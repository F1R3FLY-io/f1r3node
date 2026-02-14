// See casper/src/main/scala/coop/rchain/casper/util/rholang/InterpreterUtil.scala

use prost::bytes::Bytes;
use std::collections::{BTreeMap, HashMap, HashSet};

use block_storage::rust::{
    dag::block_dag_key_value_storage::KeyValueDagRepresentation,
    key_value_block_store::KeyValueBlockStore,
};
use crypto::rust::signatures::signed::Signed;
use models::{
    rhoapi::Par,
    rust::{
        block::state_hash::StateHash,
        block_hash::BlockHash,
        casper::{
            pretty_printer::PrettyPrinter,
            protocol::casper_message::{
                BlockMessage, DeployData, ProcessedDeploy, ProcessedSystemDeploy,
            },
        },
        validator::Validator,
    },
};
use rholang::rust::interpreter::{
    compiler::compiler::Compiler, errors::InterpreterError, system_processes::BlockData,
};
use rspace_plus_plus::rspace::{hashing::blake2b256_hash::Blake2b256Hash, history::Either};

use crate::rust::{
    block_status::BlockStatus,
    casper::CasperSnapshot,
    errors::CasperError,
    merging::{block_index::BlockIndex, dag_merger, deploy_chain_index::DeployChainIndex},
    metrics_constants::{
        CASPER_METRICS_SOURCE,
        BLOCK_PROCESSING_REPLAY_TIME_METRIC,
    },
    util::proto_util,
    BlockProcessing,
};

use super::{replay_failure::ReplayFailure, runtime_manager::RuntimeManager};

pub fn mk_term(rho: &str, normalizer_env: HashMap<String, Par>) -> Result<Par, InterpreterError> {
    Compiler::source_to_adt_with_normalizer_env(rho, normalizer_env)
}

// Returns (None, checkpoints) if the block's tuplespace hash
// does not match the computed hash based on the deploys
pub async fn validate_block_checkpoint(
    block: &BlockMessage,
    block_store: &KeyValueBlockStore,
    s: &mut CasperSnapshot,
    runtime_manager: &mut RuntimeManager,
) -> Result<BlockProcessing<Option<StateHash>>, CasperError> {
    tracing::debug!(target: "f1r3fly.casper", "before-unsafe-get-parents");
    let incoming_pre_state_hash = proto_util::pre_state_hash(block);
    let parents = proto_util::get_parents(block_store, block);
    tracing::debug!(target: "f1r3fly.casper", "before-compute-parents-post-state");
    let computed_parents_info =
        compute_parents_post_state(block_store, parents.clone(), s, runtime_manager, None);

    tracing::info!(
        "Computed parents post state for {}.",
        PrettyPrinter::build_string_block_message(&block, false)
    );

    match computed_parents_info {
        Ok((computed_pre_state_hash, rejected_deploys)) => {
            let rejected_deploy_ids: HashSet<_> = rejected_deploys.iter().cloned().collect();
            let block_rejected_deploy_sigs: HashSet<_> = block
                .body
                .rejected_deploys
                .iter()
                .map(|d| d.sig.clone())
                .collect();

            if incoming_pre_state_hash != computed_pre_state_hash {
                // TODO: at this point we may just as well terminate the replay, there's no way it will succeed.
                tracing::warn!(
                    "Computed pre-state hash {} does not equal block's pre-state hash {}.",
                    PrettyPrinter::build_string_bytes(&computed_pre_state_hash),
                    PrettyPrinter::build_string_bytes(&incoming_pre_state_hash)
                );

                return Ok(Either::Right(None));
            } else if rejected_deploy_ids != block_rejected_deploy_sigs {
                // Detailed logging for InvalidRejectedDeploy mismatch
                let extra_in_computed: Vec<_> = rejected_deploy_ids
                    .difference(&block_rejected_deploy_sigs)
                    .cloned()
                    .collect();
                let missing_in_computed: Vec<_> = block_rejected_deploy_sigs
                    .difference(&rejected_deploy_ids)
                    .cloned()
                    .collect();

                // Get all deploy signatures in the block for duplicate detection
                let all_block_deploys: Vec<_> = block
                    .body
                    .deploys
                    .iter()
                    .map(|pd| pd.deploy.sig.clone())
                    .collect();
                let mut all_deploy_sigs: Vec<_> = all_block_deploys.clone();
                all_deploy_sigs.extend(block.body.rejected_deploys.iter().map(|rd| rd.sig.clone()));

                // Find duplicates
                let mut sig_counts: HashMap<Bytes, usize> = HashMap::new();
                for sig in &all_deploy_sigs {
                    *sig_counts.entry(sig.clone()).or_insert(0) += 1;
                }
                let duplicates: Vec<_> = sig_counts
                    .into_iter()
                    .filter(|(_, count)| *count > 1)
                    .map(|(sig, _)| sig)
                    .collect();

                // Build deploy data map for correlation
                let deploy_data_map: HashMap<Bytes, &Signed<DeployData>> = block
                    .body
                    .deploys
                    .iter()
                    .map(|pd| (pd.deploy.sig.clone(), &pd.deploy))
                    .collect();

                // Helper to analyze a deploy signature
                let analyze_deploy_sig = |sig: &Bytes| -> String {
                    let sig_str = PrettyPrinter::build_string_bytes(sig);
                    let is_duplicate = if duplicates.contains(sig) {
                        " [DUPLICATE]"
                    } else {
                        ""
                    };
                    let deploy_info = match deploy_data_map.get(sig) {
                        Some(deploy) => {
                            let term_preview: String = deploy.data.term.chars().take(50).collect();
                            format!(
                                " (term={}..., timestamp={}, phloLimit={})",
                                term_preview, deploy.data.time_stamp, deploy.data.phlo_limit
                            )
                        }
                        None => " (deploy data not found in block)".to_string(),
                    };
                    format!("{}{}{}", sig_str, is_duplicate, deploy_info)
                };

                let extra_analysis: String = if extra_in_computed.is_empty() {
                    "  None".to_string()
                } else {
                    extra_in_computed
                        .iter()
                        .map(|sig| format!("  {}", analyze_deploy_sig(sig)))
                        .collect::<Vec<_>>()
                        .join("\n")
                };

                let missing_analysis: String = if missing_in_computed.is_empty() {
                    "  None".to_string()
                } else {
                    missing_in_computed
                        .iter()
                        .map(|sig| format!("  {}", analyze_deploy_sig(sig)))
                        .collect::<Vec<_>>()
                        .join("\n")
                };

                let duplicates_str: String = if duplicates.is_empty() {
                    "  None".to_string()
                } else {
                    duplicates
                        .iter()
                        .map(|sig| format!("  {}", PrettyPrinter::build_string_bytes(sig)))
                        .collect::<Vec<_>>()
                        .join("\n")
                };

                let parent_hashes: String = parents
                    .iter()
                    .map(|p| PrettyPrinter::build_string_bytes(&p.block_hash))
                    .collect::<Vec<_>>()
                    .join(", ");

                tracing::error!(
                    "\n=== InvalidRejectedDeploy Analysis ===\n\
                    Block #{} ({})\n\
                    Sender: {}\n\
                    Parents: {}\n\n\
                    Rejected deploy mismatch:\n\
                    \x20 Validator computed: {} rejected deploys\n\
                    \x20 Block contains:     {} rejected deploys\n\n\
                    Extra in computed (validator wants to reject, but block creator didn't):\n\
                    \x20 Count: {}\n{}\n\n\
                    Missing in computed (block creator rejected, but validator doesn't think should be):\n\
                    \x20 Count: {}\n{}\n\n\
                    Duplicates found in block: {}\n{}\n\n\
                    All deploys in block: {}\n\
                    All rejected in block: {}\n\
                    ========================================",
                    block.body.state.block_number,
                    PrettyPrinter::build_string_bytes(&block.block_hash),
                    PrettyPrinter::build_string_bytes(&block.sender),
                    parent_hashes,
                    rejected_deploy_ids.len(),
                    block_rejected_deploy_sigs.len(),
                    extra_in_computed.len(),
                    extra_analysis,
                    missing_in_computed.len(),
                    missing_analysis,
                    duplicates.len(),
                    duplicates_str,
                    all_block_deploys.len(),
                    block_rejected_deploy_sigs.len()
                );

                return Ok(Either::Left(BlockStatus::invalid_rejected_deploy()));
            } else {
                tracing::debug!(target: "f1r3fly.casper.replay-block", "before-process-pre-state-hash");
                // Using tracing events for async - Span[F] equivalent from Scala
                tracing::debug!(target: "f1r3fly.casper.replay-block", "replay-block-started");
                let replay_start = std::time::Instant::now();
                let replay_result =
                    replay_block(incoming_pre_state_hash, block, &mut s.dag, runtime_manager)
                        .await?;
                metrics::histogram!(BLOCK_PROCESSING_REPLAY_TIME_METRIC, "source" => CASPER_METRICS_SOURCE)
                    .record(replay_start.elapsed().as_secs_f64());
                tracing::debug!(target: "f1r3fly.casper.replay-block", "replay-block-finished");

                handle_errors(proto_util::post_state_hash(block), replay_result)
            }
        }
        Err(ex) => {
            return Ok(Either::Left(BlockStatus::exception(ex)));
        }
    }
}

async fn replay_block(
    initial_state_hash: StateHash,
    block: &BlockMessage,
    dag: &mut KeyValueDagRepresentation,
    runtime_manager: &mut RuntimeManager,
) -> Result<Either<ReplayFailure, StateHash>, CasperError> {
    // Extract deploys and system deploys from the block
    let internal_deploys = proto_util::deploys(block);
    let internal_system_deploys = proto_util::system_deploys(block);

    // Check for duplicate deploys in the block before replay
    let mut all_deploy_sigs: Vec<Bytes> = internal_deploys
        .iter()
        .map(|pd| pd.deploy.sig.clone())
        .collect();
    all_deploy_sigs.extend(block.body.rejected_deploys.iter().map(|rd| rd.sig.clone()));

    let mut sig_counts: HashMap<Bytes, usize> = HashMap::new();
    for sig in &all_deploy_sigs {
        *sig_counts.entry(sig.clone()).or_insert(0) += 1;
    }
    let deploy_duplicates: HashMap<Bytes, usize> = sig_counts
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .collect();

    if !deploy_duplicates.is_empty() {
        let duplicates_str: String = deploy_duplicates
            .iter()
            .map(|(sig, count)| {
                format!(
                    "  {} (appears {} times)",
                    PrettyPrinter::build_string_bytes(sig),
                    count
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        tracing::warn!(
            "\n=== Duplicate Deploys Detected in Block ===\n\
            Block #{} ({})\n\
            Found {} duplicate deploy signatures:\n{}\n\
            Total deploys: {}\n\
            Total rejected: {}\n\
            ============================================",
            block.body.state.block_number,
            PrettyPrinter::build_string_bytes(&block.block_hash),
            deploy_duplicates.len(),
            duplicates_str,
            internal_deploys.len(),
            block.body.rejected_deploys.len()
        );
    } else {
        tracing::debug!(
            "Block #{}: replaying {} deploys, {} rejected",
            block.body.state.block_number,
            internal_deploys.len(),
            block.body.rejected_deploys.len()
        );
    }

    // Get invalid blocks set from DAG
    let invalid_blocks_set = dag.invalid_blocks();

    // Get unseen block hashes
    let unseen_blocks_set = proto_util::unseen_block_hashes(dag, block)?;

    // Filter out invalid blocks that are unseen
    let seen_invalid_blocks_set: Vec<_> = invalid_blocks_set
        .iter()
        .filter(|invalid_block| !unseen_blocks_set.contains(&invalid_block.block_hash))
        .map(|invalid_block| invalid_block.clone())
        .collect();
    // TODO: Write test in which switching this to .filter makes it fail

    // Convert to invalid blocks map
    let invalid_blocks: HashMap<BlockHash, Validator> = seen_invalid_blocks_set
        .into_iter()
        .map(|invalid_block| (invalid_block.block_hash, invalid_block.sender))
        .collect();

    // Create block data and check if genesis
    let block_data = BlockData::from_block(block);
    let is_genesis = block.header.parents_hash_list.is_empty();

    // Implement retry logic with limit of 3 retries
    let mut attempts = 0;
    const MAX_RETRIES: usize = 3;

    loop {
        // Call the async replay_compute_state method
        let replay_result = runtime_manager
            .replay_compute_state(
                &initial_state_hash,
                internal_deploys.clone(),
                internal_system_deploys.clone(),
                &block_data,
                Some(invalid_blocks.clone()),
                is_genesis,
            )
            .await;

        match replay_result {
            Ok(computed_state_hash) => {
                // Check if computed hash matches expected hash
                if computed_state_hash == block.body.state.post_state_hash {
                    // Success - hashes match
                    return Ok(Either::Right(computed_state_hash));
                } else if attempts >= MAX_RETRIES {
                    // Give up after max retries
                    tracing::error!(
                        "Replay block {} with {} got tuple space mismatch error with error hash {}, retries details: giving up after {} retries",
                        PrettyPrinter::build_string_no_limit(&block.block_hash),
                        PrettyPrinter::build_string_no_limit(&block.body.state.post_state_hash),
                        PrettyPrinter::build_string_no_limit(&computed_state_hash),
                        attempts
                    );
                    return Ok(Either::Right(computed_state_hash));
                } else {
                    // Retry - log error and continue
                    tracing::error!(
                        "Replay block {} with {} got tuple space mismatch error with error hash {}, retries details: will retry, attempt {}",
                        PrettyPrinter::build_string_no_limit(&block.block_hash),
                        PrettyPrinter::build_string_no_limit(&block.body.state.post_state_hash),
                        PrettyPrinter::build_string_no_limit(&computed_state_hash),
                        attempts + 1
                    );
                    attempts += 1;
                }
            }
            Err(replay_error) => {
                if attempts >= MAX_RETRIES {
                    // Give up after max retries
                    tracing::error!(
                        "Replay block {} got error {:?}, retries details: giving up after {} retries",
                        PrettyPrinter::build_string_no_limit(&block.block_hash),
                        replay_error,
                        attempts
                    );
                    // Convert CasperError to ReplayFailure::InternalError
                    return Ok(Either::Left(ReplayFailure::internal_error(
                        replay_error.to_string(),
                    )));
                } else {
                    // Retry - log error and continue
                    tracing::error!(
                        "Replay block {} got error {:?}, retries details: will retry, attempt {}",
                        PrettyPrinter::build_string_no_limit(&block.block_hash),
                        replay_error,
                        attempts + 1
                    );
                    attempts += 1;
                }
            }
        }
    }
}

fn handle_errors(
    ts_hash: StateHash,
    result: Either<ReplayFailure, StateHash>,
) -> Result<BlockProcessing<Option<StateHash>>, CasperError> {
    match result {
        Either::Left(replay_failure) => match replay_failure {
            ReplayFailure::InternalError { msg } => {
                let exception = CasperError::RuntimeError(format!(
                    "Internal errors encountered while processing deploy: {}",
                    msg
                ));
                Ok(Either::Left(BlockStatus::exception(exception)))
            }

            ReplayFailure::ReplayStatusMismatch {
                initial_failed,
                replay_failed,
            } => {
                println!(
                    "Found replay status mismatch; replay failure is {} and orig failure is {}",
                    replay_failed, initial_failed
                );
                tracing::warn!(
                    "Found replay status mismatch; replay failure is {} and orig failure is {}",
                    replay_failed,
                    initial_failed
                );
                Ok(Either::Right(None))
            }

            ReplayFailure::UnusedCOMMEvent { msg } => {
                println!("Found replay exception: {}", msg);
                tracing::warn!("Found replay exception: {}", msg);
                Ok(Either::Right(None))
            }

            ReplayFailure::ReplayCostMismatch {
                initial_cost,
                replay_cost,
            } => {
                println!(
                    "Found replay cost mismatch: initial deploy cost = {}, replay deploy cost = {}",
                    initial_cost, replay_cost
                );
                tracing::warn!(
                    "Found replay cost mismatch: initial deploy cost = {}, replay deploy cost = {}",
                    initial_cost,
                    replay_cost
                );
                Ok(Either::Right(None))
            }

            ReplayFailure::SystemDeployErrorMismatch {
                play_error,
                replay_error,
            } => {
                tracing::warn!(
                        "Found system deploy error mismatch: initial deploy error message = {}, replay deploy error message = {}",
                        play_error, replay_error
                    );
                Ok(Either::Right(None))
            }
        },

        Either::Right(computed_state_hash) => {
            if ts_hash == computed_state_hash {
                // State hash in block matches computed hash!
                Ok(Either::Right(Some(computed_state_hash)))
            } else {
                // State hash in block does not match computed hash -- invalid!
                // return no state hash, do not update the state hash set
                println!(
                    "Tuplespace hash {} does not match computed hash {}.",
                    PrettyPrinter::build_string_bytes(&ts_hash),
                    PrettyPrinter::build_string_bytes(&computed_state_hash)
                );
                tracing::warn!(
                    "Tuplespace hash {} does not match computed hash {}.",
                    PrettyPrinter::build_string_bytes(&ts_hash),
                    PrettyPrinter::build_string_bytes(&computed_state_hash)
                );
                Ok(Either::Right(None))
            }
        }
    }
}

pub fn print_deploy_errors(deploy_sig: &Bytes, errors: &[InterpreterError]) {
    let deploy_info = PrettyPrinter::build_string_sig(&deploy_sig);
    let error_messages: String = errors
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    println!("Deploy ({}) errors: {}", deploy_info, error_messages);

    tracing::warn!("Deploy ({}) errors: {}", deploy_info, error_messages);
}

pub async fn compute_deploys_checkpoint(
    block_store: &mut KeyValueBlockStore,
    parents: Vec<BlockMessage>,
    deploys: Vec<Signed<DeployData>>,
    system_deploys: Vec<super::system_deploy_enum::SystemDeployEnum>,
    s: &CasperSnapshot,
    runtime_manager: &mut RuntimeManager,
    block_data: BlockData,
    invalid_blocks: HashMap<BlockHash, Validator>,
) -> Result<
    (
        StateHash,
        StateHash,
        Vec<ProcessedDeploy>,
        Vec<prost::bytes::Bytes>,
        Vec<ProcessedSystemDeploy>,
    ),
    CasperError,
> {
    // Using tracing events for async - Span[F] equivalent from Scala
    tracing::debug!(target: "f1r3fly.casper.compute-deploys-checkpoint", "compute-deploys-checkpoint-started");
    // Ensure parents are not empty
    if parents.is_empty() {
        return Err(CasperError::RuntimeError(
            "Parents must not be empty".to_string(),
        ));
    }

    // Compute parents post state
    let computed_parents_info =
        compute_parents_post_state(block_store, parents, s, runtime_manager, None)?;
    let (pre_state_hash, rejected_deploys) = computed_parents_info;

    // Compute state using runtime manager
    let result = runtime_manager
        .compute_state(
            &pre_state_hash,
            deploys,
            system_deploys,
            block_data,
            Some(invalid_blocks),
        )
        .await?;

    let (post_state_hash, processed_deploys, processed_system_deploys) = result;

    Ok((
        pre_state_hash,
        post_state_hash,
        processed_deploys,
        rejected_deploys,
        processed_system_deploys,
    ))
}

/// Bounded convergent LCA (Lowest Common Ancestor) algorithm.
///
/// Walks back from all parents simultaneously in descending block_number order,
/// tracking which original parents can reach each visited block. The first
/// block reachable from ALL parents (highest block_number) is the LCA.
///
/// Cost: O(blocks_between_parents_and_LCA) DAG lookups, which is typically
/// O(num_validators * few_levels) for a healthy network. This replaces
/// N separate all_ancestors-to-genesis traversals (each O(chain_length)).
///
/// Uses purely structural traversal (parent links only, no finalization state)
/// to guarantee determinism across all validators with the same DAG.
///
/// Returns (lca, visible_blocks) where lca is the highest common ancestor and
/// visible_blocks contains all blocks between parents and LCA (inclusive).
fn bounded_lca_walk(
    dag: &KeyValueDagRepresentation,
    parent_hashes: &[BlockHash],
) -> Result<(BlockHash, HashSet<BlockHash>), CasperError> {
    let num_parents = parent_hashes.len();

    // Frontier: block_number -> set of block hashes at that height waiting to be processed.
    // We always process the highest block_number first (using BTreeMap's last entry).
    let mut frontier: BTreeMap<i64, HashSet<BlockHash>> = BTreeMap::new();

    // For each block hash, which original parent indices (0..N-1) can reach it.
    let mut reachable_from: HashMap<BlockHash, HashSet<usize>> = HashMap::new();

    // All blocks that have been processed (visited).
    let mut visited: HashSet<BlockHash> = HashSet::new();

    // Seed the frontier with the parent blocks
    for (i, parent_hash) in parent_hashes.iter().enumerate() {
        let meta = dag.lookup_unsafe(parent_hash).map_err(|e| {
            CasperError::RuntimeError(format!("Failed to look up parent {}: {}", hex::encode(&parent_hash[..std::cmp::min(8, parent_hash.len())]), e))
        })?;
        frontier
            .entry(meta.block_number)
            .or_insert_with(HashSet::new)
            .insert(parent_hash.clone());
        reachable_from
            .entry(parent_hash.clone())
            .or_insert_with(HashSet::new)
            .insert(i);
    }

    loop {
        if frontier.is_empty() {
            // Should not happen with valid parents (all paths converge at genesis).
            // Fall back: return the first parent as LCA.
            tracing::warn!("boundedLcaWalk: frontier exhausted without finding LCA, falling back to first parent");
            return Ok((parent_hashes[0].clone(), visited));
        }

        // Pop the highest block_number layer from the frontier
        let highest_key = *frontier.keys().next_back().unwrap();
        let blocks_at_height = frontier.remove(&highest_key).unwrap();

        // Look up metadata for all blocks at this height
        let metas: Vec<_> = blocks_at_height
            .iter()
            .map(|h| dag.lookup_unsafe(h))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CasperError::RuntimeError(format!("DAG lookup failed during LCA walk: {}", e)))?;

        let mut lca_found: Option<(BlockHash, HashSet<BlockHash>)> = None;

        for meta in &metas {
            if lca_found.is_some() {
                break;
            }

            if visited.contains(&meta.block_hash) {
                // Already processed via another path
                continue;
            }

            visited.insert(meta.block_hash.clone());
            let block_reach = reachable_from
                .get(&meta.block_hash)
                .cloned()
                .unwrap_or_default();

            if block_reach.len() == num_parents {
                // Found the LCA -- all parents can reach this block
                lca_found = Some((meta.block_hash.clone(), visited.clone()));
            } else {
                // Propagate reachability to this block's parents
                for parent_hash in &meta.parents {
                    let parent_reach = reachable_from
                        .entry(parent_hash.clone())
                        .or_insert_with(HashSet::new);
                    parent_reach.extend(block_reach.iter());
                }
            }
        }

        if let Some(result) = lca_found {
            return Ok(result);
        }

        // Collect parent hashes from newly processed blocks that need
        // to be enqueued (not already visited and not already in frontier)
        let already_in_frontier: HashSet<BlockHash> =
            frontier.values().flat_map(|s| s.iter().cloned()).collect();

        let new_parents: Vec<BlockHash> = metas
            .iter()
            .filter(|m| visited.contains(&m.block_hash))
            .flat_map(|m| m.parents.iter().cloned())
            .collect::<HashSet<_>>()
            .into_iter()
            .filter(|h| !visited.contains(h) && !already_in_frontier.contains(h))
            .collect();

        // Look up block_numbers for new parents and add to frontier
        for parent_hash in new_parents {
            let parent_meta = dag.lookup_unsafe(&parent_hash).map_err(|e| {
                CasperError::RuntimeError(format!("Failed to look up parent during LCA walk: {}", e))
            })?;
            frontier
                .entry(parent_meta.block_number)
                .or_insert_with(HashSet::new)
                .insert(parent_hash);
        }
    }
}

/// Compute the merged post-state from multiple parent blocks.
/// 
/// For exploratory deploy, pass `disable_late_block_filtering_override = Some(true)` to
/// always disable late block filtering (see full merged state).
/// For normal block creation, pass `None` to use the shard config value.
pub fn compute_parents_post_state(
    block_store: &KeyValueBlockStore,
    parents: Vec<BlockMessage>,
    s: &CasperSnapshot,
    runtime_manager: &RuntimeManager,
    disable_late_block_filtering_override: Option<bool>,
) -> Result<(StateHash, Vec<Bytes>), CasperError> {
    // Span guard must live until end of scope to maintain tracing context
    let _span = tracing::debug_span!(target: "f1r3fly.casper.compute-parents-post-state", "compute-parents-post-state").entered();
    match parents.len() {
        // For genesis, use empty trie's root hash
        0 => Ok((RuntimeManager::empty_state_hash_fixed(), Vec::new())),

        // For single parent, get its post state hash
        1 => {
            let parent = &parents[0];
            Ok((proto_util::post_state_hash(parent), Vec::new()))
        }

        // Multiple parents - we might want to take some data from the parent with the most stake,
        // e.g. bonds map, slashing deploys, bonding deploys.
        // such system deploys are not mergeable, so take them from one of the parents.
        _ => {
            // Function to get or compute BlockIndex for each parent block hash
            let block_index_f = |v: &BlockHash| -> Result<BlockIndex, CasperError> {
                // Try cache first
                if let Some(cached) = runtime_manager.block_index_cache.get(v) {
                    return Ok((*cached.value()).clone());
                }

                // Cache miss - compute the BlockIndex
                let b = block_store.get_unsafe(v);
                let pre_state = &b.body.state.pre_state_hash;
                let post_state = &b.body.state.post_state_hash;
                let sender = b.sender.clone();
                let seq_num = b.seq_num;

                let mergeable_chs =
                    runtime_manager.load_mergeable_channels(post_state, sender, seq_num)?;

                let block_index = crate::rust::merging::block_index::new(
                    &b.block_hash,
                    &b.body.deploys,
                    &b.body.system_deploys,
                    &Blake2b256Hash::from_bytes_prost(pre_state),
                    &Blake2b256Hash::from_bytes_prost(post_state),
                    &runtime_manager.history_repo,
                    &mergeable_chs,
                )?;

                // Cache the result
                runtime_manager
                    .block_index_cache
                    .insert(v.clone(), block_index.clone());

                Ok(block_index)
            };

            // Compute scope: blocks visible from the parents and the LCA (lowest common ancestor).
            let parent_hashes: Vec<BlockHash> =
                parents.iter().map(|p| p.block_hash.clone()).collect();

            // Bounded convergent LCA algorithm: walk back from all parents simultaneously
            // in descending blockNum order, tracking per-parent reachability. The first
            // block reachable from ALL parents (highest blockNum) is the LCA.
            //
            // Cost: O(blocks_between_parents_and_LCA), typically O(num_validators * few_levels)
            // for a healthy network. This replaces N separate allAncestors-to-genesis traversals
            // (each O(chain_length)) with a single bounded walk.
            //
            // CRITICAL: Uses purely structural traversal (parent links only, no finalization state)
            // to guarantee determinism across all validators with the same DAG.
            //
            // Returns: (lca, visibleBlocks) where visibleBlocks contains all blocks between
            // the parents and the LCA (inclusive), which is exactly the scope needed by DagMerger.
            let (lca, visible_blocks) = bounded_lca_walk(&s.dag, &parent_hashes)?;

            // Get the LCA block to use its post-state as the merge base
            let lca_block = block_store.get_unsafe(&lca);
            let lca_state = Blake2b256Hash::from_bytes_prost(&lca_block.body.state.post_state_hash);

            // Log
            let parent_hash_str: Vec<String> = parent_hashes
                .iter()
                .map(|h| hex::encode(&h[..std::cmp::min(10, h.len())]))
                .collect();
            let lca_str = hex::encode(&lca[..std::cmp::min(10, lca.len())]);
            let lca_state_str = hex::encode(
                &lca_block.body.state.post_state_hash
                    [..std::cmp::min(10, lca_block.body.state.post_state_hash.len())],
            );

            tracing::info!(
                "computeParentsPostState: parents=[{}], LCA={} (block {}), LCA state={}, visibleBlocks={}",
                parent_hash_str.join(", "),
                lca_str,
                lca_block.body.state.block_number,
                lca_state_str,
                visible_blocks.len()
            );

            // Get disableLateBlockFiltering from override or shard config
            let disable_late_block_filtering = disable_late_block_filtering_override
                .unwrap_or(s.on_chain_state.shard_conf.disable_late_block_filtering);

            // Pass preComputedLfbAncestors = Set(lca) to avoid a redundant O(chain_length)
            // traversal inside DagMerger. Since the bounded walk only visits blocks at or
            // above the LCA, the only overlap between visibleBlocks and allAncestors(lca)
            // is the LCA itself, so Set(lca) is sufficient for the subtraction.
            let pre_computed_lfb_ancestors: HashSet<BlockHash> =
                std::iter::once(lca.clone()).collect();

            // Use DagMerger to merge parent states with scope
            let merger_result = dag_merger::merge(
                &s.dag,
                &lca,
                &lca_state,
                |hash: &BlockHash| -> Result<Vec<DeployChainIndex>, CasperError> {
                    let block_index = block_index_f(hash)?;
                    Ok(block_index.deploy_chains)
                },
                &runtime_manager.history_repo,
                dag_merger::cost_optimal_rejection_alg(),
                Some(visible_blocks),
                disable_late_block_filtering,
                Some(pre_computed_lfb_ancestors),
            )?;

            let (state, rejected) = merger_result;

            Ok((
                prost::bytes::Bytes::copy_from_slice(&state.bytes()),
                rejected,
            ))
        }
    }
}
