use std::collections::{HashMap, HashSet};

use block_storage::rust::{
    dag::block_dag_key_value_storage::KeyValueDagRepresentation,
    key_value_block_store::KeyValueBlockStore,
};
use models::rust::block_hash::BlockHash;
use models::rust::casper::pretty_printer::PrettyPrinter;
use prost::bytes::Bytes;

/// Convenience alias matching `BlockAPI`'s error type.
type ApiErr<T> = eyre::Result<T>;

/// Terminal or transitional state of a deploy as observed from the local DAG.
///
/// Clients poll `deploy_finalization_status` by deploy signature to learn
/// whether a deploy has canonically landed. Block-hash polling is insufficient
/// because a block can finalize while the effects of some of its deploys
/// were dropped during merge — `Finalized` here means the effects are in
/// canonical state, not merely that some block containing the sig finalized.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeployFinalizationState {
    /// Deploy appears in a finalized block's `body.deploys` with
    /// `is_failed=false`, and does not appear in any finalized descendant's
    /// `body.rejected_deploys`. Effects are in canonical state. Terminal.
    Finalized,
    /// Deploy appears in a finalized block with `is_failed=true` — the
    /// Rholang execution itself failed (e.g., insufficient phlo, contract
    /// error). Effects will never apply. Terminal.
    Failed,
    /// Deploy has not yet reached a canonical-finalized inclusion and has
    /// not expired. May be in deploy storage, in a non-finalized block, in
    /// the rejected-deploy buffer awaiting re-proposal, or in a block that
    /// has not yet finalized. Client should keep polling.
    Pending,
    /// `valid_after_block_number + deployLifespan` has elapsed without
    /// successful canonical inclusion. The deploy can never land. Terminal.
    Expired,
}

/// Full response payload for a deploy-finalization-status query.
#[derive(Clone, Debug)]
pub struct DeployFinalizationStatus {
    pub state: DeployFinalizationState,
    /// Number of finalized blocks in which the sig appears in
    /// `body.rejected_deploys`. Zero at submission; monotonically
    /// increases with each merge rejection that finalizes. Gives
    /// operators visibility into deploys that are contending.
    pub rejection_count: u32,
    /// Hash of the highest-block-number canonical block that contains
    /// the sig in either `body.deploys` or `body.rejected_deploys`.
    /// `None` when the sig has not yet been included in any block.
    pub latest_block_hash: Option<BlockHash>,
}

impl DeployFinalizationStatus {
    pub fn pending_unknown() -> Self {
        Self {
            state: DeployFinalizationState::Pending,
            rejection_count: 0,
            latest_block_hash: None,
        }
    }
}

/// Per-sig BFS state accumulated during the finalized-window scan.
/// Lifted out of `resolve` so the same scan can update many sigs in one
/// pass (`resolve_batch`).
struct ResolverState {
    sig_bytes: Bytes,
    valid_after_block_number: i64,
    first_seen_block_hash: BlockHash,
    rejection_count: u32,
    has_failed_finalized: bool,
    /// Highest-block-number clean inclusion + its block hash. Tracked
    /// together so the post-loop invalidation step can do a canonical-
    /// descendant ancestry comparison against `latest_rejected_event`.
    clean_finalized_event: Option<(i64, BlockHash)>,
    latest_event: Option<(i64, BlockHash)>,
    latest_rejected_event: Option<(i64, BlockHash)>,
}

impl ResolverState {
    fn new(sig_bytes: Bytes, first_seen_block_hash: BlockHash, valid_after_block_number: i64) -> Self {
        Self {
            sig_bytes,
            valid_after_block_number,
            first_seen_block_hash,
            rejection_count: 0,
            has_failed_finalized: false,
            clean_finalized_event: None,
            latest_event: None,
            latest_rejected_event: None,
        }
    }
}

/// Outcome of looking up a sig's deploy-index entry and reading its
/// first-seen block.
enum PreludeOutcome {
    /// Sig is in the deploy index and the first-seen block was readable.
    /// Carries initialized scan state.
    Active(ResolverState),
    /// Sig is unknown (deploy index miss) or first-seen block is absent
    /// from the store; either way, status is `pending_unknown()`.
    Unknown,
}

/// Per-sig prelude: deploy-index lookup, first-seen block fetch, and
/// extraction of `valid_after_block_number`. Shared by `resolve` and
/// `resolve_batch` so both entry points have identical error
/// semantics — IO failures and deploy-index inconsistencies propagate
/// as `Err`; truly missing data (unknown sig, first-seen body absent
/// from store) returns `PreludeOutcome::Unknown`. The batch caller
/// translates `Err` into "skip this merge" rather than degrading
/// individual sigs to `Pending`, so corrupted sigs are surfaced
/// honestly rather than silently masquerading as never-deployed.
fn run_prelude(
    dag: &KeyValueDagRepresentation,
    block_store: &KeyValueBlockStore,
    sig: &[u8],
) -> ApiErr<PreludeOutcome> {
    let sig_vec: Vec<u8> = sig.to_vec();
    let sig_bytes: Bytes = Bytes::copy_from_slice(sig);

    let Some(first_seen_block_hash) = dag
        .lookup_by_deploy_id(&sig_vec)
        .map_err(|e| eyre::eyre!("deploy index lookup failed: {}", e))?
    else {
        return Ok(PreludeOutcome::Unknown);
    };

    let first_seen_block = match block_store.get(&first_seen_block_hash) {
        Ok(Some(b)) => b,
        Ok(None) => {
            tracing::warn!(
                "deploy_finalization_status: sig {} indexed at block {} but block body absent from store",
                hex::encode(&sig_bytes),
                PrettyPrinter::build_string_bytes(&first_seen_block_hash)
            );
            return Ok(PreludeOutcome::Unknown);
        }
        Err(e) => {
            return Err(eyre::eyre!(
                "block_store.get failed for first-seen block {}: {}",
                PrettyPrinter::build_string_bytes(&first_seen_block_hash),
                e
            ));
        }
    };
    let valid_after_block_number = first_seen_block
        .body
        .deploys
        .iter()
        .find(|pd| pd.deploy.sig == sig_bytes)
        .map(|pd| pd.deploy.data.valid_after_block_number)
        .ok_or_else(|| {
            // Indexed-but-missing-from-body is a real corruption case
            // (deploy index points at a block that no longer claims the
            // sig in body.deploys). Surface honestly so the operator
            // sees the bug; do not silently degrade to `pending_unknown`,
            // which would make a corrupted sig indistinguishable from a
            // never-deployed one.
            eyre::eyre!(
                "deploy_finalization_status: sig {} is indexed at block {} \
                 but missing from that block's body.deploys (inconsistent state)",
                hex::encode(&sig_bytes),
                PrettyPrinter::build_string_bytes(&first_seen_block_hash),
            )
        })?;

    Ok(PreludeOutcome::Active(ResolverState::new(
        sig_bytes,
        first_seen_block_hash,
        valid_after_block_number,
    )))
}

/// Walk finalized ancestors of LFB once, updating each active sig's
/// `ResolverState` for events found in `body.deploys` and
/// `body.rejected_deploys`. The caller passes the per-sig states keyed
/// by sig; this function mutates those states in place.
///
/// Cost: one block fetch per visited block in the deploy_lifespan
/// window, regardless of how many sigs are being tracked. Sig matching
/// inside each block is a HashSet membership check.
fn bfs_finalized_window(
    dag: &KeyValueDagRepresentation,
    block_store: &KeyValueBlockStore,
    deploy_lifespan: i64,
    per_sig: &mut HashMap<Bytes, ResolverState>,
) -> ApiErr<()> {
    if per_sig.is_empty() {
        return Ok(());
    }

    let lfb_hash = dag.last_finalized_block();
    let lfb_height = dag.block_number(&lfb_hash).ok_or_else(|| {
        eyre::eyre!(
            "deploy_finalization_status: LFB {} has no block_number entry",
            PrettyPrinter::build_string_bytes(&lfb_hash),
        )
    })?;
    let scan_floor = (lfb_height - deploy_lifespan).max(0);

    // Active sigs as a HashSet for O(1) membership checks during body scans.
    // Cloning sig bytes once here avoids per-block-per-sig clones.
    let active_sigs: HashSet<Bytes> = per_sig.keys().cloned().collect();

    let mut visited: HashSet<BlockHash> = HashSet::new();
    let mut frontier: Vec<BlockHash> = vec![lfb_hash.clone()];
    while let Some(candidate_hash) = frontier.pop() {
        if !visited.insert(candidate_hash.clone()) {
            continue;
        }
        let height = match dag.block_number(&candidate_hash) {
            Some(h) => h,
            None => {
                tracing::debug!(
                    "deploy_finalization_status: no block_number for candidate {} — \
                     skipping (likely cleanup race or partial DAG)",
                    PrettyPrinter::build_string_bytes(&candidate_hash)
                );
                continue;
            }
        };
        if height < scan_floor {
            continue;
        }
        let candidate_block = match block_store.get(&candidate_hash) {
            Ok(Some(b)) => b,
            Ok(None) => {
                tracing::warn!(
                    "deploy_finalization_status: finalized-ancestor block {} absent from store — \
                     scan may miss deploy events in this block",
                    PrettyPrinter::build_string_bytes(&candidate_hash)
                );
                continue;
            }
            Err(e) => {
                tracing::warn!(
                    "deploy_finalization_status: block_store.get failed for {}: {} — \
                     continuing scan; result may be incomplete",
                    PrettyPrinter::build_string_bytes(&candidate_hash),
                    e
                );
                continue;
            }
        };

        // Enqueue every parent slot. Main-parent-only walks miss blocks
        // that reached canonical state via secondary-parent merging.
        for parent in &candidate_block.header.parents_hash_list {
            if !visited.contains(parent) {
                frontier.push(parent.clone());
            }
        }

        // Sigs found in this block — used to update each sig's
        // `latest_event` once after both scans (a sig may appear in
        // both body.deploys and body.rejected_deploys of the same block
        // in pathological dedup paths; we still only bump latest_event
        // once for that sig at this height).
        let mut seen_sigs_here: HashSet<Bytes> = HashSet::new();

        for pd in &candidate_block.body.deploys {
            if active_sigs.contains(&pd.deploy.sig) {
                seen_sigs_here.insert(pd.deploy.sig.clone());
                let state = per_sig
                    .get_mut(&pd.deploy.sig)
                    .expect("active_sigs and per_sig must agree on key set");
                if pd.is_failed {
                    state.has_failed_finalized = true;
                } else if state
                    .clean_finalized_event
                    .as_ref()
                    .map(|(h, _)| height > *h)
                    .unwrap_or(true)
                {
                    state.clean_finalized_event = Some((height, candidate_hash.clone()));
                }
            }
        }
        for rd in &candidate_block.body.rejected_deploys {
            if active_sigs.contains(&rd.sig) {
                seen_sigs_here.insert(rd.sig.clone());
                let state = per_sig
                    .get_mut(&rd.sig)
                    .expect("active_sigs and per_sig must agree on key set");
                state.rejection_count = state.rejection_count.saturating_add(1);
                if state
                    .latest_rejected_event
                    .as_ref()
                    .map(|(h, _)| height > *h)
                    .unwrap_or(true)
                {
                    state.latest_rejected_event = Some((height, candidate_hash.clone()));
                }
            }
        }
        for sig in &seen_sigs_here {
            let state = per_sig
                .get_mut(sig)
                .expect("seen_sigs_here is drawn from active_sigs / per_sig");
            if state
                .latest_event
                .as_ref()
                .map(|(h, _)| height > *h)
                .unwrap_or(true)
            {
                state.latest_event = Some((height, candidate_hash.clone()));
            }
        }
    }

    Ok(())
}

/// Apply the per-sig post-loop rules: canonical-descendant invalidation
/// of clean inclusions, latest_block_hash fallback to the first-seen
/// block, expiry rule, and final state determination.
fn finalize_sig_state(
    dag: &KeyValueDagRepresentation,
    deploy_lifespan: i64,
    state: ResolverState,
) -> DeployFinalizationStatus {
    // A rejection invalidates a clean inclusion only when the
    // rejection block is a CANONICAL-CHAIN DESCENDANT of the clean
    // block. Two reasons height alone is wrong:
    //
    //  1. Multi-parent DAGs: blocks at the same height can be siblings
    //     on separate chains. A rejection in a sibling at the same or
    //     higher height does not affect the deploy's effects in a
    //     canonical block on a different chain.
    //  2. Recovery cycles via the rejected-deploy buffer produce
    //     rejection events in non-canonical sibling blocks (validators
    //     racing to recover the same deploy). Counting these as "after"
    //     the clean inclusion creates a positive feedback loop where
    //     the deploy stays Pending while the buffer keeps re-proposing.
    //
    // Two conditions must BOTH hold for a rejection to invalidate a
    // clean inclusion:
    //
    //   (a) `is_in_main_chain(clean_block, reject_block)` — clean is
    //       in reject's main-parent ancestry. Necessary so the
    //       rejection is "downstream" of the clean inclusion.
    //   (b) `is_in_main_chain(reject_block, lfb)` — reject is itself
    //       on LFB's main-parent chain (i.e., canonical). Necessary
    //       because (a) alone is satisfied even by non-canonical
    //       sibling blocks: a sibling fork B' that has the canonical
    //       clean block A as its main parent will pass (a) yet sit
    //       outside LFB's main chain. Without (b) the resolver
    //       reports false-Pending for sigs that are genuinely in
    //       canonical state — exactly the recovery-cycle case the
    //       comment above warns about.
    //
    // Same-block (clean and rejection in the SAME block — e.g., a
    // recovery proposal whose merge step also dedup-rejected an older
    // copy in scope) is not a "descendant" and must not invalidate.
    let mut clean_finalized_height: Option<i64> =
        state.clean_finalized_event.as_ref().map(|(h, _)| *h);
    if let (Some((_, clean_block)), Some((_, reject_block))) =
        (&state.clean_finalized_event, &state.latest_rejected_event)
    {
        let lfb_hash = dag.last_finalized_block();
        let reject_is_canonical = reject_block == &lfb_hash
            || dag.is_in_main_chain(reject_block, &lfb_hash).unwrap_or(false);
        let reject_is_canonical_descendant = clean_block != reject_block
            && reject_is_canonical
            && dag.is_in_main_chain(clean_block, reject_block).unwrap_or(false);
        if reject_is_canonical_descendant {
            clean_finalized_height = None;
        }
    }

    // Account for latest_block_hash via the first-seen lookup —
    // covers the case where the sig lives only in a non-finalized
    // block (outside the finalized scan). If the first-seen block
    // somehow has no height entry, skip this fallback rather than
    // record a block_number=0 which would mis-sort against real
    // canonical events.
    let mut latest_event = state.latest_event;
    if latest_event.is_none() {
        if let Some(first_seen_height) = dag.block_number(&state.first_seen_block_hash) {
            latest_event = Some((first_seen_height, state.first_seen_block_hash.clone()));
        } else {
            tracing::debug!(
                "deploy_finalization_status: first-seen block {} has no block_number — \
                 leaving latest_block_hash empty rather than record with bogus height",
                PrettyPrinter::build_string_bytes(&state.first_seen_block_hash)
            );
        }
    }

    // Expiry rule: tip height strictly past valid_after + deployLifespan
    // AND no clean finalized inclusion. Use the DAG's overall latest
    // block number (may be higher than LFB height) as tip.
    let tip_height = dag.latest_block_number();
    let expired = tip_height > state.valid_after_block_number + deploy_lifespan
        && clean_finalized_height.is_none();

    let final_state = if state.has_failed_finalized {
        DeployFinalizationState::Failed
    } else if clean_finalized_height.is_some() {
        DeployFinalizationState::Finalized
    } else if expired {
        DeployFinalizationState::Expired
    } else {
        DeployFinalizationState::Pending
    };

    let _ = state.sig_bytes; // no longer needed past finalize

    DeployFinalizationStatus {
        state: final_state,
        rejection_count: state.rejection_count,
        latest_block_hash: latest_event.map(|(_, h)| h),
    }
}

/// Pure resolver for deploy finalization state, single-sig entry point.
/// Does not depend on the engine cell; callable from any context that
/// has a DAG representation, a block store, and the shard-level
/// `deploy_lifespan`. The gRPC / HTTP wrappers call this under their
/// async unwrap of the Casper instance.
///
/// Error semantics shared with `resolve_batch`: deploy-index
/// inconsistencies (sig indexed at a block whose body does not contain
/// the sig) propagate as `Err` so corruption is surfaced rather than
/// hidden behind `pending_unknown`. Truly absent data (unknown sig,
/// first-seen body missing from the store) returns `pending_unknown`.
///
/// The state machine is a canonical-chain scan:
///
/// 1. Look up the sig in the deploy index. Unknown sig → `Pending`.
/// 2. Fetch the first-seen block to read `valid_after_block_number`.
/// 3. Walk the finalized chain from LFB backward for `deploy_lifespan`
///    blocks, tallying clean inclusions, failed inclusions, rejections,
///    and `latest_block_hash`.
/// 4. Apply the state rules: failed finalized → `Failed`; clean finalized
///    without a later canonical-descendant rejection → `Finalized`;
///    beyond lifespan without a clean inclusion → `Expired`; otherwise
///    → `Pending`.
pub fn resolve(
    dag: &KeyValueDagRepresentation,
    block_store: &KeyValueBlockStore,
    deploy_lifespan: i64,
    sig: &[u8],
) -> ApiErr<DeployFinalizationStatus> {
    let prelude = run_prelude(dag, block_store, sig)?;
    let state = match prelude {
        PreludeOutcome::Unknown => return Ok(DeployFinalizationStatus::pending_unknown()),
        PreludeOutcome::Active(s) => s,
    };

    let mut per_sig: HashMap<Bytes, ResolverState> = HashMap::new();
    per_sig.insert(state.sig_bytes.clone(), state);

    bfs_finalized_window(dag, block_store, deploy_lifespan, &mut per_sig)?;

    let (_, state) = per_sig
        .into_iter()
        .next()
        .expect("per_sig was populated with one entry above");
    Ok(finalize_sig_state(dag, deploy_lifespan, state))
}

/// Batched resolver for many sigs in a single canonical-chain scan.
/// Optimizes the catchup-heavy hot path in
/// `compute_parents_post_state::should_admit_to_rejected_buffer`, where
/// every rejected deploy in a merge would otherwise trigger an
/// independent BFS over the same finalized window.
///
/// Cost vs. calling `resolve` per sig: with N sigs and M blocks in the
/// `deploy_lifespan` window, this is O(M + N) block fetches instead of
/// O(N · M). For a 50-rejected merge with M=200, that is 200 fetches
/// instead of 10 000.
///
/// Error semantics match `resolve`: any failure during the prelude (IO,
/// deploy-index inconsistency, LFB lookup) propagates as `Err` for the
/// whole batch. Sigs that are simply not in the deploy index (or whose
/// first-seen block is missing) yield
/// `DeployFinalizationStatus::pending_unknown()` for that sig — those
/// are absences, not corruptions.
///
/// The `compute_parents_post_state` caller wraps the batch call in a
/// "skip on Err" fallback (admit nothing for the merge step), so a
/// single corrupted sig does pause admit decisions for that one merge
/// rather than silently mislabeling the corruption as a healthy
/// `Pending`.
pub fn resolve_batch(
    dag: &KeyValueDagRepresentation,
    block_store: &KeyValueBlockStore,
    deploy_lifespan: i64,
    sigs: &HashSet<Bytes>,
) -> ApiErr<HashMap<Bytes, DeployFinalizationStatus>> {
    let mut results: HashMap<Bytes, DeployFinalizationStatus> = HashMap::new();
    if sigs.is_empty() {
        return Ok(results);
    }

    // Per-sig prelude (lenient: inconsistencies → Unknown).
    let mut per_sig: HashMap<Bytes, ResolverState> = HashMap::new();
    for sig in sigs {
        match run_prelude(dag, block_store, sig.as_ref())? {
            PreludeOutcome::Unknown => {
                results.insert(sig.clone(), DeployFinalizationStatus::pending_unknown());
            }
            PreludeOutcome::Active(state) => {
                per_sig.insert(state.sig_bytes.clone(), state);
            }
        }
    }

    // Single BFS pass — the whole point of this function.
    bfs_finalized_window(dag, block_store, deploy_lifespan, &mut per_sig)?;

    // Per-sig post-processing.
    for (sig, state) in per_sig {
        results.insert(sig, finalize_sig_state(dag, deploy_lifespan, state));
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_unknown_has_empty_fields() {
        let s = DeployFinalizationStatus::pending_unknown();
        assert_eq!(s.state, DeployFinalizationState::Pending);
        assert_eq!(s.rejection_count, 0);
        assert!(s.latest_block_hash.is_none());
    }

    #[test]
    fn states_are_distinct() {
        let all = [
            DeployFinalizationState::Finalized,
            DeployFinalizationState::Failed,
            DeployFinalizationState::Pending,
            DeployFinalizationState::Expired,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(
                    a == b,
                    i == j,
                    "state equality mismatch: {:?} vs {:?}",
                    a,
                    b
                );
            }
        }
    }
}
