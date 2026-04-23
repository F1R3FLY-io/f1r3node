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

/// Pure resolver for deploy finalization state. Does not depend on the
/// engine cell; callable from any context that has a DAG representation,
/// a block store, and the shard-level `deploy_lifespan`. The gRPC / HTTP
/// wrappers call this under their async unwrap of the Casper instance;
/// the catchup gate in `compute_parents_post_state` calls it directly.
///
/// The state machine is a canonical-chain scan:
///
/// 1. Look up the sig in the deploy index. Unknown sig → `Pending`.
/// 2. Fetch the first-seen block to read `valid_after_block_number`.
///    Consistency errors here (sig indexed but missing from body) are
///    surfaced as `Err`.
/// 3. Walk the finalized chain from LFB backward for `deploy_lifespan`
///    blocks, tallying clean inclusions, failed inclusions, rejections,
///    and `latest_block_hash`.
/// 4. Apply the state rules: failed finalized → `Failed`; clean finalized
///    without a later rejection → `Finalized`; beyond lifespan without a
///    clean inclusion → `Expired`; otherwise → `Pending`.
pub fn resolve(
    dag: &KeyValueDagRepresentation,
    block_store: &KeyValueBlockStore,
    deploy_lifespan: i64,
    sig: &[u8],
) -> ApiErr<DeployFinalizationStatus> {
    // The deploy index keys are `Vec<u8>` (`DeployId`); comparisons
    // against `pd.deploy.sig` need `Bytes`. Copy the input slice once
    // for each representation — status queries are rare enough that a
    // single O(sig.len()) allocation is immaterial against the block-
    // walk cost that follows.
    let sig_vec: Vec<u8> = sig.to_vec();
    let sig_bytes: Bytes = Bytes::copy_from_slice(sig);

    // Unknown sig → Pending with empty fields.
    let Some(first_seen_block_hash) = dag
        .lookup_by_deploy_id(&sig_vec)
        .map_err(|e| eyre::eyre!("deploy index lookup failed: {}", e))?
    else {
        return Ok(DeployFinalizationStatus::pending_unknown());
    };

    // Fetch the first-seen block to pull valid_after_block_number.
    // A sig can appear in multiple blocks post-recovery, but each shares
    // the same DeployData (same content). Any one block suffices. An
    // IO error here is a real storage failure — propagate it so callers
    // can distinguish it from a deploy that is legitimately unknown.
    let first_seen_block = match block_store.get(&first_seen_block_hash) {
        Ok(Some(b)) => b,
        Ok(None) => {
            tracing::warn!(
                "deploy_finalization_status: sig {} indexed at block {} but block body absent from store",
                hex::encode(&sig_bytes),
                PrettyPrinter::build_string_bytes(&first_seen_block_hash)
            );
            return Ok(DeployFinalizationStatus::pending_unknown());
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
            eyre::eyre!(
                "deploy_finalization_status: sig {} is indexed at block {} \
                 but missing from that block's body.deploys (inconsistent state)",
                hex::encode(&sig_bytes),
                PrettyPrinter::build_string_bytes(&first_seen_block_hash),
            )
        })?;

    // Walk the finalized chain from LFB backward for deploy_lifespan
    // blocks. This window is bounded and matches the deploy's window
    // of validity, so any canonical-state event for the deploy must
    // be in scope.
    let lfb_hash = dag.last_finalized_block();
    let lfb_height = dag.block_number(&lfb_hash).ok_or_else(|| {
        eyre::eyre!(
            "deploy_finalization_status: LFB {} has no block_number entry",
            PrettyPrinter::build_string_bytes(&lfb_hash),
        )
    })?;
    let scan_floor = (lfb_height - deploy_lifespan).max(0);
    let mut chain_hashes = vec![lfb_hash.clone()];
    chain_hashes.extend(
        dag.main_parent_chain(lfb_hash, scan_floor)
            .map_err(|e| eyre::eyre!("main_parent_chain failed: {}", e))?,
    );

    // Collect events (highest block_number first, since we walked from LFB).
    let mut rejection_count: u32 = 0;
    let mut has_failed_finalized = false;
    let mut clean_finalized_height: Option<i64> = None;
    let mut latest_event: Option<(i64, BlockHash)> = None;

    // Track latest rejection height seen. A clean inclusion gets
    // invalidated if a rejection at a strictly higher height is observed.
    let mut latest_rejected_finalized_height: Option<i64> = None;

    for candidate_hash in &chain_hashes {
        let height = match dag.block_number(candidate_hash) {
            Some(h) => h,
            None => {
                tracing::debug!(
                    "deploy_finalization_status: no block_number for candidate {} — \
                     skipping (likely cleanup race or partial DAG)",
                    PrettyPrinter::build_string_bytes(candidate_hash)
                );
                continue;
            }
        };
        let candidate_block = match block_store.get(candidate_hash) {
            Ok(Some(b)) => b,
            Ok(None) => {
                tracing::warn!(
                    "deploy_finalization_status: canonical-chain block {} absent from store — \
                     scan may miss deploy events in this block",
                    PrettyPrinter::build_string_bytes(candidate_hash)
                );
                continue;
            }
            Err(e) => {
                tracing::warn!(
                    "deploy_finalization_status: block_store.get failed for {}: {} — \
                     continuing scan; result may be incomplete",
                    PrettyPrinter::build_string_bytes(candidate_hash),
                    e
                );
                continue;
            }
        };

        let mut seen_here = false;
        for pd in &candidate_block.body.deploys {
            if pd.deploy.sig == sig_bytes {
                seen_here = true;
                if pd.is_failed {
                    has_failed_finalized = true;
                } else if clean_finalized_height.map(|h| height > h).unwrap_or(true) {
                    clean_finalized_height = Some(height);
                }
                break;
            }
        }
        for rd in &candidate_block.body.rejected_deploys {
            if rd.sig == sig_bytes {
                seen_here = true;
                rejection_count = rejection_count.saturating_add(1);
                if latest_rejected_finalized_height
                    .map(|h| height > h)
                    .unwrap_or(true)
                {
                    latest_rejected_finalized_height = Some(height);
                }
                break;
            }
        }
        if seen_here
            && latest_event
                .as_ref()
                .map(|(h, _)| height > *h)
                .unwrap_or(true)
        {
            latest_event = Some((height, candidate_hash.clone()));
        }
    }

    // Rejection after a clean inclusion invalidates that inclusion:
    // the rejected-deploy recovery mechanism means the deploy may
    // still land in a later block. Report `Pending` so the client
    // keeps polling.
    if let (Some(clean_h), Some(reject_h)) =
        (clean_finalized_height, latest_rejected_finalized_height)
    {
        if reject_h > clean_h {
            clean_finalized_height = None;
        }
    }

    // Also account for latest_block_hash via the first-seen lookup —
    // covers the case where the sig lives only in a non-finalized
    // block (outside the finalized scan). If the first-seen block
    // somehow has no height entry, skip this fallback rather than
    // record a block_number=0 which would mis-sort against real
    // canonical events.
    if latest_event.is_none() {
        if let Some(first_seen_height) = dag.block_number(&first_seen_block_hash) {
            latest_event = Some((first_seen_height, first_seen_block_hash.clone()));
        } else {
            tracing::debug!(
                "deploy_finalization_status: first-seen block {} has no block_number — \
                 leaving latest_block_hash empty rather than record with bogus height",
                PrettyPrinter::build_string_bytes(&first_seen_block_hash)
            );
        }
    }

    // Expiry rule: tip height strictly past valid_after + deployLifespan
    // AND no clean finalized inclusion. Use the DAG's overall latest
    // block number (may be higher than LFB height) as tip.
    let tip_height = dag.latest_block_number();
    let expired =
        tip_height > valid_after_block_number + deploy_lifespan && clean_finalized_height.is_none();

    let state = if has_failed_finalized {
        DeployFinalizationState::Failed
    } else if clean_finalized_height.is_some() {
        DeployFinalizationState::Finalized
    } else if expired {
        DeployFinalizationState::Expired
    } else {
        DeployFinalizationState::Pending
    };

    Ok(DeployFinalizationStatus {
        state,
        rejection_count,
        latest_block_hash: latest_event.map(|(_, h)| h),
    })
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
