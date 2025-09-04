// See casper/src/main/scala/coop/rchain/casper/LastFinalizedHeightConstraintChecker.scala

use super::{
    blocks::proposer::propose_result::CheckProposeConstraintsResult, casper::CasperSnapshot,
    errors::CasperError, validator_identity::ValidatorIdentity,
};

pub fn check(
    snapshot: &CasperSnapshot,
    validator_identity: &ValidatorIdentity,
) -> Result<CheckProposeConstraintsResult, CasperError> {
    let validator = validator_identity.public_key.bytes.clone();
    let last_finalized_block_hash = snapshot.dag.last_finalized_block();
    let height_constraint_threshold = snapshot
        .on_chain_state
        .shard_conf
        .height_constraint_threshold;

    let last_finalized_block = snapshot.dag.lookup_unsafe(&last_finalized_block_hash)?;
    let latest_message_opt = snapshot.dag.latest_message(&validator)?;
    let latest_message = latest_message_opt.ok_or(CasperError::Other(
        "Last finalized height constraint checker: Validator does not have a latest message"
            .to_string(),
    ))?;

    let height_difference = latest_message.block_number - last_finalized_block.block_number;

    log::info!(
        "Latest message is {} blocks ahead of the last finalized block",
        height_difference
    );

    if height_difference <= height_constraint_threshold {
        Ok(CheckProposeConstraintsResult::success())
    } else {
        Ok(CheckProposeConstraintsResult::too_far_ahead_of_last_finalized())
    }
}
