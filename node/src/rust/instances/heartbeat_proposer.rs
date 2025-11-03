use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use casper::rust::blocks::proposer::proposer::ProposerResult;
use casper::rust::casper::{CasperSnapshot, MultiParentCasper};
use casper::rust::casper_conf::HeartbeatConf;
use casper::rust::engine::engine_cell::EngineCell;
use casper::rust::validator_identity::ValidatorIdentity;
use models::rust::casper::protocol::casper_message::BlockMessage;
use rand::Rng;

use casper::rust::ProposeFunction;

/// Heartbeat proposer that periodically checks if a block
/// needs to be proposed to maintain liveness.
pub struct HeartbeatProposer;

impl HeartbeatProposer {
    /// Create a heartbeat proposer stream that periodically checks if a block
    /// needs to be proposed to maintain liveness.
    ///
    /// This integrates with the existing propose queue mechanism for thread safety.
    /// The heartbeat simply calls the same triggerPropose function that user deploys
    /// and explicit propose calls use, ensuring serialization through ProposerInstance.
    ///
    /// To prevent lock-step behavior between validators, the stream waits a random
    /// amount of time (0 to checkInterval) before starting the periodic checks.
    ///
    /// The heartbeat only runs on bonded validators. It checks the active validators
    /// set before proposing to avoid unnecessary attempts by unbonded nodes.
    ///
    /// # Arguments
    ///
    /// * `engine_cell` - The EngineCell to read the current Casper instance from
    /// * `trigger_propose_f` - The propose function that integrates with the propose queue
    /// * `validator_identity` - The validator identity to check if bonded
    /// * `config` - Heartbeat configuration (enabled, check_interval, max_lfb_age)
    ///
    /// # Returns
    ///
    /// Returns `Some(JoinHandle)` when the heartbeat is spawned, or `None` when
    /// disabled or trigger function is not available.
    pub fn create(
        engine_cell: Arc<EngineCell>,
        trigger_propose_f: Option<Arc<ProposeFunction>>, // same queue/function used by user-triggered proposes
        validator_identity: ValidatorIdentity,
        config: HeartbeatConf,
    ) -> Option<tokio::task::JoinHandle<()>> {
        if !config.enabled {
            return None;
        }

        let trigger = match trigger_propose_f {
            Some(f) => f,
            None => {
                tracing::warn!("Heartbeat: trigger_propose function not available, skipping spawn");
                return None;
            }
        };

        let initial_delay = random_initial_delay(config.check_interval);
        tracing::info!(
            "Heartbeat: Starting with random initial delay of {}s (check interval: {}s, max LFB age: {}s)",
            initial_delay.as_secs(),
            config.check_interval.as_secs(),
            config.max_lfb_age.as_secs()
        );

        let handle = tokio::spawn(async move {
            tokio::time::sleep(initial_delay).await;

            let mut ticker = tokio::time::interval(config.check_interval);

            loop {
                // wait next tick
                ticker.tick().await;

                tracing::debug!("Heartbeat: Checking if propose is needed");
                let eng = engine_cell.get().await;

                // Access Casper if available and run the check
                if let Some(casper) = eng.with_casper() {
                    let _ =
                        do_heartbeat_check(casper, &*trigger, &validator_identity, &config).await;
                } else {
                    tracing::debug!("Heartbeat: Casper not available yet, skipping check");
                }
            }
        });

        Some(handle)
    }
}

fn random_initial_delay(check_interval: Duration) -> Duration {
    let max_millis = check_interval.as_millis() as u64;
    let random_millis = rand::rng().random_range(0..=max_millis);
    Duration::from_millis(random_millis)
}

async fn do_heartbeat_check(
    casper: Arc<dyn MultiParentCasper + Send + Sync>,
    trigger_propose: &ProposeFunction,
    validator_identity: &ValidatorIdentity,
    config: &HeartbeatConf,
) -> Result<(), casper::rust::errors::CasperError> {
    let snapshot: CasperSnapshot = casper.get_snapshot().await?;

    let is_bonded = snapshot
        .on_chain_state
        .active_validators
        .contains(&validator_identity.public_key.bytes);

    if !is_bonded {
        tracing::info!("Heartbeat: Validator is not bonded, skipping heartbeat propose");
    } else {
        tracing::debug!("Heartbeat: Validator is bonded, checking LFB age");
        check_lfb_and_propose(casper.clone(), trigger_propose, config).await?;
    }

    Ok(())
}

async fn check_lfb_and_propose(
    casper: Arc<dyn MultiParentCasper + Send + Sync>,
    trigger_propose: &ProposeFunction,
    config: &HeartbeatConf,
) -> Result<(), casper::rust::errors::CasperError> {
    let lfb: BlockMessage = casper.last_finalized_block().await?;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u128)
        .unwrap_or(0);

    let lfb_timestamp_ms = lfb.header.timestamp as u128;

    let time_since_lfb = now - lfb_timestamp_ms;

    let should_propose = time_since_lfb > config.max_lfb_age.as_millis();

    if should_propose {
        tracing::info!(
            "Heartbeat: LFB is {}ms old (threshold: {}ms), triggering propose",
            time_since_lfb,
            config.max_lfb_age.as_millis()
        );

        let result = trigger_propose(casper.clone(), false).await?;
        match result {
            ProposerResult::Empty => {
                tracing::debug!("Heartbeat: Propose already in progress, will retry next check");
            }
            ProposerResult::Failure(status, seq_num) => {
                tracing::warn!(
                    "Heartbeat: Propose failed with {} (seqNum {})",
                    status,
                    seq_num
                );
            }
            ProposerResult::Success(_, _) => {
                tracing::info!("Heartbeat: Successfully created block");
            }
            ProposerResult::Started(seq_num) => {
                tracing::info!("Heartbeat: Async propose started (seqNum {})", seq_num);
            }
        }
    } else {
        tracing::debug!(
            "Heartbeat: LFB age is {}ms, no action needed",
            time_since_lfb
        );
    }

    Ok(())
}
