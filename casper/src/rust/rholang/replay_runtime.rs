// See casper/src/main/scala/coop/rchain/casper/rholang/RuntimeReplaySyntax.scala

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    future::Future,
    sync::{Arc, Mutex},
};

use models::{
    rhoapi::Par,
    rust::{
        block::state_hash::StateHash,
        block_hash::BlockHash,
        casper::protocol::casper_message::{
            Event, ProcessedDeploy, ProcessedSystemDeploy, SystemDeployData,
        },
        validator::Validator,
    },
};
use rholang::rust::interpreter::{
    interpreter::EvaluateResult,
    rho_runtime::{RhoRuntime, RhoRuntimeImpl},
    system_processes::BlockData,
};
use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash, history::Either,
    merger::merging_logic::NumberChannelsEndVal,
};

use crate::rust::{
    errors::CasperError,
    util::{
        event_converter,
        rholang::{
            costacc::{
                close_block_deploy::CloseBlockDeploy, pre_charge_deploy::PreChargeDeploy,
                refund_deploy::RefundDeploy, slash_deploy::SlashDeploy,
            },
            interpreter_util,
            replay_failure::ReplayFailure,
            system_deploy::SystemDeployTrait,
            system_deploy_util,
        },
    },
};

use super::runtime::{RuntimeOps, SysEvalResult};

pub struct ReplayRuntimeOps {
    pub runtime_ops: RuntimeOps,
}

impl ReplayRuntimeOps {
    pub fn new(runtime_ops: RuntimeOps) -> Self {
        Self { runtime_ops }
    }

    pub fn new_from_runtime(runtime: RhoRuntimeImpl) -> Self {
        Self {
            runtime_ops: RuntimeOps::new(runtime),
        }
    }

    /* REPLAY Compute state with deploys (genesis block) and System deploys (regular block) */

    /**
     * Evaluates (and validates) deploys and System deploys with checkpoint to valiate final state hash
     */
    #[tracing::instrument(name = "create-replay-runtime", target = "rchain.rholang.runtime.create-replay", skip_all)]
    pub async fn replay_compute_state(
        &mut self,
        start_hash: &StateHash,
        terms: Vec<ProcessedDeploy>,
        system_deploys: Vec<ProcessedSystemDeploy>,
        block_data: &BlockData,
        invalid_blocks: Option<HashMap<BlockHash, Validator>>,
        is_genesis: bool, //FIXME have a better way of knowing this. Pass the replayDeploy function maybe? - OLD
    ) -> Result<(Blake2b256Hash, Vec<NumberChannelsEndVal>), CasperError> {
        let invalid_blocks = invalid_blocks.unwrap_or_default();

        self.runtime_ops.runtime.set_block_data(block_data.clone()).await;
        self.runtime_ops.runtime.set_invalid_blocks(invalid_blocks).await;

        self.replay_deploys(start_hash, terms, system_deploys, !is_genesis, block_data)
            .await
    }

    /* REPLAY Deploy evaluators */

    /**
     * Evaluates (and validates) deploys on root hash with checkpoint to validate final state hash
     */
    pub async fn replay_deploys(
        &mut self,
        start_hash: &StateHash,
        terms: Vec<ProcessedDeploy>,
        system_deploys: Vec<ProcessedSystemDeploy>,
        with_cost_accounting: bool,
        block_data: &BlockData,
    ) -> Result<(Blake2b256Hash, Vec<NumberChannelsEndVal>), CasperError> {
        self.runtime_ops
            .runtime
            .reset(&Blake2b256Hash::from_bytes_prost(start_hash));

        let mut deploy_results = Vec::new();
        for term in terms {
            let result = self.replay_deploy_e(with_cost_accounting, &term).await?;
            deploy_results.push(result);
        }

        let mut system_deploy_results = Vec::new();
        for system_deploy in system_deploys {
            let result = self
                .replay_block_system_deploy(block_data, &system_deploy)
                .await?;
            system_deploy_results.push(result);
        }

        let mut all_mergeable = Vec::new();
        all_mergeable.extend(deploy_results);
        all_mergeable.extend(system_deploy_results);

        // Using tracing events for async - Span[F].traceI("create-checkpoint") from Scala
        tracing::debug!(target: "rchain.casper.replay-rho-runtime", "create-checkpoint-started");
        let checkpoint = self.runtime_ops.runtime.create_checkpoint();
        tracing::debug!(target: "rchain.casper.replay-rho-runtime", "create-checkpoint-finished");
        Ok((checkpoint.root, all_mergeable))
    }

    /**
     * REPLAY Evaluates deploy
     */
    pub async fn replay_deploy(
        &mut self,
        with_cost_accounting: bool,
        processed_deploy: &ProcessedDeploy,
    ) -> Option<CasperError> {
        match self
            .replay_deploy_e(with_cost_accounting, processed_deploy)
            .await
        {
            Ok(_) => None,
            Err(err) => Some(err),
        }
    }

    pub async fn replay_deploy_e(
        &mut self,
        with_cost_accounting: bool,
        processed_deploy: &ProcessedDeploy,
    ) -> Result<NumberChannelsEndVal, CasperError> {
        // Using tracing events for async - Span[F].traceI("replay-deploy") from Scala
        tracing::debug!(target: "rchain.casper.replay-rho-runtime", "replay-deploy-started");
        let mergeable_channels = Arc::new(Mutex::new(HashSet::new()));

        self.rig(processed_deploy)?;

        let eval_successful = if with_cost_accounting {
            self.process_deploy_with_cost_accounting(&processed_deploy, &mergeable_channels)
                .await?
        } else {
            self.process_deploy_without_cost_accounting(&processed_deploy, &mergeable_channels)
                .await?
        };

        self.check_replay_data_with_fix(eval_successful)?;

        let final_mergeable = mergeable_channels.lock().unwrap().clone();
        let channels_data = self
            .runtime_ops
            .get_number_channels_data(&final_mergeable)?;

        Ok(channels_data)
    }

    async fn process_deploy_with_cost_accounting(
        &mut self,
        processed_deploy: &ProcessedDeploy,
        mergeable_channels: &Arc<Mutex<HashSet<Par>>>,
    ) -> Result<bool, CasperError> {
        let mut pre_charge_deploy = PreChargeDeploy {
            charge_amount: processed_deploy.deploy.data.total_phlo_charge(),
            pk: processed_deploy.deploy.pk.clone(),
            rand: system_deploy_util::generate_pre_charge_deploy_random_seed(
                &processed_deploy.deploy,
            ),
        };

        tracing::debug!(target: "rchain.casper.replay-rho-runtime", "precharge-started");
        let precharge_result = self
            .replay_system_deploy_internal(
                &mut pre_charge_deploy,
                &processed_deploy.system_deploy_error,
            )
            .await;

        match precharge_result {
            Ok((_, mut system_eval_result)) => {
                self.runtime_ops.runtime.create_soft_checkpoint();

                if system_eval_result.errors.is_empty() {
                    let mut mc_lock = mergeable_channels.lock().unwrap();
                    mc_lock.extend(system_eval_result.mergeable.drain());
                }
                tracing::debug!(target: "rchain.casper.replay-rho-runtime", "precharge-done");
            }
            Err(err) => {
                return Err(err);
            }
        };

        let eval_successful = if processed_deploy.system_deploy_error.is_none() {
            // Run the user deploy in a transaction
            let (_, successful) = self
                .run_user_deploy(processed_deploy, mergeable_channels)
                .await?;

            if successful {
                self.runtime_ops.runtime.create_soft_checkpoint();
            }
            tracing::debug!(target: "rchain.casper.replay-rho-runtime", "deploy-eval-done");

            tracing::debug!(target: "rchain.casper.replay-rho-runtime", "refund-started");
            let mut refund_deploy = RefundDeploy {
                refund_amount: processed_deploy.refund_amount(),
                rand: system_deploy_util::generate_refund_deploy_random_seed(
                    &processed_deploy.deploy,
                ),
            };

            let refund_result = self
                .replay_system_deploy_internal(&mut refund_deploy, &None)
                .await;

            match refund_result {
                Ok((_, mut system_eval_result)) => {
                    self.runtime_ops.runtime.create_soft_checkpoint();

                    if system_eval_result.errors.is_empty() {
                        let mut mc_lock = mergeable_channels.lock().unwrap();
                        mc_lock.extend(system_eval_result.mergeable.drain());
                    }
                    tracing::debug!(target: "rchain.casper.replay-rho-runtime", "refund-done");
                }
                Err(err) => {
                    return Err(err);
                }
            }

            successful
        } else {
            // If there was an expected failure in the system deploy, skip user deploy execution
            true
        };

        tracing::debug!(target: "rchain.casper.replay-rho-runtime", "deploy-done");
        Ok(eval_successful)
    }

    async fn process_deploy_without_cost_accounting(
        &mut self,
        processed_deploy: &ProcessedDeploy,
        mergeable_channels: &Arc<Mutex<HashSet<Par>>>,
    ) -> Result<bool, CasperError> {
        self.run_user_deploy(processed_deploy, mergeable_channels)
            .await
            .map(|(_, eval_successful)| eval_successful)
    }

    async fn run_user_deploy(
        &mut self,
        processed_deploy: &ProcessedDeploy,
        mergeable_channels: &Arc<Mutex<HashSet<Par>>>,
    ) -> Result<(EvaluateResult, bool), CasperError> {
        let fallback = self.runtime_ops.runtime.create_soft_checkpoint();

        let mut user_eval_result = self.runtime_ops.evaluate(&processed_deploy.deploy).await?;

        let eval_successful = user_eval_result.errors.is_empty();

        if !eval_successful {
            interpreter_util::print_deploy_errors(
                &processed_deploy.deploy.sig,
                &user_eval_result.errors,
            );
            self.runtime_ops.runtime.revert_to_soft_checkpoint(fallback);
        } else {
            let mut mc_lock = mergeable_channels.lock().unwrap();
            mc_lock.extend(user_eval_result.mergeable.drain());
        }

        // Verify that our execution matches the expected result
        if processed_deploy.is_failed != !eval_successful {
            return Err(CasperError::ReplayFailure(
                ReplayFailure::replay_status_mismatch(processed_deploy.is_failed, !eval_successful),
            ));
        }

        if processed_deploy.cost.cost != user_eval_result.cost.value as u64 {
            return Err(CasperError::ReplayFailure(
                ReplayFailure::replay_cost_mismatch(
                    processed_deploy.cost.cost,
                    user_eval_result.cost.value as u64,
                ),
            ));
        }

        Ok((user_eval_result, eval_successful))
    }

    /* REPLAY System deploy evaluators */

    /**
     * Evaluates System deploy with checkpoint to get final state hash
     */
    pub async fn replay_block_system_deploy(
        &mut self,
        block_data: &BlockData,
        processed_system_deploy: &ProcessedSystemDeploy,
    ) -> Result<NumberChannelsEndVal, CasperError> {
        // Using tracing events for async - Span[F].traceI("replay-sys-deploy") from Scala
        tracing::debug!(target: "rchain.casper.replay-rho-runtime", "replay-sys-deploy-started");
        let system_deploy = match processed_system_deploy {
            ProcessedSystemDeploy::Succeeded {
                ref system_deploy, ..
            } => system_deploy,
            ProcessedSystemDeploy::Failed { .. } => &SystemDeployData::Empty,
        };

        match system_deploy {
            SystemDeployData::Slash {
                invalid_block_hash,
                issuer_public_key,
            } => {
                let mut slash_deploy = SlashDeploy {
                    invalid_block_hash: invalid_block_hash.clone(),
                    pk: issuer_public_key.clone(),
                    initial_rand: system_deploy_util::generate_slash_deploy_random_seed(
                        block_data.sender.bytes.clone(),
                        block_data.seq_num,
                    ),
                };

                self.rig_system_deploy(processed_system_deploy)?;
                let (map, eval_res) = self
                    .replay_system_deploy_internal(&mut slash_deploy, &None)
                    .await
                    .map(|(_, eval_result)| {
                        if eval_result.errors.is_empty() {
                            self.runtime_ops.runtime.create_soft_checkpoint();
                        }

                        let data = self
                            .runtime_ops
                            .get_number_channels_data(&eval_result.mergeable)?;
                        Ok::<(BTreeMap<Blake2b256Hash, i64>, EvaluateResult), CasperError>((
                            data,
                            eval_result,
                        ))
                    })??;

                self.check_replay_data_with_fix(eval_res.errors.is_empty())?;
                Ok(map)
            }

            SystemDeployData::CloseBlockSystemDeployData => {
                let mut close_block_deploy = CloseBlockDeploy {
                    initial_rand:
                        system_deploy_util::generate_close_deploy_random_seed_from_validator(
                            block_data.sender.bytes.clone(),
                            block_data.seq_num,
                        ),
                };

                self.rig_system_deploy(processed_system_deploy)?;

                let (map, eval_res) = self
                    .replay_system_deploy_internal(&mut close_block_deploy, &None)
                    .await
                    .map(|(_, eval_result)| {
                        if eval_result.errors.is_empty() {
                            self.runtime_ops.runtime.create_soft_checkpoint();
                        }

                        let data = self
                            .runtime_ops
                            .get_number_channels_data(&eval_result.mergeable)?;
                        Ok::<(BTreeMap<Blake2b256Hash, i64>, EvaluateResult), CasperError>((
                            data,
                            eval_result,
                        ))
                    })??;

                self.check_replay_data_with_fix(eval_res.errors.is_empty())?;
                Ok(map)
            }

            SystemDeployData::Empty => Err(CasperError::ReplayFailure(
                ReplayFailure::internal_error("Expected system deploy".to_string()),
            )),
        }
    }

    pub async fn replay_system_deploy_internal<S: SystemDeployTrait>(
        &mut self,
        system_deploy: &mut S,
        expected_failure_msg: &Option<String>,
    ) -> Result<SysEvalResult<S>, CasperError> {
        // Using tracing events for async - Span[F].withMarks("replay-system-deploy") from Scala
        tracing::debug!(target: "rchain.casper.replay-rho-runtime", "replay-system-deploy-started");
        let (result, eval_res) = self.runtime_ops.eval_system_deploy(system_deploy).await?;

        // Compare evaluation from play and replay, successful or failed
        match (expected_failure_msg, &result) {
            // Valid replay
            (None, Either::Right(_)) => {
                // Replayed successful execution
                Ok((result, eval_res))
            }
            (Some(expected_error), Either::Left(error)) => {
                let actual_error = &error.error_message;
                if expected_error == actual_error {
                    // Replayed failed execution - error messages match
                    Ok((result, eval_res))
                } else {
                    // Error messages different
                    Err(CasperError::ReplayFailure(
                        ReplayFailure::system_deploy_error_mismatch(
                            expected_error.clone(),
                            actual_error.clone(),
                        ),
                    ))
                }
            }
            // Invalid replay
            (Some(_), Either::Right(_)) => {
                // Error expected, replay successful
                Err(CasperError::ReplayFailure(
                    ReplayFailure::replay_status_mismatch(true, false),
                ))
            }
            (None, Either::Left(_)) => {
                // No error expected, replay failed
                Err(CasperError::ReplayFailure(
                    ReplayFailure::replay_status_mismatch(false, true),
                ))
            }
        }
    }

    /* Helper functions */

    pub async fn rig_with_check<A, F, Fut>(
        &self,
        processed_deploy: &ProcessedDeploy,
        action: F,
    ) -> Result<(A, bool), CasperError>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<(A, bool), CasperError>>,
    {
        // Rig the events first (synchronous operation)
        self.rig(processed_deploy)?;

        // Execute the provided async action
        let action_result = action().await;

        match action_result {
            Ok((value, eval_successful)) => {
                match self.check_replay_data_with_fix(eval_successful) {
                    Ok(_) => Ok((value, eval_successful)),
                    Err(replay_failure) => Err(CasperError::ReplayFailure(replay_failure)),
                }
            }
            Err(e) => Err(e),
        }
    }

    pub async fn rig_with_check_system_deploy<A, F, Fut>(
        &self,
        processed_system_deploy: &ProcessedSystemDeploy,
        action: F,
    ) -> Result<(A, EvaluateResult), CasperError>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<(A, EvaluateResult), CasperError>>,
    {
        self.rig_system_deploy(processed_system_deploy)?;
        let (value, eval_res) = action().await?;
        self.check_replay_data_with_fix(eval_res.errors.is_empty())?;
        Ok((value, eval_res))
    }

    pub fn rig(&self, processed_deploy: &ProcessedDeploy) -> Result<(), CasperError> {
        Ok(self.runtime_ops.runtime.rig(
            processed_deploy
                .deploy_log
                .iter()
                .map(event_converter::to_rspace_event)
                .collect(),
        )?)
    }

    pub fn rig_system_deploy(
        &self,
        processed_system_deploy: &ProcessedSystemDeploy,
    ) -> Result<(), CasperError> {
        let event_list = match processed_system_deploy {
            ProcessedSystemDeploy::Succeeded { event_list, .. } => event_list,
            ProcessedSystemDeploy::Failed { event_list, .. } => event_list,
        };

        Ok(self.runtime_ops.runtime.rig(
            event_list
                .iter()
                .map(|event: &Event| event_converter::to_rspace_event(&event))
                .collect(),
        )?)
    }

    pub fn check_replay_data_with_fix(
        &self,
        // https://rchain.atlassian.net/browse/RCHAIN-3505
        _eval_successful: bool,
    ) -> Result<(), ReplayFailure> {
        // Only check replay data for successful evaluations
        match self.runtime_ops.runtime.check_replay_data() {
            Ok(()) => Ok(()),
            Err(err) => {
                let err_msg = err.to_string();
                if err_msg.contains("unused") && err_msg.contains("COMM") {
                    Err(ReplayFailure::unused_comm_event(err_msg))
                } else {
                    Err(ReplayFailure::internal_error(format!(
                        "Replay check failed: {}",
                        err
                    )))
                }
            }
        }
    }
}
