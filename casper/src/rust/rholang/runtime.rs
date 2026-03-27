// See casper/src/main/scala/coop/rchain/casper/rholang/RuntimeSyntax.scala

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    future::Future,
    mem,
    sync::OnceLock,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use crypto::rust::{
    hash::blake2b512_random::Blake2b512Random,
    public_key::PublicKey,
    signatures::{secp256k1::Secp256k1, signatures_alg::SignaturesAlg, signed::Signed},
};
use models::{
    rhoapi::{
        expr::ExprInstance, g_unforgeable::UnfInstance, tagged_continuation::TaggedCont,
        BindPattern, GPrivate, GUnforgeable, ListParWithRandom, Par, TaggedContinuation,
    },
    rust::{
        block::state_hash::StateHash,
        block_hash::BlockHash,
        casper::{
            pretty_printer::PrettyPrinter,
            protocol::casper_message::{
                Bond, DeployData, Event, ProcessedDeploy, ProcessedSystemDeploy, SystemDeployData,
            },
        },
        normalizer_env::normalizer_env_from_deploy,
        par_map_type_mapper::ParMapTypeMapper,
        par_set_type_mapper::ParSetTypeMapper,
        sorted_par_hash_set::SortedParHashSet,
        sorted_par_map::SortedParMap,
        utils::new_freevar_par,
        validator::Validator,
    },
};
use rholang::rust::interpreter::{
    accounting::costs::Cost,
    accounting::has_cost::HasCost,
    compiler::compiler::Compiler,
    env::Env,
    errors::InterpreterError,
    interpreter::EvaluateResult,
    merging::rholang_merging_logic::RholangMergingLogic,
    rho_runtime::{bootstrap_registry, RhoRuntime, RhoRuntimeImpl},
    system_processes::{BlockData, DeployData as SystemProcessDeployData},
};
use rspace_plus_plus::rspace::{
    hashing::{blake2b256_hash::Blake2b256Hash, stable_hash_provider},
    history::{instances::radix_history::RadixHistory, Either},
    merger::merging_logic::NumberChannelsEndVal,
};

use crate::rust::{
    errors::CasperError,
    metrics_constants::{
        BLOCK_REPLAY_SYSDEPLOY_EVAL_CONSUME_RESULT_TIME_METRIC,
        BLOCK_REPLAY_SYSDEPLOY_EVAL_EVALUATE_SOURCE_TIME_METRIC, CASPER_METRICS_SOURCE,
    },
    rholang::types::eval_collector::EvalCollector,
    util::{
        construct_deploy, event_converter,
        rholang::{
            costacc::{
                close_block_deploy::CloseBlockDeploy, pre_charge_deploy::PreChargeDeploy,
                refund_deploy::RefundDeploy, slash_deploy::SlashDeploy,
            },
            interpreter_util,
            system_deploy::SystemDeployTrait,
            system_deploy_result::SystemDeployResult,
            system_deploy_user_error::{SystemDeployPlatformFailure, SystemDeployUserError},
            system_deploy_util,
            tools::Tools,
        },
    },
};

pub struct RuntimeOps {
    pub runtime: RhoRuntimeImpl,
}

impl RuntimeOps {
    pub fn new(runtime: RhoRuntimeImpl) -> Self {
        Self { runtime }
    }
}

#[allow(type_alias_bounds)]
pub type SysEvalResult<S: SystemDeployTrait> =
    (Either<SystemDeployUserError, S::Result>, EvaluateResult);

fn system_deploy_consume_all_pattern() -> BindPattern {
    BindPattern {
        patterns: vec![new_freevar_par(0, Vec::new())],
        remainder: None,
        free_count: 1,
    }
}

impl RuntimeOps {
    /**
     * Because of the history legacy, the emptyStateHash does not really represent an empty trie.
     * The `emptyStateHash` is used as genesis block pre state which the state only contains registry
     * fixed channels in the state.
     */
    pub async fn empty_state_hash(&mut self) -> Result<StateHash, CasperError> {
        self.runtime.reset(&RadixHistory::empty_root_node_hash())?;

        bootstrap_registry(&self.runtime).await;
        let checkpoint = self.runtime.create_checkpoint();
        Ok(checkpoint.root.bytes().into())
    }

    /* Compute state with deploys (genesis block) and System deploys (regular block) */

    /**
     * Evaluates deploys and System deploys with checkpoint to get final state hash
     */
    pub async fn compute_state(
        &mut self,
        start_hash: &StateHash,
        terms: Vec<Signed<DeployData>>,
        system_deploys: Vec<crate::rust::util::rholang::system_deploy_enum::SystemDeployEnum>,
        block_data: BlockData,
        invalid_blocks: HashMap<BlockHash, Validator>,
    ) -> Result<
        (
            StateHash,
            Vec<(ProcessedDeploy, NumberChannelsEndVal)>,
            Vec<(ProcessedSystemDeploy, NumberChannelsEndVal)>,
        ),
        CasperError,
    > {
        let mem_profile_enabled = std::env::var("F1R3_BLOCK_CREATOR_PHASE_SUBSTEP_PROFILE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let read_vm_rss_kb = || -> Option<usize> {
            let status = std::fs::read_to_string("/proc/self/status").ok()?;
            status
                .lines()
                .find(|line| line.starts_with("VmRSS:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|value| value.parse::<usize>().ok())
        };
        let mut rss_baseline = if mem_profile_enabled {
            read_vm_rss_kb()
        } else {
            None
        };
        let mut rss_prev = rss_baseline;
        let mut log_mem_step = |step: &str| {
            if !mem_profile_enabled {
                return;
            }
            if let Some(curr) = read_vm_rss_kb() {
                let prev = rss_prev.unwrap_or(curr);
                let baseline = rss_baseline.unwrap_or(curr);
                eprintln!(
                    "compute_state.mem step={} rss_kb={} delta_prev_kb={} delta_total_kb={}",
                    step,
                    curr,
                    curr as i64 - prev as i64,
                    curr as i64 - baseline as i64
                );
                rss_prev = Some(curr);
                if rss_baseline.is_none() {
                    rss_baseline = Some(curr);
                }
            }
        };

        // Using tracing events instead of spans for async context
        // Span[F].traceI("compute-state") equivalent from Scala
        tracing::info!(target: "f1r3fly.casper.runtime", "compute-state-started");
        log_mem_step("start");
        self.runtime.set_block_data(block_data).await;
        log_mem_step("after_set_block_data");
        self.runtime.set_invalid_blocks(invalid_blocks).await;
        log_mem_step("after_set_invalid_blocks");

        let (start_hash, processed_deploys) =
            self.play_deploys_for_state(start_hash, terms).await?;
        log_mem_step("after_play_deploys_for_state");

        let mut current_hash = start_hash;
        let mut processed_system_deploys = Vec::with_capacity(system_deploys.len());

        for (idx, system_deploy_enum) in system_deploys.into_iter().enumerate() {
            if mem_profile_enabled {
                let before_step = format!("before_system_deploy_{}", idx + 1);
                log_mem_step(&before_step);
            }
            // Match on the enum and call appropriate generic method
            let result = match system_deploy_enum {
                crate::rust::util::rholang::system_deploy_enum::SystemDeployEnum::Slash(
                    mut slash_deploy,
                ) => {
                    self.play_system_deploy(&current_hash, &mut slash_deploy)
                        .await?
                }
                crate::rust::util::rholang::system_deploy_enum::SystemDeployEnum::Close(
                    mut close_deploy,
                ) => {
                    self.play_system_deploy(&current_hash, &mut close_deploy)
                        .await?
                }
            };
            if mem_profile_enabled {
                let after_step = format!("after_system_deploy_{}", idx + 1);
                log_mem_step(&after_step);
            }

            match result {
                SystemDeployResult::PlaySucceeded {
                    state_hash,
                    processed_system_deploy,
                    mergeable_channels,
                    result: _,
                } => {
                    processed_system_deploys.push((processed_system_deploy, mergeable_channels));
                    current_hash = state_hash;
                }
                SystemDeployResult::PlayFailed {
                    processed_system_deploy: ProcessedSystemDeploy::Failed { error_msg, .. },
                } => {
                    return Err(CasperError::RuntimeError(format!(
                        "Unexpected system error during play of system deploy: {}",
                        error_msg
                    )))
                }
                SystemDeployResult::PlayFailed {
                    processed_system_deploy: ProcessedSystemDeploy::Succeeded { .. },
                } => {
                    return Err(CasperError::RuntimeError(format!(
                        "Unreachable code path. This is likely caused by a bug in the runtime."
                    )))
                }
            }
        }

        let post_state_hash = current_hash;
        log_mem_step("finish");

        tracing::info!(target: "f1r3fly.casper.runtime", "compute-state-finished");
        Ok((post_state_hash, processed_deploys, processed_system_deploys))
    }

    /**
     * Evaluates genesis deploys with checkpoint to get final state hash
     */
    pub async fn compute_genesis(
        &mut self,
        terms: Vec<Signed<DeployData>>,
        block_time: i64,
        block_number: i64,
    ) -> Result<
        (
            StateHash,
            StateHash,
            Vec<(ProcessedDeploy, NumberChannelsEndVal)>,
        ),
        CasperError,
    > {
        // Using tracing events instead of spans for async context
        // Span[F].traceI("compute-genesis") equivalent from Scala
        tracing::info!(target: "f1r3fly.casper.runtime", "compute-genesis-started");
        self.runtime
            .set_block_data(BlockData {
                time_stamp: block_time,
                block_number,
                sender: PublicKey::from_bytes(&Vec::new()),
                seq_num: 0,
            })
            .await;

        let genesis_pre_state_hash = self.empty_state_hash().await?;
        let play_result = self
            .play_deploys_for_genesis(&genesis_pre_state_hash, terms)
            .await?;

        let (post_state_hash, processed_deploys) = play_result;
        tracing::info!(target: "f1r3fly.casper.runtime", "compute-genesis-finished");
        Ok((genesis_pre_state_hash, post_state_hash, processed_deploys))
    }

    /* Deploy evaluators */

    /**
     * Evaluates deploys on root hash with checkpoint to get final state hash
     */
    pub async fn play_deploys_for_state(
        &mut self,
        start_hash: &StateHash,
        terms: Vec<Signed<DeployData>>,
    ) -> Result<(StateHash, Vec<(ProcessedDeploy, NumberChannelsEndVal)>), CasperError> {
        let mem_profile_enabled = std::env::var("F1R3_BLOCK_CREATOR_PHASE_SUBSTEP_PROFILE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let read_vm_rss_kb = || -> Option<usize> {
            let status = std::fs::read_to_string("/proc/self/status").ok()?;
            status
                .lines()
                .find(|line| line.starts_with("VmRSS:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|value| value.parse::<usize>().ok())
        };
        let mut rss_baseline = if mem_profile_enabled {
            read_vm_rss_kb()
        } else {
            None
        };
        let mut rss_prev = rss_baseline;
        let mut log_mem_step = |step: &str| {
            if !mem_profile_enabled {
                return;
            }
            if let Some(curr) = read_vm_rss_kb() {
                let prev = rss_prev.unwrap_or(curr);
                let baseline = rss_baseline.unwrap_or(curr);
                eprintln!(
                    "play_deploys_for_state.mem step={} rss_kb={} delta_prev_kb={} delta_total_kb={}",
                    step,
                    curr,
                    curr as i64 - prev as i64,
                    curr as i64 - baseline as i64
                );
                rss_prev = Some(curr);
                if rss_baseline.is_none() {
                    rss_baseline = Some(curr);
                }
            }
        };

        // Using tracing events for async - Span[F].withMarks("play-deploys") from Scala
        tracing::info!(target: "f1r3fly.casper.play-deploys", "play-deploys-started");
        log_mem_step("start");
        self.runtime
            .reset(&Blake2b256Hash::from_bytes_prost(start_hash))?;
        log_mem_step("after_reset");

        let mut res = Vec::with_capacity(terms.len());
        for (idx, deploy) in terms.into_iter().enumerate() {
            if mem_profile_enabled {
                let before = format!("before_deploy_{}", idx + 1);
                log_mem_step(&before);
            }
            res.push(self.play_deploy_with_cost_accounting(deploy).await?);
            if mem_profile_enabled {
                let after = format!("after_deploy_{}", idx + 1);
                log_mem_step(&after);
            }
        }

        log_mem_step("before_final_checkpoint");
        log_mem_step("before_final_checkpoint_create_checkpoint");
        let final_checkpoint = self.runtime.create_checkpoint();
        tracing::info!(
            target: "f1r3fly.rspace",
            checkpoint_root = %hex::encode(final_checkpoint.root.bytes()),
            deploys_count = res.len(),
            "play_deploys_for_state: checkpoint completed"
        );
        log_mem_step("after_final_checkpoint_create_checkpoint");
        log_mem_step("before_final_checkpoint_root_to_bytes");
        let final_root = final_checkpoint.root.to_bytes_prost();
        log_mem_step("after_final_checkpoint_root_to_bytes");
        log_mem_step("after_final_checkpoint");
        Ok((final_root, res))
    }

    /**
     * Evaluates deploys on root hash with checkpoint to get final state hash
     */
    pub async fn play_deploys_for_genesis(
        &mut self,
        start_hash: &StateHash,
        terms: Vec<Signed<DeployData>>,
    ) -> Result<(StateHash, Vec<(ProcessedDeploy, NumberChannelsEndVal)>), CasperError> {
        // Using tracing events for async - Span[F].withMarks("play-deploys") from Scala
        tracing::info!(target: "f1r3fly.casper.play-deploys-genesis", "play-deploys-genesis-started");
        self.runtime
            .reset(&Blake2b256Hash::from_bytes_prost(start_hash))?;

        let mut res = Vec::with_capacity(terms.len());
        for deploy in terms {
            res.push(self.process_deploy_with_mergeable_data(deploy).await?);
        }

        let final_checkpoint = self.runtime.create_checkpoint();
        Ok((final_checkpoint.root.to_bytes_prost(), res))
    }

    /**
     * Evaluates deploy with cost accounting (PoS Pre-charge and Refund calls)
     */
    pub async fn play_deploy_with_cost_accounting(
        &mut self,
        deploy: Signed<DeployData>,
    ) -> Result<(ProcessedDeploy, NumberChannelsEndVal), CasperError> {
        let mem_profile_enabled = std::env::var("F1R3_BLOCK_CREATOR_PHASE_SUBSTEP_PROFILE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let read_vm_rss_kb = || -> Option<usize> {
            let status = std::fs::read_to_string("/proc/self/status").ok()?;
            status
                .lines()
                .find(|line| line.starts_with("VmRSS:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|value| value.parse::<usize>().ok())
        };
        let mut rss_baseline = if mem_profile_enabled {
            read_vm_rss_kb()
        } else {
            None
        };
        let mut rss_prev = rss_baseline;
        let mut log_mem_step = |step: &str| {
            if !mem_profile_enabled {
                return;
            }
            if let Some(curr) = read_vm_rss_kb() {
                let prev = rss_prev.unwrap_or(curr);
                let baseline = rss_baseline.unwrap_or(curr);
                eprintln!(
                    "play_deploy_with_cost_accounting.mem step={} rss_kb={} delta_prev_kb={} delta_total_kb={}",
                    step,
                    curr,
                    curr as i64 - prev as i64,
                    curr as i64 - baseline as i64
                );
                rss_prev = Some(curr);
                if rss_baseline.is_none() {
                    rss_baseline = Some(curr);
                }
            }
        };

        // Using tracing events for async - Span[F].withMarks("play-deploy") from Scala
        tracing::debug!(target: "f1r3fly.casper.play-deploy", "play-deploy-started");
        log_mem_step("start");
        let mut eval_collector_state = EvalCollector::new();

        let deploy_pk = deploy.pk.bytes.clone();
        let deploy_pk_hex = hex::encode(&deploy_pk);
        let deploy_sig_hex = hex::encode(&deploy.sig);
        let refund_rand = system_deploy_util::generate_refund_deploy_random_seed(&deploy);
        let pre_charge_rand = system_deploy_util::generate_pre_charge_deploy_random_seed(&deploy);

        // Evaluates Pre-charge system deploy
        let pre_charge_result = {
            // Using tracing events for async - Span[F].traceI("precharge") from Scala
            tracing::debug!(target: "f1r3fly.casper.precharge", "precharge-started");
            tracing::debug!(
                "PreCharging {} for {}",
                deploy_pk_hex.as_str(),
                deploy.data.total_phlo_charge()
            );
            log_mem_step("before_precharge_internal");
            let (event_log, result, mergeable_channels) = self
                .play_system_deploy_internal(&mut PreChargeDeploy {
                    charge_amount: deploy.data.total_phlo_charge(),
                    pk: deploy.pk.clone(),
                    rand: pre_charge_rand,
                })
                .await?;
            log_mem_step("after_precharge_internal");
            eval_collector_state.add(event_log, mergeable_channels);
            log_mem_step("after_precharge_collect");
            result
        };
        log_mem_step("after_precharge");

        match pre_charge_result {
            Either::Right(_) => {
                // Evaluates user deploy
                let pd = {
                    // Using tracing events for async - Span[F].traceI("user-deploy") from Scala
                    tracing::debug!(target: "f1r3fly.casper.user-deploy", "user-deploy-started");
                    tracing::debug!("Processing user deploy {}", deploy_pk_hex.as_str());
                    // Evaluates user deploy and append event log to local state
                    {
                        let (mut pd, mc) = self.process_deploy(deploy).await?;
                        let deploy_log = mem::take(&mut pd.deploy_log);
                        eval_collector_state.add(deploy_log, mc);
                        pd
                    }
                };
                log_mem_step("after_user_deploy");

                // Evaluates Refund system deploy
                let refund_result = {
                    // Using tracing events for async - Span[F].traceI("refund") from Scala
                    tracing::debug!(target: "f1r3fly.casper.refund", "refund-started");
                    tracing::debug!(
                        "Refunding {} with {}",
                        deploy_pk_hex.as_str(),
                        pd.refund_amount()
                    );
                    let (event_log, result, mergeable_channels) = self
                        .play_system_deploy_internal(&mut RefundDeploy {
                            refund_amount: pd.refund_amount(),
                            rand: refund_rand,
                        })
                        .await?;
                    eval_collector_state.add(event_log, mergeable_channels);
                    result
                };
                log_mem_step("after_refund");

                match refund_result {
                    Either::Right(_) => {
                        // Get mergeable channels data
                        let mergeable_channels_data = self
                            .get_number_channels_data(&eval_collector_state.mergeable_channels)?;

                        let deploy_log = mem::take(&mut eval_collector_state.event_log);
                        log_mem_step("after_collect_result");

                        Ok((
                            ProcessedDeploy { deploy_log, ..pd },
                            mergeable_channels_data,
                        ))
                    }

                    Either::Left(error) => {
                        // If Pre-charge succeeds and Refund fails, it's a platform error.
                        // Include deploy identifiers so operators can quickly isolate toxic deploys.
                        let refund_amount = pd.refund_amount();
                        let failure_context = format!(
                            "{}, deploy_sig={}, deployer_pk={}, refund_amount={}",
                            error.error_message,
                            deploy_sig_hex,
                            deploy_pk_hex.as_str(),
                            refund_amount
                        );
                        metrics::counter!(
                            "casper_runtime_refund_failures_total",
                            "source" => CASPER_METRICS_SOURCE
                        )
                        .increment(1);
                        tracing::warn!("Refund failure '{}'", failure_context);
                        Err(CasperError::SystemRuntimeError(
                            SystemDeployPlatformFailure::GasRefundFailure(failure_context),
                        ))
                    }
                }
            }

            Either::Left(error) => {
                tracing::error!("Pre-charge failure '{}'", error.error_message);

                // Handle evaluation errors from PreCharge
                // - assigning 0 cost - replay should reach the same state
                let mut empty_pd = ProcessedDeploy::empty(deploy);
                empty_pd.system_deploy_error = Some(error.error_message);

                // Update result with accumulated event logs
                // Get mergeable channels data
                let mergeable_channels_data =
                    self.get_number_channels_data(&eval_collector_state.mergeable_channels)?;

                let deploy_log = mem::take(&mut eval_collector_state.event_log);

                Ok((
                    ProcessedDeploy {
                        deploy_log,
                        ..empty_pd
                    },
                    mergeable_channels_data,
                ))
            }
        }
    }

    pub async fn process_deploy(
        &mut self,
        deploy: Signed<DeployData>,
    ) -> Result<(ProcessedDeploy, HashSet<Par>), CasperError> {
        // Keep a soft checkpoint before user deploy execution so failed deploy rollback
        // preserves pre-charge side effects required by refundDeploy.
        let fallback = self.runtime.create_soft_checkpoint();

        // Evaluate deploy
        rholang::rust::interpreter::storage::charging_rspace::reset_cost_trace_seq();
        let eval_result = self.evaluate(&deploy).await?;

        tracing::debug!(
            target: "f1r3fly.casper",
            deploy_cost = eval_result.cost.value,
            phlo_limit = deploy.data.phlo_limit,
            errors_count = eval_result.errors.len(),
            event_count = eval_result.mergeable.len(),
            "process_deploy: user deploy evaluation complete"
        );

        let deploy_log = self.runtime.take_event_log();

        let eval_succeeded = eval_result.errors.is_empty();
        let deploy_sig = deploy.sig.clone();

        let deploy_result = ProcessedDeploy {
            deploy,
            cost: Cost::to_proto(eval_result.cost),
            deploy_log: deploy_log
                .into_iter()
                .map(|event| event_converter::to_casper_event(event))
                .collect(),
            is_failed: !eval_succeeded,
            system_deploy_error: None,
        };

        if !eval_succeeded {
            self.runtime.revert_to_soft_checkpoint(fallback);
            interpreter_util::print_deploy_errors(&deploy_sig, &eval_result.errors);
        }

        Ok((deploy_result, eval_result.mergeable))
    }

    pub async fn process_deploy_with_mergeable_data(
        &mut self,
        deploy: Signed<DeployData>,
    ) -> Result<(ProcessedDeploy, NumberChannelsEndVal), CasperError> {
        self.process_deploy(deploy)
            .await
            .and_then(|(pd, merge_chs)| {
                self.get_number_channels_data(&merge_chs)
                    .map(|data| (pd, data))
            })
    }

    pub fn get_number_channels_data(
        &self,
        channels: &HashSet<Par>,
    ) -> Result<NumberChannelsEndVal, CasperError> {
        let mut result = BTreeMap::new();
        for channel in channels {
            if let Some((hash, value)) = self.get_number_channel(channel)? {
                result.insert(hash, value);
            }
        }
        Ok(result)
    }

    pub fn get_number_channel(
        &self,
        channel: &Par,
    ) -> Result<Option<(Blake2b256Hash, i64)>, CasperError> {
        let ch_values = self.runtime.get_data(channel);

        if ch_values.is_empty() {
            return Ok(None);
        } else {
            let ch_hash = stable_hash_provider::hash(channel);
            if ch_values.len() != 1 {
                // Liveness-first fallback: ambiguous mergeable channel values should not wedge proposing.
                // Keep behavior deterministic by selecting the maximum observed numeric value.
                let num = ch_values
                    .iter()
                    .map(|datum| {
                        let (n, _) = RholangMergingLogic::get_number_with_rnd(&datum.a);
                        n
                    })
                    .max()
                    .ok_or_else(|| {
                        CasperError::RuntimeError(
                            "NumberChannel had values but max() returned none.".to_string(),
                        )
                    })?;

                tracing::warn!(
                    target: "f1r3fly.mergeable_channel.sanitize",
                    "NumberChannel has {} values; selecting deterministic max={} for channel {}",
                    ch_values.len(),
                    num,
                    hex::encode(ch_hash.clone().bytes()),
                );
                metrics::counter!(
                    "mergeable_channel_number_sanitized_total",
                    "source" => "casper_runtime"
                )
                .increment(1);

                return Ok(Some((ch_hash, num)));
            }

            let num_par = &ch_values[0].a;
            let (num, _) = RholangMergingLogic::get_number_with_rnd(num_par);
            Ok(Some((ch_hash, num)))
        }
    }

    /* System deploy evaluators */

    /**
     * Evaluates System deploy with checkpoint to get final state hash
     */
    pub async fn play_system_deploy<S: SystemDeployTrait>(
        &mut self,
        state_hash: &StateHash,
        system_deploy: &mut S,
    ) -> Result<SystemDeployResult<S::Result>, CasperError> {
        self.runtime
            .reset(&Blake2b256Hash::from_bytes_prost(&state_hash))?;

        let (event_log, result, mergeable_channels) =
            self.play_system_deploy_internal(system_deploy).await?;

        let final_state_hash = {
            let checkpoint = self.runtime.create_checkpoint();
            tracing::info!(
                target: "f1r3fly.rspace",
                checkpoint_root = %hex::encode(checkpoint.root.bytes()),
                "play_system_deploy: checkpoint completed"
            );
            checkpoint.root.to_bytes_prost()
        };

        match result {
            Either::Right(system_deploy_result) => {
                let mcl = self.get_number_channels_data(&mergeable_channels)?;
                if let Some(SlashDeploy {
                    invalid_block_hash,
                    pk,
                    initial_rand: _,
                }) = system_deploy.as_any().downcast_ref::<SlashDeploy>()
                {
                    Ok(SystemDeployResult::play_succeeded(
                        final_state_hash,
                        event_log,
                        SystemDeployData::create_slash(invalid_block_hash.clone(), pk.clone()),
                        mcl,
                        system_deploy_result,
                    ))
                } else if let Some(CloseBlockDeploy { .. }) =
                    system_deploy.as_any().downcast_ref::<CloseBlockDeploy>()
                {
                    Ok(SystemDeployResult::play_succeeded(
                        final_state_hash,
                        event_log,
                        SystemDeployData::create_close(),
                        mcl,
                        system_deploy_result,
                    ))
                } else {
                    Ok(SystemDeployResult::play_succeeded(
                        final_state_hash,
                        event_log,
                        SystemDeployData::Empty,
                        mcl,
                        system_deploy_result,
                    ))
                }
            }

            Either::Left(usr_err) => Ok(SystemDeployResult::play_failed(event_log, usr_err)),
        }
    }

    pub async fn play_system_deploy_internal<S: SystemDeployTrait>(
        &mut self,
        system_deploy: &mut S,
    ) -> Result<
        (
            Vec<Event>,
            Either<SystemDeployUserError, S::Result>,
            HashSet<Par>,
        ),
        CasperError,
    > {
        let mem_profile_enabled = std::env::var("F1R3_BLOCK_CREATOR_PHASE_SUBSTEP_PROFILE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let read_vm_rss_kb = || -> Option<usize> {
            let status = std::fs::read_to_string("/proc/self/status").ok()?;
            status
                .lines()
                .find(|line| line.starts_with("VmRSS:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|value| value.parse::<usize>().ok())
        };
        let deploy_type = std::any::type_name::<S>();
        let mut rss_baseline = if mem_profile_enabled {
            read_vm_rss_kb()
        } else {
            None
        };
        let mut rss_prev = rss_baseline;
        let mut log_mem_step = |step: &str| {
            if !mem_profile_enabled {
                return;
            }
            if let Some(curr) = read_vm_rss_kb() {
                let prev = rss_prev.unwrap_or(curr);
                let baseline = rss_baseline.unwrap_or(curr);
                eprintln!(
                    "play_system_deploy_internal.mem deploy_type={} step={} rss_kb={} delta_prev_kb={} delta_total_kb={}",
                    deploy_type,
                    step,
                    curr,
                    curr as i64 - prev as i64,
                    curr as i64 - baseline as i64
                );
                rss_prev = Some(curr);
                if rss_baseline.is_none() {
                    rss_baseline = Some(curr);
                }
            }
        };
        log_mem_step("start");

        // Get System deploy result / throw fatal errors for unexpected results
        let (result_or_system_deploy_error, eval_result) =
            self.eval_system_deploy(system_deploy).await?;
        log_mem_step("after_eval_system_deploy");

        let log = self.runtime.take_event_log();
        log_mem_step("after_take_event_log");
        let log = log
            .into_iter()
            .map(event_converter::to_casper_event)
            .collect();
        log_mem_step("after_convert_event_log");

        Ok((log, result_or_system_deploy_error, eval_result.mergeable))
    }

    /**
     * Evaluates System deploy (applicative errors are fatal)
     */
    pub async fn eval_system_deploy<S: SystemDeployTrait>(
        &mut self,
        system_deploy: &mut S,
    ) -> Result<SysEvalResult<S>, CasperError> {
        let mem_profile_enabled = std::env::var("F1R3_BLOCK_CREATOR_PHASE_SUBSTEP_PROFILE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let read_vm_rss_kb = || -> Option<usize> {
            let status = std::fs::read_to_string("/proc/self/status").ok()?;
            status
                .lines()
                .find(|line| line.starts_with("VmRSS:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|value| value.parse::<usize>().ok())
        };
        let deploy_type = std::any::type_name::<S>();
        let mut rss_baseline = if mem_profile_enabled {
            read_vm_rss_kb()
        } else {
            None
        };
        let mut rss_prev = rss_baseline;
        let mut log_mem_step = |step: &str| {
            if !mem_profile_enabled {
                return;
            }
            if let Some(curr) = read_vm_rss_kb() {
                let prev = rss_prev.unwrap_or(curr);
                let baseline = rss_baseline.unwrap_or(curr);
                eprintln!(
                    "eval_system_deploy.mem deploy_type={} step={} rss_kb={} delta_prev_kb={} delta_total_kb={}",
                    deploy_type,
                    step,
                    curr,
                    curr as i64 - prev as i64,
                    curr as i64 - baseline as i64
                );
                rss_prev = Some(curr);
                if rss_baseline.is_none() {
                    rss_baseline = Some(curr);
                }
            }
        };
        log_mem_step("start");

        // println!("\nEvaluating system deploy, {:?}", S::source());
        let eval_result = self.evaluate_system_source(system_deploy).await?;
        log_mem_step("after_evaluate_system_source");

        // println!("\nEval result: {:?}", eval_result);

        if !eval_result.errors.is_empty() {
            return Err(CasperError::SystemRuntimeError(
                SystemDeployPlatformFailure::UnexpectedSystemErrors(eval_result.errors),
            ));
        }
        log_mem_step("after_error_check");

        log_mem_step("before_consume_system_result");
        let consumed = self.consume_system_result(system_deploy)?;
        log_mem_step("after_consume_system_result");
        let r = match consumed {
            Some((_, vec_list)) => match vec_list.as_slice() {
                [ListParWithRandom { pars, .. }] if pars.len() == 1 => {
                    let extracted = system_deploy.extract_result(&pars[0]);
                    log_mem_step("after_extract_result");
                    Ok(extracted)
                }
                _ => Err(CasperError::SystemRuntimeError(
                    SystemDeployPlatformFailure::UnexpectedResult(
                        vec_list.iter().flat_map(|lp| lp.pars.clone()).collect(),
                    ),
                )),
            },
            None => Err(CasperError::SystemRuntimeError(
                SystemDeployPlatformFailure::ConsumeFailed,
            )),
        }?;
        log_mem_step("after_match_result");

        Ok((r, eval_result))
    }

    /**
     * Evaluates exploratory (read-only) deploy
     */
    pub async fn play_exploratory_deploy(
        &mut self,
        term: String,
        hash: &StateHash,
    ) -> Result<Vec<Par>, CasperError> {
        // Use a fresh random key pair and wall-clock timestamp for each explore-deploy.
        // This ensures the deploy's RNG seed is unique per call, so the externally-
        // computed return_name always matches the `ret` channel created inside eval_new.
        // (Matches the Scala implementation's behavior.)
        let secp = Secp256k1;
        let (priv_key, _pub_key) = secp.new_key_pair();

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        let deploy = construct_deploy::source_deploy(
            term,
            timestamp,
            // Hardcoded phlogiston limit / 1 REV if phloPrice=1
            Some(100 * 1000 * 1000),
            None,
            Some(priv_key),
            None,
            None,
        )?;

        // Create return channel as first private name created in deploy term
        let mut rand = Tools::unforgeable_name_rng(&deploy.pk, deploy.data.time_stamp);
        let return_bytes: Vec<u8> = rand.next().into_iter().map(|b| b as u8).collect();

        // Diagnostic: log the RNG seed inputs and the resulting return channel id
        tracing::info!(
            target: "f1r3fly.rholang.diag",
            pk_hex = %hex::encode(&deploy.pk.bytes),
            timestamp = deploy.data.time_stamp,
            return_name_hex = %hex::encode(&return_bytes),
            rand_position = rand.position,
            rand_path_position = rand.path_position,
            "play_exploratory_deploy: computed return_name from deploy seed"
        );

        let return_name = Par::default().with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
                id: return_bytes,
            })),
        }]);

        // Execute deploy on top of specified block hash
        self.capture_results_with_name(hash, &deploy, &return_name)
            .await
    }

    async fn play_exploratory_par(
        &mut self,
        par: Par,
        hash: &StateHash,
    ) -> Result<Vec<Par>, CasperError> {
        let mem_profile_enabled = std::env::var("F1R3_BLOCK_CREATOR_PHASE_SUBSTEP_PROFILE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let read_vm_rss_kb = || -> Option<usize> {
            let status = std::fs::read_to_string("/proc/self/status").ok()?;
            status
                .lines()
                .find(|line| line.starts_with("VmRSS:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|value| value.parse::<usize>().ok())
        };
        let mut rss_baseline = if mem_profile_enabled {
            read_vm_rss_kb()
        } else {
            None
        };
        let mut rss_prev = rss_baseline;
        let mut log_mem_step = |step: &str| {
            if !mem_profile_enabled {
                return;
            }
            if let Some(curr) = read_vm_rss_kb() {
                let prev = rss_prev.unwrap_or(curr);
                let baseline = rss_baseline.unwrap_or(curr);
                eprintln!(
                    "play_exploratory_par.mem step={} rss_kb={} delta_prev_kb={} delta_total_kb={}",
                    step,
                    curr,
                    curr as i64 - prev as i64,
                    curr as i64 - baseline as i64
                );
                rss_prev = Some(curr);
                if rss_baseline.is_none() {
                    rss_baseline = Some(curr);
                }
            }
        };
        log_mem_step("start");

        self.runtime
            .reset(&Blake2b256Hash::from_bytes_prost(hash))?;
        log_mem_step("after_reset");
        self.runtime.cost().set(Cost::unsafe_max());
        log_mem_step("after_set_cost");

        let rand = Blake2b512Random::create_from_bytes(&[0u8; 128]);
        let mut return_rand = rand.clone();
        let return_name = Par::default().with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
                id: return_rand.next().into_iter().map(|b| b as u8).collect(),
            })),
        }]);
        log_mem_step("after_build_return_name");

        let result = match self.runtime.inj(par, Env::new(), rand).await {
            Ok(()) => {
                log_mem_step("after_inj_ok");
                let data = self.get_data_par(&return_name);
                log_mem_step("after_get_data_par");
                Ok(data)
            }
            Err(err) => {
                log_mem_step("after_inj_err");
                tracing::error!("Error in play_exploratory_par: {:?}", err);
                Ok(Vec::new())
            }
        };

        let _ = self.runtime.take_event_log();
        log_mem_step("after_take_event_log");
        self.runtime
            .reset(&Blake2b256Hash::from_bytes_prost(hash))?;
        log_mem_step("after_post_query_reset");

        result
    }

    /* Checkpoints */

    /**
     * Creates soft checkpoint with rollback if result is false.
     */
    pub async fn with_soft_transaction<A, F, Fut>(&mut self, action: F) -> Result<A, CasperError>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<(A, bool), CasperError>>,
    {
        let fallback = self.runtime.create_soft_checkpoint();

        // Execute action
        let (a, success) = action().await?;

        // Revert the state if failed
        if !success {
            self.runtime.revert_to_soft_checkpoint(fallback);
        }

        Ok(a)
    }

    /* Evaluates and captures results */

    // Return channel on which result is captured is the first name
    // in the deploy term `new return in { return!(42) }`
    pub async fn capture_results(
        &mut self,
        start: &StateHash,
        deploy: &Signed<DeployData>,
    ) -> Result<Vec<Par>, CasperError> {
        // Create return channel as first unforgeable name created in deploy term
        let mut rand = Tools::unforgeable_name_rng(&deploy.pk, deploy.data.time_stamp);
        let return_name = Par::default().with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
                id: rand.next().into_iter().map(|b| b as u8).collect(),
            })),
        }]);

        self.capture_results_with_name(start, deploy, &return_name)
            .await
    }

    pub async fn capture_results_with_name(
        &mut self,
        start: &StateHash,
        deploy: &Signed<DeployData>,
        name: &Par,
    ) -> Result<Vec<Par>, CasperError> {
        match self.capture_results_with_errors(start, deploy, name).await {
            Ok(result) => Ok(result),
            Err(err) => Err(CasperError::InterpreterError(
                InterpreterError::BugFoundError(format!(
                    "Unexpected error while capturing results from Rholang: {}",
                    err
                )),
            )),
        }
    }

    pub async fn capture_results_with_errors(
        &mut self,
        start: &StateHash,
        deploy: &Signed<DeployData>,
        name: &Par,
    ) -> Result<Vec<Par>, CasperError> {
        // Step 2a: Entry beacon — confirms function was entered
        tracing::info!(
            target: "f1r3fly.rholang.diag",
            state_hash = %hex::encode(start),
            "capture_results_with_errors: ENTERED"
        );

        tracing::debug!(
            state_hash = %hex::encode(start),
            "capture_results: resetting to state hash"
        );
        self.runtime
            .reset(&Blake2b256Hash::from_bytes_prost(start))?;
        tracing::debug!("capture_results: reset succeeded");

        // Step 2b: Beacon before POST-RESET probe
        tracing::info!(
            target: "f1r3fly.rholang.diag",
            "capture_results_with_errors: about to run POST-RESET probe"
        );

        // Diagnostic: probe registry state for byte_name(14) after reset, before evaluate.
        // This tells us whether the registry's persistent continuation is accessible
        // from the history trie at this state hash.
        // Step 2c: Guard with try_lock instead of unwrap to avoid panics.
        {
            let reg_channel = Par {
                unforgeables: vec![GUnforgeable {
                    unf_instance: Some(UnfInstance::GPrivateBody(GPrivate { id: vec![14] })),
                }],
                ..Par::default()
            };

            match self.runtime.reducer.space.try_lock() {
                Ok(space_guard) => {
                    let reg_data = space_guard.get_data(&reg_channel);
                    let reg_conts = space_guard.get_waiting_continuations(vec![reg_channel.clone()]);
                    let reg_joins = space_guard.get_joins(reg_channel.clone());
                    drop(space_guard);

                    let persistent_conts = reg_conts.iter().filter(|wc| wc.persist).count();

                    tracing::info!(
                        target: "f1r3fly.rholang.diag",
                        state_hash = %hex::encode(start),
                        data_count = reg_data.len(),
                        cont_count = reg_conts.len(),
                        persistent_conts = persistent_conts,
                        join_count = reg_joins.len(),
                        "POST-RESET REGISTRY PROBE byte_name(14): data={}, conts={} (persistent={}), joins={}",
                        reg_data.len(),
                        reg_conts.len(),
                        persistent_conts,
                        reg_joins.len()
                    );

                    // If joins exist, log the join channel groups
                    for (i, join_group) in reg_joins.iter().enumerate() {
                        let join_ch_ids: Vec<String> = join_group
                            .iter()
                            .flat_map(|par| &par.unforgeables)
                            .filter_map(|u| u.unf_instance.as_ref())
                            .map(|inst| match inst {
                                UnfInstance::GPrivateBody(gp) => format!("GPrivate({})", hex::encode(&gp.id)),
                                other => format!("{:?}", other),
                            })
                            .collect();
                        tracing::info!(
                            target: "f1r3fly.rholang.diag",
                            join_idx = i,
                            join_channels = ?join_ch_ids,
                            "POST-RESET REGISTRY PROBE byte_name(14): join group #{}: {:?}",
                            i, join_ch_ids
                        );
                    }

                    // If continuations exist, log their pattern info
                    for (i, wc) in reg_conts.iter().enumerate() {
                        tracing::info!(
                            target: "f1r3fly.rholang.diag",
                            cont_idx = i,
                            persist = wc.persist,
                            pattern_count = wc.patterns.len(),
                            "POST-RESET REGISTRY PROBE byte_name(14): continuation #{}: persist={}, patterns={}",
                            i, wc.persist, wc.patterns.len()
                        );
                    }

                    if reg_conts.is_empty() && reg_joins.is_empty() {
                        tracing::warn!(
                            target: "f1r3fly.rholang.diag",
                            state_hash = %hex::encode(start),
                            "POST-RESET REGISTRY PROBE byte_name(14): NO continuations AND NO joins — \
                             registry state is NOT accessible at this state hash! \
                             This confirms the registry COMM cannot fire."
                        );
                    }
                }
                Err(_) => {
                    tracing::warn!(
                        target: "f1r3fly.rholang.diag",
                        state_hash = %hex::encode(start),
                        "POST-RESET REGISTRY PROBE byte_name(14): SKIPPED — space lock not available \
                         (another thread holds the lock)"
                    );
                }
            }
        }

        let eval_res = self.evaluate(deploy).await?;
        if !eval_res.errors.is_empty() {
            tracing::warn!(
                errors = ?eval_res.errors,
                "capture_results: evaluation produced errors"
            );
            return Err(CasperError::InterpreterError(eval_res.errors[0].clone()));
        }

        let result = self.get_data_par(name);
        if result.is_empty() {
            // Log the return channel bytes for diagnosing channel mismatch issues
            let return_ch_id: String = name
                .unforgeables
                .first()
                .and_then(|u| u.unf_instance.as_ref())
                .map(|inst| format!("{:?}", inst))
                .unwrap_or_else(|| "<unknown>".to_string());
            tracing::warn!(
                target: "f1r3fly.rspace",
                state_hash = %hex::encode(start),
                deploy_term_len = deploy.data.term.len(),
                return_channel = %return_ch_id,
                deploy_timestamp = deploy.data.time_stamp,
                "capture_results: get_data_par returned EMPTY — explore-deploy produced no results"
            );

            // Diagnostic: enumerate ALL data channels in hot store to show where data actually went
            let hot_changes = self.runtime.get_hot_changes();
            let non_empty_data: Vec<_> = hot_changes
                .iter()
                .filter(|(_, row)| !row.data.is_empty())
                .collect();
            tracing::warn!(
                target: "f1r3fly.rholang.diag",
                total_channels = hot_changes.len(),
                channels_with_data = non_empty_data.len(),
                "capture_results: hot store data channel enumeration after EMPTY result"
            );
            for (ch_keys, row) in &non_empty_data {
                // Extract GPrivate ids from the channel keys for comparison
                let ch_hex: Vec<String> = ch_keys
                    .iter()
                    .flat_map(|par| &par.unforgeables)
                    .filter_map(|u| u.unf_instance.as_ref())
                    .map(|inst| match inst {
                        UnfInstance::GPrivateBody(gp) => {
                            format!("GPrivate({})", hex::encode(&gp.id))
                        }
                        other => format!("{:?}", other),
                    })
                    .collect();
                tracing::warn!(
                    target: "f1r3fly.rholang.diag",
                    channel = ?ch_hex,
                    datum_count = row.data.len(),
                    "capture_results: hot store data channel with data"
                );
            }

            // Step 1: Enumerate ALL pending continuations in hot store to identify blocked channels
            let non_empty_conts: Vec<_> = hot_changes
                .iter()
                .filter(|(_, row)| !row.wks.is_empty())
                .collect();
            let total_pending: usize = non_empty_conts.iter().map(|(_, row)| row.wks.len()).sum();
            tracing::warn!(
                target: "f1r3fly.rholang.diag",
                channels_with_continuations = non_empty_conts.len(),
                total_pending_continuations = total_pending,
                "capture_results: hot store continuation enumeration after EMPTY result"
            );
            for (ch_keys, row) in &non_empty_conts {
                let ch_hex: Vec<String> = ch_keys
                    .iter()
                    .flat_map(|par| &par.unforgeables)
                    .filter_map(|u| u.unf_instance.as_ref())
                    .map(|inst| match inst {
                        UnfInstance::GPrivateBody(gp) => {
                            format!("GPrivate({})", hex::encode(&gp.id))
                        }
                        other => format!("{:?}", other),
                    })
                    .collect();
                let persistent_count = row.wks.iter().filter(|wc| wc.persist).count();
                let pattern_counts: Vec<usize> = row.wks.iter().map(|wc| wc.patterns.len()).collect();
                tracing::warn!(
                    target: "f1r3fly.rholang.diag",
                    channel = ?ch_hex,
                    continuation_count = row.wks.len(),
                    persistent_count = persistent_count,
                    pattern_counts = ?pattern_counts,
                    "capture_results: hot store channel with pending continuation(s)"
                );
            }
        } else {
            tracing::debug!(
                result_count = result.len(),
                "capture_results: get_data_par returned {} pars",
                result.len()
            );
        }
        Ok(result)
    }

    /* Evaluates Rholang source code */

    pub async fn evaluate(
        &mut self,
        deploy: &Signed<DeployData>,
    ) -> Result<EvaluateResult, CasperError> {
        let deploy_data = SystemProcessDeployData::from_deploy(deploy);
        self.runtime.set_deploy_data(deploy_data).await;

        let result = self
            .runtime
            .evaluate(
                &deploy.data.term,
                Cost::create(deploy.data.phlo_limit, "Evaluate deploy".to_string()),
                normalizer_env_from_deploy(deploy),
                Tools::unforgeable_name_rng(&deploy.pk, deploy.data.time_stamp),
            )
            .await;

        match result {
            Ok(eval_result) => Ok(eval_result),
            Err(e) => Err(CasperError::InterpreterError(e)),
        }
    }

    pub async fn evaluate_system_source<S: SystemDeployTrait>(
        &mut self,
        system_deploy: &mut S,
    ) -> Result<EvaluateResult, CasperError> {
        let mem_profile_enabled = std::env::var("F1R3_BLOCK_CREATOR_PHASE_SUBSTEP_PROFILE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let read_vm_rss_kb = || -> Option<usize> {
            let status = std::fs::read_to_string("/proc/self/status").ok()?;
            status
                .lines()
                .find(|line| line.starts_with("VmRSS:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|value| value.parse::<usize>().ok())
        };
        let deploy_type = std::any::type_name::<S>();
        let mut rss_baseline = if mem_profile_enabled {
            read_vm_rss_kb()
        } else {
            None
        };
        let mut rss_prev = rss_baseline;
        let mut log_mem_step = |step: &str| {
            if !mem_profile_enabled {
                return;
            }
            if let Some(curr) = read_vm_rss_kb() {
                let prev = rss_prev.unwrap_or(curr);
                let baseline = rss_baseline.unwrap_or(curr);
                eprintln!(
                    "evaluate_system_source.mem deploy_type={} step={} rss_kb={} delta_prev_kb={} delta_total_kb={}",
                    deploy_type,
                    step,
                    curr,
                    curr as i64 - prev as i64,
                    curr as i64 - baseline as i64
                );
                rss_prev = Some(curr);
                if rss_baseline.is_none() {
                    rss_baseline = Some(curr);
                }
            }
        };
        log_mem_step("start");

        // Using tracing events for async - Span[F].traceI("evaluate-system-source") from Scala
        tracing::debug!(target: "f1r3fly.casper.evaluate-system-source", "evaluate-system-source-started");
        let eval_start = Instant::now();
        log_mem_step("before_build_env");
        let env = system_deploy.env();
        log_mem_step("after_build_env");
        let rand = system_deploy.rand().clone();
        log_mem_step("after_clone_rand");
        log_mem_step("before_runtime_evaluate");
        let result = self
            .runtime
            .evaluate(
                S::source(),
                Cost::unsafe_max(),
                env,
                // TODO: Review this clone and whether to pass mut ref down into evaluate
                rand,
            )
            .await?;
        log_mem_step("after_runtime_evaluate");
        metrics::histogram!(BLOCK_REPLAY_SYSDEPLOY_EVAL_EVALUATE_SOURCE_TIME_METRIC, "source" => CASPER_METRICS_SOURCE)
            .record(eval_start.elapsed().as_secs_f64());
        Ok(result)
    }

    pub fn get_data_par(&self, channel: &Par) -> Vec<Par> {
        // Diagnostic: log the channel's GPrivate id bytes before reading
        let ch_id_hex: String = channel
            .unforgeables
            .first()
            .and_then(|u| u.unf_instance.as_ref())
            .map(|inst| match inst {
                UnfInstance::GPrivateBody(gp) => hex::encode(&gp.id),
                other => format!("{:?}", other),
            })
            .unwrap_or_else(|| "<no-unforgeable>".to_string());
        tracing::info!(
            target: "f1r3fly.rholang.diag",
            channel_gprivate_hex = %ch_id_hex,
            "get_data_par: reading from channel"
        );

        let datums = self.runtime.get_data(channel);
        tracing::debug!(
            target: "f1r3fly.rspace",
            channel = ?channel,
            datum_count = datums.len(),
            "get_data_par: retrieved {} datums from channel",
            datums.len()
        );
        datums
            .into_iter()
            .flat_map(|datum| datum.a.pars)
            .collect()
    }

    pub fn get_continuation_par(&self, channels: Vec<Par>) -> Vec<(Vec<BindPattern>, Par)> {
        self.runtime
            .get_continuations(channels)
            .into_iter()
            .filter_map(|wk| {
                if let Some(TaggedCont::ParBody(par_body)) = wk.continuation.tagged_cont {
                    Some((wk.patterns, par_body.body.unwrap()))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn consume_result(
        &mut self,
        channel: Par,
        pattern: BindPattern,
    ) -> Result<Option<(TaggedContinuation, Vec<ListParWithRandom>)>, CasperError> {
        Ok(self.runtime.consume_result(vec![channel], vec![pattern])?)
    }

    pub fn consume_system_result<S: SystemDeployTrait>(
        &mut self,
        system_deploy: &mut S,
    ) -> Result<Option<(TaggedContinuation, Vec<ListParWithRandom>)>, CasperError> {
        let _span = tracing::info_span!(target: "f1r3fly.casper.consume-system-result", "consume-system-result").entered();
        let consume_start = Instant::now();
        let return_channel = system_deploy.return_channel()?;
        let result = self.consume_result(return_channel, system_deploy_consume_all_pattern());
        metrics::histogram!(BLOCK_REPLAY_SYSDEPLOY_EVAL_CONSUME_RESULT_TIME_METRIC, "source" => CASPER_METRICS_SOURCE)
            .record(consume_start.elapsed().as_secs_f64());
        result
    }

    /* Read only Rholang evaluator helpers */

    pub async fn get_active_validators(
        &mut self,
        start_hash: &StateHash,
    ) -> Result<Vec<Validator>, CasperError> {
        let validators_pars = self
            .play_exploratory_par(Self::activate_validator_query_par().clone(), start_hash)
            .await?;

        if validators_pars.is_empty() {
            tracing::warn!(
                "No result from getActiveValidators query for state {}; treating as no active validators",
                PrettyPrinter::build_string_bytes(start_hash)
            );
            return Ok(Vec::new());
        }

        if validators_pars.len() != 1 {
            return Err(CasperError::RuntimeError(format!(
                "Incorrect number of results from query of current bonds in state {}: {}",
                PrettyPrinter::build_string_bytes(start_hash),
                validators_pars.len()
            )));
        }

        let validators = Self::to_validator_vec(validators_pars[0].to_owned())?;
        let vlds: Vec<String> = validators.iter().map(|v| hex::encode(&v)).collect();
        tracing::info!(
            "*** ACTIVE VALIDATORS FOR StateHash {}: {}",
            hex::encode(start_hash),
            vlds.join("\n")
        );

        Ok(validators)
    }

    pub async fn compute_bonds(&mut self, hash: &StateHash) -> Result<Vec<Bond>, CasperError> {
        let bonds_pars = self
            .play_exploratory_par(Self::bonds_query_par().clone(), hash)
            .await?;

        if bonds_pars.is_empty() {
            tracing::warn!(
                "No result from getBonds query for state {}; treating as empty bonds",
                PrettyPrinter::build_string_bytes(hash)
            );
            return Ok(Vec::new());
        }

        if bonds_pars.len() != 1 {
            return Err(CasperError::RuntimeError(format!(
                "Incorrect number of results from query of current bonds in state {}: {}",
                PrettyPrinter::build_string_bytes(hash),
                bonds_pars.len()
            )));
        }

        Self::to_bond_vec(bonds_pars[0].to_owned())
    }

    fn activate_validator_query_source() -> String {
        r#"
          new return, rl(`rho:registry:lookup`), poSCh in {
          rl!(`rho:system:pos`, *poSCh) |
          for(@(_, PoS) <- poSCh) {
            @PoS!("getActiveValidators", *return)
          }
        }
      "#
        .to_string()
    }

    fn activate_validator_query_par() -> &'static Par {
        static QUERY: OnceLock<Par> = OnceLock::new();
        QUERY.get_or_init(|| {
            Compiler::source_to_adt(&Self::activate_validator_query_source())
                .expect("Failed to compile active validator query source")
        })
    }

    fn bonds_query_source() -> String {
        r#"
        new return, rl(`rho:registry:lookup`), poSCh in {
          rl!(`rho:system:pos`, *poSCh) |
          for(@(_, PoS) <- poSCh) {
            @PoS!("getBonds", *return)
          }
        }
      "#
        .to_string()
    }

    fn bonds_query_par() -> &'static Par {
        static QUERY: OnceLock<Par> = OnceLock::new();
        QUERY.get_or_init(|| {
            Compiler::source_to_adt(&Self::bonds_query_source())
                .expect("Failed to compile bonds query source")
        })
    }

    fn to_validator_vec(validators_par: Par) -> Result<Vec<Validator>, CasperError> {
        if validators_par.exprs.is_empty() {
            return Ok(Vec::new());
        }

        let ps = match validators_par.exprs[0].expr_instance.as_ref().unwrap() {
            ExprInstance::ESetBody(set) => ParSetTypeMapper::eset_to_par_set(set.clone()).ps,
            _ => SortedParHashSet::create_from_empty(),
        };

        Ok(ps
            .map_iter(|v| {
                if v.exprs.len() != 1 {
                    Err(CasperError::RuntimeError(
                        "Validator in bonds map wasn't a single string.".to_string(),
                    ))
                } else {
                    match v.exprs[0].expr_instance.as_ref().unwrap() {
                        ExprInstance::GByteArray(g_byte_array) => Ok(g_byte_array.clone().into()),
                        _ => Err(CasperError::RuntimeError(
                            "Expected GByteArray in validator data".to_string(),
                        )),
                    }
                }
            })
            .collect::<Result<Vec<_>, _>>()?)
    }

    fn to_bond_vec(bonds_map: Par) -> Result<Vec<Bond>, CasperError> {
        if bonds_map.exprs.is_empty() {
            return Ok(Vec::new());
        }

        let ps = match bonds_map.exprs[0].expr_instance.as_ref().unwrap() {
            ExprInstance::EMapBody(map) => ParMapTypeMapper::emap_to_par_map(map.clone()).ps,
            _ => SortedParMap::create_from_empty(),
        };

        Ok(ps
            .map_iter(|(validator, bond)| {
                if validator.exprs.len() != 1 {
                    Err(CasperError::RuntimeError(
                        "Validator in bonds map wasn't a single string.".to_string(),
                    ))
                } else if bond.exprs.len() != 1 {
                    Err(CasperError::RuntimeError(
                        "Stake in bonds map wasn't a single string.".to_string(),
                    ))
                } else {
                    let validator_name = match validator.exprs[0].expr_instance.as_ref().unwrap() {
                        ExprInstance::GByteArray(g_byte_array) => Ok(g_byte_array.clone().into()),
                        _ => Err(CasperError::RuntimeError(
                            "Expected GByteArray in validator data".to_string(),
                        )),
                    }?;

                    let stake_amount = match bond.exprs[0].expr_instance.as_ref().unwrap() {
                        ExprInstance::GInt(g_int) => Ok(*g_int),
                        _ => Err(CasperError::RuntimeError(
                            "Expected GInt in stake data".to_string(),
                        )),
                    }?;

                    Ok(Bond {
                        validator: validator_name,
                        stake: stake_amount,
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()?)
    }
}
