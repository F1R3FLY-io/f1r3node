// See casper/src/main/scala/coop/rchain/casper/rholang/RuntimeSyntax.scala

use std::{
    collections::{BTreeMap, HashMap},
    future::Future,
    mem,
    sync::OnceLock,
    time::Instant,
};

use crypto::rust::{
    hash::blake2b512_random::Blake2b512Random, public_key::PublicKey, signatures::signed::Signed,
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
    merger::merging_logic::{MergeType, NumberChannelsEndVal},
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
        self.runtime
            .reset(&RadixHistory::empty_root_node_hash())
            .await?;

        bootstrap_registry(&self.runtime).await;
        let checkpoint = self.runtime.create_checkpoint().await;
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
        let mem_profile_enabled = crate::rust::util::rholang::mem_profiler::mem_profile_enabled();
        let read_vm_rss_kb =
            || -> Option<usize> { crate::rust::util::rholang::mem_profiler::read_vm_rss_kb() };
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
        let mem_profile_enabled = crate::rust::util::rholang::mem_profiler::mem_profile_enabled();
        let read_vm_rss_kb =
            || -> Option<usize> { crate::rust::util::rholang::mem_profiler::read_vm_rss_kb() };
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
            .reset(&Blake2b256Hash::from_bytes_prost(start_hash))
            .await?;
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
        let final_checkpoint = self.runtime.create_checkpoint().await;
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
            .reset(&Blake2b256Hash::from_bytes_prost(start_hash))
            .await?;

        let mut res = Vec::with_capacity(terms.len());
        for deploy in terms {
            res.push(self.process_deploy_with_mergeable_data(deploy).await?);
        }

        let final_checkpoint = self.runtime.create_checkpoint().await;
        Ok((final_checkpoint.root.to_bytes_prost(), res))
    }

    /**
     * Evaluates deploy with cost accounting (PoS Pre-charge and Refund calls)
     */
    pub async fn play_deploy_with_cost_accounting(
        &mut self,
        deploy: Signed<DeployData>,
    ) -> Result<(ProcessedDeploy, NumberChannelsEndVal), CasperError> {
        let mem_profile_enabled = crate::rust::util::rholang::mem_profiler::mem_profile_enabled();
        let read_vm_rss_kb =
            || -> Option<usize> { crate::rust::util::rholang::mem_profiler::read_vm_rss_kb() };
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
                            .get_number_channels_data(&eval_collector_state.mergeable_channels)
                            .await?;

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
                let mergeable_channels_data = self
                    .get_number_channels_data(&eval_collector_state.mergeable_channels)
                    .await?;

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
    ) -> Result<(ProcessedDeploy, HashMap<Par, MergeType>), CasperError> {
        // Keep a soft checkpoint before user deploy execution so failed deploy rollback
        // preserves pre-charge side effects required by refundDeploy.
        let fallback = self.runtime.create_soft_checkpoint().await;

        // Evaluate deploy
        let eval_result = self.evaluate(&deploy).await?;

        let deploy_log = self.runtime.take_event_log().await;

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
            self.runtime.revert_to_soft_checkpoint(fallback).await;
            interpreter_util::print_deploy_errors(&deploy_sig, &eval_result.errors);
        }

        Ok((deploy_result, eval_result.mergeable))
    }

    pub async fn process_deploy_with_mergeable_data(
        &mut self,
        deploy: Signed<DeployData>,
    ) -> Result<(ProcessedDeploy, NumberChannelsEndVal), CasperError> {
        let (pd, merge_chs) = self.process_deploy(deploy).await?;
        let data = self.get_number_channels_data(&merge_chs).await?;
        Ok((pd, data))
    }

    pub async fn get_number_channels_data(
        &self,
        channels: &std::collections::HashMap<
            Par,
            rspace_plus_plus::rspace::merger::merging_logic::MergeType,
        >,
    ) -> Result<NumberChannelsEndVal, CasperError> {
        let mut result = BTreeMap::new();
        for (channel, merge_type) in channels {
            if let Some((hash, value)) = self.get_number_channel(channel, *merge_type).await? {
                result.insert(hash, (value, *merge_type));
            }
        }
        Ok(result)
    }

    /// Deterministic multi-value fold for a mergeable channel that holds more
    /// than one numeric Datum at observation time. Dispatches by `MergeType`:
    /// `IntegerAdd` picks the max (conservative for vault balances);
    /// `BitmaskOr` OR-folds all bitmaps (no set bit is lost). Returns `None`
    /// for an empty input.
    pub fn fold_multi_value(values: &[i64], merge_type: MergeType) -> Option<i64> {
        if values.is_empty() {
            return None;
        }
        let folded = match merge_type {
            MergeType::IntegerAdd => *values.iter().max().unwrap(),
            MergeType::BitmaskOr => values
                .iter()
                .fold(0i64, |acc, v| ((acc as u64) | (*v as u64)) as i64),
        };
        Some(folded)
    }

    pub async fn get_number_channel(
        &self,
        channel: &Par,
        merge_type: MergeType,
    ) -> Result<Option<(Blake2b256Hash, i64)>, CasperError> {
        let ch_values = self.runtime.get_data(channel).await;

        if ch_values.is_empty() {
            return Ok(None);
        } else {
            let ch_hash = stable_hash_provider::hash(channel);
            if ch_values.len() != 1 {
                // Liveness-first fallback: ambiguous mergeable channel values should not wedge
                // proposing. Non-numeric values are skipped — they aren't candidates for the
                // numeric merge path and fall through to existing conflict handling.
                let nums: Vec<i64> = ch_values
                    .iter()
                    .filter_map(|datum| {
                        RholangMergingLogic::try_get_number_with_rnd(&datum.a).map(|(n, _)| n)
                    })
                    .collect();

                let num = match Self::fold_multi_value(&nums, merge_type) {
                    Some(n) => n,
                    None => return Ok(None),
                };

                tracing::warn!(
                    target: "f1r3fly.mergeable_channel.sanitize",
                    "NumberChannel has {} values; merge_type={:?} dispatched value={} for channel {}",
                    ch_values.len(),
                    merge_type,
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

            // Single value: opportunistic numeric read. Non-numeric values
            // (e.g., TreeHashMap leaf Maps tagged with the bitmask tag) are
            // skipped here and fall through to the existing conflict path.
            let num_par = &ch_values[0].a;
            match RholangMergingLogic::try_get_number_with_rnd(num_par) {
                Some((num, _)) => Ok(Some((ch_hash, num))),
                None => Ok(None),
            }
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
            .reset(&Blake2b256Hash::from_bytes_prost(&state_hash))
            .await?;

        let (event_log, result, mergeable_channels) =
            self.play_system_deploy_internal(system_deploy).await?;

        let final_state_hash = {
            let checkpoint = self.runtime.create_checkpoint().await;
            checkpoint.root.to_bytes_prost()
        };

        match result {
            Either::Right(system_deploy_result) => {
                let mcl = self.get_number_channels_data(&mergeable_channels).await?;
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
            HashMap<Par, MergeType>,
        ),
        CasperError,
    > {
        let mem_profile_enabled = crate::rust::util::rholang::mem_profiler::mem_profile_enabled();
        let read_vm_rss_kb =
            || -> Option<usize> { crate::rust::util::rholang::mem_profiler::read_vm_rss_kb() };
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

        let log = self.runtime.take_event_log().await;
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
        let mem_profile_enabled = crate::rust::util::rholang::mem_profiler::mem_profile_enabled();
        let read_vm_rss_kb =
            || -> Option<usize> { crate::rust::util::rholang::mem_profiler::read_vm_rss_kb() };
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
        let consumed = self.consume_system_result(system_deploy).await?;
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
    ) -> Result<(Vec<Par>, u64), CasperError> {
        let deploy_result = (|| async {
            let deploy = construct_deploy::source_deploy(
                term,
                0,
                // Hardcoded phlogiston limit / 1 REV if phloPrice=1
                Some(100 * 1000 * 1000),
                None,
                Some(construct_deploy::DEFAULT_SEC.clone()),
                None,
                None,
            )?;

            // Create return channel as first private name created in deploy term
            let mut rand = Tools::unforgeable_name_rng(&deploy.pk, deploy.data.time_stamp);
            let return_name = Par::default().with_unforgeables(vec![GUnforgeable {
                unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
                    id: rand.next().into_iter().map(|b| b as u8).collect(),
                })),
            }]);

            // Execute deploy on top of specified block hash
            self.capture_results_with_name(hash, &deploy, &return_name)
                .await
        })();

        deploy_result.await
    }

    async fn play_exploratory_par(
        &mut self,
        par: Par,
        hash: &StateHash,
    ) -> Result<Vec<Par>, CasperError> {
        let mem_profile_enabled = crate::rust::util::rholang::mem_profiler::mem_profile_enabled();
        let read_vm_rss_kb =
            || -> Option<usize> { crate::rust::util::rholang::mem_profiler::read_vm_rss_kb() };
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
            .reset(&Blake2b256Hash::from_bytes_prost(hash))
            .await?;
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
                let data = self.get_data_par(&return_name).await;
                log_mem_step("after_get_data_par");
                Ok(data)
            }
            Err(err) => {
                log_mem_step("after_inj_err");
                tracing::error!("Error in play_exploratory_par: {:?}", err);
                Ok(Vec::new())
            }
        };

        let _ = self.runtime.take_event_log().await;
        log_mem_step("after_take_event_log");
        self.runtime
            .reset(&Blake2b256Hash::from_bytes_prost(hash))
            .await?;
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
        let fallback = self.runtime.create_soft_checkpoint().await;

        // Execute action
        let (a, success) = action().await?;

        // Revert the state if failed
        if !success {
            self.runtime.revert_to_soft_checkpoint(fallback).await;
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

        let (data, _cost) = self
            .capture_results_with_name(start, deploy, &return_name)
            .await?;
        Ok(data)
    }

    pub async fn capture_results_with_name(
        &mut self,
        start: &StateHash,
        deploy: &Signed<DeployData>,
        name: &Par,
    ) -> Result<(Vec<Par>, u64), CasperError> {
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
    ) -> Result<(Vec<Par>, u64), CasperError> {
        self.runtime
            .reset(&Blake2b256Hash::from_bytes_prost(start))
            .await?;

        let eval_res = self.evaluate(deploy).await?;
        if !eval_res.errors.is_empty() {
            return Err(CasperError::InterpreterError(eval_res.errors[0].clone()));
        }

        let cost = eval_res.cost.value.max(0) as u64;
        Ok((self.get_data_par(name).await, cost))
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
        let mem_profile_enabled = crate::rust::util::rholang::mem_profiler::mem_profile_enabled();
        let read_vm_rss_kb =
            || -> Option<usize> { crate::rust::util::rholang::mem_profiler::read_vm_rss_kb() };
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

    pub async fn get_data_par(&self, channel: &Par) -> Vec<Par> {
        self.runtime
            .get_data(channel)
            .await
            .into_iter()
            .flat_map(|datum| datum.a.pars)
            .collect()
    }

    pub async fn get_continuation_par(&self, channels: Vec<Par>) -> Vec<(Vec<BindPattern>, Par)> {
        self.runtime
            .get_continuations(channels)
            .await
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

    pub async fn consume_result(
        &mut self,
        channel: Par,
        pattern: BindPattern,
    ) -> Result<Option<(TaggedContinuation, Vec<ListParWithRandom>)>, CasperError> {
        Ok(self
            .runtime
            .consume_result(vec![channel], vec![pattern])
            .await?)
    }

    pub async fn consume_system_result<S: SystemDeployTrait>(
        &mut self,
        system_deploy: &mut S,
    ) -> Result<Option<(TaggedContinuation, Vec<ListParWithRandom>)>, CasperError> {
        let consume_start = Instant::now();
        let return_channel = system_deploy.return_channel()?;
        let result = self
            .consume_result(return_channel, system_deploy_consume_all_pattern())
            .await;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_multi_value_empty_returns_none() {
        assert_eq!(
            RuntimeOps::fold_multi_value(&[], MergeType::IntegerAdd),
            None
        );
        assert_eq!(
            RuntimeOps::fold_multi_value(&[], MergeType::BitmaskOr),
            None
        );
    }

    #[test]
    fn fold_multi_value_single_returns_value() {
        assert_eq!(
            RuntimeOps::fold_multi_value(&[42], MergeType::IntegerAdd),
            Some(42)
        );
        assert_eq!(
            RuntimeOps::fold_multi_value(&[42], MergeType::BitmaskOr),
            Some(42)
        );
    }

    #[test]
    fn fold_multi_value_integer_add_returns_max() {
        // Vault-balance semantics: pick the largest observed value.
        assert_eq!(
            RuntimeOps::fold_multi_value(&[10, 5, 20, 15], MergeType::IntegerAdd),
            Some(20)
        );
    }

    #[test]
    fn fold_multi_value_bitmask_or_returns_or_fold_not_max() {
        // BitmaskOr must OR-fold all bitmaps; using max() would silently lose
        // bits set only in non-max values.
        let a = 0b00010001i64; // bits {0, 4}
        let b = 0b00100010i64; // bits {1, 5}
                               // max(a, b) = b = 0b00100010 — would lose bits {0, 4}.
                               // OR fold = 0b00110011 — bits {0, 1, 4, 5}. Correct.
        assert_eq!(
            RuntimeOps::fold_multi_value(&[a, b], MergeType::BitmaskOr),
            Some(0b00110011),
        );
        // Three-way fold sanity.
        let c = 0b01000000i64;
        assert_eq!(
            RuntimeOps::fold_multi_value(&[a, b, c], MergeType::BitmaskOr),
            Some(0b01110011),
        );
    }

    #[test]
    fn fold_multi_value_bitmask_or_commutes() {
        // Result must not depend on observation order.
        let xs = [0b0001_0001i64, 0b0010_0010, 0b0100_0100, 0b1000_1000];
        let mut ys = xs;
        ys.reverse();
        assert_eq!(
            RuntimeOps::fold_multi_value(&xs, MergeType::BitmaskOr),
            RuntimeOps::fold_multi_value(&ys, MergeType::BitmaskOr),
        );
    }

    #[test]
    fn fold_multi_value_bitmask_or_negative_high_bits_preserved() {
        // i64::MIN sets only the sign bit (bit 63). OR with a positive bitmap
        // must keep bit 63 set — no narrowing or sign-extension surprise.
        let neg = i64::MIN;
        let pos = 0b1010i64;
        let folded = RuntimeOps::fold_multi_value(&[neg, pos], MergeType::BitmaskOr).unwrap();
        assert_eq!(folded as u64, (neg as u64) | (pos as u64));
        assert_ne!(folded & i64::MIN, 0, "sign bit must remain set");
    }
}
