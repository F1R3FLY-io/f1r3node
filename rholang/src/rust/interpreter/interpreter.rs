use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::Par;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock, RwLock};
use tracing::{event, Level};

use super::accounting::_cost;
use super::accounting::costs::{parsing_cost, Cost};
use super::compiler::compiler::Compiler;
use super::errors::InterpreterError;
use super::reduce::DebruijnInterpreter;

//See rholang/src/main/scala/coop/rchain/rholang/interpreter/Interpreter.scala

// NOTE: Manual marks are used instead of trace_i() for async operations.
// This is the correct pattern for async code and matches Scala's Span[F].traceI() semantics.
#[derive(Clone, Debug, Default)]
pub struct EvaluateResult {
    pub cost: Cost,
    pub errors: Vec<InterpreterError>,
    pub mergeable: HashSet<Par>,
}

#[allow(async_fn_in_trait)]
pub trait Interpreter {
    async fn inj_attempt(
        &self,
        reducer: &DebruijnInterpreter,
        term: &str,
        initial_phlo: Cost,
        normalizer_env: HashMap<String, Par>,
        rand: Blake2b512Random,
    ) -> Result<EvaluateResult, InterpreterError>;
}

pub struct InterpreterImpl {
    c: _cost,
    merge_chs: Arc<RwLock<HashSet<Par>>>,
}

fn block_creator_phase_substep_profile_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("F1R3_BLOCK_CREATOR_PHASE_SUBSTEP_PROFILE")
            .ok()
            .map(|v| {
                let normalized = v.trim().to_ascii_lowercase();
                matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
    })
}

impl Interpreter for InterpreterImpl {
    async fn inj_attempt(
        &self,
        reducer: &DebruijnInterpreter,
        term: &str,
        initial_phlo: Cost,
        normalizer_env: HashMap<String, Par>,
        rand: Blake2b512Random,
    ) -> Result<EvaluateResult, InterpreterError> {
        let mem_profile_enabled = block_creator_phase_substep_profile_enabled();
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
                    "inj_attempt.mem step={} rss_kb={} delta_prev_kb={} delta_total_kb={}",
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

        let parsing_cost = parsing_cost(term);
        log_mem_step("after_parsing_cost");

        // Using tracing events for async context
        // Scala spans: "set-initial-cost", "charge-parsing-cost", "build-normalized-term", "reduce-term"
        // Implemented as debug events since this is an async function
        let evaluation_result: Result<EvaluateResult, InterpreterError> = {
            // Trace: set-initial-cost (matching Scala's Span[F].traceI("set-initial-cost"))
            {
                event!(
                    Level::DEBUG,
                    mark = "started-set-initial-cost",
                    "inj_attempt"
                );
                let _ = self.c.set(initial_phlo.clone());
                event!(
                    Level::DEBUG,
                    mark = "finished-set-initial-cost",
                    "inj_attempt"
                );
                log_mem_step("after_set_initial_cost");
            }

            // Trace: charge-parsing-cost (matching Scala's Span[F].traceI("charge-parsing-cost"))
            {
                event!(
                    Level::DEBUG,
                    mark = "started-charge-parsing-cost",
                    "inj_attempt"
                );
                // Scala: charge[F](parsingCost) is inside for-comprehension with .handleErrorWith at the end
                // In Rust, we must catch charge errors explicitly to match Scala's monadic error handling.
                // If charge fails (e.g., OutOfPhlogistonsError), convert to EvaluateResult with errors.
                if let Err(e) = self.c.charge(parsing_cost.clone()) {
                    event!(
                        Level::DEBUG,
                        mark = "failed-charge-parsing-cost",
                        "inj_attempt"
                    );
                    log_mem_step("charge_parsing_cost_error");
                    return self.handle_error(initial_phlo.clone(), parsing_cost.clone(), e);
                }
                event!(
                    Level::DEBUG,
                    mark = "finished-charge-parsing-cost",
                    "inj_attempt"
                );
                log_mem_step("after_charge_parsing_cost");
            }

            // Trace: build-normalized-term (matching Scala's Span[F].traceI("build-normalized-term"))
            let parsed = {
                event!(
                    Level::DEBUG,
                    mark = "started-build-normalized-term",
                    "inj_attempt"
                );
                let result =
                    match Compiler::source_to_adt_with_normalizer_env(&term, normalizer_env) {
                        Ok(p) => {
                            event!(
                                Level::DEBUG,
                                mark = "finished-build-normalized-term",
                                "inj_attempt"
                            );
                            Ok(p)
                        }
                        Err(e) => {
                            event!(
                                Level::DEBUG,
                                mark = "failed-build-normalized-term",
                                "inj_attempt"
                            );
                            log_mem_step("build_normalized_term_error");
                            Err(self.handle_error(
                                initial_phlo.clone(),
                                parsing_cost.clone(),
                                InterpreterError::ParserError(e.to_string()),
                            ))
                        }
                    };
                match result {
                    Ok(p) => p,
                    Err(err) => return err,
                }
            };
            log_mem_step("after_build_normalized_term");

            // Empty mergeable channels
            {
                let mut merge_chs_lock = self.merge_chs.write().unwrap();
                merge_chs_lock.clear();
            }
            log_mem_step("after_clear_mergeable_channels");

            // Diagnostic: log the RNG state before injection so we can compare
            // with the state in eval_new's alloc
            tracing::info!(
                target: "f1r3fly.rholang.diag",
                rand_position = rand.position,
                rand_path_position = rand.path_position,
                rand_last_block_prefix = %hex::encode(
                    &rand.last_block.iter().take(16).map(|&b| b as u8).collect::<Vec<u8>>()
                ),
                rand_hash_array_prefix = %hex::encode(
                    &rand.hash_array.iter().take(16).map(|&b| b as u8).collect::<Vec<u8>>()
                ),
                "inj_attempt: RNG state before reducer.inj"
            );

            // Trace: reduce-term (matching Scala's Span[F].traceI("reduce-term"))
            event!(Level::DEBUG, mark = "started-reduce-term", "inj_attempt");
            log_mem_step("before_reduce_term");
            let reduce_result = reducer.inj(parsed, rand).await;
            match reduce_result {
                Ok(()) => {
                    event!(Level::DEBUG, mark = "finished-reduce-term", "inj_attempt");
                    log_mem_step("after_reduce_term_ok");
                    let phlos_left = self.c.get();
                    let phlos_used = initial_phlo.value - phlos_left.value;
                    tracing::debug!(
                        target: "f1r3fly.rholang",
                        initial_phlo = initial_phlo.value,
                        phlos_left = phlos_left.value,
                        used = phlos_used,
                        "inj_attempt: deploy evaluation complete — computing cost"
                    );

                    // Diagnostic: dump hot store pending state after evaluation.
                    // Non-zero data/continuation counts indicate work that the
                    // reducer did NOT complete — potential early termination.
                    {
                        let (data_channels, data_items, cont_channels, cont_items) =
                            reducer.space.pending_state_counts();

                        tracing::info!(
                            target: "f1r3fly.rholang",
                            data_channels,
                            data_items,
                            cont_channels,
                            cont_items,
                            phlos_used,
                            "inj_attempt: hot store state after evaluation"
                        );

                        if cont_items > 0 {
                            tracing::warn!(
                                target: "f1r3fly.rholang",
                                data_channels,
                                data_items,
                                cont_channels,
                                cont_items,
                                phlos_used,
                                "inj_attempt: PENDING CONTINUATIONS after evaluation — \
                                 continuations are waiting for data that never arrived, \
                                 this may indicate the reducer terminated before all \
                                 concurrent branches completed"
                            );

                            // Enumerate the actual channels with pending
                            // continuations for cross-referencing
                            let cont_detail = reducer.space.pending_continuation_channels_debug();
                            for (i, (channels_dbg, num_conts, has_peek)) in cont_detail.iter().enumerate() {
                                tracing::warn!(
                                    target: "f1r3fly.rholang.diag",
                                    cont_idx = i,
                                    num_continuations = num_conts,
                                    has_peek,
                                    channels = %channels_dbg,
                                    "PENDING CONT[{}]: {} continuation(s), peek={}, channels={}",
                                    i, num_conts, has_peek, channels_dbg
                                );
                            }
                        }
                    }
                    log_mem_step("after_hot_store_diagnostic");

                    let mergeable_channels = { self.merge_chs.read().unwrap().clone() };

                    Ok(EvaluateResult {
                        cost: initial_phlo.clone() - phlos_left,
                        errors: Vec::new(),
                        mergeable: mergeable_channels,
                    })
                }
                Err(e) => {
                    event!(Level::DEBUG, mark = "failed-reduce-term", "inj_attempt");
                    log_mem_step("after_reduce_term_error");
                    self.handle_error(initial_phlo.clone(), parsing_cost.clone(), e)
                }
            }
        };
        log_mem_step("finish");

        evaluation_result
    }
}

impl InterpreterImpl {
    pub fn new(cost: _cost, merge_chs: Arc<RwLock<HashSet<Par>>>) -> InterpreterImpl {
        InterpreterImpl { c: cost, merge_chs }
    }

    fn handle_error(
        &self,
        initial_cost: Cost,
        parsing_cost: Cost,
        error: InterpreterError,
    ) -> Result<EvaluateResult, InterpreterError> {
        match error {
            // Parsing error consumes only parsing cost
            InterpreterError::ParserError(_) => Ok(EvaluateResult {
                cost: parsing_cost,
                errors: vec![error],
                mergeable: HashSet::new(),
            }),

            // For Out Of Phlogistons error initial cost is used because evaluated cost can be higher
            // all phlos are consumed
            InterpreterError::OutOfPhlogistonsError => Ok(EvaluateResult {
                cost: initial_cost,
                errors: vec![error],
                mergeable: HashSet::new(),
            }),

            // User triggered abort - execution failed, return cost consumed so far
            InterpreterError::UserAbortError => Ok(EvaluateResult {
                cost: initial_cost.clone() - self.c.get(),
                errors: vec![error],
                mergeable: HashSet::new(),
            }),

            // InterpreterError(s) - multiple errors are result of parallel execution
            InterpreterError::AggregateError { interpreter_errors } => Ok(EvaluateResult {
                cost: initial_cost,
                errors: interpreter_errors,
                mergeable: HashSet::new(),
            }),

            // TODO: Review why 'Compiler::source_to_adt_with_normalizer_env' doesn't pick this up
            // See 'compute_state_should_capture_rholang_parsing_errors_and_charge_for_parsing'
            InterpreterError::OperatorNotDefined {
                op: _,
                other_type: _,
            } => Ok(EvaluateResult {
                cost: parsing_cost,
                errors: vec![error],
                mergeable: HashSet::new(),
            }),

            // TODO: Review why 'Compiler::source_to_adt_with_normalizer_env' doesn't pick this up
            // See 'compute_state_should_capture_rholang_parsing_errors_and_charge_for_parsing'
            InterpreterError::OperatorExpectedError {
                op: _,
                expected: _,
                other_type: _,
            } => Ok(EvaluateResult {
                cost: parsing_cost,
                errors: vec![error],
                mergeable: HashSet::new(),
            }),

            // InterpreterError is returned as a result
            _ => Ok(EvaluateResult {
                cost: initial_cost,
                errors: vec![error],
                mergeable: HashSet::new(),
            }),
        }
    }
}
