use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::Par;
use rspace_plus_plus::rspace::merger::merging_logic::MergeType;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{event, Level};

use super::accounting::_cost;
use super::accounting::costs::{parsing_cost, Cost};
use super::compiler::compiler::Compiler;
use super::errors::InterpreterError;
use super::metrics_constants::{
    INJ_ATTEMPT_BUILD_NORMALIZED_TERM_TIME_METRIC, INJ_ATTEMPT_CHARGE_PARSING_COST_TIME_METRIC,
    INJ_ATTEMPT_REDUCE_TERM_TIME_METRIC, INJ_ATTEMPT_SET_INITIAL_COST_TIME_METRIC,
    INTERPRETER_METRICS_SOURCE,
};
use super::reduce::DebruijnInterpreter;

//See rholang/src/main/scala/coop/rchain/rholang/interpreter/Interpreter.scala

// NOTE: Manual marks are used instead of trace_i() for async operations.
// This is the correct pattern for async code and matches Scala's Span[F].traceI() semantics.
#[derive(Clone, Debug, Default)]
pub struct EvaluateResult {
    pub cost: Cost,
    pub errors: Vec<InterpreterError>,
    pub mergeable: HashMap<Par, MergeType>,
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
    merge_chs: Arc<RwLock<HashMap<Par, MergeType>>>,
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
        let parsing_cost = parsing_cost(term);

        let evaluation_result: Result<EvaluateResult, InterpreterError> = {
            // Phase: set-initial-cost
            {
                let phase_start = Instant::now();
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
                metrics::histogram!(
                    INJ_ATTEMPT_SET_INITIAL_COST_TIME_METRIC,
                    "source" => INTERPRETER_METRICS_SOURCE
                )
                .record(phase_start.elapsed().as_secs_f64());
            }

            // Phase: charge-parsing-cost. Charge can fail (OutOfPhlogistons);
            // convert that into an EvaluateResult with errors to mirror the
            // monadic error handling in the Scala reference.
            {
                let phase_start = Instant::now();
                event!(
                    Level::DEBUG,
                    mark = "started-charge-parsing-cost",
                    "inj_attempt"
                );
                if let Err(e) = self.c.charge(parsing_cost.clone()) {
                    event!(
                        Level::DEBUG,
                        mark = "failed-charge-parsing-cost",
                        "inj_attempt"
                    );
                    metrics::histogram!(
                        INJ_ATTEMPT_CHARGE_PARSING_COST_TIME_METRIC,
                        "source" => INTERPRETER_METRICS_SOURCE
                    )
                    .record(phase_start.elapsed().as_secs_f64());
                    return self.handle_error(initial_phlo.clone(), parsing_cost.clone(), e);
                }
                event!(
                    Level::DEBUG,
                    mark = "finished-charge-parsing-cost",
                    "inj_attempt"
                );
                metrics::histogram!(
                    INJ_ATTEMPT_CHARGE_PARSING_COST_TIME_METRIC,
                    "source" => INTERPRETER_METRICS_SOURCE
                )
                .record(phase_start.elapsed().as_secs_f64());
            }

            // Phase: build-normalized-term — parse the source string into an
            // AST.
            let parsed = {
                let phase_start = Instant::now();
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
                            Err(self.handle_error(
                                initial_phlo.clone(),
                                parsing_cost.clone(),
                                InterpreterError::ParserError(e.to_string()),
                            ))
                        }
                    };
                metrics::histogram!(
                    INJ_ATTEMPT_BUILD_NORMALIZED_TERM_TIME_METRIC,
                    "source" => INTERPRETER_METRICS_SOURCE
                )
                .record(phase_start.elapsed().as_secs_f64());
                match result {
                    Ok(p) => p,
                    Err(err) => return err,
                }
            };
            // Reset mergeable-channel tracking before reducing the new term.
            {
                let mut merge_chs_lock = self.merge_chs.write().await;
                merge_chs_lock.clear();
            }
            // Phase: reduce-term — execute the parsed AST through RSpace.
            let phase_start = Instant::now();
            event!(Level::DEBUG, mark = "started-reduce-term", "inj_attempt");
            let reduce_result = reducer.inj(parsed, rand).await;
            metrics::histogram!(
                INJ_ATTEMPT_REDUCE_TERM_TIME_METRIC,
                "source" => INTERPRETER_METRICS_SOURCE
            )
            .record(phase_start.elapsed().as_secs_f64());
            match reduce_result {
                Ok(()) => {
                    event!(Level::DEBUG, mark = "finished-reduce-term", "inj_attempt");
                    let phlos_left = self.c.get();
                    let mergeable_channels = { self.merge_chs.read().await.clone() };

                    Ok(EvaluateResult {
                        cost: initial_phlo.clone() - phlos_left,
                        errors: Vec::new(),
                        mergeable: mergeable_channels,
                    })
                }
                Err(e) => {
                    event!(Level::DEBUG, mark = "failed-reduce-term", "inj_attempt");
                    self.handle_error(initial_phlo.clone(), parsing_cost.clone(), e)
                }
            }
        };
        evaluation_result
    }
}

impl InterpreterImpl {
    pub fn new(cost: _cost, merge_chs: Arc<RwLock<HashMap<Par, MergeType>>>) -> InterpreterImpl {
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
                mergeable: HashMap::new(),
            }),

            // For Out Of Phlogistons error initial cost is used because evaluated cost can be higher
            // all phlos are consumed
            InterpreterError::OutOfPhlogistonsError => Ok(EvaluateResult {
                cost: initial_cost,
                errors: vec![error],
                mergeable: HashMap::new(),
            }),

            // User triggered abort - execution failed, return cost consumed so far
            InterpreterError::UserAbortError => Ok(EvaluateResult {
                cost: initial_cost.clone() - self.c.get(),
                errors: vec![error],
                mergeable: HashMap::new(),
            }),

            // InterpreterError(s) - multiple errors are result of parallel execution
            InterpreterError::AggregateError { interpreter_errors } => Ok(EvaluateResult {
                cost: initial_cost,
                errors: interpreter_errors,
                mergeable: HashMap::new(),
            }),

            // TODO: Review why 'Compiler::source_to_adt_with_normalizer_env' doesn't pick this up
            // See 'compute_state_should_capture_rholang_parsing_errors_and_charge_for_parsing'
            InterpreterError::OperatorNotDefined {
                op: _,
                other_type: _,
            } => Ok(EvaluateResult {
                cost: parsing_cost,
                errors: vec![error],
                mergeable: HashMap::new(),
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
                mergeable: HashMap::new(),
            }),

            // InterpreterError is returned as a result
            _ => Ok(EvaluateResult {
                cost: initial_cost,
                errors: vec![error],
                mergeable: HashMap::new(),
            }),
        }
    }
}
