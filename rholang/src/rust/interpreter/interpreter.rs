use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::Par;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
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

        // Using tracing events for async context
        // Scala spans: "set-initial-cost", "charge-parsing-cost", "build-normalized-term", "reduce-term"
        // Implemented as debug events since this is an async function
        let evaluation_result: Result<EvaluateResult, InterpreterError> = {
            // Trace: set-initial-cost (matching Scala's Span[F].traceI("set-initial-cost"))
            {
                event!(Level::DEBUG, mark = "started-set-initial-cost", "inj_attempt");
                let _ = self.c.set(initial_phlo.clone());
                event!(Level::DEBUG, mark = "finished-set-initial-cost", "inj_attempt");
            }
            
            // Trace: charge-parsing-cost (matching Scala's Span[F].traceI("charge-parsing-cost"))
            {
                event!(Level::DEBUG, mark = "started-charge-parsing-cost", "inj_attempt");
                // Scala: charge[F](parsingCost) is inside for-comprehension with .handleErrorWith at the end
                // In Rust, we must catch charge errors explicitly to match Scala's monadic error handling.
                // If charge fails (e.g., OutOfPhlogistonsError), convert to EvaluateResult with errors.
                if let Err(e) = self.c.charge(parsing_cost.clone()) {
                    event!(Level::DEBUG, mark = "failed-charge-parsing-cost", "inj_attempt");
                    return self.handle_error(initial_phlo.clone(), parsing_cost.clone(), e);
                }
                event!(Level::DEBUG, mark = "finished-charge-parsing-cost", "inj_attempt");
            }
            
            // Trace: build-normalized-term (matching Scala's Span[F].traceI("build-normalized-term"))
            let parsed = {
                event!(Level::DEBUG, mark = "started-build-normalized-term", "inj_attempt");
                let result = match Compiler::source_to_adt_with_normalizer_env(&term, normalizer_env) {
                    Ok(p) => {
                        event!(Level::DEBUG, mark = "finished-build-normalized-term", "inj_attempt");
                        Ok(p)
                    },
                    Err(e) => {
                        event!(Level::DEBUG, mark = "failed-build-normalized-term", "inj_attempt");
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

            // Empty mergeable channels
            {
                let mut merge_chs_lock = self.merge_chs.write().unwrap();
                merge_chs_lock.clear();
            }

            // Trace: reduce-term (matching Scala's Span[F].traceI("reduce-term"))
            event!(Level::DEBUG, mark = "started-reduce-term", "inj_attempt");
            let reduce_result = reducer.inj(parsed, rand).await;
            match reduce_result {
                Ok(()) => {
                    event!(Level::DEBUG, mark = "finished-reduce-term", "inj_attempt");
                    let phlos_left = self.c.get();
                    let mergeable_channels = { self.merge_chs.read().unwrap().clone() };

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
