use crate::rust::interpreter::compiler::exports::{FreeContext, FreeMap, SourcePosition};
// Import new span-based types
use crate::rust::interpreter::compiler::exports::{FreeContextSpan, FreeMapSpan};
use crate::rust::interpreter::compiler::normalize::VarSort;
use crate::rust::interpreter::compiler::span_utils::SpanContext;
use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::var::VarInstance::{FreeVar, Wildcard};
use models::rhoapi::var::WildcardMsg;
use models::rhoapi::Var as ModelsVar;

use super::processes::exports::Proc;
use crate::rust::interpreter::compiler::rholang_ast::Var;

// New AST imports for parallel functions
use rholang_parser::ast::{Var as NewVar, Id as NewId};

fn handle_proc_var(
    proc: &Proc,
    known_free: FreeMap<VarSort>,
) -> Result<(Option<ModelsVar>, FreeMap<VarSort>), InterpreterError> {
    // println!("\nhit handle_proc_var");
    // println!("\nknown_free: {:?}", known_free);
    match proc {
        Proc::Wildcard { line_num, col_num } => {
            let wildcard_var = ModelsVar {
                var_instance: Some(Wildcard(WildcardMsg {})),
            };
            let source_position = SourcePosition::new(*line_num, *col_num);
            Ok((Some(wildcard_var), known_free.add_wildcard(source_position)))
        }

        Proc::Var(Var {
            name,
            line_num,
            col_num,
        }) => {
            let source_position = SourcePosition::new(*line_num, *col_num);

            match known_free.get(&name) {
                None => {
                    let binding = (name.clone(), VarSort::ProcSort, source_position);
                    let new_bindings_pair = known_free.put(binding);
                    let free_var = ModelsVar {
                        var_instance: Some(FreeVar(known_free.next_level as i32)),
                    };
                    Ok((Some(free_var), new_bindings_pair))
                }
                Some(FreeContext {
                    source_position: first_source_position,
                    ..
                }) => Err(InterpreterError::UnexpectedReuseOfProcContextFree {
                    var_name: name.clone(),
                    first_use: first_source_position,
                    second_use: source_position,
                }),
            }
        }

        _ => Err(InterpreterError::NormalizerError(format!(
            "Expected Proc::Var or Proc::Wildcard, found {:?}",
            proc,
        ))),
    }
}

// coop.rchain.rholang.interpreter.compiler.normalizer.RemainderNormalizeMatcher.normalizeMatchProc
// This function is to be called in `collection_normalize_matcher`
// This handles the 'cont' field in our grammar.js for 'collection' types. AKA '_proc_remainder'
pub fn normalize_remainder(
    r: &Option<Box<Proc>>,
    known_free: FreeMap<VarSort>,
) -> Result<(Option<ModelsVar>, FreeMap<VarSort>), InterpreterError> {
    match r {
        Some(pr) => handle_proc_var(pr, known_free),
        None => Ok((None, known_free)),
    }
}

// This function handles the 'cont' field in our grammar.js for 'names' types. AKA '_name_remainder'
pub fn normalize_match_name(
    nr: &Option<Box<Proc>>,
    known_free: FreeMap<VarSort>,
) -> Result<(Option<ModelsVar>, FreeMap<VarSort>), InterpreterError> {
    match nr {
        Some(pr) => handle_proc_var(&pr, known_free),
        None => Ok((None, known_free)),
    }
}

// ============================================================================
// NEW AST PARALLEL FUNCTIONS
// ============================================================================

/// Parallel version of handle_proc_var for new AST Var types
/// Handles the direct Var<'ast> instead of Proc containing Var
fn handle_new_var<'ast>(
    var: &NewVar<'ast>,
    known_free: FreeMapSpan<VarSort>,
) -> Result<(Option<ModelsVar>, FreeMapSpan<VarSort>), InterpreterError> {
    match var {
        NewVar::Wildcard => {
            let wildcard_var = ModelsVar {
                var_instance: Some(Wildcard(WildcardMsg {})),
            };
            // Current approach: Use synthetic span since rholang-rs Wildcard lacks position data
            // 
            // IDEAL: If rholang-rs enhanced Wildcard with SourcePos:
            //   let wildcard_span = SpanContext::pos_to_span(wildcard.pos);
            // 
            // BETTER: If we had access to containing construct span:
            //   let wildcard_span = SpanContext::wildcard_span_with_context(parent_span);
            // 
            // CURRENT: Synthetic span with valid 1-based coordinates
            let wildcard_span = SpanContext::wildcard_span();
            Ok((Some(wildcard_var), known_free.add_wildcard(wildcard_span)))
        }

        NewVar::Id(NewId { name, pos }) => {
            // Extract proper source position from Id and convert to span
            let source_span = SpanContext::pos_to_span(*pos);

            match known_free.get(name) {
                None => {
                    // Use IdContextPos for single position Id types
                    let binding = (name.to_string(), VarSort::ProcSort, *pos);
                    let new_bindings_pair = known_free.put_pos(binding);
                    let free_var = ModelsVar {
                        var_instance: Some(FreeVar(known_free.next_level as i32)),
                    };
                    Ok((Some(free_var), new_bindings_pair))
                }
                Some(FreeContextSpan {
                    source_span: first_source_span,
                    ..
                }) => Err(InterpreterError::UnexpectedReuseOfProcContextFreeSpan {
                    var_name: name.to_string(),
                    first_use: first_source_span,
                    second_use: source_span,
                }),
            }
        }
    }
}

/// Parallel version of normalize_remainder for new AST types
/// Handles Option<Var<'ast>> instead of Option<Box<Proc>>
pub fn normalize_remainder_new_ast<'ast>(
    r: &Option<NewVar<'ast>>,
    known_free: FreeMapSpan<VarSort>,
) -> Result<(Option<ModelsVar>, FreeMapSpan<VarSort>), InterpreterError> {
    match r {
        Some(var) => handle_new_var(var, known_free),
        None => Ok((None, known_free)),
    }
}

/// Parallel version of normalize_match_name for new AST types
/// This handles remainder variables in Names structures
pub fn normalize_match_name_new_ast<'ast>(
    nr: &Option<NewVar<'ast>>,
    known_free: FreeMapSpan<VarSort>,
) -> Result<(Option<ModelsVar>, FreeMapSpan<VarSort>), InterpreterError> {
    match nr {
        Some(var) => handle_new_var(var, known_free),
        None => Ok((None, known_free)),
    }
}