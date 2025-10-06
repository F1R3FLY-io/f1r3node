use crate::rust::interpreter::compiler::exports::{FreeContext, FreeMap};
use crate::rust::interpreter::compiler::normalize::VarSort;
use crate::rust::interpreter::compiler::span_utils::SpanContext;
use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::var::VarInstance::{FreeVar, Wildcard};
use models::rhoapi::var::WildcardMsg;
use models::rhoapi::Var as ModelsVar;

use rholang_parser::ast::{Id, Var};

fn handle_var<'ast>(
    var: &Var<'ast>,
    known_free: FreeMap<VarSort>,
) -> Result<(Option<ModelsVar>, FreeMap<VarSort>), InterpreterError> {
    match var {
        Var::Wildcard => {
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

        Var::Id(Id { name, pos }) => {
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
                Some(FreeContext {
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

pub fn normalize_remainder<'ast>(
    r: &Option<Var<'ast>>,
    known_free: FreeMap<VarSort>,
) -> Result<(Option<ModelsVar>, FreeMap<VarSort>), InterpreterError> {
    match r {
        Some(var) => handle_var(var, known_free),
        None => Ok((None, known_free)),
    }
}

pub fn normalize_match_name<'ast>(
    nr: &Option<Var<'ast>>,
    known_free: FreeMap<VarSort>,
) -> Result<(Option<ModelsVar>, FreeMap<VarSort>), InterpreterError> {
    match nr {
        Some(var) => handle_var(var, known_free),
        None => Ok((None, known_free)),
    }
}
