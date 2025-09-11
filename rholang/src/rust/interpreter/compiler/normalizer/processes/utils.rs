use crate::rust::interpreter::compiler::normalize::{NameVisitOutputs, ProcVisitInputs};
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::errors::InterpreterError::PatternReceiveError;
use models::rhoapi::connective::ConnectiveInstance;

// Import span-based types for parallel function
use crate::rust::interpreter::compiler::normalize::{NameVisitOutputsSpan, ProcVisitInputsSpan};

pub fn fail_on_invalid_connective(
    input: &ProcVisitInputs,
    name_res: &NameVisitOutputs,
) -> Result<NameVisitOutputs, InterpreterError> {
    if input.bound_map_chain.depth() == 0 {
        // println!("fail_on_invalid_connective: input.bound_map_chain.depth() == 0");
        name_res
            .free_map
            .connectives
            .iter()
            .find_map(
                |(connective_instance, source_position)| match connective_instance {
                    ConnectiveInstance::ConnOrBody(_) => Some(PatternReceiveError(format!(
                        "\\/ (disjunction) at {:?}",
                        source_position
                    ))),
                    ConnectiveInstance::ConnNotBody(_) => Some(PatternReceiveError(format!(
                        "~ (negation) at {:?}",
                        source_position
                    ))),
                    _ => None,
                },
            )
            .map_or(Ok(name_res.clone()), Err)
    } else {
        // println!("else of input.bound_map_chain.depth() NOT EQUALS 0");
        Ok(name_res.clone())
    }
}

/// Parallel version of fail_on_invalid_connective for span-based types
pub fn fail_on_invalid_connective_span(
    input: &ProcVisitInputsSpan,
    name_res: &NameVisitOutputsSpan,
) -> Result<NameVisitOutputsSpan, InterpreterError> {
    if input.bound_map_chain.depth() == 0 {
        // println!("fail_on_invalid_connective_span: input.bound_map_chain.depth() == 0");
        name_res
            .free_map
            .connectives
            .iter()
            .find_map(
                |(connective_instance, source_span)| match connective_instance {
                    ConnectiveInstance::ConnOrBody(_) => Some(PatternReceiveError(format!(
                        "\\/ (disjunction) at {:?}",
                        source_span
                    ))),
                    ConnectiveInstance::ConnNotBody(_) => Some(PatternReceiveError(format!(
                        "~ (negation) at {:?}",
                        source_span
                    ))),
                    _ => None,
                },
            )
            .map_or(Ok(name_res.clone()), Err)
    } else {
        // println!("else of input.bound_map_chain.depth() NOT EQUALS 0");
        Ok(name_res.clone())
    }
}
