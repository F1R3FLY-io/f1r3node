use crate::rust::interpreter::compiler::normalize::{NameVisitOutputsSpan, ProcVisitInputsSpan};
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::errors::InterpreterError::PatternReceiveError;
use models::rhoapi::connective::ConnectiveInstance;

pub fn fail_on_invalid_connective_span(
    input: &ProcVisitInputsSpan,
    name_res: &NameVisitOutputsSpan,
) -> Result<NameVisitOutputsSpan, InterpreterError> {
    if input.bound_map_chain.depth() == 0 {
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
        Ok(name_res.clone())
    }
}
