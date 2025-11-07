use errors::InterpreterError;

pub mod accounting;
pub mod compiler;
pub mod contract_call;
pub mod deploy_parameters;
pub mod dispatch;
pub mod env;
pub mod errors;
pub mod interpreter;

// TODO: Port RhoRuntime.scala:125-126 when creating runtime module:
// - Span[F].trace(createPlayRuntime)
// - Span[F].trace(createReplayRuntime)
pub mod matcher;
pub mod merging;
pub mod openai_service;
pub mod pretty_printer;
pub mod reduce;
pub mod registry;
pub mod rho_runtime;
pub mod rho_type;
pub mod storage;
pub mod substitute;
pub mod system_processes;
pub mod test_utils;
pub mod util;

pub fn unwrap_option_safe<A: Clone + std::fmt::Debug>(
    opt: Option<A>,
) -> Result<A, InterpreterError> {
    opt.map(|x| x.clone()).ok_or_else(|| {
        InterpreterError::UndefinedRequiredProtobufFieldError(format!(
            "{:?}",
            std::any::type_name::<A>()
        ))
    })
}
