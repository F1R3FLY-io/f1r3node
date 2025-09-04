// See casper/src/main/scala/coop/rchain/casper/util/rholang/SystemDeployUserError.scala

use std::fmt;

use models::rhoapi::Par;
use rholang::rust::interpreter::{errors::InterpreterError, pretty_printer::PrettyPrinter};

#[derive(Debug)]
pub struct SystemDeployUserError {
    pub error_message: String,
}

impl SystemDeployUserError {
    pub fn new(error_message: String) -> Self {
        Self { error_message }
    }
}

/**
 * Fatal error - node should exit on these errors.
 */
#[derive(Debug, Clone, PartialEq)]
pub enum SystemDeployPlatformFailure {
    UnexpectedResult(Vec<Par>),
    UnexpectedSystemErrors(Vec<InterpreterError>),
    GasRefundFailure(String),
    ConsumeFailed,
}

fn show_seq_par(pars: &[Par]) -> String {
    match pars {
        [] => "Nil".to_string(),
        [single] => {
            let mut pretty_printer = PrettyPrinter::new();
            pretty_printer.build_channel_string(single)
        }
        _ => format!(
            "({})",
            pars.iter()
                .map(|p| format!("{:?}", p))
                .collect::<Vec<_>>()
                .join(",\n ")
        ),
    }
}

impl std::error::Error for SystemDeployPlatformFailure {}

impl fmt::Display for SystemDeployPlatformFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SystemDeployPlatformFailure::UnexpectedResult(results) => {
                write!(f, "Unable to proceed with {}", show_seq_par(results))
            }
            SystemDeployPlatformFailure::UnexpectedSystemErrors(errors) => {
                write!(f, "Caught errors in Rholang interpreter {:?}", errors)
            }
            SystemDeployPlatformFailure::GasRefundFailure(msg) => {
                write!(f, "Unable to refund remaining gas ({})", msg)
            }
            SystemDeployPlatformFailure::ConsumeFailed => {
                write!(f, "Unable to consume results of system deploy")
            }
        }
    }
}

impl From<SystemDeployPlatformFailure> for SystemDeployUserError {
    fn from(failure: SystemDeployPlatformFailure) -> Self {
        Self {
            error_message: format!("Platform failure: {}", failure),
        }
    }
}
