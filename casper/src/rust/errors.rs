use std::fmt;

use comm::rust::errors::CommError;
use rholang::rust::interpreter::errors::InterpreterError;
use rspace_plus_plus::rspace::errors::HistoryError;
use shared::rust::store::key_value_store::KvStoreError;

use super::util::rholang::{
    replay_failure::ReplayFailure, system_deploy_user_error::SystemDeployPlatformFailure,
};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CasperError {
    InterpreterError(InterpreterError),
    KvStoreError(KvStoreError),
    RuntimeError(String),
    SystemRuntimeError(SystemDeployPlatformFailure),
    SigningError(String),
    ReplayFailure(ReplayFailure),
    CommError(CommError),
    HistoryError(HistoryError),
    StreamError(String),
    Other(String),
}

impl fmt::Display for CasperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CasperError::InterpreterError(error) => write!(f, "Interpreter error: {}", error),
            CasperError::KvStoreError(error) => write!(f, "KvStore error: {}", error),
            CasperError::RuntimeError(error) => write!(f, "Runtime error: {}", error),
            CasperError::SystemRuntimeError(error) => write!(f, "System runtime error: {}", error),
            CasperError::SigningError(error) => write!(f, "Signing error: {}", error),
            CasperError::ReplayFailure(error) => write!(f, "Replay failure: {}", error),
            CasperError::CommError(error) => write!(f, "Comm error: {}", error),
            CasperError::HistoryError(error) => write!(f, "History error: {}", error),
            CasperError::StreamError(error) => write!(f, "Stream error: {}", error),
            CasperError::Other(error) => write!(f, "Other error: {}", error),
        }
    }
}

impl From<InterpreterError> for CasperError {
    fn from(error: InterpreterError) -> Self {
        CasperError::InterpreterError(error)
    }
}

impl From<KvStoreError> for CasperError {
    fn from(error: KvStoreError) -> Self {
        CasperError::KvStoreError(error)
    }
}

impl From<ReplayFailure> for CasperError {
    fn from(error: ReplayFailure) -> Self {
        CasperError::ReplayFailure(error)
    }
}

impl From<CommError> for CasperError {
    fn from(error: CommError) -> Self {
        CasperError::CommError(error)
    }
}
