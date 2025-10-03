pub mod deploy_runtime;
pub mod grpc_deploy_service;
pub mod grpc_propose_service;
pub mod listen_at_name;

type ServiceResult<T> = std::result::Result<T, Vec<String>>; // left the type of Err as Vec<String> for compatibility with Scala version

/// Convert an Error into the ServiceResult::Err.
fn error_to_vec<E: std::fmt::Debug>(e: E) -> Vec<String> {
    vec![format!("{e:?}")]
}
