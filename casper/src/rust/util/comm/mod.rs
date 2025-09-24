pub mod grpc_deploy_client;
pub mod grpc_propose_client;

type ServiceResult<T> = std::result::Result<T, Vec<String>>; // left the type of Err as Vec<String> for compatibility with Scala version

/// Convert an Error into the ServiceResult::Err.
fn error_to_vec<E: std::fmt::Debug>(e: E) -> Vec<String> {
    vec![format!("{e:?}")]
}
