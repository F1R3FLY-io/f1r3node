pub mod casper_packet_handler;
pub mod deploy_runtime;
pub mod fair_round_robin_dispatcher;
pub mod grpc_deploy_service;
pub mod grpc_propose_service;
pub mod listen_at_name;

type ServiceResult<T> = std::result::Result<T, Vec<String>>; // left the type of Err as Vec<String> for compatibility with Scala version

/// Convert an Error into the ServiceResult::Err.
fn to_service_error<E: std::error::Error>(e: E) -> ServiceResult<()> {
    Err(vec![e.to_string()])
}

pub fn error_to_vec<E: std::fmt::Display>(err: E) -> Vec<String> {
    vec![err.to_string()]
}
