// See node/src/main/scala/coop/rchain/node/api/package.scala

use std::time::Duration;

use models::casper::v1::deploy_service_server::DeployServiceServer;
use models::casper::v1::propose_service_server::ProposeServiceServer;
use tonic::transport::Server as TonicServer;

use crate::rust::api::deploy_grpc_service_v1::DeployGrpcServiceV1Impl;
use crate::rust::api::lsp_grpc_service::lsp::lsp_server::LspServer;
use crate::rust::api::lsp_grpc_service::LspGrpcServiceImpl;
use crate::rust::api::propose_grpc_service_v1::ProposeGrpcServiceV1Impl;
use crate::rust::api::repl_grpc_service::repl::repl_server::ReplServer;
use crate::rust::api::repl_grpc_service::ReplGrpcServiceImpl;

pub const FILE_DESCRIPTOR_SET: &[u8] =
    include_bytes!("../../../build/descriptors/reflection_protos.bin");

// Note: Deploy and Propose services are defined in the models crate
// These would be imported from models::casper::v1::{deploy_service_v1_server, propose_service_v1_server}

/// Create an internal gRPC server with all services (Repl, Propose, Deploy, Lsp)
///
/// This function creates a gRPC server that includes all available services:
/// - REPL service for executing Rholang code
/// - Propose service for block proposals
/// - Deploy service for deploying contracts and querying blocks
/// - LSP service for code validation
///
/// Returns a router that can be started with `GrpcServer::start_with_router`.
///
/// # Arguments
/// * `repl_grpc_service` - REPL service implementation
/// * `deploy_grpc_service` - Deploy service implementation
/// * `propose_grpc_service` - Propose service implementation
/// * `lsp_grpc_service` - LSP service implementation
/// * `max_message_size` - Maximum inbound message size in bytes
/// * `keep_alive_time` - Duration for keep-alive ping interval
/// * `keep_alive_timeout` - Duration to wait for keep-alive ping acknowledgment
/// * `permit_keep_alive_time` - Duration to wait for keep-alive ping without data (not directly supported in tonic)
/// * `max_connection_idle` - Maximum time a connection can be idle
/// * `max_connection_age` - Maximum age of a connection (not directly supported in tonic)
/// * `max_connection_age_grace` - Grace period for closing connections after max_connection_age (not directly supported in tonic)
pub async fn acquire_internal_server(
    repl_grpc_service: ReplGrpcServiceImpl,
    deploy_grpc_service: DeployGrpcServiceV1Impl,
    propose_grpc_service: ProposeGrpcServiceV1Impl,
    lsp_grpc_service: LspGrpcServiceImpl,
    max_message_size: usize,
    keep_alive_time: Duration,
    keep_alive_timeout: Duration,
    permit_keep_alive_time: Duration,
    max_connection_idle: Duration,
    max_connection_age: Duration,
    _max_connection_age_grace: Duration,
) -> Result<tonic::transport::server::Router, Box<dyn std::error::Error + Send + Sync>> {
    // Create adapter wrappers that implement the proto-generated server traits
    // Note: These adapters need to be implemented separately to bridge between
    // the trait-based service implementations and the proto-generated server traits
    let repl_server = ReplServer::new(repl_grpc_service);
    let lsp_server = LspServer::new(lsp_grpc_service);
    let propose_server = ProposeServiceServer::new(propose_grpc_service);
    let deploy_server = DeployServiceServer::new(deploy_grpc_service);

    let reflection_service = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
        .build_v1()?;

    // Build the server router with all services
    let router = TonicServer::builder()
        .tcp_keepalive(Some(permit_keep_alive_time))
        .max_frame_size(Some(max_message_size as u32))
        .http2_keepalive_interval(Some(keep_alive_time))
        .http2_keepalive_timeout(Some(keep_alive_timeout))
        .http2_adaptive_window(Some(true))
        .timeout(max_connection_idle)
        .max_connection_age(max_connection_age)
        .concurrency_limit_per_connection(1024)
        .add_service(repl_server)
        .add_service(lsp_server)
        .add_service(deploy_server)
        .add_service(propose_server)
        .add_service(reflection_service);

    Ok(router)
}

/// Create an external gRPC server with only the Deploy service
///
/// This function creates a gRPC server that only includes the Deploy service,
/// intended for external access without internal administrative services.
///
/// Returns a router that can be started with `GrpcServer::start_with_router`.
///
/// # Arguments
/// * `deploy_grpc_service` - Deploy service implementation
/// * `max_message_size` - Maximum inbound message size in bytes
/// * `keep_alive_time` - Duration for keep-alive ping interval
/// * `keep_alive_timeout` - Duration to wait for keep-alive ping acknowledgment
/// * `permit_keep_alive_time` - Duration to wait for keep-alive ping without data (not directly supported in tonic)
/// * `max_connection_idle` - Maximum time a connection can be idle
/// * `max_connection_age` - Maximum age of a connection (not directly supported in tonic)
/// * `max_connection_age_grace` - Grace period for closing connections after max_connection_age (not directly supported in tonic)
pub fn acquire_external_server(
    deploy_grpc_service: DeployGrpcServiceV1Impl,
    max_message_size: usize,
    keep_alive_time: Duration,
    keep_alive_timeout: Duration,
    permit_keep_alive_time: Duration,
    max_connection_idle: Duration,
    max_connection_age: Duration,
    _max_connection_age_grace: Duration,
) -> Result<tonic::transport::server::Router, Box<dyn std::error::Error + Send + Sync>> {
    // Create adapter wrappers that implement the proto-generated server traits
    // Note: These adapters need to be implemented separately to bridge between
    // the trait-based service implementations and the proto-generated server traits
    let deploy_server = DeployServiceServer::new(deploy_grpc_service);

    let reflection_service = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
        .build_v1()?;

    // Build the server router with all services
    let router = TonicServer::builder()
        .tcp_keepalive(Some(permit_keep_alive_time))
        .max_frame_size(Some(max_message_size as u32))
        .http2_keepalive_interval(Some(keep_alive_time))
        .http2_keepalive_timeout(Some(keep_alive_timeout))
        .http2_adaptive_window(Some(true))
        .timeout(max_connection_idle)
        .max_connection_age(max_connection_age)
        .concurrency_limit_per_connection(1024)
        .add_service(deploy_server)
        .add_service(reflection_service);

    Ok(router)
}
