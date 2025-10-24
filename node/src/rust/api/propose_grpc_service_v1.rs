//! Propose gRPC Service V1 implementation
//!
//! This module provides a gRPC service for block proposal functionality,
//! allowing clients to trigger block proposals and get proposal results.

use std::sync::Arc;

use casper::rust::api::block_api::BlockAPI;
use casper::rust::engine::engine_cell::EngineCell;
use casper::rust::state::instances::proposer_state::ProposerState;
use casper::rust::ProposeFunction;
use eyre::Result;
use models::casper::v1::{ProposeResponse, ProposeResultResponse};
use models::casper::{ProposeQuery, ProposeResultQuery};
use models::servicemodelapi::ServiceError;
use tokio::sync::RwLock;

/// Propose gRPC Service V1 trait defining the interface for propose operations
#[async_trait::async_trait]
pub trait ProposeGrpcServiceV1 {
    /// Trigger a block proposal
    async fn propose(&self, request: ProposeQuery) -> Result<ProposeResponse>;

    /// Get the result of the latest proposal
    async fn propose_result(&self, request: ProposeResultQuery) -> Result<ProposeResultResponse>;
}

/// Propose gRPC Service V1 implementation
pub struct ProposeGrpcServiceV1Impl {
    trigger_propose_f_opt: Option<Box<ProposeFunction>>,
    proposer_state_ref_opt: Option<Arc<RwLock<ProposerState>>>,
    engine_cell: Arc<EngineCell>,
}

impl ProposeGrpcServiceV1Impl {
    pub fn new(
        trigger_propose_f_opt: Option<Box<ProposeFunction>>,
        proposer_state_ref_opt: Option<Arc<RwLock<ProposerState>>>,
        engine_cell: Arc<EngineCell>,
    ) -> Self {
        Self {
            trigger_propose_f_opt,
            proposer_state_ref_opt,
            engine_cell,
        }
    }

    /// Helper function to convert errors to ServiceError
    fn create_service_error(message: String) -> ServiceError {
        ServiceError {
            messages: vec![message],
        }
    }

    /// Helper function to create a successful ProposeResponse
    fn create_success_propose_response(result: String) -> ProposeResponse {
        ProposeResponse {
            message: Some(models::casper::v1::propose_response::Message::Result(
                result,
            )),
        }
    }

    /// Helper function to create an error ProposeResponse
    fn create_error_propose_response(error: ServiceError) -> ProposeResponse {
        ProposeResponse {
            message: Some(models::casper::v1::propose_response::Message::Error(error)),
        }
    }

    /// Helper function to create a successful ProposeResultResponse
    fn create_success_propose_result_response(result: String) -> ProposeResultResponse {
        ProposeResultResponse {
            message: Some(models::casper::v1::propose_result_response::Message::Result(result)),
        }
    }

    /// Helper function to create an error ProposeResultResponse
    fn create_error_propose_result_response(error: ServiceError) -> ProposeResultResponse {
        ProposeResultResponse {
            message: Some(models::casper::v1::propose_result_response::Message::Error(
                error,
            )),
        }
    }
}

#[async_trait::async_trait]
impl ProposeGrpcServiceV1 for ProposeGrpcServiceV1Impl {
    async fn propose(&self, request: ProposeQuery) -> Result<ProposeResponse> {
        match &self.trigger_propose_f_opt {
            Some(trigger_propose_f) => {
                match BlockAPI::create_block(&self.engine_cell, trigger_propose_f, request.is_async)
                    .await
                {
                    Ok(result) => Ok(Self::create_success_propose_response(result)),
                    Err(e) => {
                        let error = Self::create_service_error(format!(
                            "Propose service method error: {}",
                            e
                        ));
                        Ok(Self::create_error_propose_response(error))
                    }
                }
            }
            None => {
                let error =
                    Self::create_service_error("Propose error: read-only node.".to_string());
                Ok(Self::create_error_propose_response(error))
            }
        }
    }

    async fn propose_result(&self, _request: ProposeResultQuery) -> Result<ProposeResultResponse> {
        match &self.proposer_state_ref_opt {
            Some(proposer_state_ref) => {
                let mut proposer_state = proposer_state_ref.write().await;
                match BlockAPI::get_propose_result(&mut proposer_state).await {
                    Ok(result) => Ok(Self::create_success_propose_result_response(result)),
                    Err(e) => {
                        let error = Self::create_service_error(format!(
                            "Propose service method error: {}",
                            e
                        ));
                        Ok(Self::create_error_propose_result_response(error))
                    }
                }
            }
            None => {
                let error = Self::create_service_error("Error: read-only node.".to_string());
                Ok(Self::create_error_propose_result_response(error))
            }
        }
    }
}
