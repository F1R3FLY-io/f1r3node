//! Deploy gRPC Service V1 implementation
//!
//! This module provides a gRPC service for deploy functionality,
//! allowing clients to deploy contracts, query blocks, and perform various blockchain operations.

use std::sync::Arc;

use crate::rust::web::version_info::get_version_info_str;
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use casper::rust::api::block_api::BlockAPI;
use casper::rust::api::block_report_api::BlockReportAPI;
use casper::rust::api::graph_generator::{GraphConfig, GraphzGenerator};
use casper::rust::engine::engine_cell::EngineCell;
use casper::rust::ProposeFunction;
use comm::rust::discovery::node_discovery::NodeDiscovery;
use comm::rust::rp::connect::ConnectionsCell;
use comm::rust::rp::rp_conf::RPConf;
use graphz::{GraphSerializer, ListSerializer};
use models::casper::v1::deploy_service_server::DeployService;
use models::casper::v1::{
    BlockInfoResponse, BlockResponse, BondStatusResponse, ContinuationAtNameResponse,
    DeployResponse, EventInfoResponse, ExploratoryDeployResponse, FindDeployResponse,
    IsFinalizedResponse, LastFinalizedBlockResponse, ListeningNameDataResponse,
    MachineVerifyResponse, PrivateNamePreviewResponse, RhoDataResponse, StatusResponse,
    VisualizeBlocksResponse,
};
use models::casper::{
    BlockQuery, BlocksQuery, BlocksQueryByHeight, BondStatusQuery, ContinuationAtNameQuery,
    DataAtNameByBlockQuery, DataAtNameQuery, DeployDataProto, ExploratoryDeployQuery,
    FindDeployQuery, IsFinalizedQuery, LastFinalizedBlockQuery, MachineVerifyQuery,
    PrivateNamePreviewQuery, ReportQuery, Status, VersionInfo, VisualizeDagQuery,
};
use models::servicemodelapi::ServiceError;
use tracing::error;

trait IntoServiceError {
    fn into_service_error(self) -> ServiceError;
}

impl IntoServiceError for eyre::Report {
    fn into_service_error(self) -> ServiceError {
        ServiceError {
            messages: vec![self.to_string()],
        }
    }
}

/// Deploy gRPC Service V1 implementation
#[derive(Clone)]
pub struct DeployGrpcServiceV1Impl {
    api_max_blocks_limit: i32,
    trigger_propose_f: Option<Arc<ProposeFunction>>,
    dev_mode: bool,
    network_id: String,
    shard_id: String,
    min_phlo_price: i64,
    is_node_read_only: bool,
    engine_cell: EngineCell,
    block_report_api: BlockReportAPI,
    key_value_block_store: KeyValueBlockStore,
    rp_conf: RPConf,
    connections_cell: ConnectionsCell,
    node_discovery: Arc<dyn NodeDiscovery + Send + Sync>,
}

impl DeployGrpcServiceV1Impl {
    pub fn new(
        api_max_blocks_limit: i32,
        trigger_propose_f: Option<Arc<ProposeFunction>>,
        dev_mode: bool,
        network_id: String,
        shard_id: String,
        min_phlo_price: i64,
        is_node_read_only: bool,
        engine_cell: EngineCell,
        block_report_api: BlockReportAPI,
        key_value_block_store: KeyValueBlockStore,
        rp_conf: RPConf,
        connections_cell: ConnectionsCell,
        node_discovery: Arc<dyn NodeDiscovery + Send + Sync>,
    ) -> Self {
        Self {
            api_max_blocks_limit,
            trigger_propose_f,
            dev_mode,
            network_id,
            shard_id,
            min_phlo_price,
            is_node_read_only,
            engine_cell,
            block_report_api,
            key_value_block_store,
            rp_conf,
            connections_cell,
            node_discovery,
        }
    }

    /// Helper function to convert errors to ServiceError
    fn create_service_error(message: String) -> ServiceError {
        ServiceError {
            messages: vec![message],
        }
    }

    /// Helper function to create a successful DeployResponse
    fn create_success_deploy_response(
        result: String,
    ) -> Result<tonic::Response<DeployResponse>, tonic::Status> {
        Ok(DeployResponse {
            message: Some(models::casper::v1::deploy_response::Message::Result(result)),
        }
        .into())
    }

    /// Helper function to create an error DeployResponse
    fn create_error_deploy_response(
        error: ServiceError,
    ) -> Result<tonic::Response<DeployResponse>, tonic::Status> {
        Ok(DeployResponse {
            message: Some(models::casper::v1::deploy_response::Message::Error(error)),
        }
        .into())
    }

    /// Helper function to create a successful BlockResponse
    fn create_success_block_response(
        block_info: models::casper::BlockInfo,
    ) -> Result<tonic::Response<BlockResponse>, tonic::Status> {
        Ok(BlockResponse {
            message: Some(models::casper::v1::block_response::Message::BlockInfo(
                block_info,
            )),
        }
        .into())
    }

    /// Helper function to create an error BlockResponse
    fn create_error_block_response(
        error: ServiceError,
    ) -> Result<tonic::Response<BlockResponse>, tonic::Status> {
        Ok(BlockResponse {
            message: Some(models::casper::v1::block_response::Message::Error(error)),
        }
        .into())
    }
}

#[async_trait::async_trait]
impl DeployService for DeployGrpcServiceV1Impl {
    type showMainChainStream = tokio_stream::wrappers::ReceiverStream<
        std::result::Result<BlockInfoResponse, tonic::Status>,
    >;

    type visualizeDagStream = tokio_stream::wrappers::ReceiverStream<
        std::result::Result<VisualizeBlocksResponse, tonic::Status>,
    >;
    type getBlocksStream = tokio_stream::wrappers::ReceiverStream<
        std::result::Result<BlockInfoResponse, tonic::Status>,
    >;
    type getBlocksByHeightsStream = tokio_stream::wrappers::ReceiverStream<
        std::result::Result<BlockInfoResponse, tonic::Status>,
    >;

    /// Deploy a contract
    async fn do_deploy(
        &self,
        request: tonic::Request<DeployDataProto>,
    ) -> Result<tonic::Response<DeployResponse>, tonic::Status> {
        // Convert DeployDataProto to Signed<DeployData>
        let signed_deploy =
            match models::rust::casper::protocol::casper_message::DeployData::from_proto(
                request.into_inner(),
            ) {
                Ok(signed) => signed,
                Err(err_msg) => {
                    let error = Self::create_service_error(err_msg);
                    return Self::create_error_deploy_response(error);
                }
            };

        match BlockAPI::deploy(
            &self.engine_cell,
            signed_deploy,
            &self.trigger_propose_f,
            self.min_phlo_price,
            self.is_node_read_only,
            &self.shard_id,
        )
        .await
        {
            Ok(result) => Self::create_success_deploy_response(result),
            Err(e) => {
                error!("Deploy service method error do_deploy");
                Self::create_error_deploy_response(e.into_service_error())
            }
        }
    }

    /// Get a block by hash
    async fn get_block(
        &self,
        request: tonic::Request<BlockQuery>,
    ) -> Result<tonic::Response<BlockResponse>, tonic::Status> {
        match BlockAPI::get_block(&self.engine_cell, &request.into_inner().hash).await {
            Ok(block_info) => Self::create_success_block_response(block_info),
            Err(e) => {
                error!("Deploy service method error get_block");
                Self::create_error_block_response(e.into_service_error())
            }
        }
    }

    /// Visualize the DAG
    async fn visualize_dag(
        &self,
        request: tonic::Request<VisualizeDagQuery>,
    ) -> Result<tonic::Response<Self::visualizeDagStream>, tonic::Status> {
        let request = request.into_inner();

        let depth = if request.depth <= 0 {
            self.api_max_blocks_limit
        } else {
            request.depth
        };

        let config = GraphConfig {
            show_justification_lines: request.show_justification_lines,
        };
        let start_block_number = request.start_block_number;
        let (tx, rx) = tokio::sync::mpsc::channel(128);
        let engine_cell = self.engine_cell.clone();
        let key_value_block_store = self.key_value_block_store.clone();

        tokio::spawn(async move {
            let (sender, receiver) = tokio::sync::oneshot::channel();
            let ser: Arc<dyn GraphSerializer> = Arc::new(ListSerializer::new(sender));

            match BlockAPI::visualize_dag(
                &engine_cell,
                depth,
                start_block_number,
                |ts, lfb| {
                    let ser = ser.clone();
                    let key_value_block_store = key_value_block_store.clone();
                    async move {
                        let _: graphz::Graphz = GraphzGenerator::dag_as_cluster(
                            ts,
                            lfb,
                            config,
                            ser,
                            &key_value_block_store,
                        )
                        .await?;
                        Ok(())
                    }
                },
                receiver,
            )
            .await
            {
                Ok(content) => {
                    for content_string in content {
                        let response = VisualizeBlocksResponse {
                            message: Some(
                                models::casper::v1::visualize_blocks_response::Message::Content(
                                    content_string,
                                ),
                            ),
                        };
                        if tx.send(Ok(response)).await.is_err() {
                            break;
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(tonic::Status::internal(e.to_string()))).await;
                }
            }
        });

        Ok(tonic::Response::new(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
    }

    /// Get machine verifiable DAG
    async fn machine_verifiable_dag(
        &self,
        request: tonic::Request<MachineVerifyQuery>,
    ) -> Result<tonic::Response<MachineVerifyResponse>, tonic::Status> {
        let _request = request.into_inner(); // maybe this parameter is should be removed in future, left for compatibility with Scala version
        match BlockAPI::machine_verifiable_dag(
            &self.engine_cell,
            self.api_max_blocks_limit,
            self.api_max_blocks_limit,
        )
        .await
        {
            Ok(content) => Ok(tonic::Response::new(MachineVerifyResponse {
                message: Some(
                    models::casper::v1::machine_verify_response::Message::Content(content),
                ),
            })),
            Err(e) => {
                error!("Deploy service method error machine_verifiable_dag");
                Ok(tonic::Response::new(MachineVerifyResponse {
                    message: Some(models::casper::v1::machine_verify_response::Message::Error(
                        e.into_service_error(),
                    )),
                }))
            }
        }
    }

    /// Show main chain
    async fn show_main_chain(
        &self,
        request: tonic::Request<BlocksQuery>,
    ) -> Result<tonic::Response<Self::showMainChainStream>, tonic::Status> {
        let request = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(128);
        let engine_cell = self.engine_cell.clone();

        let api_max_blocks_limit = self.api_max_blocks_limit;
        tokio::spawn(async move {
            let blocks =
                BlockAPI::show_main_chain(&engine_cell, request.depth, api_max_blocks_limit).await;

            for block_info in blocks {
                let response = BlockInfoResponse {
                    message: Some(models::casper::v1::block_info_response::Message::BlockInfo(
                        block_info,
                    )),
                };
                if tx.send(Ok(response)).await.is_err() {
                    break;
                }
            }
        });

        Ok(tonic::Response::new(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
    }

    /// Get blocks
    async fn get_blocks(
        &self,
        request: tonic::Request<BlocksQuery>,
    ) -> Result<tonic::Response<Self::getBlocksStream>, tonic::Status> {
        let request = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(128);
        let engine_cell = self.engine_cell.clone();
        let api_max_blocks_limit = self.api_max_blocks_limit;

        tokio::spawn(async move {
            match BlockAPI::get_blocks(&engine_cell, request.depth, api_max_blocks_limit).await {
                Ok(blocks) => {
                    for block_info in blocks {
                        let response = BlockInfoResponse {
                            message: Some(
                                models::casper::v1::block_info_response::Message::BlockInfo(
                                    block_info,
                                ),
                            ),
                        };
                        if tx.send(Ok(response)).await.is_err() {
                            break;
                        }
                    }
                }
                Err(e) => {
                    error!("Deploy service method error get_blocks");
                    let _ = tx.send(Err(tonic::Status::internal(e.to_string()))).await;
                }
            }
        });

        Ok(tonic::Response::new(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
    }

    /// Listen for data at name
    async fn listen_for_data_at_name(
        &self,
        request: tonic::Request<DataAtNameQuery>,
    ) -> Result<tonic::Response<ListeningNameDataResponse>, tonic::Status> {
        let request = request.into_inner();
        match BlockAPI::get_listening_name_data_response(
            &self.engine_cell,
            request.depth,
            request.name.unwrap_or_default(),
            self.api_max_blocks_limit,
        )
        .await
        {
            Ok((block_info, length)) => {
                let payload = models::casper::v1::ListeningNameDataPayload { block_info, length };
                Ok(tonic::Response::new(ListeningNameDataResponse {
                    message: Some(
                        models::casper::v1::listening_name_data_response::Message::Payload(payload),
                    ),
                }))
            }
            Err(e) => {
                error!("Deploy service method error listen_for_data_at_name");
                Ok(tonic::Response::new(ListeningNameDataResponse {
                    message: Some(
                        models::casper::v1::listening_name_data_response::Message::Error(
                            e.into_service_error(),
                        ),
                    ),
                }))
            }
        }
    }

    /// Get data at name
    async fn get_data_at_name(
        &self,
        request: tonic::Request<DataAtNameByBlockQuery>,
    ) -> Result<tonic::Response<RhoDataResponse>, tonic::Status> {
        let request = request.into_inner();
        match BlockAPI::get_data_at_par(
            &self.engine_cell,
            &request.par.unwrap_or_default(),
            request.block_hash,
            request.use_pre_state_hash,
        )
        .await
        {
            Ok((par, block)) => {
                let payload = models::casper::v1::RhoDataPayload {
                    par,
                    block: Some(block),
                };
                Ok(tonic::Response::new(RhoDataResponse {
                    message: Some(models::casper::v1::rho_data_response::Message::Payload(
                        payload,
                    )),
                }))
            }
            Err(e) => {
                error!("Deploy service method error get_data_at_name");
                Ok(tonic::Response::new(RhoDataResponse {
                    message: Some(models::casper::v1::rho_data_response::Message::Error(
                        e.into_service_error(),
                    )),
                }))
            }
        }
    }

    /// Listen for continuation at name
    async fn listen_for_continuation_at_name(
        &self,
        request: tonic::Request<ContinuationAtNameQuery>,
    ) -> Result<tonic::Response<ContinuationAtNameResponse>, tonic::Status> {
        let request = request.into_inner();
        match BlockAPI::get_listening_name_continuation_response(
            &self.engine_cell,
            request.depth,
            &request.names,
            self.api_max_blocks_limit,
        )
        .await
        {
            Ok((block_results, length)) => {
                let payload = models::casper::v1::ContinuationAtNamePayload {
                    block_results,
                    length,
                };
                Ok(tonic::Response::new(ContinuationAtNameResponse {
                    message: Some(
                        models::casper::v1::continuation_at_name_response::Message::Payload(
                            payload,
                        ),
                    ),
                }))
            }
            Err(e) => {
                error!("Deploy service method error listen_for_continuation_at_name");
                Ok(tonic::Response::new(ContinuationAtNameResponse {
                    message: Some(
                        models::casper::v1::continuation_at_name_response::Message::Error(
                            e.into_service_error(),
                        ),
                    ),
                }))
            }
        }
    }

    /// Find deploy
    async fn find_deploy(
        &self,
        request: tonic::Request<FindDeployQuery>,
    ) -> Result<tonic::Response<FindDeployResponse>, tonic::Status> {
        let request = request.into_inner();
        match BlockAPI::find_deploy(&self.engine_cell, &request.deploy_id.to_vec()).await {
            Ok(block_info) => Ok(tonic::Response::new(FindDeployResponse {
                message: Some(
                    models::casper::v1::find_deploy_response::Message::BlockInfo(block_info),
                ),
            })),
            Err(e) => {
                error!("Deploy service method error find_deploy");
                Ok(tonic::Response::new(FindDeployResponse {
                    message: Some(models::casper::v1::find_deploy_response::Message::Error(
                        e.into_service_error(),
                    )),
                }))
            }
        }
    }

    /// Preview private names
    async fn preview_private_names(
        &self,
        request: tonic::Request<PrivateNamePreviewQuery>,
    ) -> Result<tonic::Response<PrivateNamePreviewResponse>, tonic::Status> {
        let request = request.into_inner();
        match BlockAPI::preview_private_names(
            &request.user.to_vec(),
            request.timestamp,
            request.name_qty,
        ) {
            Ok(ids) => {
                let ids_bytes: Vec<prost::bytes::Bytes> =
                    ids.into_iter().map(|id| id.into()).collect();
                let payload = models::casper::v1::PrivateNamePreviewPayload { ids: ids_bytes };
                Ok(tonic::Response::new(PrivateNamePreviewResponse {
                    message: Some(
                        models::casper::v1::private_name_preview_response::Message::Payload(
                            payload,
                        ),
                    ),
                }))
            }
            Err(e) => {
                error!("Deploy service method error preview_private_names");
                Ok(tonic::Response::new(PrivateNamePreviewResponse {
                    message: Some(
                        models::casper::v1::private_name_preview_response::Message::Error(
                            e.into_service_error(),
                        ),
                    ),
                }))
            }
        }
    }

    /// Get last finalized block
    async fn last_finalized_block(
        &self,
        request: tonic::Request<LastFinalizedBlockQuery>,
    ) -> Result<tonic::Response<LastFinalizedBlockResponse>, tonic::Status> {
        let _request = request.into_inner(); // maybe this parameter is should be removed in future, left for compatibility with Scala version
        match BlockAPI::last_finalized_block(&self.engine_cell).await {
            Ok(block_info) => Ok(tonic::Response::new(LastFinalizedBlockResponse {
                message: Some(
                    models::casper::v1::last_finalized_block_response::Message::BlockInfo(
                        block_info,
                    ),
                ),
            })),
            Err(e) => {
                error!("Deploy service method error last_finalized_block");
                Ok(tonic::Response::new(LastFinalizedBlockResponse {
                    message: Some(
                        models::casper::v1::last_finalized_block_response::Message::Error(
                            e.into_service_error(),
                        ),
                    ),
                }))
            }
        }
    }

    /// Check if block is finalized
    async fn is_finalized(
        &self,
        request: tonic::Request<IsFinalizedQuery>,
    ) -> Result<tonic::Response<IsFinalizedResponse>, tonic::Status> {
        let request = request.into_inner();
        match BlockAPI::is_finalized(&self.engine_cell, &request.hash).await {
            Ok(is_finalized) => Ok(tonic::Response::new(IsFinalizedResponse {
                message: Some(
                    models::casper::v1::is_finalized_response::Message::IsFinalized(is_finalized),
                ),
            })),
            Err(e) => {
                error!("Deploy service method error is_finalized");
                Ok(tonic::Response::new(IsFinalizedResponse {
                    message: Some(models::casper::v1::is_finalized_response::Message::Error(
                        e.into_service_error(),
                    )),
                }))
            }
        }
    }

    /// Get bond status
    async fn bond_status(
        &self,
        request: tonic::Request<BondStatusQuery>,
    ) -> Result<tonic::Response<BondStatusResponse>, tonic::Status> {
        let request = request.into_inner();
        match BlockAPI::bond_status(&self.engine_cell, &request.public_key.to_vec()).await {
            Ok(is_bonded) => Ok(tonic::Response::new(BondStatusResponse {
                message: Some(models::casper::v1::bond_status_response::Message::IsBonded(
                    is_bonded,
                )),
            })),
            Err(e) => {
                error!("Deploy service method error bond_status");
                Ok(tonic::Response::new(BondStatusResponse {
                    message: Some(models::casper::v1::bond_status_response::Message::Error(
                        e.into_service_error(),
                    )),
                }))
            }
        }
    }

    /// Exploratory deploy
    async fn exploratory_deploy(
        &self,
        request: tonic::Request<ExploratoryDeployQuery>,
    ) -> Result<tonic::Response<ExploratoryDeployResponse>, tonic::Status> {
        let request = request.into_inner();
        let block_hash = if request.block_hash.is_empty() {
            None
        } else {
            Some(request.block_hash.clone())
        };

        match BlockAPI::exploratory_deploy(
            &self.engine_cell,
            request.term,
            block_hash,
            request.use_pre_state_hash,
            self.dev_mode,
        )
        .await
        {
            Ok((par, block)) => {
                let data_with_block_info = models::casper::DataWithBlockInfo {
                    post_block_data: par,
                    block: Some(block),
                };
                Ok(tonic::Response::new(ExploratoryDeployResponse {
                    message: Some(
                        models::casper::v1::exploratory_deploy_response::Message::Result(
                            data_with_block_info,
                        ),
                    ),
                }))
            }
            Err(e) => {
                error!("Deploy service method error exploratory_deploy");
                Ok(tonic::Response::new(ExploratoryDeployResponse {
                    message: Some(
                        models::casper::v1::exploratory_deploy_response::Message::Error(
                            e.into_service_error(),
                        ),
                    ),
                }))
            }
        }
    }

    /// Get event by hash
    async fn get_event_by_hash(
        &self,
        request: tonic::Request<ReportQuery>,
    ) -> Result<tonic::Response<EventInfoResponse>, tonic::Status> {
        let request = request.into_inner();

        if let Err(_) = hex::decode(&request.hash) {
            let error = Self::create_service_error(format!(
                "Request hash: {} is not valid hex string",
                request.hash
            ));
            return Ok(tonic::Response::new(EventInfoResponse {
                message: Some(models::casper::v1::event_info_response::Message::Error(
                    error,
                )),
            }));
        }

        match self
            .block_report_api
            .block_report(
                prost::bytes::Bytes::from(request.hash),
                request.force_replay,
            )
            .await
        {
            Ok(block_event_info) => Ok(tonic::Response::new(EventInfoResponse {
                message: Some(models::casper::v1::event_info_response::Message::Result(
                    block_event_info,
                )),
            })),
            Err(e) => {
                error!("Deploy service method error get_event_by_hash");
                Ok(tonic::Response::new(EventInfoResponse {
                    message: Some(models::casper::v1::event_info_response::Message::Error(
                        e.into_service_error(),
                    )),
                }))
            }
        }
    }

    /// Get blocks by heights
    async fn get_blocks_by_heights(
        &self,
        request: tonic::Request<BlocksQueryByHeight>,
    ) -> Result<tonic::Response<Self::getBlocksByHeightsStream>, tonic::Status> {
        let request = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(128);
        let engine_cell = self.engine_cell.clone();
        let api_max_blocks_limit = self.api_max_blocks_limit;

        tokio::spawn(async move {
            match BlockAPI::get_blocks_by_heights(
                &engine_cell,
                request.start_block_number,
                request.end_block_number,
                api_max_blocks_limit,
            )
            .await
            {
                Ok(blocks) => {
                    for block_info in blocks {
                        let response = BlockInfoResponse {
                            message: Some(
                                models::casper::v1::block_info_response::Message::BlockInfo(
                                    block_info,
                                ),
                            ),
                        };
                        if tx.send(Ok(response)).await.is_err() {
                            break;
                        }
                    }
                }
                Err(e) => {
                    error!("Deploy service method error get_blocks_by_heights");
                    let _ = tx.send(Err(tonic::Status::internal(e.to_string()))).await;
                }
            }
        });

        Ok(tonic::Response::new(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
    }

    /// Get status
    async fn status(
        &self,
        _request: tonic::Request<()>,
    ) -> Result<tonic::Response<StatusResponse>, tonic::Status> {
        let address = self.rp_conf.local.to_address();

        let peers = match self.connections_cell.read() {
            Ok(connections) => connections.len() as i32,
            Err(e) => {
                error!("Deploy service method error status");
                return Err(tonic::Status::internal(e.to_string()));
            }
        };

        let nodes = match self.node_discovery.peers() {
            Ok(peers) => peers.len() as i32,
            Err(e) => {
                error!("Deploy service method error status");
                return Err(tonic::Status::internal(e.to_string()));
            }
        };

        let status = Status {
            version: Some(VersionInfo {
                api: "1".to_string(),
                node: get_version_info_str(),
            }),
            address,
            network_id: self.network_id.clone(),
            shard_id: self.shard_id.clone(),
            peers,
            nodes,
            min_phlo_price: self.min_phlo_price,
        };

        Ok(tonic::Response::new(StatusResponse {
            message: Some(models::casper::v1::status_response::Message::Status(status)),
        }))
    }
}
