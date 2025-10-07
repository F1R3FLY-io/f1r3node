// See casper/src/main/scala/coop/rchain/casper/engine/GenesisCeremonyMaster.scala

use async_trait::async_trait;
use block_storage::rust::casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage;
use block_storage::rust::dag::block_dag_key_value_storage::BlockDagKeyValueStorage;
use block_storage::rust::deploy::key_value_deploy_storage::KeyValueDeployStorage;
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use comm::rust::peer_node::PeerNode;
use comm::rust::rp::connect::ConnectionsCell;
use comm::rust::rp::rp_conf::RPConf;
use comm::rust::transport::transport_layer::TransportLayer;
use models::rust::block_hash::BlockHash;
use models::rust::casper::protocol::casper_message::{ApprovedBlock, BlockMessage, CasperMessage};
use rspace_plus_plus::rspace::state::rspace_state_manager::RSpaceStateManager;
use shared::rust::shared::f1r3fly_events::F1r3flyEvents;
use std::collections::{HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::sleep;

use crate::rust::casper::{hash_set_casper, CasperShardConf, MultiParentCasper};
use crate::rust::engine::approve_block_protocol::ApproveBlockProtocolImpl;
use crate::rust::engine::block_retriever::BlockRetriever;
use crate::rust::engine::engine::{
    insert_into_block_and_dag_store, log_no_approved_block_available,
    send_no_approved_block_available, transition_to_running, Engine,
};
use crate::rust::engine::engine_cell::EngineCell;
use crate::rust::errors::CasperError;
use crate::rust::estimator::Estimator;
use crate::rust::util::rholang::runtime_manager::RuntimeManager;
use crate::rust::validator_identity::ValidatorIdentity;

pub struct GenesisCeremonyMaster<T: TransportLayer + Send + Sync + Clone + 'static> {
    approve_protocol: Arc<ApproveBlockProtocolImpl<T>>,
    transport_layer: Arc<T>,
    rp_conf_ask: RPConf,
}

impl<T: TransportLayer + Send + Sync + Clone + 'static> GenesisCeremonyMaster<T> {
    pub fn new(approve_protocol: ApproveBlockProtocolImpl<T>) -> Self {
        // In Scala these come via implicit parameters
        let transport_layer = approve_protocol.transport().clone();
        let rp_conf_ask = approve_protocol
            .conf()
            .as_ref()
            .expect("RPConf required for GenesisCeremonyMaster")
            .as_ref()
            .clone();

        Self {
            approve_protocol: Arc::new(approve_protocol),
            transport_layer,
            rp_conf_ask,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn waiting_for_approved_block_loop(
        // Infrastructure dependencies (Scala implicit parameters)
        transport_layer: Arc<T>,
        rp_conf_ask: RPConf,
        connections_cell: ConnectionsCell,
        last_approved_block: Arc<Mutex<Option<ApprovedBlock>>>,
        event_publisher: Arc<F1r3flyEvents>,
        block_retriever: Arc<BlockRetriever<T>>,
        engine_cell: Arc<EngineCell>,
        block_store: Arc<Mutex<Option<KeyValueBlockStore>>>,
        block_dag_storage: Arc<Mutex<Option<BlockDagKeyValueStorage>>>,
        deploy_storage: Arc<Mutex<Option<KeyValueDeployStorage>>>,
        casper_buffer_storage: Arc<Mutex<Option<CasperBufferKeyValueStorage>>>,
        rspace_state_manager: Arc<Mutex<Option<RSpaceStateManager>>>,
        runtime_manager: Arc<Mutex<RuntimeManager>>,
        estimator: Arc<Mutex<Option<Estimator>>>,

        // Explicit parameters from Scala (in same order as Scala signature)
        block_processing_queue: Arc<
            Mutex<VecDeque<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>>,
        >,
        blocks_in_processing: Arc<Mutex<HashSet<BlockHash>>>,
        casper_shard_conf: CasperShardConf,
        validator_id: Option<ValidatorIdentity>,
        disable_state_exporter: bool,
    ) -> Result<(), CasperError> {
        sleep(Duration::from_secs(2)).await;

        let last_approved_block_opt = last_approved_block.lock().unwrap().clone();

        match last_approved_block_opt {
            None => {
                Box::pin(Self::waiting_for_approved_block_loop(
                    transport_layer,
                    rp_conf_ask,
                    connections_cell,
                    last_approved_block,
                    event_publisher,
                    block_retriever,
                    engine_cell,
                    block_store,
                    block_dag_storage,
                    deploy_storage,
                    casper_buffer_storage,
                    rspace_state_manager,
                    runtime_manager,
                    estimator,
                    block_processing_queue,
                    blocks_in_processing,
                    casper_shard_conf,
                    validator_id,
                    disable_state_exporter,
                ))
                .await
            }
            Some(approved_block) => {
                let ab = approved_block.candidate.block.clone();
                {
                    let mut store_guard = block_store.lock().unwrap();
                    let mut dag_guard = block_dag_storage.lock().unwrap();
                    let store = store_guard.as_mut().expect("Block store not available");
                    let dag = dag_guard.as_mut().expect("BlockDag storage not available");
                    insert_into_block_and_dag_store(store, dag, &ab, approved_block.clone())?;
                }

                let casper = Self::create_casper_from_storage(
                    &transport_layer,
                    &connections_cell,
                    &rp_conf_ask,
                    &event_publisher,
                    &runtime_manager,
                    &estimator,
                    &block_store,
                    &block_dag_storage,
                    &deploy_storage,
                    &casper_buffer_storage,
                    &rspace_state_manager,
                    validator_id.clone(),
                    &casper_shard_conf,
                    ab,
                )?;

                // Scala: Engine.transitionToRunning[F](..., init = ().pure[F], ...)
                let the_init = Arc::new(|| {
                    Box::pin(async { Ok(()) })
                        as Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>>
                });

                transition_to_running(
                    block_processing_queue.clone(),
                    blocks_in_processing.clone(),
                    Arc::new(casper),
                    approved_block.clone(),
                    the_init,
                    disable_state_exporter,
                    connections_cell.clone(),
                    transport_layer.clone(),
                    rp_conf_ask.clone(),
                    block_retriever.clone(),
                    &*engine_cell,
                    &event_publisher,
                )
                .await?;

                // Scala: CommUtil[F].sendForkChoiceTipRequest
                transport_layer
                    .send_fork_choice_tip_request(&connections_cell, &rp_conf_ask)
                    .await?;

                Ok(())
            }
        }
    }

    /// Helper function to create MultiParentCasper from storage components
    /// Same logic as CasperLaunchImpl::create_casper but as static function
    #[allow(clippy::too_many_arguments)]
    fn create_casper_from_storage(
        transport_layer: &Arc<T>,
        connections_cell: &ConnectionsCell,
        rp_conf_ask: &RPConf,
        event_publisher: &Arc<F1r3flyEvents>,
        runtime_manager: &Arc<Mutex<RuntimeManager>>,
        estimator: &Arc<Mutex<Option<Estimator>>>,
        block_store: &Arc<Mutex<Option<KeyValueBlockStore>>>,
        block_dag_storage: &Arc<Mutex<Option<BlockDagKeyValueStorage>>>,
        deploy_storage: &Arc<Mutex<Option<KeyValueDeployStorage>>>,
        casper_buffer_storage: &Arc<Mutex<Option<CasperBufferKeyValueStorage>>>,
        rspace_state_manager: &Arc<Mutex<Option<RSpaceStateManager>>>,
        validator_id: Option<ValidatorIdentity>,
        casper_shard_conf: &CasperShardConf,
        ab: BlockMessage,
    ) -> Result<crate::rust::multi_parent_casper_impl::MultiParentCasperImpl<T>, CasperError> {
        let block_retriever_for_casper = BlockRetriever::new(
            transport_layer.clone(),
            Arc::new(connections_cell.clone()),
            Arc::new(rp_conf_ask.clone()),
        );

        let events_for_casper = (**event_publisher).clone();
        let runtime_manager_for_casper = runtime_manager.clone();

        let estimator_for_casper = estimator
            .lock()
            .unwrap()
            .take()
            .expect("Estimator not available");

        let block_store_for_casper = block_store
            .lock()
            .unwrap()
            .as_ref()
            .expect("Block store not available")
            .clone();

        let block_dag_storage_for_casper = block_dag_storage
            .lock()
            .unwrap()
            .take()
            .expect("BlockDag storage not available");

        let deploy_storage_for_casper = deploy_storage
            .lock()
            .unwrap()
            .take()
            .expect("Deploy storage not available");

        let casper_buffer_storage_for_casper = casper_buffer_storage
            .lock()
            .unwrap()
            .take()
            .expect("Casper buffer storage not available");

        let rspace_state_manager_for_casper = rspace_state_manager
            .lock()
            .unwrap()
            .take()
            .expect("RSpace state manager not available");

        hash_set_casper(
            block_retriever_for_casper,
            events_for_casper,
            runtime_manager_for_casper,
            estimator_for_casper,
            block_store_for_casper,
            block_dag_storage_for_casper,
            deploy_storage_for_casper,
            casper_buffer_storage_for_casper,
            validator_id,
            casper_shard_conf.clone(),
            ab,
            rspace_state_manager_for_casper,
        )
    }
}

#[async_trait(?Send)]
impl<T: TransportLayer + Send + Sync + Clone + 'static> Engine for GenesisCeremonyMaster<T> {
    async fn init(&self) -> Result<(), CasperError> {
        self.approve_protocol.run().await
    }

    async fn handle(&self, peer: PeerNode, msg: CasperMessage) -> Result<(), CasperError> {
        match msg {
            CasperMessage::ApprovedBlockRequest(approved_block_request) => {
                send_no_approved_block_available(
                    &self.rp_conf_ask,
                    &*self.transport_layer,
                    &approved_block_request.identifier,
                    peer,
                )
                .await
            }
            CasperMessage::BlockApproval(block_approval) => {
                self.approve_protocol.add_approval(block_approval).await
            }
            CasperMessage::NoApprovedBlockAvailable(no_approved_block_available) => {
                log_no_approved_block_available(&no_approved_block_available.node_identifier);
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn with_casper(&self) -> Option<&dyn MultiParentCasper> {
        None
    }

    fn clone_box(&self) -> Box<dyn Engine> {
        panic!("GenesisCeremonyMaster engine is not designed to be cloned - it transitions to Running state")
    }
}
