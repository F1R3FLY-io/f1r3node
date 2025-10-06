// See casper/src/main/scala/coop/rchain/casper/engine/GenesisValidator.scala

use async_trait::async_trait;
use std::collections::{HashMap, HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use block_storage::rust::casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage;
use block_storage::rust::dag::block_dag_key_value_storage::BlockDagKeyValueStorage;
use block_storage::rust::deploy::key_value_deploy_storage::KeyValueDeployStorage;
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use comm::rust::peer_node::PeerNode;
use comm::rust::rp::connect::ConnectionsCell;
use comm::rust::rp::rp_conf::RPConf;
use comm::rust::transport::transport_layer::TransportLayer;
use models::rust::block_hash::BlockHash;
use models::rust::casper::pretty_printer::PrettyPrinter;
use models::rust::casper::protocol::casper_message::{
    ApprovedBlock, ApprovedBlockRequest, BlockMessage, CasperMessage, NoApprovedBlockAvailable,
    UnapprovedBlock,
};
use rspace_plus_plus::rspace::state::rspace_state_manager::RSpaceStateManager;
use shared::rust::shared::f1r3fly_events::F1r3flyEvents;

use crate::rust::casper::{CasperShardConf, MultiParentCasper};
use crate::rust::engine::block_approver_protocol::BlockApproverProtocol;
use crate::rust::engine::block_retriever::BlockRetriever;
use crate::rust::engine::engine::{
    log_no_approved_block_available, send_no_approved_block_available, transition_to_initializing,
    Engine,
};
use crate::rust::engine::engine_cell::EngineCell;
use crate::rust::errors::CasperError;
use crate::rust::estimator::Estimator;
use crate::rust::multi_parent_casper_impl::MultiParentCasperImpl;
use crate::rust::util::rholang::runtime_manager::RuntimeManager;
use crate::rust::validator_identity::ValidatorIdentity;

pub struct GenesisValidator<T: TransportLayer + Send + Sync + Clone + 'static> {
    block_processing_queue:
        Arc<Mutex<VecDeque<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>>>,
    blocks_in_processing: Arc<Mutex<HashSet<BlockHash>>>,
    casper_shard_conf: CasperShardConf,
    validator_id: ValidatorIdentity,
    block_approver: BlockApproverProtocol<T>,

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

    // Scala equivalent: `private val seenCandidates = Cell.unsafe[F, Map[BlockHash, Boolean]](Map.empty)`
    // Used by isRepeated() and ack() methods to track processed UnapprovedBlock candidates
    seen_candidates: Arc<Mutex<HashMap<BlockHash, bool>>>,
}

impl<T: TransportLayer + Send + Sync + Clone + 'static> GenesisValidator<T> {
    /// Scala equivalent: Constructor for `GenesisValidator` class
    ///
    /// NOTE: Parameter types adapted to use Arc<Mutex<Option<T>>> for storage types
    /// to enable cloning from TestFixture and proper ownership transfer to Initializing.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        block_processing_queue: Arc<
            Mutex<VecDeque<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>>,
        >,
        blocks_in_processing: Arc<Mutex<HashSet<BlockHash>>>,
        casper_shard_conf: CasperShardConf,
        validator_id: ValidatorIdentity,
        block_approver: BlockApproverProtocol<T>,
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
    ) -> Self {
        Self {
            block_processing_queue,
            blocks_in_processing,
            casper_shard_conf,
            validator_id,
            block_approver,
            transport_layer,
            rp_conf_ask,
            connections_cell,
            last_approved_block,
            event_publisher,
            block_retriever,
            engine_cell,
            // Storage already wrapped in Arc<Mutex<Option>> by caller
            block_store,
            block_dag_storage,
            deploy_storage,
            casper_buffer_storage,
            rspace_state_manager,
            runtime_manager,
            estimator,
            // Scala equivalent: `private val seenCandidates = Cell.unsafe[F, Map[BlockHash, Boolean]](Map.empty)`
            seen_candidates: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn is_repeated(&self, hash: &BlockHash) -> bool {
        self.seen_candidates.lock().unwrap().contains_key(hash)
    }

    fn ack(&self, hash: BlockHash) {
        self.seen_candidates.lock().unwrap().insert(hash, true);
    }

    async fn handle_unapproved_block(
        &self,
        peer: PeerNode,
        ub: UnapprovedBlock,
    ) -> Result<(), CasperError> {
        let hash = ub.candidate.block.block_hash.clone();
        if self.is_repeated(&hash) {
            log::warn!(
                "UnapprovedBlock {} is already being verified. Dropping repeated message.",
                PrettyPrinter::build_string_no_limit(&hash)
            );
            return Ok(());
        }

        self.ack(hash);

        {
            let mut runtime_manager_guard = self.runtime_manager.lock().unwrap();

            self.block_approver
                .unapproved_block_packet_handler(
                    &mut *runtime_manager_guard,
                    &peer,
                    ub,
                    &self.casper_shard_conf.shard_name,
                )
                .await?;
        }

        // Scala: init = noop (empty F[Unit])
        let init = Arc::new(|| Box::pin(async { Ok(()) }) as Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>>);
        let validator_id_opt = Some(self.validator_id.clone());
        
        transition_to_initializing(
            &self.block_processing_queue,
            &self.blocks_in_processing,
            &self.casper_shard_conf,
            &validator_id_opt,
            init,
            true,
            false,
            &self.transport_layer,
            &self.rp_conf_ask,
            &self.connections_cell,
            &self.last_approved_block,
            &self.block_store,
            &self.block_dag_storage,
            &self.deploy_storage,
            &self.casper_buffer_storage,
            &self.rspace_state_manager,
            &self.event_publisher,
            &self.block_retriever,
            &self.engine_cell,
            &self.runtime_manager,
            &self.estimator,
        )
        .await
    }
}

#[async_trait(?Send)]
impl<T: TransportLayer + Send + Sync + Clone + 'static> Engine for GenesisValidator<T> {
    async fn init(&self) -> Result<(), CasperError> {
        Ok(())
    }

    /// Scala equivalent: `override def handle(peer: PeerNode, msg: CasperMessage): F[Unit]`
    async fn handle(&self, peer: PeerNode, msg: CasperMessage) -> Result<(), CasperError> {
        match msg {
            CasperMessage::ApprovedBlockRequest(ApprovedBlockRequest { identifier, .. }) => {
                send_no_approved_block_available(
                    &self.rp_conf_ask,
                    &*self.transport_layer,
                    &identifier,
                    peer,
                )
                .await
            }
            CasperMessage::UnapprovedBlock(ub) => self.handle_unapproved_block(peer, ub).await,
            CasperMessage::NoApprovedBlockAvailable(NoApprovedBlockAvailable {
                node_identifier,
                ..
            }) => {
                log_no_approved_block_available(&node_identifier);
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn with_casper(&self) -> Option<&dyn crate::rust::casper::MultiParentCasper> {
        None
    }

    fn clone_box(&self) -> Box<dyn Engine> {
        panic!("GenesisValidator engine is not designed to be cloned - it transitions to Initializing state")
    }
}
