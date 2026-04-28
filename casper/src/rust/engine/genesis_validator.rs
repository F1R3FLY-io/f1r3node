// See casper/src/main/scala/coop/rchain/casper/engine/GenesisValidator.scala

use async_trait::async_trait;
use dashmap::DashSet;
use std::collections::{HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;

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
use crate::rust::util::rholang::runtime_manager::RuntimeManager;
use crate::rust::validator_identity::ValidatorIdentity;

pub struct GenesisValidator<T: TransportLayer + Send + Sync + Clone + 'static> {
    block_processing_queue_tx:
        mpsc::Sender<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>,
    blocks_in_processing: Arc<DashSet<BlockHash>>,
    casper_shard_conf: CasperShardConf,
    validator_id: ValidatorIdentity,
    block_approver: BlockApproverProtocol<T>,

    transport_layer: Arc<T>,
    rp_conf_ask: RPConf,
    connections_cell: ConnectionsCell,
    last_approved_block: Arc<Mutex<Option<ApprovedBlock>>>,
    event_publisher: F1r3flyEvents,
    block_retriever: BlockRetriever<T>,
    engine_cell: Arc<EngineCell>,

    block_store: KeyValueBlockStore,
    block_dag_storage: BlockDagKeyValueStorage,
    deploy_storage: KeyValueDeployStorage,
    casper_buffer_storage: CasperBufferKeyValueStorage,
    rspace_state_manager: RSpaceStateManager,

    runtime_manager: Arc<tokio::sync::Mutex<RuntimeManager>>,
    estimator: Estimator,

    // Bounded set of seen UnapprovedBlock candidates to avoid unbounded memory growth.
    seen_candidates: Arc<Mutex<SeenCandidates>>,
    /// Shared reference to heartbeat signal for triggering immediate wake on deploy
    heartbeat_signal_ref: crate::rust::heartbeat_signal::HeartbeatSignalRef,
}

struct SeenCandidates {
    set: HashSet<BlockHash>,
    fifo: VecDeque<BlockHash>,
    max_entries: usize,
}

impl SeenCandidates {
    fn new(max_entries: usize) -> Self {
        Self {
            set: HashSet::new(),
            fifo: VecDeque::new(),
            max_entries,
        }
    }

    fn contains(&self, hash: &BlockHash) -> bool {
        self.set.contains(hash)
    }

    fn insert(&mut self, hash: BlockHash) {
        if !self.set.insert(hash.clone()) {
            return;
        }
        self.fifo.push_back(hash);
        while self.set.len() > self.max_entries {
            if let Some(oldest) = self.fifo.pop_front() {
                self.set.remove(&oldest);
            } else {
                break;
            }
        }
    }
}

fn genesis_seen_candidates_max_entries() -> usize {
    4_096
}

impl<T: TransportLayer + Send + Sync + Clone + 'static> GenesisValidator<T> {
    /// Scala equivalent: Constructor for `GenesisValidator` class
    ///
    /// NOTE: Parameter types adapted to use Arc<Mutex<Option<T>>> for storage types
    /// to enable cloning from TestFixture and proper ownership transfer to Initializing.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        block_processing_queue_tx: mpsc::Sender<(
            Arc<dyn MultiParentCasper + Send + Sync>,
            BlockMessage,
        )>,
        blocks_in_processing: Arc<DashSet<BlockHash>>,
        casper_shard_conf: CasperShardConf,
        validator_id: ValidatorIdentity,
        block_approver: BlockApproverProtocol<T>,
        transport_layer: Arc<T>,
        rp_conf_ask: RPConf,
        connections_cell: ConnectionsCell,
        last_approved_block: Arc<Mutex<Option<ApprovedBlock>>>,
        event_publisher: F1r3flyEvents,
        block_retriever: BlockRetriever<T>,
        engine_cell: Arc<EngineCell>,
        block_store: KeyValueBlockStore,
        block_dag_storage: BlockDagKeyValueStorage,
        deploy_storage: KeyValueDeployStorage,
        casper_buffer_storage: CasperBufferKeyValueStorage,
        rspace_state_manager: RSpaceStateManager,
        runtime_manager: Arc<tokio::sync::Mutex<RuntimeManager>>,
        estimator: Estimator,
        heartbeat_signal_ref: crate::rust::heartbeat_signal::HeartbeatSignalRef,
    ) -> Self {
        Self {
            block_processing_queue_tx,
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
            block_store,
            block_dag_storage,
            deploy_storage,
            casper_buffer_storage,
            rspace_state_manager,
            runtime_manager,
            estimator,
            seen_candidates: Arc::new(Mutex::new(SeenCandidates::new(
                genesis_seen_candidates_max_entries(),
            ))),
            heartbeat_signal_ref,
        }
    }

    fn is_repeated(&self, hash: &BlockHash) -> bool {
        self.seen_candidates.lock().unwrap().contains(hash)
    }

    fn ack(&self, hash: BlockHash) {
        self.seen_candidates.lock().unwrap().insert(hash);
    }

    /// Handle an ApprovedBlock that arrives while we're still in GenesisValidator state.
    ///
    /// Race scenario: boot broadcasts the UnapprovedBlock to its current connections,
    /// reaches `required_signatures` from peers that connected first, then transitions
    /// to Running and broadcasts the ApprovedBlock. A genesis validator that joined
    /// boot's connections AFTER the UnapprovedBlock broadcasts but BEFORE the
    /// ApprovedBlock broadcast has never seen the UnapprovedBlock — there is no path
    /// out of `GenesisValidator` state via the signing flow because there is nothing
    /// to sign.
    ///
    /// Recovery: transition to `Initializing`. `Initializing::init` proactively sends
    /// `ApprovedBlockRequest` to bootstrap, and its `handle` accepts the response,
    /// validates it, and transitions to `Running` — the same path a late-joining
    /// non-genesis node already takes.
    ///
    /// No `seen_candidates` dedup here: that set is for `UnapprovedBlock` repeats and
    /// would conflate "we've seen the candidate's content" with "we've already
    /// transitioned for this ApprovedBlock". A successful `transition_to_initializing`
    /// replaces the engine, so subsequent `ApprovedBlock` messages route to
    /// `Initializing::handle` rather than back here. Concurrent duplicates during the
    /// brief transition window are safe — the engine_cell write serializes them and
    /// `Initializing::init`'s `ApprovedBlockRequest` is idempotent at bootstrap.
    async fn handle_approved_block_late(
        &self,
        approved_block: ApprovedBlock,
    ) -> Result<(), CasperError> {
        let hash = approved_block.candidate.block.block_hash.clone();
        tracing::info!(
            "Received ApprovedBlock {} while in GenesisValidator state — transitioning to Initializing for late-joiner recovery",
            PrettyPrinter::build_string_no_limit(&hash)
        );

        let init = Arc::new(|| {
            Box::pin(async { Ok(()) })
                as Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>>
        });
        let validator_id_opt = Some(self.validator_id.clone());

        transition_to_initializing(
            &self.block_processing_queue_tx,
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
            self.event_publisher.clone(),
            self.block_retriever.clone(),
            &self.engine_cell,
            &self.runtime_manager,
            &self.estimator,
            &self.heartbeat_signal_ref,
        )
        .await
    }

    async fn handle_unapproved_block(
        &self,
        peer: PeerNode,
        ub: UnapprovedBlock,
    ) -> Result<(), CasperError> {
        let hash = ub.candidate.block.block_hash.clone();
        if self.is_repeated(&hash) {
            tracing::warn!(
                "UnapprovedBlock {} is already being verified. Dropping repeated message.",
                PrettyPrinter::build_string_no_limit(&hash)
            );
            return Ok(());
        }

        self.ack(hash);

        {
            let mut runtime_manager_guard = self.runtime_manager.lock().await;

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
        let init = Arc::new(|| {
            Box::pin(async { Ok(()) })
                as Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>>
        });
        let validator_id_opt = Some(self.validator_id.clone());

        transition_to_initializing(
            &self.block_processing_queue_tx,
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
            self.event_publisher.clone(),
            self.block_retriever.clone(),
            &self.engine_cell,
            &self.runtime_manager,
            &self.estimator,
            &self.heartbeat_signal_ref,
        )
        .await
    }
}

#[async_trait]
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
            CasperMessage::ApprovedBlock(approved_block) => {
                self.handle_approved_block_late(approved_block).await
            }
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

    fn with_casper(&self) -> Option<Arc<dyn crate::rust::casper::MultiParentCasper + Send + Sync>> {
        None
    }
}
