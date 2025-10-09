// See casper/src/main/scala/coop/rchain/casper/engine/Engine.scala

use async_trait::async_trait;
use block_storage::rust::casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage;
use block_storage::rust::dag::block_dag_key_value_storage::BlockDagKeyValueStorage;
use block_storage::rust::deploy::key_value_deploy_storage::KeyValueDeployStorage;
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use comm::rust::peer_node::PeerNode;
use comm::rust::rp::connect::ConnectionsCell;
use comm::rust::rp::rp_conf::RPConf;
use comm::rust::transport::transport_layer::{Blob, TransportLayer};
use models::rust::block_hash::BlockHash;
use models::rust::casper::pretty_printer::PrettyPrinter;
use models::rust::casper::protocol::casper_message::{
    ApprovedBlock, BlockMessage, CasperMessage, NoApprovedBlockAvailable, StoreItemsMessage,
};
use models::rust::casper::protocol::packet_type_tag::ToPacket;
use shared::rust::shared::f1r3fly_event::F1r3flyEvent;
use shared::rust::shared::f1r3fly_events::F1r3flyEvents;
use std::collections::{HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

use crate::rust::casper::CasperShardConf;
use crate::rust::casper::MultiParentCasper;
use crate::rust::engine::block_retriever::BlockRetriever;
use crate::rust::engine::engine_cell::EngineCell;
use crate::rust::engine::running::Running;
use crate::rust::errors::CasperError;
use crate::rust::estimator::Estimator;
use crate::rust::util::rholang::runtime_manager::RuntimeManager;
use crate::rust::validator_identity::ValidatorIdentity;
use rspace_plus_plus::rspace::state::rspace_state_manager::RSpaceStateManager;

/// Object-safe Engine trait that matches Scala Engine[F] behavior.
/// Note: we expose `with_casper() -> Option<&MultiParentCasper>` as an accessor,
/// and provide Scala-like `with_casper(f, default)` via `EngineDynExt`.
#[async_trait(?Send)]
pub trait Engine: Send + Sync {
    async fn init(&self) -> Result<(), CasperError>;

    async fn handle(&self, peer: PeerNode, msg: CasperMessage) -> Result<(), CasperError>;

    /// Returns the casper instance if this engine wraps one.
    /// Used by `EngineDynExt::with_casper(...)` to emulate Scala semantics.
    fn with_casper(&self) -> Option<&dyn MultiParentCasper>;
}

/// Trait for engines that provide withCasper functionality
/// This matches the Scala Engine[F] withCasper method behavior
#[async_trait(?Send)]
pub trait EngineDynExt {
    async fn with_casper<A, F>(
        &self,
        f: F,
        default: Result<A, CasperError>,
    ) -> Result<A, CasperError>
    where
        for<'a> F: FnOnce(
            &'a dyn MultiParentCasper,
        ) -> Pin<Box<dyn Future<Output = Result<A, CasperError>> + 'a>>,
        A: Sized;
}

#[async_trait(?Send)]
impl<T: Engine + ?Sized> EngineDynExt for T {
    async fn with_casper<A, F>(
        &self,
        f: F,
        default: Result<A, CasperError>,
    ) -> Result<A, CasperError>
    where
        for<'a> F: FnOnce(
            &'a dyn MultiParentCasper,
        ) -> Pin<Box<dyn Future<Output = Result<A, CasperError>> + 'a>>,
        A: Sized,
    {
        match self.with_casper() {
            Some(casper) => f(casper).await,
            None => default,
        }
    }
}

pub fn noop() -> impl Engine {
    #[derive(Clone)]
    struct NoopEngine;

    #[async_trait(?Send)]
    impl Engine for NoopEngine {
        async fn init(&self) -> Result<(), CasperError> {
            Ok(())
        }

        async fn handle(&self, _peer: PeerNode, _msg: CasperMessage) -> Result<(), CasperError> {
            Ok(())
        }

        fn with_casper(&self) -> Option<&dyn MultiParentCasper> {
            None
        }
    }

    NoopEngine
}

pub fn log_no_approved_block_available(identifier: &str) {
    log::info!(
        "No approved block available on node {}. Will request again in 10 seconds.",
        identifier
    )
}

/*
 * Note the ordering of the insertions is important.
 * We always want the block dag store to be a subset of the block store.
 */
pub fn insert_into_block_and_dag_store(
    block_store: &mut KeyValueBlockStore,
    block_dag_storage: &mut BlockDagKeyValueStorage,
    genesis: &BlockMessage,
    approved_block: ApprovedBlock,
) -> Result<(), CasperError> {
    block_store.put(genesis.block_hash.clone(), genesis)?;
    block_dag_storage.insert(genesis, false, true)?;
    block_store.put_approved_block(&approved_block)?;
    Ok(())
}

pub async fn send_no_approved_block_available(
    rp_conf_ask: &RPConf,
    transport_layer: &impl TransportLayer,
    identifier: &str,
    peer: PeerNode,
) -> Result<(), CasperError> {
    let local = rp_conf_ask.local.clone();
    // TODO: remove NoApprovedBlockAvailable.nodeIdentifier, use `sender` provided by TransportLayer
    let no_approved_block_available = NoApprovedBlockAvailable {
        node_identifier: local.to_string(),
        identifier: identifier.to_string(),
    }
    .to_proto();

    let msg = Blob {
        sender: local,
        packet: no_approved_block_available.mk_packet(),
    };

    transport_layer.stream(&peer, &msg).await?;
    Ok(())
}

// NOTE: Changed to use trait object (dyn MultiParentCasper) instead of generic T
// based on discussion with Steven for TestFixture compatibility
pub async fn transition_to_running<U: TransportLayer + Send + Sync + 'static>(
    block_processing_queue: Arc<
        Mutex<VecDeque<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>>,
    >,
    blocks_in_processing: Arc<Mutex<HashSet<BlockHash>>>,
    casper: Arc<dyn MultiParentCasper + Send + Sync>,
    approved_block: ApprovedBlock,
    the_init: Arc<
        dyn Fn() -> Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>> + Send + Sync,
    >,
    disable_state_exporter: bool,
    connections_cell: ConnectionsCell,
    transport: Arc<U>,
    conf: RPConf,
    block_retriever: Arc<BlockRetriever<U>>,
    engine_cell: &EngineCell,
    event_log: &F1r3flyEvents,
) -> Result<(), CasperError> {
    let approved_block_info =
        PrettyPrinter::build_string_block_message(&approved_block.candidate.block, true);

    log::info!(
        "Making a transition to Running state. Approved {}",
        approved_block_info
    );

    // Publish EnteredRunningState event
    let block_hash_string =
        PrettyPrinter::build_string_no_limit(&approved_block.candidate.block.block_hash);
    event_log
        .publish(F1r3flyEvent::entered_running_state(block_hash_string))
        .map_err(|e| {
            CasperError::Other(format!(
                "Failed to publish EnteredRunningState event: {}",
                e
            ))
        })?;

    let running = Running::new(
        block_processing_queue,
        blocks_in_processing,
        casper,
        approved_block,
        the_init,
        disable_state_exporter,
        connections_cell,
        transport,
        conf,
        block_retriever,
    );

    engine_cell.set(Arc::new(running)).await;

    Ok(())
}

// NOTE about Scala parity:
// In Scala `Engine.transitionToInitializing`, fs2 queues are created internally via
// `Queue.bounded[F, BlockMessage](50)` and `Queue.bounded[F, StoreItemsMessage](50)` and
// passed to `Initializing`. In Rust we return the senders of newly created channels to the
// caller and keep the receivers inside `Initializing`.
// Rationale:
// - Ownership/visibility: without a shared effect environment (like F[_]) external producers
//   (transport/tests) would have no handles to feed messages into the engine, causing hangs.
//   Returning senders ensures producers can enqueue LFS responses, mirroring Scala tests that
//   enqueue directly into queues.
// - Behavior equivalence: `Initializing` still consumes from these channels; Scala used bounded(50),
//   here we use unbounded for simplicity and low test traffic. If strict bounds are needed later,
//   we can switch to `mpsc::channel(50)` and still return the senders.
// NOTE: Parameter types adapted to match GenesisValidator changes (Arc wrappers, trait objects)
// based on discussion with Steven for TestFixture compatibility
pub async fn transition_to_initializing<U: TransportLayer + Send + Sync + Clone + 'static>(
    block_processing_queue: &Arc<
        Mutex<VecDeque<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>>,
    >,
    blocks_in_processing: &Arc<Mutex<HashSet<BlockHash>>>,
    casper_shard_conf: &CasperShardConf,
    validator_id: &Option<ValidatorIdentity>,
    init: Arc<
        dyn Fn() -> Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>> + Send + Sync,
    >,
    trim_state: bool,
    disable_state_exporter: bool,
    transport_layer: &Arc<U>,
    rp_conf_ask: &RPConf,
    connections_cell: &ConnectionsCell,
    last_approved_block: &Arc<Mutex<Option<ApprovedBlock>>>,
    block_store_arc: &Arc<Mutex<Option<KeyValueBlockStore>>>,
    block_dag_storage_arc: &Arc<Mutex<Option<BlockDagKeyValueStorage>>>,
    deploy_storage_arc: &Arc<Mutex<Option<KeyValueDeployStorage>>>,
    casper_buffer_storage_arc: &Arc<Mutex<Option<CasperBufferKeyValueStorage>>>,
    rspace_state_manager_arc: &Arc<Mutex<Option<RSpaceStateManager>>>,
    event_publisher: &Arc<F1r3flyEvents>,
    block_retriever: &Arc<BlockRetriever<U>>,
    engine_cell: &Arc<EngineCell>,
    runtime_manager_arc: &Arc<Mutex<RuntimeManager>>,
    estimator_arc: &Arc<Mutex<Option<Estimator>>>,
) -> Result<(), CasperError> {
    // Create channels and return senders so caller can feed LFS responses (Scala: expose queues)
    let (block_tx, block_rx) = mpsc::unbounded_channel::<BlockMessage>();
    let (tuple_tx, tuple_rx) = mpsc::unbounded_channel::<StoreItemsMessage>();

    // Take owned resources from shared Arcs for Initializing
    let block_store = block_store_arc
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| CasperError::RuntimeError("Block store not available".to_string()))?;
    let block_dag_storage = block_dag_storage_arc
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| CasperError::RuntimeError("BlockDag storage not available".to_string()))?;
    let deploy_storage = deploy_storage_arc
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| CasperError::RuntimeError("Deploy storage not available".to_string()))?;
    let casper_buffer_storage = casper_buffer_storage_arc
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| {
            CasperError::RuntimeError("Casper buffer storage not available".to_string())
        })?;
    let rspace_state_manager =
        rspace_state_manager_arc
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| {
                CasperError::RuntimeError("RSpace state manager not available".to_string())
            })?;

    // RuntimeManager is now Arc<Mutex<RuntimeManager>>, so we clone the Arc instead of taking
    let runtime_manager = runtime_manager_arc.clone();

    let estimator = estimator_arc
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| CasperError::RuntimeError("Estimator not available".to_string()))?;

    let initializing = crate::rust::engine::initializing::Initializing::new(
        (**transport_layer).clone(),
        rp_conf_ask.clone(),
        connections_cell.clone(),
        last_approved_block.clone(),
        block_store,
        block_dag_storage,
        deploy_storage,
        casper_buffer_storage,
        rspace_state_manager,
        block_processing_queue.clone(),
        blocks_in_processing.clone(),
        casper_shard_conf.clone(),
        validator_id.clone(),
        init,
        block_tx.clone(),
        block_rx,
        tuple_tx.clone(),
        tuple_rx,
        trim_state,
        disable_state_exporter,
        event_publisher.clone(),
        block_retriever.clone(),
        engine_cell.clone(),
        runtime_manager,
        estimator,
    );

    engine_cell.set(Arc::new(initializing)).await;

    Ok(())
}
