// See node/src/main/scala/coop/rchain/node/runtime/Setup.scala

use block_storage::rust::{
    casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage,
    dag::block_dag_key_value_storage::BlockDagKeyValueStorage,
    deploy::key_value_deploy_storage::KeyValueDeployStorage,
    finality::LastFinalizedKeyValueStorage, key_value_block_store::KeyValueBlockStore,
};
use models::{
    rhoapi::Par,
    rust::{
        block_hash::BlockHash,
        casper::protocol::casper_message::{ApprovedBlock, BlockMessage},
    },
};
use std::{
    collections::{HashSet, VecDeque},
    sync::{Arc, Mutex},
};
use tokio::sync::{mpsc, oneshot, RwLock};
use tracing::info;

use casper::rust::{
    blocks::{
        block_processor::BlockProcessor,
        proposer::proposer::{ProductionProposer, ProposerResult},
    },
    casper::{Casper, MultiParentCasper},
    engine::{
        block_retriever::BlockRetriever, casper_launch::CasperLaunch, engine_cell::EngineCell,
    },
    errors::CasperError,
    estimator::Estimator,
    genesis::genesis::Genesis,
    reporting_casper,
    safety_oracle::CliqueOracleImpl,
    state::instances::ProposerState,
    storage::rnode_key_value_store_manager::new_key_value_store_manager,
    util::rholang::runtime_manager::RuntimeManager,
    validator_identity::ValidatorIdentity,
    ProposeFunction,
};
use comm::rust::{
    discovery::node_discovery::NodeDiscovery,
    p2p::packet_handler::PacketHandler,
    rp::{connect::ConnectionsCell, rp_conf::RPConf},
    transport::transport_layer::TransportLayer,
};
use crypto::rust::private_key::PrivateKey;
use rholang::rust::interpreter::{matcher::r#match::Matcher, rho_runtime};
use rspace_plus_plus::rspace::{
    shared::key_value_store_manager::KeyValueStoreManager,
    state::rspace_state_manager::RSpaceStateManager,
};
use shared::rust::shared::f1r3fly_events::F1r3flyEvents;

use crate::rust::{
    api::{admin_web_api::AdminWebApi, web_api::WebApi},
    configuration::NodeConf,
    runtime::{
        api_servers::APIServers,
        node_runtime::{CasperLoop, EngineInit},
    },
    web::reporting_routes::ReportingHttpRoutes,
};

pub async fn setup_node_program<T: TransportLayer + Send + Sync + Clone + 'static>(
    rp_connections: ConnectionsCell,
    rp_conf: RPConf,
    transport_layer: Arc<T>,
    block_retriever: BlockRetriever<T>,
    conf: NodeConf,
    event_publisher: F1r3flyEvents,
    node_discovery: Arc<dyn NodeDiscovery>,
    last_approved_block: Arc<Mutex<Option<ApprovedBlock>>>,
) -> Result<
    (
        Arc<dyn PacketHandler>,
        APIServers,
        CasperLoop,
        CasperLoop,
        EngineInit,
        Arc<dyn CasperLaunch>,
        ReportingHttpRoutes,
        Arc<dyn WebApi>,
        Arc<dyn AdminWebApi>,
        Option<ProductionProposer<T>>,
        mpsc::UnboundedReceiver<(
            Arc<dyn Casper + Send + Sync>,
            bool,
            oneshot::Sender<ProposerResult>,
        )>,
        // TODO: move towards having a single node state - OLD
        Option<RwLock<ProposerState>>,
        BlockProcessor<T>,
        Arc<Mutex<HashSet<BlockHash>>>,
        mpsc::UnboundedSender<(Arc<dyn Casper + Send + Sync>, BlockMessage)>,
        Option<Arc<ProposeFunction>>,
    ),
    CasperError,
> {
    // TODO: Span

    // RNode key-value store manager / manages LMDB databases
    let mut rnode_store_manager = new_key_value_store_manager(conf.storage.data_dir, None);

    // Block storage
    let block_store = KeyValueBlockStore::create_from_kvm(&mut rnode_store_manager).await?;

    // Last finalized Block storage
    let last_finalized_storage =
        LastFinalizedKeyValueStorage::create_from_kvm(&mut rnode_store_manager).await?;

    // Migrate LastFinalizedStorage to BlockDagStorage
    let lfb_require_migration = last_finalized_storage.require_migration()?;
    if lfb_require_migration {
        info!("Migrating LastFinalizedStorage to BlockDagStorage.");
        last_finalized_storage
            .migrate_lfb(&mut rnode_store_manager, &block_store)
            .await?;
    }

    // Block DAG storage
    let block_dag_storage = BlockDagKeyValueStorage::new(&mut rnode_store_manager).await?;

    // Casper requesting blocks cache
    let casper_buffer_storage =
        CasperBufferKeyValueStorage::new_from_kvm(&mut rnode_store_manager).await?;

    // Deploy storage
    let deploy_storage = KeyValueDeployStorage::new(&mut rnode_store_manager).await?;
    let deploy_storage_arc = Arc::new(Mutex::new(deploy_storage.clone()));

    // Safety oracle (clique oracle implementation)
    let oracle = CliqueOracleImpl;

    // Estimator
    let estimator = Estimator::apply(
        conf.casper.max_number_of_parents,
        Some(conf.casper.max_parent_depth),
    );

    // Runtime for `rnode eval`
    let eval_runtime = {
        let eval_stores = rnode_store_manager
            .eval_stores()
            .await
            .map_err(|e| CasperError::Other(format!("Failed to get eval stores: {}", e)))?;

        rho_runtime::create_runtime_from_kv_store(
            eval_stores,
            Par::default(),
            false,
            &mut Vec::new(),
            Arc::new(Box::new(Matcher)),
        )
        .await
    };

    // Runtime manager (play and replay runtimes)
    let (runtime_manager, history_repo) = {
        let rspace_stores = rnode_store_manager
            .r_space_stores()
            .await
            .map_err(|e| CasperError::Other(format!("Failed to get rspace stores: {}", e)))?;

        let mergeable_store = RuntimeManager::mergeable_store(&mut rnode_store_manager).await?;
        RuntimeManager::create_with_history(
            rspace_stores,
            mergeable_store,
            Genesis::non_negative_mergeable_tag_name(),
        )
    };

    // Reporting runtime
    let reporting_runtime = if conf.api_server.enable_reporting {
        // In reporting replay channels map is not needed
        let rspace_stores = rnode_store_manager
            .r_space_stores()
            .await
            .map_err(|e| CasperError::Other(format!("Failed to get rspace stores: {}", e)))?;
        reporting_casper::rho_reporter(&rspace_stores, &block_store, &block_dag_storage)
    } else {
        reporting_casper::noop()
    };

    // RSpace state manager (for CasperLaunch)
    // Note: rnodeStateManager is created in Scala but never used, so we only create rspaceStateManager
    let rspace_state_manager = {
        let exporter = history_repo.exporter();
        let importer = history_repo.importer();
        RSpaceStateManager::new(exporter, importer)
    };

    // Engine dynamic reference
    let engine_cell = EngineCell::init();

    // Block processor queue (for CasperLaunch - uses VecDeque internally)
    let block_processor_queue = Arc::new(Mutex::new(VecDeque::<(
        Arc<dyn MultiParentCasper + Send + Sync>,
        BlockMessage,
    )>::new()));

    // Block processor queue channels (for external communication)
    let (block_processor_queue_tx, block_processor_queue_rx) =
        mpsc::unbounded_channel::<(Arc<dyn Casper + Send + Sync>, BlockMessage)>();

    // Block processing state - set of items currently in processing
    let block_processor_state_ref = Arc::new(Mutex::new(HashSet::<BlockHash>::new()));

    // Block processor
    let block_processor = casper::rust::blocks::block_processor::new_block_processor(
        block_store.clone(),
        casper_buffer_storage.clone(),
        block_dag_storage.clone(),
        block_retriever.clone(),
        transport_layer.clone(),
        rp_connections.clone(),
        rp_conf.clone(),
    );

    // Proposer instance
    let validator_identity_opt = ValidatorIdentity::from_private_key_with_logging(
        conf.casper.validator_private_key.as_deref(),
    );

    let proposer = validator_identity_opt.map(|validator_identity| {
        // Parse dummy deployer key from config
        let dummy_deploy_opt = conf
            .dev
            .deployer_private_key
            .as_ref()
            .and_then(|key_hex| hex::decode(key_hex).ok())
            .map(|bytes| {
                let private_key = PrivateKey::from_bytes(&bytes);
                // TODO: Make term for dummy deploy configurable - OLD
                (private_key, "Nil".to_string())
            });

        casper::rust::blocks::proposer::proposer::new_proposer(
            validator_identity,
            dummy_deploy_opt,
            runtime_manager.clone(),
            block_store.clone(),
            deploy_storage_arc.clone(),
            block_retriever.clone(),
            transport_layer.clone(),
            rp_connections.clone(),
            rp_conf.clone(),
            event_publisher.clone(),
        )
    });

    // Propose request is a tuple - Casper, async flag and deferred proposer result that will be resolved by proposer
    let (proposer_queue_tx, proposer_queue_rx) = mpsc::unbounded_channel::<(
        Arc<dyn Casper + Send + Sync>,
        bool,
        oneshot::Sender<ProposerResult>,
    )>();

    // Trigger propose function - wraps proposerQueue to provide propose functionality
    let trigger_propose_f_opt: Option<Arc<ProposeFunction>> = if proposer.is_some() {
        let queue_tx = proposer_queue_tx.clone();
        Some(Arc::new(
            move |casper: Arc<dyn MultiParentCasper + Send + Sync>, is_async: bool| {
                let queue_tx = queue_tx.clone();
                // Downcast to Arc<dyn Casper + Send + Sync> for the queue (MultiParentCasper extends Casper)
                let casper_for_queue: Arc<dyn Casper + Send + Sync> = casper;

                Box::pin(async move {
                    // Create oneshot channel
                    let (result_tx, result_rx) = oneshot::channel::<ProposerResult>();

                    // Send to proposer queue
                    queue_tx
                        .send((casper_for_queue, is_async, result_tx))
                        .map_err(|e| {
                            CasperError::Other(format!("Failed to send to proposer queue: {}", e))
                        })?;

                    // Wait for result
                    result_rx.await.map_err(|e| {
                        CasperError::Other(format!("Failed to receive proposer result: {}", e))
                    })
                })
            },
        ))
    } else {
        None
    };

    // Proposer state ref - created if trigger_propose_f_opt exists
    let proposer_state_ref_opt: Option<RwLock<ProposerState>> = trigger_propose_f_opt
        .as_ref()
        .map(|_| RwLock::new(ProposerState::default()));

    // CasperLaunch - orchestrates the launch of the Casper consensus
    let casper_launch = {
        // Determine which propose function to use based on autopropose config
        let propose_f_for_launch = if conf.autopropose {
            trigger_propose_f_opt.clone()
        } else {
            None
        };

        // Create CasperLaunch with all dependencies
        Arc::new(casper::rust::engine::casper_launch::CasperLaunchImpl::new(
            // Infrastructure dependencies
            transport_layer,
            rp_conf,
            rp_connections,
            last_approved_block,
            event_publisher,
            block_retriever,
            Arc::new(engine_cell.clone()),
            block_store,
            block_dag_storage,
            deploy_storage,
            casper_buffer_storage,
            rspace_state_manager,
            Arc::new(tokio::sync::Mutex::new(runtime_manager.clone())),
            estimator,
            // Explicit parameters
            block_processor_queue.clone(),
            block_processor_state_ref.clone(),
            propose_f_for_launch,
            conf.casper.clone(),
            !conf.protocol_client.disable_lfs,
            conf.protocol_server.disable_state_exporter,
        )) as Arc<dyn CasperLaunch>
    };

    todo!()
}
