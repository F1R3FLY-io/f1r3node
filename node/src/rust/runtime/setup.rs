// See node/src/main/scala/coop/rchain/node/runtime/Setup.scala

// Imports needed for function signature and return type
use std::{
    collections::HashSet,
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
};
use tokio::sync::{mpsc, oneshot, RwLock};

use models::rust::{
    block_hash::BlockHash,
    casper::protocol::casper_message::{ApprovedBlock, BlockMessage},
};

use casper::rust::{
    blocks::{
        block_processor::BlockProcessor,
        proposer::proposer::{ProductionProposer, ProposerResult},
    },
    casper::{Casper, MultiParentCasper},
    engine::{block_retriever::BlockRetriever, casper_launch::CasperLaunch},
    errors::CasperError,
    state::instances::ProposerState,
    ProposeFunction,
};

use comm::rust::{
    discovery::node_discovery::NodeDiscovery, p2p::packet_handler::PacketHandler,
    rp::connect::ConnectionsCell, transport::transport_layer::TransportLayer,
};

use shared::rust::shared::f1r3fly_events::F1r3flyEvents;

use crate::rust::{
    api::{admin_web_api::AdminWebApi, web_api::WebApi},
    configuration::NodeConf,
    runtime::{
        api_servers::APIServers,
        node_runtime::{CasperLoop, EngineInit},
    },
    web::reporting_routes::{ReportingHttpRoutes, ReportingRoutes},
};

pub async fn setup_node_program<T: TransportLayer + Send + Sync + Clone + 'static>(
    rp_connections: ConnectionsCell,
    rp_conf_cell: comm::rust::rp::rp_conf::RPConfCell,
    transport_layer: Arc<T>,
    block_retriever: BlockRetriever<T>,
    conf: NodeConf,
    event_publisher: F1r3flyEvents,
    node_discovery: Arc<dyn NodeDiscovery + Send + Sync>,
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
        Arc<dyn WebApi + Send + Sync + 'static>,
        Arc<dyn AdminWebApi + Send + Sync + 'static>,
        Option<ProductionProposer<T>>,
        mpsc::UnboundedReceiver<(
            Arc<dyn Casper + Send + Sync>,
            bool,
            oneshot::Sender<ProposerResult>,
        )>,
        mpsc::UnboundedSender<(
            Arc<dyn Casper + Send + Sync>,
            bool,
            oneshot::Sender<ProposerResult>,
        )>,
        Option<Arc<RwLock<ProposerState>>>,
        BlockProcessor<T>,
        Arc<Mutex<HashSet<BlockHash>>>,
        mpsc::UnboundedSender<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>,
        mpsc::UnboundedReceiver<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>,
        Option<Arc<ProposeFunction>>,
        Arc<casper::rust::api::block_report_api::BlockReportAPI>,
        block_storage::rust::key_value_block_store::KeyValueBlockStore,
    ),
    CasperError,
> {
    // TODO: Span

    // RNode key-value store manager / manages LMDB databases
    let mut rnode_store_manager = {
        use casper::rust::storage::rnode_key_value_store_manager::new_key_value_store_manager;

        new_key_value_store_manager(conf.storage.data_dir, None)
    };

    // Block storage
    let block_store = {
        use block_storage::rust::key_value_block_store::KeyValueBlockStore;

        KeyValueBlockStore::create_from_kvm(&mut rnode_store_manager).await?
    };

    // Last finalized Block storage
    let last_finalized_storage = {
        use block_storage::rust::finality::LastFinalizedKeyValueStorage;

        LastFinalizedKeyValueStorage::create_from_kvm(&mut rnode_store_manager).await?
    };

    // Migrate LastFinalizedStorage to BlockDagStorage
    let lfb_require_migration = last_finalized_storage.require_migration()?;
    if lfb_require_migration {
        use tracing::info;

        info!("Migrating LastFinalizedStorage to BlockDagStorage.");
        last_finalized_storage
            .migrate_lfb(&mut rnode_store_manager, &block_store)
            .await?;
    }

    // Block DAG storage
    let block_dag_storage = {
        use block_storage::rust::dag::block_dag_key_value_storage::BlockDagKeyValueStorage;

        BlockDagKeyValueStorage::new(&mut rnode_store_manager).await?
    };

    // Casper requesting blocks cache
    let casper_buffer_storage = {
        use block_storage::rust::casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage;

        CasperBufferKeyValueStorage::new_from_kvm(&mut rnode_store_manager).await?
    };

    // Deploy storage
    let (deploy_storage, deploy_storage_arc) = {
        use block_storage::rust::deploy::key_value_deploy_storage::KeyValueDeployStorage;

        let deploy_storage = KeyValueDeployStorage::new(&mut rnode_store_manager).await?;
        let deploy_storage_arc = Arc::new(Mutex::new(deploy_storage.clone()));
        (deploy_storage, deploy_storage_arc)
    };

    // Safety oracle (clique oracle implementation)
    let oracle = {
        use casper::rust::safety_oracle::CliqueOracleImpl;

        CliqueOracleImpl
    };

    // Estimator
    let estimator = {
        use casper::rust::estimator::Estimator;

        Estimator::apply(
            conf.casper.max_number_of_parents,
            Some(conf.casper.max_parent_depth),
        )
    };

    // Runtime for `rnode eval`
    let eval_runtime = {
        use models::rhoapi::Par;
        use rholang::rust::interpreter::{matcher::r#match::Matcher, rho_runtime};
        use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;

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
        use casper::rust::genesis::genesis::Genesis;
        use casper::rust::util::rholang::runtime_manager::RuntimeManager;
        use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;

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
    let reporting_runtime = {
        use casper::rust::reporting_casper;
        use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;

        if conf.api_server.enable_reporting {
            // In reporting replay channels map is not needed
            let rspace_stores = rnode_store_manager
                .r_space_stores()
                .await
                .map_err(|e| CasperError::Other(format!("Failed to get rspace stores: {}", e)))?;
            reporting_casper::rho_reporter(&rspace_stores, &block_store, &block_dag_storage)
        } else {
            reporting_casper::noop()
        }
    };

    // RSpace state manager (for CasperLaunch)
    // Note: rnodeStateManager is created in Scala but never used, so we only create rspaceStateManager
    let rspace_state_manager = {
        use rspace_plus_plus::rspace::state::rspace_state_manager::RSpaceStateManager;

        let exporter = history_repo.exporter();
        let importer = history_repo.importer();
        RSpaceStateManager::new(exporter, importer)
    };

    // Engine dynamic reference
    let engine_cell = {
        use casper::rust::engine::engine_cell::EngineCell;

        EngineCell::init()
    };

    // Block processor queue - mpsc channel connecting producers (CasperLaunch, Running)
    // to consumer (BlockProcessorInstance)
    let (block_processor_queue_tx, block_processor_queue_rx) =
        mpsc::unbounded_channel::<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>();

    // Block processing state - set of items currently in processing
    let block_processor_state_ref = Arc::new(Mutex::new(HashSet::<BlockHash>::new()));

    // Read RPConf once for use in multiple places
    let rp_conf = rp_conf_cell
        .read()
        .map_err(|e| CasperError::Other(format!("Failed to read RPConf: {}", e)))?;

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
    let validator_identity_opt = {
        use casper::rust::validator_identity::ValidatorIdentity;

        ValidatorIdentity::from_private_key_with_logging(
            conf.casper.validator_private_key.as_deref(),
        )
    };

    let proposer = validator_identity_opt.map(|validator_identity| {
        use crypto::rust::private_key::PrivateKey;

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
    // Wrapped in Arc for sharing across multiple API instances
    let proposer_state_ref_opt: Option<Arc<RwLock<ProposerState>>> = trigger_propose_f_opt
        .as_ref()
        .map(|_| Arc::new(RwLock::new(ProposerState::default())));

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
            transport_layer.clone(),
            rp_conf.clone(),
            rp_connections.clone(),
            last_approved_block,
            event_publisher,
            block_retriever.clone(),
            Arc::new(engine_cell.clone()),
            block_store.clone(),
            block_dag_storage.clone(),
            deploy_storage,
            casper_buffer_storage.clone(),
            rspace_state_manager,
            Arc::new(tokio::sync::Mutex::new(runtime_manager.clone())),
            estimator.clone(),
            // Explicit parameters
            block_processor_queue_tx.clone(),
            block_processor_state_ref.clone(),
            propose_f_for_launch,
            conf.casper.clone(),
            !conf.protocol_client.disable_lfs,
            conf.protocol_server.disable_state_exporter,
        )) as Arc<dyn CasperLaunch>
    };

    // Packet handler - handles incoming Casper protocol messages
    // Note: Scala has a commented-out fairDispatcher option (Setup.scala:268-277) that uses
    // round-robin dispatching with queue management. Currently using simple handler.
    let packet_handler = casper::rust::util::comm::casper_packet_handler::CasperPacketHandler::new(
        engine_cell.clone(),
    );
    let packet_handler: Arc<dyn PacketHandler> = Arc::new(packet_handler);

    // Reporting store - storage for block event reports with LZ4 compression
    let reporting_store =
        casper::rust::report_store::report_store(&mut rnode_store_manager).await?;

    // Block Report API - API for block reporting
    let block_report_api = casper::rust::api::block_report_api::BlockReportAPI::new(
        reporting_runtime,
        reporting_store,
        engine_cell.clone(),
        block_store.clone(),
        oracle,
    );

    // API Servers - gRPC services for REPL, Deploy, Propose, and LSP
    let is_node_read_only = conf.casper.validator_private_key.is_none();

    // Conditional propose function for autopropose
    let propose_f_for_api = if conf.autopropose && conf.dev.deployer_private_key.is_some() {
        trigger_propose_f_opt.clone()
    } else {
        None
    };

    // Clone block_report_api before passing to api_servers since we'll use it later for transaction API and return value
    let block_report_api_for_transaction = block_report_api.clone();
    let block_report_api_for_return = block_report_api.clone();

    // Clone trigger_propose_f_opt before passing to api_servers since we'll use it later for web_api, admin_web_api, and return value
    let trigger_propose_f_opt_for_web_api = trigger_propose_f_opt.clone();
    let trigger_propose_f_opt_for_admin_web_api = trigger_propose_f_opt.clone();
    let trigger_propose_f_opt_for_return = trigger_propose_f_opt.clone();

    // Clone proposer_state_ref_opt before passing to api_servers since we'll use it later for admin_web_api and return value
    let proposer_state_ref_opt_for_admin_web_api = proposer_state_ref_opt.clone();
    let proposer_state_ref_opt_for_return = proposer_state_ref_opt.clone();

    let api_servers = APIServers::build(
        eval_runtime,
        trigger_propose_f_opt,
        proposer_state_ref_opt,
        conf.api_server.max_blocks_limit as i32,
        conf.dev_mode,
        propose_f_for_api,
        block_report_api,
        conf.protocol_server.network_id.clone(),
        conf.casper.shard_name.clone(),
        conf.casper.min_phlo_price,
        is_node_read_only,
        engine_cell.clone(),
        block_store.clone(),
        rp_conf_cell.clone(),
        rp_connections.clone(),
        node_discovery.clone(),
    );

    // Reporting HTTP Routes - REST API for block reporting and tracing
    // Note: In Rust with Axum, BlockReportAPI is accessed via State extraction
    // at runtime rather than being captured at route creation time
    let reporting_routes = ReportingRoutes::create_router();

    // Casper Loop - maintenance loop body for Casper consensus
    // This closure is executed repeatedly to:
    // 1. Fetch missing block dependencies from CasperBuffer
    // 2. Maintain requested blocks with timeout management
    // 3. Sleep for the configured interval
    let casper_loop = {
        let engine_cell_clone = engine_cell.clone();
        let block_retriever_clone = block_retriever.clone();
        let requested_blocks_timeout = conf.casper.requested_blocks_timeout;
        let casper_loop_interval = conf.casper.casper_loop_interval;

        move || -> Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>> {
            let engine_cell = engine_cell_clone.clone();
            let block_retriever = block_retriever_clone.clone();

            Box::pin(async move {
                // Read the engine from engine cell
                let engine = engine_cell.get().await;

                // Fetch dependencies from CasperBuffer
                if let Some(casper) = engine.with_casper() {
                    casper.fetch_dependencies().await?;
                }

                // Maintain RequestedBlocks for Casper
                block_retriever
                    .request_all(requested_blocks_timeout)
                    .await?;

                // Sleep for the configured interval
                tokio::time::sleep(casper_loop_interval).await;

                Ok::<(), CasperError>(())
            })
        }
    };

    // Update Fork Choice Loop - requests fork choice tips if node is stuck
    // Broadcast fork choice tips request if current fork choice is more than
    // `forkChoiceStaleThreshold` old, which indicates the node might be stuck.
    // For details, see Running::update_fork_choice_tips_if_stuck description.
    let update_fork_choice_loop = {
        let engine_cell_clone = engine_cell.clone();
        let transport_layer_clone = transport_layer.clone();
        let rp_connections_clone = rp_connections.clone();
        let rp_conf_cell_clone = rp_conf_cell.clone();
        let fork_choice_check_interval = conf.casper.fork_choice_check_if_stale_interval;
        let fork_choice_stale_threshold = conf.casper.fork_choice_stale_threshold;

        move || -> Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>> {
            let engine_cell = engine_cell_clone.clone();
            let transport_layer = transport_layer_clone.clone();
            let rp_connections = rp_connections_clone.clone();
            let rp_conf_cell = rp_conf_cell_clone.clone();

            Box::pin(async move {
                // Sleep first
                tokio::time::sleep(fork_choice_check_interval).await;

                // Read current RPConf
                let rp_conf = rp_conf_cell
                    .read()
                    .map_err(|e| CasperError::Other(e.to_string()))?;

                // Call the standalone function
                casper::rust::engine::running::update_fork_choice_tips_if_stuck(
                    &engine_cell,
                    &transport_layer,
                    &rp_connections,
                    &rp_conf,
                    fork_choice_stale_threshold,
                )
                .await?;

                Ok::<(), CasperError>(())
            })
        }
    };

    // Engine Init - reads engine from engine cell and calls init
    let engine_init = {
        let engine_cell_clone = engine_cell.clone();

        move || -> Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>> {
            let engine_cell = engine_cell_clone.clone();

            Box::pin(async move {
                let engine = engine_cell.get().await;
                engine.init().await?;
                Ok::<(), CasperError>(())
            })
        }
    };

    // Scala has: runtimeCleanup = NodeRuntime.cleanup(rnodeStoreManager)
    // But it's commented out in NodeRuntime.scala line 321:
    //   //_ <- addShutdownHook(servers, runtimeCleanup, blockStore)
    //
    // Rust implementation notes:
    // - The store managers (LmdbDirStoreManager, LmdbStoreManager) have both:
    //   1. async shutdown() methods for graceful cleanup
    //   2. Drop implementations for fallback cleanup
    // - shutdown() should be called explicitly for proper async cleanup
    // - This should be implemented in the main runtime's signal handler
    //   (SIGTERM, SIGINT, etc.) before program exit
    // - For now, Drop implementations will handle cleanup on program exit
    //
    // When implementing, add shutdown call like:
    //   rnode_store_manager.shutdown().await?;

    let transaction_api = {
        use crate::rust::web::transaction::{transfer_unforgeable, TransactionAPIImpl};

        let transfer_unforgeable_par = transfer_unforgeable();
        TransactionAPIImpl::new(block_report_api_for_transaction, transfer_unforgeable_par)
    };

    let cache_transaction_api = {
        use crate::rust::web::transaction::cache_transaction_api;

        cache_transaction_api(transaction_api, &mut rnode_store_manager)
            .await
            .map_err(|e| {
                CasperError::Other(format!("Failed to create cache transaction API: {}", e))
            })?
    };

    // Web API - HTTP REST API implementation
    let web_api = {
        use crate::rust::api::web_api::WebApiImpl;

        let is_node_read_only = conf.casper.validator_private_key.is_none();

        // Conditional propose function for autopropose
        let trigger_propose_f = if conf.autopropose && conf.dev.deployer_private_key.is_some() {
            trigger_propose_f_opt_for_web_api
        } else {
            None
        };

        WebApiImpl::new(
            conf.api_server.max_blocks_limit as i32,
            conf.dev_mode,
            conf.protocol_server.network_id.clone(),
            conf.casper.shard_name.clone(),
            conf.casper.min_phlo_price,
            is_node_read_only,
            cache_transaction_api,
            Arc::new(engine_cell.clone()),
            rp_conf_cell.clone(),
            rp_connections.clone(),
            node_discovery.clone(),
            trigger_propose_f,
        )
    };

    // Admin Web API - Admin HTTP REST API implementation
    let admin_web_api = {
        use crate::rust::api::admin_web_api::AdminWebApiImpl;

        AdminWebApiImpl::new(
            trigger_propose_f_opt_for_admin_web_api,
            proposer_state_ref_opt_for_admin_web_api,
            Arc::new(engine_cell.clone()),
        )
    };

    // Return all initialized components
    Ok((
        packet_handler,
        api_servers,
        Arc::new(casper_loop),
        Arc::new(update_fork_choice_loop),
        Arc::new(engine_init),
        casper_launch,
        reporting_routes,
        Arc::new(web_api),
        Arc::new(admin_web_api),
        proposer,
        proposer_queue_rx,
        proposer_queue_tx,
        proposer_state_ref_opt_for_return,
        block_processor,
        block_processor_state_ref,
        block_processor_queue_tx,
        block_processor_queue_rx,
        trigger_propose_f_opt_for_return,
        Arc::new(block_report_api_for_return),
        block_store,
    ))
}
