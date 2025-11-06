// See casper/src/main/scala/coop/rchain/casper/engine/CasperLaunch.scala

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::rust::casper::{hash_set_casper, CasperShardConf, MultiParentCasper};
use crate::rust::casper_conf::CasperConf;
use crate::rust::engine::approve_block_protocol::ApproveBlockProtocolFactory;
use crate::rust::engine::block_approver_protocol::BlockApproverProtocol;
use crate::rust::engine::block_retriever::BlockRetriever;
use crate::rust::engine::engine::{transition_to_initializing, transition_to_running};
use crate::rust::engine::engine_cell::EngineCell;
use crate::rust::engine::genesis_ceremony_master::GenesisCeremonyMaster;
use crate::rust::engine::genesis_validator::GenesisValidator;
use crate::rust::errors::CasperError;
use crate::rust::estimator::Estimator;
use crate::rust::multi_parent_casper_impl::MultiParentCasperImpl;
use crate::rust::util::bonds_parser::BondsParser;
use crate::rust::util::rholang::runtime_manager::RuntimeManager;
use crate::rust::util::vault_parser::VaultParser;
use crate::rust::validator_identity::ValidatorIdentity;
use async_trait::async_trait;
use block_storage::rust::casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage;
use block_storage::rust::dag::block_dag_key_value_storage::BlockDagKeyValueStorage;
use block_storage::rust::deploy::key_value_deploy_storage::KeyValueDeployStorage;
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use comm::rust::rp::connect::ConnectionsCell;
use comm::rust::rp::rp_conf::RPConf;
use comm::rust::transport::transport_layer::TransportLayer;
use models::rust::block_hash::BlockHash;
use models::rust::casper::pretty_printer::PrettyPrinter;
use models::rust::casper::protocol::casper_message::{ApprovedBlock, BlockMessage, CasperMessage};
use rspace_plus_plus::rspace::state::rspace_state_manager::RSpaceStateManager;
use shared::rust::shared::f1r3fly_events::F1r3flyEvents;
use std::collections::{HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::time::SystemTime;

#[async_trait]
pub trait CasperLaunch {
    async fn launch(&self) -> Result<(), CasperError>;
}

pub struct CasperLaunchImpl<T: TransportLayer + Send + Sync + Clone + 'static> {
    // Infrastructure dependencies (Scala implicit parameters - Transport, State, Storage, etc.)
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
    runtime_manager: Arc<tokio::sync::Mutex<RuntimeManager>>,
    estimator: Arc<Mutex<Option<Estimator>>>,
    casper_shard_conf: CasperShardConf,

    // Explicit parameters from Scala (in same order as Scala signature)
    block_processing_queue:
        Arc<Mutex<VecDeque<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>>>,
    blocks_in_processing: Arc<Mutex<HashSet<BlockHash>>>,
    propose_f_opt: Option<Arc<crate::rust::ProposeFunction>>,
    conf: CasperConf,
    trim_state: bool,
    disable_state_exporter: bool,
}

impl<T: TransportLayer + Send + Sync + Clone + 'static> CasperLaunchImpl<T> {
    /// Helper method to create MultiParentCasper instance
    /// Scala equivalent: MultiParentCasper.hashSetCasper[F](validatorId, casperShardConf, ab)
    fn create_casper(
        &self,
        validator_id: Option<ValidatorIdentity>,
        ab: BlockMessage,
    ) -> Result<MultiParentCasperImpl<T>, CasperError> {
        // Scala: implicit val requestedBlocks: RequestedBlocks[F] = Ref.unsafe[F, Map[BlockHash, RequestState]](Map.empty)
        let requested_blocks = Arc::new(Mutex::new(HashMap::new()));
        
        // Scala: implicit val blockRetriever: BlockRetriever[F] = BlockRetriever.of[F]
        let block_retriever_for_casper = BlockRetriever::new(
            requested_blocks,
            self.transport_layer.clone(),
            self.connections_cell.clone(),
            self.rp_conf_ask.clone(),
        );

        let events_for_casper = (*self.event_publisher).clone();
        let runtime_manager = self.runtime_manager.clone();

        let estimator = self
            .estimator
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| CasperError::RuntimeError("Estimator not available".to_string()))?;

        let block_store = self
            .block_store
            .lock()
            .unwrap()
            .as_ref()
            .ok_or_else(|| CasperError::RuntimeError("Block store not available".to_string()))?
            .clone();

        let block_dag_storage = self
            .block_dag_storage
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| {
                CasperError::RuntimeError("BlockDag storage not available".to_string())
            })?;

        let deploy_storage =
            self.deploy_storage.lock().unwrap().take().ok_or_else(|| {
                CasperError::RuntimeError("Deploy storage not available".to_string())
            })?;

        let casper_buffer_storage = self
            .casper_buffer_storage
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| {
                CasperError::RuntimeError("Casper buffer storage not available".to_string())
            })?;

        hash_set_casper(
            block_retriever_for_casper,
            events_for_casper,
            runtime_manager,
            estimator,
            block_store,
            block_dag_storage,
            deploy_storage,
            casper_buffer_storage,
            validator_id,
            self.casper_shard_conf.clone(),
            ab,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
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
        runtime_manager: Arc<tokio::sync::Mutex<RuntimeManager>>,
        estimator: Arc<Mutex<Option<Estimator>>>,
        // Explicit parameters (matching Scala signature order)
        block_processing_queue: Arc<
            Mutex<VecDeque<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>>,
        >,
        blocks_in_processing: Arc<Mutex<HashSet<BlockHash>>>,
        propose_f_opt: Option<Arc<crate::rust::ProposeFunction>>,
        conf: CasperConf,
        trim_state: bool,
        disable_state_exporter: bool,
    ) -> Self {
        // Scala equivalent: val casperShardConf = CasperShardConf(...)
        let casper_shard_conf = CasperShardConf {
            fault_tolerance_threshold: conf.fault_tolerance_threshold,
            shard_name: conf.shard_name.clone(),
            parent_shard_id: conf.parent_shard_id.clone(),
            finalization_rate: conf.finalization_rate,
            max_number_of_parents: conf.max_number_of_parents,
            max_parent_depth: conf.max_parent_depth,
            synchrony_constraint_threshold: conf.synchrony_constraint_threshold as f32,
            height_constraint_threshold: conf.height_constraint_threshold,
            deploy_lifespan: 50,
            casper_version: 1,
            config_version: 1,
            bond_minimum: conf.genesis_block_data.bond_minimum,
            bond_maximum: conf.genesis_block_data.bond_maximum,
            epoch_length: conf.genesis_block_data.epoch_length,
            quarantine_length: conf.genesis_block_data.quarantine_length,
            min_phlo_price: conf.min_phlo_price,
        };

        Self {
            // Infrastructure dependencies (implicit parameters)
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
            casper_shard_conf,
            // Explicit parameters
            block_processing_queue,
            blocks_in_processing,
            propose_f_opt,
            conf,
            trim_state,
            disable_state_exporter,
        }
    }

    async fn connect_to_existing_network(
        &self,
        approved_block: ApprovedBlock,
        disable_state_exporter: bool,
    ) -> Result<(), CasperError> {
        async fn ask_peers_for_fork_choice_tips<T: TransportLayer + Send + Sync + Clone>(
            transport_layer: &T,
            connections_cell: &ConnectionsCell,
            rp_conf_ask: &RPConf,
        ) -> Result<(), CasperError> {
            transport_layer
                .send_fork_choice_tip_request(connections_cell, rp_conf_ask)
                .await?;
            Ok(())
        }

        async fn send_buffer_pendants_to_casper<T: TransportLayer + Send + Sync + Clone>(
            casper: Arc<dyn MultiParentCasper + Send + Sync>,
            casper_buffer_storage: &Arc<Mutex<Option<CasperBufferKeyValueStorage>>>,
            block_store: &Arc<Mutex<Option<KeyValueBlockStore>>>,
            block_retriever: &Arc<BlockRetriever<T>>,
            block_processing_queue: &Arc<
                Mutex<VecDeque<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>>,
            >,
        ) -> Result<(), CasperError> {
            println!("sendBufferPendantsToCasper");

            let pendants = casper_buffer_storage
                .lock()
                .unwrap()
                .as_ref()
                .expect("Casper buffer storage not available")
                .get_pendants();

            // Filter pendants to only those that exist in BlockStore
            let mut pendants_stored = Vec::new();
            for hash_serde in pendants.iter() {
                // Convert BlockHashSerde wrapper to BlockHash (Bytes)
                let hash: BlockHash = hash_serde.0.clone();

                // Check if this hash exists in BlockStore
                let contains = block_store
                    .lock()
                    .unwrap()
                    .as_ref()
                    .expect("Block store not available")
                    .contains(&hash)?;

                // If block exists, add hash to filtered list
                if contains {
                    pendants_stored.push(hash);
                }
            }

            log::info!(
                "Checking pendant hashes: {} items in CasperBuffer.",
                pendants_stored.len()
            );

            // Process each pendant hash and send block to Casper for processing
            for hash in pendants_stored {
                // Retrieve block from BlockStore (returns Option)
                let block = block_store
                    .lock()
                    .unwrap()
                    .as_ref()
                    .expect("Block store not available")
                    .get(&hash)?;

                if let Some(block) = block {
                    log::info!(
                        "Pendant {} is available in BlockStore, sending to Casper.",
                        PrettyPrinter::build_string(
                            CasperMessage::BlockMessage(block.clone()),
                            true
                        )
                    );

                    // Check if block already exists in DAG
                    let dag_contains = casper.dag_contains(&hash);

                    // Log error if block unexpectedly exists in DAG (database inconsistency)
                    if dag_contains {
                        log::error!(
                            "Pendant {} is available in DAG, database is supposedly in inconsistent state.",
                            PrettyPrinter::build_string(CasperMessage::BlockMessage(block.clone()), true)
                        );
                    }

                    // Acknowledge that we received this block
                    block_retriever.ack_receive(hash).await?;

                    // Add block to processing queue for validation and addition to DAG
                    let mut queue = block_processing_queue.lock().unwrap();
                    queue.push_back((casper.clone(), block));
                }
            }

            Ok(())
        }

        let validator_id = ValidatorIdentity::from_private_key_with_logging(
            self.conf.validator_private_key.as_deref(),
        );

        let ab = approved_block.candidate.block.clone();

        let casper = self.create_casper(validator_id.clone(), ab)?;
        let casper_arc = Arc::new(casper);

        // Scala equivalent: init = for { _ <- askPeersForForkChoiceTips; _ <- sendBufferPendantsToCasper(casper); _ <- proposeFOpt.traverse(...) } yield ()
        // Create lazy async init computation (matches Scala F[Unit])

        // Note: Double cloning is necessary because:
        // 1. First clone: capture in outer closure (needs to be Fn, not FnOnce)
        // 2. Second clone: move into inner async block
        let transport_layer_for_init = self.transport_layer.clone();
        let connections_cell_for_init = self.connections_cell.clone();
        let rp_conf_ask_for_init = self.rp_conf_ask.clone();
        let casper_for_init = casper_arc.clone();
        let casper_buffer_storage_for_init = self.casper_buffer_storage.clone();
        let block_store_for_init = self.block_store.clone();
        let block_retriever_for_init = self.block_retriever.clone();
        let block_processing_queue_for_init = self.block_processing_queue.clone();
        let propose_f_opt_for_init = self.propose_f_opt.clone();

        let the_init = Arc::new(move || {
            let transport_layer = transport_layer_for_init.clone();
            let connections_cell = connections_cell_for_init.clone();
            let rp_conf_ask = rp_conf_ask_for_init.clone();
            let casper = casper_for_init.clone();
            let casper_buffer_storage = casper_buffer_storage_for_init.clone();
            let block_store = block_store_for_init.clone();
            let block_retriever = block_retriever_for_init.clone();
            let block_processing_queue = block_processing_queue_for_init.clone();
            let propose_f_opt = propose_f_opt_for_init.clone();

            Box::pin(async move {
                ask_peers_for_fork_choice_tips(&*transport_layer, &connections_cell, &rp_conf_ask)
                    .await?;

                send_buffer_pendants_to_casper(
                    casper.clone(),
                    &casper_buffer_storage,
                    &block_store,
                    &block_retriever,
                    &block_processing_queue,
                )
                .await?;

                if let Some(propose_f) = propose_f_opt.as_ref() {
                    propose_f(casper.as_ref(), true)?;
                }

                Ok(())
            }) as Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>>
        });

        // Scala equivalent: Engine.transitionToRunning[F](...)
        transition_to_running(
            self.block_processing_queue.clone(),
            self.blocks_in_processing.clone(),
            casper_arc,
            approved_block,
            the_init,
            disable_state_exporter,
            self.connections_cell.clone(),
            self.transport_layer.clone(),
            self.rp_conf_ask.clone(),
            self.block_retriever.clone(),
            &*self.engine_cell,
            &self.event_publisher,
        )
        .await?;

        Ok(())
    }

    async fn connect_as_genesis_validator(&self) -> Result<(), CasperError> {
        println!("connectAsGenesisValidator");

        let timestamp = self
            .conf
            .genesis_block_data
            .deploy_timestamp
            .unwrap_or_else(|| {
                SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as i64
            });

        let bonds = BondsParser::parse_with_autogen(
            &self.conf.genesis_block_data.bonds_file,
            self.conf.genesis_ceremony.autogen_shard_size as usize,
        )
        .map_err(|e| CasperError::RuntimeError(format!("Failed to parse bonds: {}", e)))?;

        let validator_id = ValidatorIdentity::from_private_key_with_logging(
            self.conf.validator_private_key.as_deref(),
        )
        .ok_or_else(|| {
            CasperError::RuntimeError(
                "Validator identity required for genesis validator".to_string(),
            )
        })?;

        let vaults =
            VaultParser::parse_from_path_str(&self.conf.genesis_block_data.wallets_file)
                .map_err(|e| CasperError::RuntimeError(format!("Failed to parse vaults: {}", e)))?;

        let bap = BlockApproverProtocol::new(
            validator_id.clone(),
            timestamp,
            vaults,
            bonds,
            self.conf.genesis_block_data.bond_minimum,
            self.conf.genesis_block_data.bond_maximum,
            self.conf.genesis_block_data.epoch_length,
            self.conf.genesis_block_data.quarantine_length,
            self.conf.genesis_block_data.number_of_active_validators,
            self.conf.genesis_ceremony.required_signatures,
            self.conf
                .genesis_block_data
                .pos_multi_sig_public_keys
                .clone(),
            self.conf.genesis_block_data.pos_multi_sig_quorum,
            self.transport_layer.clone(),
            Arc::new(self.rp_conf_ask.clone()),
        )?;

        // Scala equivalent: EngineCell[F].set(new GenesisValidator(...))
        let genesis_validator = GenesisValidator::new(
            self.block_processing_queue.clone(),
            self.blocks_in_processing.clone(),
            self.casper_shard_conf.clone(),
            validator_id,
            bap,
            self.transport_layer.clone(),
            self.rp_conf_ask.clone(),
            self.connections_cell.clone(),
            self.last_approved_block.clone(),
            self.event_publisher.clone(),
            self.block_retriever.clone(),
            self.engine_cell.clone(),
            self.block_store.clone(),
            self.block_dag_storage.clone(),
            self.deploy_storage.clone(),
            self.casper_buffer_storage.clone(),
            self.rspace_state_manager.clone(),
            self.runtime_manager.clone(),
            self.estimator.clone(),
        );

        self.engine_cell.set(Arc::new(genesis_validator)).await;

        Ok(())
    }

    async fn init_bootstrap(&self, disable_state_exporter: bool) -> Result<(), CasperError> {
        println!("initBootstrap");

        let validator_id = ValidatorIdentity::from_private_key_with_logging(
            self.conf.validator_private_key.as_deref(),
        );

        // Scala equivalent: abp <- ApproveBlockProtocol.of[F](...)
        let abp = ApproveBlockProtocolFactory::create(
            self.conf.genesis_block_data.bonds_file.clone(),
            self.conf.genesis_ceremony.autogen_shard_size,
            self.conf.genesis_block_data.wallets_file.clone(),
            self.conf.genesis_block_data.bond_minimum,
            self.conf.genesis_block_data.bond_maximum,
            self.conf.genesis_block_data.epoch_length,
            self.conf.genesis_block_data.quarantine_length,
            self.conf.genesis_block_data.number_of_active_validators,
            self.casper_shard_conf.shard_name.clone(),
            self.conf.genesis_block_data.deploy_timestamp,
            self.conf.genesis_ceremony.required_signatures,
            self.conf.genesis_ceremony.approve_duration,
            self.conf.genesis_ceremony.approve_interval,
            self.conf.genesis_block_data.genesis_block_number,
            self.conf
                .genesis_block_data
                .pos_multi_sig_public_keys
                .clone(),
            self.conf.genesis_block_data.pos_multi_sig_quorum,
            &mut *self.runtime_manager.lock().await,
            self.last_approved_block.clone(),
            None, // metrics
            None, // event_log
            self.transport_layer.clone(),
            Arc::new(self.connections_cell.clone()),
            Arc::new(self.rp_conf_ask.clone()),
        )
        .await?;

        // Scala equivalent: Concurrent[F].start(GenesisCeremonyMaster.waitingForApprovedBlockLoop[F](...))
        tokio::spawn({
            let block_processing_queue = self.block_processing_queue.clone();
            let blocks_in_processing = self.blocks_in_processing.clone();
            let casper_shard_conf = self.casper_shard_conf.clone();
            let validator_id = validator_id.clone();
            let transport_layer = self.transport_layer.clone();
            let rp_conf_ask = self.rp_conf_ask.clone();
            let connections_cell = self.connections_cell.clone();
            let last_approved_block = self.last_approved_block.clone();
            let block_store = self.block_store.clone();
            let block_dag_storage = self.block_dag_storage.clone();
            let deploy_storage = self.deploy_storage.clone();
            let casper_buffer_storage = self.casper_buffer_storage.clone();
            let event_publisher = self.event_publisher.clone();
            let block_retriever = self.block_retriever.clone();
            let engine_cell = self.engine_cell.clone();
            let runtime_manager = self.runtime_manager.clone();
            let estimator = self.estimator.clone();

            async move {
                if let Err(e) = GenesisCeremonyMaster::waiting_for_approved_block_loop(
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
                    runtime_manager,
                    estimator,
                    block_processing_queue,
                    blocks_in_processing,
                    casper_shard_conf,
                    validator_id,
                    disable_state_exporter,
                )
                .await
                {
                    log::error!("waitingForApprovedBlockLoop failed: {:?}", e);
                }
            }
        });

        let genesis_ceremony_master = GenesisCeremonyMaster::new(Arc::new(abp));
        self.engine_cell
            .set(Arc::new(genesis_ceremony_master))
            .await;

        Ok(())
    }

    async fn connect_and_query_approved_block(
        &self,
        trim_state: bool,
        disable_state_exporter: bool,
    ) -> Result<(), CasperError> {
        let validator_id = ValidatorIdentity::from_private_key_with_logging(
            self.conf.validator_private_key.as_deref(),
        );

        // Scala: CommUtil[F].requestApprovedBlock(trimState) - passed as init to transitionToInitializing
        let transport_layer_for_init = self.transport_layer.clone();
        let rp_conf_ask_for_init = self.rp_conf_ask.clone();

        let init = Arc::new(move || {
            let transport_layer = transport_layer_for_init.clone();
            let rp_conf_ask = rp_conf_ask_for_init.clone();

            Box::pin(async move {
                transport_layer
                    .request_approved_block(&rp_conf_ask, Some(trim_state))
                    .await?;
                Ok(())
            }) as Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>>
        });

        // Scala equivalent: Engine.transitionToInitializing(...)
        transition_to_initializing(
            &self.block_processing_queue,
            &self.blocks_in_processing,
            &self.casper_shard_conf,
            &validator_id,
            init,
            trim_state,
            disable_state_exporter,
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
        .await?;

        Ok(())
    }
}

#[async_trait]
impl<T: TransportLayer + Send + Sync + Clone + 'static> CasperLaunch for CasperLaunchImpl<T> {
    async fn launch(&self) -> Result<(), CasperError> {
        let approved_block_opt = self
            .block_store
            .lock()
            .unwrap()
            .as_ref()
            .expect("Block store not available")
            .get_approved_block()?;

        let (msg, action_result) = match approved_block_opt {
            Some(approved_block) => {
                let msg = "Approved block found, reconnecting to existing network";
                let action_result = self
                    .connect_to_existing_network(approved_block, self.disable_state_exporter)
                    .await;
                (msg, action_result)
            }

            None if self.conf.genesis_ceremony.genesis_validator_mode => {
                let msg = "Approved block not found, taking part in ceremony as genesis validator";
                let action_result = self.connect_as_genesis_validator().await;
                (msg, action_result)
            }

            None if self.conf.genesis_ceremony.ceremony_master_mode => {
                let msg = "Approved block not found, taking part in ceremony as ceremony master";
                let action_result = self.init_bootstrap(self.disable_state_exporter).await;
                (msg, action_result)
            }

            None => {
                let msg = "Approved block not found, connecting to existing network";
                let action_result = self
                    .connect_and_query_approved_block(self.trim_state, self.disable_state_exporter)
                    .await;
                (msg, action_result)
            }
        };

        // Scala equivalent: case (msg, action) => Log[F].info(msg) >> action
        log::info!("{}", msg);
        action_result
    }
}
