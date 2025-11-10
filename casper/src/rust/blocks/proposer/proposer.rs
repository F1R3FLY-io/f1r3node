// See casper/src/main/scala/coop/rchain/casper/blocks/proposer/Proposer.scala

use log;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

use block_storage::rust::{
    deploy::key_value_deploy_storage::KeyValueDeployStorage,
    key_value_block_store::KeyValueBlockStore,
};
use comm::rust::{
    rp::{connect::ConnectionsCell, rp_conf::RPConf},
    transport::transport_layer::TransportLayer,
};
use crypto::rust::private_key::PrivateKey;
use models::rust::casper::protocol::casper_message::BlockMessage;
use shared::rust::shared::f1r3fly_events::F1r3flyEvents;

use crate::rust::{
    blocks::proposer::{
        block_creator,
        propose_result::{
            BlockCreatorResult, CheckProposeConstraintsFailure, CheckProposeConstraintsResult,
            ProposeFailure, ProposeResult,
        },
    },
    casper::{Casper, CasperSnapshot},
    engine::block_retriever::BlockRetriever,
    errors::CasperError,
    last_finalized_height_constraint_checker,
    multi_parent_casper_impl::{self},
    synchrony_constraint_checker,
    util::rholang::runtime_manager::RuntimeManager,
    validator_identity::ValidatorIdentity,
    ValidBlockProcessing,
};

use super::propose_result::ProposeStatus;

// Traits for dependency injection and testing
pub trait CasperSnapshotProvider {
    async fn get_casper_snapshot(
        &self,
        casper: Arc<dyn Casper + Send + Sync + 'static>,
    ) -> Result<CasperSnapshot, CasperError>;
}

pub trait ActiveValidatorChecker {
    fn check_active_validator(
        &self,
        casper_snapshot: &CasperSnapshot,
        validator_identity: &ValidatorIdentity,
    ) -> CheckProposeConstraintsResult;
}

pub trait StakeChecker {
    async fn check_enough_base_stake(
        &self,
        casper_snapshot: &CasperSnapshot,
    ) -> Result<CheckProposeConstraintsResult, CasperError>;
}

pub trait HeightChecker {
    async fn check_finalized_height(
        &self,
        casper_snapshot: &CasperSnapshot,
    ) -> Result<CheckProposeConstraintsResult, CasperError>;
}

pub trait BlockCreator {
    async fn create_block(
        &mut self,
        casper_snapshot: &CasperSnapshot,
        validator_identity: &ValidatorIdentity,
        dummy_deploy_opt: Option<(PrivateKey, String)>,
        allow_empty_blocks: bool,
    ) -> Result<BlockCreatorResult, CasperError>;
}

pub trait BlockValidator {
    async fn validate_block(
        &self,
        casper: Arc<dyn Casper + Send + Sync + 'static>,
        casper_snapshot: &mut CasperSnapshot,
        block: &BlockMessage,
    ) -> Result<ValidBlockProcessing, CasperError>;
}

pub trait ProposeEffectHandler {
    async fn handle_propose_effect(
        &mut self,
        casper: Arc<dyn Casper + Send + Sync + 'static>,
        block: &BlockMessage,
    ) -> Result<(), CasperError>;
}

pub enum ProposerResult {
    Empty,
    Success(ProposeStatus, BlockMessage),
    Failure(ProposeStatus, i32),
    Started(i32),
}

impl ProposerResult {
    pub fn empty() -> Self {
        Self::Empty
    }

    pub fn success(status: ProposeStatus, block: BlockMessage) -> Self {
        Self::Success(status, block)
    }

    pub fn failure(status: ProposeStatus, seq_number: i32) -> Self {
        Self::Failure(status, seq_number)
    }

    pub fn started(seq_number: i32) -> Self {
        Self::Started(seq_number)
    }
}

pub struct Proposer<C, A, S, H, BC, BV, E>
where
    C: CasperSnapshotProvider,
    A: ActiveValidatorChecker,
    S: StakeChecker,
    H: HeightChecker,
    BC: BlockCreator,
    BV: BlockValidator,
    E: ProposeEffectHandler,
{
    pub validator: Arc<ValidatorIdentity>,
    pub dummy_deploy_opt: Option<(PrivateKey, String)>,
    pub casper_snapshot_provider: C,
    pub active_validator_checker: A,
    pub stake_checker: S,
    pub height_checker: H,
    pub block_creator: BC,
    pub block_validator: BV,
    pub propose_effect_handler: E,
}

impl<C, A, S, H, BC, BV, E> Proposer<C, A, S, H, BC, BV, E>
where
    C: CasperSnapshotProvider,
    A: ActiveValidatorChecker,
    S: StakeChecker,
    H: HeightChecker,
    BC: BlockCreator,
    BV: BlockValidator,
    E: ProposeEffectHandler,
{
    pub fn new(
        validator: Arc<ValidatorIdentity>,
        dummy_deploy_opt: Option<(PrivateKey, String)>,
        casper_snapshot_provider: C,
        active_validator_checker: A,
        stake_checker: S,
        height_checker: H,
        block_creator: BC,
        block_validator: BV,
        propose_effect_handler: E,
    ) -> Self {
        Self {
            validator,
            dummy_deploy_opt,
            casper_snapshot_provider,
            active_validator_checker,
            stake_checker,
            height_checker,
            block_creator,
            block_validator,
            propose_effect_handler,
        }
    }

    // This is the whole logic of propose
    async fn do_propose(
        &mut self,
        casper_snapshot: &mut CasperSnapshot,
        casper: Arc<dyn Casper + Send + Sync + 'static>,
    ) -> Result<(ProposeResult, Option<BlockMessage>), CasperError> {
        // check if node is allowed to propose a block
        let constraint_check = self.check_propose_constraints(casper_snapshot).await?;

        match constraint_check {
            CheckProposeConstraintsResult::Failure(failure) => Ok((
                ProposeResult::failure(ProposeFailure::CheckConstraintsFailure(failure)),
                None,
            )),
            CheckProposeConstraintsResult::Success => {
                let block_result = self
                    .block_creator
                    .create_block(
                        casper_snapshot,
                        &self.validator,
                        self.dummy_deploy_opt.clone(),
                        false,
                    )
                    .await?;

                match block_result {
                    BlockCreatorResult::NoNewDeploys => {
                        Ok((ProposeResult::failure(ProposeFailure::NoNewDeploys), None))
                    }
                    BlockCreatorResult::Created(block) => {
                        let validation_result = self
                            .block_validator
                            .validate_block(casper.clone(), casper_snapshot, &block)
                            .await?;

                        match validation_result {
                            ValidBlockProcessing::Right(valid_status) => {
                                self.propose_effect_handler
                                    .handle_propose_effect(casper, &block)
                                    .await?;
                                Ok((ProposeResult::success(valid_status), Some(block)))
                            }
                            ValidBlockProcessing::Left(invalid_reason) => {
                                return Err(CasperError::RuntimeError(format!(
                                    "Validation of self created block failed with reason: {:?}, cancelling propose.",
                                    invalid_reason
                                )));
                            }
                        }
                    }
                }
            }
        }
    }

    // Check if proposer can issue a block
    pub async fn check_propose_constraints(
        &mut self,
        casper_snapshot: &CasperSnapshot,
    ) -> Result<CheckProposeConstraintsResult, CasperError> {
        match self
            .active_validator_checker
            .check_active_validator(casper_snapshot, &self.validator)
        {
            CheckProposeConstraintsResult::Failure(CheckProposeConstraintsFailure::NotBonded) => {
                Ok(CheckProposeConstraintsResult::not_bonded())
            }
            _ => {
                // Run both async checks in parallel
                let (stake_result, height_result) = tokio::join!(
                    self.stake_checker.check_enough_base_stake(casper_snapshot),
                    self.height_checker.check_finalized_height(casper_snapshot)
                );

                // Handle any errors from the async calls
                let stake_check = stake_result?;
                let height_check = height_result?;

                // Pick some result that is not Success, or return Success
                let results = vec![stake_check, height_check];
                for result in results {
                    match result {
                        CheckProposeConstraintsResult::Success => continue,
                        failure => return Ok(failure),
                    }
                }

                Ok(CheckProposeConstraintsResult::success())
            }
        }
    }

    pub async fn propose(
        &mut self,
        casper: Arc<dyn Casper + Send + Sync + 'static>,
        is_async: bool,
        propose_id_sender: oneshot::Sender<ProposerResult>,
    ) -> Result<(ProposeResult, Option<BlockMessage>), CasperError> {
        // Using tracing events instead of spans for async context
        // Span[F].traceI("do-propose") equivalent from Scala
        tracing::info!(target: "f1r3fly.casper.proposer", "do-propose-started");
        tracing::debug!(target: "f1r3fly.casper.proposer", "started-do-propose");
        
        fn get_validator_next_seq_number(
            casper_snapshot: &CasperSnapshot,
            validator_public_key: &[u8],
        ) -> i32 {
            casper_snapshot
                .max_seq_nums
                .get(validator_public_key)
                .map(|seq| *seq + 1)
                .unwrap_or(1) as i32
        }

        let start_time = std::time::Instant::now();

        // get snapshot to serve as a base for propose
        let mut casper_snapshot = self
            .casper_snapshot_provider
            .get_casper_snapshot(casper.clone())
            .await?;

        let elapsed = start_time.elapsed();
        log::info!("getCasperSnapshot [{}ms]", elapsed.as_millis());

        let result = if is_async {
            let next_seq =
                get_validator_next_seq_number(&casper_snapshot, &self.validator.public_key.bytes);
            let _ = propose_id_sender.send(ProposerResult::started(next_seq));

            // propose
            self.do_propose(&mut casper_snapshot, casper.clone())
                .await?
        } else {
            // propose
            let result = self.do_propose(&mut casper_snapshot, casper).await?;

            let (propose_result, block_opt) = &result;
            let proposer_result = match block_opt {
                None => {
                    let seq_number = get_validator_next_seq_number(
                        &casper_snapshot,
                        &self.validator.public_key.bytes,
                    );
                    ProposerResult::failure(propose_result.propose_status.clone(), seq_number)
                }
                Some(block) => {
                    ProposerResult::success(propose_result.propose_status.clone(), block.clone())
                }
            };
            let _ = propose_id_sender.send(proposer_result);

            result
        };

        tracing::debug!(target: "f1r3fly.casper.proposer", "finished-do-propose");
        tracing::info!(target: "f1r3fly.casper.proposer", "do-propose-finished");
        Ok(result)
    }
}

pub type ProductionProposer<T> = Proposer<
    ProductionCasperSnapshotProvider,
    ProductionActiveValidatorChecker,
    ProductionStakeChecker,
    ProductionHeightChecker,
    ProductionBlockCreator,
    ProductionBlockValidator,
    ProductionProposeEffectHandler<T>,
>;

pub fn new_proposer<T: TransportLayer + Send + Sync>(
    validator: ValidatorIdentity,
    dummy_deploy_opt: Option<(PrivateKey, String)>,
    runtime_manager: RuntimeManager,
    block_store: KeyValueBlockStore,
    deploy_storage: Arc<Mutex<KeyValueDeployStorage>>,
    block_retriever: BlockRetriever<T>,
    transport: Arc<T>,
    connections_cell: ConnectionsCell,
    conf: RPConf,
    event_publisher: F1r3flyEvents,
) -> ProductionProposer<T> {
    let validator_arc = Arc::new(validator);

    Proposer::new(
        validator_arc.clone(),
        dummy_deploy_opt,
        ProductionCasperSnapshotProvider,
        ProductionActiveValidatorChecker,
        ProductionStakeChecker::new(
            runtime_manager.clone(),
            block_store.clone(),
            validator_arc.clone(),
        ),
        ProductionHeightChecker::new(validator_arc),
        ProductionBlockCreator::new(deploy_storage, runtime_manager.clone(), block_store.clone()),
        ProductionBlockValidator,
        ProductionProposeEffectHandler::new(
            block_store,
            block_retriever,
            transport,
            connections_cell,
            conf,
            event_publisher,
        ),
    )
}

pub struct ProductionCasperSnapshotProvider;
impl CasperSnapshotProvider for ProductionCasperSnapshotProvider {
    async fn get_casper_snapshot(
        &self,
        casper: Arc<dyn Casper + Send + Sync + 'static>,
    ) -> Result<CasperSnapshot, CasperError> {
        casper.get_snapshot().await
    }
}

pub struct ProductionActiveValidatorChecker;
impl ActiveValidatorChecker for ProductionActiveValidatorChecker {
    fn check_active_validator(
        &self,
        casper_snapshot: &CasperSnapshot,
        validator_identity: &ValidatorIdentity,
    ) -> CheckProposeConstraintsResult {
        if casper_snapshot
            .on_chain_state
            .active_validators
            .contains(&validator_identity.public_key.bytes)
        {
            CheckProposeConstraintsResult::success()
        } else {
            CheckProposeConstraintsResult::not_bonded()
        }
    }
}

pub struct ProductionStakeChecker {
    runtime_manager: RuntimeManager,
    block_store: KeyValueBlockStore,
    validator: Arc<ValidatorIdentity>,
}

impl ProductionStakeChecker {
    pub fn new(
        runtime_manager: RuntimeManager,
        block_store: KeyValueBlockStore,
        validator: Arc<ValidatorIdentity>,
    ) -> Self {
        Self {
            runtime_manager,
            block_store,
            validator,
        }
    }
}

impl StakeChecker for ProductionStakeChecker {
    async fn check_enough_base_stake(
        &self,
        casper_snapshot: &CasperSnapshot,
    ) -> Result<CheckProposeConstraintsResult, CasperError> {
        synchrony_constraint_checker::check(
            casper_snapshot,
            &self.runtime_manager,
            &self.block_store,
            &self.validator,
        )
        .await
    }
}

pub struct ProductionHeightChecker {
    validator: Arc<ValidatorIdentity>,
}

impl ProductionHeightChecker {
    pub fn new(validator: Arc<ValidatorIdentity>) -> Self {
        Self { validator }
    }
}

impl HeightChecker for ProductionHeightChecker {
    async fn check_finalized_height(
        &self,
        casper_snapshot: &CasperSnapshot,
    ) -> Result<CheckProposeConstraintsResult, CasperError> {
        last_finalized_height_constraint_checker::check(casper_snapshot, &self.validator)
    }
}

pub struct ProductionBlockCreator {
    deploy_storage: Arc<Mutex<KeyValueDeployStorage>>,
    runtime_manager: RuntimeManager,
    block_store: KeyValueBlockStore,
}

impl ProductionBlockCreator {
    pub fn new(
        deploy_storage: Arc<Mutex<KeyValueDeployStorage>>,
        runtime_manager: RuntimeManager,
        block_store: KeyValueBlockStore,
    ) -> Self {
        Self {
            deploy_storage,
            runtime_manager,
            block_store,
        }
    }
}

impl BlockCreator for ProductionBlockCreator {
    async fn create_block(
        &mut self,
        casper_snapshot: &CasperSnapshot,
        validator_identity: &ValidatorIdentity,
        dummy_deploy_opt: Option<(PrivateKey, String)>,
        allow_empty_blocks: bool,
    ) -> Result<BlockCreatorResult, CasperError> {
        block_creator::create(
            casper_snapshot,
            validator_identity,
            dummy_deploy_opt,
            self.deploy_storage.clone(),
            &mut self.runtime_manager,
            &mut self.block_store,
            allow_empty_blocks,
        )
        .await
    }
}

pub struct ProductionBlockValidator;
impl BlockValidator for ProductionBlockValidator {
    async fn validate_block(
        &self,
        casper: Arc<dyn Casper + Send + Sync + 'static>,
        casper_snapshot: &mut CasperSnapshot,
        block: &BlockMessage,
    ) -> Result<ValidBlockProcessing, CasperError> {
        casper.validate(block, casper_snapshot).await
    }
}

pub struct ProductionProposeEffectHandler<T: TransportLayer + Send + Sync> {
    block_store: KeyValueBlockStore,
    block_retriever: BlockRetriever<T>,
    transport: Arc<T>,
    connections_cell: ConnectionsCell,
    conf: RPConf,
    event_publisher: F1r3flyEvents,
}

impl<T: TransportLayer + Send + Sync> ProductionProposeEffectHandler<T> {
    pub fn new(
        block_store: KeyValueBlockStore,
        block_retriever: BlockRetriever<T>,
        transport: Arc<T>,
        connections_cell: ConnectionsCell,
        conf: RPConf,
        event_publisher: F1r3flyEvents,
    ) -> Self {
        Self {
            block_store,
            block_retriever,
            transport,
            connections_cell,
            conf,
            event_publisher,
        }
    }
}

impl<T: TransportLayer + Send + Sync> ProposeEffectHandler for ProductionProposeEffectHandler<T> {
    async fn handle_propose_effect(
        &mut self,
        casper: Arc<dyn Casper + Send + Sync + 'static>,
        block: &BlockMessage,
    ) -> Result<(), CasperError> {
        // store block
        self.block_store.put_block_message(block)?;

        // save changes to Casper
        casper.handle_valid_block(block).await?;

        // inform block retriever about block
        self.block_retriever
            .ack_in_casper(block.block_hash.clone())
            .await?;

        // broadcast hash to peers
        self.transport
            .send_block_hash(
                &self.connections_cell,
                &self.conf,
                &block.block_hash,
                &block.sender,
            )
            .await?;

        // Publish event
        self.event_publisher
            .publish(multi_parent_casper_impl::created_event(block))
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?;

        Ok(())
    }
}
