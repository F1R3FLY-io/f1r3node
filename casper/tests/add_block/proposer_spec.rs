// See casper/src/test/scala/coop/rchain/casper/addblock/ProposerSpec.scala

use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

use crate::{
    helper::block_dag_storage_fixture::with_storage,
    util::{
        genesis_builder::DEFAULT_VALIDATOR_SKS,
        rholang::resources::{mk_dummy_casper_snapshot, mk_runtime_manager},
    },
};
use casper::rust::{
    blocks::proposer::{
        propose_result::{BlockCreatorResult, CheckProposeConstraintsResult},
        proposer::{
            ActiveValidatorChecker, BlockCreator, BlockValidator, CasperSnapshotProvider,
            HeightChecker, ProposeEffectHandler, Proposer, StakeChecker,
        },
    },
    casper::{Casper, CasperSnapshot},
    errors::CasperError,
    validator_identity::ValidatorIdentity,
    ValidBlockProcessing,
};
use crypto::rust::private_key::PrivateKey;
use models::rust::casper::protocol::casper_message::BlockMessage;

// Test implementations for different scenarios
pub struct TestCasperSnapshotProvider;
impl CasperSnapshotProvider for TestCasperSnapshotProvider {
    async fn get_casper_snapshot(
        &self,
        _: &mut impl Casper,
    ) -> Result<CasperSnapshot, CasperError> {
        Ok(mk_dummy_casper_snapshot())
    }
}

pub struct AlwaysNotActiveChecker;
impl ActiveValidatorChecker for AlwaysNotActiveChecker {
    fn check_active_validator(
        &self,
        _: &CasperSnapshot,
        _: &ValidatorIdentity,
    ) -> CheckProposeConstraintsResult {
        CheckProposeConstraintsResult::not_bonded()
    }
}

pub struct AlwaysActiveChecker;
impl ActiveValidatorChecker for AlwaysActiveChecker {
    fn check_active_validator(
        &self,
        _: &CasperSnapshot,
        _: &ValidatorIdentity,
    ) -> CheckProposeConstraintsResult {
        CheckProposeConstraintsResult::success()
    }
}

pub struct AlwaysNotEnoughBlocksStakeChecker;
impl StakeChecker for AlwaysNotEnoughBlocksStakeChecker {
    async fn check_enough_base_stake(
        &self,
        _: &CasperSnapshot,
    ) -> Result<CheckProposeConstraintsResult, CasperError> {
        Ok(CheckProposeConstraintsResult::not_enough_new_block())
    }
}

pub struct AlwaysTooFarAheadChecker;
impl HeightChecker for AlwaysTooFarAheadChecker {
    async fn check_finalized_height(
        &self,
        _: &CasperSnapshot,
    ) -> Result<CheckProposeConstraintsResult, CasperError> {
        Ok(CheckProposeConstraintsResult::too_far_ahead_of_last_finalized())
    }
}

pub struct OkProposeConstraintStakeChecker;
impl StakeChecker for OkProposeConstraintStakeChecker {
    async fn check_enough_base_stake(
        &self,
        _: &CasperSnapshot,
    ) -> Result<CheckProposeConstraintsResult, CasperError> {
        Ok(CheckProposeConstraintsResult::success())
    }
}

pub struct OkHeightChecker;
impl HeightChecker for OkHeightChecker {
    async fn check_finalized_height(
        &self,
        _: &CasperSnapshot,
    ) -> Result<CheckProposeConstraintsResult, CasperError> {
        Ok(CheckProposeConstraintsResult::success())
    }
}

pub struct TestBlockCreator;
impl BlockCreator for TestBlockCreator {
    async fn create_block(
        &mut self,
        _: &CasperSnapshot,
        _: &ValidatorIdentity,
        _: Option<(PrivateKey, String)>,
    ) -> Result<BlockCreatorResult, CasperError> {
        use models::rust::block_implicits::get_random_block_default;
        Ok(BlockCreatorResult::Created(get_random_block_default()))
    }
}

pub struct TestBlockValidator;
impl BlockValidator for TestBlockValidator {
    async fn validate_block(
        &self,
        _: &mut impl Casper,
        _: &mut CasperSnapshot,
        _: &BlockMessage,
    ) -> Result<ValidBlockProcessing, CasperError> {
        use casper::rust::block_status::ValidBlock;
        Ok(ValidBlockProcessing::Right(ValidBlock::Valid))
    }
}

pub struct AlwaysUnsuccessfulValidator;
impl BlockValidator for AlwaysUnsuccessfulValidator {
    async fn validate_block(
        &self,
        _: &mut impl Casper,
        _: &mut CasperSnapshot,
        _: &BlockMessage,
    ) -> Result<ValidBlockProcessing, CasperError> {
        use casper::rust::block_status::{BlockError, InvalidBlock};
        Ok(ValidBlockProcessing::Left(BlockError::Invalid(
            InvalidBlock::InvalidFormat,
        )))
    }
}

pub struct TestProposeEffectHandler;
impl ProposeEffectHandler for TestProposeEffectHandler {
    async fn handle_propose_effect(
        &mut self,
        _: &mut impl Casper,
        _: &BlockMessage,
    ) -> Result<(), CasperError> {
        Ok(())
    }
}

use std::sync::atomic::{AtomicI32, Ordering};

// Global variable to track propose effects (similar to proposeEffectVar in Scala)
static PROPOSE_EFFECT_VAR: AtomicI32 = AtomicI32::new(0);

pub struct TrackingProposeEffectHandler {
    value: i32,
}

impl TrackingProposeEffectHandler {
    pub fn new(value: i32) -> Self {
        Self { value }
    }
}

impl ProposeEffectHandler for TrackingProposeEffectHandler {
    async fn handle_propose_effect(
        &mut self,
        _: &mut impl Casper,
        _: &BlockMessage,
    ) -> Result<(), CasperError> {
        PROPOSE_EFFECT_VAR.store(self.value, Ordering::SeqCst);
        Ok(())
    }
}

fn get_propose_effect_var() -> i32 {
    PROPOSE_EFFECT_VAR.load(Ordering::SeqCst)
}

fn reset_propose_effect_var() {
    PROPOSE_EFFECT_VAR.store(0, Ordering::SeqCst);
}

fn dummy_validator_identity() -> ValidatorIdentity {
    ValidatorIdentity::new(&DEFAULT_VALIDATOR_SKS[0])
}

#[tokio::test]
async fn proposer_should_reject_to_propose_if_proposer_is_not_active_validator() {
    with_storage(|block_store, block_dag_storage| async move {
        let runtime_manager = mk_runtime_manager("block-query-response-api-test", None).await;
        let validator_identity = Arc::new(dummy_validator_identity());

        let mut proposer = Proposer::new(
            validator_identity,
            None,
            TestCasperSnapshotProvider,
            AlwaysNotActiveChecker,
            OkProposeConstraintStakeChecker,
            OkHeightChecker,
            TestBlockCreator,
            TestBlockValidator,
            TestProposeEffectHandler,
        );

        use crate::helper::no_ops_casper_effect::NoOpsCasperEffect;
        use std::collections::HashMap;

        let dag_representation = block_dag_storage.get_representation();
        let mut casper = NoOpsCasperEffect::new(
            Some(HashMap::new()),
            None,
            Arc::new(Mutex::new(runtime_manager)),
            block_store,
            dag_representation,
        );

        let (sender, _) = oneshot::channel();

        let result = proposer.propose(&mut casper, false, sender).await;

        match result {
            Ok((propose_result, block_opt)) => {
                use casper::rust::blocks::proposer::propose_result::{
                    CheckProposeConstraintsFailure, ProposeFailure, ProposeStatus,
                };

                assert!(matches!(
                    propose_result.propose_status,
                    ProposeStatus::Failure(ProposeFailure::CheckConstraintsFailure(
                        CheckProposeConstraintsFailure::NotBonded
                    ))
                ));
                assert!(block_opt.is_none());
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    })
    .await
}

#[tokio::test]
async fn proposer_should_reject_to_propose_if_synchrony_constraint_not_met() {
    with_storage(|block_store, block_dag_storage| async move {
        let runtime_manager = mk_runtime_manager("block-query-response-api-test", None).await;
        let validator_identity = Arc::new(dummy_validator_identity());

        let mut proposer = Proposer::new(
            validator_identity,
            None,
            TestCasperSnapshotProvider,
            AlwaysActiveChecker,               // permissive - validator is active
            AlwaysNotEnoughBlocksStakeChecker, // synchrony constraint is not met
            OkHeightChecker,                   // permissive
            TestBlockCreator,
            TestBlockValidator,
            TestProposeEffectHandler,
        );

        use crate::helper::no_ops_casper_effect::NoOpsCasperEffect;
        use std::collections::HashMap;

        let dag_representation = block_dag_storage.get_representation();
        let mut casper = NoOpsCasperEffect::new(
            Some(HashMap::new()),
            None,
            Arc::new(Mutex::new(runtime_manager)),
            block_store,
            dag_representation,
        );

        let (sender, _) = oneshot::channel();

        let result = proposer.propose(&mut casper, false, sender).await;

        match result {
            Ok((propose_result, block_opt)) => {
                use casper::rust::blocks::proposer::propose_result::{
                    CheckProposeConstraintsFailure, ProposeFailure, ProposeStatus,
                };

                assert!(matches!(
                    propose_result.propose_status,
                    ProposeStatus::Failure(ProposeFailure::CheckConstraintsFailure(
                        CheckProposeConstraintsFailure::NotEnoughNewBlocks
                    ))
                ));
                assert!(block_opt.is_none());
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    })
    .await
}

#[tokio::test]
async fn proposer_should_reject_to_propose_if_last_finalized_height_constraint_not_met() {
    with_storage(|block_store, block_dag_storage| async move {
        let runtime_manager = mk_runtime_manager("block-query-response-api-test", None).await;
        let validator_identity = Arc::new(dummy_validator_identity());

        let mut proposer = Proposer::new(
            validator_identity,
            None,
            TestCasperSnapshotProvider,
            AlwaysActiveChecker,             // permissive - validator is active
            OkProposeConstraintStakeChecker, // permissive
            AlwaysTooFarAheadChecker,        // height constraint is not met
            TestBlockCreator,
            TestBlockValidator,
            TestProposeEffectHandler,
        );

        use crate::helper::no_ops_casper_effect::NoOpsCasperEffect;
        use std::collections::HashMap;

        let dag_representation = block_dag_storage.get_representation();
        let mut casper = NoOpsCasperEffect::new(
            Some(HashMap::new()),
            None,
            Arc::new(Mutex::new(runtime_manager)),
            block_store,
            dag_representation,
        );

        let (sender, _) = oneshot::channel();

        let result = proposer.propose(&mut casper, false, sender).await;

        match result {
            Ok((propose_result, block_opt)) => {
                use casper::rust::blocks::proposer::propose_result::{
                    CheckProposeConstraintsFailure, ProposeFailure, ProposeStatus,
                };

                assert!(matches!(
                    propose_result.propose_status,
                    ProposeStatus::Failure(ProposeFailure::CheckConstraintsFailure(
                        CheckProposeConstraintsFailure::TooFarAheadOfLastFinalized
                    ))
                ));
                assert!(block_opt.is_none());
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    })
    .await
}

#[tokio::test]
async fn proposer_should_shut_down_the_node_if_block_created_is_not_successfully_replayed() {
    with_storage(|block_store, block_dag_storage| async move {
        let runtime_manager = mk_runtime_manager("block-query-response-api-test", None).await;
        let validator_identity = Arc::new(dummy_validator_identity());

        let mut proposer = Proposer::new(
            validator_identity,
            None,
            TestCasperSnapshotProvider,
            AlwaysActiveChecker,             // permissive - validator is active
            OkProposeConstraintStakeChecker, // permissive
            OkHeightChecker,                 // permissive
            TestBlockCreator,                // creates a block
            AlwaysUnsuccessfulValidator,     // validation fails
            TestProposeEffectHandler,        // handles effects
        );

        use crate::helper::no_ops_casper_effect::NoOpsCasperEffect;
        use std::collections::HashMap;

        let dag_representation = block_dag_storage.get_representation();
        let mut casper = NoOpsCasperEffect::new(
            Some(HashMap::new()),
            None,
            Arc::new(Mutex::new(runtime_manager)),
            block_store,
            dag_representation,
        );

        let (sender, _) = oneshot::channel();

        let result = proposer.propose(&mut casper, false, sender).await;

        // Should return an error when block validation fails
        match result {
            Ok(_) => panic!("Expected error when block validation fails"),
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(error_msg.contains("Validation of self created block failed"));
            }
        }
    })
    .await
}

#[tokio::test]
async fn proposer_should_execute_propose_effects_if_block_created_successfully_replayed() {
    with_storage(|block_store, block_dag_storage| async move {
        // Reset the effect variable before test
        reset_propose_effect_var();

        let runtime_manager = mk_runtime_manager("block-query-response-api-test", None).await;
        let validator_identity = Arc::new(dummy_validator_identity());

        let mut proposer = Proposer::new(
            validator_identity,
            None,
            TestCasperSnapshotProvider,
            AlwaysActiveChecker,             // permissive - validator is active
            OkProposeConstraintStakeChecker, // permissive
            OkHeightChecker,                 // permissive
            TestBlockCreator,                // creates a block
            TestBlockValidator,              // validates successfully
            TrackingProposeEffectHandler::new(10), // tracks effects with value 10
        );

        use crate::helper::no_ops_casper_effect::NoOpsCasperEffect;
        use std::collections::HashMap;

        let dag_representation = block_dag_storage.get_representation();
        let mut casper = NoOpsCasperEffect::new(
            Some(HashMap::new()),
            None,
            Arc::new(Mutex::new(runtime_manager)),
            block_store,
            dag_representation,
        );

        let (sender, _) = oneshot::channel();

        let result = proposer.propose(&mut casper, false, sender).await;

        match result {
            Ok((propose_result, block_opt)) => {
                use casper::rust::block_status::ValidBlock;
                use casper::rust::blocks::proposer::propose_result::{
                    ProposeStatus, ProposeSuccess,
                };

                assert!(matches!(
                    propose_result.propose_status,
                    ProposeStatus::Success(ProposeSuccess {
                        result: ValidBlock::Valid
                    })
                ));
                assert!(block_opt.is_some());
                // Verify that the propose effect was executed
                assert_eq!(get_propose_effect_var(), 10);
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    })
    .await
}
