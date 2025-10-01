// See casper/src/test/scala/coop/rchain/casper/engine/GenesisValidatorSpec.scala
use shared::rust::shared::f1r3fly_events::{EventPublisher, EventPublisherFactory};

use crate::engine::setup::TestFixture;
use casper::rust::engine::engine::Engine;
use casper::rust::engine::engine_cell::EngineCell;
use casper::rust::engine::genesis_validator::GenesisValidator;
use models::rust::casper::protocol::casper_message::{ApprovedBlockCandidate, UnapprovedBlock, BlockMessage};
use comm::rust::rp::connect::{Connections, ConnectionsCell};
use std::sync::{Arc, Mutex};

struct GenesisValidatorSpec;

impl GenesisValidatorSpec {
    fn event_bus() -> Box<dyn EventPublisher> {
        EventPublisherFactory::noop()
    }

    // TODO should be moved to Rust BlockApproverProtocolTest.createUnapproved, when BlockApproverProtocolTest will be created
    fn create_unapproved(required_sigs: i32, block: &BlockMessage) -> UnapprovedBlock {
        UnapprovedBlock {
            candidate: ApprovedBlockCandidate {
                block: block.clone(),
                required_sigs,
            },
            timestamp: 0,
            duration: 0,
        }
    }

    async fn respond_on_unapproved_block_messages_with_block_approval() {
        // TODO: Implement after fixing genesis_validator parameters
        // let _event_bus = Self::event_bus();
        // let fixture = TestFixture::new().await;
        // ...
        unimplemented!("Test temporarily disabled until GenesisValidator::new() parameters are filled in")
    }

    async fn should_not_respond_to_any_other_message() {
        // TODO: Implement after fixing genesis_validator parameters
        unimplemented!("Test temporarily disabled until GenesisValidator::new() parameters are filled in")
    }
}

// Temporarily commented out until we fill in GenesisValidator::new() parameters
// #[tokio::test]
// async fn respond_on_unapproved_block_messages_with_block_approval() {
//     GenesisValidatorSpec::respond_on_unapproved_block_messages_with_block_approval().await;
// }

// #[tokio::test]
// async fn should_not_respond_to_any_other_message() {
//     GenesisValidatorSpec::should_not_respond_to_any_other_message().await;
// }
