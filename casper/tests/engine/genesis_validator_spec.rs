// See casper/src/test/scala/coop/rchain/casper/engine/GenesisValidatorSpec.scala
use shared::rust::shared::f1r3fly_events::{EventPublisher, EventPublisherFactory};

use crate::engine::setup::TestFixture;
use casper::rust::engine::block_approver_protocol::BlockApproverProtocol;
use casper::rust::engine::genesis_validator::GenesisValidator;
use comm::rust::rp::protocol_helper::packet_with_content;
use models::rust::casper::protocol::casper_message::{
    ApprovedBlock, ApprovedBlockCandidate, ApprovedBlockRequest, BlockMessage, BlockRequest,
    CasperMessage, NoApprovedBlockAvailable, UnapprovedBlock,
};
use std::sync::Arc;
use std::time::Duration;

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
        let _event_bus = Self::event_bus();

        let fixture = TestFixture::new().await;

        // Scala: implicit val engineCell: EngineCell[Task] = Cell.unsafe[Task, Engine[Task]](Engine.noop)
        // Rust: Use engine_cell from fixture instead of creating a new one
        // TestFixture already creates engine_cell with unsafe_init() (equivalent to Cell.unsafe with Engine.noop)

        let expected_candidate = ApprovedBlockCandidate {
            block: fixture.genesis.clone(),
            required_sigs: fixture.required_sigs,
        };

        let unapproved_block = Self::create_unapproved(fixture.required_sigs, &fixture.genesis);

        let test = async {
            let genesis_validator = GenesisValidator::new(
                fixture.block_processing_queue_tx.clone(),
                fixture.blocks_in_processing.clone(),
                fixture.casper_shard_conf.clone(),
                fixture.validator_id.clone(),
                fixture.bap.clone(),
                fixture.transport_layer.clone(),
                fixture.rp_conf_ask.clone(),
                fixture.connections_cell.clone(),
                fixture.last_approved_block.clone(),
                fixture.event_publisher.clone(),
                fixture.block_retriever.clone(),
                fixture.engine_cell.clone(), // use fixture.engine_cell instead of new one
                fixture.block_store.clone(),
                fixture.block_dag_storage.clone(),
                fixture.deploy_storage.clone(),
                fixture.casper_buffer_storage.clone(),
                fixture.rspace_state_manager.clone(),
                fixture.runtime_manager.clone(),
                fixture.estimator.clone(),
                casper::rust::heartbeat_signal::new_heartbeat_signal_ref(),
                Duration::from_secs(1),
            );

            fixture.engine_cell.set(Arc::new(genesis_validator)).await;

            // Scala: _ <- engineCell.read >>= (_.handle(local, unapprovedBlock))
            let engine = fixture.engine_cell.get().await;
            engine
                .handle(
                    fixture.local.clone(),
                    CasperMessage::UnapprovedBlock(unapproved_block),
                )
                .await
                .expect("Failed to handle unapproved block");

            // Scala: blockApproval = BlockApproverProtocol.getBlockApproval(expectedCandidate, validatorId)
            let block_approval = BlockApproverProtocol::get_block_approval(
                &fixture.bap.clone(),
                &expected_candidate,
            );

            // Scala: expectedPacket = ProtocolHelper.packet(local, networkId, blockApproval.toProto)
            let expected_packet = packet_with_content(
                &fixture.local,
                &fixture.network_id,
                block_approval.to_proto(),
            );

            // Scala: val lastMessage = transportLayer.requests.last
            //        assert(lastMessage.peer == local && lastMessage.msg == expectedPacket)
            let last_message = fixture
                .transport_layer
                .get_all_requests()
                .last()
                .expect("No requests in transport layer")
                .clone();

            assert_eq!(last_message.peer, fixture.local);
            assert_eq!(last_message.msg, expected_packet);
        };

        test.await;
    }

    async fn should_not_respond_to_any_other_message() {
        let _event_bus = Self::event_bus();

        let fixture = TestFixture::new().await;

        // Scala: val approvedBlockRequest = ApprovedBlockRequest("test")
        let approved_block_request = ApprovedBlockRequest {
            identifier: "test".to_string(),
            trim_state: false,
        };

        let test = async {
            // Scala: engineCell.set(new GenesisValidator(...))
            let genesis_validator = GenesisValidator::new(
                fixture.block_processing_queue_tx.clone(),
                fixture.blocks_in_processing.clone(),
                fixture.casper_shard_conf.clone(),
                fixture.validator_id.clone(),
                fixture.bap.clone(),
                fixture.transport_layer.clone(),
                fixture.rp_conf_ask.clone(),
                fixture.connections_cell.clone(),
                fixture.last_approved_block.clone(),
                fixture.event_publisher.clone(),
                fixture.block_retriever.clone(),
                fixture.engine_cell.clone(),
                fixture.block_store.clone(),
                fixture.block_dag_storage.clone(),
                fixture.deploy_storage.clone(),
                fixture.casper_buffer_storage.clone(),
                fixture.rspace_state_manager.clone(),
                fixture.runtime_manager.clone(),
                fixture.estimator.clone(),
                casper::rust::heartbeat_signal::new_heartbeat_signal_ref(),
                Duration::from_secs(1),
            );

            fixture.engine_cell.set(Arc::new(genesis_validator)).await;

            // Scala: engineCell.read >>= (_.handle(local, approvedBlockRequest))
            let engine = fixture.engine_cell.get().await;
            engine
                .handle(
                    fixture.local.clone(),
                    CasperMessage::ApprovedBlockRequest(approved_block_request),
                )
                .await
                .expect("Failed to handle approved block request");

            // head = transportLayer.requests.head
            let head = fixture
                .transport_layer
                .get_all_requests()
                .first()
                .expect("No requests in transport layer")
                .clone();

            let expected_response = packet_with_content(
                &fixture.local,
                &fixture.network_id,
                NoApprovedBlockAvailable {
                    node_identifier: fixture.local.to_string(),
                    identifier: "test".to_string(),
                }
                .to_proto(),
            );

            assert_eq!(head.peer, fixture.local);
            assert_eq!(head.msg, expected_response);

            // Scala: transportLayer.reset()
            fixture.transport_layer.reset();

            // Scala: blockRequest = BlockRequest(ByteString.copyFromUtf8("base16Hash"))
            let block_request = BlockRequest {
                hash: prost::bytes::Bytes::from("base16Hash".as_bytes().to_vec()),
            };

            let engine = fixture.engine_cell.get().await;
            engine
                .handle(
                    fixture.local.clone(),
                    CasperMessage::BlockRequest(block_request),
                )
                .await
                .expect("Failed to handle block request");

            // Verify transport layer has no requests (GenesisValidator doesn't respond to BlockRequest)
            assert!(
                fixture.transport_layer.get_all_requests().is_empty(),
                "GenesisValidator should not respond to BlockRequest"
            );
        };

        test.await;
    }

    ///
    /// When a genesis validator receives an ApprovedBlock (ceremony already completed
    /// on another path), it should transition to Initializing to sync approved state.
    async fn transition_to_initializing_when_approved_block_received() {
        let _event_bus = Self::event_bus();

        let fixture = TestFixture::new().await;

        let test = async {
            // Build a valid ApprovedBlock from the genesis block
            let approved_block_candidate = ApprovedBlockCandidate {
                block: fixture.genesis.clone(),
                required_sigs: 0,
            };
            // The handler doesn't validate signatures — it just transitions.
            let approved_block = ApprovedBlock {
                candidate: approved_block_candidate,
                sigs: vec![],
            };

            let genesis_validator = GenesisValidator::new(
                fixture.block_processing_queue_tx.clone(),
                fixture.blocks_in_processing.clone(),
                fixture.casper_shard_conf.clone(),
                fixture.validator_id.clone(),
                fixture.bap.clone(),
                fixture.transport_layer.clone(),
                fixture.rp_conf_ask.clone(),
                fixture.connections_cell.clone(),
                fixture.last_approved_block.clone(),
                fixture.event_publisher.clone(),
                fixture.block_retriever.clone(),
                fixture.engine_cell.clone(),
                fixture.block_store.clone(),
                fixture.block_dag_storage.clone(),
                fixture.deploy_storage.clone(),
                fixture.casper_buffer_storage.clone(),
                fixture.rspace_state_manager.clone(),
                fixture.runtime_manager.clone(),
                fixture.estimator.clone(),
                casper::rust::heartbeat_signal::new_heartbeat_signal_ref(),
                Duration::from_secs(1),
            );

            fixture.engine_cell.set(Arc::new(genesis_validator)).await;

            // Record the engine reference before handling ApprovedBlock
            let engine_before = fixture.engine_cell.get().await;

            // Handle ApprovedBlock — should transition to Initializing
            engine_before
                .handle(
                    fixture.local.clone(),
                    CasperMessage::ApprovedBlock(approved_block),
                )
                .await
                .expect("Failed to handle approved block");

            // Verify engine transitioned (engine cell was updated with a new engine instance)
            let engine_after = fixture.engine_cell.get().await;
            assert!(
                !Arc::ptr_eq(&engine_before, &engine_after),
                "Engine should have transitioned to Initializing after receiving ApprovedBlock"
            );
        };

        test.await;
    }

    ///
    /// When GenesisValidator.init() is called, it spawns a background task that
    /// periodically requests ApprovedBlock from bootstrap. This test verifies that
    /// the background task sends an ApprovedBlockRequest after the initial delay.
    async fn request_approved_block_from_bootstrap_after_initial_delay() {
        let _event_bus = Self::event_bus();

        let fixture = TestFixture::new().await;

        // Default TransportLayerStub already returns Ok(()) for send()

        let test = async {
            // Use a short approve_interval for test speed
            let approve_interval = Duration::from_millis(100);

            let genesis_validator = GenesisValidator::new(
                fixture.block_processing_queue_tx.clone(),
                fixture.blocks_in_processing.clone(),
                fixture.casper_shard_conf.clone(),
                fixture.validator_id.clone(),
                fixture.bap.clone(),
                fixture.transport_layer.clone(),
                fixture.rp_conf_ask.clone(),
                fixture.connections_cell.clone(),
                fixture.last_approved_block.clone(),
                fixture.event_publisher.clone(),
                fixture.block_retriever.clone(),
                fixture.engine_cell.clone(),
                fixture.block_store.clone(),
                fixture.block_dag_storage.clone(),
                fixture.deploy_storage.clone(),
                fixture.casper_buffer_storage.clone(),
                fixture.rspace_state_manager.clone(),
                fixture.runtime_manager.clone(),
                fixture.estimator.clone(),
                casper::rust::heartbeat_signal::new_heartbeat_signal_ref(),
                approve_interval,
            );

            fixture
                .engine_cell
                .set(Arc::new(genesis_validator))
                .await;

            // Call init — starts background task requesting ApprovedBlock.
            let engine = fixture.engine_cell.get().await;
            engine
                .init()
                .await
                .expect("Failed to init genesis validator");

            // Poll for the request — allow up to 5s for the initial delay + request
            let max_attempts = 50;
            let poll_interval = Duration::from_millis(100);

            for attempt in 0..max_attempts {
                let requests = fixture.transport_layer.get_all_requests();
                if !requests.is_empty() {
                    // Request appeared — test passes
                    return;
                }
                if attempt == max_attempts - 1 {
                    panic!(
                        "GenesisValidator.init should send ApprovedBlockRequest within timeout. \
                         No requests sent after {} attempts.",
                        max_attempts
                    );
                }
                tokio::time::sleep(poll_interval).await;
            }
        };

        test.await;
    }
}

#[tokio::test]
async fn respond_on_unapproved_block_messages_with_block_approval() {
    GenesisValidatorSpec::respond_on_unapproved_block_messages_with_block_approval().await;
}

#[tokio::test]
async fn should_not_respond_to_any_other_message() {
    GenesisValidatorSpec::should_not_respond_to_any_other_message().await;
}

#[tokio::test]
async fn transition_to_initializing_when_approved_block_received() {
    GenesisValidatorSpec::transition_to_initializing_when_approved_block_received().await;
}

#[tokio::test]
async fn request_approved_block_from_bootstrap_after_initial_delay() {
    GenesisValidatorSpec::request_approved_block_from_bootstrap_after_initial_delay().await;
}
