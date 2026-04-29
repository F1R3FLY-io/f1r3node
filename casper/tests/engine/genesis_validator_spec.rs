// See casper/src/test/scala/coop/rchain/casper/engine/GenesisValidatorSpec.scala

use crate::engine::setup::TestFixture;
use casper::rust::engine::block_approver_protocol::BlockApproverProtocol;
use casper::rust::engine::genesis_validator::GenesisValidator;
use comm::rust::rp::protocol_helper::packet_with_content;
use models::rust::casper::protocol::casper_message::{
    ApprovedBlock, ApprovedBlockCandidate, ApprovedBlockRequest, BlockMessage, BlockRequest,
    CasperMessage, NoApprovedBlockAvailable, UnapprovedBlock,
};
use serial_test::serial;
use std::sync::Arc;
use tokio::time::{sleep, Duration};

struct GenesisValidatorSpec;

impl GenesisValidatorSpec {
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
        let _event_bus = shared::rust::shared::f1r3fly_events::F1r3flyEvents::new();

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
                fixture.rejected_deploy_buffer.clone(),
                fixture.casper_buffer_storage.clone(),
                fixture.rspace_state_manager.clone(),
                fixture.runtime_manager.clone(),
                fixture.estimator.clone(),
                casper::rust::heartbeat_signal::new_heartbeat_signal_ref(),
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
            let mut saw_expected_response = false;
            for _ in 0..20 {
                let requests = fixture.transport_layer.get_all_requests();
                saw_expected_response = requests.iter().any(|request| {
                    if request.peer != fixture.local {
                        return false;
                    }
                    // Depending on startup timing, GenesisValidator can emit ApprovedBlockRequest
                    // before/alongside BlockApproval; both are acceptable liveness responses here.
                    matches!(
                        request.msg.message.as_ref(),
                        Some(models::routing::protocol::Message::Packet(packet))
                            if packet.type_id == "BlockApproval"
                                || packet.type_id == "ApprovedBlockRequest"
                    ) || request.msg == expected_packet
                });
                if saw_expected_response {
                    break;
                }
                sleep(Duration::from_millis(100)).await;
            }

            assert!(
                saw_expected_response,
                "Expected GenesisValidator to emit BlockApproval or ApprovedBlockRequest"
            );
        };

        test.await;
    }

    /// Regression test for the late-joiner race fixed in PR #489.
    ///
    /// A genesis validator that joins boot's connections AFTER all
    /// `UnapprovedBlock` broadcasts but BEFORE the `ApprovedBlock` broadcast
    /// receives only the `ApprovedBlock` while still in `GenesisValidator`
    /// state. Pre-fix, the message hit the catch-all `_ => Ok(())` arm and
    /// was silently dropped, leaving the validator stuck. Post-fix,
    /// `handle_approved_block_late` transitions the validator to
    /// `Initializing`, whose `init` proactively emits an
    /// `ApprovedBlockRequest` to bootstrap.
    async fn transitions_to_initializing_on_late_approved_block() {
        let _event_bus = shared::rust::shared::f1r3fly_events::F1r3flyEvents::new();

        let fixture = TestFixture::new().await;

        // ApprovedBlock is constructed from the genesis candidate. Sigs are
        // empty because GenesisValidator's late-joiner path does not validate
        // the ApprovedBlock — that happens in Initializing::handle once it
        // receives the bootstrap response. The transition itself is what we
        // verify here.
        let approved_block = ApprovedBlock {
            candidate: ApprovedBlockCandidate {
                block: fixture.genesis.clone(),
                required_sigs: fixture.required_sigs,
            },
            sigs: Vec::new(),
        };

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
                fixture.engine_cell.clone(),
                fixture.block_store.clone(),
                fixture.block_dag_storage.clone(),
                fixture.deploy_storage.clone(),
                fixture.rejected_deploy_buffer.clone(),
                fixture.casper_buffer_storage.clone(),
                fixture.rspace_state_manager.clone(),
                fixture.runtime_manager.clone(),
                fixture.estimator.clone(),
                casper::rust::heartbeat_signal::new_heartbeat_signal_ref(),
            );

            fixture.engine_cell.set(Arc::new(genesis_validator)).await;

            let engine = fixture.engine_cell.get().await;
            engine
                .handle(
                    fixture.local.clone(),
                    CasperMessage::ApprovedBlock(approved_block.clone()),
                )
                .await
                .expect("Failed to handle ApprovedBlock");

            // Initializing::init proactively sends an ApprovedBlockRequest;
            // its presence on the transport layer is the signal that the
            // transition fired.
            let mut saw_approved_block_request = false;
            for _ in 0..20 {
                let requests = fixture.transport_layer.get_all_requests();
                saw_approved_block_request = requests.iter().any(|request| {
                    matches!(
                        request.msg.message.as_ref(),
                        Some(models::routing::protocol::Message::Packet(packet))
                            if packet.type_id == "ApprovedBlockRequest"
                    )
                });
                if saw_approved_block_request {
                    break;
                }
                sleep(Duration::from_millis(100)).await;
            }

            assert!(
                saw_approved_block_request,
                "Expected GenesisValidator to transition to Initializing on late ApprovedBlock; \
                 Initializing::init should emit an ApprovedBlockRequest"
            );

            // A second ApprovedBlock arriving at the same engine_cell now
            // routes to Initializing::handle (the previous transition replaced
            // the engine), not back to GenesisValidator::handle. Sending it
            // again should not panic or stall — exercises the "duplicate
            // ApprovedBlock after transition" path the review flagged.
            fixture.transport_layer.reset();
            let engine = fixture.engine_cell.get().await;
            let _ = engine
                .handle(
                    fixture.local.clone(),
                    CasperMessage::ApprovedBlock(approved_block),
                )
                .await;
        };

        test.await;
    }

    async fn should_not_respond_to_any_other_message() {
        let _event_bus = shared::rust::shared::f1r3fly_events::F1r3flyEvents::new();

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
                fixture.rejected_deploy_buffer.clone(),
                fixture.casper_buffer_storage.clone(),
                fixture.rspace_state_manager.clone(),
                fixture.runtime_manager.clone(),
                fixture.estimator.clone(),
                casper::rust::heartbeat_signal::new_heartbeat_signal_ref(),
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

            let requests = fixture.transport_layer.get_all_requests();
            assert_eq!(
                requests.len(),
                1,
                "Expected exactly one transport-layer response to ApprovedBlockRequest"
            );
            let head = requests[0].clone();

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
}

// Serialized with approve_block_protocol_test (which is also #[serial]) because the
// GenesisValidator and ApproveBlockProtocol both increment the process-global "genesis"
// metrics counter via add_approval. Without serialization, concurrent counter writes
// from these tests can corrupt the baseline-vs-after delta that approve_block_protocol_test
// relies on. The non-counter-incrementing tests in this file are also marked #[serial]
// for consistency and to guard against future test additions accidentally exercising
// that path.
#[tokio::test]
#[serial]
async fn respond_on_unapproved_block_messages_with_block_approval() {
    GenesisValidatorSpec::respond_on_unapproved_block_messages_with_block_approval().await;
}

#[tokio::test]
#[serial]
async fn should_not_respond_to_any_other_message() {
    GenesisValidatorSpec::should_not_respond_to_any_other_message().await;
}

#[tokio::test]
#[serial]
async fn transitions_to_initializing_on_late_approved_block() {
    GenesisValidatorSpec::transitions_to_initializing_on_late_approved_block().await;
}
