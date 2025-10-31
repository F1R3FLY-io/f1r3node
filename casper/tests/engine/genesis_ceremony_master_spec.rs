// See casper/src/test/scala/coop/rchain/casper/engine/GenesisCeremonyMasterSpec.scala

use crate::engine::approve_block_protocol_test::create_approval;
use crate::engine::setup::TestFixture;
use casper::rust::engine::approve_block_protocol::ApproveBlockProtocolFactory;
use casper::rust::engine::engine_cell::EngineCell;
use casper::rust::engine::genesis_ceremony_master::GenesisCeremonyMaster;
use comm::rust::test_instances::TransportLayerStub;
use models::casper::ApprovedBlockProto;
use models::routing::protocol::Message;
use models::rust::casper::protocol::casper_message::{
    ApprovedBlock, ApprovedBlockCandidate, CasperMessage,
};
use prost::Message as ProstMessage;
use shared::rust::shared::metrics_test::MetricsTestImpl;
use std::sync::Arc;
use std::time::Duration;

struct GenesisCeremonyMasterSpec;

impl GenesisCeremonyMasterSpec {
    async fn make_transition_to_running_state_after_block_approved() {
        // NOTE: LocalSet is required because the_init closure in ApproveBlockProtocol
        // captures !Send types. In Scala, Task doesn't require Send, but Rust tokio::spawn does.
        // LocalSet allows running !Send futures on a single thread.
        let local = tokio::task::LocalSet::new();

        local.run_until(async {
            let fixture = TestFixture::new().await;

            let required_sigs = 0;

            // interval and duration don't really matter since we don't require and signs from validators
            let interval = Duration::from_millis(1);
            let duration = Duration::from_secs(1);
            let start_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;

            let engine_cell = Arc::new(EngineCell::init());

            async fn wait_until_casper_is_defined(
                engine_cell: &Arc<EngineCell>,
            ) {
                let engine = engine_cell.get().await;

                match engine.with_casper() {
                    Some(_casper) => {},
                    None => {
                        tokio::time::sleep(Duration::from_secs(3)).await;
                        Box::pin(wait_until_casper_is_defined(engine_cell)).await
                    }
                }
            }

            let test = async {
                let abp = ApproveBlockProtocolFactory::unsafe_new_with_infrastructure(
                    fixture.genesis.clone(),
                    required_sigs,
                    duration,
                    interval,
                    Arc::new(MetricsTestImpl::new()),
                    fixture.event_publisher.clone(),
                    fixture.transport_layer.clone(),
                    Some(Arc::new(fixture.connections_cell.clone())),
                    Some(Arc::new(fixture.rp_conf_ask.clone())),
                    fixture.last_approved_block.clone(),
                );

                // TODO When ApproveBlockProtocolImpl gets Clone derive, we can:
                // - Change GenesisCeremonyMaster::new() to accept ApproveBlockProtocolImpl directly
                // - Clone abp for c1: let abp_for_run = abp.clone()
                // - Remove this Arc wrapper
                let abp_arc = Arc::new(abp);

                let genesis_ceremony_master = GenesisCeremonyMaster::new(abp_arc.clone());
                engine_cell
                    .set(Arc::new(genesis_ceremony_master))
                    .await;

                // с1
                let abp_for_run = abp_arc.clone();
                tokio::task::spawn_local(async move {
                    if let Err(e) = abp_for_run.run().await {
                        log::error!("approve_protocol.run() failed: {:?}", e);
                    }
                });

                // с2
                let engine_cell_for_loop = engine_cell.clone();
                let transport_layer = fixture.transport_layer.clone();
                let rp_conf_ask = fixture.rp_conf_ask.clone();
                let connections_cell = fixture.connections_cell.clone();
                let last_approved_block = fixture.last_approved_block.clone();
                let event_publisher = fixture.event_publisher.clone();
                let block_retriever = fixture.block_retriever.clone();
                let block_store = fixture.block_store.clone();
                let block_dag_storage = fixture.block_dag_storage.clone();
                let deploy_storage = fixture.deploy_storage.clone();
                let casper_buffer_storage = fixture.casper_buffer_storage.clone();
                let runtime_manager = fixture.runtime_manager.clone();
                let estimator = fixture.estimator.clone();
                let block_processing_queue = fixture.block_processing_queue.clone();
                let blocks_in_processing = fixture.blocks_in_processing.clone();
                let casper_shard_conf = fixture.casper_shard_conf.clone();
                let validator_id = Some(fixture.validator_id.clone());

                tokio::task::spawn_local(async move {
                    if let Err(e) = GenesisCeremonyMaster::<TransportLayerStub>::waiting_for_approved_block_loop(
                        transport_layer,
                        rp_conf_ask,
                        connections_cell,
                        last_approved_block,
                        &event_publisher,
                        block_retriever,
                        engine_cell_for_loop,
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
                        true,
                    )
                    .await
                    {
                        log::error!("waitingForApprovedBlockLoop failed: {:?}", e);
                    }
                });

                let approved_block_candidate = ApprovedBlockCandidate {
                    block: fixture.genesis.clone(),
                    required_sigs,
                };

                // Note: Creating two approvals because BlockApproval doesn't implement Clone (protobuf generated)
                // In Scala, same blockApproval is reused, but Rust requires separate instances
                let block_approval_1 = create_approval(
                    &approved_block_candidate,
                    &fixture.validator_sk,
                    &fixture.validator_pk,
                );
                let block_approval_2 = create_approval(
                    &approved_block_candidate,
                    &fixture.validator_sk,
                    &fixture.validator_pk,
                );

                engine_cell
                    .get()
                    .await
                    .handle(
                        fixture.local.clone(),
                        CasperMessage::BlockApproval(block_approval_1),
                    )
                    .await
                    .expect("Failed to handle block approval");

                let timeout_future = tokio::time::sleep(Duration::from_secs(180));
                let possibly_сasper = wait_until_casper_is_defined(&engine_cell);

                tokio::select! {
                    _ = timeout_future => {
                        panic!("Timeout: Casper was not defined within 3 minutes");
                    }
                    _ = possibly_сasper => {}
                }

                let block_opt = fixture
                    .block_store
                    .lock()
                    .unwrap()
                    .as_ref()
                    .expect("Block store not available")
                    .get(&fixture.genesis.block_hash)
                    .expect("Failed to get block");
                assert!(block_opt.is_some(), "Genesis block should be in BlockStore");
                assert_eq!(block_opt.unwrap(), fixture.genesis);

                let engine_final = engine_cell.get().await;
                assert!(
                    engine_final.with_casper().is_some(),
                    "Engine should be Running with Casper"
                );

                let last_approved_block = fixture.last_approved_block.lock().unwrap().clone();
                assert!(
                    last_approved_block.is_some(),
                    "LastApprovedBlock should be set"
                );

                engine_cell
                    .get()
                    .await
                    .handle(
                        fixture.local.clone(),
                        CasperMessage::BlockApproval(block_approval_2),
                    )
                    .await
                    .expect("Failed to handle second block approval");

                let head = fixture.transport_layer.get_all_requests();
                assert!(!head.is_empty(), "Transport layer should have requests");

                let proto = ApprovedBlockProto::decode(
                    head[0]
                        .msg
                        .message
                        .as_ref()
                        .and_then(|m| match m {
                            Message::Packet(p) => Some(p.content.as_ref()),
                            _ => None,
                        })
                        .expect("No packet in message"),
                )
                .expect("Failed to decode ApprovedBlockProto");

                let approved_block =
                    ApprovedBlock::from_proto(proto).expect("Failed to parse ApprovedBlock");

                assert_eq!(
                    approved_block.sigs,
                    last_approved_block.as_ref().unwrap().sigs,
                    "ApprovedBlock signatures should match LastApprovedBlock"
                );
        };

        test.await;
        }).await;
    }
}

#[tokio::test]
async fn make_transition_to_running_state_after_block_approved() {
    GenesisCeremonyMasterSpec::make_transition_to_running_state_after_block_approved().await;
}
