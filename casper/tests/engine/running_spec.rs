// See casper/src/test/scala/coop/rchain/casper/engine/RunningSpec.scala

use casper::rust::{casper::MultiParentCasper, engine::engine::Engine};
use models::rust::{
    block_implicits::get_random_block,
    casper::protocol::casper_message::{
        BlockRequest, CasperMessage, ForkChoiceTipRequest, HasBlock,
    },
};
use prost::bytes::Bytes;
use std::collections::HashSet;

use crate::engine::setup::{to_casper_message, TestFixture};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn engine_should_enqueue_block_message_for_processing() {
        let fixture = TestFixture::new().await;
        let block_message = get_random_block(
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        );

        let signed_block = fixture.validator_id.sign_block(&block_message);

        fixture
            .engine
            .handle(
                fixture.local.clone(),
                CasperMessage::BlockMessage(signed_block.clone()),
            )
            .await
            .unwrap();

        // Verify the block was enqueued for processing (following Scala test behavior)
        // This matches the Scala test pattern: getRandomBlock() -> signBlock() -> handle() -> check queue
        assert!(
            fixture
                .is_block_in_processing_queue(&signed_block.block_hash)
                .await,
            "Block should be enqueued in processing queue after being handled"
        );
    }

    #[tokio::test]
    async fn engine_should_respond_to_block_request() {
        let fixture = TestFixture::new().await;

        // Scala: blockStore.put(genesis.blockHash, genesis) (line 79)
        // Insert genesis block into store before testing BlockRequest response
        let genesis = fixture.genesis.clone();
        fixture
            .block_store
            .put(genesis.block_hash.clone(), &genesis)
            .expect("Failed to put genesis block");

        let block_request = BlockRequest {
            hash: genesis.block_hash.clone(),
        };

        fixture
            .engine
            .handle(
                fixture.local.clone(),
                CasperMessage::BlockRequest(block_request),
            )
            .await
            .unwrap();

        assert_eq!(fixture.transport_layer.request_count(), 1);
        let sent_request = fixture.transport_layer.pop_request().unwrap();
        assert_eq!(sent_request.peer, fixture.local);
        if let CasperMessage::BlockMessage(sent_msg) = to_casper_message(sent_request.msg) {
            assert_eq!(sent_msg, genesis);
        } else {
            panic!("Expected BlockMessage");
        }
    }

    #[tokio::test]
    async fn engine_should_respond_to_approved_block_request() {
        let fixture = TestFixture::new().await;

        // Scala: Similar to BlockRequest test, genesis needs to be in store
        // Insert genesis block into store before testing ApprovedBlockRequest response
        let genesis_block = fixture.genesis.clone();
        fixture
            .block_store
            .put(genesis_block.block_hash.clone(), &genesis_block)
            .expect("Failed to put genesis block");

        let approved_block_request =
            models::rust::casper::protocol::casper_message::ApprovedBlockRequest {
                identifier: "test".to_string(),
                trim_state: false,
            };
        let expected_approved_block =
            models::rust::casper::protocol::casper_message::ApprovedBlock {
                candidate: models::rust::casper::protocol::casper_message::ApprovedBlockCandidate {
                    block: genesis_block,
                    required_sigs: 0,
                },
                sigs: Vec::new(),
            };

        fixture
            .engine
            .handle(
                fixture.local.clone(),
                CasperMessage::ApprovedBlockRequest(approved_block_request),
            )
            .await
            .unwrap();

        assert_eq!(fixture.transport_layer.request_count(), 1);
        let sent_request = fixture.transport_layer.pop_request().unwrap();
        assert_eq!(sent_request.peer, fixture.local);
        if let CasperMessage::ApprovedBlock(sent_msg) = to_casper_message(sent_request.msg) {
            assert_eq!(sent_msg, expected_approved_block);
        } else {
            panic!("Expected ApprovedBlock");
        }
    }

    #[tokio::test]
    async fn engine_should_respond_to_fork_choice_tip_request() {
        let mut fixture = TestFixture::new().await;

        // Step 1: Create a request object
        let request = ForkChoiceTipRequest {};

        // Step 2: Create 2 blocks with empty sender
        let mut block1 = get_random_block(
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        );
        block1.sender = Bytes::new(); // Empty sender

        let mut block2 = get_random_block(
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        );
        block2.sender = Bytes::new(); // Empty sender

        // Step 3: Insert blocks in blockDagStorage (following Scala implementation)
        // This matches the Scala pattern: blockDagStorage.insert(block1, false)
        fixture.casper.insert_block(block1.clone(), false);
        fixture.casper.insert_block(block2.clone(), false);

        // Step 4: Get tips from casper.blockDag (this happens inside the engine)
        let dag = fixture.casper.block_dag().await.unwrap();
        let tips_from_dag: Vec<_> = dag
            .latest_messages_map
            .iter()
            .map(|entry| entry.value().clone())
            .collect();

        // Step 5: Call engine.handle with local peer and request object
        fixture
            .engine
            .handle(
                fixture.local.clone(),
                CasperMessage::ForkChoiceTipRequest(request),
            )
            .await
            .unwrap();

        // Step 6: Get requests from transportLayer
        let requests = fixture.transport_layer.get_all_requests();

        // Step 7: Create Expected Tip value
        let expected_tips: HashSet<_> = tips_from_dag.into_iter().collect();

        // Step 8: Assert peer in head in requests in transport layer is local
        assert!(!requests.is_empty());
        let first_request = &requests[0];
        assert_eq!(first_request.peer, fixture.local);
        let second_request = &requests[1];
        assert_eq!(second_request.peer, fixture.local);

        // Step 9: Assert requests matches to expected tips value
        let mut received_tips = HashSet::new();
        for request in requests {
            if let CasperMessage::HasBlock(HasBlock { hash }) = to_casper_message(request.msg) {
                received_tips.insert(hash);
            }
        }

        assert_eq!(received_tips, expected_tips);
    }
}
