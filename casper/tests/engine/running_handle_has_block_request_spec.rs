// See casper/src/test/scala/coop/rchain/casper/engine/RunningHandleHasBlockRequestSpec.scala

use crate::engine::setup::TestFixture;
use comm::rust::peer_node::{Endpoint, NodeIdentifier, PeerNode};
use models::{
    casper::HasBlockProto,
    routing::{protocol::Message::Packet, Protocol},
    rust::{
        block_hash::BlockHash,
        casper::protocol::casper_message::{HasBlock, HasBlockRequest},
    },
};
use prost::{bytes::Bytes, Message};

const HASH_BYTES: &[u8] = b"hash";

struct TestContext {
    hbr: HasBlockRequest,
    // Note: Using full TestFixture for convenience, though this test only needs
    // engine and transport_layer. The overhead is acceptable for test simplicity.
    fixture: TestFixture,
}

impl TestContext {
    async fn new() -> Self {
        let hash = Bytes::from(HASH_BYTES.to_vec());
        let hbr = HasBlockRequest { hash: hash.clone() };
        let fixture = TestFixture::new().await;

        Self { hbr, fixture }
    }

    fn endpoint(port: u16) -> Endpoint {
        Endpoint {
            host: "host".to_string(),
            tcp_port: port as u32,
            udp_port: port as u32,
        }
    }

    fn peer_node(name: &str, port: u16) -> PeerNode {
        PeerNode {
            id: NodeIdentifier {
                key: Bytes::from(name.as_bytes().to_vec()),
            },
            endpoint: Self::endpoint(port),
        }
    }

    fn to_has_block(protocol: &Protocol) -> HasBlock {
        if let Some(message) = &protocol.message {
            if let Packet(packet_data) = message {
                if let Ok(hb) = HasBlockProto::decode(packet_data.content.as_ref()) {
                    return HasBlock::from_proto(hb);
                }
            }
        }
        panic!("Could not convert protocol to HasBlock");
    }

    // Scala: private def alwaysSuccess: PeerNode => Protocol => CommErr[Unit] = kp(kp(Right(())))
    // Note: Not ported because TransportLayerTestImpl doesn't require setting success responses.

    // Scala: override def beforeEach(): Unit
    // Note: Not ported because in Rust we create a fresh TestContext for each test,
    // eliminating the need for explicit setup/teardown.
}

#[tokio::test]
async fn running_handle_has_block_request_if_given_block_is_stored_should_send_back_has_block_message_to_the_sender(
) {
    let ctx = TestContext::new().await;
    // given
    let sender = TestContext::peer_node("peer", 40400);
    let block_lookup = |_hash: BlockHash| true;
    // then
    ctx.fixture
        .engine
        .handle_has_block_request(sender.clone(), ctx.hbr.clone(), block_lookup)
        .await
        .unwrap();
    // then
    let (peer, protocol_msg) = ctx
        .fixture
        .transport_layer
        .get_request(0)
        .expect("No request found");

    assert_eq!(peer, sender);

    let has_block = TestContext::to_has_block(&protocol_msg);
    assert_eq!(has_block.hash, Bytes::from(HASH_BYTES.to_vec()));

    assert_eq!(ctx.fixture.transport_layer.request_count(), 1);
}

#[tokio::test]
async fn running_handle_has_block_request_if_given_block_is_not_stored_in_block_store_should_do_nothing(
) {
    let ctx = TestContext::new().await;
    // given
    let sender = TestContext::peer_node("peer", 40400);

    let block_lookup = |_hash: BlockHash| false;
    // then
    ctx.fixture
        .engine
        .handle_has_block_request(sender.clone(), ctx.hbr.clone(), block_lookup)
        .await
        .unwrap();
    // then
    assert_eq!(
        ctx.fixture.transport_layer.request_count(),
        0,
        "Should have no messages"
    );
}
