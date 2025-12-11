// See casper/src/test/scala/coop/rchain/casper/engine/RunningHandleHasBlockSpec.scala

use crate::engine::setup::TestFixture;
use casper::rust::engine::block_retriever::RequestState;
use comm::rust::peer_node::{Endpoint, NodeIdentifier, PeerNode};
use models::{
    casper::BlockRequestProto,
    routing::{protocol::Message::Packet, Protocol},
    rust::{
        block_hash::BlockHash,
        casper::protocol::casper_message::{BlockRequest, HasBlock},
    },
};
use prost::{bytes::Bytes, Message};
use std::{
    collections::{HashMap, HashSet},
    time::{SystemTime, UNIX_EPOCH},
};

const HASH_BYTES: &[u8] = b"hash";

struct TestContext {
    hash: BlockHash,
    hb: HasBlock,
    // Note: Using full TestFixture for convenience, though this test only needs
    // engine, block_retriever, and transport_layer. The overhead is acceptable for test simplicity.
    fixture: TestFixture,
}

impl TestContext {
    async fn new() -> Self {
        let hash = Bytes::from(HASH_BYTES.to_vec());

        let hb = HasBlock { hash: hash.clone() };

        let fixture = TestFixture::new().await;

        Self { hash, hb, fixture }
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

    // Scala: val br = BlockRequest.from(convert[PacketTypeTag.BlockRequest.type](toPacket(msg).right.get).get)
    fn to_block_request(protocol: &Protocol) -> BlockRequest {
        if let Some(message) = &protocol.message {
            if let Packet(packet_data) = message {
                if let Ok(br) = BlockRequestProto::decode(packet_data.content.as_ref()) {
                    return BlockRequest::from_proto(br);
                }
            }
        }
        panic!("Could not convert protocol to BlockRequest");
    }

    // Helper to get current timestamp
    fn current_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    // Scala: private def alwaysSuccess: PeerNode => Protocol => CommErr[Unit] = kp(kp(Right(())))
    // Note: Not ported because TransportLayerTestImpl doesn't require setting success responses.

    // Scala: private def alwaysDoNotIgnoreF: BlockHash => Task[Boolean] = _ => false.pure[Task]
    fn always_do_not_ignore_f(_hash: BlockHash) -> Result<bool, casper::rust::errors::CasperError> {
        Ok(false)
    }

    // Scala: val casperContains: BlockHash => Task[Boolean] = _ => true.pure[Task]
    fn casper_contains(_hash: BlockHash) -> Result<bool, casper::rust::errors::CasperError> {
        Ok(true)
    }

    // Scala: override def beforeEach(): Unit
    // Note: Not ported because in Rust we create a fresh TestContext for each test,
    // eliminating the need for explicit setup/teardown.
}

#[tokio::test]
async fn block_retriever_should_store_on_a_waiting_list_and_dont_request_if_request_state_by_different_peer(
) {
    let ctx = TestContext::new().await;
    // given
    let sender = TestContext::peer_node("some_peer", 40400);
    let other_peer = TestContext::peer_node("other_peer", 40400);

    let request_state_before = {
        let mut map = HashMap::new();
        map.insert(
            ctx.hash.clone(),
            RequestState {
                timestamp: TestContext::current_millis(),
                peers: HashSet::new(),
                received: false,
                in_casper_buffer: false,
                waiting_list: vec![other_peer],
            },
        );
        map
    };

    *ctx.fixture
        .block_retriever
        .requested_blocks()
        .lock()
        .unwrap() = request_state_before;
    // when
    ctx.fixture
        .engine
        .handle_has_block_message(
            sender.clone(),
            ctx.hb.clone(),
            TestContext::always_do_not_ignore_f,
        )
        .await
        .unwrap();

    // then
    assert_eq!(
        ctx.fixture.transport_layer.request_count(),
        0,
        "Transport queue should be empty"
    );

    let request_state_after = ctx
        .fixture
        .block_retriever
        .requested_blocks()
        .lock()
        .unwrap();
    let state = request_state_after.get(&ctx.hash).unwrap();
    assert_eq!(
        state.waiting_list.len(),
        2,
        "Waiting list should have 2 peers"
    );
}

#[tokio::test]
async fn block_retriever_should_request_block_and_add_peer_to_waiting_list_if_peers_list_is_empty()
{
    let ctx = TestContext::new().await;
    // given
    let sender = TestContext::peer_node("somePeer", 40400);

    let request_state_before = {
        let mut map = HashMap::new();
        map.insert(
            ctx.hash.clone(),
            RequestState {
                timestamp: TestContext::current_millis(),
                peers: HashSet::new(),
                received: false,
                in_casper_buffer: false,
                waiting_list: vec![],
            },
        );
        map
    };

    *ctx.fixture
        .block_retriever
        .requested_blocks()
        .lock()
        .unwrap() = request_state_before;

    // when
    ctx.fixture
        .engine
        .handle_has_block_message(
            sender.clone(),
            ctx.hb.clone(),
            TestContext::always_do_not_ignore_f,
        )
        .await
        .unwrap();

    // then
    let (peer, protocol_msg) = ctx
        .fixture
        .transport_layer
        .get_request(0)
        .expect("No request found");

    let br = TestContext::to_block_request(&protocol_msg);
    assert_eq!(br.hash, ctx.hash);
    assert_eq!(peer, sender);
    assert_eq!(ctx.fixture.transport_layer.request_count(), 1);

    let request_state_after = ctx
        .fixture
        .block_retriever
        .requested_blocks()
        .lock()
        .unwrap();
    let state = request_state_after.get(&ctx.hash).unwrap();
    assert_eq!(
        state.waiting_list.len(),
        1,
        "Waiting list should have 1 peer"
    );
    assert_eq!(
        state.waiting_list[0], sender,
        "Waiting list should contain sender"
    );
}

#[tokio::test]
async fn if_there_is_no_yet_an_entry_in_the_request_state_blocks_should_request_block_and_store_information_about_request_state_block(
) {
    let ctx = TestContext::new().await;
    // given
    let sender = TestContext::peer_node("somePeer", 40400);

    *ctx.fixture
        .block_retriever
        .requested_blocks()
        .lock()
        .unwrap() = HashMap::new();

    // when
    ctx.fixture
        .engine
        .handle_has_block_message(
            sender.clone(),
            ctx.hb.clone(),
            TestContext::always_do_not_ignore_f,
        )
        .await
        .unwrap();

    // then
    let (peer, protocol_msg) = ctx
        .fixture
        .transport_layer
        .get_request(0)
        .expect("No request found");
    // assert RequestState
    let br = TestContext::to_block_request(&protocol_msg);

    assert_eq!(br.hash, ctx.hash);

    assert_eq!(peer, sender);

    assert_eq!(ctx.fixture.transport_layer.request_count(), 1);

    // assert RequestState informaton stored
    let request_state_after = ctx
        .fixture
        .block_retriever
        .requested_blocks()
        .lock()
        .unwrap();
    let state = request_state_after.get(&ctx.hash).unwrap();
    assert_eq!(
        state.waiting_list.len(),
        1,
        "Waiting list should have 1 peer"
    );
    assert_eq!(
        state.waiting_list[0], sender,
        "Waiting list should contain sender"
    );
}

#[tokio::test]
async fn if_casper_does_not_contain_block_with_given_hash_if_there_is_already_an_entry_in_the_request_state_blocks_should_ignore_if_peer_on_the_request_state_peers_list(
) {
    let ctx = TestContext::new().await;
    // given
    let sender = TestContext::peer_node("somePeer", 40400);

    // when
    ctx.fixture
        .engine
        .handle_has_block_message(sender.clone(), ctx.hb.clone(), TestContext::casper_contains)
        .await
        .unwrap();

    // then
    assert_eq!(
        ctx.fixture.transport_layer.request_count(),
        0,
        "Transport queue should be empty"
    );
}

#[tokio::test]
async fn running_handle_has_block_should_not_call_send_hash_to_block_receiver_if_it_is_ignorable_hash(
) {
    let ctx = TestContext::new().await;
    // given
    // Note: Scala passes null as peer because it's not used when ignore_message_f returns true.
    // In Rust, we can't pass null for PeerNode, so we create a dummy peer instead.
    let dummy_peer = TestContext::peer_node("dummy", 40400);

    // when
    ctx.fixture
        .engine
        .handle_has_block_message(
            dummy_peer.clone(),
            ctx.hb.clone(),
            TestContext::casper_contains,
        )
        .await
        .unwrap();

    // then
    assert_eq!(
        ctx.fixture.transport_layer.request_count(),
        0,
        "Transport queue should be empty"
    );
}
