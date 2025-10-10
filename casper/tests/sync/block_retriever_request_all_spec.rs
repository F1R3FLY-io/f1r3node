// See casper/src/test/scala/coop/rchain/casper/sync/BlockRetrieverRequesAllSpec.scala

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use casper::rust::{
        engine::block_retriever::{AdmitHashReason, BlockRetriever},
        protocol::{extract_packet_from_protocol, verify_block_request, verify_has_block_request},
    };
    use comm::rust::{
        peer_node::PeerNode,
        rp::connect::{Connections, ConnectionsCell},
        test_instances::{create_rp_conf_ask, TransportLayerStub},
    };
    use models::rust::block_hash::BlockHash;

    use crate::engine::setup;

    // Test constants
    const TEST_HASH_BYTES: &[u8] = b"newHash";
    const TIMEOUT_SECONDS: u64 = 240;

    #[derive(Debug, Clone, PartialEq)]
    struct TestReason;
    impl Into<AdmitHashReason> for TestReason {
        fn into(self) -> AdmitHashReason {
            AdmitHashReason::HasBlockMessageReceived
        }
    }

    struct TestFixture {
        hash: BlockHash,
        timeout: Duration,
        local_peer: PeerNode,
        block_retriever: BlockRetriever<TransportLayerStub>,
        transport_layer: Arc<TransportLayerStub>,
    }

    impl TestFixture {
        fn new() -> Self {
            let hash = BlockHash::from(TEST_HASH_BYTES.to_vec());
            let timeout = Duration::from_secs(TIMEOUT_SECONDS);
            let local_peer = setup::peer_node("src", 40400);

            let connections_cell = ConnectionsCell {
                peers: Arc::new(Mutex::new(Connections::from_vec(vec![local_peer.clone()]))),
            };
            let rp_conf = create_rp_conf_ask(local_peer.clone(), None, None);
            let transport_layer = Arc::new(TransportLayerStub::new());
            let block_retriever = BlockRetriever::new(
                transport_layer.clone(),
                connections_cell.clone(),
                rp_conf.clone(),
            );

            Self {
                hash,
                timeout,
                local_peer,
                block_retriever,
                transport_layer,
            }
        }

        fn reset(&self) {
            self.transport_layer.reset();
        }

        // Helper function to create a peer node for testing
        fn peer_node(name: &str, port: u32) -> PeerNode {
            setup::peer_node(name, port)
        }

        // Helper function to create a timed out timestamp
        fn create_timed_out_timestamp(&self) -> u64 {
            use std::time::{SystemTime, UNIX_EPOCH};
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            now.saturating_sub((2 * self.timeout.as_millis()) as u64)
        }
    }

    mod maintain_requested_blocks {
        use super::*;

        mod for_every_block_that_was_requested {
            use super::*;

            mod if_block_request_is_still_within_timeout {
                use super::*;

                #[tokio::test]
                async fn should_keep_the_request_not_touch() {
                    let fixture = TestFixture::new();
                    fixture.reset();

                    // Given - create a request that won't be timed out
                    let test_reason = TestReason;
                    let result = fixture
                        .block_retriever
                        .admit_hash(fixture.hash.clone(), None, test_reason.into())
                        .await;
                    assert!(result.is_ok());

                    // Verify initial state - should have 1 request
                    let initial_count = fixture
                        .block_retriever
                        .get_requested_blocks_count()
                        .await
                        .expect("Should be able to get requested blocks count");
                    assert_eq!(initial_count, 1);

                    // When - call requestAll with the timeout
                    let result = fixture.block_retriever.request_all(fixture.timeout).await;
                    assert!(result.is_ok());

                    // Then - verify the request is still there (not removed)
                    let final_count = fixture
                        .block_retriever
                        .get_requested_blocks_count()
                        .await
                        .expect("Should be able to get requested blocks count");
                    assert_eq!(final_count, 1);
                }
            }
        }

        mod if_block_was_not_delivered_within_given_timeout {
            use super::*;
            use std::collections::HashSet;

            mod if_waiting_list_is_not_empty {
                use super::*;

                #[tokio::test]
                async fn should_request_block_from_first_peer_on_a_waiting_list() {
                    let fixture = TestFixture::new();
                    fixture.reset();

                    // Given - setup a timed out request with waiting list
                    let waiting1 = TestFixture::peer_node("waiting1", 40401);
                    let waiting2 = TestFixture::peer_node("waiting2", 40402);
                    let peer = TestFixture::peer_node("peer", 40403);

                    let waiting_list = vec![waiting1.clone(), waiting2.clone()];
                    let peers = HashSet::from([peer]);
                    let timed_out_timestamp = fixture.create_timed_out_timestamp();

                    let request_state = casper::rust::engine::block_retriever::RequestState {
                        timestamp: timed_out_timestamp,
                        peers,
                        received: false,
                        in_casper_buffer: false,
                        waiting_list,
                    };

                    fixture
                        .block_retriever
                        .set_request_state_for_test(fixture.hash.clone(), request_state)
                        .await
                        .expect("Should be able to set request state");

                    // When - call requestAll
                    let result = fixture.block_retriever.request_all(fixture.timeout).await;
                    assert!(result.is_ok());

                    // Then - should have exactly one request to the first waiting peer
                    let request_count = fixture.transport_layer.request_count();
                    assert_eq!(request_count, 1);

                    let (recipient, protocol) = fixture.transport_layer.get_request(0).unwrap();
                    assert_eq!(recipient, waiting1);

                    // Verify the protocol message is BlockRequest and contains the correct hash
                    let packet = extract_packet_from_protocol(&protocol)
                        .expect("Should be able to extract packet from protocol");
                    verify_block_request(&packet, &fixture.hash)
                        .expect("Should be BlockRequest with correct hash");
                }

                #[tokio::test]
                async fn should_move_that_peer_from_the_waiting_list_to_the_requested_set() {
                    let fixture = TestFixture::new();
                    fixture.reset();

                    // Given - setup a timed out request with waiting list
                    let waiting1 = TestFixture::peer_node("waiting1", 40401);
                    let waiting2 = TestFixture::peer_node("waiting2", 40402);
                    let peer = TestFixture::peer_node("peer", 40403);

                    let waiting_list = vec![waiting1.clone(), waiting2.clone()];
                    let peers = HashSet::from([peer.clone()]);
                    let timed_out_timestamp = fixture.create_timed_out_timestamp();

                    let request_state = casper::rust::engine::block_retriever::RequestState {
                        timestamp: timed_out_timestamp,
                        peers,
                        received: false,
                        in_casper_buffer: false,
                        waiting_list,
                    };

                    fixture
                        .block_retriever
                        .set_request_state_for_test(fixture.hash.clone(), request_state)
                        .await
                        .expect("Should be able to set request state");

                    // When - call requestAll
                    let result = fixture.block_retriever.request_all(fixture.timeout).await;
                    assert!(result.is_ok());

                    // Then - verify the peer was moved from waiting list to peers set
                    let request_state_after = fixture
                        .block_retriever
                        .get_request_state_for_test(&fixture.hash)
                        .await
                        .expect("Should be able to get request state")
                        .expect("Request state should exist");

                    // waiting1 should be removed from waiting list
                    assert_eq!(request_state_after.waiting_list, vec![waiting2]);

                    // waiting1 should be added to peers set
                    let expected_peers = HashSet::from([peer, waiting1]);
                    assert_eq!(request_state_after.peers, expected_peers);
                }

                #[tokio::test]
                async fn timestamp_is_reset() {
                    let fixture = TestFixture::new();
                    fixture.reset();

                    // Given - setup a timed out request with waiting list
                    let waiting1 = TestFixture::peer_node("waiting1", 40401);
                    let waiting2 = TestFixture::peer_node("waiting2", 40402);
                    let peer = TestFixture::peer_node("peer", 40403);

                    let waiting_list = vec![waiting1.clone(), waiting2.clone()];
                    let peers = HashSet::from([peer]);
                    let timed_out_timestamp = fixture.create_timed_out_timestamp();

                    let request_state = casper::rust::engine::block_retriever::RequestState {
                        timestamp: timed_out_timestamp,
                        peers,
                        received: false,
                        in_casper_buffer: false,
                        waiting_list,
                    };

                    fixture
                        .block_retriever
                        .set_request_state_for_test(fixture.hash.clone(), request_state)
                        .await
                        .expect("Should be able to set request state");

                    // Get the current time just before calling requestAll
                    let current_time_before = {
                        use std::time::{SystemTime, UNIX_EPOCH};
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64
                    };

                    // When - call requestAll
                    let result = fixture.block_retriever.request_all(fixture.timeout).await;
                    assert!(result.is_ok());

                    // Get the current time just after calling requestAll
                    let current_time_after = {
                        use std::time::{SystemTime, UNIX_EPOCH};
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64
                    };

                    // Then - verify the timestamp was reset to the current time (matches Scala: requestedAfter.timestamp shouldBe time.clock)
                    let request_state_after = fixture
                        .block_retriever
                        .get_request_state_for_test(&fixture.hash)
                        .await
                        .expect("Should be able to get request state")
                        .expect("Request state should exist");

                    // The timestamp should be between the current time before and after the call
                    // This matches the Scala semantics where requestedAfter.timestamp shouldBe time.clock
                    assert!(
                        request_state_after.timestamp >= current_time_before,
                        "Timestamp should be at least the current time before the call"
                    );
                    assert!(
                        request_state_after.timestamp <= current_time_after,
                        "Timestamp should be at most the current time after the call"
                    );
                }
            }

            mod if_waiting_list_has_one_peer_left {
                use super::*;

                #[tokio::test]
                async fn should_broadcast_has_block_request() {
                    let fixture = TestFixture::new();
                    fixture.reset();

                    // Given - setup a timed out request with one peer left in waiting list
                    let last_peer = TestFixture::peer_node("lastPeer", 40401);
                    let peer = TestFixture::peer_node("peer", 40403);
                    let peers = HashSet::from([peer]);
                    let timed_out_timestamp = fixture.create_timed_out_timestamp();

                    let request_state = casper::rust::engine::block_retriever::RequestState {
                        timestamp: timed_out_timestamp,
                        peers,
                        received: false,
                        in_casper_buffer: false,
                        waiting_list: vec![last_peer.clone()], // One peer left in waiting list
                    };

                    fixture
                        .block_retriever
                        .set_request_state_for_test(fixture.hash.clone(), request_state)
                        .await
                        .expect("Should be able to set request state");

                    // When - call requestAll
                    let result = fixture.block_retriever.request_all(fixture.timeout).await;
                    assert!(result.is_ok());

                    // Then - should have exactly 2 requests: BlockRequest + HasBlockRequest
                    let request_count = fixture.transport_layer.request_count();
                    assert_eq!(request_count, 2);

                    // First request should be BlockRequest to the last peer
                    let (recipient, protocol) = fixture.transport_layer.get_request(0).unwrap();
                    assert_eq!(recipient, last_peer);

                    let packet = extract_packet_from_protocol(&protocol)
                        .expect("Should be able to extract packet from protocol");
                    verify_block_request(&packet, &fixture.hash)
                        .expect("Should be BlockRequest with correct hash");

                    // Second request should be HasBlockRequest (broadcast)
                    let (_, protocol1) = fixture.transport_layer.get_request(1).unwrap();
                    let packet1 = extract_packet_from_protocol(&protocol1)
                        .expect("Should be able to extract packet from protocol");
                    verify_has_block_request(&packet1, &fixture.hash)
                        .expect("Should be HasBlockRequest with correct hash");
                }

                #[tokio::test]
                async fn should_not_send_requests_to_other_peers() {
                    let fixture = TestFixture::new();
                    fixture.reset();

                    // Given - setup a timed out request with empty waiting list
                    let peer = TestFixture::peer_node("peer", 40403);
                    let peers = HashSet::from([peer]);
                    let timed_out_timestamp = fixture.create_timed_out_timestamp();

                    let request_state = casper::rust::engine::block_retriever::RequestState {
                        timestamp: timed_out_timestamp,
                        peers,
                        received: false,
                        in_casper_buffer: false,
                        waiting_list: Vec::new(), // Empty waiting list
                    };

                    fixture
                        .block_retriever
                        .set_request_state_for_test(fixture.hash.clone(), request_state)
                        .await
                        .expect("Should be able to set request state");

                    // When - call requestAll
                    let result = fixture.block_retriever.request_all(fixture.timeout).await;
                    assert!(result.is_ok());

                    // Then - should not send any requests
                    let request_count = fixture.transport_layer.request_count();
                    assert_eq!(request_count, 0);
                }

                #[tokio::test]
                async fn should_remove_the_entry_from_requested_block_lists_when_block_is_in_casper_buffer_and_after_timeout(
                ) {
                    let fixture = TestFixture::new();
                    fixture.reset();

                    // Given - setup a timed out request that's in casper buffer with empty waiting list
                    let peer = TestFixture::peer_node("peer", 40403);
                    let peers = HashSet::from([peer]);
                    let timed_out_timestamp = fixture.create_timed_out_timestamp();

                    let request_state = casper::rust::engine::block_retriever::RequestState {
                        timestamp: timed_out_timestamp,
                        peers,
                        received: false,
                        in_casper_buffer: true, // Already in casper buffer
                        waiting_list: Vec::new(),
                    };

                    fixture
                        .block_retriever
                        .set_request_state_for_test(fixture.hash.clone(), request_state)
                        .await
                        .expect("Should be able to set request state");

                    // When - call requestAll
                    let result = fixture.block_retriever.request_all(fixture.timeout).await;
                    assert!(result.is_ok());

                    // Then - should have no requests (entry should be removed)
                    let final_count = fixture
                        .block_retriever
                        .get_requested_blocks_count()
                        .await
                        .expect("Should be able to get requested blocks count");
                    assert_eq!(final_count, 0);
                }
            }
        }
    }
}
