// See casper/src/test/scala/coop/rchain/casper/sync/BlockRetrieverSpec.scala

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

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

    #[derive(Debug, Clone, PartialEq)]
    struct TestReason;
    impl Into<AdmitHashReason> for TestReason {
        fn into(self) -> AdmitHashReason {
            AdmitHashReason::HasBlockMessageReceived
        }
    }

    struct TestFixture {
        hash: BlockHash,
        local_peer: PeerNode,
        peer: PeerNode,
        second_peer: PeerNode,
        block_retriever: BlockRetriever<TransportLayerStub>,
        transport_layer: Arc<TransportLayerStub>,
    }

    impl TestFixture {
        fn new() -> Self {
            let hash = BlockHash::from(TEST_HASH_BYTES.to_vec());
            let local_peer = setup::peer_node("src", 40400);
            let peer = setup::peer_node("peer", 40400);
            let second_peer = setup::peer_node("secondPeer", 40400);

            let connections_cell = ConnectionsCell {
                peers: Arc::new(Mutex::new(Connections::from_vec(vec![local_peer.clone()]))),
            };
            let rp_conf = create_rp_conf_ask(local_peer.clone(), None, None);
            let transport_layer = Arc::new(TransportLayerStub::new());
            let requested_blocks = Arc::new(Mutex::new(HashMap::new()));
            let block_retriever = BlockRetriever::new(
                requested_blocks,
                transport_layer.clone(),
                connections_cell,
                rp_conf,
            );

            Self {
                hash,
                local_peer,
                peer,
                second_peer,
                block_retriever,
                transport_layer,
            }
        }

        fn reset(&self) {
            self.transport_layer.reset();
        }

        async fn setup_known_hash(&self) {
            let test_reason = TestReason;
            let result = self
                .block_retriever
                .admit_hash(self.hash.clone(), None, test_reason.into())
                .await;
            assert!(result.is_ok());
            // Reset transport after setup to clear the initial request
            self.transport_layer.reset();
        }
    }

    mod admit_hash {
        use super::*;

        mod when_hash_is_unknown {
            use super::*;

            #[tokio::test]
            async fn should_add_record_for_hash() {
                let fixture = TestFixture::new();
                let test_reason = TestReason;

                // Call admitHash with unknown hash
                let result = fixture
                    .block_retriever
                    .admit_hash(fixture.hash.clone(), None, test_reason.into())
                    .await;

                // Should succeed
                assert!(result.is_ok());

                // Check that hash was added to requested blocks
                let requested_count = fixture
                    .block_retriever
                    .get_requested_blocks_count()
                    .await
                    .expect("Should be able to get requested blocks count");
                assert_eq!(requested_count, 1);

                // Verify the hash is known through is_received
                let is_known = fixture
                    .block_retriever
                    .is_received(fixture.hash.clone())
                    .await
                    .is_ok();
                assert!(is_known);
            }

            mod when_source_peer_is_unknown {
                use super::*;

                #[tokio::test]
                async fn should_broadcast_has_block_request_and_only_has_block_request() {
                    let fixture = TestFixture::new();
                    fixture.reset();
                    let test_reason = TestReason;

                    // Call admitHash with unknown hash and no peer
                    let result = fixture
                        .block_retriever
                        .admit_hash(fixture.hash.clone(), None, test_reason.into())
                        .await;

                    // Should succeed
                    assert!(result.is_ok());

                    // Should have exactly one request
                    let request_count = fixture.transport_layer.request_count();
                    assert_eq!(request_count, 1);

                    // Get the request and verify it's sent to the correct peer
                    let (peer, protocol) = fixture.transport_layer.get_request(0).unwrap();

                    // In broadcast, the request should be sent to our local peer (from connections)
                    assert_eq!(peer, fixture.local_peer);

                    // Verify the protocol message is HasBlockRequest and contains the correct hash
                    let packet = extract_packet_from_protocol(&protocol)
                        .expect("Should be able to extract packet from protocol");
                    verify_has_block_request(&packet, &fixture.hash)
                        .expect("Should be HasBlockRequest with correct hash");
                }
            }

            mod when_source_peer_is_known {
                use super::*;

                #[tokio::test]
                async fn should_send_block_request_and_only_block_request() {
                    let fixture = TestFixture::new();
                    fixture.reset();
                    let test_reason = TestReason;

                    // Call admitHash with unknown hash and a specific peer
                    let result = fixture
                        .block_retriever
                        .admit_hash(
                            fixture.hash.clone(),
                            Some(fixture.peer.clone()),
                            test_reason.into(),
                        )
                        .await;

                    // Should succeed
                    assert!(result.is_ok());

                    // Should have exactly one request
                    let request_count = fixture.transport_layer.request_count();
                    assert_eq!(request_count, 1);

                    // Get the request and verify it's sent to the correct peer
                    let (recipient, protocol) = fixture.transport_layer.get_request(0).unwrap();
                    assert_eq!(recipient, fixture.peer);

                    // Verify the protocol message is BlockRequest and contains the correct hash
                    let packet = extract_packet_from_protocol(&protocol)
                        .expect("Should be able to extract packet from protocol");
                    verify_block_request(&packet, &fixture.hash)
                        .expect("Should be BlockRequest with correct hash");
                }
            }
        }

        mod when_hash_is_known {
            use super::*;

            mod when_source_peer_is_unknown {
                use super::*;

                #[tokio::test]
                async fn should_ignore_hash() {
                    let fixture = TestFixture::new();
                    fixture.setup_known_hash().await;
                    let test_reason = TestReason;

                    // Call admitHash again with the same hash and no peer
                    let result = fixture
                        .block_retriever
                        .admit_hash(fixture.hash.clone(), None, test_reason.into())
                        .await;

                    // Should succeed but return Ignore status
                    assert!(result.is_ok());
                    let admit_result = result.unwrap();
                    assert_eq!(
                        admit_result.status,
                        casper::rust::engine::block_retriever::AdmitHashStatus::Ignore
                    );
                }
            }

            mod when_source_peer_is_known {
                use super::*;

                #[tokio::test]
                async fn should_request_block_from_peer_if_sources_list_was_empty() {
                    let fixture = TestFixture::new();
                    fixture.setup_known_hash().await;
                    let test_reason = TestReason;

                    // Call admitHash with known hash and a specific peer
                    let result = fixture
                        .block_retriever
                        .admit_hash(
                            fixture.hash.clone(),
                            Some(fixture.peer.clone()),
                            test_reason.into(),
                        )
                        .await;

                    // Should succeed
                    assert!(result.is_ok());

                    // Should have exactly one request (since sources list was empty)
                    let request_count = fixture.transport_layer.request_count();
                    assert_eq!(request_count, 1);

                    // Get the request and verify it's sent to the correct peer
                    let (recipient, protocol) = fixture.transport_layer.get_request(0).unwrap();
                    assert_eq!(recipient, fixture.peer);

                    // Verify the protocol message is BlockRequest and contains the correct hash
                    let packet = extract_packet_from_protocol(&protocol)
                        .expect("Should be able to extract packet from protocol");
                    verify_block_request(&packet, &fixture.hash)
                        .expect("Should be BlockRequest with correct hash");
                }

                #[tokio::test]
                async fn should_ignore_hash_if_peer_is_already_in_sources_list() {
                    let fixture = TestFixture::new();
                    fixture.setup_known_hash().await;

                    // First call to add peer to sources list
                    let result1 = fixture
                        .block_retriever
                        .admit_hash(
                            fixture.hash.clone(),
                            Some(fixture.peer.clone()),
                            TestReason.into(),
                        )
                        .await;
                    assert!(result1.is_ok());

                    // Reset transport to clear the first request
                    fixture.transport_layer.reset();

                    // Second call with the same peer (should be ignored)
                    let result2 = fixture
                        .block_retriever
                        .admit_hash(
                            fixture.hash.clone(),
                            Some(fixture.peer.clone()),
                            TestReason.into(),
                        )
                        .await;

                    // Should succeed but return Ignore status
                    assert!(result2.is_ok());
                    let admit_result = result2.unwrap();
                    assert_eq!(
                        admit_result.status,
                        casper::rust::engine::block_retriever::AdmitHashStatus::Ignore
                    );
                }

                #[tokio::test]
                async fn should_add_peer_to_sources_list_if_it_is_absent() {
                    let fixture = TestFixture::new();
                    fixture.setup_known_hash().await;

                    // Add first peer
                    let result1 = fixture
                        .block_retriever
                        .admit_hash(
                            fixture.hash.clone(),
                            Some(fixture.peer.clone()),
                            TestReason.into(),
                        )
                        .await;
                    assert!(result1.is_ok());

                    // Add second peer (different from first)
                    let result2 = fixture
                        .block_retriever
                        .admit_hash(
                            fixture.hash.clone(),
                            Some(fixture.second_peer.clone()),
                            TestReason.into(),
                        )
                        .await;
                    assert!(result2.is_ok());

                    // Check that both peers were added by verifying the status
                    let admit_result2 = result2.unwrap();
                    assert_eq!(admit_result2.status, casper::rust::engine::block_retriever::AdmitHashStatus::NewSourcePeerAddedToRequest);

                    // Now we can directly check the waiting list size
                    let waiting_list_size = fixture
                        .block_retriever
                        .get_waiting_list_size(&fixture.hash)
                        .await
                        .expect("Should be able to get waiting list size");
                    assert_eq!(waiting_list_size, 2);
                }

                #[tokio::test]
                async fn should_not_request_for_block_from_peer_if_sources_list_was_not_empty() {
                    let fixture = TestFixture::new();
                    fixture.setup_known_hash().await;

                    // Add first peer (this will send a request since list is empty)
                    let result1 = fixture
                        .block_retriever
                        .admit_hash(
                            fixture.hash.clone(),
                            Some(fixture.peer.clone()),
                            TestReason.into(),
                        )
                        .await;
                    assert!(result1.is_ok());

                    // Add second peer (this should NOT send a request since list is not empty)
                    let result2 = fixture
                        .block_retriever
                        .admit_hash(
                            fixture.hash.clone(),
                            Some(fixture.second_peer.clone()),
                            TestReason.into(),
                        )
                        .await;
                    assert!(result2.is_ok());

                    // Should have exactly one request (only from the first peer)
                    let request_count = fixture.transport_layer.request_count();
                    assert_eq!(request_count, 1);
                }
            }
        }
    }
}
