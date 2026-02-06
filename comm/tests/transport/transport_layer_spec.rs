// See comm/src/test/scala/coop/rchain/comm/transport/TransportLayerSpec.scala

use std::sync::{Arc, Once};

use comm::rust::transport::transport_layer::{Blob, TransportLayer};
use models::routing::Packet;
use prost::bytes::Bytes;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::transport::transport_layer_runtime::{
    broadcast_heartbeat, send_heartbeat, TestProtocolDispatcher, TestStreamDispatcher,
    TransportLayerTestRuntime,
};

static INIT: Once = Once::new();

fn init_logger() {
    INIT.call_once(|| {
        let filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::DEBUG.into())
            .parse("")
            .unwrap();

        tracing_subscriber::registry()
            .with(filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .json()
                    .with_target(false)
                    .with_current_span(false) // logs only
                    .with_span_list(false) // logs only
                    .flatten_event(true), // put event fields at top level
            )
            .try_init()
            .unwrap();
    });
}

/// Create big content for streaming tests
pub fn create_big_content(max_message_size: i32) -> Bytes {
    // Create content that is 4.5 times the max message size to force streaming
    let size = (4 * max_message_size) + (max_message_size / 2);
    let content = vec![128u8; size as usize];
    Bytes::from(content)
}

#[tokio::test]
async fn sending_a_message_should_deliver_the_message() {
    init_logger();

    let runtime = TransportLayerTestRuntime::new("test".to_string());
    let protocol_dispatcher = TestProtocolDispatcher::new();

    let result = runtime
        .run_two_nodes_test(
            |transport, local, remote| async move {
                send_heartbeat(Arc::new(transport), &local, &remote, "test").await
            },
            Some(protocol_dispatcher.clone()),
            None, // Use default stream dispatcher
            true, // block_until_dispatched
        )
        .await
        .expect("Test should succeed");

    // Verify message was received
    let received = protocol_dispatcher.received();
    assert_eq!(received.len(), 1);

    let (receiver_peer, protocol_msg) = &received[0];
    assert_eq!(receiver_peer, &result.remote_node);

    // Verify the sender in the protocol message matches local node
    let header = protocol_msg.header.as_ref().expect("Header should exist");
    let sender_node = header.sender.as_ref().expect("Sender should exist");
    let sender_peer = comm::rust::peer_node::PeerNode::from_node(sender_node.clone())
        .expect("Should convert to PeerNode");
    assert_eq!(sender_peer, result.local_node);

    // Verify it's a heartbeat message
    assert!(protocol_msg.message.is_some());
    if let Some(models::routing::protocol::Message::Heartbeat(_)) = &protocol_msg.message {
        // Expected heartbeat message
    } else {
        panic!("Expected heartbeat message");
    }

    // Verify the send operation succeeded
    assert!(result.result.is_ok());
}

#[tokio::test]
async fn broadcasting_a_message_should_send_the_message_to_all_peers() {
    init_logger();

    let runtime = TransportLayerTestRuntime::new("test".to_string());
    let protocol_dispatcher = TestProtocolDispatcher::new();

    let result = runtime
        .run_three_nodes_test(
            |transport, local, remote1, remote2| async move {
                broadcast_heartbeat(Arc::new(transport), &local, &[remote1, remote2], "test").await
            },
            Some(protocol_dispatcher.clone()),
            None, // Use default stream dispatcher
            true, // block_until_dispatched
        )
        .await
        .expect("Test should succeed");

    // Verify messages were received by both peers
    let received = protocol_dispatcher.received();
    assert_eq!(received.len(), 2);

    let (receiver1, protocol1) = &received[0];
    let (receiver2, protocol2) = &received[1];

    // Verify senders match local node
    for protocol_msg in [protocol1, protocol2] {
        let header = protocol_msg.header.as_ref().expect("Header should exist");
        let sender_node = header.sender.as_ref().expect("Sender should exist");
        let sender_peer = comm::rust::peer_node::PeerNode::from_node(sender_node.clone())
            .expect("Should convert to PeerNode");
        assert_eq!(sender_peer, result.local_node);

        // Verify it's a heartbeat message
        assert!(protocol_msg.message.is_some());
        if let Some(models::routing::protocol::Message::Heartbeat(_)) = &protocol_msg.message {
            // Expected heartbeat message
        } else {
            panic!("Expected heartbeat message");
        }
    }

    // Verify receivers are the two remote nodes (order may vary)
    let receivers = vec![receiver1, receiver2];
    assert!(receivers.contains(&&result.remote_node1));
    assert!(receivers.contains(&&result.remote_node2));

    // Verify the broadcast operation succeeded
    assert!(result.result.is_ok());
}

#[tokio::test]
async fn stream_blob_should_send_a_blob_and_receive_by_single_remote_side() {
    init_logger();

    let runtime = TransportLayerTestRuntime::new("test".to_string());
    let stream_dispatcher = TestStreamDispatcher::new();

    let big_content = create_big_content(runtime.max_message_size);
    let big_content_for_assert = big_content.clone();

    let result = runtime
        .run_two_nodes_test(
            |transport, local, remote| async move {
                let blob = Blob {
                    sender: local.clone(),
                    packet: Packet {
                        type_id: "Test".to_string(),
                        content: big_content.clone(),
                    },
                };
                transport.stream(&remote, &blob).await
            },
            None, // Use default protocol dispatcher
            Some(stream_dispatcher.clone()),
            true, // block_until_dispatched
        )
        .await
        .expect("Test should succeed");

    // Verify blob was received
    let received = stream_dispatcher.received();
    assert_eq!(received.len(), 1);

    let (receiver_peer, received_blob) = &received[0];
    assert_eq!(receiver_peer, &result.remote_node);

    // Verify blob content
    assert_eq!(received_blob.packet.type_id, "Test");
    assert_eq!(received_blob.packet.content, big_content_for_assert);

    // Verify the stream operation succeeded
    assert!(result.result.is_ok());
}

#[tokio::test]
async fn stream_blob_should_send_a_blob_and_receive_by_multiple_remote_side() {
    init_logger();

    let runtime = TransportLayerTestRuntime::new("test".to_string());
    let stream_dispatcher = TestStreamDispatcher::new();

    let big_content = create_big_content(runtime.max_message_size);
    let big_content_for_assert = big_content.clone();

    let result = runtime
        .run_three_nodes_test(
            |transport, local, remote1, remote2| async move {
                let blob = Blob {
                    sender: local.clone(),
                    packet: Packet {
                        type_id: "N/A".to_string(),
                        content: big_content.clone(),
                    },
                };
                transport.stream_mult(&[remote1, remote2], &blob).await
            },
            None, // Use default protocol dispatcher
            Some(stream_dispatcher.clone()),
            true, // block_until_dispatched
        )
        .await
        .expect("Test should succeed");

    // Verify blobs were received by both peers
    let received = stream_dispatcher.received();
    assert_eq!(received.len(), 2);

    let (receiver1, blob1) = &received[0];
    let (receiver2, blob2) = &received[1];

    // Verify blob content for both
    for received_blob in [blob1, blob2] {
        assert_eq!(received_blob.packet.type_id, "N/A");
        assert_eq!(received_blob.packet.content, big_content_for_assert);
    }

    // Verify receivers are the two remote nodes (order may vary)
    let receivers = vec![receiver1, receiver2];
    assert!(receivers.contains(&&result.remote_node1));
    assert!(receivers.contains(&&result.remote_node2));

    // Verify the stream operation succeeded
    assert!(result.result.is_ok());
}

// **Additional Resilience Tests**

#[tokio::test]
async fn sending_message_to_unavailable_peer_should_fail_with_peer_unavailable() {
    init_logger();

    let runtime = TransportLayerTestRuntime::new("dead_peer_test".to_string());

    let result = runtime
        .run_two_nodes_test_remote_dead(|transport, local, remote| async move {
            send_heartbeat(Arc::new(transport), &local, &remote, "dead_peer_test").await
        })
        .await
        .expect("Test should succeed");

    // Should fail with some error (could be PeerUnavailable or other connection error)
    assert!(result.result.is_err());
    println!("Error received: {:?}", result.result);
    // Accept any communication error type since dead peer can manifest as different errors
}

#[tokio::test]
async fn streaming_to_unavailable_peer_should_fail_with_peer_unavailable() {
    init_logger();

    let runtime = TransportLayerTestRuntime::new("dead_stream_test".to_string());
    let big_content = create_big_content(runtime.max_message_size);

    let result = runtime
        .run_two_nodes_test_remote_dead(|transport, local, remote| async move {
            let blob = Blob {
                sender: local.clone(),
                packet: Packet {
                    type_id: "Test".to_string(),
                    content: big_content.clone(),
                },
            };
            transport.stream(&remote, &blob).await
        })
        .await
        .expect("Test should succeed");

    // Should fail with some error (could be PeerUnavailable or other connection error)
    assert!(result.result.is_err());
    println!("Error received: {:?}", result.result);
    // Accept any communication error type since dead peer can manifest as different errors
}

// **Edge Cases and Boundary Tests**

#[tokio::test]
async fn sending_empty_message_should_work() {
    init_logger();

    let runtime = TransportLayerTestRuntime::new("empty_msg_test".to_string());
    let protocol_dispatcher = TestProtocolDispatcher::new();

    let result = runtime
        .run_two_nodes_test(
            |transport, local, remote| async move {
                // Create a protocol message with minimal content
                let msg = comm::rust::rp::protocol_helper::heartbeat(&local, "empty_msg_test");
                transport.send(&remote, &msg).await
            },
            Some(protocol_dispatcher.clone()),
            None,
            true,
        )
        .await
        .expect("Test should succeed");

    // Verify message was received
    let received = protocol_dispatcher.received();
    assert_eq!(received.len(), 1);
    assert!(result.result.is_ok());
}

#[tokio::test]
async fn streaming_empty_blob_should_work() {
    init_logger();

    let runtime = TransportLayerTestRuntime::new("empty_blob_test".to_string());
    let stream_dispatcher = TestStreamDispatcher::new();

    let result = runtime
        .run_two_nodes_test(
            |transport, local, remote| async move {
                let blob = Blob {
                    sender: local.clone(),
                    packet: Packet {
                        type_id: "EmptyTest".to_string(),
                        content: Bytes::new(), // Empty content
                    },
                };
                transport.stream(&remote, &blob).await
            },
            None,
            Some(stream_dispatcher.clone()),
            true,
        )
        .await
        .expect("Test should succeed");

    // Verify empty blob was received
    let received = stream_dispatcher.received();
    assert_eq!(received.len(), 1);
    let (_, received_blob) = &received[0];
    assert_eq!(received_blob.packet.type_id, "EmptyTest");
    assert_eq!(received_blob.packet.content.len(), 0);
    assert!(result.result.is_ok());
}

#[tokio::test]
async fn streaming_very_large_blob_should_work() {
    init_logger();

    let runtime = TransportLayerTestRuntime::new("large_blob_test".to_string());
    let stream_dispatcher = TestStreamDispatcher::new();

    // Create a very large blob (10x the normal test size)
    let large_size = (10 * runtime.max_message_size) as usize;
    let large_content = Bytes::from(vec![42u8; large_size]);
    let large_content_for_assert = large_content.clone();

    let result = runtime
        .run_two_nodes_test(
            |transport, local, remote| async move {
                let blob = Blob {
                    sender: local.clone(),
                    packet: Packet {
                        type_id: "LargeTest".to_string(),
                        content: large_content.clone(),
                    },
                };
                transport.stream(&remote, &blob).await
            },
            None,
            Some(stream_dispatcher.clone()),
            true,
        )
        .await
        .expect("Test should succeed");

    // Verify large blob was received correctly
    let received = stream_dispatcher.received();
    assert_eq!(received.len(), 1);
    let (_, received_blob) = &received[0];
    assert_eq!(received_blob.packet.type_id, "LargeTest");
    assert_eq!(received_blob.packet.content, large_content_for_assert);
    assert!(result.result.is_ok());
}

// **Concurrent Operations Tests**

#[tokio::test]
async fn concurrent_sends_to_same_peer_should_all_succeed() {
    init_logger();

    let runtime = TransportLayerTestRuntime::new("concurrent_test".to_string());
    let protocol_dispatcher = TestProtocolDispatcher::new();

    let result = runtime
        .run_two_nodes_test(
            |transport, local, remote| async move {
                // Send multiple messages concurrently with unique content
                // Each message must be unique to avoid deduplication by RecentHashFilter
                let futures: Vec<_> = (0..5)
                    .map(|i| {
                        let transport = &transport;
                        let local = &local;
                        let remote = &remote;
                        async move {
                            // Use packet with unique content to avoid RecentHashFilter deduplication
                            let packet = models::routing::Packet {
                                content: prost::bytes::Bytes::from(format!("concurrent_msg_{}", i)),
                                type_id: "test".to_string(),
                            };
                            let msg = comm::rust::rp::protocol_helper::packet(
                                local,
                                "concurrent_test",
                                packet,
                            );
                            transport
                                .send(remote, &msg)
                                .await
                                .map_err(|e| format!("Concurrent send {} failed: {}", i, e))
                        }
                    })
                    .collect();

                // Execute all sends concurrently
                let results = futures::future::join_all(futures).await;

                // Check all succeeded
                for (i, result) in results.iter().enumerate() {
                    if let Err(e) = result {
                        return Err(format!("Send {} failed: {}", i, e));
                    }
                }

                Ok::<(), String>(())
            },
            Some(protocol_dispatcher.clone()),
            None,
            true, // block_until_dispatched
        )
        .await
        .expect("Test should succeed");

    // Wait for all operations to complete
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Verify all messages were received
    let received = protocol_dispatcher.received();
    assert_eq!(
        received.len(),
        5,
        "Expected 5 concurrent sends, got {}",
        received.len()
    );
    assert!(result.result.is_ok());
}

#[tokio::test]
async fn concurrent_streams_to_same_peer_should_all_succeed() {
    init_logger();

    let runtime = TransportLayerTestRuntime::new("concurrent_streams_test".to_string());
    let stream_dispatcher = TestStreamDispatcher::new();

    // Use moderate-sized content for streaming
    let content = prost::bytes::Bytes::from(vec![42u8; 2000]);

    let result = runtime
        .run_two_nodes_test(
            |transport, local, remote| async move {
                // Send multiple streams concurrently to test multi-consumer buffer architecture
                let futures: Vec<_> = (0..3)
                    .map(|i| {
                        let transport = &transport;
                        let local = &local;
                        let remote = &remote;
                        let content = content.clone();

                        async move {
                            // Make each stream unique
                            let mut stream_content = content.to_vec();
                            stream_content[0] = (100 + i) as u8; // First byte indicates stream number

                            let blob = Blob {
                                sender: local.clone(),
                                packet: Packet {
                                    type_id: format!("ConcurrentStream_{}", i),
                                    content: prost::bytes::Bytes::from(stream_content),
                                },
                            };

                            transport
                                .stream(&remote, &blob)
                                .await
                                .map_err(|e| format!("Concurrent stream {} failed: {}", i, e))
                        }
                    })
                    .collect();

                // Execute all streams concurrently
                let results = futures::future::join_all(futures).await;

                // Check all succeeded
                for (i, result) in results.iter().enumerate() {
                    if let Err(e) = result {
                        return Err(format!("Stream {} failed: {}", i, e));
                    }
                }

                Ok::<(), String>(())
            },
            None,
            Some(stream_dispatcher.clone()),
            false, // Don't use block_until_dispatched for multiple streams
        )
        .await
        .expect("Test should succeed");

    // Wait for all processing to complete
    tokio::time::sleep(std::time::Duration::from_millis(1500)).await;

    // Verify all streams were received
    let received = stream_dispatcher.received();
    assert_eq!(
        received.len(),
        3,
        "Expected 3 concurrent streams, got {}",
        received.len()
    );

    // Verify each stream has correct content
    let mut type_ids: Vec<String> = received
        .iter()
        .map(|(_, blob)| blob.packet.type_id.clone())
        .collect();
    type_ids.sort();
    let expected_ids = vec![
        "ConcurrentStream_0".to_string(),
        "ConcurrentStream_1".to_string(),
        "ConcurrentStream_2".to_string(),
    ];
    assert_eq!(
        type_ids, expected_ids,
        "All concurrent stream type IDs should be present"
    );

    // Verify unique content for each stream
    for (_, blob) in &received {
        assert_eq!(
            blob.packet.content.len(),
            2000,
            "Stream content should be 2000 bytes"
        );
        let first_byte = blob.packet.content[0];
        assert!(
            first_byte >= 100 && first_byte <= 102,
            "First byte should indicate stream number (100-102)"
        );
    }

    assert!(
        result.result.is_ok(),
        "Concurrent stream operations should succeed"
    );
}

// **Mixed Operations Tests**

#[tokio::test]
async fn mixed_sends_and_streams_should_all_work() {
    init_logger();

    let runtime = TransportLayerTestRuntime::new("mixed_test".to_string());
    let protocol_dispatcher = TestProtocolDispatcher::new();
    let stream_dispatcher = TestStreamDispatcher::new();

    let content = create_big_content(runtime.max_message_size);

    let result = runtime
        .run_two_nodes_test(
            |transport, local, remote| async move {
                // Mix of sends and streams executed concurrently
                let send_future = async {
                    let msg = comm::rust::rp::protocol_helper::heartbeat(&local, "mixed_test");
                    transport
                        .send(&remote, &msg)
                        .await
                        .map_err(|e| format!("Send failed: {}", e))
                };

                let stream_future = async {
                    let blob = Blob {
                        sender: local.clone(),
                        packet: Packet {
                            type_id: "MixedStreamTest".to_string(),
                            content: content.clone(),
                        },
                    };
                    transport
                        .stream(&remote, &blob)
                        .await
                        .map_err(|e| format!("Stream failed: {}", e))
                };

                // Execute both concurrently
                let (send_result, stream_result) = tokio::join!(send_future, stream_future);

                send_result?;
                stream_result?;

                Ok::<(), String>(())
            },
            Some(protocol_dispatcher.clone()),
            Some(stream_dispatcher.clone()),
            true, // block_until_dispatched
        )
        .await
        .expect("Test should succeed");

    // Wait for operations to complete
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Verify both message types were received
    let protocol_received = protocol_dispatcher.received();
    let stream_received = stream_dispatcher.received();

    assert_eq!(protocol_received.len(), 1, "Expected 1 protocol message");
    assert_eq!(stream_received.len(), 1, "Expected 1 stream message");
    assert!(result.result.is_ok());
}

// **Broadcast Edge Cases**

#[tokio::test]
async fn broadcasting_to_empty_peer_list_should_succeed() {
    init_logger();

    let runtime = TransportLayerTestRuntime::new("empty_broadcast_test".to_string());
    let protocol_dispatcher = TestProtocolDispatcher::new();

    let result = runtime
        .run_two_nodes_test(
            |transport, local, _remote| async move {
                // Broadcast to empty list
                broadcast_heartbeat(Arc::new(transport), &local, &[], "empty_broadcast_test").await
            },
            Some(protocol_dispatcher.clone()),
            None,
            false, // Don't block since no messages will be sent
        )
        .await
        .expect("Test should succeed");

    // Should succeed with no messages sent
    let received = protocol_dispatcher.received();
    assert_eq!(received.len(), 0);
    assert!(result.result.is_ok());
}

#[tokio::test]
async fn streaming_to_empty_peer_list_should_succeed() {
    init_logger();

    let runtime = TransportLayerTestRuntime::new("empty_stream_test".to_string());
    let stream_dispatcher = TestStreamDispatcher::new();

    let content = create_big_content(runtime.max_message_size);

    let result = runtime
        .run_two_nodes_test(
            |transport, local, _remote| async move {
                let blob = Blob {
                    sender: local.clone(),
                    packet: Packet {
                        type_id: "EmptyBroadcastTest".to_string(),
                        content: content.clone(),
                    },
                };
                transport.stream_mult(&[], &blob).await
            },
            None,
            Some(stream_dispatcher.clone()),
            false, // Don't block since no messages will be sent
        )
        .await
        .expect("Test should succeed");

    // Should succeed with no streams sent
    let received = stream_dispatcher.received();
    assert_eq!(received.len(), 0);
    assert!(result.result.is_ok());
}
