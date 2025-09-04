// See comm/src/test/scala/coop/rchain/comm/discovery/KademliaRPCSpec.scala

use super::kademlia_rpc_runtime::*;
use comm::rust::discovery::{grpc_kademlia_rpc::GrpcKademliaRPC, kademlia_rpc::KademliaRPC};
use comm::rust::peer_node::{Endpoint, NodeIdentifier, PeerNode};
use rand::RngCore;
use std::time::Duration;

#[tokio::test]
async fn ping_remote_peer_send_and_receive_positive_response() {
    let runtime = TestRuntime::new("test".to_string());

    let execute_fn = |kademlia_rpc: GrpcKademliaRPC, _local: PeerNode, remote: PeerNode| async move {
        kademlia_rpc.ping(&remote).await.unwrap_or(false)
    };

    // Run test with default handlers
    let (test_result, ping_handler, _lookup_handler) = run_two_nodes_test(
        &runtime, execute_fn, None, // Use default ping handler
        None, // Use default lookup handler
    )
    .await
    .expect("Test should succeed");

    let received = ping_handler.received();
    assert_eq!(test_result.result, true);
    assert_eq!(received.len(), 1);

    let (receiver, sender) = &received[0];
    assert_eq!(receiver, &test_result.remote_node);
    assert_eq!(sender, &test_result.local_node);
}

#[tokio::test]
async fn ping_remote_peer_send_twice_and_receive_positive_responses() {
    let runtime = TestRuntime::new("test".to_string());

    let execute_fn = |kademlia_rpc: GrpcKademliaRPC, _local: PeerNode, remote: PeerNode| async move {
        let r1 = kademlia_rpc.ping(&remote).await.unwrap_or(false);
        let r2 = kademlia_rpc.ping(&remote).await.unwrap_or(false);
        (r1, r2)
    };

    let (test_result, ping_handler, _lookup_handler) = run_two_nodes_test(
        &runtime, execute_fn, None, // Use default ping handler
        None, // Use default lookup handler
    )
    .await
    .expect("Test should succeed");

    let received = ping_handler.received();
    assert_eq!(test_result.result, (true, true));
    assert_eq!(received.len(), 2);

    let (receiver1, sender1) = &received[0];
    let (receiver2, sender2) = &received[1];

    assert_eq!(receiver1, &test_result.remote_node);
    assert_eq!(receiver2, &test_result.remote_node);

    assert_eq!(sender1, &test_result.local_node);
    assert_eq!(sender2, &test_result.local_node);
}

#[tokio::test]
async fn ping_remote_peer_response_takes_too_long_get_negative_result() {
    let runtime = TestRuntime::new("test".to_string());

    let execute_fn = |kademlia_rpc: GrpcKademliaRPC, _local: PeerNode, remote: PeerNode| async move {
        kademlia_rpc.ping(&remote).await.unwrap_or(false)
    };

    let ping_handler_with_delay = TestPingHandler::with_delay(Duration::from_secs(1));

    let (test_result, ping_handler, _lookup_handler) = run_two_nodes_test(
        &runtime,
        execute_fn,
        Some(ping_handler_with_delay), // Use ping handler with 1 second delay
        None,                          // Use default lookup handler
    )
    .await
    .expect("Test should succeed");

    let received = ping_handler.received();
    assert_eq!(test_result.result, false);
    assert_eq!(received.len(), 1);

    let (receiver, sender) = &received[0];
    assert_eq!(receiver, &test_result.remote_node);
    assert_eq!(sender, &test_result.local_node);
}

#[tokio::test]
async fn ping_remote_peer_peer_is_not_listening_get_negative_result() {
    let runtime = TestRuntime::new("test".to_string());

    let execute_fn = |kademlia_rpc: GrpcKademliaRPC, _local: PeerNode, remote: PeerNode| async move {
        kademlia_rpc.ping(&remote).await.unwrap_or(false)
    };

    // Use the remote dead variant - no server will be started
    let (test_result, _ping_handler, _lookup_handler) = run_two_nodes_test_remote_dead(
        &runtime, execute_fn,
        None, // Use default ping handler (won't be called since no server)
        None, // Use default lookup handler
    )
    .await
    .expect("Test should succeed");

    assert_eq!(test_result.result, false);
}

#[tokio::test]
async fn lookup_remote_peer_send_and_receive_list_of_peers() {
    let runtime = TestRuntime::new("test".to_string());

    let mut key = [0u8; 40];
    rand::rng().fill_bytes(&mut key);

    let other_peer = PeerNode {
        id: NodeIdentifier {
            key: prost::bytes::Bytes::from(key.to_vec()),
        },
        endpoint: Endpoint::new("1.2.3.4".to_string(), 0, 0),
    };

    let execute_fn = |kademlia_rpc: GrpcKademliaRPC, _local: PeerNode, remote: PeerNode| {
        let key = key.clone();
        async move { kademlia_rpc.lookup(&key, &remote).await.unwrap_or_default() }
    };

    let lookup_handler = TestLookupHandler::with_response(vec![other_peer.clone()]);

    let (test_result, _ping_handler, lookup_handler) = run_two_nodes_test(
        &runtime,
        execute_fn,
        None,                 // Use default ping handler
        Some(lookup_handler), // Use lookup handler with response
    )
    .await
    .expect("Test should succeed");

    let received = lookup_handler.received();
    assert_eq!(test_result.result, vec![other_peer]);
    assert_eq!(received.len(), 1);

    let (receiver, (sender, k)) = &received[0];
    assert_eq!(receiver, &test_result.remote_node);
    assert_eq!(sender, &test_result.local_node);
    assert_eq!(k, &key);
}

#[tokio::test]
async fn lookup_remote_peer_filter_out_invalid_address() {
    let runtime = TestRuntime::new("test".to_string());

    let mut key = [0u8; 40];
    rand::rng().fill_bytes(&mut key);

    let other_peer = PeerNode {
        id: NodeIdentifier {
            key: prost::bytes::Bytes::from(key.to_vec()),
        },
        endpoint: Endpoint::new("1.2.3.4".to_string(), 0, 0),
    };

    let invalid_peer = PeerNode {
        id: NodeIdentifier {
            key: prost::bytes::Bytes::from(key.to_vec()),
        },
        endpoint: Endpoint::new("0.0.0.0".to_string(), 0, 0),
    };

    let execute_fn = |kademlia_rpc: GrpcKademliaRPC, _local: PeerNode, remote: PeerNode| {
        let key = key.clone();
        async move { kademlia_rpc.lookup(&key, &remote).await.unwrap_or_default() }
    };

    let lookup_handler = TestLookupHandler::with_response(vec![other_peer.clone(), invalid_peer]);

    let (test_result, _ping_handler, lookup_handler) = run_two_nodes_test(
        &runtime,
        execute_fn,
        None,                 // Use default ping handler
        Some(lookup_handler), // Use lookup handler with response including invalid address
    )
    .await
    .expect("Test should succeed");

    let received = lookup_handler.received();
    assert_eq!(test_result.result, vec![other_peer]);
    assert_eq!(received.len(), 1);

    let (receiver, (sender, k)) = &received[0];
    assert_eq!(receiver, &test_result.remote_node);
    assert_eq!(sender, &test_result.local_node);
    assert_eq!(k, &key);
}

#[tokio::test]
async fn lookup_remote_peer_response_takes_too_long_get_empty_list() {
    let runtime = TestRuntime::new("test".to_string());

    let mut key = [0u8; 40];
    rand::rng().fill_bytes(&mut key);

    let execute_fn = |kademlia_rpc: GrpcKademliaRPC, _local: PeerNode, remote: PeerNode| {
        let key = key.clone();
        async move { kademlia_rpc.lookup(&key, &remote).await.unwrap_or_default() }
    };

    let lookup_handler_with_delay = TestLookupHandler::with_delay(Duration::from_secs(1));

    let (test_result, _ping_handler, lookup_handler) = run_two_nodes_test(
        &runtime,
        execute_fn,
        None,                            // Use default ping handler
        Some(lookup_handler_with_delay), // Use lookup handler with delay
    )
    .await
    .expect("Test should succeed");

    let received = lookup_handler.received();
    assert_eq!(test_result.result, Vec::<PeerNode>::new());
    assert_eq!(received.len(), 1);

    let (receiver, (sender, k)) = &received[0];
    assert_eq!(receiver, &test_result.remote_node);
    assert_eq!(sender, &test_result.local_node);
    assert_eq!(k, &key);
}

#[tokio::test]
async fn lookup_remote_peer_peer_is_not_listening_get_empty_list() {
    let runtime = TestRuntime::new("test".to_string());

    let mut key = [0u8; 40];
    rand::rng().fill_bytes(&mut key);

    let execute_fn = |kademlia_rpc: GrpcKademliaRPC, _local: PeerNode, remote: PeerNode| {
        let key = key.clone();
        async move { kademlia_rpc.lookup(&key, &remote).await.unwrap_or_default() }
    };

    // Use the remote dead variant - no server will be started
    let (test_result, _ping_handler, _lookup_handler) = run_two_nodes_test_remote_dead(
        &runtime, execute_fn, None, // Use default ping handler
        None, // Use default lookup handler
    )
    .await
    .expect("Test should succeed");

    assert_eq!(test_result.result, Vec::<PeerNode>::new());
}
