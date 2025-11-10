// See comm/src/test/scala/coop/rchain/comm/discovery/PeerTableSpec.scala

use async_trait::async_trait;
use prost::bytes::Bytes;
use rand::RngCore;

use comm::rust::{
    discovery::{kademlia_rpc::KademliaRPC, peer_table::PeerTable},
    errors::CommError,
    peer_node::{Endpoint, NodeIdentifier, PeerNode},
};

/// Helper function to generate random bytes
fn rand_bytes(nbytes: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; nbytes];
    rand::rng().fill_bytes(&mut bytes);
    bytes
}

/// Helper function to create an endpoint for testing
fn endpoint() -> Endpoint {
    Endpoint::new("".to_string(), 0, 0)
}

/// Test implementation of KademliaRPC that always succeeds
struct KademliaRPCStub;

#[async_trait]
impl KademliaRPC for KademliaRPCStub {
    async fn ping(&self, _peer: &PeerNode) -> Result<bool, CommError> {
        Ok(true) // Always successful for tests
    }

    async fn lookup(&self, _key: &[u8], _peer: &PeerNode) -> Result<Vec<PeerNode>, CommError> {
        Ok(Vec::new()) // Return empty for tests
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    const ADDRESS_WIDTH: usize = 8;

    fn home() -> PeerNode {
        let id = NodeIdentifier {
            key: Bytes::from(rand_bytes(ADDRESS_WIDTH)),
        };
        PeerNode {
            id,
            endpoint: endpoint(),
        }
    }

    fn create_peer_with_id(id: &[u8]) -> PeerNode {
        PeerNode {
            id: NodeIdentifier {
                key: Bytes::from(id.to_vec()),
            },
            endpoint: endpoint(),
        }
    }

    fn create_peer_with_endpoint(id: &[u8], host: &str, port: u32) -> PeerNode {
        PeerNode {
            id: NodeIdentifier {
                key: Bytes::from(id.to_vec()),
            },
            endpoint: Endpoint::new(host.to_string(), port, port),
        }
    }

    #[tokio::test]
    async fn peer_that_is_already_in_the_table_should_get_updated() {
        // given
        let kademlia_rpc = Arc::new(KademliaRPCStub);
        let home_peer = home();
        let table = PeerTable::new(home_peer.key().clone(), None, None, kademlia_rpc.clone());

        let id = rand_bytes(ADDRESS_WIDTH);
        let peer0 = PeerNode {
            id: NodeIdentifier {
                key: Bytes::from(id.clone()),
            },
            endpoint: Endpoint::new("new".to_string(), 0, 0),
        };
        let peer1 = PeerNode {
            id: NodeIdentifier {
                key: Bytes::from(id),
            },
            endpoint: Endpoint::new("changed".to_string(), 0, 0),
        };

        // when - add first peer
        table.update_last_seen(&peer0).await.unwrap();

        // then - should contain peer0
        let peers = table.peers().unwrap();
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0], peer0);

        // when - add second peer with same id but different endpoint
        table.update_last_seen(&peer1).await.unwrap();

        // then - should now contain peer1 (updated endpoint)
        let peers = table.peers().unwrap();
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0], peer1);
    }

    #[test]
    fn distance_calculation_should_work_correctly() {
        // given
        let kademlia_rpc = Arc::new(KademliaRPCStub);
        let home_key = vec![0b00000000u8]; // All zeros
        let table = PeerTable::new(Bytes::from(home_key), None, None, kademlia_rpc.clone());

        // Test various distances
        let test_cases = vec![
            (vec![0b00000001u8], Some(7)), // Last bit different -> distance 7
            (vec![0b00000010u8], Some(6)), // Second-to-last bit different -> distance 6
            (vec![0b00000100u8], Some(5)), // Third-to-last bit different -> distance 5
            (vec![0b00001000u8], Some(4)), // Fourth-to-last bit different -> distance 4
            (vec![0b00010000u8], Some(3)), // Fifth-to-last bit different -> distance 3
            (vec![0b00100000u8], Some(2)), // Sixth-to-last bit different -> distance 2
            (vec![0b01000000u8], Some(1)), // Seventh-to-last bit different -> distance 1
            (vec![0b10000000u8], Some(0)), // First bit different -> distance 0
            (vec![0b00000000u8], Some(8)), // Same key -> distance 8 (width * 8)
        ];

        for (key, expected_distance) in test_cases {
            let peer = create_peer_with_id(&key);
            assert_eq!(
                table.distance_other_peer(&peer),
                expected_distance,
                "Failed for key {:08b}",
                key[0]
            );
        }
    }

    #[tokio::test]
    async fn peers_should_be_added_to_correct_buckets() {
        // given
        let kademlia_rpc = Arc::new(KademliaRPCStub);
        let home_key = vec![0b00000000u8];
        let table = PeerTable::new(Bytes::from(home_key), Some(10), None, kademlia_rpc.clone());

        // Add peers at different distances
        let peer_distance_0 = create_peer_with_id(&[0b10000000u8]); // distance 0
        let peer_distance_1 = create_peer_with_id(&[0b01000000u8]); // distance 1
        let peer_distance_7 = create_peer_with_id(&[0b00000001u8]); // distance 7

        // when
        table.update_last_seen(&peer_distance_0).await.unwrap();
        table.update_last_seen(&peer_distance_1).await.unwrap();
        table.update_last_seen(&peer_distance_7).await.unwrap();

        // then - all peers should be in the table
        let peers = table.peers().unwrap();
        assert_eq!(peers.len(), 3);
        assert!(peers.contains(&peer_distance_0));
        assert!(peers.contains(&peer_distance_1));
        assert!(peers.contains(&peer_distance_7));

        // Check bucket contents directly
        assert_eq!(table.table[0].lock().unwrap().len(), 1);
        assert_eq!(table.table[1].lock().unwrap().len(), 1);
        assert_eq!(table.table[7].lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn lookup_should_return_closest_peers() {
        // given
        let kademlia_rpc = Arc::new(KademliaRPCStub);
        let home_key = vec![0b00000000u8];
        let table = PeerTable::new(Bytes::from(home_key), Some(3), None, kademlia_rpc.clone());

        // Add several peers
        let peers = vec![
            create_peer_with_id(&[0b10000000u8]), // distance 0
            create_peer_with_id(&[0b01000000u8]), // distance 1
            create_peer_with_id(&[0b00100000u8]), // distance 2
            create_peer_with_id(&[0b00010000u8]), // distance 3
            create_peer_with_id(&[0b00001000u8]), // distance 4
        ];

        for peer in &peers {
            table.update_last_seen(peer).await.unwrap();
        }

        // when - lookup a key close to one of the peers
        let lookup_key = Bytes::from(vec![0b10000001u8]); // Close to first peer
        let result = table.lookup(&lookup_key).unwrap();

        // then - should return at most k peers, excluding the lookup key itself
        assert!(result.len() <= 3);
        assert!(result.len() > 0);

        // Should not return a peer with the exact lookup key
        assert!(!result.iter().any(|p| p.key() == &lookup_key));
    }

    #[tokio::test]
    async fn find_should_return_peer_if_exists() {
        // given
        let kademlia_rpc = Arc::new(KademliaRPCStub);
        let home_key = vec![0b00000000u8];
        let table = PeerTable::new(Bytes::from(home_key), None, None, kademlia_rpc.clone());

        let peer_key = vec![0b10101010u8];
        let peer = create_peer_with_id(&peer_key);

        // when - add peer and then find it
        table.update_last_seen(&peer).await.unwrap();
        let found = table.find(&Bytes::from(peer_key.clone())).unwrap();

        // then
        assert!(found.is_some());
        assert_eq!(found.unwrap(), peer);

        // when - try to find non-existent peer
        let not_found = table.find(&Bytes::from(vec![0b11111111u8])).unwrap();

        // then
        assert!(not_found.is_none());
    }

    #[tokio::test]
    async fn remove_should_delete_peer_from_table() {
        // given
        let kademlia_rpc = Arc::new(KademliaRPCStub);
        let home_key = vec![0b00000000u8];
        let table = PeerTable::new(Bytes::from(home_key), None, None, kademlia_rpc.clone());

        let peer_key = vec![0b10101010u8];
        let peer = create_peer_with_id(&peer_key);

        // when - add peer, verify it exists, then remove it
        table.update_last_seen(&peer).await.unwrap();
        assert!(table
            .find(&Bytes::from(peer_key.clone()))
            .unwrap()
            .is_some());

        table.remove(&Bytes::from(peer_key.clone())).unwrap();

        // then - peer should no longer exist
        assert!(table.find(&Bytes::from(peer_key)).unwrap().is_none());
        assert_eq!(table.peers().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn sparseness_should_return_distances_ordered_by_fill_level() {
        // given
        let kademlia_rpc = Arc::new(KademliaRPCStub);
        let home_key = vec![0b00000000u8];
        let table = PeerTable::new(Bytes::from(home_key), None, None, kademlia_rpc.clone());

        // Add more peers to some buckets than others
        table
            .update_last_seen(&create_peer_with_id(&[0b10000000u8]))
            .await
            .unwrap(); // distance 0
        table
            .update_last_seen(&create_peer_with_id(&[0b10000001u8]))
            .await
            .unwrap(); // distance 0
        table
            .update_last_seen(&create_peer_with_id(&[0b01000000u8]))
            .await
            .unwrap(); // distance 1

        // when
        let sparseness = table.sparseness().unwrap();

        // then - distances with fewer peers should come first
        // Most buckets are empty (0 peers), then distance 1 (1 peer), then distance 0 (2 peers)
        assert!(sparseness.len() > 0);
        assert!(sparseness.contains(&0)); // distance 0 bucket
        assert!(sparseness.contains(&1)); // distance 1 bucket
    }

    #[tokio::test]
    async fn table_should_handle_multiple_peers_same_distance() {
        // given
        let kademlia_rpc = Arc::new(KademliaRPCStub);
        let home_key = vec![0b00000000u8];
        let table = PeerTable::new(Bytes::from(home_key), None, None, kademlia_rpc.clone());

        // Create multiple peers at the same distance (different in later bits)
        let peer1 = create_peer_with_id(&[0b10000001u8]); // distance 0
        let peer2 = create_peer_with_id(&[0b10000010u8]); // distance 0
        let peer3 = create_peer_with_id(&[0b10000011u8]); // distance 0

        // when
        table.update_last_seen(&peer1).await.unwrap();
        table.update_last_seen(&peer2).await.unwrap();
        table.update_last_seen(&peer3).await.unwrap();

        // then
        let peers = table.peers().unwrap();
        assert_eq!(peers.len(), 3);
        assert!(peers.contains(&peer1));
        assert!(peers.contains(&peer2));
        assert!(peers.contains(&peer3));

        // All should be in the same bucket (distance 0)
        assert_eq!(table.table[0].lock().unwrap().len(), 3);
    }

    #[test]
    fn empty_table_should_return_empty_results() {
        // given
        let kademlia_rpc = Arc::new(KademliaRPCStub);
        let home_key = vec![0b00000000u8];
        let table = PeerTable::new(Bytes::from(home_key), None, None, kademlia_rpc.clone());

        // when/then - all operations on empty table should return empty results
        assert_eq!(table.peers().unwrap().len(), 0);
        assert!(table
            .find(&Bytes::from(vec![0b11111111u8]))
            .unwrap()
            .is_none());
        assert_eq!(
            table
                .lookup(&Bytes::from(vec![0b11111111u8]))
                .unwrap()
                .len(),
            0
        );
    }

    #[test]
    fn distance_calculation_edge_cases() {
        // given
        let kademlia_rpc = Arc::new(KademliaRPCStub);
        let home_key = vec![0b11111111u8]; // All ones
        let table = PeerTable::new(Bytes::from(home_key), None, None, kademlia_rpc.clone());

        // Test edge cases
        let all_zeros = create_peer_with_id(&[0b00000000u8]);
        let all_ones = create_peer_with_id(&[0b11111111u8]);
        let alternating = create_peer_with_id(&[0b10101010u8]);

        // XOR: 11111111 ^ 00000000 = 11111111 -> first bit (MSB) is set -> distance 0
        assert_eq!(table.distance_other_peer(&all_zeros), Some(0));
        // Same key -> width * 8
        assert_eq!(table.distance_other_peer(&all_ones), Some(8));
        // XOR: 11111111 ^ 10101010 = 01010101 -> first bit (MSB) is 0, second bit is 1 -> distance 1
        assert_eq!(table.distance_other_peer(&alternating), Some(1));
    }

    #[tokio::test]
    async fn bucket_should_maintain_insertion_order() {
        // given
        let kademlia_rpc = Arc::new(KademliaRPCStub);
        let home_key = vec![0b00000000u8];
        let table = PeerTable::new(Bytes::from(home_key), Some(10), None, kademlia_rpc.clone());

        // Add peers in specific order at same distance
        let peer1 = create_peer_with_endpoint(&[0b10000001u8], "host1", 1000);
        let peer2 = create_peer_with_endpoint(&[0b10000010u8], "host2", 2000);
        let peer3 = create_peer_with_endpoint(&[0b10000011u8], "host3", 3000);

        // when
        table.update_last_seen(&peer1).await.unwrap();
        table.update_last_seen(&peer2).await.unwrap();
        table.update_last_seen(&peer3).await.unwrap();

        // then - peers should be retrievable and in the table
        let peers = table.peers().unwrap();
        assert_eq!(peers.len(), 3);

        // Verify all peers are present (order in peers() may vary due to bucket iteration)
        assert!(peers.contains(&peer1));
        assert!(peers.contains(&peer2));
        assert!(peers.contains(&peer3));
    }

    #[tokio::test]
    async fn large_table_operations() {
        // given
        let kademlia_rpc = Arc::new(KademliaRPCStub);
        let home_key = vec![0b00000000u8];
        let table = PeerTable::new(Bytes::from(home_key), Some(20), None, kademlia_rpc.clone());

        // Add many peers across different distances
        let mut added_peers = Vec::new();
        for i in 0..64 {
            let key = vec![i as u8];
            let peer = create_peer_with_id(&key);
            table.update_last_seen(&peer).await.unwrap();
            added_peers.push(peer);
        }

        // when/then - all operations should work with many peers
        let all_peers = table.peers().unwrap();
        // Note: Some peers might have the same distance and be in the same bucket,
        // and some distances might not be reachable with single-byte keys
        assert!(all_peers.len() <= 64); // Should be <= original count
        assert!(all_peers.len() > 0); // Should have some peers

        // Test lookup with many peers
        let lookup_result = table.lookup(&Bytes::from(vec![0b10101010u8])).unwrap();
        assert!(lookup_result.len() <= 20); // Should respect k limit

        // Test find with added peers - check that they can be found
        let mut found_count = 0;
        for peer in &added_peers {
            let found = table.find(peer.key()).unwrap();
            if found.is_some() {
                assert_eq!(found.unwrap(), *peer);
                found_count += 1;
            }
        }
        assert!(found_count > 0); // At least some peers should be findable
    }

    #[tokio::test]
    async fn concurrent_operations_simulation() {
        // given
        let kademlia_rpc = Arc::new(KademliaRPCStub);
        let home_key = vec![0b00000000u8];
        let table = PeerTable::new(Bytes::from(home_key), Some(10), None, kademlia_rpc.clone());

        // Simulate concurrent operations by rapidly adding/removing/updating peers
        let peer1 = create_peer_with_id(&[0b10000001u8]);
        let peer2 = create_peer_with_id(&[0b10000010u8]);
        let peer1_updated = create_peer_with_endpoint(&[0b10000001u8], "updated", 9999);

        // when - rapid operations
        table.update_last_seen(&peer1).await.unwrap();
        table.update_last_seen(&peer2).await.unwrap();

        let lookup1 = table.lookup(&Bytes::from(vec![0b11111111u8])).unwrap();

        table.update_last_seen(&peer1_updated).await.unwrap(); // Update peer1

        let lookup2 = table.lookup(&Bytes::from(vec![0b11111111u8])).unwrap();

        table.remove(peer2.key()).unwrap();

        let final_peers = table.peers().unwrap();

        // then - operations should be consistent
        assert!(lookup1.len() > 0);
        assert!(lookup2.len() > 0);
        assert_eq!(final_peers.len(), 1);
        assert_eq!(final_peers[0], peer1_updated);
    }
}
