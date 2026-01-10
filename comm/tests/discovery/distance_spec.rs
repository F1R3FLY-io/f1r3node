// See comm/src/test/scala/coop/rchain/comm/discovery/DistanceSpec.scala

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

/// Helper function to encode bytes as hex string (equivalent to Base16.encode)
fn encode_hex(bytes: &[u8]) -> String {
    hex::encode(bytes)
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

// Test helper functions for creating one-off keys
fn one_offs(key: &[u8]) -> Vec<Vec<u8>> {
    let width = key.len();
    let mut result = Vec::new();

    for i in 0..width {
        for j in (0..=7).rev() {
            let mut k1 = vec![0u8; key.len()];
            k1.copy_from_slice(key);
            k1[i] = k1[i] ^ (1 << j);
            result.push(k1);
        }
    }

    result
}

fn test_key(key: &[u8]) -> bool {
    let kademlia_rpc = std::sync::Arc::new(KademliaRPCStub);
    let table = PeerTable::new(Bytes::from(key.to_vec()), None, None, kademlia_rpc.clone());
    let one_off_keys = one_offs(key);

    let distances: Vec<Option<usize>> = one_off_keys
        .iter()
        .map(|k| table.distance_other_key(&Bytes::from(k.clone())))
        .collect();

    let expected: Vec<Option<usize>> = (0..8 * key.len()).map(Some).collect();

    distances == expected
}

// Macro to generate test modules for different widths
macro_rules! mod_tests {
    ($width:expr) => {
        paste::paste! {
            mod [<width_ $width>] {
                use super::*;

                #[test]
                fn [<node_with_key_all_zeroes_should_compute_distance_correctly_ $width>]() {
                    let k0 = vec![0u8; $width];
                    assert!(test_key(&k0), "Key all zeroes ({})", encode_hex(&k0));
                }

                #[test]
                fn [<node_with_key_all_ones_should_compute_distance_correctly_ $width>]() {
                    let k1 = vec![0xff; $width];
                    assert!(test_key(&k1), "Key all ones ({})", encode_hex(&k1));
                }

                #[test]
                fn [<node_with_random_key_should_compute_distance_correctly_ $width>]() {
                    let kr = rand_bytes($width);
                    assert!(test_key(&kr), "Random key ({})", encode_hex(&kr));
                }

                #[test]
                fn [<empty_table_of_width_should_have_no_peers_ $width>]() {
                    let kademlia_rpc = std::sync::Arc::new(KademliaRPCStub);
                    let kr = rand_bytes($width);
                    let table = PeerTable::new(Bytes::from(kr), None, None, kademlia_rpc.clone());

                    let peers = table.peers().unwrap();
                    assert_eq!(peers.len(), 0);
                }

                #[test]
                fn [<empty_table_should_return_no_peers_ $width>]() {
                    let kademlia_rpc = std::sync::Arc::new(KademliaRPCStub);
                    let kr = rand_bytes($width);
                    let table = PeerTable::new(Bytes::from(kr), None, None, kademlia_rpc.clone());

                    let peers = table.peers().unwrap();
                    assert_eq!(peers.len(), 0);
                }

                #[test]
                fn [<empty_table_should_return_no_values_on_lookup_ $width>]() {
                    let kademlia_rpc = std::sync::Arc::new(KademliaRPCStub);
                    let kr = rand_bytes($width);
                    let table = PeerTable::new(Bytes::from(kr), None, None, kademlia_rpc);

                    let lookup_result = table.lookup(&Bytes::from(rand_bytes($width))).unwrap();
                    assert_eq!(lookup_result.len(), 0);
                }

                #[tokio::test]
                async fn [<table_should_add_key_at_most_once_ $width>]() {
                    let kademlia_rpc = std::sync::Arc::new(KademliaRPCStub);
                    let kr = rand_bytes($width);
                    let table = PeerTable::new(Bytes::from(kr.clone()), Some(20), None, kademlia_rpc.clone());
                    let to_add = one_offs(&kr)[0].clone();
                    let dist = table.distance_other_key(&Bytes::from(to_add.clone())).unwrap();

                    for _i in 1..=10 {
                        let peer = PeerNode {
                            id: NodeIdentifier { key: Bytes::from(to_add.clone()) },
                            endpoint: endpoint(),
                        };
                        table.update_last_seen(&peer).await.unwrap();

                        // Check that the bucket at distance `dist` has exactly 1 peer
                        let peers = table.peers().unwrap();
                        let peers_at_distance: Vec<_> = peers
                            .iter()
                            .filter(|p| table.distance_other_peer(p) == Some(dist))
                            .collect();
                        assert_eq!(peers_at_distance.len(), 1);
                    }
                }

                #[tokio::test]
                async fn [<table_with_peers_at_all_distances_should_have_no_empty_buckets_ $width>]() {
                    let kademlia_rpc = std::sync::Arc::new(KademliaRPCStub);
                    let kr = rand_bytes($width);
                    let table = PeerTable::new(Bytes::from(kr.clone()), Some(20), None, kademlia_rpc.clone());

                    for k in one_offs(&kr) {
                        let peer = PeerNode {
                            id: NodeIdentifier { key: Bytes::from(k) },
                            endpoint: endpoint(),
                        };
                        table.update_last_seen(&peer).await.unwrap();
                    }

                    // Check that we have peers at all distances
                    let peers = table.peers().unwrap();
                    assert_eq!(peers.len(), 8 * $width);
                }

                #[tokio::test]
                async fn [<table_should_return_min_k_peers_on_lookup_ $width>]() {
                    let kademlia_rpc = std::sync::Arc::new(KademliaRPCStub);
                    let kr = rand_bytes($width);
                    let table = PeerTable::new(Bytes::from(kr.clone()), Some(20), None, kademlia_rpc.clone());
                    let kr_one_offs = one_offs(&kr);

                    for k in &kr_one_offs {
                        let peer = PeerNode {
                            id: NodeIdentifier { key: Bytes::from(k.clone()) },
                            endpoint: endpoint(),
                        };
                        table.update_last_seen(&peer).await.unwrap();
                    }

                    let random_key = rand_bytes($width);
                    let expected = if kr_one_offs.iter().any(|k| k == &random_key) {
                        8 * $width - 1
                    } else {
                        8 * $width
                    };

                    let lookup_result = table.lookup(&Bytes::from(random_key)).unwrap();
                    assert_eq!(lookup_result.len(), std::cmp::min(20, expected));
                }

                #[tokio::test]
                async fn [<table_should_not_return_sought_peer_on_lookup_ $width>]() {
                    let kademlia_rpc = std::sync::Arc::new(KademliaRPCStub);
                    let kr = rand_bytes($width);
                    let table = PeerTable::new(Bytes::from(kr.clone()), Some(20), None, kademlia_rpc.clone());

                    for k in one_offs(&kr) {
                        let peer = PeerNode {
                            id: NodeIdentifier { key: Bytes::from(k) },
                            endpoint: endpoint(),
                        };
                        table.update_last_seen(&peer).await.unwrap();
                    }

                    let peers = table.peers().unwrap();
                    if let Some(target) = peers.get($width * 4) {
                        let resp = table.lookup(target.key()).unwrap();
                        assert!(resp.iter().all(|p| p.key() != target.key()));
                    }
                }

                #[tokio::test]
                async fn [<table_should_return_all_peers_when_sequenced_ $width>]() {
                    let kademlia_rpc = std::sync::Arc::new(KademliaRPCStub);
                    let kr = rand_bytes($width);
                    let table = PeerTable::new(Bytes::from(kr.clone()), Some(20), None, kademlia_rpc.clone());

                    for k in one_offs(&kr) {
                        let peer = PeerNode {
                            id: NodeIdentifier { key: Bytes::from(k) },
                            endpoint: endpoint(),
                        };
                        table.update_last_seen(&peer).await.unwrap();
                    }

                    let peers = table.peers().unwrap();
                    assert_eq!(peers.len(), 8 * $width);
                }

                #[tokio::test]
                async fn [<table_should_find_each_added_peer_ $width>]() {
                    let kademlia_rpc = std::sync::Arc::new(KademliaRPCStub);
                    let kr = rand_bytes($width);
                    let table = PeerTable::new(Bytes::from(kr.clone()), Some(20), None, kademlia_rpc.clone());

                    for k in one_offs(&kr) {
                        let peer = PeerNode {
                            id: NodeIdentifier { key: Bytes::from(k.clone()) },
                            endpoint: endpoint(),
                        };
                        table.update_last_seen(&peer).await.unwrap();
                    }

                    for k in one_offs(&kr) {
                        let expected_peer = PeerNode {
                            id: NodeIdentifier { key: Bytes::from(k.clone()) },
                            endpoint: endpoint(),
                        };
                        let found = table.find(&Bytes::from(k)).unwrap();
                        assert_eq!(found, Some(expected_peer));
                    }
                }
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_node_of_width_n_bytes_should_have_distance_to_itself_equal_to_8n() {
        let kademlia_rpc = std::sync::Arc::new(KademliaRPCStub);

        for i in 1..=64 {
            let home_key = Bytes::from(rand_bytes(i));
            let home = PeerNode {
                id: NodeIdentifier {
                    key: home_key.clone(),
                },
                endpoint: endpoint(),
            };
            let table = PeerTable::new(home_key.clone(), None, None, kademlia_rpc.clone());

            let distance = table.distance_other_peer(&home);
            assert_eq!(distance, Some(8 * i));
        }
    }

    // Generate test modules for different widths using macro
    mod_tests!(1);
    mod_tests!(2);
    mod_tests!(4);
    mod_tests!(8);
    mod_tests!(16);
    mod_tests!(32);
    mod_tests!(64);
    mod_tests!(128);
    mod_tests!(256);
}
