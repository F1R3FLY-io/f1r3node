// See comm/src/test/scala/coop/rchain/comm/discovery/KademliaSpec.scala

use std::sync::{Arc, Mutex};

use prost::bytes::Bytes;

use comm::rust::{
    discovery::{kademlia_rpc::KademliaRPC, peer_table::PeerTable},
    errors::CommError,
    peer_node::{Endpoint, NodeIdentifier, PeerNode},
};

use async_trait::async_trait;

/// Helper function to create a peer from binary string representation
fn create_peer(id: &str) -> PeerNode {
    let id_value = u8::from_str_radix(id, 2).unwrap();
    let bytes = vec![id_value];
    let node_id = NodeIdentifier {
        key: Bytes::from(bytes),
    };
    let endpoint = Endpoint::new(id.to_string(), 0, 0);
    PeerNode {
        id: node_id,
        endpoint,
    }
}

/// Mock KademliaRPC implementation for testing
struct KademliaRPCMock {
    returns: bool,
    pinged_peers: Arc<Mutex<Vec<PeerNode>>>,
}

impl KademliaRPCMock {
    fn new(returns: bool) -> Self {
        Self {
            returns,
            pinged_peers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn get_pinged_peers(&self) -> Vec<PeerNode> {
        self.pinged_peers.lock().unwrap().clone()
    }
}

#[async_trait]
impl KademliaRPC for KademliaRPCMock {
    async fn ping(&self, peer: &PeerNode) -> Result<bool, CommError> {
        self.pinged_peers.lock().unwrap().push(peer.clone());
        if self.returns {
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn lookup(&self, _key: &[u8], _peer: &PeerNode) -> Result<Vec<PeerNode>, CommError> {
        Ok(Vec::new())
    }
}

/// Helper function to get bucket entries at a specific distance
fn bucket_entries_at(distance: Option<usize>, table: &PeerTable<KademliaRPCMock>) -> Vec<PeerNode> {
    match distance {
        Some(d) => {
            if let Some(bucket) = table.table.get(d) {
                if let Ok(entries) = bucket.lock() {
                    entries.iter().map(|entry| entry.entry.clone()).collect()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        }
        None => Vec::new(),
    }
}

/// Helper to set up bucket 4 as full (with 3 peers: peer1, peer2, peer3)
async fn that_bucket_4_is_full<'a>(table: &PeerTable<'a, KademliaRPCMock>) {
    let peer1 = create_peer("00001000");
    let peer2 = create_peer("00001001");
    let peer3 = create_peer("00001010");

    table.update_last_seen(&peer1).await.unwrap();
    table.update_last_seen(&peer2).await.unwrap();
    table.update_last_seen(&peer3).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test peer definitions - these match the Scala version
    fn local() -> PeerNode {
        create_peer("00000001")
    }

    fn peer0() -> PeerNode {
        create_peer("00000010")
    }

    fn peer1() -> PeerNode {
        create_peer("00001000")
    }

    fn peer2() -> PeerNode {
        create_peer("00001001")
    }

    fn peer3() -> PeerNode {
        create_peer("00001010")
    }

    fn peer4() -> PeerNode {
        create_peer("00001100")
    }

    const DISTANCE_4: Option<usize> = Some(4);
    const DISTANCE_6: Option<usize> = Some(6);

    mod peer_table_with_1_byte_addresses_and_k_3 {
        use super::*;

        mod when_adding_a_peer_to_an_empty_table {
            use super::*;

            #[tokio::test]
            async fn should_add_it_to_a_bucket_according_to_its_distance() {
                // given
                let ping = KademliaRPCMock::new(true);
                let table = PeerTable::new(local().key().clone(), Some(3), None, &ping);
                assert_eq!(table.distance_other_peer(&peer0()), DISTANCE_6);

                // when
                table.update_last_seen(&peer0()).await.unwrap();

                // then
                assert_eq!(bucket_entries_at(DISTANCE_6, &table), vec![peer0()]);
            }

            #[tokio::test]
            async fn should_not_ping_the_peer() {
                // given
                let ping = KademliaRPCMock::new(true);
                let table = PeerTable::new(local().key().clone(), Some(3), None, &ping);

                // when
                table.update_last_seen(&peer0()).await.unwrap();

                // then
                assert_eq!(ping.get_pinged_peers(), Vec::<PeerNode>::new());
            }
        }

        mod when_adding_a_peer_when_that_peer_already_exists_but_with_different_ip {
            use super::*;

            #[tokio::test]
            async fn should_replace_peer_with_new_entry_the_one_with_new_ip() {
                // given
                let ping = KademliaRPCMock::new(true);
                let table = PeerTable::new(local().key().clone(), Some(3), None, &ping);
                table.update_last_seen(&peer1()).await.unwrap();

                // when
                let new_peer1 = PeerNode {
                    id: peer1().id.clone(),
                    endpoint: Endpoint::new("otherIP".to_string(), 0, 0),
                };
                table.update_last_seen(&new_peer1).await.unwrap();

                // then
                assert_eq!(bucket_entries_at(DISTANCE_4, &table), vec![new_peer1]);
            }

            #[tokio::test]
            async fn should_move_peer_to_the_end_of_the_bucket_meaning_its_been_seen_lately() {
                // given
                let ping = KademliaRPCMock::new(true);
                let table = PeerTable::new(local().key().clone(), Some(3), None, &ping);
                table.update_last_seen(&peer2()).await.unwrap();
                table.update_last_seen(&peer1()).await.unwrap();
                table.update_last_seen(&peer3()).await.unwrap();
                assert_eq!(
                    bucket_entries_at(DISTANCE_4, &table),
                    vec![peer2(), peer1(), peer3()]
                );

                // when
                let new_peer1 = PeerNode {
                    id: peer1().id.clone(),
                    endpoint: Endpoint::new("otherIP".to_string(), 0, 0),
                };
                table.update_last_seen(&new_peer1).await.unwrap();

                // then
                assert_eq!(
                    bucket_entries_at(DISTANCE_4, &table),
                    vec![peer2(), peer3(), new_peer1]
                );
            }
        }

        mod when_adding_a_peer_to_a_table_where_corresponding_bucket_is_filled_but_not_full {
            use super::*;

            #[tokio::test]
            async fn should_add_peer_to_the_end_of_the_bucket_meaning_its_been_seen_lately() {
                // given
                let ping = KademliaRPCMock::new(true);
                let table = PeerTable::new(local().key().clone(), Some(3), None, &ping);
                table.update_last_seen(&peer2()).await.unwrap();
                table.update_last_seen(&peer3()).await.unwrap();
                assert_eq!(
                    bucket_entries_at(DISTANCE_4, &table),
                    vec![peer2(), peer3()]
                );

                // when
                table.update_last_seen(&peer1()).await.unwrap();

                // then
                assert_eq!(
                    bucket_entries_at(DISTANCE_4, &table),
                    vec![peer2(), peer3(), peer1()]
                );
            }

            #[tokio::test]
            async fn no_peers_should_be_pinged() {
                // given
                let ping = KademliaRPCMock::new(true);
                let table = PeerTable::new(local().key().clone(), Some(3), None, &ping);
                table.update_last_seen(&peer2()).await.unwrap();
                table.update_last_seen(&peer3()).await.unwrap();
                assert_eq!(
                    bucket_entries_at(DISTANCE_4, &table),
                    vec![peer2(), peer3()]
                );

                // when
                table.update_last_seen(&peer1()).await.unwrap();

                // then
                assert_eq!(ping.get_pinged_peers(), Vec::<PeerNode>::new());
            }
        }

        mod when_adding_a_peer_to_a_table_where_corresponding_bucket_is_full {
            use super::*;

            #[tokio::test]
            async fn should_ping_the_oldest_peer_to_check_if_it_responds() {
                // given
                let ping = KademliaRPCMock::new(true);
                let table = PeerTable::new(local().key().clone(), Some(3), None, &ping);
                that_bucket_4_is_full(&table).await;

                // when
                table.update_last_seen(&peer4()).await.unwrap();

                // then
                assert_eq!(ping.get_pinged_peers(), vec![peer1()]);
            }

            mod and_oldest_peer_is_responding_to_ping {
                use super::*;

                #[tokio::test]
                async fn should_drop_the_new_peer() {
                    // given
                    let ping = KademliaRPCMock::new(true);
                    let table = PeerTable::new(local().key().clone(), Some(3), None, &ping);
                    that_bucket_4_is_full(&table).await;

                    // when
                    table.update_last_seen(&peer4()).await.unwrap();

                    // then
                    assert_eq!(
                        bucket_entries_at(DISTANCE_4, &table),
                        vec![peer2(), peer3(), peer1()]
                    );
                }
            }

            mod and_oldest_peer_is_not_responding_to_ping {
                use super::*;

                #[tokio::test]
                async fn should_add_the_new_peer_and_drop_the_oldest_one() {
                    // given
                    let ping = KademliaRPCMock::new(false);
                    let table = PeerTable::new(local().key().clone(), Some(3), None, &ping);
                    that_bucket_4_is_full(&table).await;

                    // when
                    table.update_last_seen(&peer4()).await.unwrap();

                    // then
                    assert_eq!(
                        bucket_entries_at(DISTANCE_4, &table),
                        vec![peer2(), peer3(), peer4()]
                    );
                }
            }
        }
    }
}
