// See comm/src/main/scala/coop/rchain/comm/discovery/KademliaNodeDiscovery.scala

use crate::rust::{
    discovery::node_discovery::NodeDiscovery,
    errors::CommError,
    peer_node::{NodeIdentifier, PeerNode},
};
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use std::{collections::HashSet, sync::Arc};

use super::{kademlia_rpc::KademliaRPC, kademlia_store::KademliaStore};

#[derive(Clone)]
pub struct KademliaNodeDiscovery<T: KademliaRPC> {
    node_id: NodeIdentifier,
    store: Arc<KademliaStore<T>>,
    rpc: Arc<T>,
}

#[async_trait::async_trait]
impl<T: KademliaRPC + Send + Sync + 'static> NodeDiscovery for KademliaNodeDiscovery<T> {
    async fn discover(&self) -> Result<(), CommError> {
        self.discover_raw(&self.node_id).await
    }

    fn peers(&self) -> Result<Vec<PeerNode>, CommError> {
        self.store.peers()
    }

    fn remove_peer(&self, peer: &PeerNode) -> Result<(), CommError> {
        self.store.remove(&peer.id.key)
    }
}

impl<T: KademliaRPC> KademliaNodeDiscovery<T> {
    pub fn new(store: Arc<KademliaStore<T>>, rpc: Arc<T>, node_id: NodeIdentifier) -> Self {
        Self {
            node_id,
            store,
            rpc,
        }
    }

    /**
     * Return up to `limit` candidate peers.
     *
     * Currently, this function determines the distances in the table that are
     * least populated and searches for more peers to fill those. It asks one
     * node for peers at one distance, then moves on to the next node and
     * distance. The queried nodes are not in any particular order. For now, this
     * function should be called with a relatively small `limit` parameter like
     * 10 to avoid making too many unproductive network calls.
     */
    async fn discover_raw(&self, id: &NodeIdentifier) -> Result<(), CommError> {
        let peers = self.store.peers()?;
        let dists = self.store.sparseness()?;

        // Shuffle the peers randomly
        let mut peer_list = peers;
        let mut rng = SmallRng::from_os_rng();
        peer_list.shuffle(&mut rng);

        let result = self
            .find(id, 10, &dists, &peer_list, &HashSet::new(), 0)
            .await?;

        // Update last seen for all discovered peers
        for peer in &result {
            self.store.update_last_seen(peer).await?;
        }

        Ok(())
    }

    pub fn peers(&self) -> Result<Vec<PeerNode>, CommError> {
        self.store.peers()
    }

    async fn find(
        &self,
        id: &NodeIdentifier,
        limit: usize,
        dists: &[usize],
        peer_set: &[PeerNode],
        potentials: &HashSet<PeerNode>,
        i: usize,
    ) -> Result<Vec<PeerNode>, CommError> {
        if !peer_set.is_empty() && potentials.len() < limit && i < dists.len() {
            let dist = dists[i];
            /*
             * The general idea is to ask a peer for its peers around a certain
             * distance from our own key. So, construct a key that first differs
             * from ours at bit position dist.
             */
            let mut target = id.key.to_vec();
            let byte_index = dist / 8;
            let different_bit = 1 << (dist % 8);
            target[byte_index] = target[byte_index] ^ different_bit; // A key at a distance dist from me

            let peers = self.rpc.lookup(&target, peer_set.first().unwrap()).await?;
            let filtered = self.filter(&peers, potentials, id)?;

            let mut new_potentials = potentials.clone();
            new_potentials.extend(filtered);

            Box::pin(self.find(id, limit, dists, &peer_set[1..], &new_potentials, i + 1)).await
        } else {
            Ok(potentials.iter().cloned().collect())
        }
    }

    fn filter(
        &self,
        peers: &[PeerNode],
        potentials: &HashSet<PeerNode>,
        id: &NodeIdentifier,
    ) -> Result<HashSet<PeerNode>, CommError> {
        let mut result = HashSet::new();

        for peer in peers {
            // Skip if already in potentials
            if potentials.contains(peer) {
                continue;
            }

            // Skip if peer has the same key as our node
            if peer.id.key == id.key {
                continue;
            }

            // Skip if peer is already in our store
            if self.store.find(&peer.id.key)?.is_some() {
                continue;
            }

            result.insert(peer.clone());
        }

        Ok(result)
    }
}
