// See comm/src/main/scala/coop/rchain/comm/discovery/PeerTable.scala

use std::sync::{Arc, Mutex};

use prost::bytes::Bytes;

use crate::rust::{errors::CommError, peer_node::PeerNode};

use super::kademlia_rpc::KademliaRPC;

// Lookup table for log 2 (highest bit set) of an 8-bit unsigned
// integer. Entry 0 is unused.
static DLUT: [u8; 256] = [
    // 4 elements
    0, 7, 6, 6, // 4 elements of value 5
    5, 5, 5, 5, // 8 elements of value 4
    4, 4, 4, 4, 4, 4, 4, 4, // 16 elements of value 3
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, // 32 elements of value 2
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    // 64 elements of value 1
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // 128 elements of value 0
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

#[derive(Clone)]
pub struct PeerTableEntry {
    pub entry: PeerNode,
    pub pinging: bool,
    pub key: Bytes,
}

impl PeerTableEntry {
    pub fn new(entry: PeerNode) -> Self {
        let key = entry.key().clone();
        Self {
            entry,
            pinging: false,
            key,
        }
    }
}

/** `PeerTable` implements the routing table used in the Kademlia
 * network discovery and routing protocol.
 *
 */
pub struct PeerTable<T: KademliaRPC> {
    local_key: Bytes,
    k: u32,
    /// Concurrency factor: system allows up to alpha outstanding network
    /// requests at a time. Currently unused in basic PeerTable operations,
    /// but would be used in full Kademlia iterative lookup algorithms
    /// for controlling parallel network requests to avoid overwhelming the network.
    alpha: u32,
    kademlia_rpc: Arc<T>,
    width: usize,
    pub table: Vec<Arc<Mutex<Vec<PeerTableEntry>>>>,
}

impl<T: KademliaRPC> PeerTable<T> {
    pub fn new(local_key: Bytes, k: Option<u32>, alpha: Option<u32>, kademlia_rpc: Arc<T>) -> Self {
        let width = local_key.len();

        Self {
            local_key,
            k: k.unwrap_or(20),
            alpha: alpha.unwrap_or(3),
            kademlia_rpc,
            width,
            table: (0..8 * width)
                .map(|_| Arc::new(Mutex::new(Vec::new())))
                .collect(),
        }
    }

    /// Get the concurrency factor (alpha) for network operations
    pub fn alpha(&self) -> u32 {
        self.alpha
    }

    /** Computes Kademlia XOR distance.
     *
     * Returns the length of the longest common prefix in bits between
     * the two sequences `a` and `b`. As in Ethereum's implementation,
     * "closer" nodes have higher distance values.
     *
     * @return `Some(Int)` if `a` and `b` are comparable in this table,
     * `None` otherwise.
     */
    fn distance(&self, a: &Bytes, b: &Bytes) -> Option<usize> {
        if a.len() != self.width || b.len() != self.width {
            return None;
        }

        for idx in 0..self.width {
            let xor_result = a[idx] ^ b[idx];
            if xor_result != 0 {
                return Some(8 * idx + DLUT[xor_result as usize] as usize);
            }
        }

        Some(8 * self.width)
    }

    pub fn distance_other_key(&self, other: &Bytes) -> Option<usize> {
        self.distance(&self.local_key, other)
    }

    pub fn distance_other_peer(&self, other: &PeerNode) -> Option<usize> {
        self.distance(&self.local_key, other.key())
    }

    /// Update the last seen time for a peer in the routing table.
    /// This function implements the same logic as the Scala version.
    pub async fn update_last_seen(&self, peer: &PeerNode) -> Result<(), CommError> {
        // Find appropriate bucket
        let bucket = self.get_bucket(peer)?;

        if let Some(bucket_mutex) = bucket {
            // Try to add/update or pick old peer
            let old_peer_candidate = self.add_update_or_pick_old_peer(&bucket_mutex, peer)?;

            // If we have an old peer to ping, do so
            if let Some(old_entry) = old_peer_candidate {
                self.ping_and_update(&bucket_mutex, peer, old_entry).await?;
            }
        }

        Ok(())
    }

    /// Find the appropriate bucket for a peer
    fn get_bucket(
        &self,
        peer: &PeerNode,
    ) -> Result<Option<&Arc<Mutex<Vec<PeerTableEntry>>>>, CommError> {
        let dist = self.distance(&self.local_key, peer.key());

        match dist {
            Some(d) if d < 8 * self.width => {
                if let Some(bucket) = self.table.get(d) {
                    Ok(Some(bucket))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    /// Add/update peer or pick an old peer to ping
    fn add_update_or_pick_old_peer(
        &self,
        bucket: &Arc<Mutex<Vec<PeerTableEntry>>>,
        peer: &PeerNode,
    ) -> Result<Option<PeerTableEntry>, CommError> {
        let mut entries = bucket.lock().map_err(|_| {
            CommError::InternalCommunicationError("Failed to acquire bucket lock".to_string())
        })?;

        // Check if peer already exists
        if let Some(pos) = entries.iter().position(|entry| entry.key == *peer.key()) {
            // Remove old entry and add new one (move to back - newest position)
            entries.remove(pos);
            entries.push(PeerTableEntry::new(peer.clone()));
            return Ok(None);
        }

        // No existing entry - check if we can add without exceeding k
        if entries.len() < self.k as usize {
            entries.push(PeerTableEntry::new(peer.clone()));
            return Ok(None);
        }

        // Bucket is full - find first entry that isn't being pinged
        for entry in entries.iter_mut() {
            if !entry.pinging {
                entry.pinging = true;
                return Ok(Some(entry.clone()));
            }
        }

        // All entries are being pinged
        Ok(None)
    }

    /// Ping an old peer and update the bucket based on response
    async fn ping_and_update(
        &self,
        bucket: &Arc<Mutex<Vec<PeerTableEntry>>>,
        new_peer: &PeerNode,
        mut old_entry: PeerTableEntry,
    ) -> Result<(), CommError> {
        // Ping the old peer and check if it returns true (successful ping)
        let ping_successful = self
            .kademlia_rpc
            .ping(&old_entry.entry)
            .await
            .map(|result| result)
            .unwrap_or(false);

        let mut entries = bucket.lock().map_err(|_| {
            CommError::InternalCommunicationError("Failed to acquire bucket lock".to_string())
        })?;

        // Remove the old entry from the bucket
        if let Some(pos) = entries.iter().position(|entry| entry.key == old_entry.key) {
            entries.remove(pos);
        }

        if ping_successful {
            // Old peer responded - move it to back (newest position) and clear pinging flag
            old_entry.pinging = false;
            entries.push(old_entry);
        } else {
            // Old peer didn't respond - add new peer instead
            entries.push(PeerTableEntry::new(new_peer.clone()));
        }

        Ok(())
    }

    /// Remove a peer with the given key.
    pub fn remove(&self, key: &Bytes) -> Result<(), CommError> {
        let dist = self.distance(&self.local_key, key);

        if let Some(index) = dist {
            if index < 8 * self.width {
                if let Some(bucket) = self.table.get(index) {
                    let mut entries = bucket.lock().map_err(|_| {
                        CommError::InternalCommunicationError(
                            "Failed to acquire bucket lock".to_string(),
                        )
                    })?;

                    if let Some(pos) = entries.iter().position(|entry| entry.key == *key) {
                        entries.remove(pos);
                    }
                }
            }
        }

        Ok(())
    }

    /// Return the `k` nodes closest to `key` that this table knows about,
    /// sorted ascending by distance to `key`.
    pub fn lookup(&self, key: &Bytes) -> Result<Vec<PeerNode>, CommError> {
        let dist = self.distance(&self.local_key, key);

        match dist {
            Some(index) => {
                let mut entries = Vec::new();

                // Look in buckets from index onwards
                for i in index..8 * self.width {
                    if entries.len() >= self.k as usize {
                        break;
                    }

                    if let Some(bucket) = self.table.get(i) {
                        let bucket_entries = bucket.lock().map_err(|_| {
                            CommError::InternalCommunicationError(
                                "Failed to acquire bucket lock".to_string(),
                            )
                        })?;

                        for entry in bucket_entries.iter() {
                            if entry.entry.key() != key {
                                entries.push(entry.entry.clone());
                            }
                        }
                    }
                }

                // Look in buckets before index
                for i in (0..index).rev() {
                    if entries.len() >= self.k as usize {
                        break;
                    }

                    if let Some(bucket) = self.table.get(i) {
                        let bucket_entries = bucket.lock().map_err(|_| {
                            CommError::InternalCommunicationError(
                                "Failed to acquire bucket lock".to_string(),
                            )
                        })?;

                        for entry in bucket_entries.iter() {
                            entries.push(entry.entry.clone());
                        }
                    }
                }

                // Sort by distance to key
                entries.sort_by(|a, b| {
                    let dist_a = self.distance(key, a.key()).unwrap_or(0);
                    let dist_b = self.distance(key, b.key()).unwrap_or(0);
                    dist_b.cmp(&dist_a) // Descending order (closer nodes have higher distance)
                });

                // Take at most k entries
                entries.truncate(self.k as usize);
                Ok(entries)
            }
            None => Ok(Vec::new()),
        }
    }

    /// Return `Some[PeerNode]` if `key` names an entry in the table.
    pub fn find(&self, key: &Bytes) -> Result<Option<PeerNode>, CommError> {
        let dist = self.distance(&self.local_key, key);

        if let Some(d) = dist {
            if let Some(bucket) = self.table.get(d) {
                let entries = bucket.lock().map_err(|_| {
                    CommError::InternalCommunicationError(
                        "Failed to acquire bucket lock".to_string(),
                    )
                })?;

                for entry in entries.iter() {
                    if entry.entry.key() == key {
                        return Ok(Some(entry.entry.clone()));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Return a sequence of all the `PeerNode`s in the table.
    pub fn peers(&self) -> Result<Vec<PeerNode>, CommError> {
        let mut all_peers = Vec::new();

        for bucket in &self.table {
            let entries = bucket.lock().map_err(|_| {
                CommError::InternalCommunicationError("Failed to acquire bucket lock".to_string())
            })?;

            for entry in entries.iter() {
                all_peers.push(entry.entry.clone());
            }
        }

        Ok(all_peers)
    }

    /// Return all distances in order from least to most filled.
    ///
    /// Optionally, ignore any distance closer than [[limit]].
    pub fn sparseness(&self) -> Result<Vec<usize>, CommError> {
        let mut distance_sizes: Vec<(usize, usize)> = Vec::new();

        for (i, bucket) in self.table.iter().take(256).enumerate() {
            let entries = bucket.lock().map_err(|_| {
                CommError::InternalCommunicationError("Failed to acquire bucket lock".to_string())
            })?;

            distance_sizes.push((entries.len(), i));
        }

        // Sort by size (least filled first)
        distance_sizes.sort_by_key(|(size, _)| *size);

        Ok(distance_sizes.into_iter().map(|(_, idx)| idx).collect())
    }
}
