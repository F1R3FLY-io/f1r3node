// See shared/src/main/scala/coop/rchain/state/TrieExporter.scala

use crate::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use shared::rust::Byte;
use shared::rust::store::key_value_store::KvStoreError;

// Type of the key to uniquely defines the trie / in RSpace this is the hash of the trie
pub type KeyHash = Blake2b256Hash;

// Type of the full path to the node
// - it contains parent nodes with indexes and node itself at the end
pub type NodePath = Vec<(KeyHash, Option<Byte>)>;

pub type Value = Vec<u8>;

// Defines basic operation to traverse tries and convert to path indexed list
pub trait TrieExporter {
    // Get trie nodes with offset from start path and number of nodes
    // - skipping nodes can be expensive as taking nodes
    fn get_nodes(&self, start_path: NodePath, skip: i32, take: i32) -> Vec<TrieNode<KeyHash>>;

    // Get history values / from branch nodes in the trie
    fn get_history_items(&self, keys: Vec<KeyHash>) -> Result<Vec<(KeyHash, Value)>, KvStoreError>;

    // Get data values / from leaf nodes in the trie
    fn get_data_items(&self, keys: Vec<KeyHash>) -> Result<Vec<(KeyHash, Value)>, KvStoreError>;
}

#[derive(Clone)]
pub struct TrieNode<KeyHash> {
    pub hash: KeyHash,
    pub is_leaf: bool,
    pub path: Vec<(KeyHash, Option<u8>)>,
}

impl<KeyHash> TrieNode<KeyHash> {
    pub fn new(hash: KeyHash, is_leaf: bool, path: Vec<(KeyHash, Option<u8>)>) -> Self {
        TrieNode {
            hash,
            is_leaf,
            path,
        }
    }
}
