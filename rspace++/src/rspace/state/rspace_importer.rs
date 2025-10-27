use rayon::{
    ThreadPoolBuilder,
    iter::{IntoParallelRefIterator, ParallelIterator},
};
use shared::rust::ByteVector;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash,
    history::cold_store::PersistedData,
    shared::{trie_exporter::KeyHash, trie_importer::TrieImporter},
    state::rspace_exporter::RSpaceExporterInstance,
};
use std::hash::Hash;

// See rspace/src/main/scala/coop/rchain/rspace/state/RSpaceImporter.scala
pub trait RSpaceImporter: TrieImporter + Send + Sync {
    fn get_history_item(&self, hash: KeyHash) -> Option<ByteVector>;
}

pub struct RSpaceImporterInstance;

impl RSpaceImporterInstance {
    pub fn validate_state_items(
        history_items: Vec<(Blake2b256Hash, ByteVector)>,
        data_items: Vec<(Blake2b256Hash, ByteVector)>,
        start_path: Vec<(Blake2b256Hash, Option<u8>)>,
        chunk_size: i32,
        skip: i32,
        get_from_history: Arc<dyn RSpaceImporter>,
    ) -> () {
        let received_history_size = history_items.len() as i32;
        let is_end = || received_history_size < chunk_size;

        // Validate history items size
        let validate_history_size = || {
            let size_is_valid = || received_history_size == chunk_size || is_end();
            if !size_is_valid() {
                panic!(
                    "RSpace Importer: Input size of history items is not valid. Expected chunk size {}, received {}.",
                    chunk_size, received_history_size
                )
            }
        };

        // Validate history hashes
        let get_and_validate_history_items = || {
            let mut validated_items = Vec::new();
            for (hash, trie_bytes) in history_items.clone() {
                let trie_hash = Blake2b256Hash::new(&trie_bytes);
                if hash == trie_hash {
                    validated_items.push((trie_hash.bytes(), trie_bytes));
                } else {
                    panic!(
                        "RSpace Importer: Trie hash does not match decoded trie, key: {}, decoded: {}",
                        hex::encode(hash.bytes()),
                        hex::encode(trie_hash.bytes())
                    );
                }
            }
            validated_items
        };

        // Validate data hashes
        let validate_data_items_hashes = {
            let pool = ThreadPoolBuilder::new().num_threads(64).build().unwrap();
            pool.install(|| {
                data_items.par_iter().try_for_each(|(hash, value_bytes)| {
                    let persisted_data: PersistedData = bincode::deserialize(value_bytes).unwrap();
                    let bytes = match persisted_data {
                        PersistedData::Joins(ref leaf) => bincode::serialize(leaf).unwrap(),
                        PersistedData::Data(ref leaf) => bincode::serialize(leaf).unwrap(),
                        PersistedData::Continuations(ref leaf) => bincode::serialize(leaf).unwrap(),
                    };
                    let data_hash = Blake2b256Hash::new(&bytes);
                    if *hash != data_hash {
                        Err(format!(
                            "Data hash does not match decoded data, key: {}, decoded: {}.",
                            hex::encode(hash.bytes()),
                            hex::encode(data_hash.bytes())
                        ))
                    } else {
                        Ok(())
                    }
                })
            })
        };

        let get_node = |st: HashMap<ByteVector, ByteVector>| {
            Arc::new(move |hash: &ByteVector| match st.get(hash) {
                Some(value) => Some(value.clone()),
                None => {
                    match get_from_history
                        .get_history_item(Blake2b256Hash::from_bytes(hash.to_vec()))
                    {
                        Some(bytes) => Some(bytes),
                        None => panic!(
                            "RSpace Importer: Trie hash not found in received items or in history store, hash: {}",
                            hex::encode(Blake2b256Hash::new(&hash).bytes())
                        ),
                    }
                }
            })
        };

        // Validate chunk size.
        validate_history_size();

        // Validate tries from received history items.
        let trie_map: HashMap<ByteVector, ByteVector> =
            get_and_validate_history_items().into_iter().collect();

        let get_node_function = get_node(trie_map);
        // Traverse trie and extract nodes / the same as in export. Nodes must match hashed keys.
        let nodes = RSpaceExporterInstance::traverse_history(
            start_path,
            skip,
            chunk_size,
            get_node_function,
        );

        // Extract history and data keys.
        let (leafs, non_leafs): (Vec<_>, Vec<_>) = nodes.into_iter().partition(|node| node.is_leaf);
        let history_keys: Vec<Blake2b256Hash> = non_leafs
            .into_iter()
            .map(|non_leaf| non_leaf.hash)
            .collect();
        let data_keys: Vec<Blake2b256Hash> = leafs.into_iter().map(|leaf| leaf.hash).collect();

        // Validate keys / cryptographic proof that store chunk is not corrupted or modified.
        let history_item_keys: Vec<Blake2b256Hash> =
            history_items.into_iter().map(|item| item.0).collect();
        // println!("\nhistory_item_keys: {:?}", history_item_keys);
        // println!("\nhistory_keys: {:?}", history_keys);
        // TODO: This might need to check ordering too
        if !check_same_elements(history_item_keys, history_keys) {
            panic!("RSpace Importer: History items are corrupted")
        }

        let data_item_keys: Vec<Blake2b256Hash> =
            data_items.clone().into_iter().map(|item| item.0).collect();
        // println!("\ndata_item_keys: {:?}", data_item_keys);
        // println!("\ndata_keys: {:?}", data_keys);
        // TODO: This might need to check ordering too
        if !check_same_elements(data_item_keys, data_keys) {
            panic!("RSpace Importer: Data items are corrupted")
        }

        validate_data_items_hashes.expect("RSpace Importer: Unable to validate data items hashes");
    }
}

fn check_same_elements<T: Hash + Eq>(vec1: Vec<T>, vec2: Vec<T>) -> bool {
    let set1: HashSet<_> = vec1.into_iter().collect();
    let set2: HashSet<_> = vec2.into_iter().collect();
    set1 == set2
}
