use std::sync::{Arc, Mutex};

use shared::rust::{Byte, ByteVector};

use crate::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash, state::rspace_exporter::RSpaceExporter,
};

//See rspace/src/main/scala/coop/rchain/rspace/state/exporters/RSpaceExporterItems.scala
pub struct StoreItems<KeyHash, Value> {
    pub items: Vec<(KeyHash, Value)>,
    pub last_path: Vec<(KeyHash, Option<Byte>)>,
}

pub struct RSpaceExporterItems;

impl RSpaceExporterItems {
    pub fn get_history(
        &self,
        exporter: Arc<Mutex<Box<dyn RSpaceExporter>>>,
        start_path: Vec<(Blake2b256Hash, Option<Byte>)>,
        skip: i32,
        take: i32,
    ) -> StoreItems<Blake2b256Hash, ByteVector> {
        let exporter = exporter.lock().unwrap();

        // Traverse and collect tuple space nodes
        let nodes = exporter.get_nodes(start_path, skip, take);

        // Get last entry / raise exception if empty (partial function)
        let last_entry = nodes.last().expect("EmptyHistoryException");

        // Load all history items / without leafs
        let history_keys: Vec<_> = nodes.iter().filter(|node| !node.is_leaf).collect();
        let keys: Vec<Blake2b256Hash> = history_keys
            .iter()
            .map(|key| key.hash.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        let items = exporter
            .get_history_items(keys)
            .expect("Failed to get history items in rspace_exporter_items");

        let mut last_path = last_entry.path.clone();
        last_path.push((last_entry.hash.clone(), None));
        StoreItems { items, last_path }
    }

    pub fn get_data(
        &self,
        exporter: Arc<Mutex<Box<dyn RSpaceExporter>>>,
        start_path: Vec<(Blake2b256Hash, Option<Byte>)>,
        skip: i32,
        take: i32,
    ) -> StoreItems<Blake2b256Hash, ByteVector> {
        let exporter = exporter.lock().unwrap();

        // Traverse and collect tuple space nodes
        let nodes = exporter.get_nodes(start_path, skip, take);

        // Get last entry / raise exception if empty (partial function)
        let last_entry = nodes.last().expect("EmptyHistoryException");

        // Load all data items / without history
        let data_keys: Vec<_> = nodes.iter().filter(|node| !node.is_leaf).collect();
        let keys: Vec<Blake2b256Hash> = data_keys
            .iter()
            .map(|key| key.hash.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        let items = exporter
            .get_data_items(keys)
            .expect("Failed to get history items in rspace_exporter_items");

        let mut last_path = last_entry.path.clone();
        last_path.push((last_entry.hash.clone(), None));
        StoreItems { items, last_path }
    }

    pub fn get_history_and_data(
        exporter: Arc<dyn RSpaceExporter>,
        start_path: Vec<(Blake2b256Hash, Option<Byte>)>,
        skip: i32,
        take: i32,
    ) -> (StoreItems<Blake2b256Hash, ByteVector>, StoreItems<Blake2b256Hash, ByteVector>) {
        // Traverse and collect tuple space nodes
        let nodes = exporter.get_nodes(start_path, skip, take);

        // Get last entry / raise exception if empty (partial function)
        let last_entry = nodes.last().expect("EmptyHistoryException");

        // Split by leaf nodes
        let (leafs, non_leafs): (Vec<_>, Vec<_>) = nodes.iter().partition(|node| node.is_leaf);
        let history_keys: Vec<Blake2b256Hash> = non_leafs
            .iter()
            .map(|key| key.hash.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        let data_keys: Vec<Blake2b256Hash> = leafs
            .iter()
            .map(|key| key.hash.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Load all history and data items
        let history_items = exporter
            .get_history_items(history_keys)
            .expect("Failed to get history items in rspace_exporter_items");
        let data_items = exporter
            .get_data_items(data_keys)
            .expect("Failed to get history items in rspace_exporter_items");

        // println!("\ndata_items in exporter_items: {:?}", data_items);

        let mut history_last_path = last_entry.path.clone();
        history_last_path.push((last_entry.hash.clone(), None));
        let history = StoreItems {
            items: history_items,
            last_path: history_last_path,
        };

        let mut data_last_path = last_entry.path.clone();
        data_last_path.push((last_entry.hash.clone(), None));
        let data = StoreItems {
            items: data_items,
            last_path: data_last_path,
        };

        (history, data)
    }
}
