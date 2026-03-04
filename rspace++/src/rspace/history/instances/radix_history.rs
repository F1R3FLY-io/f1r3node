// See rspace/src/main/scala/coop/rchain/rspace/history/instances/RadixHistory.
// scala

use std::collections::HashSet;
use std::sync::Arc;

use shared::rust::ByteVector;
use shared::rust::store::key_value_store::KeyValueStore;

use crate::rspace::errors::HistoryError;
use crate::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use crate::rspace::history::history::History;
use crate::rspace::history::history_action::{HistoryAction, HistoryActionTrait};
use crate::rspace::history::radix_tree::{Node, RadixTreeImpl, empty_node, hash_node};

pub struct RadixHistory {
    root_hash: Blake2b256Hash,
    root_node: Node,
    imple: RadixTreeImpl,
    store: Arc<dyn KeyValueStore>,
}

impl RadixHistory {
    pub fn create(
        root: Blake2b256Hash,
        store: Arc<dyn KeyValueStore>,
    ) -> Result<RadixHistory, HistoryError> {
        let imple = RadixTreeImpl::new(store.clone());
        let node = imple.load_node(root.bytes(), Some(true))?;

        Ok(RadixHistory {
            root_hash: root,
            root_node: node,
            imple,
            store,
        })
    }

    pub fn create_store(store: Arc<dyn KeyValueStore>) -> Arc<dyn KeyValueStore> { store }

    pub fn empty_root_node_hash() -> Blake2b256Hash {
        let node_hash_bytes = hash_node(&empty_node()).0;
        let node_hash = Blake2b256Hash::from_bytes(node_hash_bytes);
        node_hash
    }

    fn has_no_duplicates(&self, actions: &Vec<HistoryAction>) -> bool {
        let keys: HashSet<_> = actions.iter().map(|action| action.key()).collect();
        keys.len() == actions.len()
    }
}

impl History for RadixHistory {
    fn read(&self, key: ByteVector) -> Result<Option<ByteVector>, HistoryError> {
        let read_result = self.imple.read(&self.root_node, key.as_slice())?;
        Ok(read_result)
    }

    fn process(&self, actions: Vec<HistoryAction>) -> Result<Box<dyn History>, HistoryError> {
        let mem_profile_enabled = std::env::var("F1R3_BLOCK_CREATOR_PHASE_SUBSTEP_PROFILE")
            .map(|v| {
                let normalized = v.trim().to_ascii_lowercase();
                normalized == "1" || normalized == "true" || normalized == "yes"
            })
            .unwrap_or(false);
        let read_rss_kb = || -> Option<u64> {
            let status = std::fs::read_to_string("/proc/self/status").ok()?;
            let line = status.lines().find(|l| l.starts_with("VmRSS:"))?;
            let mut parts = line.split_whitespace();
            let _ = parts.next();
            parts.next()?.parse::<u64>().ok()
        };
        let mut mem_prev_kb = if mem_profile_enabled {
            read_rss_kb()
        } else {
            None
        };
        let mem_base_kb = mem_prev_kb;
        let mut log_mem_step = |step: &str| {
            if !mem_profile_enabled {
                return;
            }
            if let Some(curr_kb) = read_rss_kb() {
                let prev_kb = mem_prev_kb.unwrap_or(curr_kb);
                let base_kb = mem_base_kb.unwrap_or(curr_kb);
                let delta_prev_kb = curr_kb as i64 - prev_kb as i64;
                let delta_total_kb = curr_kb as i64 - base_kb as i64;
                eprintln!(
                    "radix_history.process.mem step={} rss_kb={} delta_prev_kb={} \
                     delta_total_kb={}",
                    step, curr_kb, delta_prev_kb, delta_total_kb
                );
                mem_prev_kb = Some(curr_kb);
            }
        };
        log_mem_step("start");

        log_mem_step("before_has_no_duplicates");
        if !self.has_no_duplicates(&actions) {
            log_mem_step("duplicate_actions_error");
            return Err(HistoryError::ActionError(
                "Cannot process duplicate actions on one key.".to_string(),
            ));
        }
        log_mem_step("after_has_no_duplicates");

        log_mem_step("before_make_actions");
        let new_root_node_opt = self.imple.make_actions(&self.root_node, actions)?;
        log_mem_step("after_make_actions");

        match new_root_node_opt {
            Some(new_root_node) => {
                log_mem_step("before_save_node");
                let node_hash_bytes = self.imple.save_node(new_root_node.clone());
                log_mem_step("after_save_node");
                let root_hash = Blake2b256Hash::from_bytes(node_hash_bytes);
                // Avoid cloning RadixTreeImpl caches into each checkpointed history instance.
                // A fresh tree backed by the same store preserves correctness and reduces
                // allocator pressure from DashMap clone paths.
                log_mem_step("before_new_imple");
                let new_imple = RadixTreeImpl::new(self.store.clone());
                log_mem_step("after_new_imple");
                let new_history = RadixHistory {
                    root_hash,
                    root_node: new_root_node,
                    imple: new_imple,
                    store: self.store.clone(),
                };
                log_mem_step("before_commit");
                self.imple.commit()?;
                log_mem_step("after_commit");

                log_mem_step("before_clear_write_cache");
                self.imple.clear_write_cache();
                log_mem_step("after_clear_write_cache");
                log_mem_step("before_clear_read_cache");
                self.imple.clear_read_cache();
                log_mem_step("after_clear_read_cache");
                log_mem_step("finish_some");

                Ok(Box::new(new_history))
            }
            None => {
                log_mem_step("none_no_changes");
                let result = Box::new(RadixHistory {
                    root_hash: self.root_hash.clone(),
                    root_node: self.root_node.clone(),
                    imple: RadixTreeImpl::new(self.store.clone()),
                    store: self.store.clone(),
                });
                log_mem_step("finish_none");
                Ok(result)
            }
        }
    }

    fn root(&self) -> Blake2b256Hash { self.root_hash.clone() }

    fn reset(&self, root: &Blake2b256Hash) -> Result<Box<dyn History>, HistoryError> {
        let imple = RadixTreeImpl::new(self.store.clone());
        let node = imple.load_node(root.bytes(), Some(true))?;

        Ok(Box::new(RadixHistory {
            root_hash: root.clone(),
            root_node: node,
            imple,
            store: self.store.clone(),
        }))
    }
}
