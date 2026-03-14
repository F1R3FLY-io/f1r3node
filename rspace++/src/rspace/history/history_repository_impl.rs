use std::marker::PhantomData;
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use shared::rust::store::key_value_store::KeyValueStore;
use tracing::{Level, debug};

use super::cold_store::{ContinuationsLeaf, DataLeaf, JoinsLeaf};
use super::history_action::{DeleteAction, HistoryAction, InsertAction};
use super::history_reader::HistoryReader;
use super::history_repository::{PREFIX_DATUM, PREFIX_JOINS, PREFIX_KONT};
use super::instances::rspace_history_reader_impl::RSpaceHistoryReaderImpl;
use crate::rspace::errors::HistoryError;
use crate::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use crate::rspace::hashing::stable_hash_provider::{hash, hash_from_vec};
use crate::rspace::history::cold_store::PersistedData;
use crate::rspace::history::history::History;
use crate::rspace::history::history_repository::HistoryRepository;
use crate::rspace::history::root_repository::RootRepository;
use crate::rspace::hot_store_action::DeleteAction::{DeleteContinuations, DeleteData, DeleteJoins};
use crate::rspace::hot_store_action::HotStoreAction;
use crate::rspace::hot_store_action::InsertAction::{InsertContinuations, InsertData, InsertJoins};
use crate::rspace::hot_store_trie_action::{
    HotStoreTrieAction, TrieDeleteAction, TrieDeleteConsume, TrieDeleteJoins, TrieDeleteProduce,
    TrieInsertAction, TrieInsertConsume, TrieInsertJoins, TrieInsertProduce,
};
use crate::rspace::serializers::serializers::{encode_continuations, encode_datums, encode_joins};
use crate::rspace::state::rspace_exporter::RSpaceExporter;
use crate::rspace::state::rspace_importer::RSpaceImporter;

// See rspace/src/main/scala/coop/rchain/rspace/history/HistoryRepositoryImpl.
// scala
pub struct HistoryRepositoryImpl<C, P, A, K> {
    pub current_history: Arc<Mutex<Box<dyn History>>>,
    pub roots_repository: Arc<Mutex<RootRepository>>,
    pub leaf_store: Arc<dyn KeyValueStore>,
    pub rspace_exporter: Arc<dyn RSpaceExporter>,
    pub rspace_importer: Arc<dyn RSpaceImporter>,
    pub _marker: PhantomData<(C, P, A, K)>,
}

type ColdAction = (Blake2b256Hash, Option<PersistedData>);
const CHECKPOINT_PARALLEL_ACTIONS_THRESHOLD_DEFAULT: usize = 256;
const CHECKPOINT_PARALLEL_ACTIONS_THRESHOLD_ENV: &str =
    "F1R3_HISTORY_CHECKPOINT_PARALLEL_ACTIONS_THRESHOLD";
const BLOCK_CREATOR_PHASE_SUBSTEP_PROFILE_ENV: &str = "F1R3_BLOCK_CREATOR_PHASE_SUBSTEP_PROFILE";

impl<C, P, A, K> HistoryRepositoryImpl<C, P, A, K>
where
    C: Clone + Send + Sync + Serialize,
    P: Clone + Send + Sync + Serialize,
    A: Clone + Send + Sync + Serialize,
    K: Clone + Send + Sync + Serialize,
{
    fn checkpoint_noop_clone(
        &self,
    ) -> Box<dyn HistoryRepository<C, P, A, K> + Send + Sync + 'static>
    where
        C: for<'a> Deserialize<'a> + 'static,
        P: for<'a> Deserialize<'a> + 'static,
        A: for<'a> Deserialize<'a> + 'static,
        K: for<'a> Deserialize<'a> + 'static,
    {
        Box::new(HistoryRepositoryImpl {
            current_history: self.current_history.clone(),
            roots_repository: self.roots_repository.clone(),
            leaf_store: self.leaf_store.clone(),
            rspace_exporter: self.rspace_exporter.clone(),
            rspace_importer: self.rspace_importer.clone(),
            _marker: PhantomData,
        })
    }

    fn checkpoint_parallel_actions_threshold() -> usize {
        static VALUE: OnceLock<usize> = OnceLock::new();
        *VALUE.get_or_init(|| {
            std::env::var(CHECKPOINT_PARALLEL_ACTIONS_THRESHOLD_ENV)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(CHECKPOINT_PARALLEL_ACTIONS_THRESHOLD_DEFAULT)
        })
    }

    fn block_creator_phase_substep_profile_enabled() -> bool {
        static VALUE: OnceLock<bool> = OnceLock::new();
        *VALUE.get_or_init(|| {
            std::env::var(BLOCK_CREATOR_PHASE_SUBSTEP_PROFILE_ENV)
                .map(|value| {
                    let normalized = value.trim().to_ascii_lowercase();
                    normalized == "1" || normalized == "true" || normalized == "yes"
                })
                .unwrap_or(false)
        })
    }

    fn should_parallelize_checkpoint_actions(actions_len: usize) -> bool {
        actions_len >= Self::checkpoint_parallel_actions_threshold()
    }

    fn measure(&self, actions: &Vec<HotStoreAction<C, P, A, K>>) -> () {
        if !tracing::enabled!(Level::DEBUG) {
            return;
        }
        for p in self.compute_measure(actions) {
            debug!("{}", p);
        }
    }

    fn compute_measure(&self, actions: &Vec<HotStoreAction<C, P, A, K>>) -> Vec<String> {
        actions
            .into_par_iter()
            .map(|action| match action {
                HotStoreAction::Insert(InsertData(i)) => {
                    let key = hash(&i.channel).bytes();
                    let data = encode_datums(&i.data);
                    format!("{};insert-data;{};{}", hex::encode(key), data.len(), i.data.len())
                }
                HotStoreAction::Insert(InsertContinuations(i)) => {
                    let key = hash_from_vec(&i.channels).bytes();
                    let data = encode_continuations(&i.continuations);
                    format!(
                        "{};insert-continuation;{};{}",
                        hex::encode(key),
                        data.len(),
                        i.continuations.len()
                    )
                }
                HotStoreAction::Insert(InsertJoins(i)) => {
                    let key = hash(&i.channel).bytes();
                    let data = encode_joins(&i.joins);
                    format!("{};insert-join;{};", hex::encode(key), data.len())
                }
                HotStoreAction::Delete(DeleteData(d)) => {
                    let key = hash(&d.channel).bytes();
                    format!("{};delete-data;0", hex::encode(key))
                }
                HotStoreAction::Delete(DeleteContinuations(d)) => {
                    let key = hash_from_vec(&d.channels).bytes();
                    format!("{};delete-continuation;0", hex::encode(key))
                }
                HotStoreAction::Delete(DeleteJoins(d)) => {
                    let key = hash(&d.channel).bytes();
                    format!("{};delete-join;0", hex::encode(key))
                }
            })
            .collect()
    }

    fn calculate_storage_actions(
        &self,
        action: &HotStoreTrieAction<C, P, A, K>,
    ) -> (ColdAction, HistoryAction) {
        let mem_profile_enabled = Self::block_creator_phase_substep_profile_enabled();
        let action_kind = match action {
            HotStoreTrieAction::TrieInsertAction(TrieInsertAction::TrieInsertProduce(_)) => {
                "insert_produce"
            }
            HotStoreTrieAction::TrieInsertAction(TrieInsertAction::TrieInsertConsume(_)) => {
                "insert_consume"
            }
            HotStoreTrieAction::TrieInsertAction(TrieInsertAction::TrieInsertJoins(_)) => {
                "insert_joins"
            }
            HotStoreTrieAction::TrieInsertAction(TrieInsertAction::TrieInsertBinaryProduce(_)) => {
                "insert_binary_produce"
            }
            HotStoreTrieAction::TrieInsertAction(TrieInsertAction::TrieInsertBinaryConsume(_)) => {
                "insert_binary_consume"
            }
            HotStoreTrieAction::TrieInsertAction(TrieInsertAction::TrieInsertBinaryJoins(_)) => {
                "insert_binary_joins"
            }
            HotStoreTrieAction::TrieDeleteAction(TrieDeleteAction::TrieDeleteProduce(_)) => {
                "delete_produce"
            }
            HotStoreTrieAction::TrieDeleteAction(TrieDeleteAction::TrieDeleteConsume(_)) => {
                "delete_consume"
            }
            HotStoreTrieAction::TrieDeleteAction(TrieDeleteAction::TrieDeleteJoins(_)) => {
                "delete_joins"
            }
        };
        let read_rss_kb = || -> Option<u64> {
            let status = std::fs::read_to_string("/proc/self/status").ok()?;
            let line = status.lines().find(|l| l.starts_with("VmRSS:"))?;
            let mut parts = line.split_whitespace();
            let _ = parts.next();
            parts.next()?.parse::<u64>().ok()
        };
        let start_kb = if mem_profile_enabled {
            read_rss_kb()
        } else {
            None
        };
        if mem_profile_enabled {
            if let Some(rss_kb) = start_kb {
                eprintln!(
                    "history_repo.calculate_storage_actions.mem step=start action_kind={} \
                     rss_kb={}",
                    action_kind, rss_kb
                );
            }
        }

        let result = match action {
            HotStoreTrieAction::TrieInsertAction(TrieInsertAction::TrieInsertProduce(i)) => {
                let data = encode_datums(&i.data);
                let data_leaf = DataLeaf { bytes: data };
                let data_leaf_encoded = bincode::serialize(&data_leaf)
                    .expect("History Repository Impl: Unable to serialize DataLeaf");
                let data_hash = Blake2b256Hash::new(&data_leaf_encoded);

                (
                    (data_hash.clone(), Some(PersistedData::Data(data_leaf))),
                    HistoryAction::Insert(InsertAction {
                        key: prepend_bytes(PREFIX_DATUM, &i.hash.bytes()),
                        hash: data_hash,
                    }),
                )
            }
            HotStoreTrieAction::TrieInsertAction(TrieInsertAction::TrieInsertConsume(i)) => {
                let data = encode_continuations(&i.continuations);
                let continuations_leaf = ContinuationsLeaf { bytes: data };
                let continuations_leaf_encoded = bincode::serialize(&continuations_leaf)
                    .expect("History Repository Impl: Unable to serialize ContinuationsLeaf");
                let continuations_hash = Blake2b256Hash::new(&continuations_leaf_encoded);

                (
                    (
                        continuations_hash.clone(),
                        Some(PersistedData::Continuations(continuations_leaf)),
                    ),
                    HistoryAction::Insert(InsertAction {
                        key: prepend_bytes(PREFIX_KONT, &i.hash.bytes()),
                        hash: continuations_hash,
                    }),
                )
            }
            HotStoreTrieAction::TrieInsertAction(TrieInsertAction::TrieInsertJoins(i)) => {
                let data = encode_joins(&i.joins);
                let joins_leaf = JoinsLeaf { bytes: data };
                let joins_leaf_encoded = bincode::serialize(&joins_leaf)
                    .expect("History Repository Impl: Unable to serialize JoinsLeaf");
                let joins_hash = Blake2b256Hash::new(&joins_leaf_encoded);

                (
                    (joins_hash.clone(), Some(PersistedData::Joins(joins_leaf))),
                    HistoryAction::Insert(InsertAction {
                        key: prepend_bytes(PREFIX_JOINS, &i.hash.bytes()),
                        hash: joins_hash,
                    }),
                )
            }
            HotStoreTrieAction::TrieInsertAction(TrieInsertAction::TrieInsertBinaryProduce(i)) => {
                // Sort data before serializing for deterministic hashing regardless of
                // insertion order
                let mut sorted_data = i.data.clone();
                sorted_data.sort();
                let data =
                    bincode::serialize(&sorted_data).expect("Failed to serialize Vec<Vec<u8>>");
                let data_leaf = DataLeaf { bytes: data };
                let data_leaf_encoded = bincode::serialize(&data_leaf)
                    .expect("History Repository Impl: Unable to serialize DataLeaf");
                let data_hash = Blake2b256Hash::new(&data_leaf_encoded);

                (
                    (data_hash.clone(), Some(PersistedData::Data(data_leaf))),
                    HistoryAction::Insert(InsertAction {
                        key: prepend_bytes(PREFIX_DATUM, &i.hash.bytes()),
                        hash: data_hash,
                    }),
                )
            }
            HotStoreTrieAction::TrieInsertAction(TrieInsertAction::TrieInsertBinaryConsume(i)) => {
                // Sort continuations before serializing for deterministic hashing regardless of
                // insertion order
                let mut sorted_continuations = i.continuations.clone();
                sorted_continuations.sort();
                let data = bincode::serialize(&sorted_continuations)
                    .expect("Failed to serialize Vec<Vec<u8>>");
                let continuations_leaf = ContinuationsLeaf { bytes: data };
                let continuations_leaf_encoded = bincode::serialize(&continuations_leaf)
                    .expect("History Repository Impl: Unable to serialize ContinuationsLeaf");
                let continuations_hash = Blake2b256Hash::new(&continuations_leaf_encoded);

                (
                    (
                        continuations_hash.clone(),
                        Some(PersistedData::Continuations(continuations_leaf)),
                    ),
                    HistoryAction::Insert(InsertAction {
                        key: prepend_bytes(PREFIX_KONT, &i.hash.bytes()),
                        hash: continuations_hash,
                    }),
                )
            }
            HotStoreTrieAction::TrieInsertAction(TrieInsertAction::TrieInsertBinaryJoins(i)) => {
                // Sort joins before serializing for deterministic hashing regardless of
                // insertion order
                let mut sorted_joins = i.joins.clone();
                sorted_joins.sort();
                let data =
                    bincode::serialize(&sorted_joins).expect("Failed to serialize Vec<Vec<u8>>");
                let joins_leaf = JoinsLeaf { bytes: data };
                let joins_leaf_encoded = bincode::serialize(&joins_leaf)
                    .expect("History Repository Impl: Unable to serialize JoinsLeaf");
                let joins_hash = Blake2b256Hash::new(&joins_leaf_encoded);

                (
                    (joins_hash.clone(), Some(PersistedData::Joins(joins_leaf))),
                    HistoryAction::Insert(InsertAction {
                        key: prepend_bytes(PREFIX_JOINS, &i.hash.bytes()),
                        hash: joins_hash,
                    }),
                )
            }
            HotStoreTrieAction::TrieDeleteAction(TrieDeleteAction::TrieDeleteProduce(d)) => (
                (d.hash.clone(), None),
                HistoryAction::Delete(DeleteAction {
                    key: prepend_bytes(PREFIX_DATUM, &d.hash.bytes()),
                }),
            ),
            HotStoreTrieAction::TrieDeleteAction(TrieDeleteAction::TrieDeleteConsume(d)) => (
                (d.hash.clone(), None),
                HistoryAction::Delete(DeleteAction {
                    key: prepend_bytes(PREFIX_KONT, &d.hash.bytes()),
                }),
            ),
            HotStoreTrieAction::TrieDeleteAction(TrieDeleteAction::TrieDeleteJoins(d)) => (
                (d.hash.clone(), None),
                HistoryAction::Delete(DeleteAction {
                    key: prepend_bytes(PREFIX_JOINS, &d.hash.bytes()),
                }),
            ),
        };
        if mem_profile_enabled {
            if let Some(end_kb) = read_rss_kb() {
                let base_kb = start_kb.unwrap_or(end_kb);
                let delta_kb = end_kb as i64 - base_kb as i64;
                eprintln!(
                    "history_repo.calculate_storage_actions.mem step=finish action_kind={} \
                     rss_kb={} delta_total_kb={}",
                    action_kind, end_kb, delta_kb
                );
            }
        }
        result
    }

    fn transform(
        &self,
        hot_store_action: HotStoreAction<C, P, A, K>,
    ) -> HotStoreTrieAction<C, P, A, K> {
        let mem_profile_enabled = Self::block_creator_phase_substep_profile_enabled();
        let action_kind = match &hot_store_action {
            HotStoreAction::Insert(InsertData(_)) => "insert_data",
            HotStoreAction::Insert(InsertContinuations(_)) => "insert_continuations",
            HotStoreAction::Insert(InsertJoins(_)) => "insert_joins",
            HotStoreAction::Delete(DeleteData(_)) => "delete_data",
            HotStoreAction::Delete(DeleteContinuations(_)) => "delete_continuations",
            HotStoreAction::Delete(DeleteJoins(_)) => "delete_joins",
        };
        let read_rss_kb = || -> Option<u64> {
            let status = std::fs::read_to_string("/proc/self/status").ok()?;
            let line = status.lines().find(|l| l.starts_with("VmRSS:"))?;
            let mut parts = line.split_whitespace();
            let _ = parts.next();
            parts.next()?.parse::<u64>().ok()
        };
        let log_step_delta = |step: &str, start_kb: Option<u64>| {
            if !mem_profile_enabled {
                return;
            }
            if let (Some(before), Some(after)) = (start_kb, read_rss_kb()) {
                let delta_kb = after as i64 - before as i64;
                if delta_kb != 0 {
                    eprintln!(
                        "history_repo.transform.step.mem action_kind={} step={} rss_kb={} \
                         delta_kb={}",
                        action_kind, step, after, delta_kb
                    );
                }
            }
        };
        let start_kb = if mem_profile_enabled {
            read_rss_kb()
        } else {
            None
        };

        let transformed = match hot_store_action {
            HotStoreAction::Insert(InsertData(i)) => {
                let before_hash = if mem_profile_enabled {
                    read_rss_kb()
                } else {
                    None
                };
                let key = hash(&i.channel);
                log_step_delta("after_hash_data_channel", before_hash);
                let before_take = if mem_profile_enabled {
                    read_rss_kb()
                } else {
                    None
                };
                let data = i.data;
                log_step_delta("after_take_insert_data_payload", before_take);
                let before_new = if mem_profile_enabled {
                    read_rss_kb()
                } else {
                    None
                };
                let trie_insert_action =
                    TrieInsertAction::TrieInsertProduce(TrieInsertProduce::new(key, data));
                log_step_delta("after_new_trie_insert_produce", before_new);
                HotStoreTrieAction::TrieInsertAction(trie_insert_action)
            }
            HotStoreAction::Insert(InsertContinuations(i)) => {
                let before_hash = if mem_profile_enabled {
                    read_rss_kb()
                } else {
                    None
                };
                let key = hash_from_vec(&i.channels);
                log_step_delta("after_hash_continuations_channels", before_hash);
                let before_take = if mem_profile_enabled {
                    read_rss_kb()
                } else {
                    None
                };
                let continuations = i.continuations;
                log_step_delta("after_take_insert_continuations_payload", before_take);
                let before_new = if mem_profile_enabled {
                    read_rss_kb()
                } else {
                    None
                };
                let trie_insert_action =
                    TrieInsertAction::TrieInsertConsume(TrieInsertConsume::new(key, continuations));
                log_step_delta("after_new_trie_insert_consume", before_new);
                HotStoreTrieAction::TrieInsertAction(trie_insert_action)
            }
            HotStoreAction::Insert(InsertJoins(i)) => {
                let before_hash = if mem_profile_enabled {
                    read_rss_kb()
                } else {
                    None
                };
                let key = hash(&i.channel);
                log_step_delta("after_hash_joins_channel", before_hash);
                let before_take = if mem_profile_enabled {
                    read_rss_kb()
                } else {
                    None
                };
                let joins = i.joins;
                log_step_delta("after_take_insert_joins_payload", before_take);
                let before_new = if mem_profile_enabled {
                    read_rss_kb()
                } else {
                    None
                };
                let trie_insert_action =
                    TrieInsertAction::TrieInsertJoins(TrieInsertJoins::new(key, joins));
                log_step_delta("after_new_trie_insert_joins", before_new);
                HotStoreTrieAction::TrieInsertAction(trie_insert_action)
            }
            HotStoreAction::Delete(DeleteData(d)) => {
                let before_hash = if mem_profile_enabled {
                    read_rss_kb()
                } else {
                    None
                };
                let key = hash(&d.channel);
                log_step_delta("after_hash_delete_data_channel", before_hash);
                let before_new = if mem_profile_enabled {
                    read_rss_kb()
                } else {
                    None
                };
                let trie_delete_action =
                    TrieDeleteAction::TrieDeleteProduce(TrieDeleteProduce::new(key));
                log_step_delta("after_new_trie_delete_produce", before_new);
                HotStoreTrieAction::TrieDeleteAction(trie_delete_action)
            }
            HotStoreAction::Delete(DeleteContinuations(d)) => {
                let before_hash = if mem_profile_enabled {
                    read_rss_kb()
                } else {
                    None
                };
                let key = hash_from_vec(&d.channels);
                log_step_delta("after_hash_delete_continuations_channels", before_hash);
                let before_new = if mem_profile_enabled {
                    read_rss_kb()
                } else {
                    None
                };
                let trie_delete_action =
                    TrieDeleteAction::TrieDeleteConsume(TrieDeleteConsume::new(key));
                log_step_delta("after_new_trie_delete_consume", before_new);
                HotStoreTrieAction::TrieDeleteAction(trie_delete_action)
            }
            HotStoreAction::Delete(DeleteJoins(d)) => {
                let before_hash = if mem_profile_enabled {
                    read_rss_kb()
                } else {
                    None
                };
                let key = hash(&d.channel);
                log_step_delta("after_hash_delete_joins_channel", before_hash);
                let before_new = if mem_profile_enabled {
                    read_rss_kb()
                } else {
                    None
                };
                let trie_delete_action =
                    TrieDeleteAction::TrieDeleteJoins(TrieDeleteJoins::new(key));
                log_step_delta("after_new_trie_delete_joins", before_new);
                HotStoreTrieAction::TrieDeleteAction(trie_delete_action)
            }
        };

        if mem_profile_enabled {
            if let Some(end_kb) = read_rss_kb() {
                let base_kb = start_kb.unwrap_or(end_kb);
                let delta_kb = end_kb as i64 - base_kb as i64;
                if delta_kb != 0 {
                    eprintln!(
                        "history_repo.transform.mem action_kind={} rss_kb={} delta_total_kb={}",
                        action_kind, end_kb, delta_kb
                    );
                }
            }
        }

        transformed
    }
}

impl<C, P, A, K> HistoryRepository<C, P, A, K> for HistoryRepositoryImpl<C, P, A, K>
where
    C: Clone + Send + Sync + Serialize + for<'a> Deserialize<'a> + 'static,
    P: Clone + Send + Sync + Serialize + for<'a> Deserialize<'a> + 'static,
    A: Clone + Send + Sync + Serialize + for<'a> Deserialize<'a> + 'static,
    K: Clone + Send + Sync + Serialize + for<'a> Deserialize<'a> + 'static,
{
    fn checkpoint(
        &self,
        actions: Vec<HotStoreAction<C, P, A, K>>,
    ) -> Box<dyn HistoryRepository<C, P, A, K> + Send + Sync + 'static> {
        let mem_profile_enabled = Self::block_creator_phase_substep_profile_enabled();
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
                    "history_repo.checkpoint.mem step={} rss_kb={} delta_prev_kb={} \
                     delta_total_kb={}",
                    step, curr_kb, delta_prev_kb, delta_total_kb
                );
                mem_prev_kb = Some(curr_kb);
            }
        };
        log_mem_step("start");

        if actions.is_empty() {
            log_mem_step("actions_empty_noop_clone");
            return self.checkpoint_noop_clone();
        }

        log_mem_step("before_measure");
        let _ = self.measure(&actions);
        log_mem_step("after_measure");

        let trie_actions: Vec<_> = if Self::should_parallelize_checkpoint_actions(actions.len()) {
            actions
                .into_par_iter()
                .map(|action| self.transform(action))
                .collect()
        } else {
            actions
                .into_iter()
                .map(|action| self.transform(action))
                .collect()
        };
        log_mem_step("after_transform_actions");

        log_mem_step("before_do_checkpoint");
        let hr = self.do_checkpoint(trie_actions);
        log_mem_step("after_do_checkpoint");
        log_mem_step("finish");
        hr
    }

    fn do_checkpoint(
        &self,
        trie_actions: Vec<HotStoreTrieAction<C, P, A, K>>,
    ) -> Box<dyn HistoryRepository<C, P, A, K> + Send + Sync + 'static> {
        let mem_profile_enabled = Self::block_creator_phase_substep_profile_enabled();
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
                    "history_repo.do_checkpoint.mem step={} rss_kb={} delta_prev_kb={} \
                     delta_total_kb={}",
                    step, curr_kb, delta_prev_kb, delta_total_kb
                );
                mem_prev_kb = Some(curr_kb);
            }
        };
        log_mem_step("start");

        if trie_actions.is_empty() {
            log_mem_step("actions_empty_noop_clone");
            return self.checkpoint_noop_clone();
        }

        let storage_actions: Vec<(ColdAction, HistoryAction)> =
            if Self::should_parallelize_checkpoint_actions(trie_actions.len()) {
                trie_actions
                    .par_iter()
                    .map(|a| self.calculate_storage_actions(a))
                    .collect()
            } else {
                trie_actions
                    .iter()
                    .map(|a| self.calculate_storage_actions(a))
                    .collect()
            };
        log_mem_step("after_calculate_storage_actions");

        let mut cold_actions: Vec<(Blake2b256Hash, PersistedData)> = Vec::new();
        let mut history_actions: Vec<HistoryAction> = Vec::with_capacity(storage_actions.len());
        for ((key, maybe_data), history) in storage_actions {
            if let Some(data) = maybe_data {
                cold_actions.push((key, data));
            }
            history_actions.push(history);
        }
        log_mem_step("after_partition_cold_and_history_actions");

        // save new root for state after checkpoint
        let store_root = |root| {
            let roots_repo_lock = self
                .roots_repository
                .lock()
                .expect("History Repository Impl: Unable to acquire roots repository lock");
            roots_repo_lock.commit(root)
        };

        // store cold data
        let store_leaves = {
            let serialized_cold_actions = cold_actions
                .into_iter()
                .map(|(key, value)| {
                    let serialized_key = bincode::serialize(&key)
                        .expect("History Respository Impl: Failed to serialize");
                    let serialized_value = bincode::serialize(&value)
                        .expect("History Respository Impl: Failed to serialize");
                    (serialized_key, serialized_value)
                })
                .collect();

            // println!("\nserialized_cold_actions: {:?}", serialized_cold_actions);

            self.leaf_store
                .put_if_absent(serialized_cold_actions)
                .expect("History Repository Impl: Failed to put if absent");
        };
        log_mem_step("after_store_leaves");

        // store everything related to history (history data, new root and populate
        // cache for new root)
        let store_history = {
            log_mem_step("before_history_lock");
            let process_result = {
                let history_lock = self
                    .current_history
                    .lock()
                    .expect("History Repository Impl: Unable to acquire history lock");
                log_mem_step("after_history_lock");
                log_mem_step("before_history_process");
                let processed = history_lock.process(history_actions);
                log_mem_step("after_history_process_call");
                processed
            };
            let result_history = process_result.unwrap();
            log_mem_step("after_history_process_unwrap");
            result_history
        };
        log_mem_step("after_store_history");

        let new_root = store_history.root();
        log_mem_step("after_new_root");
        store_root(&new_root).expect("History Repository Impl: Unable to store root");
        log_mem_step("after_store_root");

        let combined = {
            let leaves = store_leaves;
            let history = store_history;
            (leaves, history)
        };
        let (_, new_history) = combined;
        log_mem_step("after_unpack_combined");

        let result = Box::new(HistoryRepositoryImpl {
            current_history: Arc::new(Mutex::new(new_history)),
            roots_repository: self.roots_repository.clone(),
            leaf_store: self.leaf_store.clone(),
            rspace_exporter: self.rspace_exporter.clone(),
            rspace_importer: self.rspace_importer.clone(),
            _marker: PhantomData,
        });
        log_mem_step("after_build_result");
        log_mem_step("finish");
        result
    }

    fn reset(
        &self,
        root: &Blake2b256Hash,
    ) -> Result<Box<dyn HistoryRepository<C, P, A, K> + Send + Sync + 'static>, HistoryError> {
        debug!("[HistoryRepositoryImpl] reset to {}", root);

        let roots_lock = self
            .roots_repository
            .lock()
            .expect("History Repository Impl: Unable to acquire roots repository lock");
        roots_lock.validate_and_set_current_root(root.clone())?;

        let history_lock = self
            .current_history
            .lock()
            .expect("History Repository Impl: Unable to acquire history lock");
        let next = history_lock.reset(root)?;

        Ok(Box::new(HistoryRepositoryImpl {
            current_history: Arc::new(Mutex::new(next)),
            roots_repository: self.roots_repository.clone(),
            leaf_store: self.leaf_store.clone(),
            rspace_exporter: self.rspace_exporter.clone(),
            rspace_importer: self.rspace_importer.clone(),
            _marker: PhantomData,
        }))
    }

    fn history(&self) -> Arc<Mutex<Box<dyn History>>> { self.current_history.clone() }

    fn exporter(&self) -> Arc<dyn RSpaceExporter> { self.rspace_exporter.clone() }

    fn importer(&self) -> Arc<dyn RSpaceImporter> { self.rspace_importer.clone() }

    fn get_history_reader(
        &self,
        state_hash: &Blake2b256Hash,
    ) -> Result<Box<dyn HistoryReader<Blake2b256Hash, C, P, A, K>>, HistoryError> {
        let history_lock = self
            .current_history
            .lock()
            .expect("History Repository Impl: Unable to acquire history lock");
        let history_repo = history_lock.reset(&state_hash)?;
        Ok(Box::new(RSpaceHistoryReaderImpl::new(history_repo, self.leaf_store.clone())))
    }

    fn get_history_reader_struct(
        &self,
        state_hash: &Blake2b256Hash,
    ) -> Result<RSpaceHistoryReaderImpl<C, P, A, K>, HistoryError> {
        let history_lock = self
            .current_history
            .lock()
            .expect("History Repository Impl: Unable to acquire history lock");
        let history_repo = history_lock.reset(state_hash)?;
        Ok(RSpaceHistoryReaderImpl::new(history_repo, self.leaf_store.clone()))
    }

    fn root(&self) -> Blake2b256Hash {
        let history_lock = self
            .current_history
            .lock()
            .expect("History Repository Impl: Unable to acquire history lock");
        history_lock.root()
    }

    fn record_root(&self, root: &Blake2b256Hash) -> Result<(), HistoryError> {
        let roots_repo = self
            .roots_repository
            .lock()
            .expect("History Repository Impl: Unable to acquire roots_repository lock");
        roots_repo.commit(root).map_err(HistoryError::from)
    }
}

pub fn prepend_bytes(element: u8, _bytes: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.push(element);
    bytes.extend(_bytes.iter().cloned());
    bytes
}
