use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
#[cfg(test)]
use proptest::prelude::*;
#[cfg(test)]
use rand::{Rng, thread_rng};
use tracing::warn;

use super::errors::RSpaceError;

use crate::rspace::history::history_reader::HistoryReaderBase;
use crate::rspace::hot_store_action::{
    DeleteAction, DeleteContinuations, DeleteData, DeleteJoins, HotStoreAction, InsertAction,
    InsertContinuations, InsertData, InsertJoins,
};
use crate::rspace::internal::{Datum, Row, WaitingContinuation};
use crate::rspace::metrics_constants::{
    HOT_STORE_HISTORY_CONT_CACHE_ITEMS_METRIC, HOT_STORE_HISTORY_CONT_CACHE_SIZE_METRIC,
    HOT_STORE_HISTORY_DATA_CACHE_ITEMS_METRIC, HOT_STORE_HISTORY_DATA_CACHE_SIZE_METRIC,
    HOT_STORE_HISTORY_JOINS_CACHE_ITEMS_METRIC, HOT_STORE_HISTORY_JOINS_CACHE_SIZE_METRIC,
    HOT_STORE_STATE_CONT_ITEMS_METRIC, HOT_STORE_STATE_CONT_SIZE_METRIC,
    HOT_STORE_STATE_DATA_ITEMS_METRIC, HOT_STORE_STATE_DATA_SIZE_METRIC,
    HOT_STORE_STATE_INSTALLED_CONT_ITEMS_METRIC, HOT_STORE_STATE_INSTALLED_CONT_SIZE_METRIC,
    HOT_STORE_STATE_INSTALLED_JOINS_ITEMS_METRIC, HOT_STORE_STATE_INSTALLED_JOINS_SIZE_METRIC,
    HOT_STORE_STATE_JOINS_ITEMS_METRIC, HOT_STORE_STATE_JOINS_SIZE_METRIC, RSPACE_METRICS_SOURCE,
};

const MAX_HISTORY_STORE_CACHE_ENTRIES: usize = 512;
const MAX_HISTORY_STORE_CACHE_CONT_ITEMS: usize = 8192;
const MAX_HISTORY_STORE_CACHE_DATA_ITEMS: usize = 8192;
const MAX_HISTORY_STORE_CACHE_JOIN_ITEMS: usize = 8192;
const HOT_STORE_STATE_METRICS_UPDATE_INTERVAL_MS_DEFAULT: u64 = 250;
const HOT_STORE_STATE_METRICS_UPDATE_INTERVAL_MS_ENV: &str =
    "F1R3_HOT_STORE_STATE_METRICS_UPDATE_INTERVAL_MS";
const HOT_STORE_HISTORY_CACHE_METRICS_UPDATE_INTERVAL_MS_DEFAULT: u64 = 250;
const HOT_STORE_HISTORY_CACHE_METRICS_UPDATE_INTERVAL_MS_ENV: &str =
    "F1R3_HOT_STORE_HISTORY_CACHE_METRICS_UPDATE_INTERVAL_MS";

// See rspace/src/main/scala/coop/rchain/rspace/HotStore.scala
pub trait HotStore<C: Clone + Hash + Eq, P: Clone, A: Clone, K: Clone>: Sync + Send {
    fn get_continuations(&self, channels: &[C]) -> Vec<WaitingContinuation<P, K>>;
    fn put_continuation(&self, channels: &[C], wc: WaitingContinuation<P, K>) -> Option<bool>;
    fn install_continuation(&self, channels: &[C], wc: WaitingContinuation<P, K>) -> Option<()>;
    fn remove_continuation(&self, channels: &[C], index: i32) -> Option<()>;

    fn get_data(&self, channel: &C) -> Vec<Datum<A>>;
    fn put_datum(&self, channel: &C, d: Datum<A>) -> ();
    fn remove_datum(&self, channel: &C, index: i32) -> Result<(), RSpaceError>;

    fn get_joins(&self, channel: &C) -> Vec<Vec<C>>;
    fn put_join(&self, channel: &C, join: &[C]) -> Option<()>;
    fn install_join(&self, channel: &C, join: &[C]) -> Option<()>;
    fn remove_join(&self, channel: &C, join: &[C]) -> Option<()>;

    fn changes(&self) -> Vec<HotStoreAction<C, P, A, K>>;
    fn to_map(&self) -> HashMap<Vec<C>, Row<P, A, K>>;
    fn snapshot(&self) -> HotStoreState<C, P, A, K>;

    fn print(&self) -> ();
    fn clear(&self) -> ();

    // See rspace/src/test/scala/coop/rchain/rspace/test/package.scala
    fn is_empty(&self) -> bool;

    /// Returns lightweight pending state counts for diagnostics:
    /// (data_channels, data_items, continuation_channels, continuation_items)
    fn state_counts(&self) -> (usize, usize, usize, usize);

    /// Returns debug info for each pending continuation channel:
    /// Vec of (channels_debug_string, num_continuations, has_peek)
    fn continuation_channels_debug(&self) -> Vec<(String, usize, bool)>;
}

pub fn new_dashmap<K: std::cmp::Eq + std::hash::Hash, V>() -> DashMap<K, V> { DashMap::new() }

#[derive(Default, Debug, Clone)]
pub struct HotStoreState<C, P, A, K>
where
    C: Eq + Hash,
    A: Clone,
    P: Clone,
    K: Clone,
{
    pub continuations: DashMap<Vec<C>, Vec<WaitingContinuation<P, K>>>,
    pub installed_continuations: DashMap<Vec<C>, WaitingContinuation<P, K>>,
    pub data: DashMap<C, Vec<Datum<A>>>,
    pub joins: DashMap<C, Vec<Vec<C>>>,
    pub installed_joins: DashMap<C, Vec<Vec<C>>>,
}

// This impl is needed for hot_store_spec.rs
#[cfg(test)]
impl<C, P, A, K> HotStoreState<C, P, A, K>
where
    C: Eq + Hash + Debug + Arbitrary + Default + Clone,
    A: Clone + Debug + Arbitrary + Default,
    P: Clone + Debug + Arbitrary + Default,
    K: Clone + Debug + Arbitrary + Default,
{
    fn random_vec<T>(size: usize) -> Vec<T>
    where T: Default + Clone {
        let mut rng = thread_rng();
        (0..size)
            .map(|_| T::default())
            .collect::<Vec<T>>()
            .iter()
            .cloned()
            .take(rng.gen_range(0..size + 1))
            .collect()
    }

    pub fn random_state() -> Self {
        let channels: Vec<C> = HotStoreState::<C, P, A, K>::random_vec(10);
        let continuations: Vec<WaitingContinuation<P, K>> =
            HotStoreState::<C, P, A, K>::random_vec(10);
        let installed_continuations = WaitingContinuation::default();
        let data: Vec<Datum<A>> = HotStoreState::<C, P, A, K>::random_vec(10);
        let channel = C::default();
        let joins: Vec<Vec<C>> = HotStoreState::<C, P, A, K>::random_vec(10);
        let installed_joins: Vec<Vec<C>> = HotStoreState::<C, P, A, K>::random_vec(10);

        HotStoreState {
            continuations: DashMap::from_iter(vec![(channels.clone(), continuations.clone())]),
            installed_continuations: DashMap::from_iter(vec![(
                channels.clone(),
                installed_continuations.clone(),
            )]),
            data: DashMap::from_iter(vec![(channel.clone(), data.clone())]),
            joins: DashMap::from_iter(vec![(channel.clone(), joins)]),
            installed_joins: DashMap::from_iter(vec![(channel, installed_joins)]),
        }
    }
}

#[derive(Default)]
struct HistoryStoreCache<C, P, A, K>
where
    C: Eq + Hash,
    A: Clone,
    P: Clone,
    K: Clone,
{
    continuations: DashMap<Vec<C>, Vec<WaitingContinuation<P, K>>>,
    datums: DashMap<C, Vec<Datum<A>>>,
    joins: DashMap<C, Vec<Vec<C>>>,
}

struct InMemHotStore<C, P, A, K>
where
    C: Eq + Hash + Sync + Send,
    A: Clone + Sync + Send,
    P: Clone + Sync + Send,
    K: Clone + Sync + Send,
{
    hot_store_state: Arc<Mutex<HotStoreState<C, P, A, K>>>,
    history_store_cache: Arc<Mutex<HistoryStoreCache<C, P, A, K>>>,
    history_reader_base: Box<dyn HistoryReaderBase<C, P, A, K>>,
}

// See rspace/src/main/scala/coop/rchain/rspace/HotStore.scala
impl<C, P, A, K> HotStore<C, P, A, K> for InMemHotStore<C, P, A, K>
where
    C: Clone + Debug + Hash + Eq + Send + Sync,
    P: Clone + Debug + Send + Sync,
    A: Clone + Debug + Send + Sync,
    K: Clone + Debug + Send + Sync,
{
    fn snapshot(&self) -> HotStoreState<C, P, A, K> {
        let hot_store_state_lock = self.hot_store_state.lock().unwrap();
        HotStoreState {
            continuations: hot_store_state_lock.continuations.clone(),
            installed_continuations: hot_store_state_lock.installed_continuations.clone(),
            data: hot_store_state_lock.data.clone(),
            joins: hot_store_state_lock.joins.clone(),
            installed_joins: hot_store_state_lock.installed_joins.clone(),
        }
    }

    // Continuations

    fn get_continuations(&self, channels: &[C]) -> Vec<WaitingContinuation<P, K>> {
        let (continuations, installed) = {
            let state = self.hot_store_state.lock().unwrap();
            (
                state.continuations.get(channels).map(|c| c.clone()),
                state
                    .installed_continuations
                    .get(channels)
                    .map(|c| c.clone()),
            )
        };

        let result = match (continuations, installed) {
            (Some(conts), Some(inst)) => {
                let mut result = Vec::with_capacity(conts.len() + 1);
                result.push(inst);
                result.extend(conts);
                result
            }
            (Some(conts), None) => conts,
            (None, Some(inst)) => {
                // Read-only fallthrough: return history continuations WITHOUT caching in
                // hot store state. Caching here would cause changes() to re-emit unchanged
                // continuations with potentially different channel serialization.
                let from_history_store = self.get_cont_from_history_store(channels);
                let mut result = Vec::with_capacity(from_history_store.len() + 1);
                result.push(inst);
                result.extend(from_history_store);
                result
            }
            (None, None) => {
                // Read-only fallthrough: return history continuations WITHOUT caching in
                // hot store state. Caching here would cause changes() to re-emit unchanged
                // continuations with potentially different channel serialization.
                let from_history_store = self.get_cont_from_history_store(channels);
                let persistent_count = from_history_store.iter().filter(|wc| wc.persist).count();
                tracing::debug!(
                    target: "f1r3fly.rspace",
                    channels = ?channels,
                    history_conts = from_history_store.len(),
                    persistent_conts = persistent_count,
                    "get_continuations: fell through to history, found {}",
                    from_history_store.len()
                );
                from_history_store
            }
        };
        let state = self.hot_store_state.lock().unwrap();
        Self::update_hot_store_state_metrics(&state);
        result
    }

    fn put_continuation(&self, channels: &[C], wc: WaitingContinuation<P, K>) -> Option<bool> {

        let mut inserted = false;
        let has_existing = {
            let state = self.hot_store_state.lock().unwrap();
            let has = state.continuations.get(channels).is_some();
            has
        };
        let from_history_store = if has_existing {
            None
        } else {
            Some(self.get_cont_from_history_store(channels))
        };

        let state = self.hot_store_state.lock().unwrap();
        let wc_identity = Self::continuation_identity(&wc);
        match state.continuations.entry(channels.to_vec()) {
            Entry::Occupied(mut occupied) => {
                if !occupied
                    .get()
                    .iter()
                    .any(|existing| Self::continuation_identity(existing) == wc_identity)
                {
                    occupied.get_mut().insert(0, wc);
                    inserted = true;
                }
            }
            Entry::Vacant(vacant) => {
                let mut new_continuations = from_history_store.unwrap_or_default();
                if !new_continuations
                    .iter()
                    .any(|existing| Self::continuation_identity(existing) == wc_identity)
                {
                    new_continuations.insert(0, wc);
                    inserted = true;
                }
                vacant.insert(new_continuations);
            }
        }
        Self::update_hot_store_state_metrics(&state);
        Some(inserted)
    }

    fn install_continuation(&self, channels: &[C], wc: WaitingContinuation<P, K>) -> Option<()> {
        // println!("hit install_continuation");
        let state = self.hot_store_state.lock().unwrap();
        let _ = state.installed_continuations.insert(channels.to_vec(), wc);
        Self::update_hot_store_state_metrics(&state);

        // println!("installed_continuation result: {:?}", result);
        // println!("to_map: {:?}\n", self.print());

        Some(())
    }

    fn remove_continuation(&self, channels: &[C], index: i32) -> Option<()> {

        let state = self.hot_store_state.lock().unwrap();
        let is_installed = state.installed_continuations.get(channels).is_some();
        let removing_installed = is_installed && index == 0;
        let removed_index = if is_installed { index - 1 } else { index };

        let result = if removing_installed {
            warn!("Attempted to remove an installed continuation");
            None
        } else {
            match state.continuations.entry(channels.to_vec()) {
                Entry::Occupied(mut occupied) => {
                    let len = occupied.get().len();
                    let out_of_bounds = removed_index < 0 || removed_index as usize >= len;
                    if out_of_bounds {
                        warn!(index, "Index out of bounds when removing continuation");
                        None
                    } else {
                        occupied.get_mut().remove(removed_index as usize);
                        Some(())
                    }
                }
                Entry::Vacant(vacant) => {
                    let mut from_history_store = self.get_cont_from_history_store(channels);
                    let len = from_history_store.len();
                    let out_of_bounds = removed_index < 0 || removed_index as usize >= len;
                    if out_of_bounds {
                        warn!(index, "Index out of bounds when removing continuation");
                        vacant.insert(from_history_store);
                        None
                    } else {
                        from_history_store.remove(removed_index as usize);
                        vacant.insert(from_history_store);
                        Some(())
                    }
                }
            }
        };
        Self::update_hot_store_state_metrics(&state);
        result
    }

    // Data

    fn get_data(&self, channel: &C) -> Vec<Datum<A>> {
        let maybe_data = {
            self.hot_store_state
                .lock()
                .unwrap()
                .data
                .get(channel)
                .map(|data| data.clone())
        };

        let hot_state_had_entry = maybe_data.is_some();
        let result = if let Some(data) = maybe_data {
            tracing::debug!(
                target: "f1r3fly.rspace.history",
                channel = ?channel,
                data_count = data.len(),
                source = "hot_state",
                "get_data: hot state hit ({} datums)",
                data.len()
            );
            data
        } else {
            // Read-only fallthrough: return history data WITHOUT caching in hot store state.
            // The history_store_cache provides read caching. Caching here would cause
            // changes() to re-emit unchanged data with a potentially different channel
            // serialization, orphaning the original trie entry.
            let data = self.get_data_from_history_store(channel);
            tracing::debug!(
                target: "f1r3fly.rspace.history",
                channel = ?channel,
                data_count = data.len(),
                source = "history_fallback",
                "get_data: hot state miss, fell through to history ({} datums)",
                data.len()
            );
            data
        };
        // LFS diagnostic: log when get_data returns empty for 32-byte GPrivate channels.
        // These are the channels that trigger DEAD END during treeHashMap replay.
        if result.is_empty() {
            let ch_dbg = format!("{:?}", channel);
            if ch_dbg.contains("GPrivateBody") && ch_dbg.len() > 200 {
                tracing::warn!(
                    target: "f1r3fly.rspace.lfs_diag",
                    channel = %ch_dbg,
                    hot_state_had_entry,
                    "GET_DATA EMPTY: 32-byte GPrivate channel returned 0 datums \
                     (hot_state={}, history=empty)",
                    if hot_state_had_entry { "had-entry-but-empty" } else { "miss" }
                );
            }
        }
        let state = self.hot_store_state.lock().unwrap();
        Self::update_hot_store_state_metrics(&state);
        result
    }

    fn put_datum(&self, channel: &C, d: Datum<A>) -> () {

        // Phase 5e: log put_datum calls on 32-byte GPrivate channels to trace
        // spurious data mutations on peek-only channels like treeHashMapCh
        if tracing::enabled!(target: "f1r3fly.rholang.diag", tracing::Level::WARN) {
            let ch_dbg = format!("{:?}", channel);
            if ch_dbg.contains("GPrivateBody") && ch_dbg.len() > 200 {
                let gprivate_hex: String = ch_dbg
                    .find("id: [")
                    .and_then(|start| {
                        ch_dbg[start..].find(']').map(|end| {
                            ch_dbg[start + 5..start + end].to_string()
                        })
                    })
                    .unwrap_or_else(|| "<unknown>".to_string());
                let existing_count = {
                    let state = self.hot_store_state.lock().unwrap();
                    state.data.get(channel).map(|d| d.len())
                };
                tracing::warn!(
                    target: "f1r3fly.rholang.diag",
                    gprivate_id = %gprivate_hex,
                    persist = d.persist,
                    existing_in_hot_state = ?existing_count,
                    "PUT_DATUM called on 32-byte GPrivate channel — \
                     existing_in_hot_state={:?}, persist={}",
                    existing_count, d.persist
                );
            }
        }

        let has_existing = {
            let state = self.hot_store_state.lock().unwrap();
            let has = state.data.get(channel).is_some();
            has
        };
        let from_history_store = if has_existing {
            None
        } else {
            Some(self.get_data_from_history_store(channel))
        };

        let state = self.hot_store_state.lock().unwrap();
        match state.data.entry(channel.clone()) {
            Entry::Occupied(mut occupied) => {
                occupied.get_mut().insert(0, d);
            }
            Entry::Vacant(vacant) => {
                let mut new_data = from_history_store.unwrap_or_default();
                new_data.insert(0, d);
                vacant.insert(new_data);
            }
        }
        Self::update_hot_store_state_metrics(&state);
    }

    fn remove_datum(&self, channel: &C, index: i32) -> Result<(), RSpaceError> {

        // Phase 5e: log remove_datum calls on 32-byte GPrivate channels — this is the
        // primary suspect for spurious DeleteData on peek-only channels like treeHashMapCh.
        // When remove_datum hits the Vacant path, it loads from history (1 datum), removes
        // it, and stores the resulting empty vector — which changes() then emits as DeleteData.
        if tracing::enabled!(target: "f1r3fly.rholang.diag", tracing::Level::WARN) {
            let ch_dbg = format!("{:?}", channel);
            if ch_dbg.contains("GPrivateBody") && ch_dbg.len() > 200 {
                let gprivate_hex: String = ch_dbg
                    .find("id: [")
                    .and_then(|start| {
                        ch_dbg[start..].find(']').map(|end| {
                            ch_dbg[start + 5..start + end].to_string()
                        })
                    })
                    .unwrap_or_else(|| "<unknown>".to_string());
                let existing_in_hot = {
                    let state = self.hot_store_state.lock().unwrap();
                    state.data.get(channel).map(|d| d.len())
                };
                tracing::warn!(
                    target: "f1r3fly.rholang.diag",
                    gprivate_id = %gprivate_hex,
                    index,
                    existing_in_hot_state = ?existing_in_hot,
                    "REMOVE_DATUM called on 32-byte GPrivate channel — \
                     index={}, existing_in_hot_state={:?}. \
                     If existing_in_hot_state=None, this will load from history and \
                     store the result (potentially empty) in hot state, causing \
                     spurious DeleteData on peek-only channels.",
                    index, existing_in_hot
                );
            }
        }

        let state = self.hot_store_state.lock().unwrap();
        let result = match state.data.entry(channel.clone()) {
            Entry::Occupied(mut occupied) => {
                let out_of_bounds = index < 0 || index as usize >= occupied.get().len();
                if out_of_bounds {
                    Err(RSpaceError::BugFoundError(format!(
                        "Index {} out of bounds when removing datum (len={})",
                        index,
                        occupied.get().len()
                    )))
                } else {
                    occupied.get_mut().remove(index as usize);
                    Ok(())
                }
            }
            Entry::Vacant(vacant) => {
                let mut from_history_store = self.get_data_from_history_store(channel);
                let out_of_bounds = index < 0 || index as usize >= from_history_store.len();
                if out_of_bounds {
                    let len = from_history_store.len();
                    vacant.insert(from_history_store);
                    Err(RSpaceError::BugFoundError(format!(
                        "Index {} out of bounds when removing datum (len={})",
                        index, len
                    )))
                } else {
                    from_history_store.remove(index as usize);
                    vacant.insert(from_history_store);
                    Ok(())
                }
            }
        };
        Self::update_hot_store_state_metrics(&state);
        result
    }

    // Joins

    fn get_joins(&self, channel: &C) -> Vec<Vec<C>> {
        // println!("\nHit get_joins");

        let (joins, installed_joins) = {
            let state = self.hot_store_state.lock().unwrap();
            (
                state.joins.get(channel).map(|j| j.clone()),
                state.installed_joins.get(channel).map(|j| j.clone()),
            )
        };

        let result = match joins {
            Some(joins_data) => {
                // println!("Found joins in store");
                let mut result = Vec::new();
                if let Some(installed) = installed_joins {
                    result.extend(installed);
                }
                result.extend(joins_data);
                result
            }
            None => {
                // Read-only fallthrough: return history joins WITHOUT caching in
                // hot store state. Caching here would cause changes() to re-emit
                // unchanged joins with a potentially different channel serialization,
                // orphaning the original trie entry.
                let from_history_store = self.get_joins_from_history_store(channel);
                tracing::debug!(
                    target: "f1r3fly.rspace",
                    channel = ?channel,
                    history_joins = from_history_store.len(),
                    "get_joins: fell through to history, found {}",
                    from_history_store.len()
                );

                let mut result = Vec::new();
                if let Some(installed) = installed_joins {
                    result.extend(installed);
                }
                result.extend(from_history_store);
                result
            }
        };
        let state = self.hot_store_state.lock().unwrap();
        Self::update_hot_store_state_metrics(&state);
        result
    }

    fn put_join(&self, channel: &C, join: &[C]) -> Option<()> {

        let has_existing = {
            let state = self.hot_store_state.lock().unwrap();
            let has = state.joins.get(channel).is_some();
            has
        };
        let from_history_store = if has_existing {
            None
        } else {
            Some(self.get_joins_from_history_store(channel))
        };

        let ch_dbg_hash = {
            let dbg = format!("{:?}", channel);
            super::hashing::blake2b256_hash::Blake2b256Hash::new(dbg.as_bytes())
        };
        tracing::info!(
            target: "f1r3fly.rspace.cost_trace",
            ch = %hex::encode(&ch_dbg_hash.bytes()[..8]),
            has_existing,
            history_joins = from_history_store.as_ref().map_or(0, |j| j.len()),
            "PUT_JOIN: ch={} has_existing={} history_joins={}",
            hex::encode(&ch_dbg_hash.bytes()[..8]),
            has_existing,
            from_history_store.as_ref().map_or(0, |j| j.len())
        );

        let state = self.hot_store_state.lock().unwrap();
        match state.joins.entry(channel.clone()) {
            Entry::Occupied(mut occupied) => {
                if !occupied.get().iter().any(|j| j.as_slice() == join) {
                    occupied.get_mut().insert(0, join.to_vec());
                }
            }
            Entry::Vacant(vacant) => {
                let mut joins = from_history_store.unwrap_or_default();
                if !joins.iter().any(|j| j.as_slice() == join) {
                    joins.insert(0, join.to_vec());
                }
                vacant.insert(joins);
            }
        }
        Self::update_hot_store_state_metrics(&state);
        Some(())
    }

    fn install_join(&self, channel: &C, join: &[C]) -> Option<()> {
        // println!("hit install_join");
        let state = self.hot_store_state.lock().unwrap();
        match state.installed_joins.entry(channel.clone()) {
            Entry::Occupied(mut occupied) => {
                if !occupied.get().iter().any(|j| j.as_slice() == join) {
                    occupied.get_mut().insert(0, join.to_vec());
                }
            }
            Entry::Vacant(vacant) => {
                vacant.insert(vec![join.to_vec()]);
            }
        }
        Self::update_hot_store_state_metrics(&state);
        Some(())
    }

    fn remove_join(&self, channel: &C, join: &[C]) -> Option<()> {

        let state = self.hot_store_state.lock().unwrap();
        let current_continuations = {
            let mut conts = state
                .installed_continuations
                .get(join)
                .map(|c| vec![c.clone()])
                .unwrap_or_else(Vec::new);
            conts.extend(
                state
                    .continuations
                    .get(join)
                    .map(|continuations| continuations.clone())
                    .unwrap_or_else(|| self.get_cont_from_history_store(join)),
            );
            conts
        };

        // Remove join is called when continuation is removed, so it can be called when
        // continuations are present in which case we just want to skip removal.
        let do_remove = current_continuations.is_empty();

        let ch_dbg_hash = {
            let dbg = format!("{:?}", channel);
            super::hashing::blake2b256_hash::Blake2b256Hash::new(dbg.as_bytes())
        };
        let has_hot_entry = state.joins.get(channel).is_some();
        tracing::info!(
            target: "f1r3fly.rspace.cost_trace",
            ch = %hex::encode(&ch_dbg_hash.bytes()[..8]),
            do_remove,
            conts_count = current_continuations.len(),
            has_hot_entry,
            "REMOVE_JOIN: ch={} do_remove={} conts={} hot_entry={}",
            hex::encode(&ch_dbg_hash.bytes()[..8]),
            do_remove, current_continuations.len(), has_hot_entry
        );

        let result = if !do_remove {
            // Continuations still exist, so we skip removal. No need to cache
            // history joins into hot store state — doing so would cause changes()
            // to re-emit unchanged joins with a potentially different channel
            // serialization, orphaning the original trie entry.
            Some(())
        } else {
            match state.joins.entry(channel.clone()) {
                Entry::Occupied(mut occupied) => {
                    if let Some(idx) = occupied.get().iter().position(|x| x.as_slice() == join) {
                        occupied.get_mut().remove(idx);
                    } else {
                        warn!("Join not found when removing join");
                    }
                    Some(())
                }
                Entry::Vacant(vacant) => {
                    let mut joins_in_history_store = self.get_joins_from_history_store(channel);
                    if let Some(idx) = joins_in_history_store
                        .iter()
                        .position(|x| x.as_slice() == join)
                    {
                        joins_in_history_store.remove(idx);
                    } else {
                        warn!("Join not found when removing join");
                    }
                    vacant.insert(joins_in_history_store);
                    Some(())
                }
            }
        };
        Self::update_hot_store_state_metrics(&state);
        result
    }

    fn changes(&self) -> Vec<HotStoreAction<C, P, A, K>> {
        // NOTE: Channel normalization (clearing locally_free) is performed upstream
        // in produce_inner/consume_inner before channels enter the hot store. Ideally
        // we would also normalize here as a defensive measure, but C is generic and
        // adding a NormalizeForHashing trait bound would ripple through HotStore,
        // ISpace, RSpace, ReplayRSpace, ReportingRSpace, and all test files. Instead,
        // we use the Debug representation as a canary to detect if any non-normalized
        // channel ever reaches changes().
        let cache = self.hot_store_state.lock().unwrap();
        let continuations: Vec<HotStoreAction<C, P, A, K>> = cache
            .continuations
            .iter()
            .map(|entry| {
                let (k, v) = entry.pair();
                if v.is_empty() {
                    tracing::warn!(
                        target: "f1r3fly.rspace",
                        channels = ?k,
                        "changes(): emitting DeleteContinuations — channel will be cleared from trie"
                    );
                    HotStoreAction::Delete(DeleteAction::DeleteContinuations(DeleteContinuations {
                        channels: k.clone(),
                    }))
                } else {
                    HotStoreAction::Insert(InsertAction::InsertContinuations(InsertContinuations {
                        channels: k.clone(),
                        continuations: v.clone(),
                    }))
                }
            })
            .collect();

        let data: Vec<HotStoreAction<C, P, A, K>> = cache
            .data
            .iter()
            .map(|entry| {
                let (k, v) = entry.pair();
                if v.is_empty() {
                    tracing::warn!(
                        target: "f1r3fly.rholang.diag",
                        channel = ?k,
                        "changes(): emitting DeleteData for channel"
                    );
                    HotStoreAction::Delete(DeleteAction::DeleteData(DeleteData {
                        channel: k.clone(),
                    }))
                } else {
                    HotStoreAction::Insert(InsertAction::InsertData(InsertData {
                        channel: k.clone(),
                        data: v.clone(),
                    }))
                }
            })
            .collect();

        let joins: Vec<HotStoreAction<C, P, A, K>> = cache
            .joins
            .iter()
            .map(|entry| {
                let (k, v) = entry.pair();
                if v.is_empty() {
                    tracing::warn!(
                        target: "f1r3fly.rholang.diag",
                        channel = ?k,
                        "changes(): emitting DeleteJoins for channel"
                    );
                    HotStoreAction::Delete(DeleteAction::DeleteJoins(DeleteJoins {
                        channel: k.clone(),
                    }))
                } else {
                    HotStoreAction::Insert(InsertAction::InsertJoins(InsertJoins {
                        channel: k.clone(),
                        joins: v.clone(),
                    }))
                }
            })
            .collect();

        let all = [continuations, data, joins].concat();

        // Canary: detect non-normalized channels (non-empty locally_free) that
        // slipped past produce_inner/consume_inner normalization. The Debug repr
        // of Par includes "locally_free: [...]" — if it contains non-empty content,
        // the channel hash will differ from normalized versions, causing trie
        // lookup failures.
        if tracing::enabled!(target: "f1r3fly.rholang.diag", tracing::Level::WARN) {
            for action in &all {
                let channel_debug = match action {
                    HotStoreAction::Insert(InsertAction::InsertData(i)) => {
                        Some(format!("{:?}", i.channel))
                    }
                    HotStoreAction::Delete(DeleteAction::DeleteData(d)) => {
                        Some(format!("{:?}", d.channel))
                    }
                    HotStoreAction::Insert(InsertAction::InsertJoins(i)) => {
                        Some(format!("{:?}", i.channel))
                    }
                    HotStoreAction::Delete(DeleteAction::DeleteJoins(d)) => {
                        Some(format!("{:?}", d.channel))
                    }
                    _ => None, // Continuations use Vec<C> keys, checked separately
                };
                if let Some(debug_str) = channel_debug {
                    // Check for non-empty locally_free in the Debug output.
                    // A normalized channel has "locally_free: []" — anything else
                    // indicates the channel was not normalized before entering the
                    // hot store.
                    if let Some(pos) = debug_str.find("locally_free: [") {
                        let after = &debug_str[pos + "locally_free: [".len()..];
                        if !after.starts_with(']') {
                            tracing::error!(
                                target: "f1r3fly.rholang.diag",
                                channel_debug = %debug_str,
                                "changes(): CANARY — channel has non-empty locally_free! \
                                 This will cause trie hash mismatch. Channel was not \
                                 normalized in produce_inner/consume_inner."
                            );
                        }
                    }
                }
                // Check continuation channels (Vec<C>)
                let cont_channels_debug = match action {
                    HotStoreAction::Insert(InsertAction::InsertContinuations(i)) => {
                        Some(format!("{:?}", i.channels))
                    }
                    HotStoreAction::Delete(DeleteAction::DeleteContinuations(d)) => {
                        Some(format!("{:?}", d.channels))
                    }
                    _ => None,
                };
                if let Some(debug_str) = cont_channels_debug {
                    if let Some(pos) = debug_str.find("locally_free: [") {
                        let after = &debug_str[pos + "locally_free: [".len()..];
                        if !after.starts_with(']') {
                            tracing::error!(
                                target: "f1r3fly.rholang.diag",
                                channels_debug = %debug_str,
                                "changes(): CANARY — continuation channels have non-empty \
                                 locally_free! This will cause trie hash mismatch."
                            );
                        }
                    }
                }
            }
        }

        tracing::info!(
            target: "f1r3fly.rholang.diag",
            total_actions = all.len(),
            data_inserts = all.iter().filter(|a| matches!(a, HotStoreAction::Insert(InsertAction::InsertData(_)))).count(),
            data_deletes = all.iter().filter(|a| matches!(a, HotStoreAction::Delete(DeleteAction::DeleteData(_)))).count(),
            cont_inserts = all.iter().filter(|a| matches!(a, HotStoreAction::Insert(InsertAction::InsertContinuations(_)))).count(),
            cont_deletes = all.iter().filter(|a| matches!(a, HotStoreAction::Delete(DeleteAction::DeleteContinuations(_)))).count(),
            join_inserts = all.iter().filter(|a| matches!(a, HotStoreAction::Insert(InsertAction::InsertJoins(_)))).count(),
            join_deletes = all.iter().filter(|a| matches!(a, HotStoreAction::Delete(DeleteAction::DeleteJoins(_)))).count(),
            "changes(): checkpoint action summary"
        );

        all
    }

    fn to_map(&self) -> HashMap<Vec<C>, Row<P, A, K>> {
        let state = self.hot_store_state.lock().unwrap();
        let data = state
            .data
            .iter()
            .map(|entry| {
                let (k, v) = entry.pair();
                (vec![k.clone()], v.clone())
            })
            .collect::<HashMap<_, _>>();

        let all_continuations = {
            let mut all = state
                .continuations
                .iter()
                .map(|entry| {
                    let (k, v) = entry.pair();
                    (k.clone(), v.clone())
                })
                .collect::<HashMap<_, _>>();
            for (k, v) in state.installed_continuations.iter().map(|entry| {
                let (k, v) = entry.pair();
                (k.clone(), v.clone())
            }) {
                all.entry(k).or_insert_with(Vec::new).push(v);
            }
            all
        };

        let mut map = HashMap::new();

        // Include channels with data (and their continuations if any)
        for (k, v) in data.into_iter() {
            let row = Row {
                data: v,
                wks: all_continuations.get(&k).cloned().unwrap_or_else(Vec::new),
            };
            if !(row.data.is_empty() && row.wks.is_empty()) {
                map.insert(k, row);
            }
        }

        // Include channels with only continuations (no data)
        for (k, v) in all_continuations.into_iter() {
            if !map.contains_key(&k) && !v.is_empty() {
                map.insert(k, Row { data: Vec::new(), wks: v });
            }
        }

        map
    }

    fn print(&self) {
        let hot_store_state = self.hot_store_state.lock().unwrap();
        println!("\nHot Store");

        println!("Continuations:");
        for entry in hot_store_state.continuations.iter() {
            let (key, value) = entry.pair();
            println!("Key: {:?}, Value: {:?}", key, value);
        }

        println!("\nInstalled Continuations:");
        for entry in hot_store_state.installed_continuations.iter() {
            let (key, value) = entry.pair();
            println!("Key: {:?}, Value: {:?}", key, value);
        }

        println!("\nData:");
        for entry in hot_store_state.data.iter() {
            let (key, value) = entry.pair();
            println!("Key: {:?}, Value: {:?}", key, value);
        }

        println!("\nJoins:");
        for entry in hot_store_state.joins.iter() {
            let (key, value) = entry.pair();
            println!("Key: {:?}, Value: {:?}", key, value);
        }

        println!("\nInstalled Joins:");
        for entry in hot_store_state.installed_joins.iter() {
            let (key, value) = entry.pair();
            println!("Key: {:?}, Value: {:?}", key, value);
        }

        let history_cache_state = self.history_store_cache.lock().unwrap();
        println!("\nHistory Cache");

        println!("Continuations:");
        for entry in history_cache_state.continuations.iter() {
            let (key, value) = entry.pair();
            println!("Key: {:?}, Value: {:?}", key, value);
        }

        println!("\nData:");
        for entry in history_cache_state.datums.iter() {
            let (key, value) = entry.pair();
            println!("Key: {:?}, Value: {:?}", key, value);
        }

        println!("\nJoins:");
        for entry in history_cache_state.joins.iter() {
            let (key, value) = entry.pair();
            println!("Key: {:?}, Value: {:?}", key, value);
        }

        // println!("\nHistory");
        // println!("Continuations: {:?}",
        // self.history_reader_base.get_continuations(&channels));
    }

    fn clear(&self) {
        let mut state = self.hot_store_state.lock().unwrap();
        state.continuations = DashMap::new();
        state.installed_continuations = DashMap::new();
        state.data = DashMap::new();
        state.joins = DashMap::new();
        state.installed_joins = DashMap::new();
        drop(state);

        let history_cache = self.history_store_cache.lock().unwrap();
        history_cache.continuations.clear();
        history_cache.datums.clear();
        history_cache.joins.clear();
        metrics::gauge!(HOT_STORE_HISTORY_CONT_CACHE_SIZE_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(0.0);
        metrics::gauge!(HOT_STORE_HISTORY_DATA_CACHE_SIZE_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(0.0);
        metrics::gauge!(HOT_STORE_HISTORY_JOINS_CACHE_SIZE_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(0.0);
        metrics::gauge!(HOT_STORE_HISTORY_CONT_CACHE_ITEMS_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(0.0);
        metrics::gauge!(HOT_STORE_HISTORY_DATA_CACHE_ITEMS_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(0.0);
        metrics::gauge!(HOT_STORE_HISTORY_JOINS_CACHE_ITEMS_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(0.0);
        metrics::gauge!(HOT_STORE_STATE_INSTALLED_CONT_SIZE_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(0.0);
        metrics::gauge!(HOT_STORE_STATE_INSTALLED_JOINS_SIZE_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(0.0);
        metrics::gauge!(HOT_STORE_STATE_INSTALLED_CONT_ITEMS_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(0.0);
        metrics::gauge!(HOT_STORE_STATE_INSTALLED_JOINS_ITEMS_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(0.0);

        let state = self.hot_store_state.lock().unwrap();
        Self::update_hot_store_state_metrics(&state);
    }

    // See rspace/src/test/scala/coop/rchain/rspace/test/package.scala
    fn is_empty(&self) -> bool {
        let store_actions = self.changes();
        let has_insert_actions = store_actions
            .into_iter()
            .any(|action| matches!(action, HotStoreAction::Insert(_)));

        !has_insert_actions
    }

    fn state_counts(&self) -> (usize, usize, usize, usize) {
        let state = self.hot_store_state.lock().expect("hot_store_state lock poisoned");
        let data_channels = state.data.len();
        let data_items: usize = state.data.iter().map(|e| e.value().len()).sum();
        let cont_channels = state.continuations.len();
        let cont_items: usize = state.continuations.iter().map(|e| e.value().len()).sum();
        (data_channels, data_items, cont_channels, cont_items)
    }

    fn continuation_channels_debug(&self) -> Vec<(String, usize, bool)> {
        let state = self.hot_store_state.lock().expect("hot_store_state lock poisoned");
        state
            .continuations
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .map(|entry| {
                let channels_dbg = format!("{:?}", entry.key());
                let count = entry.value().len();
                let has_peek = entry.value().iter().any(|wc| !wc.peeks.is_empty());
                (channels_dbg, count, has_peek)
            })
            .collect()
    }

}

impl<C, P, A, K> InMemHotStore<C, P, A, K>
where
    C: Clone + Debug + Hash + Eq + Sync + Send,
    P: Clone + Debug + Sync + Send,
    A: Clone + Debug + Sync + Send,
    K: Clone + Debug + Sync + Send,
{
    fn continuation_identity(wc: &WaitingContinuation<P, K>) -> String {
        format!(
            "{:?}|{:?}|{}|{:?}",
            wc.patterns, wc.continuation, wc.persist, wc.peeks
        )
    }

    fn now_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    fn state_metrics_update_interval_ms() -> u64 {
        static VALUE: OnceLock<u64> = OnceLock::new();
        *VALUE.get_or_init(|| {
            std::env::var(HOT_STORE_STATE_METRICS_UPDATE_INTERVAL_MS_ENV)
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(HOT_STORE_STATE_METRICS_UPDATE_INTERVAL_MS_DEFAULT)
        })
    }

    fn history_cache_metrics_update_interval_ms() -> u64 {
        static VALUE: OnceLock<u64> = OnceLock::new();
        *VALUE.get_or_init(|| {
            std::env::var(HOT_STORE_HISTORY_CACHE_METRICS_UPDATE_INTERVAL_MS_ENV)
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(HOT_STORE_HISTORY_CACHE_METRICS_UPDATE_INTERVAL_MS_DEFAULT)
        })
    }

    fn should_emit_metrics(last_emit_at_ms: &AtomicU64, update_interval_ms: u64) -> bool {
        if update_interval_ms == 0 {
            return true;
        }

        let now = Self::now_millis();
        loop {
            let last = last_emit_at_ms.load(Ordering::Relaxed);
            if now.saturating_sub(last) < update_interval_ms {
                return false;
            }
            if last_emit_at_ms
                .compare_exchange_weak(last, now, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
        }
    }

    fn update_hot_store_state_metrics(state: &HotStoreState<C, P, A, K>) {
        static LAST_EMIT_AT_MS: AtomicU64 = AtomicU64::new(0);
        if !Self::should_emit_metrics(&LAST_EMIT_AT_MS, Self::state_metrics_update_interval_ms()) {
            return;
        }

        let cont_items: usize = state
            .continuations
            .iter()
            .map(|entry| entry.value().len())
            .sum();
        let data_items: usize = state.data.iter().map(|entry| entry.value().len()).sum();
        let joins_items: usize = state.joins.iter().map(|entry| entry.value().len()).sum();
        let installed_cont_items = state.installed_continuations.len();
        let installed_joins_items: usize = state
            .installed_joins
            .iter()
            .map(|entry| entry.value().len())
            .sum();

        metrics::gauge!(HOT_STORE_STATE_CONT_SIZE_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(state.continuations.len() as f64);
        metrics::gauge!(HOT_STORE_STATE_DATA_SIZE_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(state.data.len() as f64);
        metrics::gauge!(HOT_STORE_STATE_JOINS_SIZE_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(state.joins.len() as f64);
        metrics::gauge!(HOT_STORE_STATE_INSTALLED_CONT_SIZE_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(state.installed_continuations.len() as f64);
        metrics::gauge!(HOT_STORE_STATE_INSTALLED_JOINS_SIZE_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(state.installed_joins.len() as f64);
        metrics::gauge!(HOT_STORE_STATE_CONT_ITEMS_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(cont_items as f64);
        metrics::gauge!(HOT_STORE_STATE_DATA_ITEMS_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(data_items as f64);
        metrics::gauge!(HOT_STORE_STATE_JOINS_ITEMS_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(joins_items as f64);
        metrics::gauge!(HOT_STORE_STATE_INSTALLED_CONT_ITEMS_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(installed_cont_items as f64);
        metrics::gauge!(HOT_STORE_STATE_INSTALLED_JOINS_ITEMS_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(installed_joins_items as f64);
    }

    fn update_history_cache_metrics(cache: &HistoryStoreCache<C, P, A, K>) {
        static LAST_EMIT_AT_MS: AtomicU64 = AtomicU64::new(0);
        if !Self::should_emit_metrics(
            &LAST_EMIT_AT_MS,
            Self::history_cache_metrics_update_interval_ms(),
        ) {
            return;
        }

        let cont_items: usize = cache
            .continuations
            .iter()
            .map(|entry| entry.value().len())
            .sum();
        let data_items: usize = cache.datums.iter().map(|entry| entry.value().len()).sum();
        let joins_items: usize = cache.joins.iter().map(|entry| entry.value().len()).sum();

        metrics::gauge!(HOT_STORE_HISTORY_CONT_CACHE_SIZE_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(cache.continuations.len() as f64);
        metrics::gauge!(HOT_STORE_HISTORY_DATA_CACHE_SIZE_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(cache.datums.len() as f64);
        metrics::gauge!(HOT_STORE_HISTORY_JOINS_CACHE_SIZE_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(cache.joins.len() as f64);
        metrics::gauge!(HOT_STORE_HISTORY_CONT_CACHE_ITEMS_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(cont_items as f64);
        metrics::gauge!(HOT_STORE_HISTORY_DATA_CACHE_ITEMS_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(data_items as f64);
        metrics::gauge!(HOT_STORE_HISTORY_JOINS_CACHE_ITEMS_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .set(joins_items as f64);
    }

    fn enforce_history_cache_bounds(cache: &HistoryStoreCache<C, P, A, K>) {
        let cont_items: usize = cache
            .continuations
            .iter()
            .map(|entry| entry.value().len())
            .sum();
        let data_items: usize = cache.datums.iter().map(|entry| entry.value().len()).sum();
        let joins_items: usize = cache.joins.iter().map(|entry| entry.value().len()).sum();

        if cache.continuations.len() >= MAX_HISTORY_STORE_CACHE_ENTRIES ||
            cont_items >= MAX_HISTORY_STORE_CACHE_CONT_ITEMS
        {
            cache.continuations.clear();
        }
        if cache.datums.len() >= MAX_HISTORY_STORE_CACHE_ENTRIES ||
            data_items >= MAX_HISTORY_STORE_CACHE_DATA_ITEMS
        {
            cache.datums.clear();
        }
        if cache.joins.len() >= MAX_HISTORY_STORE_CACHE_ENTRIES ||
            joins_items >= MAX_HISTORY_STORE_CACHE_JOIN_ITEMS
        {
            cache.joins.clear();
        }
    }

    fn get_cont_from_history_store(&self, channels: &[C]) -> Vec<WaitingContinuation<P, K>> {
        let cache = self.history_store_cache.lock().unwrap();
        Self::enforce_history_cache_bounds(&cache);
        let channels_vec = channels.to_vec();
        let entry = cache.continuations.entry(channels_vec.clone());
        let result = match entry {
            Entry::Occupied(o) => {
                let cached = o.get().clone();
                tracing::debug!(
                    target: "f1r3fly.rspace.history",
                    channels = ?channels,
                    cont_count = cached.len(),
                    source = "cache",
                    "get_cont_from_history_store: cache hit ({} continuations)",
                    cached.len()
                );
                cached
            }
            Entry::Vacant(v) => {
                let ks = self.history_reader_base.get_continuations(&channels_vec);

                tracing::debug!(
                    target: "f1r3fly.rspace.history",
                    channels = ?channels,
                    cont_count = ks.len(),
                    source = "history_reader",
                    "get_cont_from_history_store: cache miss, history returned {} continuations",
                    ks.len()
                );

                if tracing::enabled!(target: "f1r3fly.rspace.matcher", tracing::Level::DEBUG) {
                    for (i, wc) in ks.iter().enumerate() {
                        tracing::debug!(
                            target: "f1r3fly.rspace.matcher",
                            channels = ?channels,
                            cont_idx = i,
                            num_patterns = wc.patterns.len(),
                            persist = wc.persist,
                            patterns = ?wc.patterns,
                            "get_cont_from_history_store: loaded continuation #{} ({} patterns, persist={})",
                            i, wc.patterns.len(), wc.persist
                        );
                    }
                }

                v.insert(ks.clone());
                ks
            }
        };
        Self::update_history_cache_metrics(&cache);
        result
    }

    fn get_data_from_history_store(&self, channel: &C) -> Vec<Datum<A>> {
        let cache = self.history_store_cache.lock().unwrap();
        Self::enforce_history_cache_bounds(&cache);
        let entry = cache.datums.entry(channel.clone());
        let result = match entry {
            Entry::Occupied(o) => {
                let cached = o.get().clone();
                tracing::debug!(
                    target: "f1r3fly.rspace.history",
                    channel = ?channel,
                    data_count = cached.len(),
                    source = "cache",
                    "get_data_from_history_store: cache hit ({} datums)",
                    cached.len()
                );
                cached
            }
            Entry::Vacant(v) => {
                let datums = self.history_reader_base.get_data(channel);
                tracing::debug!(
                    target: "f1r3fly.rspace.history",
                    channel = ?channel,
                    data_count = datums.len(),
                    source = "history_reader",
                    "get_data_from_history_store: cache miss, history returned {} datums",
                    datums.len()
                );

                // Phase 5d Step 3: for GPrivate channels returning 0 datums,
                // log channel identity for cross-referencing with init deploy
                // checkpoint InsertData entries.
                if datums.is_empty() {
                    let ch_dbg = format!("{:?}", channel);
                    // Heuristic: 32-byte GPrivate channels have long debug
                    // representations (>200 chars) containing "id: [...]"
                    if ch_dbg.len() > 200 || ch_dbg.contains("GPrivateBody") {
                        let gprivate_hex: String = ch_dbg
                            .find("id: [")
                            .and_then(|start| {
                                ch_dbg[start..].find(']').map(|end| {
                                    ch_dbg[start + 5..start + end].to_string()
                                })
                            })
                            .unwrap_or_else(|| "<unknown>".to_string());

                        // Check ALL locally_free occurrences in the Debug string,
                        // not just the top-level one. Nested structures (Send,
                        // Receive, New, Match) also have locally_free that
                        // affects bincode serialization and thus the history hash.
                        let locally_free_fields: Vec<String> = ch_dbg
                            .match_indices("locally_free: [")
                            .filter_map(|(pos, _)| {
                                let after = &ch_dbg[pos + "locally_free: [".len()..];
                                if after.starts_with(']') {
                                    None // empty, skip
                                } else {
                                    // Extract the content up to the closing bracket
                                    after.find(']').map(|end| {
                                        format!("@{}: [{}]", pos, &after[..end])
                                    })
                                }
                            })
                            .collect();

                        tracing::warn!(
                            target: "f1r3fly.rholang.diag",
                            gprivate_id = %gprivate_hex,
                            nonempty_locally_free_count = locally_free_fields.len(),
                            nonempty_locally_free = ?locally_free_fields,
                            channel = %ch_dbg,
                            "HISTORY DATA MISS: GPrivate channel returned 0 \
                             datums from history. gprivate_id={}, non-empty locally_free={}",
                            gprivate_hex, locally_free_fields.len()
                        );
                    }
                }

                v.insert(datums.clone());
                datums
            }
        };
        Self::update_history_cache_metrics(&cache);
        result
    }

    fn get_joins_from_history_store(&self, channel: &C) -> Vec<Vec<C>> {
        let cache = self.history_store_cache.lock().unwrap();
        Self::enforce_history_cache_bounds(&cache);
        let ch_dbg = format!("{:?}", channel);
        let is_byte_name_14 = ch_dbg.contains("id: [14]");
        let entry = cache.joins.entry(channel.clone());
        let result = match entry {
            Entry::Occupied(o) => {
                let cached = o.get().clone();
                if is_byte_name_14 {
                    tracing::info!(
                        target: "f1r3fly.rholang.diag",
                        cached_joins = cached.len(),
                        channel_debug = %ch_dbg,
                        "get_joins_from_history_store byte_name(14): CACHE HIT, {} joins",
                        cached.len()
                    );
                } else {
                    tracing::debug!(
                        target: "f1r3fly.rspace",
                        channel = ?channel,
                        cached_joins = cached.len(),
                        "get_joins_from_history_store: cache hit"
                    );
                }
                cached
            }
            Entry::Vacant(v) => {
                let joins = self.history_reader_base.get_joins(&channel);
                if is_byte_name_14 {
                    // Log each join group for byte_name(14) at INFO level
                    for (i, join_group) in joins.iter().enumerate() {
                        let join_dbg: Vec<String> = join_group.iter().map(|c| format!("{:?}", c)).collect();
                        tracing::info!(
                            target: "f1r3fly.rholang.diag",
                            join_idx = i,
                            join_channels = ?join_dbg,
                            "get_joins_from_history_store byte_name(14): history join group #{}: {:?}",
                            i, join_dbg
                        );
                    }
                    tracing::info!(
                        target: "f1r3fly.rholang.diag",
                        history_joins = joins.len(),
                        channel_debug = %ch_dbg,
                        "get_joins_from_history_store byte_name(14): CACHE MISS, history returned {} joins",
                        joins.len()
                    );
                } else {
                    tracing::debug!(
                        target: "f1r3fly.rspace",
                        channel = ?channel,
                        history_joins = joins.len(),
                        "get_joins_from_history_store: cache miss, queried history"
                    );
                }
                v.insert(joins.clone());
                joins
            }
        };
        Self::update_history_cache_metrics(&cache);
        result
    }
}

pub struct HotStoreInstances;

impl HotStoreInstances {
    pub fn create_from_mhs_and_hr<C, P, A, K>(
        hot_store_state_ref: Arc<Mutex<HotStoreState<C, P, A, K>>>,
        history_reader_base: Box<dyn HistoryReaderBase<C, P, A, K>>,
    ) -> Box<dyn HotStore<C, P, A, K>>
    where
        C: Default + Clone + Debug + Eq + Hash + Send + Sync + 'static,
        P: Default + Clone + Debug + Send + Sync + 'static,
        A: Default + Clone + Debug + Send + Sync + 'static,
        K: Default + Clone + Debug + Send + Sync + 'static,
    {
        Box::new(InMemHotStore {
            hot_store_state: hot_store_state_ref,
            history_store_cache: Arc::new(Mutex::new(HistoryStoreCache::default())),
            history_reader_base,
        })
    }

    pub fn create_from_hs_and_hr<C, P, A, K>(
        cache: HotStoreState<C, P, A, K>,
        history_reader: Box<dyn HistoryReaderBase<C, P, A, K>>,
    ) -> Box<dyn HotStore<C, P, A, K>>
    where
        C: Default + Clone + Debug + Eq + Hash + Send + Sync + 'static,
        P: Default + Clone + Debug + Send + Sync + 'static,
        A: Default + Clone + Debug + Send + Sync + 'static,
        K: Default + Clone + Debug + Send + Sync + 'static,
    {
        let cache = Arc::new(Mutex::new(cache));
        let store = HotStoreInstances::create_from_mhs_and_hr(cache, history_reader);
        store
    }

    pub fn create_from_hr<C, P, A, K>(
        history_reader: Box<dyn HistoryReaderBase<C, P, A, K>>,
    ) -> Box<dyn HotStore<C, P, A, K>>
    where
        C: Default + Clone + Debug + Eq + Hash + 'static + Send + Sync,
        P: Default + Clone + Debug + 'static + Send + Sync,
        A: Default + Clone + Debug + 'static + Send + Sync,
        K: Default + Clone + Debug + 'static + Send + Sync,
    {
        HotStoreInstances::create_from_hs_and_hr(HotStoreState::default(), history_reader)
    }
}
