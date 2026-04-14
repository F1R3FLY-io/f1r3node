use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use std::collections::hash_map::Entry;
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
const HOT_STORE_STATE_METRICS_UPDATE_INTERVAL_MS: u64 = 250;
const HOT_STORE_HISTORY_CACHE_METRICS_UPDATE_INTERVAL_MS: u64 = 250;

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

    fn set_state(&self, state: HotStoreState<C, P, A, K>);
}

pub fn new_hashmap<K: std::cmp::Eq + std::hash::Hash, V>() -> HashMap<K, V> { HashMap::new() }

#[derive(Default, Debug, Clone)]
pub struct HotStoreState<C, P, A, K>
where
    C: Eq + Hash,
    A: Clone,
    P: Clone,
    K: Clone,
{
    pub continuations: HashMap<Vec<C>, Vec<WaitingContinuation<P, K>>>,
    pub installed_continuations: HashMap<Vec<C>, WaitingContinuation<P, K>>,
    pub data: HashMap<C, Vec<Datum<A>>>,
    pub joins: HashMap<C, Vec<Vec<C>>>,
    pub installed_joins: HashMap<C, Vec<Vec<C>>>,
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
            continuations: HashMap::from_iter(vec![(channels.clone(), continuations.clone())]),
            installed_continuations: HashMap::from_iter(vec![(
                channels.clone(),
                installed_continuations.clone(),
            )]),
            data: HashMap::from_iter(vec![(channel.clone(), data.clone())]),
            joins: HashMap::from_iter(vec![(channel.clone(), joins)]),
            installed_joins: HashMap::from_iter(vec![(channel, installed_joins)]),
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
    continuations: HashMap<Vec<C>, Vec<WaitingContinuation<P, K>>>,
    datums: HashMap<C, Vec<Datum<A>>>,
    joins: HashMap<C, Vec<Vec<C>>>,
}

struct InMemHotStore<C, P, A, K>
where
    C: Eq + Hash + Sync + Send,
    A: Clone + Sync + Send,
    P: Clone + Sync + Send,
    K: Clone + Sync + Send,
{
    state: std::sync::RwLock<HotStoreState<C, P, A, K>>,
    history_cache: std::sync::RwLock<HistoryStoreCache<C, P, A, K>>,
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
        let state = self.state.read().expect("hot store state read lock");
        HotStoreState {
            continuations: state.continuations.clone(),
            installed_continuations: state.installed_continuations.clone(),
            data: state.data.clone(),
            joins: state.joins.clone(),
            installed_joins: state.installed_joins.clone(),
        }
    }

    // Continuations

    fn get_continuations(&self, channels: &[C]) -> Vec<WaitingContinuation<P, K>> {
        let state = self.state.read().expect("hot store state read lock");
        let continuations = state.continuations.get(channels).cloned();
        let installed = state.installed_continuations.get(channels).cloned();
        drop(state);

        match (continuations, installed) {
            (Some(conts), Some(inst)) => {
                let mut result = Vec::with_capacity(conts.len() + 1);
                result.push(inst);
                result.extend(conts);
                result
            }
            (Some(conts), None) => conts,
            (None, Some(inst)) => {
                let from_history_store = self.get_cont_from_history_store(channels);
                self.state.write().expect("hot store state write lock")
                    .continuations
                    .insert(channels.to_vec(), from_history_store.clone());
                let mut result = Vec::with_capacity(from_history_store.len() + 1);
                result.push(inst);
                result.extend(from_history_store);
                result
            }
            (None, None) => {
                let from_history_store = self.get_cont_from_history_store(channels);
                self.state.write().expect("hot store state write lock")
                    .continuations
                    .insert(channels.to_vec(), from_history_store.clone());
                from_history_store
            }
        }
    }

    fn put_continuation(&self, channels: &[C], wc: WaitingContinuation<P, K>) -> Option<bool> {
        let mut inserted = false;
        let has_existing = self.state.read().expect("hot store state read lock").continuations.get(channels).is_some();
        let from_history_store = if has_existing {
            None
        } else {
            Some(self.get_cont_from_history_store(channels))
        };

        let mut state = self.state.write().expect("hot store state write lock");
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
        Some(inserted)
    }

    fn install_continuation(&self, channels: &[C], wc: WaitingContinuation<P, K>) -> Option<()> {
        self.state.write().expect("hot store state write lock").installed_continuations.insert(channels.to_vec(), wc);
        Some(())
    }

    fn remove_continuation(&self, channels: &[C], index: i32) -> Option<()> {
        let mut state = self.state.write().expect("hot store state write lock");
        let is_installed = state.installed_continuations.get(channels).is_some();
        let removing_installed = is_installed && index == 0;
        let removed_index = if is_installed { index - 1 } else { index };

        if removing_installed {
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
        }
    }

    // Data

    fn get_data(&self, channel: &C) -> Vec<Datum<A>> {
        let maybe_data = self.state.read().expect("hot store state read lock").data.get(channel).cloned();

        if let Some(data) = maybe_data {
            data
        } else {
            let data = self.get_data_from_history_store(channel);
            self.state.write().expect("hot store state write lock")
                .data
                .insert(channel.clone(), data.clone());
            data
        }
    }

    fn put_datum(&self, channel: &C, d: Datum<A>) -> () {
        let has_existing = self.state.read().expect("hot store state read lock").data.get(channel).is_some();
        let from_history_store = if has_existing {
            None
        } else {
            Some(self.get_data_from_history_store(channel))
        };

        let mut state = self.state.write().expect("hot store state write lock");
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
    }

    fn remove_datum(&self, channel: &C, index: i32) -> Result<(), RSpaceError> {
        let mut state = self.state.write().expect("hot store state write lock");
        match state.data.entry(channel.clone()) {
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
        }
    }

    // Joins

    fn get_joins(&self, channel: &C) -> Vec<Vec<C>> {
        let state = self.state.read().expect("hot store state read lock");
        let joins = state.joins.get(channel).cloned();
        let installed_joins = state.installed_joins.get(channel).cloned();
        drop(state);

        match joins {
            Some(joins_data) => {
                let mut result = Vec::new();
                if let Some(installed) = installed_joins {
                    result.extend(installed);
                }
                result.extend(joins_data);
                result
            }
            None => {
                let from_history_store = self.get_joins_from_history_store(channel);
                self.state.write().expect("hot store state write lock")
                    .joins
                    .insert(channel.clone(), from_history_store.clone());
                let mut result = Vec::new();
                if let Some(installed) = installed_joins {
                    result.extend(installed);
                }
                result.extend(from_history_store);
                result
            }
        }
    }

    fn put_join(&self, channel: &C, join: &[C]) -> Option<()> {
        let has_existing = self.state.read().expect("hot store state read lock").joins.get(channel).is_some();
        let from_history_store = if has_existing {
            None
        } else {
            Some(self.get_joins_from_history_store(channel))
        };

        let mut state = self.state.write().expect("hot store state write lock");
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
        Some(())
    }

    fn install_join(&self, channel: &C, join: &[C]) -> Option<()> {
        let mut state = self.state.write().expect("hot store state write lock");
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
        Some(())
    }

    fn remove_join(&self, channel: &C, join: &[C]) -> Option<()> {
        let mut state = self.state.write().expect("hot store state write lock");
        let has_join_in_state = state.joins.get(channel).is_some();
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
                    .cloned()
                    .unwrap_or_else(|| self.get_cont_from_history_store(join)),
            );
            conts
        };

        let do_remove = current_continuations.is_empty();

        if !do_remove {
            if !has_join_in_state {
                let joins_in_history_store = self.get_joins_from_history_store(channel);
                state.joins.insert(channel.clone(), joins_in_history_store);
            }
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
        }
    }

    fn changes(&self) -> Vec<HotStoreAction<C, P, A, K>> {
        let cache = self.state.read().expect("hot store state read lock");
        let continuations: Vec<HotStoreAction<C, P, A, K>> = cache
            .continuations
            .iter()
            .map(|(k, v)| {
                if v.is_empty() {
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
                let (k, v) = entry;
                if v.is_empty() {
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
                let (k, v) = entry;
                if v.is_empty() {
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

        [continuations, data, joins].concat()
    }

    fn to_map(&self) -> HashMap<Vec<C>, Row<P, A, K>> {
        let state = self.state.read().expect("hot store state read lock");
        let data = state
            .data
            .iter()
            .map(|entry| {
                let (k, v) = entry;
                (vec![k.clone()], v.clone())
            })
            .collect::<HashMap<_, _>>();

        let all_continuations = {
            let mut all = state
                .continuations
                .iter()
                .map(|entry| {
                    let (k, v) = entry;
                    (k.clone(), v.clone())
                })
                .collect::<HashMap<_, _>>();
            for (k, v) in state.installed_continuations.iter().map(|entry| {
                let (k, v) = entry;
                (k.clone(), v.clone())
            }) {
                all.entry(k).or_insert_with(Vec::new).push(v);
            }
            all
        };

        let mut map = HashMap::new();

        for (k, v) in data.into_iter() {
            let row = Row {
                data: v,
                wks: all_continuations.get(&k).cloned().unwrap_or_else(Vec::new),
            };
            if !(row.data.is_empty() && row.wks.is_empty()) {
                map.insert(k, row);
            }
        }

        map
    }

    fn print(&self) {
        let hot_store_state = self.state.read().expect("hot store state read lock");
        println!("\nHot Store");

        println!("Continuations:");
        for entry in hot_store_state.continuations.iter() {
            let (key, value) = (entry.0, entry.1);
            println!("Key: {:?}, Value: {:?}", key, value);
        }

        println!("\nInstalled Continuations:");
        for entry in hot_store_state.installed_continuations.iter() {
            let (key, value) = (entry.0, entry.1);
            println!("Key: {:?}, Value: {:?}", key, value);
        }

        println!("\nData:");
        for entry in hot_store_state.data.iter() {
            let (key, value) = (entry.0, entry.1);
            println!("Key: {:?}, Value: {:?}", key, value);
        }

        println!("\nJoins:");
        for entry in hot_store_state.joins.iter() {
            let (key, value) = (entry.0, entry.1);
            println!("Key: {:?}, Value: {:?}", key, value);
        }

        println!("\nInstalled Joins:");
        for entry in hot_store_state.installed_joins.iter() {
            let (key, value) = (entry.0, entry.1);
            println!("Key: {:?}, Value: {:?}", key, value);
        }

        let history_cache_state = self.history_cache.read().expect("history cache read lock");
        println!("\nHistory Cache");

        println!("Continuations:");
        for entry in history_cache_state.continuations.iter() {
            let (key, value) = (entry.0, entry.1);
            println!("Key: {:?}, Value: {:?}", key, value);
        }

        println!("\nData:");
        for entry in history_cache_state.datums.iter() {
            let (key, value) = (entry.0, entry.1);
            println!("Key: {:?}, Value: {:?}", key, value);
        }

        println!("\nJoins:");
        for entry in history_cache_state.joins.iter() {
            let (key, value) = (entry.0, entry.1);
            println!("Key: {:?}, Value: {:?}", key, value);
        }

    }

    fn clear(&self) {
        let mut state = self.state.write().expect("hot store state write lock");
        state.continuations.clear();
        state.installed_continuations.clear();
        state.data.clear();
        state.joins.clear();
        state.installed_joins.clear();
        drop(state);

        let mut history_cache = self.history_cache.write().expect("history cache write lock");
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

        Self::update_hot_store_state_metrics(&self.state.read().expect("hot store state read lock"));
    }

    // See rspace/src/test/scala/coop/rchain/rspace/test/package.scala
    fn is_empty(&self) -> bool {
        let store_actions = self.changes();
        let has_insert_actions = store_actions
            .into_iter()
            .any(|action| matches!(action, HotStoreAction::Insert(_)));

        !has_insert_actions
    }

    fn set_state(&self, new_state: HotStoreState<C, P, A, K>) {
        *self.state.write().expect("hot store state write lock for set_state") = new_state;
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
        HOT_STORE_STATE_METRICS_UPDATE_INTERVAL_MS
    }

    fn history_cache_metrics_update_interval_ms() -> u64 {
        HOT_STORE_HISTORY_CACHE_METRICS_UPDATE_INTERVAL_MS
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
            .values()
            .map(|v| v.len())
            .sum();
        let data_items: usize = state.data.values().map(|v| v.len()).sum();
        let joins_items: usize = state.joins.values().map(|v| v.len()).sum();
        let installed_cont_items = state.installed_continuations.len();
        let installed_joins_items: usize = state
            .installed_joins
            .values()
            .map(|v| v.len())
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
            .values()
            .map(|v| v.len())
            .sum();
        let data_items: usize = cache.datums.values().map(|v| v.len()).sum();
        let joins_items: usize = cache.joins.values().map(|v| v.len()).sum();

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

    fn enforce_history_cache_bounds(cache: &mut HistoryStoreCache<C, P, A, K>) {
        let cont_items: usize = cache
            .continuations
            .values()
            .map(|v| v.len())
            .sum();
        let data_items: usize = cache.datums.values().map(|v| v.len()).sum();
        let joins_items: usize = cache.joins.values().map(|v| v.len()).sum();

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
        let mut cache = self.history_cache.write().expect("history cache write lock");
        Self::enforce_history_cache_bounds(&mut cache);
        let channels_vec = channels.to_vec();
        let entry = cache.continuations.entry(channels_vec.clone());
        let result = match entry {
            Entry::Occupied(o) => o.get().clone(),
            Entry::Vacant(v) => {
                let ks = self.history_reader_base.get_continuations(&channels_vec);
                v.insert(ks.clone());
                ks
            }
        };
        Self::update_history_cache_metrics(&cache);
        result
    }

    fn get_data_from_history_store(&self, channel: &C) -> Vec<Datum<A>> {
        let mut cache = self.history_cache.write().expect("history cache write lock");
        Self::enforce_history_cache_bounds(&mut cache);
        let entry = cache.datums.entry(channel.clone());
        let result = match entry {
            Entry::Occupied(o) => o.get().clone(),
            Entry::Vacant(v) => {
                let datums = self.history_reader_base.get_data(channel);
                v.insert(datums.clone());
                datums
            }
        };
        Self::update_history_cache_metrics(&cache);
        result
    }

    fn get_joins_from_history_store(&self, channel: &C) -> Vec<Vec<C>> {
        let mut cache = self.history_cache.write().expect("history cache write lock");
        Self::enforce_history_cache_bounds(&mut cache);
        let entry = cache.joins.entry(channel.clone());
        let result = match entry {
            Entry::Occupied(o) => o.get().clone(),
            Entry::Vacant(v) => {
                let joins = self.history_reader_base.get_joins(&channel);
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
        Box::new(InMemHotStore {
            state: std::sync::RwLock::new(cache),
            history_cache: std::sync::RwLock::new(HistoryStoreCache::default()),
            history_reader_base: history_reader,
        })
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
