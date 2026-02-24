use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use proptest::prelude::*;
use rand::{Rng, thread_rng};

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

// See rspace/src/main/scala/coop/rchain/rspace/HotStore.scala
pub trait HotStore<C: Clone + Hash + Eq, P: Clone, A: Clone, K: Clone>: Sync + Send {
    fn get_continuations(&self, channels: &[C]) -> Vec<WaitingContinuation<P, K>>;
    fn put_continuation(&self, channels: &[C], wc: WaitingContinuation<P, K>) -> Option<()>;
    fn install_continuation(&self, channels: &[C], wc: WaitingContinuation<P, K>) -> Option<()>;
    fn remove_continuation(&self, channels: &[C], index: i32) -> Option<()>;

    fn get_data(&self, channel: &C) -> Vec<Datum<A>>;
    fn put_datum(&self, channel: &C, d: Datum<A>) -> ();
    fn remove_datum(&self, channel: &C, index: i32) -> Option<()>;

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
                let from_history_store = self.get_cont_from_history_store(channels);
                self.hot_store_state
                    .lock()
                    .unwrap()
                    .continuations
                    .insert(channels.to_vec(), from_history_store.clone());
                let mut result = Vec::with_capacity(from_history_store.len() + 1);
                result.push(inst);
                result.extend(from_history_store);
                result
            }
            (None, None) => {
                let from_history_store = self.get_cont_from_history_store(channels);
                self.hot_store_state
                    .lock()
                    .unwrap()
                    .continuations
                    .insert(channels.to_vec(), from_history_store.clone());
                from_history_store
            }
        };
        let state = self.hot_store_state.lock().unwrap();
        Self::update_hot_store_state_metrics(&state);
        result
    }

    fn put_continuation(&self, channels: &[C], wc: WaitingContinuation<P, K>) -> Option<()> {
        // println!("\nHit put_continuation");
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
        match state.continuations.entry(channels.to_vec()) {
            Entry::Occupied(mut occupied) => {
                occupied.get_mut().insert(0, wc);
            }
            Entry::Vacant(vacant) => {
                let mut new_continuations = from_history_store.unwrap_or_default();
                new_continuations.insert(0, wc);
                vacant.insert(new_continuations);
            }
        }
        Self::update_hot_store_state_metrics(&state);
        Some(())
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
            println!("WARNING: Attempted to remove an installed continuation");
            None
        } else {
            match state.continuations.entry(channels.to_vec()) {
                Entry::Occupied(mut occupied) => {
                    let len = occupied.get().len();
                    let out_of_bounds = removed_index < 0 || removed_index as usize >= len;
                    if out_of_bounds {
                        println!("WARNING: Index {index} out of bounds when removing continuation");
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
                        println!("WARNING: Index {index} out of bounds when removing continuation");
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

        let result = if let Some(data) = maybe_data {
            data
        } else {
            let data = self.get_data_from_history_store(channel);
            self.hot_store_state
                .lock()
                .unwrap()
                .data
                .insert(channel.clone(), data.clone());
            data
        };
        let state = self.hot_store_state.lock().unwrap();
        Self::update_hot_store_state_metrics(&state);
        result
    }

    fn put_datum(&self, channel: &C, d: Datum<A>) -> () {
        // println!("\nHit put_datum, channel: {:?}, data: {:?}", channel, d);
        // println!("\nHit put_datum, data: {:?}", d);
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

    fn remove_datum(&self, channel: &C, index: i32) -> Option<()> {
        let state = self.hot_store_state.lock().unwrap();
        let result = match state.data.entry(channel.clone()) {
            Entry::Occupied(mut occupied) => {
                let out_of_bounds = index < 0 || index as usize >= occupied.get().len();
                if out_of_bounds {
                    println!("WARNING: Index {index} out of bounds when removing datum");
                    None
                } else {
                    occupied.get_mut().remove(index as usize);
                    Some(())
                }
            }
            Entry::Vacant(vacant) => {
                let mut from_history_store = self.get_data_from_history_store(channel);
                let out_of_bounds = index < 0 || index as usize >= from_history_store.len();
                if out_of_bounds {
                    println!("WARNING: Index {index} out of bounds when removing datum");
                    vacant.insert(from_history_store);
                    None
                } else {
                    from_history_store.remove(index as usize);
                    vacant.insert(from_history_store);
                    Some(())
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
                let from_history_store = self.get_joins_from_history_store(channel);
                self.hot_store_state
                    .lock()
                    .unwrap()
                    .joins
                    .insert(channel.clone(), from_history_store.clone());
                // println!("No joins found in store");
                // println!("Inserted into store. Returning from history");

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
                    .map(|continuations| continuations.clone())
                    .unwrap_or_else(|| self.get_cont_from_history_store(join)),
            );
            conts
        };

        // Remove join is called when continuation is removed, so it can be called when
        // continuations are present in which case we just want to skip removal.
        let do_remove = current_continuations.is_empty();

        let result = if !do_remove {
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
                        println!("WARNING: Join not found when removing join");
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
                        println!("WARNING: Join not found when removing join");
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
        let cache = self.hot_store_state.lock().unwrap();
        let continuations: Vec<HotStoreAction<C, P, A, K>> = cache
            .continuations
            .iter()
            .map(|entry| {
                let (k, v) = entry.pair();
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
                let (k, v) = entry.pair();
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
                let (k, v) = entry.pair();
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
}

impl<C, P, A, K> InMemHotStore<C, P, A, K>
where
    C: Clone + Debug + Hash + Eq + Sync + Send,
    P: Clone + Debug + Sync + Send,
    A: Clone + Debug + Sync + Send,
    K: Clone + Debug + Sync + Send,
{
    fn update_hot_store_state_metrics(state: &HotStoreState<C, P, A, K>) {
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
        let cache = self.history_store_cache.lock().unwrap();
        Self::enforce_history_cache_bounds(&cache);
        let entry = cache.datums.entry(channel.clone());
        let result = match entry {
            Entry::Occupied(o) => o.get().clone(),
            Entry::Vacant(v) => {
                let datums = self.history_reader_base.get_data(channel);
                // println!("\ndatums from history store: {:?}", datums);
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
