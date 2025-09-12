use crate::rspace::history::history_reader::HistoryReaderBase;
use crate::rspace::hot_store_action::{
    DeleteAction, DeleteContinuations, DeleteData, DeleteJoins, HotStoreAction, InsertAction,
    InsertContinuations, InsertData, InsertJoins,
};
use crate::rspace::internal::{Datum, Row, WaitingContinuation};
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use proptest::prelude::*;
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

// See rspace/src/main/scala/coop/rchain/rspace/HotStore.scala
pub trait HotStore<C: Clone + Hash + Eq, P: Clone, A: Clone, K: Clone>: Sync + Send {
    fn get_continuations(&self, channels: Vec<C>) -> Vec<WaitingContinuation<P, K>>;
    fn put_continuation(&self, channels: Vec<C>, wc: WaitingContinuation<P, K>) -> Option<()>;
    fn install_continuation(&self, channels: Vec<C>, wc: WaitingContinuation<P, K>) -> Option<()>;
    fn remove_continuation(&self, channels: Vec<C>, index: i32) -> Option<()>;

    fn get_data(&self, channel: &C) -> Vec<Datum<A>>;
    fn put_datum(&self, channel: C, d: Datum<A>) -> ();
    fn remove_datum(&self, channel: C, index: i32) -> Option<()>;

    fn get_joins(&self, channel: C) -> Vec<Vec<C>>;
    fn put_join(&self, channel: C, join: Vec<C>) -> Option<()>;
    fn install_join(&self, channel: C, join: Vec<C>) -> Option<()>;
    fn remove_join(&self, channel: C, join: Vec<C>) -> Option<()>;

    fn changes(&self) -> Vec<HotStoreAction<C, P, A, K>>;
    fn to_map(&self) -> HashMap<Vec<C>, Row<P, A, K>>;
    fn snapshot(&self) -> HotStoreState<C, P, A, K>;

    fn print(&self) -> ();
    fn clear(&self) -> ();

    // See rspace/src/test/scala/coop/rchain/rspace/test/package.scala
    fn is_empty(&self) -> bool;
}

pub fn new_dashmap<K: std::cmp::Eq + std::hash::Hash, V>() -> DashMap<K, V> {
    DashMap::new()
}

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
    where
        T: Default + Clone,
    {
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

    fn get_continuations(&self, channels: Vec<C>) -> Vec<WaitingContinuation<P, K>> {
        let from_history_store: Vec<WaitingContinuation<P, K>> =
            self.get_cont_from_history_store(&channels);

        let maybe_continuations = {
            let state = self.hot_store_state.lock().unwrap();
            state
                .continuations
                .get(&channels)
                .map(|continuations| continuations.clone())
        };

        let maybe_installed_continuation = {
            let state = self.hot_store_state.lock().unwrap();
            state
                .installed_continuations
                .get(&channels)
                .map(|continuations| continuations.clone())
        };

        match maybe_continuations {
            Some(continuations) => match maybe_installed_continuation {
                Some(installed_continuation) => {
                    let mut result = vec![installed_continuation];
                    result.extend(continuations);
                    result
                }
                None => continuations,
            },
            None => {
                self.hot_store_state
                    .lock()
                    .unwrap()
                    .continuations
                    .insert(channels, from_history_store.clone());
                match maybe_installed_continuation {
                    Some(installed_continuation) => {
                        let mut result = vec![installed_continuation];
                        result.extend(from_history_store);
                        result
                    }
                    None => from_history_store,
                }
            }
        }
    }

    fn put_continuation(&self, channels: Vec<C>, wc: WaitingContinuation<P, K>) -> Option<()> {
        // println!("\nHit put_continuation");

        let from_history_store: Vec<WaitingContinuation<P, K>> =
            self.get_cont_from_history_store(&channels);
        // println!("\nfrom_history_store: {:?}", from_history_store);

        let state = self.hot_store_state.lock().unwrap();
        let current_continuations = state
            .continuations
            .get(&channels)
            .map(|c| c.clone())
            .unwrap_or(from_history_store);
        let new_continuations = vec![wc]
            .into_iter()
            .chain(current_continuations.into_iter())
            .collect();
        state.continuations.insert(channels, new_continuations);
        Some(())
    }

    fn install_continuation(&self, channels: Vec<C>, wc: WaitingContinuation<P, K>) -> Option<()> {
        // println!("hit install_continuation");
        let state = self.hot_store_state.lock().unwrap();
        let _ = state.installed_continuations.insert(channels, wc);

        // println!("installed_continuation result: {:?}", result);
        // println!("to_map: {:?}\n", self.print());

        Some(())
    }

    fn remove_continuation(&self, channels: Vec<C>, index: i32) -> Option<()> {
        let from_history_store: Vec<WaitingContinuation<P, K>> =
            self.get_cont_from_history_store(&channels);

        let state = self.hot_store_state.lock().unwrap();
        let current_continuations = state
            .continuations
            .get(&channels)
            .map(|c| c.clone())
            .unwrap_or(from_history_store);
        let installed_continuation = state.installed_continuations.get(&channels);
        let is_installed = installed_continuation.is_some();

        let removing_installed = is_installed && index == 0;
        let removed_index = if is_installed { index - 1 } else { index };
        // println!("Index: {}", index);
        // println!("Removed Index: {}", removed_index);
        // println!("Current Continuations Length: {}", current_continuations.len());

        let out_of_bounds =
            removed_index < 0 || removed_index as usize >= current_continuations.len();

        if removing_installed {
            state.continuations.insert(channels, current_continuations);
            println!("WARNING: Attempted to remove an installed continuation");
            None
        } else if out_of_bounds {
            state.continuations.insert(channels, current_continuations);
            println!("WARNING: Index {index} out of bounds when removing continuation");
            None
        } else {
            let mut new_continuations = current_continuations;
            new_continuations.remove(removed_index as usize);
            state.continuations.insert(channels, new_continuations);
            Some(())
        }
    }

    // Data

    fn get_data(&self, channel: &C) -> Vec<Datum<A>> {
        let from_history_store: Vec<Datum<A>> = self.get_data_from_history_store(channel);

        // println!("\nfrom_history_store in hot store get_data: {:?}", from_history_store);

        let maybe_data = {
            let state = self.hot_store_state.lock().unwrap();
            state.data.get(channel).map(|data| data.clone())
        };

        match maybe_data {
            Some(data) => data,
            None => {
                self.hot_store_state
                    .lock()
                    .unwrap()
                    .data
                    .insert(channel.clone(), from_history_store.clone());
                from_history_store
            }
        }
    }

    fn put_datum(&self, channel: C, d: Datum<A>) -> () {
        // println!("\nHit put_datum, channel: {:?}, data: {:?}", channel, d);
        // println!("\nHit put_datum, data: {:?}", d);

        let from_history_store: Vec<Datum<A>> = self.get_data_from_history_store(&channel);
        // println!(
        //     "\nfrom_history_store in put_datum: {:?}",
        //     from_history_store
        // );

        let state = self.hot_store_state.lock().unwrap();
        let mut update_data = state
            .data
            .get(&channel)
            .map(|d| d.clone())
            .unwrap_or(from_history_store);
        update_data.insert(0, d);
        let _ = state.data.insert(channel, update_data);
    }

    fn remove_datum(&self, channel: C, index: i32) -> Option<()> {
        let from_history_store: Vec<Datum<A>> = self.get_data_from_history_store(&channel);

        let state = self.hot_store_state.lock().unwrap();
        let current_datums = state
            .data
            .get(&channel)
            .map(|c| c.clone())
            .unwrap_or(from_history_store);
        let out_of_bounds = index as usize >= current_datums.len();

        if out_of_bounds {
            state.data.insert(channel, current_datums);
            println!("WARNING: Index {index} out of bounds when removing datum");
            None
        } else {
            let mut new_datums = current_datums;
            new_datums.remove(index as usize);
            state.data.insert(channel, new_datums);
            Some(())
        }
    }

    // Joins

    fn get_joins(&self, channel: C) -> Vec<Vec<C>> {
        // println!("\nHit get_joins");

        let from_history_store: Vec<Vec<C>> = self.get_joins_from_history_store(&channel);
        // println!(
        //     "\nfrom_history_store in get_joins: {:?}",
        //     from_history_store
        // );

        let maybe_joins = {
            let state = self.hot_store_state.lock().unwrap();
            state.joins.get(&channel).map(|joins| joins.clone())
        };

        match maybe_joins {
            Some(joins) => {
                // println!("Found joins in store");
                let installed_joins = {
                    let state = self.hot_store_state.lock().unwrap();
                    state
                        .installed_joins
                        .get(&channel)
                        .map(|joins| joins.clone())
                        .unwrap_or(Vec::new())
                };

                let mut result = installed_joins;
                result.extend(joins);
                result
            }
            None => {
                // println!("No joins found in store");
                {
                    let state = self.hot_store_state.lock().unwrap();
                    state
                        .joins
                        .insert(channel.clone(), from_history_store.clone());
                }
                // println!("Inserted into store. Returning from history");

                let installed_joins = {
                    let state = self.hot_store_state.lock().unwrap();
                    state
                        .installed_joins
                        .get(&channel)
                        .map(|joins| joins.clone())
                        .unwrap_or(Vec::new())
                };

                let mut result = installed_joins;
                result.extend(from_history_store);
                result
            }
        }
    }

    fn put_join(&self, channel: C, join: Vec<C>) -> Option<()> {
        let from_history_store: Vec<Vec<C>> = self.get_joins_from_history_store(&channel);

        let state = self.hot_store_state.lock().unwrap();
        let current_joins = state
            .joins
            .get(&channel)
            .map(|j| j.clone())
            .unwrap_or(from_history_store);
        if current_joins.contains(&join) {
            Some(())
        } else {
            let new_joins = vec![join]
                .into_iter()
                .chain(current_joins.into_iter())
                .collect();
            state.joins.insert(channel, new_joins);
            Some(())
        }
    }

    fn install_join(&self, channel: C, join: Vec<C>) -> Option<()> {
        // println!("hit install_join");
        let state = self.hot_store_state.lock().unwrap();
        let current_installed_joins = state
            .installed_joins
            .get(&channel)
            .map(|c| c.clone())
            .unwrap_or(Vec::new());
        if !current_installed_joins.contains(&join) {
            let mut new_installed_joins = current_installed_joins;
            new_installed_joins.insert(0, join);
            let _ = state.installed_joins.insert(channel, new_installed_joins);
        }
        Some(())
    }

    fn remove_join(&self, channel: C, join: Vec<C>) -> Option<()> {
        let joins_in_history_store: Vec<Vec<C>> = self.get_joins_from_history_store(&channel);
        let continuations_in_history_store: Vec<WaitingContinuation<P, K>> =
            self.get_cont_from_history_store(&join);

        let state = self.hot_store_state.lock().unwrap();
        let current_joins = state
            .joins
            .get(&channel)
            .map(|j| j.clone())
            .unwrap_or(joins_in_history_store);

        let current_continuations = {
            let mut conts = state
                .installed_continuations
                .get(&join)
                .map(|c| vec![c.clone()])
                .unwrap_or_else(Vec::new);
            conts.extend(
                state
                    .continuations
                    .get(&join)
                    .map(|continuations| continuations.clone())
                    .unwrap_or(continuations_in_history_store),
            );
            conts
        };

        let index = current_joins.iter().position(|x| *x == join);
        let out_of_bounds = index.is_none();

        // Remove join is called when continuation is removed, so it can be called when
        // continuations are present in which case we just want to skip removal.
        let do_remove = current_continuations.is_empty();

        if do_remove {
            if out_of_bounds {
                println!("WARNING: Join not found when removing join");
                state.joins.insert(channel, current_joins);
                Some(())
            } else {
                let mut new_joins = current_joins;
                new_joins.remove(index.unwrap());
                state.joins.insert(channel, new_joins);
                Some(())
            }
        } else {
            state.joins.insert(channel, current_joins);
            Some(())
        }
    }

    fn changes(&self) -> Vec<HotStoreAction<C, P, A, K>> {
        let cache = self.hot_store_state.lock().unwrap();
        let continuations: Vec<HotStoreAction<C, P, A, K>> = cache
            .continuations
            .clone()
            .into_iter()
            .map(|(k, v)| {
                if v.is_empty() {
                    HotStoreAction::Delete(DeleteAction::DeleteContinuations(DeleteContinuations {
                        channels: k,
                    }))
                } else {
                    HotStoreAction::Insert(InsertAction::InsertContinuations(InsertContinuations {
                        channels: k,
                        continuations: v,
                    }))
                }
            })
            .collect();

        let data: Vec<HotStoreAction<C, P, A, K>> = cache
            .data
            .clone()
            .into_iter()
            .map(|(k, v)| {
                if v.is_empty() {
                    HotStoreAction::Delete(DeleteAction::DeleteData(DeleteData { channel: k }))
                } else {
                    HotStoreAction::Insert(InsertAction::InsertData(InsertData {
                        channel: k,
                        data: v,
                    }))
                }
            })
            .collect();

        let joins: Vec<HotStoreAction<C, P, A, K>> = cache
            .joins
            .clone()
            .into_iter()
            .map(|(k, v)| {
                if v.is_empty() {
                    HotStoreAction::Delete(DeleteAction::DeleteJoins(DeleteJoins { channel: k }))
                } else {
                    HotStoreAction::Insert(InsertAction::InsertJoins(InsertJoins {
                        channel: k,
                        joins: v,
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
        // println!("Continuations: {:?}", self.history_reader_base.get_continuations(channels));
    }

    fn clear(&self) {
        let mut state = self.hot_store_state.lock().unwrap();
        state.continuations = DashMap::new();
        state.installed_continuations = DashMap::new();
        state.data = DashMap::new();
        state.joins = DashMap::new();
        state.installed_joins = DashMap::new();
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
    fn get_cont_from_history_store(&self, channels: &Vec<C>) -> Vec<WaitingContinuation<P, K>> {
        let cache = self.history_store_cache.lock().unwrap();
        let entry = cache.continuations.entry(channels.clone());
        match entry {
            Entry::Occupied(o) => o.get().clone(),
            Entry::Vacant(v) => {
                let ks = self.history_reader_base.get_continuations(channels);
                v.insert(ks.clone());
                ks
            }
        }
    }

    fn get_data_from_history_store(&self, channel: &C) -> Vec<Datum<A>> {
        let cache = self.history_store_cache.lock().unwrap();
        let entry = cache.datums.entry(channel.clone());
        match entry {
            Entry::Occupied(o) => o.get().clone(),
            Entry::Vacant(v) => {
                let datums = self.history_reader_base.get_data(channel);
                // println!("\ndatums from history store: {:?}", datums);
                v.insert(datums.clone());
                datums
            }
        }
    }

    fn get_joins_from_history_store(&self, channel: &C) -> Vec<Vec<C>> {
        let cache = self.history_store_cache.lock().unwrap();
        let entry = cache.joins.entry(channel.clone());
        match entry {
            Entry::Occupied(o) => o.get().clone(),
            Entry::Vacant(v) => {
                let joins = self.history_reader_base.get_joins(channel);
                v.insert(joins.clone());
                joins
            }
        }
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
