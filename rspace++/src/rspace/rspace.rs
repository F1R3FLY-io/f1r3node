// See rspace/src/main/scala/coop/rchain/rspace/RSpace.scala

use super::checkpoint::SoftCheckpoint;
use super::errors::HistoryRepositoryError;
use super::errors::RSpaceError;
use super::hashing::blake2b256_hash::Blake2b256Hash;
use super::history::history_reader::HistoryReader;
use super::history::instances::radix_history::RadixHistory;
use super::r#match::Match;
use super::replay_rspace::ReplayRSpace;
use super::logging::BasicLogger;
use super::rspace_interface::CONSUME_COMM_LABEL;
use super::rspace_interface::ContResult;
use super::rspace_interface::ISpace;
use super::rspace_interface::MaybeConsumeResult;
use super::rspace_interface::MaybeProduceCandidate;
use super::rspace_interface::MaybeProduceResult;
use super::rspace_interface::PRODUCE_COMM_LABEL;
use super::rspace_interface::RSpaceResult;
use super::trace::Log;
use super::trace::event::COMM;
use super::trace::event::Consume;
use super::trace::event::Event;
use super::trace::event::IOEvent;
use super::trace::event::Produce;
use crate::rspace::checkpoint::Checkpoint;
use crate::rspace::history::history_repository::HistoryRepository;
use crate::rspace::history::history_repository::HistoryRepositoryInstances;
use crate::rspace::hot_store::{HotStore, HotStoreInstances};
use crate::rspace::internal::*;
use crate::rspace::space_matcher::SpaceMatcher;
use dashmap::DashMap;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use shared::rust::store::key_value_store::KeyValueStore;
use std::collections::BTreeMap;
use std::collections::{BTreeSet, HashMap};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct RSpaceStore {
    pub history: Arc<Mutex<Box<dyn KeyValueStore>>>,
    pub roots: Arc<Mutex<Box<dyn KeyValueStore>>>,
    pub cold: Arc<Mutex<Box<dyn KeyValueStore>>>,
}

#[repr(C)]
#[derive(Clone)]
pub struct RSpace<C, P, A, K> {
    pub history_repository: Arc<Box<dyn HistoryRepository<C, P, A, K>>>,
    pub store: Arc<Box<dyn HotStore<C, P, A, K>>>,
    installs: Arc<Mutex<HashMap<Vec<C>, Install<P, K>>>>,
    event_log: Log,
    produce_counter: BTreeMap<Produce, i32>,
    matcher: Arc<Box<dyn Match<P, A>>>,
}

impl<C, P, A, K> SpaceMatcher<C, P, A, K> for RSpace<C, P, A, K>
where
    C: Clone + Debug + Default + Serialize + std::hash::Hash + Ord + Eq + 'static + Sync + Send,
    P: Clone + Debug + Default + Serialize + 'static + Sync + Send,
    A: Clone + Debug + Default + Serialize + 'static + Sync + Send,
    K: Clone + Debug + Default + Serialize + 'static + Sync + Send,
{
}

impl<C, P, A, K> ISpace<C, P, A, K> for RSpace<C, P, A, K>
where
    C: Clone + Debug + Default + Serialize + std::hash::Hash + Ord + Eq + 'static + Sync + Send,
    P: Clone + Debug + Default + Serialize + 'static + Sync + Send,
    A: Clone + Debug + Default + Serialize + 'static + Sync + Send,
    K: Clone + Debug + Default + Serialize + 'static + Sync + Send,
{
    fn create_checkpoint(&mut self) -> Result<Checkpoint, RSpaceError> {
        // println!("\nhit rspace++ create_checkpoint");
        // println!("\nspace in create_checkpoint: {:?}", self.store.to_map().len());
        let changes = self.store.changes();
        let next_history = self.history_repository.checkpoint(&changes);
        self.history_repository = Arc::new(next_history);

        let log = self.event_log.clone();
        self.event_log = Vec::new();
        self.produce_counter = BTreeMap::new();

        let history_reader = self
            .history_repository
            .get_history_reader(&self.history_repository.root())?;

        self.create_new_hot_store(history_reader);
        self.restore_installs();

        // println!("\nspace after create_checkpoint: {:?}", self.store.to_map().len());

        Ok(Checkpoint {
            root: self.history_repository.root(),
            log,
        })
    }

    fn reset(&mut self, root: &Blake2b256Hash) -> Result<(), RSpaceError> {
        // println!("\nhit rspace++ reset, root: {:?}", root);
        let next_history = self.history_repository.reset(root)?;
        self.history_repository = Arc::new(next_history);

        self.event_log = Vec::new();
        self.produce_counter = BTreeMap::new();

        let history_reader = self.history_repository.get_history_reader(root)?;
        self.create_new_hot_store(history_reader);
        self.restore_installs();

        Ok(())
    }

    fn consume_result(
        &mut self,
        _channel: Vec<C>,
        _pattern: Vec<P>,
    ) -> Result<Option<(K, Vec<A>)>, RSpaceError> {
        panic!("\nERROR: RSpace consume_result should not be called here");
    }

    fn get_data(&self, channel: &C) -> Vec<Datum<A>> {
        self.store.get_data(channel)
    }

    fn get_waiting_continuations(&self, channels: Vec<C>) -> Vec<WaitingContinuation<P, K>> {
        self.store.get_continuations(channels)
    }

    fn get_joins(&self, channel: C) -> Vec<Vec<C>> {
        self.store.get_joins(channel)
    }

    fn clear(&mut self) -> Result<(), RSpaceError> {
        self.reset(&RadixHistory::empty_root_node_hash())
    }

    fn to_map(&self) -> HashMap<Vec<C>, Row<P, A, K>> {
        self.store.to_map()
    }

    fn create_soft_checkpoint(&mut self) -> SoftCheckpoint<C, P, A, K> {
        // println!("\nhit rspace++ create_soft_checkpoint");
        // println!("current hot_store state: {:?}", self.store.snapshot());

        let cache_snapshot = self.store.snapshot();
        let curr_event_log = self.event_log.clone();
        let curr_produce_counter = self.produce_counter.clone();

        self.event_log = Vec::new();
        self.produce_counter = BTreeMap::new();

        SoftCheckpoint {
            cache_snapshot,
            log: curr_event_log,
            produce_counter: curr_produce_counter,
        }
    }

    fn revert_to_soft_checkpoint(
        &mut self,
        checkpoint: SoftCheckpoint<C, P, A, K>,
    ) -> Result<(), RSpaceError> {
        let history = &self.history_repository;
        let history_reader = history.get_history_reader(&history.root())?;
        let hot_store = HotStoreInstances::create_from_mhs_and_hr(
            Arc::new(Mutex::new(checkpoint.cache_snapshot)),
            history_reader.base(),
        );

        self.store = Arc::new(hot_store);
        self.event_log = checkpoint.log;
        self.produce_counter = checkpoint.produce_counter;

        Ok(())
    }

    fn consume(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<MaybeConsumeResult<C, P, A, K>, RSpaceError> {
        // println!("\nrspace consume");
        // println!("channels: {:?}", channels);
        // println!("space in consume before: {:?}", self.store.to_map().len());

        if channels.is_empty() {
            panic!("RUST ERROR: channels can't be empty");
        } else if channels.len() != patterns.len() {
            panic!("RUST ERROR: channels.length must equal patterns.length");
        } else {
            let consume_ref = Consume::create(&channels, &patterns, &continuation, persist);

            let result =
                self.locked_consume(channels, patterns, continuation, persist, peeks, consume_ref);
            // println!("locked_consume result: {:?}", result);
            // println!("\nspace in consume after: {:?}", self.store.to_map().len());
            result
        }
    }

    fn produce(
        &mut self,
        channel: C,
        data: A,
        persist: bool,
    ) -> Result<MaybeProduceResult<C, P, A, K>, RSpaceError> {
        // println!("\nrspace produce");
        // println!("space in produce: {:?}", self.store.to_map().len());
        // println!("\nHit produce, data: {:?}", data);
        // println!("\n\nHit produce, channel: {:?}", channel);

        let produce_ref = Produce::create(&channel, &data, persist);
        let result = self.locked_produce(channel, data, persist, produce_ref);
        // println!("\nlocked_produce result: {:?}", result);
        // println!("\nspace in produce: {:?}", self.store.to_map().len());
        result
    }

    fn install(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
    ) -> Result<Option<(K, Vec<A>)>, RSpaceError> {
        self.locked_install(channels, patterns, continuation)
    }

    fn rig_and_reset(&mut self, _start_root: Blake2b256Hash, _log: Log) -> Result<(), RSpaceError> {
        panic!("\nERROR: RSpace rig_and_reset should not be called here");
    }

    fn rig(&self, _log: Log) -> Result<(), RSpaceError> {
        panic!("\nERROR: RSpace rig should not be called here");
    }

    fn check_replay_data(&self) -> Result<(), RSpaceError> {
        panic!("\nERROR: RSpace check_replay_data should not be called here");
    }

    fn is_replay(&self) -> bool {
        false
    }

    fn update_produce(&mut self, produce_ref: Produce) -> () {
        for event in self.event_log.iter_mut() {
            match event {
                Event::IoEvent(IOEvent::Produce(produce)) => {
                    if produce.hash == produce_ref.hash {
                        *produce = produce_ref.clone();
                    }
                }

                Event::Comm(comm) => {
                    let COMM {
                        produces: _produces,
                        times_repeated: _times_repeated,
                        ..
                    } = comm;

                    let updated_comm = COMM {
                        produces: _produces
                            .iter()
                            .map(|p| {
                                if p.hash == produce_ref.hash {
                                    produce_ref.clone()
                                } else {
                                    p.clone()
                                }
                            })
                            .collect(),
                        times_repeated: _times_repeated
                            .iter()
                            .map(|(k, v)| {
                                if k.hash == produce_ref.hash {
                                    (produce_ref.clone(), v.clone())
                                } else {
                                    (k.clone(), v.clone())
                                }
                            })
                            .collect(),
                        ..comm.clone()
                    };

                    *comm = updated_comm;
                }

                _ => continue,
            }
        }
    }
}

impl<C, P, A, K> RSpace<C, P, A, K>
where
    C: Clone + Debug + Default + Serialize + Hash + Ord + Eq + 'static + Sync + Send,
    P: Clone + Debug + Default + Serialize + 'static + Sync + Send,
    A: Clone + Debug + Default + Serialize + 'static + Sync + Send,
    K: Clone + Debug + Default + Serialize + 'static + Sync + Send,
{
    /**
     * Creates [[RSpace]] from [[HistoryRepository]] and [[HotStore]].
     */
    pub fn apply(
        history_repository: Arc<Box<dyn HistoryRepository<C, P, A, K>>>,
        store: Box<dyn HotStore<C, P, A, K>>,
        matcher: Arc<Box<dyn Match<P, A>>>,
    ) -> RSpace<C, P, A, K>
    where
        C: Clone + Debug + Ord + Hash,
        P: Clone + Debug,
        A: Clone + Debug,
        K: Clone + Debug,
    {
        RSpace {
            history_repository,
            store: Arc::new(store),
            matcher,
            installs: Arc::new(Mutex::new(HashMap::new())),
            event_log: Vec::new(),
            produce_counter: BTreeMap::new(),
        }
    }

    pub fn create(
        store: RSpaceStore,
        matcher: Arc<Box<dyn Match<P, A>>>,
    ) -> Result<RSpace<C, P, A, K>, HistoryRepositoryError>
    where
        C: Clone
            + Debug
            + Default
            + Send
            + Sync
            + Serialize
            + Ord
            + Hash
            + for<'a> Deserialize<'a>
            + 'static,
        P: Clone + Debug + Default + Send + Sync + Serialize + for<'a> Deserialize<'a> + 'static,
        A: Clone + Debug + Default + Send + Sync + Serialize + for<'a> Deserialize<'a> + 'static,
        K: Clone + Debug + Default + Send + Sync + Serialize + for<'a> Deserialize<'a> + 'static,
    {
        let setup = Self::create_history_repo(store).unwrap();
        let (history_reader, store) = setup;
        let space = Self::apply(Arc::new(history_reader), store, matcher);
        Ok(space)
    }

    pub fn create_with_replay(
        store: RSpaceStore,
        matcher: Arc<Box<dyn Match<P, A>>>,
    ) -> Result<(RSpace<C, P, A, K>, ReplayRSpace<C, P, A, K>), HistoryRepositoryError>
    where
        C: Clone
            + Debug
            + Default
            + Send
            + Sync
            + Serialize
            + Ord
            + Hash
            + for<'a> Deserialize<'a>
            + 'static,
        P: Clone + Debug + Default + Send + Sync + Serialize + for<'a> Deserialize<'a> + 'static,
        A: Clone + Debug + Default + Send + Sync + Serialize + for<'a> Deserialize<'a> + 'static,
        K: Clone + Debug + Default + Send + Sync + Serialize + for<'a> Deserialize<'a> + 'static,
    {
        let setup = Self::create_history_repo(store).unwrap();
        let (history_repo, store) = setup;
        let history_repo_arc = Arc::new(history_repo);
        // Play
        let space = Self::apply(history_repo_arc.clone(), store, matcher.clone());
        // Replay
        let history_reader: Box<dyn HistoryReader<Blake2b256Hash, C, P, A, K>> =
            history_repo_arc.get_history_reader(&history_repo_arc.root())?;
        let replay_store = HotStoreInstances::create_from_hr(history_reader.base());
        let replay = ReplayRSpace::apply_with_logger(
            history_repo_arc.clone(),
            Arc::new(replay_store),
            matcher.clone(),
            Box::new(BasicLogger::new()),
        );
        Ok((space, replay))
    }

    /**
     * Creates [[HistoryRepository]] and [[HotStore]].
     */
    pub fn create_history_repo(
        store: RSpaceStore,
    ) -> Result<
        (Box<dyn HistoryRepository<C, P, A, K>>, Box<dyn HotStore<C, P, A, K>>),
        HistoryRepositoryError,
    >
    where
        C: Clone
            + Debug
            + Default
            + Send
            + Sync
            + Serialize
            + for<'a> Deserialize<'a>
            + Eq
            + Hash
            + 'static,
        P: Clone + Debug + Default + Send + Sync + Serialize + for<'a> Deserialize<'a> + 'static,
        A: Clone + Debug + Default + Send + Sync + Serialize + for<'a> Deserialize<'a> + 'static,
        K: Clone + Debug + Default + Send + Sync + Serialize + for<'a> Deserialize<'a> + 'static,
    {
        let history_repo =
            HistoryRepositoryInstances::lmdb_repository(store.history, store.roots, store.cold)?;

        let history_reader = history_repo.get_history_reader(&history_repo.root())?;

        let hot_store = HotStoreInstances::create_from_hr(history_reader.base());

        Ok((history_repo, hot_store))
    }

    fn produce_counters(&self, produce_refs: Vec<Produce>) -> BTreeMap<Produce, i32> {
        produce_refs
            .into_iter()
            .map(|p| (p.clone(), self.produce_counter.get(&p).unwrap_or(&0).clone()))
            .collect()
    }

    fn locked_consume(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
        consume_ref: Consume,
    ) -> Result<MaybeConsumeResult<C, P, A, K>, RSpaceError> {
        // println!("\nHit locked_consume");
        // println!(
        //     "consume: searching for data matching <patterns: {:?}> at <channels: {:?}>",
        //     patterns, channels
        // );

        self.log_consume(consume_ref.clone(), &channels, &patterns, &continuation, persist, &peeks);

        let channel_to_indexed_data = self.fetch_channel_to_index_data(&channels);
        // println!("\nchannel_to_indexed_data: {:?}", channel_to_indexed_data);
        let zipped: Vec<(C, P)> = channels
            .iter()
            .cloned()
            .zip(patterns.iter().cloned())
            .collect();
        let options: Option<Vec<ConsumeCandidate<C, A>>> = self
            .extract_data_candidates(&self.matcher, zipped, channel_to_indexed_data, Vec::new())
            .into_iter()
            .collect();

        // println!("options: {:?}", options);

        let wk = WaitingContinuation {
            patterns,
            continuation,
            persist,
            peeks: peeks.clone(),
            source: consume_ref.clone(),
        };

        match options {
            Some(data_candidates) => {
                let produce_counters_closure =
                    |produces: Vec<Produce>| self.produce_counters(produces);

                self.log_comm(
                    &data_candidates,
                    &channels,
                    wk.clone(),
                    COMM::new(
                        data_candidates.clone(),
                        consume_ref.clone(),
                        peeks.clone(),
                        produce_counters_closure,
                    ),
                    CONSUME_COMM_LABEL,
                );
                self.store_persistent_data(data_candidates.clone(), &peeks);
                // println!(
                //     "consume: data found for <patterns: {:?}> at <channels: {:?}>",
                //     patterns, channels
                // );
                Ok(self.wrap_result(channels, wk, consume_ref, data_candidates))
            }
            None => {
                self.store_waiting_continuation(channels, wk);
                Ok(None)
            }
        }
    }

    /*
     * Here, we create a cache of the data at each channel as `channelToIndexedData`
     * which is used for finding matches.  When a speculative match is found, we can
     * remove the matching datum from the remaining data candidates in the cache.
     *
     * Put another way, this allows us to speculatively remove matching data without
     * affecting the actual store contents.
     */
    fn fetch_channel_to_index_data(&self, channels: &Vec<C>) -> DashMap<C, Vec<(Datum<A>, i32)>> {
        let map = DashMap::new();
        for c in channels {
            let data = self.store.get_data(c);
            let shuffled_data = self.shuffle_with_index(data);
            map.insert(c.clone(), shuffled_data);
        }
        map
    }

    fn locked_produce(
        &mut self,
        channel: C,
        data: A,
        persist: bool,
        produce_ref: Produce,
    ) -> Result<MaybeProduceResult<C, P, A, K>, RSpaceError> {
        // println!("\nHit locked_produce");
        let grouped_channels = self.store.get_joins(channel.clone());
        // println!("\ngrouped_channels: {:?}", grouped_channels);
        // println!(
        //     "produce: searching for matching continuations at <grouped_channels: {:?}>",
        //     grouped_channels
        // );
        let _ = self.log_produce(produce_ref.clone(), &channel, &data, persist);
        let extracted = self.extract_produce_candidate(
            grouped_channels,
            channel.clone(),
            Datum {
                a: data.clone(),
                persist,
                source: produce_ref.clone(),
            },
        );

        // println!("extracted in lockedProduce: {:?}", extracted);

        match extracted {
            Some(produce_candidate) => Ok(self
                .process_match_found(produce_candidate)
                .map(|consume_result| (consume_result.0, consume_result.1, produce_ref))),
            None => Ok(self.store_data(channel, data, persist, produce_ref)),
        }
    }

    /*
     * Find produce candidate
     *
     * NOTE: On Rust side, we are NOT passing functions through. Instead just the data.
     * And then in 'run_matcher_for_channels' we call the functions defined below
     */
    fn extract_produce_candidate(
        &self,
        grouped_channels: Vec<Vec<C>>,
        bat_channel: C,
        data: Datum<A>,
    ) -> MaybeProduceCandidate<C, P, A, K> {
        // println!("\nHit extract_produce_candidate");

        let fetch_matching_continuations =
            |channels: Vec<C>| -> Vec<(WaitingContinuation<P, K>, i32)> {
                let continuations = self.store.get_continuations(channels);
                self.shuffle_with_index(continuations)
            };

        /*
         * Here, we create a cache of the data at each channel as `channelToIndexedData`
         * which is used for finding matches.  When a speculative match is found, we can
         * remove the matching datum from the remaining data candidates in the cache.
         *
         * Put another way, this allows us to speculatively remove matching data without
         * affecting the actual store contents.
         *
         * In this version, we also add the produced data directly to this cache.
         */
        let fetch_matching_data = |channel| -> (C, Vec<(Datum<A>, i32)>) {
            let data_vec = self.store.get_data(&channel);
            let mut shuffled_data = self.shuffle_with_index(data_vec);
            if channel == bat_channel {
                shuffled_data.insert(0, (data.clone(), -1));
            }
            (channel, shuffled_data)
        };

        self.run_matcher_for_channels(
            grouped_channels,
            fetch_matching_continuations,
            fetch_matching_data,
        )
    }

    fn process_match_found(
        &mut self,
        pc: ProduceCandidate<C, P, A, K>,
    ) -> MaybeConsumeResult<C, P, A, K> {
        let ProduceCandidate {
            channels,
            continuation,
            continuation_index,
            data_candidates,
        } = pc;

        let WaitingContinuation {
            patterns: _patterns,
            continuation: _cont,
            persist,
            peeks,
            source: consume_ref,
        } = &continuation;

        let produce_counters_closure = |produces: Vec<Produce>| self.produce_counters(produces);
        self.log_comm(
            &data_candidates,
            &channels,
            continuation.clone(),
            COMM::new(
                data_candidates.clone(),
                consume_ref.clone(),
                peeks.clone(),
                produce_counters_closure,
            ),
            PRODUCE_COMM_LABEL,
        );

        if !persist {
            self.store
                .remove_continuation(channels.clone(), continuation_index);
        }

        self.remove_matched_datum_and_join(channels.clone(), data_candidates.clone());

        // println!(
        //     "produce: matching continuation found at <channels: {:?}>",
        //     channels
        // );

        self.wrap_result(channels, continuation.clone(), consume_ref.clone(), data_candidates)
    }

    fn log_comm(
        &mut self,
        _data_candidates: &Vec<ConsumeCandidate<C, A>>,
        _channels: &Vec<C>,
        _wk: WaitingContinuation<P, K>,
        comm: COMM,
        _label: &str,
    ) -> COMM {
        self.event_log.insert(0, Event::Comm(comm.clone()));
        comm
    }

    fn log_consume(
        &mut self,
        consume_ref: Consume,
        _channels: &Vec<C>,
        _patterns: &Vec<P>,
        _continuation: &K,
        _persist: bool,
        _peeks: &BTreeSet<i32>,
    ) -> Consume {
        self.event_log
            .insert(0, Event::IoEvent(IOEvent::Consume(consume_ref.clone())));

        consume_ref
    }

    fn log_produce(
        &mut self,
        produce_ref: Produce,
        _channel: &C,
        _data: &A,
        persist: bool,
    ) -> Produce {
        self.event_log
            .insert(0, Event::IoEvent(IOEvent::Produce(produce_ref.clone())));
        if !persist {
            // let entry = self.produce_counter.entry(produce_ref.clone()).or_insert(0);
            // *entry += 1;
            match self.produce_counter.get(&produce_ref) {
                Some(current_count) => self
                    .produce_counter
                    .insert(produce_ref.clone(), current_count + 1),
                None => self.produce_counter.insert(produce_ref.clone(), 1),
            };
        }

        produce_ref
    }

    pub fn spawn(&self) -> Result<Self, RSpaceError> {
        let history_repo = &self.history_repository;
        let next_history = history_repo.reset(&history_repo.root())?;
        let history_reader = next_history.get_history_reader(&next_history.root())?;
        let hot_store = HotStoreInstances::create_from_hr(history_reader.base());
        let mut rspace = RSpace::apply(Arc::new(next_history), hot_store, self.matcher.clone());
        rspace.restore_installs();

        // println!("\nRSpace Store in spawn: ");
        // rspace.store.print().await;

        // println!("\nRSpace History Store in spawn: ");
        // rspace.history_repository.

        Ok(rspace)
    }

    /* RSpaceOps */

    fn store_waiting_continuation(
        &self,
        channels: Vec<C>,
        wc: WaitingContinuation<P, K>,
    ) -> MaybeConsumeResult<C, P, A, K> {
        // println!("\nHit store_waiting_continuation");
        self.store.put_continuation(channels.clone(), wc);
        for channel in channels.iter() {
            self.store.put_join(channel.clone(), channels.clone());
            // println!("consume: no data found, storing <(patterns, continuation): ({:?}, {:?})> at <channels: {:?}>", wc.patterns, wc.continuation, channels)
        }
        None
    }

    fn store_data(
        &self,
        channel: C,
        data: A,
        persist: bool,
        produce_ref: Produce,
    ) -> MaybeProduceResult<C, P, A, K> {
        // println!("\nHit store_data");
        // println!("\nHit store_data, data: {:?}", data);
        self.store.put_datum(
            channel,
            Datum {
                a: data,
                persist,
                source: produce_ref,
            },
        );
        // println!(
        //     "produce: persisted <data: {:?}> at <channel: {:?}>",
        //     data, channel
        // );

        None
    }

    fn store_persistent_data(
        &self,
        mut data_candidates: Vec<ConsumeCandidate<C, A>>,
        _peeks: &BTreeSet<i32>,
    ) -> Option<Vec<()>> {
        data_candidates.sort_by(|a, b| b.datum_index.cmp(&a.datum_index));
        let results: Vec<_> = data_candidates
            .into_iter()
            .rev()
            .map(|consume_candidate| {
                let ConsumeCandidate {
                    channel,
                    datum: Datum { persist, .. },
                    removed_datum: _,
                    datum_index,
                } = consume_candidate;

                if !persist {
                    self.store.remove_datum(channel, datum_index)
                } else {
                    Some(())
                }
            })
            .collect();

        if results.iter().any(|res| res.is_none()) {
            None
        } else {
            Some(results.into_iter().filter_map(|x| x).collect())
        }
    }

    fn restore_installs(&mut self) -> () {
        let installs = self.installs.lock().unwrap().clone();
        // println!("\ninstalls: {:?}", installs);
        for (channels, install) in installs {
            self.install(channels, install.patterns, install.continuation)
                .unwrap();
        }
    }

    fn locked_install(
        &mut self,
        // &self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
    ) -> Result<Option<(K, Vec<A>)>, RSpaceError> {
        // println!("\nhit locked_install");
        // println!("channels: {:?}", channels);
        // println!("patterns: {:?}", patterns);
        // println!("continuation: {:?}", continuation);
        if channels.len() != patterns.len() {
            panic!("RUST ERROR: channels.length must equal patterns.length");
        } else {
            // println!(
            //     "install: searching for data matching <patterns: {:?}> at <channels: {:?}>",
            //     patterns, channels
            // );

            let consume_ref = Consume::create(&channels, &patterns, &continuation, true);
            let channel_to_indexed_data = self.fetch_channel_to_index_data(&channels);
            // println!("channel_to_indexed_data in locked_install: {:?}", channel_to_indexed_data);
            let zipped: Vec<(C, P)> = channels
                .iter()
                .cloned()
                .zip(patterns.iter().cloned())
                .collect();
            let options: Option<Vec<ConsumeCandidate<C, A>>> = self
                .extract_data_candidates(&self.matcher, zipped, channel_to_indexed_data, Vec::new())
                .into_iter()
                .collect();

            // println!("options in locked_install: {:?}", options);

            match options {
                None => {
                    self.installs.lock().unwrap().insert(
                        channels.clone(),
                        Install {
                            patterns: patterns.clone(),
                            continuation: continuation.clone(),
                        },
                    );

                    self.store.install_continuation(
                        channels.clone(),
                        WaitingContinuation {
                            patterns,
                            continuation,
                            persist: true,
                            peeks: BTreeSet::default(),
                            source: consume_ref,
                        },
                    );

                    for channel in channels.iter() {
                        self.store.install_join(channel.clone(), channels.clone());
                    }
                    // println!(
                    //     "storing <(patterns, continuation): ({:?}, {:?})> at <channels: {:?}>",
                    //     patterns, continuation, channels
                    // );
                    // println!("store length after install: {:?}\n", self.store.to_map().len());
                    Ok(None)
                }
                Some(_) => Err(RSpaceError::BugFoundError(
                    "RUST ERROR: Installing can be done only on startup".to_string(),
                )),
            }
        }
    }

    fn create_new_hot_store(
        &mut self,
        history_reader: Box<dyn HistoryReader<Blake2b256Hash, C, P, A, K>>,
    ) -> () {
        let next_hot_store = HotStoreInstances::create_from_hr(history_reader.base());
        self.store = Arc::new(next_hot_store);
    }

    fn wrap_result(
        &self,
        channels: Vec<C>,
        wk: WaitingContinuation<P, K>,
        _consume_ref: Consume,
        data_candidates: Vec<ConsumeCandidate<C, A>>,
    ) -> MaybeConsumeResult<C, P, A, K> {
        // println!("\nhit wrap_result");

        let cont_result = ContResult {
            continuation: wk.continuation,
            persistent: wk.persist,
            channels,
            patterns: wk.patterns,
            peek: !wk.peeks.is_empty(),
        };

        let rspace_results = data_candidates
            .into_iter()
            .map(|data_candidate| RSpaceResult {
                channel: data_candidate.channel,
                matched_datum: data_candidate.datum.a,
                removed_datum: data_candidate.removed_datum,
                persistent: data_candidate.datum.persist,
            })
            .collect();

        Some((cont_result, rspace_results))
    }

    fn remove_matched_datum_and_join(
        &self,
        channels: Vec<C>,
        mut data_candidates: Vec<ConsumeCandidate<C, A>>,
    ) -> Option<Vec<()>> {
        data_candidates.sort_by(|a, b| b.datum_index.cmp(&a.datum_index));
        let results: Vec<_> = data_candidates
            .into_iter()
            .rev()
            .map(|consume_candidate| {
                let ConsumeCandidate {
                    channel,
                    datum: Datum { persist, .. },
                    removed_datum: _,
                    datum_index,
                } = consume_candidate;

                let channels_clone = channels.clone();
                if datum_index >= 0 && !persist {
                    self.store.remove_datum(channel.clone(), datum_index);
                }
                self.store.remove_join(channel, channels_clone);

                Some(())
            })
            .collect();

        if results.iter().any(|res| res.is_none()) {
            None
        } else {
            Some(results.into_iter().filter_map(|x| x).collect())
        }
    }

    fn run_matcher_for_channels(
        &self,
        grouped_channels: Vec<Vec<C>>,
        fetch_matching_continuations: impl Fn(Vec<C>) -> Vec<(WaitingContinuation<P, K>, i32)>,
        fetch_matching_data: impl Fn(C) -> (C, Vec<(Datum<A>, i32)>),
    ) -> MaybeProduceCandidate<C, P, A, K> {
        let mut remaining = grouped_channels;

        loop {
            match remaining.split_first() {
                Some((channels, rest)) => {
                    let match_candidates = fetch_matching_continuations(channels.to_vec());
                    // println!("match_candidates: {:?}", match_candidates);
                    let fetch_data: Vec<_> = channels
                        .iter()
                        .map(|c| fetch_matching_data(c.clone()))
                        .collect();

                    let channel_to_indexed_data_list: Vec<(C, Vec<(Datum<A>, i32)>)> =
                        fetch_data.into_iter().filter_map(|x| Some(x)).collect();
                    // println!("channel_to_indexed_data_list: {:?}", channel_to_indexed_data_list);

                    let first_match = self.extract_first_match(
                        &self.matcher,
                        channels.to_vec(),
                        match_candidates,
                        channel_to_indexed_data_list.into_iter().collect(),
                    );

                    // println!("first_match in run_matcher_for_channels: {:?}", first_match);

                    match first_match {
                        Some(produce_candidate) => return Some(produce_candidate),
                        None => remaining = rest.to_vec(),
                    }
                }
                None => {
                    // println!("returning none in in run_matcher_for_channels");
                    return None;
                }
            }
        }
    }

    fn shuffle_with_index<D>(&self, t: Vec<D>) -> Vec<(D, i32)> {
        let mut rng = thread_rng();
        let mut indexed_vec = t
            .into_iter()
            .enumerate()
            .map(|(i, d)| (d, i as i32))
            .collect::<Vec<_>>();
        indexed_vec.shuffle(&mut rng);
        indexed_vec
    }
}
