// See /home/spreston/src/firefly/f1r3fly/rspace/src/main/scala/coop/rchain/
// rspace/ReplayRSpace.scala

// NOTE: Manual marks are used instead of trace_i() because async functions
// are not compatible with the current Span trait's FnOnce closure pattern.
// This matches Scala's Span[F].traceI() semantics for async operations.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Instant;
use std::sync::{Arc, Mutex};

use dashmap::DashMap;
use serde::Serialize;
use tracing::{Level, event};

use super::checkpoint::SoftCheckpoint;
use super::errors::RSpaceError;
use super::hashing::blake2b256_hash::Blake2b256Hash;
use super::history::history_reader::HistoryReader;
use super::history::instances::radix_history::RadixHistory;
use super::logging::{BasicLogger, RSpaceLogger};
use super::r#match::Match;
use super::metrics_constants::{
    CONSUME_COMM_LABEL, PRODUCE_COMM_LABEL, REPLAY_RSPACE_METRICS_SOURCE,
    REPLAY_WAITING_CONTINUATIONS_CHANNEL_DEPTH_METRIC,
    REPLAY_WAITING_CONTINUATIONS_ESTIMATE_METRIC,
    REPLAY_WAITING_CONTINUATIONS_MATCHED_TOTAL_METRIC,
    REPLAY_WAITING_CONTINUATIONS_STORED_TOTAL_METRIC,
};
use super::rspace_interface::{
    ContResult, ISpace, MaybeConsumeResult, MaybeProduceCandidate, MaybeProduceResult, RSpaceResult,
};
use super::trace::Log;
use super::trace::event::{COMM, Consume, Event, IOEvent, Produce};
use crate::rspace::checkpoint::Checkpoint;
use crate::rspace::history::history_repository::HistoryRepository;
use crate::rspace::hot_store::{HotStore, HotStoreInstances};
use crate::rspace::internal::*;
use crate::rspace::space_matcher::SpaceMatcher;

#[repr(C)]
#[derive(Clone)]
pub struct ReplayRSpace<C, P, A, K> {
    pub history_repository: Arc<std::sync::RwLock<Arc<Box<dyn HistoryRepository<C, P, A, K> + Send + Sync + 'static>>>>,
    pub store: Arc<std::sync::RwLock<Arc<Box<dyn HotStore<C, P, A, K>>>>>,
    installs: Arc<Mutex<HashMap<Vec<C>, Install<P, K>>>>,
    event_log: Arc<Mutex<Log>>,
    produce_counter: Arc<Mutex<BTreeMap<Produce, i32>>>,
    matcher: Arc<Box<dyn Match<P, A>>>,
    pub replay_data: Arc<Mutex<MultisetMultiMap<IOEvent, COMM>>>,
    logger: Arc<Mutex<Box<dyn RSpaceLogger<C, P, A, K>>>>,
    replay_waiting_continuations_estimate: Arc<AtomicI64>,
    phase_a_locks: Arc<DashMap<u64, Arc<Mutex<()>>>>,
    phase_b_locks: Arc<DashMap<u64, Arc<Mutex<()>>>>,
}

impl<C, P, A, K> ReplayRSpace<C, P, A, K>
where
    C: Clone + Debug + Default + Serialize + std::hash::Hash + Ord + Eq + 'static + Sync + Send,
    P: Clone + Debug + Default + Serialize + 'static + Sync + Send,
    A: Clone + Debug + Default + Serialize + 'static + Sync + Send,
    K: Clone + Debug + Default + Serialize + 'static + Sync + Send,
{
    pub fn get_store(&self) -> Arc<Box<dyn HotStore<C, P, A, K>>> {
        self.store.read().expect("store read lock").clone()
    }

    pub fn get_history_repository(&self) -> Arc<Box<dyn HistoryRepository<C, P, A, K> + Send + Sync + 'static>> {
        self.history_repository.read().expect("history read lock").clone()
    }

    fn channel_hash(channel: &C) -> u64 {
        use std::hash::Hasher;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        channel.hash(&mut hasher);
        hasher.finish()
    }

    fn acquire_locks(
        lock_map: &DashMap<u64, Arc<Mutex<()>>>,
        keys: &[u64],
    ) -> ChannelLockGuard {
        let mut sorted_keys: Vec<u64> = keys.to_vec();
        sorted_keys.sort();
        sorted_keys.dedup();

        let held: Vec<HeldLock> = sorted_keys
            .iter()
            .map(|k| {
                let lock = lock_map
                    .entry(*k)
                    .or_insert_with(|| Arc::new(Mutex::new(())))
                    .clone();
                HeldLock::acquire(lock)
            })
            .collect();

        ChannelLockGuard { _held: held }
    }

    fn consume_lock(&self, channel_hashes: &[u64]) -> (ChannelLockGuard, ChannelLockGuard) {
        let phase_a = Self::acquire_locks(&self.phase_a_locks, channel_hashes);
        let phase_b = Self::acquire_locks(&self.phase_b_locks, channel_hashes);
        (phase_a, phase_b)
    }

    fn produce_lock(&self, channel: &C) -> (ChannelLockGuard, ChannelLockGuard) {
        let channel_hash = Self::channel_hash(channel);
        let phase_a = Self::acquire_locks(&self.phase_a_locks, &[channel_hash]);

        let store = self.get_store();
        let join_hashes: Vec<u64> = store
            .get_joins(channel)
            .into_iter()
            .flatten()
            .map(|ch| Self::channel_hash(&ch))
            .collect();

        let phase_b = Self::acquire_locks(&self.phase_b_locks, &join_hashes);
        (phase_a, phase_b)
    }
}

struct HeldLock {
    _guard: std::sync::MutexGuard<'static, ()>,
    _lock: Arc<Mutex<()>>,
}

impl HeldLock {
    fn acquire(lock: Arc<Mutex<()>>) -> Self {
        // SAFETY: The Arc is held in the struct alongside the guard,
        // ensuring the Mutex lives as long as the guard. Fields drop
        // in declaration order, so _guard drops before _lock.
        let guard = lock.lock().expect("channel lock");
        let guard: std::sync::MutexGuard<'static, ()> = unsafe { std::mem::transmute(guard) };
        HeldLock {
            _guard: guard,
            _lock: lock,
        }
    }
}

struct ChannelLockGuard {
    _held: Vec<HeldLock>,
}

impl<C, P, A, K> SpaceMatcher<C, P, A, K> for ReplayRSpace<C, P, A, K>
where
    C: Clone + Debug + Default + Serialize + std::hash::Hash + Ord + Eq + 'static + Sync + Send,
    P: Clone + Debug + Default + Serialize + 'static + Sync + Send,
    A: Clone + Debug + Default + Serialize + 'static + Sync + Send,
    K: Clone + Debug + Default + Serialize + 'static + Sync + Send,
{
}

impl<C, P, A, K> ISpace<C, P, A, K> for ReplayRSpace<C, P, A, K>
where
    C: Clone + Debug + Default + Serialize + std::hash::Hash + Ord + Eq + 'static + Sync + Send,
    P: Clone + Debug + Default + Serialize + 'static + Sync + Send,
    A: Clone + Debug + Default + Serialize + 'static + Sync + Send,
    K: Clone + Debug + Default + Serialize + 'static + Sync + Send,
{
    fn create_checkpoint(&self) -> Result<Checkpoint, RSpaceError> {
        self.check_replay_data()?;

        let changes = self.get_store().changes();
        let next_history = self.get_history_repository().checkpoint(changes);
        *self.history_repository.write().expect("history write lock") = Arc::new(next_history);

        let history_reader = self
            .get_history_repository()
            .get_history_reader(&self.get_history_repository().root())?;

        self.create_new_hot_store(history_reader);
        self.restore_installs();

        Ok(Checkpoint {
            root: self.get_history_repository().root(),
            log: Vec::new(),
        })
    }

    fn reset(&self, root: &Blake2b256Hash) -> Result<(), RSpaceError> {
        let next_history = self.get_history_repository().reset(root)?;
        *self.history_repository.write().expect("history write lock") = Arc::new(next_history);

        *self.event_log.lock().expect("event log lock") = Vec::new();
        *self.produce_counter.lock().expect("produce counter lock") = BTreeMap::new();
        self.phase_a_locks.clear();
        self.phase_b_locks.clear();

        let history_reader = self.get_history_repository().get_history_reader(root)?;
        self.create_new_hot_store(history_reader);
        self.restore_installs();
        self.replay_waiting_continuations_estimate
            .store(0, Ordering::Relaxed);
        metrics::gauge!(
            REPLAY_WAITING_CONTINUATIONS_ESTIMATE_METRIC,
            "source" => REPLAY_RSPACE_METRICS_SOURCE
        )
        .set(0.0);

        Ok(())
    }

    fn consume_result(
        &self,
        _channel: Vec<C>,
        _pattern: Vec<P>,
    ) -> Result<Option<(K, Vec<A>)>, RSpaceError> {
        panic!("\nERROR: ReplayRSpace consume_result should not be called here");
    }

    fn get_data(&self, channel: &C) -> Vec<Datum<A>> { self.get_store().get_data(channel) }

    fn get_waiting_continuations(&self, channels: Vec<C>) -> Vec<WaitingContinuation<P, K>> {
        self.get_store().get_continuations(&channels)
    }

    fn get_joins(&self, channel: C) -> Vec<Vec<C>> { self.get_store().get_joins(&channel) }

    fn clear(&self) -> Result<(), RSpaceError> {
        self.replay_data.lock().expect("replay data lock").clear();
        self.reset(&RadixHistory::empty_root_node_hash())
    }

    fn get_root(&self) -> Blake2b256Hash { self.get_history_repository().root() }

    fn to_map(&self) -> HashMap<Vec<C>, Row<P, A, K>> { self.get_store().to_map() }

    fn create_soft_checkpoint(&self) -> SoftCheckpoint<C, P, A, K> {
        let cache_snapshot = self.get_store().snapshot();
        let curr_event_log = std::mem::take(&mut *self.event_log.lock().expect("event log lock"));
        let curr_produce_counter = std::mem::take(&mut *self.produce_counter.lock().expect("produce counter lock"));

        SoftCheckpoint {
            cache_snapshot,
            log: curr_event_log,
            produce_counter: curr_produce_counter,
        }
    }

    fn take_event_log(&self) -> Log {
        let curr_event_log = std::mem::take(&mut *self.event_log.lock().expect("event log lock"));
        let _ = std::mem::take(&mut *self.produce_counter.lock().expect("produce counter lock"));
        curr_event_log
    }

    fn revert_to_soft_checkpoint(
        &self,
        checkpoint: SoftCheckpoint<C, P, A, K>,
    ) -> Result<(), RSpaceError> {
        let history = self.get_history_repository();
        let history_reader = history.get_history_reader(&history.root())?;
        let hot_store = HotStoreInstances::create_from_hs_and_hr(
            checkpoint.cache_snapshot,
            history_reader.base(),
        );

        *self.store.write().expect("store write lock") = Arc::new(hot_store);
        *self.event_log.lock().expect("event log lock") = checkpoint.log;
        *self.produce_counter.lock().expect("produce counter lock") = checkpoint.produce_counter;

        Ok(())
    }

    fn consume(
        &self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<MaybeConsumeResult<C, P, A, K>, RSpaceError> {
        if channels.is_empty() {
            panic!("RUST ERROR: channels can't be empty");
        } else if channels.len() != patterns.len() {
            panic!("RUST ERROR: channels.length must equal patterns.length");
        } else {
            let consume_ref = Consume::create(&channels, &patterns, &continuation, persist);
            let channel_hashes: Vec<u64> = channels.iter().map(|ch| Self::channel_hash(ch)).collect();
            let _lock_guard = self.consume_lock(&channel_hashes);

            metrics::counter!("replay_rspace.consume.calls", "source" => "rspace").increment(1);
            let start = Instant::now();
            let result =
                self.locked_consume(channels, patterns, continuation, persist, peeks, consume_ref);
            metrics::histogram!("replay_consume_time_seconds", "source" => "rspace")
                .record(start.elapsed().as_secs_f64());
            result
        }
    }

    fn produce(
        &self,
        channel: C,
        data: A,
        persist: bool,
    ) -> Result<MaybeProduceResult<C, P, A, K>, RSpaceError> {
        let produce_ref = Produce::create(&channel, &data, persist);
        let _lock_guard = self.produce_lock(&channel);
        metrics::counter!("replay_rspace.produce.calls", "source" => "rspace").increment(1);
        let start = Instant::now();
        let result = self.locked_produce(channel, data, persist, produce_ref);
        metrics::histogram!("replay_produce_time_seconds", "source" => "rspace")
            .record(start.elapsed().as_secs_f64());
        result
    }

    fn install(
        &self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
    ) -> Result<Option<(K, Vec<A>)>, RSpaceError> {
        self.locked_install_internal(channels, patterns, continuation, true)
    }

    fn rig_and_reset(&self, start_root: Blake2b256Hash, log: Log) -> Result<(), RSpaceError> {
        self.rig(log)?;
        self.reset(&start_root)
    }

    fn rig(&self, log: Log) -> Result<(), RSpaceError> {
        let (io_events, comm_events): (Vec<_>, Vec<_>) =
            log.iter().partition(|event| match event {
                Event::IoEvent(IOEvent::Produce(_)) => true,
                Event::IoEvent(IOEvent::Consume(_)) => true,
                Event::Comm(_) => false,
            });

        // Create a set of the "new" IOEvents
        let new_stuff: HashSet<_> = io_events.into_iter().collect();

        // Create and prepare the ReplayData table
        let replay_data = self.replay_data.lock().expect("replay data lock");
        replay_data.clear();

        for event in comm_events {
            match event {
                Event::Comm(comm) => {
                    let comm_cloned = comm.clone();
                    let (consume, produces) = (comm_cloned.consume, comm_cloned.produces);
                    let produce_io_events: Vec<IOEvent> = produces
                        .into_iter()
                        .map(|produce| IOEvent::Produce(produce))
                        .collect();

                    let mut io_events = produce_io_events.clone();
                    io_events.insert(0, IOEvent::Consume(consume));

                    for io_event in io_events {
                        let io_event_converted: Event = match io_event {
                            IOEvent::Produce(ref p) => Event::IoEvent(IOEvent::Produce(p.clone())),
                            IOEvent::Consume(ref c) => Event::IoEvent(IOEvent::Consume(c.clone())),
                        };

                        if new_stuff.contains(&io_event_converted) {
                            replay_data.add_binding(io_event, comm.clone());
                        }
                    }
                    Ok(())
                }
                _ => Err(RSpaceError::BugFoundError(
                    "BUG FOUND: only COMM events are expected here".to_string(),
                )),
            }?
        }

        Ok(())
    }

    fn check_replay_data(&self) -> Result<(), RSpaceError> {
        let replay_data = self.replay_data.lock().expect("replay data lock");
        if replay_data.is_empty() {
            Ok(())
        } else {
            Err(RSpaceError::BugFoundError(format!(
                "Unused COMM event: replayData multimap has {} elements left",
                replay_data.map.len()
            )))
        }
    }

    fn is_replay(&self) -> bool { true }

    fn update_produce(&self, produce_ref: Produce) -> () {
        for event in self.event_log.lock().expect("event log lock").iter_mut() {
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

impl<C, P, A, K> ReplayRSpace<C, P, A, K>
where
    C: Clone + Debug + Default + Serialize + Hash + Ord + Eq + 'static + Sync + Send,
    P: Clone + Debug + Default + Serialize + 'static + Sync + Send,
    A: Clone + Debug + Default + Serialize + 'static + Sync + Send,
    K: Clone + Debug + Default + Serialize + 'static + Sync + Send,
{
    /**
     * Creates [[ReplayRSpace]] from [[HistoryRepository]] and [[HotStore]].
     */
    pub fn apply(
        history_repository: Arc<Box<dyn HistoryRepository<C, P, A, K> + Send + Sync + 'static>>,
        store: Arc<Box<dyn HotStore<C, P, A, K>>>,
        matcher: Arc<Box<dyn Match<P, A>>>,
    ) -> ReplayRSpace<C, P, A, K>
    where
        C: Clone + Debug + Ord + Hash,
        P: Clone + Debug,
        A: Clone + Debug,
        K: Clone + Debug,
    {
        ReplayRSpace {
            history_repository: Arc::new(std::sync::RwLock::new(history_repository)),
            store: Arc::new(std::sync::RwLock::new(store)),
            matcher,
            installs: Arc::new(Mutex::new(HashMap::new())),
            event_log: Arc::new(Mutex::new(Vec::new())),
            produce_counter: Arc::new(Mutex::new(BTreeMap::new())),
            replay_data: Arc::new(Mutex::new(MultisetMultiMap::empty())),
            logger: Arc::new(Mutex::new(Box::new(BasicLogger::new()))),
            replay_waiting_continuations_estimate: Arc::new(AtomicI64::new(0)),
            phase_a_locks: Arc::new(DashMap::new()),
            phase_b_locks: Arc::new(DashMap::new()),
        }
    }

    pub fn apply_with_logger(
        history_repository: Arc<Box<dyn HistoryRepository<C, P, A, K> + Send + Sync + 'static>>,
        store: Arc<Box<dyn HotStore<C, P, A, K>>>,
        matcher: Arc<Box<dyn Match<P, A>>>,
        logger: Box<dyn RSpaceLogger<C, P, A, K>>,
    ) -> ReplayRSpace<C, P, A, K>
    where
        C: Clone + Debug + Ord + Hash,
        P: Clone + Debug,
        A: Clone + Debug,
        K: Clone + Debug,
    {
        ReplayRSpace {
            history_repository: Arc::new(std::sync::RwLock::new(history_repository)),
            store: Arc::new(std::sync::RwLock::new(store)),
            matcher,
            installs: Arc::new(Mutex::new(HashMap::new())),
            event_log: Arc::new(Mutex::new(Vec::new())),
            produce_counter: Arc::new(Mutex::new(BTreeMap::new())),
            replay_data: Arc::new(Mutex::new(MultisetMultiMap::empty())),
            logger: Arc::new(Mutex::new(logger)),
            replay_waiting_continuations_estimate: Arc::new(AtomicI64::new(0)),
            phase_a_locks: Arc::new(DashMap::new()),
            phase_b_locks: Arc::new(DashMap::new()),
        }
    }

    fn inc_replay_waiting_continuations(&self, channels: &[C]) {
        metrics::counter!(
            REPLAY_WAITING_CONTINUATIONS_STORED_TOTAL_METRIC,
            "source" => REPLAY_RSPACE_METRICS_SOURCE
        )
        .increment(1);
        let estimate = self
            .replay_waiting_continuations_estimate
            .fetch_add(1, Ordering::Relaxed) +
            1;
        metrics::gauge!(
            REPLAY_WAITING_CONTINUATIONS_ESTIMATE_METRIC,
            "source" => REPLAY_RSPACE_METRICS_SOURCE
        )
        .set(estimate as f64);
        let channel_depth = self.get_store().get_continuations(channels).len();
        metrics::histogram!(
            REPLAY_WAITING_CONTINUATIONS_CHANNEL_DEPTH_METRIC,
            "source" => REPLAY_RSPACE_METRICS_SOURCE
        )
        .record(channel_depth as f64);
    }

    fn dec_replay_waiting_continuations(&self) {
        let mut current = self
            .replay_waiting_continuations_estimate
            .load(Ordering::Relaxed);
        loop {
            if current <= 0 {
                metrics::gauge!(
                    REPLAY_WAITING_CONTINUATIONS_ESTIMATE_METRIC,
                    "source" => REPLAY_RSPACE_METRICS_SOURCE
                )
                .set(0.0);
                return;
            }
            match self
                .replay_waiting_continuations_estimate
                .compare_exchange_weak(current, current - 1, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => {
                    metrics::gauge!(
                        REPLAY_WAITING_CONTINUATIONS_ESTIMATE_METRIC,
                        "source" => REPLAY_RSPACE_METRICS_SOURCE
                    )
                    .set((current - 1) as f64);
                    return;
                }
                Err(observed) => current = observed,
            }
        }
    }

    fn inc_replay_waiting_continuations_matched_total(&self) {
        metrics::counter!(
            REPLAY_WAITING_CONTINUATIONS_MATCHED_TOTAL_METRIC,
            "source" => REPLAY_RSPACE_METRICS_SOURCE
        )
        .increment(1);
    }

    fn mark_replay_waiting_continuation_match(&self) {
        if self
            .replay_waiting_continuations_estimate
            .load(Ordering::Relaxed) >
            0
        {
            self.dec_replay_waiting_continuations();
        }
        self.inc_replay_waiting_continuations_matched_total();
    }

    fn produce_counters(&self, produce_refs: &[Produce]) -> BTreeMap<Produce, i32> {
        produce_refs
            .iter()
            .cloned()
            .map(|p| (p.clone(), self.produce_counter.lock().expect("produce counter lock").get(&p).unwrap_or(&0).clone()))
            .collect()
    }

    #[inline]
    fn get_produce_count(&self, produce_ref: &Produce) -> i32 {
        *self.produce_counter.lock().expect("produce counter lock").get(produce_ref).unwrap_or(&0)
    }

    fn locked_consume(
        &self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
        consume_ref: Consume,
    ) -> Result<MaybeConsumeResult<C, P, A, K>, RSpaceError> {
        // Span[F].traceI("locked-consume") from Scala - works because this is NOT async
        let _span = tracing::info_span!(target: "f1r3fly.rspace", "locked-consume").entered();
        event!(Level::DEBUG, mark = "started-locked-consume", "locked_consume");

        self.log_consume(consume_ref.clone(), &channels, &patterns, &continuation, persist, &peeks);

        let wk = WaitingContinuation {
            patterns: patterns.clone(),
            continuation,
            persist,
            peeks: peeks.clone(),
            source: consume_ref.clone(),
        };

        let comms_option = self
            .replay_data
            .lock()
            .unwrap()
            .map
            .get(&IOEvent::Consume(consume_ref.clone()))
            .map(|comms| {
                comms
                    .iter()
                    .map(|tuple| tuple.0.clone())
                    .collect::<Vec<_>>()
            });
        match comms_option {
            None => Ok(self.store_waiting_continuation(channels, wk)),
            Some(comms_list) => {
                match self.get_comm_and_consume_candidates(
                    channels.clone(),
                    patterns,
                    comms_list.clone(),
                ) {
                    None => {
                        Ok(self.store_waiting_continuation(channels, wk))
                    }
                    Some((_, data_candidates)) => {
                        let produce_counters_closure =
                            |produces: &[Produce]| self.produce_counters(produces);

                        let comm_ref = COMM::new(
                            &data_candidates,
                            consume_ref.clone(),
                            peeks.clone(),
                            produce_counters_closure,
                        );

                        self.log_comm(
                            &data_candidates,
                            &channels,
                            wk.clone(),
                            comm_ref.clone(),
                            "comm.consume",
                        );

                        assert!(
                            comms_list.contains(&comm_ref),
                            "{}",
                            format!(
                                "COMM Event {:?} was not contained in the trace {:?}",
                                comm_ref, comms_list
                            )
                        );

                        let _ = self.store_persistent_data(data_candidates.clone(), &peeks);
                        let _ = self.remove_bindings_for(comm_ref);
                        Ok(self.wrap_result(channels, wk, consume_ref, data_candidates))
                    }
                }
            }
        }
    }

    /*
     * Here, we create a cache of the data at each channel as
     * `channelToIndexedData` which is used for finding matches.  When a
     * speculative match is found, we can remove the matching datum from the
     * remaining data candidates in the cache.
     *
     * Put another way, this allows us to speculatively remove matching data
     * without affecting the actual store contents.
     */
    fn fetch_channel_to_index_data(&self, channels: &[C]) -> HashMap<C, Vec<(Datum<A>, i32)>> {
        let mut map = HashMap::with_capacity(channels.len());
        for c in channels {
            let data = self.get_store().get_data(c);
            let indexed_data: Vec<(Datum<A>, i32)> = data
                .into_iter()
                .enumerate()
                .map(|(i, d)| (d, i as i32))
                .collect();
            map.insert(c.clone(), indexed_data);
        }
        map
    }

    fn get_comm_and_consume_candidates(
        &self,
        channels: Vec<C>,
        patterns: Vec<P>,
        comms: Vec<COMM>,
    ) -> Option<(COMM, Vec<ConsumeCandidate<C, A>>)> {
        let run_matcher = |comm: COMM| -> Option<Vec<ConsumeCandidate<C, A>>> {
            self.run_matcher_consume(channels.clone(), patterns.clone(), comm)
        };

        self.get_comm_or_candidate(comms, run_matcher)
    }

    fn run_matcher_consume(
        &self,
        channels: Vec<C>,
        patterns: Vec<P>,
        comm: COMM,
    ) -> Option<Vec<ConsumeCandidate<C, A>>> {
        let mut channel_to_indexed_data_list: Vec<(C, Vec<(Datum<A>, i32)>)> = Vec::new();

        for c in &channels {
            let data = self.get_store().get_data(c);
            let filtered_data: Vec<(Datum<A>, i32)> = data
                .into_iter()
                .zip(0..)
                .filter(|(d, i)| self.matches(comm.clone(), (d.clone(), *i)))
                .collect();
            channel_to_indexed_data_list.push((c.clone(), filtered_data));
        }

        let mut channel_to_indexed_data_map: HashMap<C, Vec<(Datum<A>, i32)>> =
            channel_to_indexed_data_list.into_iter().collect();

        let pairs: Vec<(C, P)> = channels.into_iter().zip(patterns.into_iter()).collect();
        let result = self
            .extract_data_candidates(
                &self.matcher,
                &pairs,
                &mut channel_to_indexed_data_map,
            )
            .into_iter()
            .collect::<Option<Vec<_>>>();

        result
    }

    fn locked_produce(
        &self,
        channel: C,
        data: A,
        persist: bool,
        produce_ref: Produce,
    ) -> Result<MaybeProduceResult<C, P, A, K>, RSpaceError> {
        // Span[F].traceI("locked-produce") from Scala - works because this is NOT async
        let _span = tracing::info_span!(target: "f1r3fly.rspace", "locked-produce").entered();
        event!(Level::DEBUG, mark = "started-locked-produce", "locked_produce");

        let grouped_channels = self.get_store().get_joins(&channel);

        self.log_produce(produce_ref.clone(), &channel, &data, persist);

        let io_event_and_comm = self
            .replay_data
            .lock()
            .unwrap()
            .map
            .clone()
            .into_iter()
            .find(|(io_event, _)| match io_event {
                IOEvent::Produce(p) => p.hash == produce_ref.hash,
                _ => false,
            });
        match io_event_and_comm {
            None => Ok(self.store_data(channel, data, persist, produce_ref)),
            Some((_, comms_list)) => {
                let comms: Vec<_> = comms_list.iter().map(|tuple| tuple.0.clone()).collect();

                match self.get_comm_or_produce_candidate(
                    channel.clone(),
                    data.clone(),
                    persist,
                    comms.clone(),
                    produce_ref.clone(),
                    grouped_channels,
                ) {
                    Some((comm, pc)) => Ok(self.handle_match(pc, comms).map(|consume_result| {
                            let p = comm
                                .produces
                                .into_iter()
                                .find(|p| p.hash == produce_ref.hash);
                            (consume_result.0, consume_result.1, p.unwrap_or_else(|| produce_ref))
                        })),
                    None => {
                        Ok(self.store_data(channel, data, persist, produce_ref))
                    }
                }
            }
        }
    }

    fn get_comm_or_produce_candidate(
        &self,
        channel: C,
        data: A,
        persist: bool,
        comms: Vec<COMM>,
        produce_ref: Produce,
        grouped_channels: Vec<Vec<C>>,
    ) -> Option<(COMM, ProduceCandidate<C, P, A, K>)> {
        let run_matcher = |comm: COMM| -> Option<ProduceCandidate<C, P, A, K>> {
            self.run_matcher_produce(
                channel.clone(),
                data.clone(),
                persist,
                comm,
                produce_ref.clone(),
                grouped_channels.clone(),
            )
        };

        self.get_comm_or_candidate(comms, run_matcher)
    }

    fn run_matcher_produce(
        &self,
        channel: C,
        data: A,
        persist: bool,
        comm: COMM,
        produce_ref: Produce,
        grouped_channels: Vec<Vec<C>>,
    ) -> Option<ProduceCandidate<C, P, A, K>> {
        self.run_matcher_for_channels(
            grouped_channels,
            |channels| {
                let continuations = self.get_store().get_continuations(&channels);
                continuations
                    .into_iter()
                    .enumerate()
                    .filter(|(_, wc)| comm.consume == wc.source)
                    .map(|(i, wc)| (wc, i as i32))
                    .collect::<Vec<_>>()
            },
            |c| {
                let store_data = self.get_store().get_data(&c);
                let datum_tuples = store_data
                    .into_iter()
                    .enumerate()
                    .map(|(i, d)| (d, i as i32))
                    .collect::<Vec<_>>();

                let mut result = datum_tuples;
                if c == channel {
                    result.insert(
                        0,
                        (
                            Datum {
                                a: data.clone(),
                                persist,
                                source: produce_ref.clone(),
                            },
                            -1,
                        ),
                    );
                }

                (
                    c.clone(),
                    result
                        .into_iter()
                        .filter(|(datum, i)| self.matches(comm.clone(), (datum.clone(), *i)))
                        .collect(),
                )
            },
        )
    }

    fn matches(&self, comm: COMM, datum_with_index: (Datum<A>, i32)) -> bool {
        let datum = datum_with_index.0;
        let x = comm.produces.contains(&datum.source);
        let res = x && self.was_repeated_enough_times(comm, datum);
        res
    }

    fn was_repeated_enough_times(&self, comm: COMM, datum: Datum<A>) -> bool {
        if !datum.persist {
            let x = *comm.times_repeated.get(&datum.source).unwrap_or(&0) ==
                self.get_produce_count(&datum.source);
            x
        } else {
            true
        }
    }

    fn handle_match(
        &self,
        pc: ProduceCandidate<C, P, A, K>,
        comms: Vec<COMM>,
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

        let produce_counters_closure = |produces: &[Produce]| self.produce_counters(produces);
        let comm_ref = COMM::new(
            &data_candidates,
            consume_ref.clone(),
            peeks.clone(),
            produce_counters_closure,
        );

        self.log_comm(
            &data_candidates,
            &channels,
            continuation.clone(),
            comm_ref.clone(),
            "comm.produce",
        );

        assert!(
            comms.contains(&comm_ref),
            "COMM Event {:?} was not contained in the trace {:?}",
            comm_ref,
            comms
        );

        if !persist {
            self.get_store()
                .remove_continuation(&channels, continuation_index);
            self.mark_replay_waiting_continuation_match();
        } else {
            self.mark_replay_waiting_continuation_match();
        }

        let _ = self.remove_matched_datum_and_join(channels.clone(), data_candidates.clone());
        let _ = self.remove_bindings_for(comm_ref);
        self.wrap_result(channels, continuation.clone(), consume_ref.clone(), data_candidates)
    }

    fn remove_bindings_for(&self, comm_ref: COMM) -> () {
        let replay_data = self.replay_data.lock().expect("replay data lock");
        replay_data
            .remove_binding_in_place(&IOEvent::Consume(comm_ref.consume.clone()), &comm_ref);

        for produce_ref in comm_ref.produces.iter() {
            replay_data
                .remove_binding_in_place(&IOEvent::Produce(produce_ref.clone()), &comm_ref);
        }
    }

    pub fn log_comm(
        &self,
        data_candidates: &Vec<ConsumeCandidate<C, A>>,
        channels: &Vec<C>,
        wk: WaitingContinuation<P, K>,
        comm: COMM,
        label: &str,
    ) {
        // Record metrics using constants to avoid memory leaks
        let metric_label = match label {
            "comm.consume" => CONSUME_COMM_LABEL,
            "comm.produce" => PRODUCE_COMM_LABEL,
            _ => {
                // This should never happen, but guards against future errors.
                tracing::warn!("log_comm called with unexpected dynamic label: {}", label);
                "" // Return an empty string to avoid creating a metric
            }
        };

        if !metric_label.is_empty() {
            metrics::counter!(metric_label, "source" => REPLAY_RSPACE_METRICS_SOURCE).increment(1);
        }

        // Call logger for reporting events
        if let Ok(logger_guard) = self.logger.lock() {
            logger_guard.log_comm(data_candidates, channels, wk, comm, label);
        }
    }

    pub fn log_consume(
        &self,
        consume_ref: Consume,
        channels: &Vec<C>,
        patterns: &Vec<P>,
        continuation: &K,
        persist: bool,
        peeks: &BTreeSet<i32>,
    ) {
        // Call logger for reporting events
        if let Ok(logger_guard) = self.logger.lock() {
            logger_guard.log_consume(consume_ref, channels, patterns, continuation, persist, peeks);
        }
    }

    pub fn log_produce(&self, produce_ref: Produce, channel: &C, data: &A, persist: bool) {
        // Call logger for reporting events
        if let Ok(logger_guard) = self.logger.lock() {
            logger_guard.log_produce(produce_ref.clone(), channel, data, persist);
        }

        if !persist {
            let mut counter = self.produce_counter.lock().expect("produce counter lock");
            let current = counter.get(&produce_ref).copied().unwrap_or(0);
            counter.insert(produce_ref.clone(), current + 1);
        }
    }

    fn get_comm_or_candidate<Candidate>(
        &self,
        comms: Vec<COMM>,
        run_matcher: impl Fn(COMM) -> Option<Candidate>,
    ) -> Option<(COMM, Candidate)> {
        let go = |cs: Vec<COMM>| match cs.as_slice() {
            [] => {
                let msg = "List comms must not be empty";
                panic!("{}", msg);
            }
            [comm_ref] => match run_matcher(comm_ref.clone()) {
                Some(data_candidates) => Ok(Ok((comm_ref.clone(), data_candidates))),
                None => Ok(Err(comm_ref.clone())),
            },
            [comm_ref, rem @ ..] => match run_matcher(comm_ref.clone()) {
                Some(data_candidates) => Ok(Ok((comm_ref.clone(), data_candidates))),
                None => Err(rem.to_vec()),
            },
        };

        let mut cs = comms;
        loop {
            match go(cs.clone()) {
                Ok(Ok(comm_or_candidate)) => return Some(comm_or_candidate),
                Ok(Err(_)) => return None,
                Err(new_cs) => cs = new_cs,
            }
        }
    }

    // This function may need to clear 'replay_data'
    pub fn spawn(&self) -> Result<Self, RSpaceError> {
        // Span[F].withMarks("spawn") from Scala - works because this is NOT async
        let _span = tracing::info_span!(target: "f1r3fly.rspace", "spawn").entered();
        event!(Level::DEBUG, mark = "started-spawn", "spawn");

        let history_repo = self.get_history_repository();
        let next_history = history_repo.reset(&history_repo.root())?;
        let history_reader = next_history.get_history_reader(&next_history.root())?;
        let hot_store = HotStoreInstances::create_from_hr(history_reader.base());
        let rspace =
            Self::apply(Arc::new(next_history), Arc::new(hot_store), self.matcher.clone());
        rspace.restore_installs();

        // Mark the completion of spawn operation
        event!(Level::DEBUG, mark = "finished-spawn", "spawn");
        Ok(rspace)
    }

    /* RSpaceOps */

    fn store_waiting_continuation(
        &self,
        channels: Vec<C>,
        wc: WaitingContinuation<P, K>,
    ) -> MaybeConsumeResult<C, P, A, K> {
        if self
            .get_store()
            .put_continuation(&channels, wc.clone())
            .unwrap_or(false)
        {
            self.inc_replay_waiting_continuations(&channels);
        }
        for channel in channels.iter() {
            self.get_store().put_join(channel, &channels);
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
        self.get_store().put_datum(&channel, Datum {
            a: data,
            persist,
            source: produce_ref,
        });

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
                    self.get_store().remove_datum(&channel, datum_index).ok()
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

    fn restore_installs(&self) -> () {
        // Move out the install map to avoid cloning the whole structure on each
        // restore.
        let installs = {
            let mut installs_lock = self.installs.lock().unwrap();
            std::mem::take(&mut *installs_lock)
        };
        {
            let mut installs_lock = self.installs.lock().unwrap();
            installs_lock.reserve(installs.len());
        }

        for (channels, install) in installs {
            self.locked_install_internal(channels, install.patterns, install.continuation, true)
                .unwrap();
        }
    }

    fn locked_install_internal(
        &self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
        record_install: bool,
    ) -> Result<Option<(K, Vec<A>)>, RSpaceError> {
        if channels.len() != patterns.len() {
            Err(RSpaceError::BugFoundError(
                "RUST ERROR: channels.length must equal patterns.length".to_string(),
            ))
        } else {
            let consume_ref = Consume::create(&channels, &patterns, &continuation, true);
            let mut channel_to_indexed_data = self.fetch_channel_to_index_data(&channels);
            let zipped: Vec<(C, P)> = channels
                .iter()
                .cloned()
                .zip(patterns.iter().cloned())
                .collect();
            let options: Option<Vec<ConsumeCandidate<C, A>>> = self
                .extract_data_candidates(&self.matcher, &zipped, &mut channel_to_indexed_data)
                .into_iter()
                .collect();

            match options {
                None => {
                    if record_install {
                        self.installs
                            .lock()
                            .unwrap()
                            .insert(channels.clone(), Install {
                                patterns: patterns.clone(),
                                continuation: continuation.clone(),
                            });
                    }

                    self.get_store()
                        .install_continuation(&channels, WaitingContinuation {
                            patterns,
                            continuation,
                            persist: true,
                            peeks: BTreeSet::default(),
                            source: consume_ref,
                        });

                    for channel in channels.iter() {
                        self.get_store().install_join(channel, &channels);
                    }
                    Ok(None)
                }
                Some(_) => Err(RSpaceError::BugFoundError(
                    "RUST ERROR: Installing can be done only on startup".to_string(),
                )),
            }
        }
    }

    fn create_new_hot_store(
        &self,
        history_reader: Box<dyn HistoryReader<Blake2b256Hash, C, P, A, K>>,
    ) -> () {
        let next_hot_store = HotStoreInstances::create_from_hr(history_reader.base());
        *self.store.write().expect("store write lock") = Arc::new(next_hot_store);
    }

    fn wrap_result(
        &self,
        channels: Vec<C>,
        wk: WaitingContinuation<P, K>,
        _consume_ref: Consume,
        data_candidates: Vec<ConsumeCandidate<C, A>>,
    ) -> MaybeConsumeResult<C, P, A, K> {
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
                    if self.get_store().remove_datum(&channel, datum_index).is_err() {
                        return None;
                    }
                }
                self.get_store().remove_join(&channel, &channels_clone);

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
                    let channel_to_indexed_data: HashMap<C, Vec<(Datum<A>, i32)>> = channels
                        .iter()
                        .map(|c| fetch_matching_data(c.clone()))
                        .collect();

                    let first_match = self.extract_first_match(
                        &self.matcher,
                        channels.to_vec(),
                        match_candidates,
                        channel_to_indexed_data,
                    );

                    match first_match {
                        Some(produce_candidate) => return Some(produce_candidate),
                        None => remaining = rest.to_vec(),
                    }
                }
                None => {
                    return None;
                }
            }
        }
    }

}
