// See rspace/src/main/scala/coop/rchain/rspace/RSpace.scala

// NOTE: Manual marks are used instead of trace_i()/with_marks() because
// the functions are not async-compatible with Span trait's closure pattern.
// This matches Scala's Span[F].traceI() and withMarks() semantics.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::collections::hash_map::DefaultHasher;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Instant;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use shared::rust::store::key_value_store::KeyValueStore;
use tracing::{Level, event};

use super::checkpoint::SoftCheckpoint;
use super::errors::{HistoryRepositoryError, RSpaceError};
use super::hashing::blake2b256_hash::Blake2b256Hash;
use super::hashing::stable_hash_provider::hash as channel_hash;
use super::history::history_reader::HistoryReader;
use super::history::instances::radix_history::RadixHistory;
use super::logging::BasicLogger;
use super::r#match::Match;
use super::metrics_constants::{
    CHANGES_SPAN, CONSUME_COMM_LABEL, HISTORY_CHECKPOINT_SPAN, LOCKED_CONSUME_SPAN,
    LOCKED_PRODUCE_SPAN, PRODUCE_COMM_LABEL, RESET_SPAN, REVERT_SOFT_CHECKPOINT_SPAN,
    RSPACE_METRICS_SOURCE,
};
use super::replay_rspace::ReplayRSpace;
use super::rspace_interface::{
    ContResult, ISpace, MaybeConsumeResult, MaybeProduceCandidate, MaybeProduceResult, RSpaceResult,
};
use super::trace::Log;
use super::trace::event::{COMM, Consume, Event, IOEvent, Produce};
use crate::rspace::checkpoint::Checkpoint;
use crate::rspace::history::history_repository::{HistoryRepository, HistoryRepositoryInstances};
use crate::rspace::hot_store::{HotStore, HotStoreInstances};
use crate::rspace::hot_store_action::{DeleteAction, HotStoreAction, InsertAction};
use crate::rspace::internal::*;
use crate::rspace::space_matcher::SpaceMatcher;

#[derive(Clone)]
pub struct RSpaceStore {
    pub history: Arc<dyn KeyValueStore>,
    pub roots: Arc<dyn KeyValueStore>,
    pub cold: Arc<dyn KeyValueStore>,
}

/// Guard that holds a per-channel-group lock.
///
/// Owns the `Arc<Mutex<()>>` to keep the mutex alive for the duration
/// of the guard, and the `MutexGuard` that actually holds the lock.
/// Using a raw pointer to work around the self-referential lifetime issue:
/// the `MutexGuard` borrows from the `Mutex` inside the `Arc`, but Rust
/// cannot express this directly. The `Arc` ensures the `Mutex` lives as
/// long as this struct, and Drop releases in the correct order.
pub struct ChannelGroupGuard {
    _guard: std::sync::MutexGuard<'static, ()>,
    _lock: Arc<std::sync::Mutex<()>>,
}

impl ChannelGroupGuard {
    pub fn new(lock: Arc<std::sync::Mutex<()>>) -> Self {
        // SAFETY: The Arc keeps the Mutex alive. We transmute the lifetime
        // to 'static because we store the Arc alongside the guard, guaranteeing
        // the Mutex outlives the guard. The guard is dropped before the Arc
        // because struct fields are dropped in declaration order.
        let guard = unsafe {
            let mutex_ref: &std::sync::Mutex<()> = &*lock;
            let static_ref: &'static std::sync::Mutex<()> = std::mem::transmute(mutex_ref);
            static_ref.lock().expect("channel group lock poisoned")
        };
        ChannelGroupGuard {
            _guard: guard,
            _lock: lock,
        }
    }
}

#[repr(C)]
#[derive(Clone)]
pub struct RSpace<C, P, A, K> {
    pub history_repository: Arc<RwLock<Arc<Box<dyn HistoryRepository<C, P, A, K> + Send + Sync + 'static>>>>,
    pub store: Arc<RwLock<Arc<Box<dyn HotStore<C, P, A, K>>>>>,
    installs: Arc<Mutex<BTreeMap<Vec<C>, Install<P, K>>>>,
    event_log: Arc<Mutex<Log>>,
    produce_counter: Arc<Mutex<BTreeMap<Produce, i32>>>,
    matcher: Arc<Box<dyn Match<P, A>>>,
    channel_locks: Arc<DashMap<u64, Arc<std::sync::Mutex<()>>>>,
}

fn block_creator_phase_substep_profile_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("F1R3_BLOCK_CREATOR_PHASE_SUBSTEP_PROFILE")
            .ok()
            .map(|v| {
                let normalized = v.trim().to_ascii_lowercase();
                matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
    })
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
    fn create_checkpoint(&self) -> Result<Checkpoint, RSpaceError> {
        // Span[F].withMarks("create-checkpoint") from Scala - works because this is NOT
        // async
        let _span = tracing::info_span!(target: "f1r3fly.rspace", "create-checkpoint").entered();
        event!(Level::DEBUG, mark = "started-create-checkpoint", "create_checkpoint");
        let mem_profile_enabled = block_creator_phase_substep_profile_enabled();
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
                    "create_checkpoint.mem step={} rss_kb={} delta_prev_kb={} delta_total_kb={}",
                    step, curr_kb, delta_prev_kb, delta_total_kb
                );
                mem_prev_kb = Some(curr_kb);
            }
        };
        log_mem_step("start");

        // Get changes with span
        let changes = {
            let _changes_span =
                tracing::info_span!(target: "f1r3fly.rspace", CHANGES_SPAN).entered();
            self.get_store().changes()
        };
        // Diagnostic: count state changes by type for checkpoint
        {
            let mut insert_data = 0usize;
            let mut insert_cont = 0usize;
            let mut insert_join = 0usize;
            let mut delete_data = 0usize;
            let mut delete_cont = 0usize;
            let mut delete_join = 0usize;

            let detail_enabled = tracing::enabled!(
                target: "f1r3fly.rspace.checkpoint_detail",
                tracing::Level::DEBUG
            );

            for action in &changes {
                match action {
                    HotStoreAction::Insert(InsertAction::InsertData(id)) => {
                        insert_data += 1;
                        if detail_enabled {
                            tracing::debug!(
                                target: "f1r3fly.rspace.checkpoint_detail",
                                channel = ?id.channel,
                                data_count = id.data.len(),
                                "checkpoint_detail: InsertData"
                            );
                        }
                    }
                    HotStoreAction::Insert(InsertAction::InsertContinuations(ic)) => {
                        insert_cont += 1;
                        if detail_enabled {
                            let persistent_count = ic.continuations.iter().filter(|wc| wc.persist).count();
                            tracing::debug!(
                                target: "f1r3fly.rspace.checkpoint_detail",
                                channels = ?ic.channels,
                                cont_count = ic.continuations.len(),
                                persistent_count,
                                "checkpoint_detail: InsertContinuations ({} total, {} persistent)",
                                ic.continuations.len(), persistent_count
                            );
                        }
                    }
                    HotStoreAction::Insert(InsertAction::InsertJoins(ij)) => {
                        insert_join += 1;
                        if detail_enabled {
                            tracing::debug!(
                                target: "f1r3fly.rspace.checkpoint_detail",
                                channel = ?ij.channel,
                                join_groups = ij.joins.len(),
                                "checkpoint_detail: InsertJoins ({} groups)",
                                ij.joins.len()
                            );
                        }
                    }
                    HotStoreAction::Delete(DeleteAction::DeleteData(dd)) => {
                        delete_data += 1;
                        if detail_enabled {
                            tracing::debug!(
                                target: "f1r3fly.rspace.checkpoint_detail",
                                channel = ?dd.channel,
                                "checkpoint_detail: DeleteData"
                            );
                        }
                    }
                    HotStoreAction::Delete(DeleteAction::DeleteContinuations(dc)) => {
                        delete_cont += 1;
                        if detail_enabled {
                            tracing::debug!(
                                target: "f1r3fly.rspace.checkpoint_detail",
                                channels = ?dc.channels,
                                "checkpoint_detail: DeleteContinuations"
                            );
                        }
                    }
                    HotStoreAction::Delete(DeleteAction::DeleteJoins(dj)) => {
                        delete_join += 1;
                        if detail_enabled {
                            tracing::debug!(
                                target: "f1r3fly.rspace.checkpoint_detail",
                                channel = ?dj.channel,
                                "checkpoint_detail: DeleteJoins"
                            );
                        }
                    }
                }
            }
            tracing::debug!(
                target: "f1r3fly.rspace",
                total_changes = changes.len(),
                insert_data,
                insert_cont,
                insert_join,
                delete_data,
                delete_cont,
                delete_join,
                "checkpoint: committing state changes"
            );
            // LFS diagnostic: log checkpoint summary at INFO level
            tracing::info!(
                target: "f1r3fly.rspace.lfs_diag",
                total_changes = changes.len(),
                insert_data,
                insert_cont,
                insert_join,
                delete_data,
                delete_cont,
                delete_join,
                "CHECKPOINT: committing {} changes (data: +{} -{}, cont: +{} -{}, join: +{} -{})",
                changes.len(), insert_data, delete_data,
                insert_cont, delete_cont, insert_join, delete_join
            );
        }

        log_mem_step("after_store_changes");

        // Create history checkpoint with span
        let next_history = {
            let _history_span =
                tracing::info_span!(target: "f1r3fly.rspace", HISTORY_CHECKPOINT_SPAN).entered();
            let hr = self.history_repository.read().expect("history_repository read lock in create_checkpoint");
            hr.checkpoint(changes)
        };
        log_mem_step("after_history_checkpoint");
        {
            let mut hr = self.history_repository.write().expect("history_repository write lock in create_checkpoint (set)");
            *hr = Arc::new(next_history);
        }
        log_mem_step("after_set_history_repository");

        let log = std::mem::take(&mut *self.event_log.lock().expect("event_log lock in create_checkpoint"));
        log_mem_step("after_take_event_log");
        let _ = std::mem::take(&mut *self.produce_counter.lock().expect("produce_counter lock in create_checkpoint"));
        log_mem_step("after_take_produce_counter");

        let history_reader = {
            let hr = self.history_repository.read().expect("history_repository read lock in create_checkpoint (reader)");
            hr.get_history_reader(&hr.root())?
        };
        log_mem_step("after_get_history_reader");

        self.create_new_hot_store(history_reader);
        log_mem_step("after_create_new_hot_store");
        self.restore_installs();
        log_mem_step("after_restore_installs");

        // Mark the completion of create-checkpoint
        event!(Level::DEBUG, mark = "finished-create-checkpoint", "create_checkpoint");
        log_mem_step("finish");

        Ok(Checkpoint {
            root: self.history_repository.read().expect("history_repository read lock in create_checkpoint (root)").root(),
            log,
        })
    }

    fn reset(&self, root: &Blake2b256Hash) -> Result<(), RSpaceError> {
        let _span = tracing::info_span!(target: "f1r3fly.rspace", RESET_SPAN).entered();
        tracing::debug!(
            target: "f1r3fly.rspace",
            root_hash = ?root,
            "reset: loading state from root"
        );

        let next_history = {
            let hr = self.history_repository.read().expect("history_repository read lock in reset");
            hr.reset(root)?
        };
        {
            let mut hr = self.history_repository.write().expect("history_repository write lock in reset (set)");
            *hr = Arc::new(next_history);
        }

        *self.event_log.lock().expect("event_log lock in reset") = Vec::new();
        *self.produce_counter.lock().expect("produce_counter lock in reset") = BTreeMap::new();

        let history_reader = {
            let hr = self.history_repository.read().expect("history_repository read lock in reset (reader)");
            hr.get_history_reader(root)?
        };
        self.create_new_hot_store(history_reader);
        self.restore_installs();

        Ok(())
    }

    fn consume_result(
        &self,
        _channel: Vec<C>,
        _pattern: Vec<P>,
    ) -> Result<Option<(K, Vec<A>)>, RSpaceError> {
        panic!("\nERROR: RSpace consume_result should not be called here");
    }

    fn get_data(&self, channel: &C) -> Vec<Datum<A>> { self.get_store().get_data(channel) }

    fn get_waiting_continuations(&self, channels: Vec<C>) -> Vec<WaitingContinuation<P, K>> {
        self.get_store().get_continuations(&channels)
    }

    fn get_joins(&self, channel: C) -> Vec<Vec<C>> { self.get_store().get_joins(&channel) }

    fn clear(&self) -> Result<(), RSpaceError> {
        self.reset(&RadixHistory::empty_root_node_hash())
    }

    fn get_root(&self) -> Blake2b256Hash { self.history_repository.read().expect("history_repository read lock in get_root").root() }

    fn to_map(&self) -> HashMap<Vec<C>, Row<P, A, K>> { self.get_store().to_map() }

    fn create_soft_checkpoint(&self) -> SoftCheckpoint<C, P, A, K> {
        // println!("\nhit rspace++ create_soft_checkpoint");
        // println!("current hot_store state: {:?}", self.store.snapshot());

        let cache_snapshot = self.get_store().snapshot();
        let curr_event_log = std::mem::take(&mut *self.event_log.lock().expect("event_log lock in create_soft_checkpoint"));
        let curr_produce_counter = std::mem::take(&mut *self.produce_counter.lock().expect("produce_counter lock in create_soft_checkpoint"));

        SoftCheckpoint {
            cache_snapshot,
            log: curr_event_log,
            produce_counter: curr_produce_counter,
        }
    }

    fn take_event_log(&self) -> Log {
        let curr_event_log = std::mem::take(&mut *self.event_log.lock().expect("event_log lock in take_event_log"));
        let _ = std::mem::take(&mut *self.produce_counter.lock().expect("produce_counter lock in take_event_log"));
        curr_event_log
    }

    fn revert_to_soft_checkpoint(
        &self,
        checkpoint: SoftCheckpoint<C, P, A, K>,
    ) -> Result<(), RSpaceError> {
        let _span =
            tracing::info_span!(target: "f1r3fly.rspace", REVERT_SOFT_CHECKPOINT_SPAN).entered();
        let history_reader = {
            let history = self.history_repository.read().expect("history_repository read lock in revert_to_soft_checkpoint");
            history.get_history_reader(&history.root())?
        };
        let hot_store = HotStoreInstances::create_from_mhs_and_hr(
            Arc::new(checkpoint.cache_snapshot),
            history_reader.base(),
        );

        self.create_new_hot_store_from(hot_store);
        *self.event_log.lock().expect("event_log lock in revert_to_soft_checkpoint") = checkpoint.log;
        *self.produce_counter.lock().expect("produce_counter lock in revert_to_soft_checkpoint") = checkpoint.produce_counter;

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
        // println!("\nrspace consume");
        // println!("channels: {:?}", channels);
        // println!("space in consume before: {:?}", self.get_store().to_map().len());

        if channels.is_empty() {
            panic!("RUST ERROR: channels can't be empty");
        } else if channels.len() != patterns.len() {
            panic!("RUST ERROR: channels.length must equal patterns.length");
        } else {
            let consume_ref = Consume::create(&channels, &patterns, &continuation, persist);

            let start = Instant::now();
            let result = self.locked_consume(
                &channels,
                &patterns,
                &continuation,
                persist,
                &peeks,
                &consume_ref,
            );
            let duration = start.elapsed();
            metrics::histogram!("comm_consume_time_seconds", "source" => RSPACE_METRICS_SOURCE)
                .record(duration.as_secs_f64());
            // println!("locked_consume result: {:?}", result);
            // println!("\nspace in consume after: {:?}", self.store.to_map().len());
            result
        }
    }

    fn produce(
        &self,
        channel: C,
        data: A,
        persist: bool,
    ) -> Result<MaybeProduceResult<C, P, A, K>, RSpaceError> {
        // println!("\nrspace produce");
        // println!("space in produce: {:?}", self.get_store().to_map().len());
        // println!("\nHit produce, data: {:?}", data);
        // println!("\n\nHit produce, channel: {:?}", channel);

        let produce_ref = Produce::create(&channel, &data, persist);
        let start = Instant::now();
        let result = self.locked_produce(channel, data, persist, &produce_ref);
        let duration = start.elapsed();
        metrics::histogram!("comm_produce_time_seconds", "source" => RSPACE_METRICS_SOURCE)
            .record(duration.as_secs_f64());
        // println!("\nlocked_produce result: {:?}", result);
        // println!("\nspace in produce: {:?}", self.store.to_map().len());
        result
    }

    fn install(
        &self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
    ) -> Result<Option<(K, Vec<A>)>, RSpaceError> {
        let start = Instant::now();
        let result = self.locked_install_internal(channels, patterns, continuation, true);
        let duration = start.elapsed();
        metrics::histogram!("install_time_seconds", "source" => RSPACE_METRICS_SOURCE)
            .record(duration.as_secs_f64());
        result
    }

    fn rig_and_reset(&self, _start_root: Blake2b256Hash, _log: Log) -> Result<(), RSpaceError> {
        panic!("\nERROR: RSpace rig_and_reset should not be called here");
    }

    fn rig(&self, _log: Log) -> Result<(), RSpaceError> {
        panic!("\nERROR: RSpace rig should not be called here");
    }

    fn check_replay_data(&self) -> Result<(), RSpaceError> {
        panic!("\nERROR: RSpace check_replay_data should not be called here");
    }

    fn is_replay(&self) -> bool { false }

    fn update_produce(&self, produce_ref: Produce) -> () {
        let mut event_log = self.event_log.lock().expect("event_log lock in update_produce");
        for event in event_log.iter_mut() {
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

    fn pending_state_counts(&self) -> (usize, usize, usize, usize) {
        self.get_store().state_counts()
    }

    fn pending_continuation_channels_debug(&self) -> Vec<(String, usize, bool)> {
        self.get_store().continuation_channels_debug()
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
        history_repository: Arc<Box<dyn HistoryRepository<C, P, A, K> + Send + Sync + 'static>>,
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
            history_repository: Arc::new(RwLock::new(history_repository)),
            store: Arc::new(RwLock::new(Arc::new(store))),
            matcher,
            installs: Arc::new(Mutex::new(BTreeMap::new())),
            event_log: Arc::new(Mutex::new(Vec::new())),
            produce_counter: Arc::new(Mutex::new(BTreeMap::new())),
            channel_locks: Arc::new(DashMap::new()),
        }
    }

    /// Returns a clone of the store Arc for lock-free read access.
    /// Uses RwLock::read() so multiple produce/consume operations can access
    /// the store concurrently. The HotStore trait methods use interior mutability
    /// (DashMap), so callers can use the returned Arc without holding any lock.
    pub fn get_store(&self) -> Arc<Box<dyn HotStore<C, P, A, K>>> {
        self.store.read().expect("store read lock in get_store").clone()
    }

    /// Returns a clone of the history_repository Arc for lock-free read access.
    pub fn get_history_repository(&self) -> Arc<Box<dyn HistoryRepository<C, P, A, K> + Send + Sync + 'static>> {
        self.history_repository.read().expect("history_repository read lock in get_history_repository").clone()
    }

    /// Acquires a per-channel-group lock for the given set of channels.
    ///
    /// Channels are hashed individually, sorted for deterministic ordering
    /// (preventing deadlocks from different channel orderings), then combined
    /// into a single key that identifies the channel group. The lock is
    /// created on first access and cached in the `channel_locks` DashMap.
    fn lock_channel_group(&self, channels: &[C]) -> ChannelGroupGuard {
        let mut hashes: Vec<u64> = channels.iter().map(|c| {
            let mut h = DefaultHasher::new();
            c.hash(&mut h);
            h.finish()
        }).collect();
        hashes.sort();

        let mut hasher = DefaultHasher::new();
        for h in &hashes {
            h.hash(&mut hasher);
        }
        let key = hasher.finish();

        let lock = self.channel_locks
            .entry(key)
            .or_insert_with(|| Arc::new(std::sync::Mutex::new(())))
            .clone();
        ChannelGroupGuard::new(lock)
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
        (
            Box<dyn HistoryRepository<C, P, A, K> + Send + Sync + 'static>,
            Box<dyn HotStore<C, P, A, K>>,
        ),
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

    fn produce_counters(&self, produce_refs: &[Produce]) -> BTreeMap<Produce, i32> {
        let pc = self.produce_counter.lock().expect("produce_counter lock in produce_counters");
        produce_refs
            .iter()
            .cloned()
            .map(|p| (p.clone(), pc.get(&p).unwrap_or(&0).clone()))
            .collect()
    }

    fn locked_consume(
        &self,
        channels: &[C],
        patterns: &[P],
        continuation: &K,
        persist: bool,
        peeks: &BTreeSet<i32>,
        consume_ref: &Consume,
    ) -> Result<MaybeConsumeResult<C, P, A, K>, RSpaceError> {
        // Span[F].traceI("locked-consume") from Scala
        let _span = tracing::info_span!(target: "f1r3fly.rspace", LOCKED_CONSUME_SPAN).entered();
        event!(Level::DEBUG, mark = "started-locked-consume", "locked_consume");

        // Acquire per-channel-group lock for this consume's channel set
        let _channel_guard = self.lock_channel_group(channels);

        // println!("\nHit locked_consume");
        // println!(
        //     "consume: searching for data matching <patterns: {:?}> at <channels:
        // {:?}>",     patterns, channels
        // );

        // Diagnostic: log channel hashes for cross-referencing validator writes vs observer reads
        if tracing::enabled!(target: "f1r3fly.rspace.channel_hash", tracing::Level::DEBUG) {
            for (i, ch) in channels.iter().enumerate() {
                let ch_hash = channel_hash(ch);
                tracing::debug!(
                    target: "f1r3fly.rspace.channel_hash",
                    channel_idx = i,
                    channel = ?ch,
                    channel_hash = %ch_hash,
                    persist,
                    op = "consume",
                    "locked_consume: channel[{}] hash={}",
                    i, ch_hash
                );
            }
        }

        self.log_consume(consume_ref, channels, patterns, continuation, persist, peeks);

        // Diagnostic: log consumes on registry channels 14/15/16
        for ch in channels.iter() {
            let ch_dbg = format!("{:?}", ch);
            for byte_id in [14u8, 15, 16] {
                let pattern = format!("id: [{}]", byte_id);
                if ch_dbg.contains(&pattern) {
                    tracing::debug!(
                        target: "f1r3fly.rspace",
                        channel_id = byte_id,
                        persist,
                        patterns_count = patterns.len(),
                        "consume on registry channel"
                    );

                    // Step 4: When a persistent consume targets byte_name(14),
                    // log the serialized bytes and hash as ground truth for
                    // comparing against produce-time lookups.
                    if byte_id == 14 && persist {
                        let serialized_bytes = bincode::serialize(ch).expect("serialize channel for diag");
                        let ch_hash = Blake2b256Hash::new(&serialized_bytes);
                        let channels_dbg: Vec<String> = channels.iter().map(|c| format!("{:?}", c)).collect();
                        tracing::info!(
                            target: "f1r3fly.rholang.diag",
                            serialized_hex = %hex::encode(&serialized_bytes),
                            channel_hash = %ch_hash,
                            channel_debug = %ch_dbg,
                            persist,
                            patterns_count = patterns.len(),
                            all_channels_count = channels.len(),
                            "CONSUME on byte_name(14) [GENESIS GROUND TRUTH]: hash={}, serialized={} bytes, channels={:?}",
                            ch_hash,
                            serialized_bytes.len(),
                            channels_dbg
                        );
                    }
                }
            }
        }

        let channel_to_indexed_data = self.fetch_channel_to_index_data(channels);
        // LFS diagnostic: log peek operations with channel hash and data availability
        if !peeks.is_empty() {
            for (i, ch) in channels.iter().enumerate() {
                let ch_hash = channel_hash(ch);
                let has_data = channel_to_indexed_data
                    .get(&ch.clone())
                    .map_or(false, |d| !d.is_empty());
                tracing::info!(
                    target: "f1r3fly.rspace.lfs_diag",
                    channel_idx = i,
                    channel_hash = %hex::encode(ch_hash.bytes()),
                    has_data,
                    data_count = channel_to_indexed_data.get(&ch.clone()).map_or(0, |d| d.len()),
                    "PEEK_LOOKUP: channel_hash={} has_data={} data_count={}",
                    hex::encode(&ch_hash.bytes()[..8]),
                    has_data,
                    channel_to_indexed_data.get(&ch.clone()).map_or(0, |d| d.len())
                );
            }
        }
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
            patterns: patterns.to_vec(),
            continuation: continuation.clone(),
            persist,
            peeks: peeks.clone(),
            source: consume_ref.clone(),
        };

        match options {
            Some(data_candidates) => {

                tracing::debug!(
                    target: "f1r3fly.rspace",
                    channels = ?channels,
                    data_candidates_count = data_candidates.len(),
                    persist = wk.persist,
                    "locked_consume: COMM fired (data found)"
                );

                let produce_counters_closure =
                    |produces: &[Produce]| self.produce_counters(produces);

                self.log_comm(
                    channels,
                    &wk,
                    COMM::new(
                        &data_candidates,
                        consume_ref.clone(),
                        peeks.clone(),
                        produce_counters_closure,
                    ),
                    "comm.consume",
                );
                self.store_persistent_data(channels, &data_candidates, peeks);
                event!(Level::DEBUG, mark = "finished-locked-consume", "locked_consume");
                Ok(self.wrap_result(channels, &wk, consume_ref, &data_candidates))
            }
            None => {

                tracing::debug!(
                    target: "f1r3fly.rspace",
                    channels = ?channels,
                    persist = wk.persist,
                    "locked_consume: no match, storing continuation"
                );

                // Phase 4: When a peek blocks, log channel details, check
                // history, and decode any data found to reveal the tree hash
                // map contents.
                if !peeks.is_empty() {
                    for (i, ch) in channels.iter().enumerate() {
                        let ch_dbg = format!("{:?}", ch);
                        // Only log for 32-byte GPrivate channels (skip system channels)
                        if ch_dbg.len() > 200 {
                            let data_from_store = self.get_store().get_data(ch);
                            let conts_from_store = self.get_store().get_continuations(&[ch.clone()]);
                            let joins_from_store = self.get_store().get_joins(ch);
                            let serialized = bincode::serialize(ch).expect("serialize channel for peek diag");
                            let ch_hash = Blake2b256Hash::new(&serialized);

                            // Extract GPrivate hex for channel identification
                            let gprivate_hex: String = ch_dbg
                                .find("id: [")
                                .and_then(|start| {
                                    ch_dbg[start..].find(']').map(|end| {
                                        ch_dbg[start + 5..start + end].to_string()
                                    })
                                })
                                .unwrap_or_else(|| "<unknown>".to_string());

                            tracing::warn!(
                                target: "f1r3fly.rholang.diag",
                                channel_idx = i,
                                channel_hash = %ch_hash,
                                gprivate_id = %gprivate_hex,
                                data_count = data_from_store.len(),
                                conts_count = conts_from_store.len(),
                                joins_count = joins_from_store.len(),
                                serialized_len = serialized.len(),
                                serialized_hex_prefix = %hex::encode(&serialized[..serialized.len().min(64)]),
                                "PEEK BLOCKED: no data on 32-byte GPrivate channel — \
                                 data={}, conts={}, joins={}, hash={}",
                                data_from_store.len(),
                                conts_from_store.len(),
                                joins_from_store.len(),
                                ch_hash
                            );

                            // Phase 5d Step 2: detect "dead end" — no data AND no
                            // existing continuations means nothing will ever wake
                            // this peek-consume. The treeHashMap node data is
                            // missing from the trie.
                            if data_from_store.is_empty() && conts_from_store.is_empty() {
                                tracing::error!(
                                    target: "f1r3fly.rspace.lfs_diag",
                                    channel_idx = i,
                                    channel_hash_full = %hex::encode(ch_hash.bytes()),
                                    channel_hash_short = %ch_hash,
                                    gprivate_id = %gprivate_hex,
                                    serialized_hex = %hex::encode(&serialized),
                                    serialized_len = serialized.len(),
                                    "DEAD END: peek-consume on GPrivate channel has NO data \
                                     AND NO existing continuations — this channel's data is \
                                     missing from both hot store and history trie. \
                                     Search validator logs for this channel_hash_full to verify \
                                     if the data exists on the validator."
                                );
                            }

                            // If data EXISTS but peek didn't match, decode and
                            // log each datum's content for diagnosis
                            for (d_idx, datum) in data_from_store.iter().enumerate() {
                                let datum_dbg = format!("{:?}", datum.a);
                                let datum_preview = if datum_dbg.len() > 500 {
                                    format!("{}...[truncated]", &datum_dbg[..500])
                                } else {
                                    datum_dbg
                                };
                                tracing::warn!(
                                    target: "f1r3fly.rholang.diag",
                                    datum_idx = d_idx,
                                    persist = datum.persist,
                                    datum_preview = %datum_preview,
                                    "PEEK BLOCKED: datum[{}] on channel — persist={}, content={}",
                                    d_idx, datum.persist, datum_preview
                                );
                            }

                            // Log each pattern for cross-reference with data
                            for (p_idx, pat) in patterns.iter().enumerate() {
                                let pat_dbg = format!("{:?}", pat);
                                let pat_preview = if pat_dbg.len() > 300 {
                                    format!("{}...[truncated]", &pat_dbg[..300])
                                } else {
                                    pat_dbg
                                };
                                tracing::warn!(
                                    target: "f1r3fly.rholang.diag",
                                    pattern_idx = p_idx,
                                    pattern_preview = %pat_preview,
                                    "PEEK BLOCKED: pattern[{}] = {}",
                                    p_idx, pat_preview
                                );
                            }
                        }
                    }
                }

                event!(Level::DEBUG, mark = "finished-locked-consume", "locked_consume");
                self.store_waiting_continuation(channels.to_vec(), wk);
                Ok(None)
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
    fn fetch_channel_to_index_data(&self, channels: &[C]) -> DashMap<C, Vec<(Datum<A>, i32)>> {
        let map = DashMap::with_capacity(channels.len());
        for c in channels {
            let data = self.get_store().get_data(c);
            let shuffled_data = self.order_by_hash_with_index(data, |d| &d.source.hash);
            map.insert(c.clone(), shuffled_data);
        }
        map
    }

    fn locked_produce(
        &self,
        channel: C,
        data: A,
        persist: bool,
        produce_ref: &Produce,
    ) -> Result<MaybeProduceResult<C, P, A, K>, RSpaceError> {
        // Span[F].traceI("locked-produce") from Scala
        let _span = tracing::info_span!(target: "f1r3fly.rspace", LOCKED_PRODUCE_SPAN).entered();
        event!(Level::DEBUG, mark = "started-locked-produce", "locked_produce");

        // Diagnostic: log channel hash for cross-referencing validator writes vs observer reads
        if tracing::enabled!(target: "f1r3fly.rspace.channel_hash", tracing::Level::DEBUG) {
            let ch_hash = channel_hash(&channel);
            tracing::debug!(
                target: "f1r3fly.rspace.channel_hash",
                channel = ?channel,
                channel_hash = %ch_hash,
                persist,
                op = "produce",
                "locked_produce: channel hash={}",
                ch_hash
            );
        }

        let grouped_channels = self.get_store().get_joins(&channel);
        tracing::debug!(
            target: "f1r3fly.rspace",
            channel = ?channel,
            joins_count = grouped_channels.len(),
            persist,
            "locked_produce: get_joins returned {} channel groups",
            grouped_channels.len()
        );

        // Diagnostic: when joins=0 for a 32-byte unforgeable, check if conts/data exist anyway
        if grouped_channels.is_empty()
            && tracing::enabled!(target: "f1r3fly.rspace.orphan_produce", tracing::Level::DEBUG)
        {
            let ch_dbg = format!("{:?}", channel);
            // Only log for 32-byte unforgeable channels (skip short explore-deploy channels)
            if ch_dbg.contains("GPrivateBody") && ch_dbg.len() > 200 {
                let conts = self.get_store().get_continuations(&[channel.clone()]);
                let data_at_ch = self.get_store().get_data(&channel);
                tracing::debug!(
                    target: "f1r3fly.rspace.orphan_produce",
                    channel = ?channel,
                    conts_count = conts.len(),
                    persistent_conts = conts.iter().filter(|wc| wc.persist).count(),
                    data_count = data_at_ch.len(),
                    "orphan_produce: joins=0 but channel has {} conts ({} persistent) and {} data",
                    conts.len(),
                    conts.iter().filter(|wc| wc.persist).count(),
                    data_at_ch.len()
                );
            }
        }

        // Diagnostic: targeted byte_name(14) registry channel probe during produce
        {
            let ch_dbg = format!("{:?}", channel);
            if ch_dbg.contains("id: [14]") {
                let serialized_bytes = bincode::serialize(&channel).expect("serialize channel for diag");
                let ch_hash = Blake2b256Hash::new(&serialized_bytes);
                let conts = self.get_store().get_continuations(&[channel.clone()]);
                let data_at_ch = self.get_store().get_data(&channel);
                tracing::info!(
                    target: "f1r3fly.rholang.diag",
                    joins_count = grouped_channels.len(),
                    conts_count = conts.len(),
                    persistent_conts = conts.iter().filter(|wc| wc.persist).count(),
                    data_count = data_at_ch.len(),
                    serialized_hex = %hex::encode(&serialized_bytes),
                    channel_hash = %ch_hash,
                    channel_debug = %ch_dbg,
                    persist,
                    "PRODUCE on byte_name(14): joins={}, conts={} (persistent={}), data={}",
                    grouped_channels.len(),
                    conts.len(),
                    conts.iter().filter(|wc| wc.persist).count(),
                    data_at_ch.len()
                );
            }
        }

        self.log_produce(produce_ref, &channel, &data, persist);

        // Try each channel group under its own per-channel-group lock.
        // This allows independent channel groups to proceed concurrently
        // while serializing operations on the same channel group.
        let datum = Datum {
            a: data.clone(),
            persist,
            source: produce_ref.clone(),
        };

        let mut extracted: MaybeProduceCandidate<C, P, A, K> = None;
        for channels in &grouped_channels {
            let _channel_guard = self.lock_channel_group(channels);
            let candidate = self.extract_produce_candidate_for_group(
                channels.clone(),
                channel.clone(),
                datum.clone(),
            );
            if candidate.is_some() {
                extracted = candidate;
                break;
            }
        }

        match extracted {
            Some(produce_candidate) => {

                tracing::info!(
                    target: "f1r3fly.rspace.cost_trace",
                    produce_hash = %hex::encode(produce_ref.hash.bytes()),
                    channel_hash = %hex::encode(produce_ref.channel_hash.bytes()),
                    persist,
                    "PRODUCE_HIT: produce COMM fired on validator"
                );
                tracing::debug!(
                    target: "f1r3fly.rspace",
                    channel = ?channel,
                    persist,
                    "locked_produce: COMM fired (continuation found)"
                );
                // Diagnostic: log byte_name(14) COMM success with the data that was produced
                if format!("{:?}", channel).contains("id: [14]") {
                    let data_dbg = format!("{:?}", data);
                    // Truncate to avoid flooding logs with full data
                    let data_preview = if data_dbg.len() > 500 {
                        format!("{}...[truncated at 500 of {} chars]", &data_dbg[..500], data_dbg.len())
                    } else {
                        data_dbg
                    };
                    tracing::info!(
                        target: "f1r3fly.rholang.diag",
                        persist,
                        data_preview = %data_preview,
                        "PRODUCE on byte_name(14): COMM FIRED — registry lookup matched, data={}", data_preview
                    );
                }
                event!(Level::DEBUG, mark = "finished-locked-produce", "locked_produce");
                Ok(self
                    .process_match_found(produce_candidate)
                    .map(|consume_result| {
                        (consume_result.0, consume_result.1, produce_ref.clone())
                    }))
            }
            None => {
                tracing::info!(
                    target: "f1r3fly.rspace.cost_trace",
                    produce_hash = %hex::encode(produce_ref.hash.bytes()),
                    channel_hash = %hex::encode(produce_ref.channel_hash.bytes()),
                    persist,
                    "PRODUCE_STORE: produce stored without COMM (validator)"
                );
                tracing::debug!(
                    target: "f1r3fly.rspace",
                    channel = ?channel,
                    persist,
                    "locked_produce: no match, storing data"
                );
                // Diagnostic: log byte_name(14) COMM failure
                if format!("{:?}", channel).contains("id: [14]") {
                    tracing::warn!(
                        target: "f1r3fly.rholang.diag",
                        persist,
                        "PRODUCE on byte_name(14): NO MATCH — registry COMM did NOT fire, data stored without matching"
                    );
                }
                event!(Level::DEBUG, mark = "finished-locked-produce", "locked_produce");
                Ok(self.store_data(channel, data, persist, produce_ref.clone()))
            }
        }
    }

    /*
     * Find produce candidate for a single channel group.
     *
     * This is called under the per-channel-group lock, allowing independent
     * channel groups to proceed concurrently.
     */
    fn extract_produce_candidate_for_group(
        &self,
        channels: Vec<C>,
        bat_channel: C,
        data: Datum<A>,
    ) -> MaybeProduceCandidate<C, P, A, K> {
        let match_candidates: Vec<(WaitingContinuation<P, K>, i32)> = {
            let continuations = self.get_store().get_continuations(&channels);
            self.order_by_hash_with_index(continuations, |wc| &wc.source.hash)
        };

        let channel_to_indexed_data: DashMap<C, Vec<(Datum<A>, i32)>> = channels
            .iter()
            .map(|c| {
                let data_vec = self.get_store().get_data(c);
                let mut shuffled_data = self.order_by_hash_with_index(data_vec, |d| &d.source.hash);
                if *c == bat_channel {
                    shuffled_data.insert(0, (data.clone(), -1));
                }
                (c.clone(), shuffled_data)
            })
            .collect();

        self.extract_first_match(
            &self.matcher,
            channels,
            match_candidates,
            channel_to_indexed_data,
        )
    }

    /*
     * Find produce candidate (iterates through ALL channel groups).
     *
     * NOTE: This method is retained for reference but is no longer called
     * from locked_produce, which now uses extract_produce_candidate_for_group
     * with per-channel-group locking.
     *
     * NOTE: On Rust side, we are NOT passing functions through. Instead just the
     * data. And then in 'run_matcher_for_channels' we call the functions
     * defined below
     */
    #[allow(dead_code)]
    fn extract_produce_candidate(
        &self,
        grouped_channels: Vec<Vec<C>>,
        bat_channel: C,
        data: Datum<A>,
    ) -> MaybeProduceCandidate<C, P, A, K> {
        // println!("\nHit extract_produce_candidate");

        let fetch_matching_continuations =
            |channels: Vec<C>| -> Vec<(WaitingContinuation<P, K>, i32)> {
                let continuations = self.get_store().get_continuations(&channels);
                self.order_by_hash_with_index(continuations, |wc| &wc.source.hash)
            };

        /*
         * Here, we create a cache of the data at each channel as
         * `channelToIndexedData` which is used for finding matches.  When a
         * speculative match is found, we can remove the matching datum from
         * the remaining data candidates in the cache.
         *
         * Put another way, this allows us to speculatively remove matching data
         * without affecting the actual store contents.
         *
         * In this version, we also add the produced data directly to this cache.
         */
        let fetch_matching_data = |channel| -> (C, Vec<(Datum<A>, i32)>) {
            let data_vec = self.get_store().get_data(&channel);
            let mut shuffled_data = self.order_by_hash_with_index(data_vec, |d| &d.source.hash);
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
        &self,
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

        let produce_counters_closure = |produces: &[Produce]| self.produce_counters(produces);
        self.log_comm(
            &channels,
            &continuation,
            COMM::new(
                &data_candidates,
                consume_ref.clone(),
                peeks.clone(),
                produce_counters_closure,
            ),
            "comm.produce",
        );

        if !persist {
            self.get_store()
                .remove_continuation(&channels, continuation_index);
        }

        self.remove_matched_datum_and_join(&channels, &data_candidates, peeks);

        // println!(
        //     "produce: matching continuation found at <channels: {:?}>",
        //     channels
        // );

        self.wrap_result(&channels, &continuation, consume_ref, &data_candidates)
    }

    fn log_comm(
        &self,
        _channels: &[C],
        _wk: &WaitingContinuation<P, K>,
        comm: COMM,
        label: &str,
    ) {
        // Increment counter FIRST (matching Scala) using constants to avoid memory
        // leaks Labels are always "comm.consume" or "comm.produce" based on the
        // RSpace implementation
        match label {
            "comm.consume" => {
                metrics::counter!(CONSUME_COMM_LABEL, "source" => RSPACE_METRICS_SOURCE)
                    .increment(1);
            }
            "comm.produce" => {
                metrics::counter!(PRODUCE_COMM_LABEL, "source" => RSPACE_METRICS_SOURCE)
                    .increment(1);
            }
            _ => {
                // This should never happen, but log if it does
                tracing::warn!("Unexpected label in log_comm: {}", label);
            }
        }

        // Then update event log (RSpace-specific behavior)
        self.event_log.lock().expect("event_log lock in log_comm").insert(0, Event::Comm(comm));
    }

    fn log_consume(
        &self,
        consume_ref: &Consume,
        _channels: &[C],
        _patterns: &[P],
        _continuation: &K,
        _persist: bool,
        _peeks: &BTreeSet<i32>,
    ) {
        self.event_log.lock().expect("event_log lock in log_consume")
            .insert(0, Event::IoEvent(IOEvent::Consume(consume_ref.clone())));
    }

    fn log_produce(&self, produce_ref: &Produce, _channel: &C, _data: &A, persist: bool) {
        self.event_log.lock().expect("event_log lock in log_produce")
            .insert(0, Event::IoEvent(IOEvent::Produce(produce_ref.clone())));
        if !persist {
            let mut pc = self.produce_counter.lock().expect("produce_counter lock in log_produce");
            let current_count = pc.get(produce_ref).copied().unwrap_or(0);
            pc.insert(produce_ref.clone(), current_count + 1);
        }
    }

    pub fn spawn(&self) -> Result<Self, RSpaceError> {
        let parent_root = self.get_history_repository().root();
        self.spawn_at(&parent_root)
    }

    /// Creates a child RSpace positioned at the given state root.
    ///
    /// Unlike `spawn()`, which inherits the parent's current (possibly stale) root,
    /// this method creates the child directly at the specified state — ensuring the
    /// history reader and hot store are consistent with the target block state from
    /// the start.
    pub fn spawn_at(&self, root: &Blake2b256Hash) -> Result<Self, RSpaceError> {
        let _span = tracing::info_span!(target: "f1r3fly.rspace", "spawn").entered();
        event!(Level::DEBUG, mark = "started-spawn", "spawn");

        let history_repo = self.get_history_repository();
        tracing::debug!(
            target: "f1r3fly.rspace",
            root = ?root,
            "spawn_at: creating child RSpace at specified root"
        );

        let next_history = history_repo.reset(root)?;
        let history_reader = next_history.get_history_reader(&next_history.root())?;
        let hot_store = HotStoreInstances::create_from_hr(history_reader.base());
        let rspace = RSpace::apply(Arc::new(next_history), hot_store, self.matcher.clone());

        // Copy parent's system contract installs so restore_installs() can re-install them.
        // This makes spawn self-contained — callers don't need to separately set up
        // system contracts. Note: create_rho_runtime() also installs system contracts
        // via create_rho_env(), so for the standard explore-deploy path this is redundant
        // but harmless. For other spawn() callers, this ensures correctness.
        {
            let parent_installs = self.installs.lock().expect("parent installs lock poisoned");
            let mut child_installs = rspace.installs.lock().expect("child installs lock poisoned");
            tracing::debug!(
                target: "f1r3fly.rspace",
                parent_installs_count = parent_installs.len(),
                "spawn_at: copying parent installs to child"
            );
            *child_installs = parent_installs.clone();
        }

        rspace.restore_installs();

        event!(Level::DEBUG, mark = "finished-spawn", "spawn");
        Ok(rspace)
    }

    /* RSpaceOps */

    fn store_waiting_continuation(
        &self,
        channels: Vec<C>,
        wc: WaitingContinuation<P, K>,
    ) -> MaybeConsumeResult<C, P, A, K> {
        // println!("\nHit store_waiting_continuation");
        let channel_hashes: Vec<_> = channels.iter().map(|ch| channel_hash(ch)).collect();
        tracing::debug!(
            target: "f1r3fly.rspace",
            channels = ?channels,
            channel_hashes = ?channel_hashes,
            persist = wc.persist,
            "store_waiting_continuation: storing continuation and joins"
        );
        let _ = self.get_store().put_continuation(&channels, wc);
        for channel in channels.iter() {
            self.get_store().put_join(channel, &channels);
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
        if tracing::enabled!(target: "f1r3fly.rspace.channel_hash", tracing::Level::DEBUG) {
            let ch_hash = channel_hash(&channel);
            tracing::debug!(
                target: "f1r3fly.rspace.channel_hash",
                channel = ?channel,
                channel_hash = %ch_hash,
                persist,
                op = "store_data",
                "store_data: persisting datum at channel hash={}",
                ch_hash
            );
        }
        self.get_store().put_datum(&channel, Datum {
            a: data,
            persist,
            source: produce_ref,
        });

        None
    }

    fn store_persistent_data(
        &self,
        channels: &[C],
        data_candidates: &Vec<ConsumeCandidate<C, A>>,
        peeks: &BTreeSet<i32>,
    ) -> Option<Vec<()>> {
        let mut sorted_candidates: Vec<_> = data_candidates.iter().collect();
        sorted_candidates.sort_by(|a, b| b.datum_index.cmp(&a.datum_index));
        let results: Vec<_> = sorted_candidates
            .into_iter()
            .rev()
            .map(|consume_candidate| {
                let ConsumeCandidate {
                    channel,
                    datum: Datum { persist, .. },
                    removed_datum: _,
                    datum_index,
                } = consume_candidate;

                let channel_idx = channels
                    .iter()
                    .position(|c| c == channel)
                    .expect("ConsumeCandidate channel must exist in channels list") as i32;
                let is_peeked = peeks.contains(&channel_idx);

                if !persist && !is_peeked {
                    // Phase 5e: log caller context before remove_datum to trace
                    // spurious DeleteData on peek-only channels like treeHashMapCh
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
                            tracing::warn!(
                                target: "f1r3fly.rholang.diag",
                                gprivate_id = %gprivate_hex,
                                caller = "store_persistent_data",
                                persist,
                                is_peeked,
                                datum_index,
                                channel_idx,
                                num_channels = channels.len(),
                                peeks = ?peeks,
                                "store_persistent_data: about to remove_datum on 32-byte \
                                 GPrivate — persist={}, is_peeked={}, datum_index={}, \
                                 channel_idx={}, peeks={:?}",
                                persist, is_peeked, datum_index, channel_idx, peeks
                            );
                        }
                    }
                    self.get_store().remove_datum(channel, *datum_index).ok()
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
        // restore.  BTreeMap iteration order is deterministic (sorted by key),
        // ensuring install_join calls happen in the same order on every node.
        let installs = {
            let mut installs_lock = self.installs.lock().expect("installs lock in restore_installs");
            std::mem::take(&mut *installs_lock)
        };

        for (channels, install) in installs {
            self.locked_install_internal(channels, install.patterns, install.continuation, true)
                .expect("locked_install_internal failed in restore_installs");
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
            panic!("RUST ERROR: channels.length must equal patterns.length");
        } else {
            // LFS diagnostic: check if continuations already exist for these channels
            let existing_installed = self.installs.lock().unwrap().contains_key(&channels);
            let existing_conts = self.get_store().get_continuations(&channels);
            if !existing_conts.is_empty() || existing_installed {
                tracing::warn!(
                    target: "f1r3fly.rspace.lfs_diag",
                    channel_count = channels.len(),
                    existing_installed,
                    existing_cont_count = existing_conts.len(),
                    "INSTALL DUPLICATE: install() called on channels that already have \
                     continuations — this may cause state divergence during replay"
                );
            }

            let consume_ref = Consume::create(&channels, &patterns, &continuation, true);
            let channel_to_indexed_data = self.fetch_channel_to_index_data(&channels);
            let zipped: Vec<(C, P)> = channels
                .iter()
                .cloned()
                .zip(patterns.iter().cloned())
                .collect();
            let options: Option<Vec<ConsumeCandidate<C, A>>> = self
                .extract_data_candidates(&self.matcher, zipped, channel_to_indexed_data, Vec::new())
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
        *self.store.write().expect("store write lock in create_new_hot_store") = Arc::new(next_hot_store);
    }

    fn create_new_hot_store_from(
        &self,
        hot_store: Box<dyn HotStore<C, P, A, K>>,
    ) -> () {
        *self.store.write().expect("store write lock in create_new_hot_store_from") = Arc::new(hot_store);
    }

    fn wrap_result(
        &self,
        channels: &[C],
        wk: &WaitingContinuation<P, K>,
        _consume_ref: &Consume,
        data_candidates: &Vec<ConsumeCandidate<C, A>>,
    ) -> MaybeConsumeResult<C, P, A, K> {
        // println!("\nhit wrap_result");

        let cont_result = ContResult {
            continuation: wk.continuation.clone(),
            persistent: wk.persist,
            channels: channels.to_vec(),
            patterns: wk.patterns.clone(),
            peek: !wk.peeks.is_empty(),
        };

        let rspace_results = data_candidates
            .iter()
            .map(|data_candidate| RSpaceResult {
                channel: data_candidate.channel.clone(),
                matched_datum: data_candidate.datum.a.clone(),
                removed_datum: data_candidate.removed_datum.clone(),
                persistent: data_candidate.datum.persist,
            })
            .collect();

        Some((cont_result, rspace_results))
    }

    fn remove_matched_datum_and_join(
        &self,
        channels: &[C],
        data_candidates: &[ConsumeCandidate<C, A>],
        peeks: &BTreeSet<i32>,
    ) -> Option<Vec<()>> {
        let mut sorted_candidates: Vec<_> = data_candidates.iter().collect();
        sorted_candidates.sort_by(|a, b| b.datum_index.cmp(&a.datum_index));
        let results: Vec<_> = sorted_candidates
            .into_iter()
            .rev()
            .map(|consume_candidate| {
                let ConsumeCandidate {
                    channel,
                    ref datum,
                    removed_datum: _,
                    datum_index,
                } = consume_candidate;
                let persist = datum.persist;

                // Determine if this channel was peeked in the continuation.
                // Peeked channels should not have their data removed.
                let channel_idx = channels
                    .iter()
                    .position(|c| c == channel)
                    .expect("ConsumeCandidate channel must exist in channels list") as i32;
                let is_peeked = peeks.contains(&channel_idx);

                if *datum_index >= 0 && !persist && !is_peeked {
                    // Phase 5e: log caller context before remove_datum to trace
                    // spurious DeleteData on peek-only channels like treeHashMapCh
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
                            tracing::warn!(
                                target: "f1r3fly.rholang.diag",
                                gprivate_id = %gprivate_hex,
                                caller = "remove_matched_datum_and_join",
                                persist,
                                is_peeked,
                                datum_index,
                                channel_idx,
                                num_channels = channels.len(),
                                peeks = ?peeks,
                                "remove_matched_datum_and_join: about to remove_datum on \
                                 32-byte GPrivate — persist={}, is_peeked={}, datum_index={}, \
                                 channel_idx={}, peeks={:?}",
                                persist, is_peeked, datum_index, channel_idx, peeks
                            );
                        }
                    }
                    if self.get_store().remove_datum(&channel, *datum_index).is_err() {
                        return None;
                    }
                } else if *datum_index < 0 && is_peeked {
                    // On-the-fly produced data matched a waiting peek continuation.
                    // The data was never stored, but peek semantics require it to
                    // persist. Store it now so future consumers can find it.
                    self.get_store().put_datum(channel, datum.clone());
                }
                self.get_store().remove_join(&channel, &channels);

                Some(())
            })
            .collect();

        if results.iter().any(|res| res.is_none()) {
            None
        } else {
            Some(results.into_iter().filter_map(|x| x).collect())
        }
    }

    // Retained for reference; no longer called from locked_produce which now
    // uses extract_produce_candidate_for_group with per-channel-group locking.
    #[allow(dead_code)]
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

    /// Order candidates deterministically by content hash for fair matching.
    ///
    /// Replaces the previous `thread_rng()` shuffle with content-hash ordering.
    /// This preserves fairness (Blake2b256 hashes are uniformly distributed,
    /// so different data values hash to different positions with no systematic
    /// bias) while being deterministic (same data → same hash → same order).
    /// Determinism is required for consensus: all validators evaluating the
    /// same block must produce the same COMM events and state hash.
    fn order_by_hash_with_index<D>(
        &self,
        t: Vec<D>,
        hash_fn: impl Fn(&D) -> &Blake2b256Hash,
    ) -> Vec<(D, i32)> {
        let mut indexed_vec = t
            .into_iter()
            .enumerate()
            .map(|(i, d)| (d, i as i32))
            .collect::<Vec<_>>();
        indexed_vec.sort_by(|(a, _), (b, _)| hash_fn(a).cmp(&hash_fn(b)));
        indexed_vec
    }
}
