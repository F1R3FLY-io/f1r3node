// See /home/spreston/src/firefly/f1r3fly/rspace/src/main/scala/coop/rchain/rspace/ReplayRSpace.scala

// NOTE: Manual marks are used instead of trace_i() because async functions
// are not compatible with the current Span trait's FnOnce closure pattern.
// This matches Scala's Span[F].traceI() semantics for async operations.

use super::checkpoint::SoftCheckpoint;
use super::errors::RSpaceError;
use super::hashing::blake2b256_hash::Blake2b256Hash;
use super::history::history_reader::HistoryReader;
use super::history::instances::radix_history::RadixHistory;
use super::logging::{BasicLogger, RSpaceLogger};
use super::metrics_constants::{CONSUME_COMM_LABEL, PRODUCE_COMM_LABEL, REPLAY_RSPACE_METRICS_SOURCE};
use super::r#match::Match;
use super::rspace_interface::ContResult;
use super::rspace_interface::ISpace;
use super::rspace_interface::MaybeConsumeResult;
use super::rspace_interface::MaybeProduceCandidate;
use super::rspace_interface::MaybeProduceResult;
use super::rspace_interface::RSpaceResult;
use super::trace::Log;
use super::trace::event::COMM;
use super::trace::event::Consume;
use super::trace::event::Event;
use super::trace::event::IOEvent;
use super::trace::event::Produce;
use crate::rspace::checkpoint::Checkpoint;
use crate::rspace::history::history_repository::HistoryRepository;
use crate::rspace::hot_store::{HotStore, HotStoreInstances};
use crate::rspace::internal::*;
use crate::rspace::space_matcher::SpaceMatcher;
use dashmap::DashMap;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::Serialize;
use tracing::{event, Level};
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::collections::{BTreeSet, HashMap};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

#[repr(C)]
#[derive(Clone)]
pub struct ReplayRSpace<C, P, A, K> {
    pub history_repository: Arc<Box<dyn HistoryRepository<C, P, A, K> + Send + Sync + 'static>>,
    pub store: Arc<Box<dyn HotStore<C, P, A, K>>>,
    installs: Arc<Mutex<HashMap<Vec<C>, Install<P, K>>>>,
    event_log: Log,
    produce_counter: BTreeMap<Produce, i32>,
    matcher: Arc<Box<dyn Match<P, A>>>,
    // pub ops: RSpaceOps<C, P, A, K>,
    pub replay_data: MultisetMultiMap<IOEvent, COMM>,
    logger: Arc<Mutex<Box<dyn RSpaceLogger<C, P, A, K>>>>,
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
    fn create_checkpoint(&mut self) -> Result<Checkpoint, RSpaceError> {
        // println!("\nhit rspace++ create_checkpoint");

        self.check_replay_data()?;

        let changes = self.store.changes();
        let next_history = self.history_repository.checkpoint(&changes);
        self.history_repository = Arc::new(next_history);

        let history_reader = self
            .history_repository
            .get_history_reader(&self.history_repository.root())?;

        self.create_new_hot_store(history_reader);
        self.restore_installs();

        Ok(Checkpoint {
            root: self.history_repository.root(),
            log: Vec::new(),
        })
    }

    fn reset(&mut self, root: &Blake2b256Hash) -> Result<(), RSpaceError> {
        // println!("\nhit rspace++ reset");
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
        panic!("\nERROR: ReplayRSpace consume_result should not be called here");
    }

    fn get_data(&self, channel: &C) -> Vec<Datum<A>> {
        self.store.get_data(channel)
    }

    fn get_waiting_continuations(&self, channels: Vec<C>) -> Vec<WaitingContinuation<P, K>> {
        self.store.get_continuations(&channels)
    }

    fn get_joins(&self, channel: C) -> Vec<Vec<C>> {
        self.store.get_joins(&channel)
    }

    fn clear(&mut self) -> Result<(), RSpaceError> {
        self.replay_data.clear();
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
        // println!("\nHit consume");

        if channels.is_empty() {
            panic!("RUST ERROR: channels can't be empty");
        } else if channels.len() != patterns.len() {
            panic!("RUST ERROR: channels.length must equal patterns.length");
        } else {
            let consume_ref =
                Consume::create(&channels, &patterns, &continuation, persist);

            let result =
                self.locked_consume(channels, patterns, continuation, persist, peeks, consume_ref);
            // println!("\nlocked_consume result: {:?}", result);
            result
        }
    }

    fn produce(
        &mut self,
        channel: C,
        data: A,
        persist: bool,
    ) -> Result<MaybeProduceResult<C, P, A, K>, RSpaceError> {
        // println!("\nHit produce");
        // println!("\nto_map: {:?}", self.store.to_map());
        // println!("\nHit produce, data: {:?}", data);
        // println!("\n\nHit produce, channel: {:?}", channel);

        let produce_ref = Produce::create(&channel, &data, persist);
        let result = self.locked_produce(channel, data, persist, produce_ref);
        // println!("\nlocked_produce result: {:?}", result);
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

    fn rig_and_reset(&mut self, start_root: Blake2b256Hash, log: Log) -> Result<(), RSpaceError> {
        self.rig(log)?;
        self.reset(&start_root)
    }

    fn rig(&self, log: Log) -> Result<(), RSpaceError> {
        // println!("\nlog len in rust rig: {:?}", log.len());
        let (io_events, comm_events): (Vec<_>, Vec<_>) =
            log.iter().partition(|event| match event {
                Event::IoEvent(IOEvent::Produce(_)) => true,
                Event::IoEvent(IOEvent::Consume(_)) => true,
                Event::Comm(_) => false,
            });

        // Create a set of the "new" IOEvents
        let new_stuff: HashSet<_> = io_events.into_iter().collect();

        // Create and prepare the ReplayData table
        self.replay_data.clear();

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
                            // println!("\nadd_binding in rig");
                            self.replay_data.add_binding(io_event, comm.clone());
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
        if self.replay_data.is_empty() {
            Ok(())
        } else {
            Err(RSpaceError::BugFoundError(format!(
                "Unused COMM event: replayData multimap has {} elements left",
                self.replay_data.map.len()
            )))
        }
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
            history_repository,
            store,
            matcher,
            installs: Arc::new(Mutex::new(HashMap::new())),
            event_log: Vec::new(),
            produce_counter: BTreeMap::new(),
            replay_data: MultisetMultiMap::empty(),
            logger: Arc::new(Mutex::new(Box::new(BasicLogger::new()))),
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
            history_repository,
            store,
            matcher,
            installs: Arc::new(Mutex::new(HashMap::new())),
            event_log: Vec::new(),
            produce_counter: BTreeMap::new(),
            replay_data: MultisetMultiMap::empty(),
            logger: Arc::new(Mutex::new(logger)),
        }
    }

    fn produce_counters(&self, produce_refs: Vec<Produce>) -> BTreeMap<Produce, i32> {
        produce_refs
            .into_iter()
            .map(|p| (p.clone(), self.produce_counter.get(&p).unwrap_or(&0).clone()))
            .collect()
    }

    #[inline]
    fn get_produce_count(&self, produce_ref: &Produce) -> i32 {
        *self.produce_counter.get(produce_ref).unwrap_or(&0)
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
        // Span[F].traceI("locked-consume") from Scala - works because this is NOT async
        let _span = tracing::info_span!(target: "f1r3fly.rspace", "locked-consume").entered();
        event!(Level::DEBUG, mark = "started-locked-consume", "locked_consume");
        
        // println!(
        //     "consume: searching for data matching <patterns: {:?}> at <channels: {:?}>",
        //     patterns, channels
        // );

        self.log_consume(consume_ref.clone(), &channels, &patterns, &continuation, persist, &peeks);

        let wk = WaitingContinuation {
            patterns: patterns.clone(),
            continuation,
            persist,
            peeks: peeks.clone(),
            source: consume_ref.clone(),
        };

        // println!("\nreplay_data len in replay_consume: {:?}", self.replay_data.map.len());

        let comms_option = self
            .replay_data
            .map
            .get(&IOEvent::Consume(consume_ref.clone()))
            .map(|comms| {
                comms
                    .iter()
                    .map(|tuple| tuple.0.clone())
                    .collect::<Vec<_>>()
            });

        // println!("\ncomms_options in replay_consume Some?: {:?}", comms_option.is_some());

        match comms_option {
            None => Ok(self.store_waiting_continuation(channels, wk)),
            Some(comms_list) => {
                match self.get_comm_and_consume_candidates(
                    channels.clone(),
                    patterns,
                    comms_list.clone(),
                ) {
                    None => {
                        // println!("\nwas none");
                        Ok(self.store_waiting_continuation(channels, wk))
                    }
                    Some((_, data_candidates)) => {
                        let produce_counters_closure =
                            |produces: Vec<Produce>| self.produce_counters(produces);

                        let comm_ref = COMM::new(
                            data_candidates.clone(),
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
                        // println!(
                        //     "consume: data found for <patterns: {:?}> at <channels: {:?}>",
                        //     patterns, channels
                        // );
                        let _ = self.remove_bindings_for(comm_ref);
                        Ok(self.wrap_result(channels, wk, consume_ref, data_candidates))
                    }
                }
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
    fn fetch_channel_to_index_data(&self, channels: &[C]) -> DashMap<C, Vec<(Datum<A>, i32)>> {
        let map = DashMap::with_capacity(channels.len());
        for c in channels {
            let data = self.store.get_data(c);
            let shuffled_data = self.shuffle_with_index(data);
            map.insert(c.clone(), shuffled_data);
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
            let data = self.store.get_data(c);
            // println!("\ndata len: {}", data.len());
            let filtered_data: Vec<(Datum<A>, i32)> = data
                .into_iter()
                .zip(0..)
                .filter(|(d, i)| self.matches(comm.clone(), (d.clone(), *i)))
                .collect();
            channel_to_indexed_data_list.push((c.clone(), filtered_data));
        }

        // println!("\nchannelToIndexedDataList: {:#?}", channel_to_indexed_data_list);

        let channel_to_indexed_data_map: DashMap<C, Vec<(Datum<A>, i32)>> =
            channel_to_indexed_data_list.into_iter().collect();

        let result = self
            .extract_data_candidates(
                &self.matcher,
                channels.into_iter().zip(patterns.into_iter()).collect(),
                channel_to_indexed_data_map,
                vec![],
            )
            .into_iter()
            .collect::<Option<Vec<_>>>();

        result
    }

    fn locked_produce(
        &mut self,
        channel: C,
        data: A,
        persist: bool,
        produce_ref: Produce,
    ) -> Result<MaybeProduceResult<C, P, A, K>, RSpaceError> {
        // Span[F].traceI("locked-produce") from Scala - works because this is NOT async
        let _span = tracing::info_span!(target: "f1r3fly.rspace", "locked-produce").entered();
        event!(Level::DEBUG, mark = "started-locked-produce", "locked_produce");
        
        // println!("\nHit replay_locked_produce");

        let grouped_channels = self.store.get_joins(&channel);
        // println!(
        //     "produce: searching for matching continuations at <grouped_channels: {:?}>",
        //     grouped_channels
        // );
        self.log_produce(produce_ref.clone(), &channel, &data, persist);

        let io_event_and_comm = self
            .replay_data
            .map
            .clone()
            .into_iter()
            .find(|(io_event, _)| match io_event {
                IOEvent::Produce(p) => p.hash == produce_ref.hash,
                _ => false,
            });

        // println!("\nreplay_data in replay_produce: {:?}", self.replay_data);
        // println!("\ncomms_options in replay_produce Some?: {:?}", comms_option.is_some());

        match io_event_and_comm {
            None => Ok(self.store_data(channel, data, persist, produce_ref)),
            Some((_, comms_list)) => {
                let comms: Vec<_> = comms_list
                    .iter()
                    .map(|tuple| tuple.0.clone())
                    .collect();

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
                        // println!("\nwas none");
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
                let continuations = self.store.get_continuations(&channels);
                continuations
                    .into_iter()
                    .enumerate()
                    .filter(|(_, wc)| comm.consume == wc.source)
                    .map(|(i, wc)| (wc, i as i32))
                    .collect::<Vec<_>>()
            },
            |c| {
                let store_data = self.store.get_data(&c);
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
        // println!("\ncomm in matches: {:?}", comm);
        let datum = datum_with_index.0;
        let x = comm.produces.contains(&datum.source);
        let res = x && self.was_repeated_enough_times(comm, datum);
        // println!("\ncomm.produce.contains: {:?}", x);
        // println!("\nmatches result: {:?}", res);
        res
    }

    fn was_repeated_enough_times(&self, comm: COMM, datum: Datum<A>) -> bool {
        // println!("\ncomm in was_repeated_enough_times: {:?}", comm);
        // println!("\n\ndatum in was_repeated_enough_times: {:?}", datum);
        // println!("\nproduce_counter: {:?}", self.produce_counter);
        if !datum.persist {
            let x = *comm.times_repeated.get(&datum.source).unwrap_or(&0)
                == self.get_produce_count(&datum.source);
            // println!("\nwas_repeated_enough_times result: {:?}", x);
            x
        } else {
            true
        }
    }

    fn handle_match(
        &mut self,
        pc: ProduceCandidate<C, P, A, K>,
        comms: Vec<COMM>,
    ) -> MaybeConsumeResult<C, P, A, K> {
        // println!("\nhit handle_match");

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
        let comm_ref = COMM::new(
            data_candidates.clone(),
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
            let _ = self
                .store
                .remove_continuation(&channels, continuation_index);
        };

        let _ = self.remove_matched_datum_and_join(channels.clone(), data_candidates.clone());
        // println!("produce: matching continuation found at <channels: {:?}>", channels);
        let _ = self.remove_bindings_for(comm_ref);
        self.wrap_result(channels, continuation.clone(), consume_ref.clone(), data_candidates)
    }

    fn remove_bindings_for(&mut self, comm_ref: COMM) -> () {
        // println!("\nhit remove_bindings_for");

        let updated_replays = self.replay_data.clone();
        updated_replays.remove_binding_in_place(&IOEvent::Consume(comm_ref.consume.clone()), &comm_ref);

        for produce_ref in comm_ref.produces.iter() {
            updated_replays.remove_binding_in_place(&IOEvent::Produce(produce_ref.clone()), &comm_ref);
        }

        self.replay_data = updated_replays;
    }

    pub fn log_comm(
        &mut self,
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
                log::warn!("log_comm called with unexpected dynamic label: {}", label);
                "" // Return an empty string to avoid creating a metric
            }
        };

        if !metric_label.is_empty() {
            metrics::counter!(metric_label, "source" => REPLAY_RSPACE_METRICS_SOURCE).increment(1);
        }
        
        // Call logger for reporting events
        if let Ok(mut logger_guard) = self.logger.lock() {
            logger_guard.log_comm(data_candidates, channels, wk, comm, label);
        }
    }

    pub fn log_consume(
        &mut self,
        consume_ref: Consume,
        channels: &Vec<C>,
        patterns: &Vec<P>,
        continuation: &K,
        persist: bool,
        peeks: &BTreeSet<i32>,
    ) {
        // Call logger for reporting events
        if let Ok(mut logger_guard) = self.logger.lock() {
            logger_guard.log_consume(consume_ref, channels, patterns, continuation, persist, peeks);
        }
    }

    pub fn log_produce(
        &mut self,
        produce_ref: Produce,
        channel: &C,
        data: &A,
        persist: bool,
    ) {
        // Call logger for reporting events
        if let Ok(mut logger_guard) = self.logger.lock() {
            logger_guard.log_produce(produce_ref.clone(), channel, data, persist);
        }
        
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
        
        let history_repo = &self.history_repository;
        let next_history = history_repo.reset(&history_repo.root())?;
        let history_reader = next_history.get_history_reader(&next_history.root())?;
        let hot_store = HotStoreInstances::create_from_hr(history_reader.base());
        let mut rspace =
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
        // println!("\nHit store_waiting_continuation");
        self.store.put_continuation(&channels, wc);
        for channel in channels.iter() {
            self.store.put_join(channel, &channels);
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
        self.store.put_datum(&channel, Datum {
            a: data,
            persist,
            source: produce_ref,
        });
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
                    self.store.remove_datum(&channel, datum_index)
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
            Err(RSpaceError::BugFoundError(
                "RUST ERROR: channels.length must equal patterns.length".to_string(),
            ))
        } else {
            // println!(
            //     "install: searching for data matching <patterns: {:?}> at <channels: {:?}>",
            //     patterns, channels
            // );

            let consume_ref =
                Consume::create(&channels, &patterns, &continuation, true);
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

                    self.store
                        .install_continuation(&channels, WaitingContinuation {
                            patterns,
                            continuation,
                            persist: true,
                            peeks: BTreeSet::default(),
                            source: consume_ref,
                        },
                    );

                    for channel in channels.iter() {
                        self.store.install_join(channel, &channels);
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
                    self.store.remove_datum(&channel, datum_index);
                }
                self.store.remove_join(&channel, &channels_clone);

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
