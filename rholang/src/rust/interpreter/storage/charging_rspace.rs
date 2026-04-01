// See rholang/src/main/scala/coop/rchain/rholang/interpreter/storage/ChargingRSpace.scala

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::rust::interpreter::{
    accounting::{
        _cost,
        costs::{
            comm_event_storage_cost, event_storage_cost, storage_cost_consume,
            storage_cost_produce, Cost,
        },
    },
    errors::InterpreterError,
};
use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::{
    tagged_continuation::TaggedCont, BindPattern, ListParWithRandom, Par, TaggedContinuation,
};
use rspace_plus_plus::rspace::{
    checkpoint::{Checkpoint, SoftCheckpoint},
    errors::RSpaceError,
    hashing::blake2b256_hash::Blake2b256Hash,
    internal::{Datum, Row, WaitingContinuation},
    rspace_interface::{ContResult, ISpace, MaybeConsumeResult, MaybeProduceResult, RSpaceResult},
    trace::{event::Produce, Log},
    util::unpack_option,
};

pub struct ChargingRSpace;

/// Global sequence counter for cost trace alignment across validator/observer.
static COST_TRACE_SEQ: AtomicU64 = AtomicU64::new(0);

/// Reset the sequence counter (call at the start of each deploy evaluation).
pub fn reset_cost_trace_seq() {
    COST_TRACE_SEQ.store(0, Ordering::Relaxed);
}

fn next_cost_trace_seq() -> u64 {
    COST_TRACE_SEQ.fetch_add(1, Ordering::Relaxed)
}


#[derive(Clone)]
pub enum TriggeredBy {
    Consume {
        id: Blake2b512Random,
        persistent: bool,
        channels_count: i64,
    },
    Produce {
        id: Blake2b512Random,
        persistent: bool,
        channels_count: i64,
    },
}

fn consume_id(continuation: TaggedContinuation) -> Result<Blake2b512Random, InterpreterError> {
    //TODO: Make ScalaBodyRef-s have their own random state and merge it during its COMMs - OLD
    match continuation.tagged_cont.unwrap() {
        TaggedCont::ParBody(par_with_random) => Ok(Blake2b512Random::create_from_bytes(
            &par_with_random.random_state,
        )),
        TaggedCont::ScalaBodyRef(value) => {
            Ok(Blake2b512Random::create_from_bytes(&value.to_be_bytes()))
        }
    }
}

impl ChargingRSpace {
    pub fn charging_rspace<T>(
        space: T,
        cost: _cost,
    ) -> impl ISpace<Par, BindPattern, ListParWithRandom, TaggedContinuation> + Clone
    where
        T: ISpace<Par, BindPattern, ListParWithRandom, TaggedContinuation> + Clone,
    {
        #[derive(Clone)]
        struct ChargingRSpace<T> {
            space: T,
            cost: _cost,
        }

        impl<T: ISpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>
            ISpace<Par, BindPattern, ListParWithRandom, TaggedContinuation> for ChargingRSpace<T>
        {
            fn consume(
                &self,
                channels: Vec<Par>,
                patterns: Vec<BindPattern>,
                continuation: TaggedContinuation,
                persist: bool,
                peeks: BTreeSet<i32>,
            ) -> Result<
                MaybeConsumeResult<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
                RSpaceError,
            > {
                let seq = next_cost_trace_seq();
                let cost_before = self.cost.get().value;
                let upfront = storage_cost_consume(
                    channels.clone(),
                    patterns.clone(),
                    continuation.clone(),
                );
                self.cost.charge(upfront.clone())?;

                let consume_res = self.space.consume(
                    channels.clone(),
                    patterns,
                    continuation.clone(),
                    persist,
                    peeks,
                )?;

                let comm_fired = consume_res.is_some();
                // Normalize: when a COMM fires from the consume side, report it
                // as produce-triggered so cost accounting is deterministic
                // regardless of evaluation order. This allows concurrent
                // evaluation (join_all) without COST_MISMATCH between validator
                // and observer. When no COMM fires, use consume-triggered
                // semantics (the consume just stored its continuation).
                let triggered_by = if comm_fired {
                    let (_, data_list) = consume_res.as_ref().expect("comm_fired is true");
                    let first_data = data_list.first().expect("COMM must have at least one produce");
                    TriggeredBy::Produce {
                        id: Blake2b512Random::create_from_bytes(&first_data.removed_datum.random_state),
                        persistent: first_data.persistent,
                        channels_count: 1,
                    }
                } else {
                    let id = consume_id(continuation)?;
                    TriggeredBy::Consume {
                        id,
                        persistent: persist,
                        channels_count: channels.len() as i64,
                    }
                };
                handle_result(
                    consume_res.clone(),
                    triggered_by,
                    self.cost.clone(),
                )?;
                let cost_after = self.cost.get().value;
                // Diagnostic 1: compute channel hashes for cross-node comparison
                let channels_hash_str = if tracing::enabled!(target: "f1r3fly.rspace.cost_trace", tracing::Level::INFO) {
                    channels.iter().map(|ch| {
                        let bytes = bincode::serialize(ch).unwrap_or_default();
                        let hash = Blake2b256Hash::new(&bytes);
                        hex::encode(&hash.bytes()[..8])
                    }).collect::<Vec<_>>().join(",")
                } else {
                    String::new()
                };
                tracing::info!(
                    target: "f1r3fly.rspace.cost_trace",
                    seq,
                    op = "consume",
                    channels_hash = %channels_hash_str,
                    upfront_charge = upfront.value,
                    comm_fired,
                    persist,
                    channels_count = channels.len(),
                    cost_before,
                    cost_after,
                    net_delta = cost_after - cost_before,
                                        "COST_TRACE_OP: seq={} op=consume ch=[{}] comm={} persist={} delta={} total={}",
                    seq, channels_hash_str, comm_fired, persist, cost_after - cost_before, cost_after
                );
                Ok(consume_res)
            }

            fn produce(
                &self,
                channel: Par,
                data: ListParWithRandom,
                persist: bool,
            ) -> Result<
                MaybeProduceResult<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
                RSpaceError,
            > {
                let seq = next_cost_trace_seq();
                let cost_before = self.cost.get().value;
                let upfront = storage_cost_produce(channel.clone(), data.clone());
                self.cost.charge(upfront.clone())?;
                // Diagnostic 1: compute channel hash before move
                let channel_hash_str = if tracing::enabled!(target: "f1r3fly.rspace.cost_trace", tracing::Level::INFO) {
                    let bytes = bincode::serialize(&channel).unwrap_or_default();
                    let hash = Blake2b256Hash::new(&bytes);
                    hex::encode(&hash.bytes()[..8])
                } else {
                    String::new()
                };
                let produce_res = self.space.produce(channel, data.clone(), persist)?;
                let comm_fired = produce_res.is_some();
                let common_result = produce_res
                    .clone()
                    .map(|(cont, data_list, _)| (cont, data_list));
                handle_result(
                    common_result,
                    TriggeredBy::Produce {
                        id: Blake2b512Random::create_from_bytes(&data.random_state),
                        persistent: persist,
                        channels_count: 1,
                    },
                    self.cost.clone(),
                )?;
                let cost_after = self.cost.get().value;
                let rand_hex = hex::encode(&data.random_state.iter().take(16).copied().collect::<Vec<u8>>());
                let data_rand_hash = {
                    hex::encode(Blake2b256Hash::new(&data.random_state).bytes())
                };
                tracing::info!(
                    target: "f1r3fly.rspace.cost_trace",
                    seq,
                    op = "produce",
                    channel_hash = %channel_hash_str,
                    upfront_charge = upfront.value,
                    comm_fired,
                    persist,
                    cost_before,
                    cost_after,
                    net_delta = cost_after - cost_before,
                    rand_state = %rand_hex,
                    data_rand_hash = %data_rand_hash,
                    data_rand_len = data.random_state.len(),
                                        "COST_TRACE_OP: seq={} op=produce ch={} comm={} persist={} rand_hash={} delta={} total={}",
                    seq, channel_hash_str, comm_fired, persist, &data_rand_hash[..16], cost_after - cost_before, cost_after
                );
                Ok(produce_res)
            }

            fn install(
                &self,
                channels: Vec<Par>,
                patterns: Vec<BindPattern>,
                continuation: TaggedContinuation,
            ) -> Result<Option<(TaggedContinuation, Vec<ListParWithRandom>)>, RSpaceError>
            {
                self.space.install(channels, patterns, continuation)
            }

            fn create_checkpoint(&self) -> Result<Checkpoint, RSpaceError> {
                self.space.create_checkpoint()
            }

            fn get_data(&self, channel: &Par) -> Vec<Datum<ListParWithRandom>> {
                self.space.get_data(channel)
            }

            fn get_waiting_continuations(
                &self,
                channels: Vec<Par>,
            ) -> Vec<WaitingContinuation<BindPattern, TaggedContinuation>> {
                self.space.get_waiting_continuations(channels)
            }

            fn get_joins(&self, channel: Par) -> Vec<Vec<Par>> {
                self.space.get_joins(channel)
            }

            fn clear(&self) -> Result<(), RSpaceError> {
                self.space.clear()
            }

            fn get_root(&self) -> Blake2b256Hash {
                self.space.get_root()
            }

            fn reset(&self, root: &Blake2b256Hash) -> Result<(), RSpaceError> {
                self.space.reset(root)
            }

            fn consume_result(
                &self,
                channel: Vec<Par>,
                pattern: Vec<BindPattern>,
            ) -> Result<Option<(TaggedContinuation, Vec<ListParWithRandom>)>, RSpaceError>
            {
                let consume_res = self.space.consume(
                    channel,
                    pattern,
                    TaggedContinuation::default(),
                    false,
                    BTreeSet::new(),
                )?;
                Ok(unpack_option(&consume_res))
            }

            fn to_map(
                &self,
            ) -> HashMap<Vec<Par>, Row<BindPattern, ListParWithRandom, TaggedContinuation>>
            {
                self.space.to_map()
            }

            fn create_soft_checkpoint(
                &self,
            ) -> SoftCheckpoint<Par, BindPattern, ListParWithRandom, TaggedContinuation>
            {
                self.space.create_soft_checkpoint()
            }

            fn take_event_log(&self) -> Log {
                self.space.take_event_log()
            }

            fn revert_to_soft_checkpoint(
                &self,
                checkpoint: SoftCheckpoint<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
            ) -> Result<(), RSpaceError> {
                self.space.revert_to_soft_checkpoint(checkpoint)
            }

            fn rig_and_reset(
                &self,
                start_root: Blake2b256Hash,
                log: Log,
            ) -> Result<(), RSpaceError> {
                self.space.rig_and_reset(start_root, log)
            }

            fn rig(&self, log: Log) -> Result<(), RSpaceError> {
                self.space.rig(log)
            }

            fn check_replay_data(&self) -> Result<(), RSpaceError> {
                self.space.check_replay_data()
            }

            fn is_replay(&self) -> bool {
                self.space.is_replay()
            }

            fn update_produce(&self, produce: Produce) -> () {
                self.space.update_produce(produce)
            }

            fn pending_state_counts(&self) -> (usize, usize, usize, usize) {
                self.space.pending_state_counts()
            }

            fn pending_continuation_channels_debug(&self) -> Vec<(String, usize, bool)> {
                self.space.pending_continuation_channels_debug()
            }
        }

        ChargingRSpace { space, cost }
    }
}

fn handle_result(
    result: MaybeConsumeResult<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
    triggered_by: TriggeredBy,
    cost: _cost,
) -> Result<(), InterpreterError> {
    let triggered_by_id = match triggered_by.clone() {
        TriggeredBy::Consume { id, .. } => id,
        TriggeredBy::Produce { id, .. } => id,
    };
    let triggered_by_channels_count = match triggered_by {
        TriggeredBy::Consume { channels_count, .. } => channels_count,
        TriggeredBy::Produce { .. } => 1,
    };
    let triggered_by_persistent = match triggered_by {
        TriggeredBy::Consume { persistent, .. } => persistent,
        TriggeredBy::Produce { persistent, .. } => persistent,
    };
    let triggered_by_id_bytes = triggered_by_id.to_bytes();

    match result {
        Some((cont, data_list)) => {
            let consume_id = consume_id(cont.continuation.clone())?;

            // We refund for non-persistent continuations, and for the persistent continuation triggering the comm.
            // That persistent continuation is going to be charged for (without refund) once it has no matches in TS.
            let consume_id_bytes = consume_id.to_bytes();
            let consume_refund_applies =
                !cont.persistent || consume_id_bytes == triggered_by_id_bytes;
            let refund_for_consume = if consume_refund_applies {
                storage_cost_consume(
                    cont.channels.clone(),
                    cont.patterns.clone(),
                    cont.continuation.clone(),
                )
            } else {
                Cost::create(0, "refund_for_consume")
            };

            let refund_for_produces =
                refund_for_removing_produces(data_list.len(), data_list, cont.clone(), triggered_by);

            let last_iteration = !triggered_by_persistent;
            let event_cost = if last_iteration {
                event_storage_cost(triggered_by_channels_count).value
            } else {
                0
            };
            let comm_cost = comm_event_storage_cost(cont.channels.len() as i64).value;

            tracing::info!(
                target: "f1r3fly.rspace.cost_trace",
                consume_refund = refund_for_consume.value,
                consume_refund_applies,
                produce_refund = refund_for_produces.value,
                cont_persistent = cont.persistent,
                triggered_by_persistent,
                last_iteration,
                event_cost,
                comm_cost,
                data_count = cont.channels.len(),
                "COST_TRACE_COMM: consume_refund={} produce_refund={} event={} comm={} cont_persist={} trig_persist={} last_iter={}",
                refund_for_consume.value, refund_for_produces.value,
                event_cost, comm_cost, cont.persistent, triggered_by_persistent, last_iteration
            );

            cost.charge(Cost::create(
                -refund_for_consume.value,
                "consume storage refund",
            ))?;
            cost.charge(Cost::create(
                -refund_for_produces.value,
                "produces storage refund",
            ))?;

            if last_iteration {
                cost.charge(event_storage_cost(triggered_by_channels_count))?;
            }

            cost.charge(comm_event_storage_cost(cont.channels.len() as i64))
        }
        None => cost.charge(event_storage_cost(triggered_by_channels_count)),
    }
}

fn refund_for_removing_produces(
    total_data_count: usize,
    data_list: Vec<RSpaceResult<Par, ListParWithRandom>>,
    cont: ContResult<Par, BindPattern, TaggedContinuation>,
    triggered_by: TriggeredBy,
) -> Cost {
    let triggered_id = match triggered_by {
        TriggeredBy::Consume { id, .. } => id,
        TriggeredBy::Produce { id, .. } => id,
    };
    let triggered_id_bytes = triggered_id.to_bytes();

    let removed_data: Vec<(RSpaceResult<Par, ListParWithRandom>, Par)> = data_list
        .into_iter()
        .zip(cont.channels.into_iter())
        .enumerate()
        // A persistent produce is charged for upfront before reaching the TS, and needs to be refunded
        // after each iteration it matches an existing consume. We treat it as 'removed' on each such iteration.
        // It is going to be 'not removed' and charged for on the last iteration, where it doesn't match anything.
        .filter(|(i, (data, _))| {
            let random_state_matches = data.removed_datum.random_state == triggered_id_bytes;
            let passes = !data.persistent || random_state_matches;
            tracing::info!(
                target: "f1r3fly.rspace.cost_trace",
                idx = i,
                persistent = data.persistent,
                random_state_matches,
                passes,
                random_state_hex = %hex::encode(&data.removed_datum.random_state),
                triggered_id_hex = %hex::encode(&triggered_id_bytes),
                "COST_TRACE_REFUND_FILTER: idx={} persistent={} rs_match={} passes={}",
                i, data.persistent, random_state_matches, passes
            );
            passes
        })
        .map(|(_, pair)| pair)
        .collect();

    let removed_count = removed_data.len();
    let result = removed_data
        .into_iter()
        .map(|(data, channel)| storage_cost_produce(channel, data.removed_datum))
        .fold(
            Cost::create(0, "refund_for_removing_produces init"),
            |acc, cost| {
                Cost::create(
                    acc.value + cost.value,
                    "refund_for_removing_produces operation",
                )
            },
        );

    tracing::info!(
        target: "f1r3fly.rspace.cost_trace",
        total_data_count,
        removed_count,
        total_refund = result.value,
        "COST_TRACE_REFUND_TOTAL: {}/{} items refunded, total={}",
        removed_count, total_data_count, result.value
    );

    result
}
