// See rholang/src/main/scala/coop/rchain/rholang/interpreter/storage/ChargingRSpace.scala

use std::collections::{BTreeSet, HashMap};

use async_trait::async_trait;

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

#[derive(Clone)]
pub enum TriggeredBy {
    Consume {
        id: Vec<u8>,
        persistent: bool,
        channels_count: i64,
    },
    Produce {
        id: Vec<u8>,
        persistent: bool,
        channels_count: i64,
    },
}

//TODO: Make ScalaBodyRef-s have their own random state and merge it during its COMMs - OLD
fn consume_id_bytes(continuation: &TaggedContinuation) -> Result<Vec<u8>, InterpreterError> {
    match continuation.tagged_cont.as_ref().unwrap() {
        TaggedCont::ParBody(par_with_random) => Ok(par_with_random.random_state.clone()),
        TaggedCont::ScalaBodyRef(value) => Ok(value.to_be_bytes().to_vec()),
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

        #[async_trait]
        impl<T: ISpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>
            ISpace<Par, BindPattern, ListParWithRandom, TaggedContinuation> for ChargingRSpace<T>
        {
            async fn consume(
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
                self.cost.charge(storage_cost_consume(
                    channels.clone(),
                    patterns.clone(),
                    continuation.clone(),
                ))?;

                let consume_res = self.space.consume(
                    channels.clone(),
                    patterns,
                    continuation.clone(),
                    persist,
                    peeks,
                ).await?;

                let id = consume_id_bytes(&continuation)?;
                handle_result(
                    consume_res.clone(),
                    TriggeredBy::Consume {
                        id,
                        persistent: persist,
                        channels_count: channels.len() as i64,
                    },
                    self.cost.clone(),
                )?;
                Ok(consume_res)
            }

            async fn produce(
                &self,
                channel: Par,
                data: ListParWithRandom,
                persist: bool,
            ) -> Result<
                MaybeProduceResult<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
                RSpaceError,
            > {
                self.cost
                    .charge(storage_cost_produce(channel.clone(), data.clone()))?;
                let produce_res = self.space.produce(channel, data.clone(), persist).await?;
                let common_result = produce_res
                    .clone()
                    .map(|(cont, data_list, _)| (cont, data_list));
                handle_result(
                    common_result,
                    TriggeredBy::Produce {
                        id: data.random_state.clone(),
                        persistent: persist,
                        channels_count: 1,
                    },
                    self.cost.clone(),
                )?;
                Ok(produce_res)
            }

            async fn install(
                &self,
                channels: Vec<Par>,
                patterns: Vec<BindPattern>,
                continuation: TaggedContinuation,
            ) -> Result<Option<(TaggedContinuation, Vec<ListParWithRandom>)>, RSpaceError>
            {
                self.space.install(channels, patterns, continuation).await
            }

            async fn create_checkpoint(&self) -> Result<Checkpoint, RSpaceError> {
                self.space.create_checkpoint().await
            }

            async fn get_data(&self, channel: &Par) -> Vec<Datum<ListParWithRandom>> {
                self.space.get_data(channel).await
            }

            async fn get_waiting_continuations(
                &self,
                channels: Vec<Par>,
            ) -> Vec<WaitingContinuation<BindPattern, TaggedContinuation>> {
                self.space.get_waiting_continuations(channels).await
            }

            async fn get_joins(&self, channel: Par) -> Vec<Vec<Par>> {
                self.space.get_joins(channel).await
            }

            async fn clear(&self) -> Result<(), RSpaceError> {
                self.space.clear().await
            }

            async fn get_root(&self) -> Blake2b256Hash {
                self.space.get_root().await
            }

            async fn reset(&self, root: &Blake2b256Hash) -> Result<(), RSpaceError> {
                self.space.reset(root).await
            }

            async fn consume_result(
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
                ).await?;
                Ok(unpack_option(&consume_res))
            }

            async fn to_map(
                &self,
            ) -> HashMap<Vec<Par>, Row<BindPattern, ListParWithRandom, TaggedContinuation>>
            {
                self.space.to_map().await
            }

            async fn create_soft_checkpoint(
                &self,
            ) -> SoftCheckpoint<Par, BindPattern, ListParWithRandom, TaggedContinuation>
            {
                self.space.create_soft_checkpoint().await
            }

            async fn take_event_log(&self) -> Log {
                self.space.take_event_log().await
            }

            async fn revert_to_soft_checkpoint(
                &self,
                checkpoint: SoftCheckpoint<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
            ) -> Result<(), RSpaceError> {
                self.space.revert_to_soft_checkpoint(checkpoint).await
            }

            async fn rig_and_reset(
                &self,
                start_root: Blake2b256Hash,
                log: Log,
            ) -> Result<(), RSpaceError> {
                self.space.rig_and_reset(start_root, log).await
            }

            async fn rig(&self, log: Log) -> Result<(), RSpaceError> {
                self.space.rig(log).await
            }

            async fn check_replay_data(&self) -> Result<(), RSpaceError> {
                self.space.check_replay_data().await
            }

            async fn is_replay(&self) -> bool {
                self.space.is_replay().await
            }

            async fn update_produce(&self, produce: Produce) -> () {
                self.space.update_produce(produce).await
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
    let triggered_by_id_bytes = match triggered_by.clone() {
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

    match result {
        Some((cont, data_list)) => {
            let consume_id_bytes = consume_id_bytes(&cont.continuation)?;

            // We refund for non-persistent continuations, and for the persistent continuation triggering the comm.
            // That persistent continuation is going to be charged for (without refund) once it has no matches in TS.
            let refund_for_consume =
                if !cont.persistent || consume_id_bytes == triggered_by_id_bytes {
                    storage_cost_consume(
                        cont.channels.clone(),
                        cont.patterns.clone(),
                        cont.continuation.clone(),
                    )
                } else {
                    Cost::create(0, "refund_for_consume")
                };

            let refund_for_produces =
                refund_for_removing_produces(data_list, cont.clone(), triggered_by);

            cost.charge(Cost::create(
                -refund_for_consume.value,
                "consume storage refund",
            ))?;
            cost.charge(Cost::create(
                -refund_for_produces.value,
                "produces storage refund",
            ))?;

            let last_iteration = !triggered_by_persistent;

            if last_iteration {
                cost.charge(event_storage_cost(triggered_by_channels_count))?;
            }

            cost.charge(comm_event_storage_cost(cont.channels.len() as i64))
        }
        None => cost.charge(event_storage_cost(triggered_by_channels_count)),
    }
}

fn refund_for_removing_produces(
    data_list: Vec<RSpaceResult<Par, ListParWithRandom>>,
    cont: ContResult<Par, BindPattern, TaggedContinuation>,
    triggered_by: TriggeredBy,
) -> Cost {
    let triggered_id_bytes = match triggered_by {
        TriggeredBy::Consume { id, .. } => id,
        TriggeredBy::Produce { id, .. } => id,
    };

    let removed_data: Vec<(RSpaceResult<Par, ListParWithRandom>, Par)> = data_list
        .into_iter()
        .zip(cont.channels.into_iter())
        // A persistent produce is charged for upfront before reaching the TS, and needs to be refunded
        // after each iteration it matches an existing consume. We treat it as 'removed' on each such iteration.
        // It is going to be 'not removed' and charged for on the last iteration, where it doesn't match anything.
        .filter(|(data, _)| {
            !data.persistent || data.removed_datum.random_state == triggered_id_bytes
        })
        .collect();

    removed_data
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
        )
}
