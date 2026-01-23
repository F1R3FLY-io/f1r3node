// See rspace/src/main/scala/coop/rchain/rspace/merger/StateChange.scala

use dashmap::DashMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use crate::rspace::{
    errors::HistoryError,
    hashing::{blake2b256_hash::Blake2b256Hash, stable_hash_provider},
    history::{
        history_reader::HistoryReader,
        instances::rspace_history_reader_impl::RSpaceHistoryReaderImpl,
    },
};

use super::{
    channel_change::ChannelChange,
    event_log_index::EventLogIndex,
    merging_logic::{consumes_affected, produces_affected},
};

/**
 * Datum changes are referenced by channel, continuation changes are references by consume.
 * In addition, map from consume channels to binary representation of a join in trie have to be maintained.
 * This is because only hashes of channels are available in log event, and computing a join binary to be
 * inserted or removed on merge requires channels before hashing.
 */
#[derive(Debug, Clone)]
pub struct StateChange {
    pub datums_changes: DashMap<Blake2b256Hash, ChannelChange<Vec<u8>>>,
    pub cont_changes: DashMap<Vec<Blake2b256Hash>, ChannelChange<Vec<u8>>>,
    pub consume_channels_to_join_serialized_map: DashMap<Vec<Blake2b256Hash>, Vec<u8>>,
}

impl PartialEq for StateChange {
    fn eq(&self, other: &Self) -> bool {
        // Compare by counting entries and checking if all keys and values match
        // This is an approximate equality check since DashMap doesn't implement PartialEq

        // Check if maps have same size
        if self.datums_changes.len() != other.datums_changes.len()
            || self.cont_changes.len() != other.cont_changes.len()
            || self.consume_channels_to_join_serialized_map.len()
                != other.consume_channels_to_join_serialized_map.len()
        {
            return false;
        }

        // Check all datums_changes match
        for entry in self.datums_changes.iter() {
            let key = entry.key();
            let value = entry.value();

            if let Some(other_value) = other.datums_changes.get(key) {
                if value.added != other_value.added || value.removed != other_value.removed {
                    return false;
                }
            } else {
                return false;
            }
        }

        // Check all cont_changes match
        for entry in self.cont_changes.iter() {
            let key = entry.key();
            let value = entry.value();

            // Find matching key in other.cont_changes
            let found = other.cont_changes.iter().any(|other_entry| {
                let other_key = other_entry.key();
                let other_value = other_entry.value();

                key == other_key
                    && value.added == other_value.added
                    && value.removed == other_value.removed
            });

            if !found {
                return false;
            }
        }

        // Check all join maps match
        for entry in self.consume_channels_to_join_serialized_map.iter() {
            let key = entry.key();
            let value = entry.value();

            // Find matching key in other.consume_channels_to_join_serialized_map
            let found = other
                .consume_channels_to_join_serialized_map
                .iter()
                .any(|other_entry| {
                    let other_key = other_entry.key();
                    let other_value = other_entry.value();

                    key == other_key && value == other_value
                });

            if !found {
                return false;
            }
        }

        true
    }
}

impl PartialOrd for StateChange {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Implement a custom ordering based on sizes
        // This is a simple implementation that treats StateChange as comparable
        // For a real implementation, you'd need to define a sensible ordering

        // Compare by the number of entries in each map
        let self_total = self.datums_changes.len()
            + self.cont_changes.len()
            + self.consume_channels_to_join_serialized_map.len();
        let other_total = other.datums_changes.len()
            + other.cont_changes.len()
            + other.consume_channels_to_join_serialized_map.len();

        self_total.partial_cmp(&other_total)
    }
}

impl Hash for StateChange {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash datums_changes
        let mut datum_keys_and_values = Vec::new();
        for entry in self.datums_changes.iter() {
            let key = entry.key().clone();
            let value = entry.value();

            let mut key_hasher = std::collections::hash_map::DefaultHasher::new();
            key.hash(&mut key_hasher);
            let key_hash = key_hasher.finish();

            let added = value.added.clone();
            let removed = value.removed.clone();

            datum_keys_and_values.push((key_hash, added, removed));
        }

        // Sort for deterministic hashing
        datum_keys_and_values.sort_by_key(|k| k.0);
        for (key_hash, added, removed) in datum_keys_and_values {
            key_hash.hash(state);

            // Hash added values
            let mut added_hashes = Vec::new();
            for item in added {
                let mut item_hasher = std::collections::hash_map::DefaultHasher::new();
                item.hash(&mut item_hasher);
                added_hashes.push(item_hasher.finish());
            }
            added_hashes.sort_unstable();
            for h in added_hashes {
                h.hash(state);
            }

            // Hash removed values
            let mut removed_hashes = Vec::new();
            for item in removed {
                let mut item_hasher = std::collections::hash_map::DefaultHasher::new();
                item.hash(&mut item_hasher);
                removed_hashes.push(item_hasher.finish());
            }
            removed_hashes.sort_unstable();
            for h in removed_hashes {
                h.hash(state);
            }
        }

        // Hash cont_changes
        let mut cont_keys_and_values = Vec::new();
        for entry in self.cont_changes.iter() {
            let key = entry.key().clone();
            let value = entry.value();

            // Hash the collection of Blake2b256Hash
            let mut key_hasher = std::collections::hash_map::DefaultHasher::new();
            for hash in &key {
                hash.hash(&mut key_hasher);
            }
            let key_hash = key_hasher.finish();

            let added = value.added.clone();
            let removed = value.removed.clone();

            cont_keys_and_values.push((key_hash, added, removed));
        }

        // Sort for deterministic hashing
        cont_keys_and_values.sort_by_key(|k| k.0);
        for (key_hash, added, removed) in cont_keys_and_values {
            key_hash.hash(state);

            // Hash added values
            let mut added_hashes = Vec::new();
            for item in added {
                let mut item_hasher = std::collections::hash_map::DefaultHasher::new();
                item.hash(&mut item_hasher);
                added_hashes.push(item_hasher.finish());
            }
            added_hashes.sort_unstable();
            for h in added_hashes {
                h.hash(state);
            }

            // Hash removed values
            let mut removed_hashes = Vec::new();
            for item in removed {
                let mut item_hasher = std::collections::hash_map::DefaultHasher::new();
                item.hash(&mut item_hasher);
                removed_hashes.push(item_hasher.finish());
            }
            removed_hashes.sort_unstable();
            for h in removed_hashes {
                h.hash(state);
            }
        }

        // Hash consume_channels_to_join_serialized_map
        let mut join_keys_and_values = Vec::new();
        for entry in self.consume_channels_to_join_serialized_map.iter() {
            let key = entry.key().clone();
            let value = entry.value().clone();

            // Hash the collection of Blake2b256Hash
            let mut key_hasher = std::collections::hash_map::DefaultHasher::new();
            for hash in &key {
                hash.hash(&mut key_hasher);
            }
            let key_hash = key_hasher.finish();

            let mut value_hasher = std::collections::hash_map::DefaultHasher::new();
            value.hash(&mut value_hasher);
            let value_hash = value_hasher.finish();

            join_keys_and_values.push((key_hash, value_hash));
        }

        // Sort for deterministic hashing
        join_keys_and_values.sort_by_key(|k| k.0);
        for (key_hash, value_hash) in join_keys_and_values {
            key_hash.hash(state);
            value_hash.hash(state);
        }
    }
}

impl Eq for StateChange {}

impl Ord for StateChange {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Implement total ordering consistent with partial_cmp
        // Compare by the number of entries in each map
        let self_total = self.datums_changes.len()
            + self.cont_changes.len()
            + self.consume_channels_to_join_serialized_map.len();
        let other_total = other.datums_changes.len()
            + other.cont_changes.len()
            + other.consume_channels_to_join_serialized_map.len();

        self_total.cmp(&other_total)
    }
}

impl StateChange {
    pub fn new<C, P, A, K>(
        pre_state_reader: RSpaceHistoryReaderImpl<C, P, A, K>,
        post_state_reader: RSpaceHistoryReaderImpl<C, P, A, K>,
        event_log_index: &EventLogIndex,
    ) -> Result<Self, HistoryError>
    where
        C: Clone + for<'a> Deserialize<'a> + Serialize + 'static + Sync + Send,
        P: Clone + for<'a> Deserialize<'a> + 'static + Sync + Send,
        A: Clone + for<'a> Deserialize<'a> + 'static + Sync + Send,
        K: Clone + for<'a> Deserialize<'a> + 'static + Sync + Send,
    {
        let datums_diff = DashMap::new();
        let cont_diff = DashMap::new();
        // Since event log only contains hashes of channels, so to know which join stored corresponds to channels,
        // this index have to be maintained
        let joins_map = DashMap::new();

        let produces_affected = produces_affected(&event_log_index);

        // Deduplicate by channel hash before processing
        // Without this, if multiple Produces have the same channel_hash, we'd compute changes multiple times
        let unique_produce_channels: HashSet<Blake2b256Hash> = produces_affected
            .0
            .iter()
            .map(|produce| produce.channel_hash.clone())
            .collect();

        let channels_of_consumes_affected = consumes_affected(&event_log_index)
            .0
            .into_iter()
            .map(|consume| consume.channel_hashes)
            .collect::<Vec<_>>();

        // Deduplicate consume channels as well
        let unique_consume_channels: HashSet<Vec<Blake2b256Hash>> =
            channels_of_consumes_affected.into_iter().collect();
        let channels_of_consumes_affected: Vec<Vec<Blake2b256Hash>> =
            unique_consume_channels.into_iter().collect();

        // Process produces in parallel - each unique channel is processed exactly once
        unique_produce_channels
            .par_iter()
            .try_for_each(|history_pointer| {
                let change = Self::compute_value_change(
                    history_pointer,
                    |h| pre_state_reader.get_data_proj_binary(h),
                    |h| post_state_reader.get_data_proj_binary(h),
                )?;

                // Since each channel is processed exactly once, we can just insert directly
                datums_diff.insert(history_pointer.clone(), change);

                Ok::<(), HistoryError>(())
            })?;

        // Process consumes in parallel - each unique consume channels set is processed exactly once
        channels_of_consumes_affected
            .par_iter()
            .try_for_each(|consume_channels| {
                let consume_channels = consume_channels.clone();
                let history_pointer = stable_hash_provider::hash_from_hashes(&consume_channels);

                let change = Self::compute_value_change(
                    &history_pointer,
                    |h| pre_state_reader.get_continuations_proj_binary(h),
                    |h| post_state_reader.get_continuations_proj_binary(h),
                )?;

                // Since each consume channels set is processed exactly once, we can just insert directly
                cont_diff.insert(consume_channels, change);

                Ok::<(), HistoryError>(())
            })?;

        // Process joins in parallel
        channels_of_consumes_affected.par_iter().try_for_each(|consume_channels| {
            let mut consume_channels = consume_channels.clone();
            let history_pointer = consume_channels[0].clone();
            let pre = pre_state_reader.get_joins(&history_pointer)?;
            let post = post_state_reader.get_joins(&history_pointer)?;

            // find join which match channels
            let join = pre
                .into_iter()
                .chain(post)
                .find(|join| {
                    let mut join_channels = join
                        .iter()
                        .map(|item| stable_hash_provider::hash(item))
                        .collect::<Vec<_>>();
                    // sorting is required because channels of a consume in event log and channels of a join in
                    // history might not be ordered the same way
                    consume_channels.sort();
                    join_channels.sort();
                    *consume_channels == join_channels
                })
                .expect("Tuple space inconsistency found: channel of consume does not contain join record corresponding to the consume channels.");

            let raw_join = bincode::serialize(&join)
                .expect("Unable to serialize join");
            joins_map.insert(consume_channels, raw_join);
            Ok::<(), HistoryError>(())
        })?;

        // Check for errors - empty changes
        let has_empty_produce_changes = datums_diff.iter().any(|entry| {
            let change = entry.value();
            change.added.is_empty() && change.removed.is_empty()
        });

        if has_empty_produce_changes {
            return Err(HistoryError::MergeError(
                "State change compute logic error: empty channel change for produce.".to_string(),
            ));
        }

        let has_empty_consume_changes = cont_diff.iter().any(|entry| {
            let change = entry.value();
            change.added.is_empty() && change.removed.is_empty()
        });

        if has_empty_consume_changes {
            return Err(HistoryError::MergeError(
                "State change compute logic error: empty channel change for consume.".to_string(),
            ));
        }

        Ok(Self {
            datums_changes: datums_diff,
            cont_changes: cont_diff,
            consume_channels_to_join_serialized_map: joins_map,
        })
    }

    /// Compute multiset difference.
    /// Removes each element of `to_remove` from `from` exactly once.
    fn multiset_diff(from: &[Vec<u8>], to_remove: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let mut result = from.to_vec();
        for item in to_remove {
            if let Some(pos) = result.iter().position(|x| x == item) {
                result.remove(pos);
            }
        }
        result
    }

    fn compute_value_change(
        history_pointer: &Blake2b256Hash,
        start_value: impl Fn(&Blake2b256Hash) -> Result<Vec<Vec<u8>>, HistoryError>,
        end_value: impl Fn(&Blake2b256Hash) -> Result<Vec<Vec<u8>>, HistoryError>,
    ) -> Result<ChannelChange<Vec<u8>>, HistoryError> {
        let start = start_value(history_pointer)?;
        let end = end_value(history_pointer)?;

        // Use multiset diff for correct merge semantics
        // added = endValue diff startValue (items in end that aren't in start, multiset)
        let added = Self::multiset_diff(&end, &start);

        // deleted = startValue diff endValue (items in start that aren't in end, multiset)
        let deleted = Self::multiset_diff(&start, &end);

        Ok(ChannelChange {
            added,
            removed: deleted,
        })
    }

    pub fn empty() -> Self {
        Self {
            datums_changes: DashMap::new(),
            cont_changes: DashMap::new(),
            consume_channels_to_join_serialized_map: DashMap::new(),
        }
    }

    pub fn combine(self, other: Self) -> Self {
        let datums_changes = self.datums_changes;
        let cont_changes = self.cont_changes;
        let consume_channels_to_join_serialized_map = self.consume_channels_to_join_serialized_map;

        // Combine datum changes
        for (key, value) in other.datums_changes {
            match datums_changes.entry(key) {
                dashmap::mapref::entry::Entry::Occupied(mut entry) => {
                    let current = entry.get_mut();
                    current.added.extend(value.added);
                    current.removed.extend(value.removed);
                }
                dashmap::mapref::entry::Entry::Vacant(entry) => {
                    entry.insert(value);
                }
            }
        }

        // Combine continuation changes
        for (key, value) in other.cont_changes {
            match cont_changes.entry(key) {
                dashmap::mapref::entry::Entry::Occupied(mut entry) => {
                    let current = entry.get_mut();
                    current.added.extend(value.added);
                    current.removed.extend(value.removed);
                }
                dashmap::mapref::entry::Entry::Vacant(entry) => {
                    entry.insert(value);
                }
            }
        }

        // Combine join maps (newer values take precedence)
        for (key, value) in other.consume_channels_to_join_serialized_map {
            consume_channels_to_join_serialized_map.insert(key, value);
        }

        Self {
            datums_changes,
            cont_changes,
            consume_channels_to_join_serialized_map,
        }
    }
}
