// See rspace/src/main/scala/coop/rchain/rspace/merger/StateChangeMerger.scala

use shared::rust::ByteVector;

use crate::rspace::{
    errors::HistoryError,
    hashing::{blake2b256_hash::Blake2b256Hash, stable_hash_provider},
    history::history_reader::HistoryReader,
    hot_store_trie_action::{
        HotStoreTrieAction, TrieDeleteAction, TrieDeleteConsume, TrieDeleteJoins,
        TrieDeleteProduce, TrieInsertAction, TrieInsertBinaryConsume, TrieInsertBinaryJoins,
        TrieInsertBinaryProduce,
    },
};

use super::{
    channel_change::ChannelChange, merging_logic::NumberChannelsDiff, state_change::StateChange,
};

/**
 * This classes are used to compute joins.
 * Consume value pointer that stores continuations on some channel is identified by channels involved in.
 * Therefore when no continuations on some consume is left and the whole consume ponter is removed -
 * no joins with corresponding seq of channels exist in tuple space. So join should be removed.
 * */
pub enum JoinActionKind {
    AddJoin(Vec<Blake2b256Hash>),
    RemoveJoin(Vec<Blake2b256Hash>),
}

pub struct ConsumeAndJoinActions<C: Clone, P: Clone, A: Clone, K: Clone> {
    consume_action: HotStoreTrieAction<C, P, A, K>,
    join_action: Option<JoinActionKind>,
}

pub fn compute_trie_actions<C: Clone, P: Clone, A: Clone, K: Clone>(
    changes: &StateChange,
    base_reader: &Box<dyn HistoryReader<Blake2b256Hash, C, P, A, K>>,
    mergeable_chs: &NumberChannelsDiff,
    handle_channel_change: impl Fn(
        &Blake2b256Hash,
        &ChannelChange<Vec<u8>>,
        &NumberChannelsDiff,
    ) -> Result<Option<HotStoreTrieAction<C, P, A, K>>, HistoryError>,
) -> Result<Vec<HotStoreTrieAction<C, P, A, K>>, HistoryError> {
    // Sort continuation changes by hash of consume channels for deterministic ordering
    let mut cont_changes_sorted: Vec<_> = changes
        .cont_changes
        .iter()
        .map(|ref_multi| (ref_multi.key().clone(), ref_multi.value().clone()))
        .collect();
    cont_changes_sorted.sort_by_key(|(consume_channels, _)| stable_hash_provider::hash_from_hashes(consume_channels));

    let consume_with_join_actions: Vec<ConsumeAndJoinActions<C, P, A, K>> = cont_changes_sorted
        .iter()
        .map(|(consume_channels, channel_change)| {
            // Use hash_from_hashes to match EXEC path's hash_from_vec behavior:
            // The EXEC path uses hash_from_vec(&channels) which serializes each channel,
            // hashes each, sorts, concatenates, and hashes again.
            // Since consume_channels here is already Vec<Blake2b256Hash>, we use hash_from_hashes
            // which sorts, concatenates, and hashes - matching the EXEC behavior.
            let history_pointer = stable_hash_provider::hash_from_hashes(consume_channels);
            let init = base_reader.get_continuations_proj_binary(&history_pointer)?;

            let new_val = {
                // Use multiset diff: remove each item in 'removed' exactly once from 'init'
                let mut result = StateChange::multiset_diff(&init, &channel_change.removed);
                result.extend(channel_change.added.clone());
                result
            };

            if init == new_val {
                Err(HistoryError::MergeError(
                    "Merging logic error: empty consume change when computing trie action."
                        .to_string(),
                ))
            } else if init.is_empty() {
                // No konts were in base state and some are added - insert konts and add join.
                Ok(ConsumeAndJoinActions {
                    consume_action: HotStoreTrieAction::TrieInsertAction(
                        TrieInsertAction::TrieInsertBinaryConsume(TrieInsertBinaryConsume {
                            hash: history_pointer,
                            continuations: new_val,
                        }),
                    ),
                    join_action: Some(JoinActionKind::AddJoin(consume_channels.clone())),
                })
            } else if new_val.is_empty() {
                // All konts present in base are removed - remove consume, remove join.
                Ok(ConsumeAndJoinActions {
                    consume_action: HotStoreTrieAction::TrieDeleteAction(
                        TrieDeleteAction::TrieDeleteConsume(TrieDeleteConsume {
                            hash: history_pointer,
                        }),
                    ),
                    join_action: Some(JoinActionKind::RemoveJoin(consume_channels.clone())),
                })
            } else {
                // Konts were updated but consume is present in base state - update konts, no joins.
                Ok(ConsumeAndJoinActions {
                    consume_action: HotStoreTrieAction::TrieInsertAction(
                        TrieInsertAction::TrieInsertBinaryConsume(TrieInsertBinaryConsume {
                            hash: history_pointer,
                            continuations: new_val,
                        }),
                    ),
                    join_action: None,
                })
            }
        })
        .collect::<Result<Vec<ConsumeAndJoinActions<C, P, A, K>>, HistoryError>>()?;

    let consume_trie_actions = consume_with_join_actions
        .iter()
        .map(|consume_and_join_action| consume_and_join_action.consume_action.clone())
        .collect::<Vec<_>>();

    // Sort datum changes by history pointer for deterministic ordering
    let mut datums_changes_sorted: Vec<_> = changes
        .datums_changes
        .iter()
        .map(|ref_multi| (ref_multi.key().clone(), ref_multi.value().clone()))
        .collect();
    datums_changes_sorted.sort_by_key(|(history_pointer, _)| history_pointer.clone());

    let produce_trie_actions = datums_changes_sorted
        .iter()
        .map(|(history_pointer, changes)| {
            handle_channel_change(history_pointer, changes, mergeable_chs).and_then(|action| {
                action.map(Ok).unwrap_or_else(|| {
                    make_trie_action(
                        history_pointer,
                        |hash| base_reader.get_data_proj_binary(hash),
                        changes,
                        |hash| {
                            HotStoreTrieAction::TrieDeleteAction(
                                TrieDeleteAction::TrieDeleteProduce(TrieDeleteProduce {
                                    hash: hash.clone(),
                                }),
                            )
                        },
                        |hash, data| {
                            HotStoreTrieAction::TrieInsertAction(
                                TrieInsertAction::TrieInsertBinaryProduce(
                                    TrieInsertBinaryProduce {
                                        hash: hash.clone(),
                                        data,
                                    },
                                ),
                            )
                        },
                    )
                })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Process joins changes
    let joins_channels_to_body_map = &changes.consume_channels_to_join_serialized_map;
    let mut joins_changes = std::collections::HashMap::new();

    // Collect join changes from consume actions
    for consume_and_join_action in &consume_with_join_actions {
        if let Some(join_action) = &consume_and_join_action.join_action {
            let join_channels = match join_action {
                JoinActionKind::AddJoin(chs) => chs,
                JoinActionKind::RemoveJoin(chs) => chs,
            };

            // Get the serialized join data for these channels
            if let Some(join_data) = joins_channels_to_body_map.get(join_channels) {
                // Update the joins_changes for each channel
                for channel in join_channels {
                    let current_val = joins_changes
                        .entry(channel.clone())
                        .or_insert_with(ChannelChange::empty);

                    match join_action {
                        JoinActionKind::AddJoin(_) => {
                            current_val.added.push(join_data.clone());
                        }
                        JoinActionKind::RemoveJoin(_) => {
                            current_val.removed.push(join_data.clone());
                        }
                    }
                }
            } else {
                return Err(HistoryError::MergeError(
                    "No ByteVector value for join found when merging when computing trie action."
                        .to_string(),
                ));
            }
        }
    }

    // Process joins for each channel
    let joins_trie_actions = joins_changes
        .iter()
        .map(|(history_pointer, changes)| {
            make_trie_action(
                history_pointer,
                |hash| base_reader.get_joins_proj_binary(hash),
                changes,
                |hash| {
                    HotStoreTrieAction::TrieDeleteAction(TrieDeleteAction::TrieDeleteJoins(
                        TrieDeleteJoins { hash: hash.clone() },
                    ))
                },
                |hash, joins| {
                    HotStoreTrieAction::TrieInsertAction(TrieInsertAction::TrieInsertBinaryJoins(
                        TrieInsertBinaryJoins {
                            hash: hash.clone(),
                            joins,
                        },
                    ))
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Combine all trie actions
    let mut result = Vec::new();
    result.extend(produce_trie_actions);
    result.extend(consume_trie_actions);
    result.extend(joins_trie_actions);

    Ok(result)
}

fn make_trie_action<C: Clone, P: Clone, A: Clone, K: Clone>(
    history_pointer: &Blake2b256Hash,
    init_value: impl Fn(&Blake2b256Hash) -> Result<Vec<ByteVector>, HistoryError>,
    changes: &ChannelChange<ByteVector>,
    remove_action: impl Fn(&Blake2b256Hash) -> HotStoreTrieAction<C, P, A, K>,
    update_action: impl Fn(&Blake2b256Hash, Vec<ByteVector>) -> HotStoreTrieAction<C, P, A, K>,
) -> Result<HotStoreTrieAction<C, P, A, K>, HistoryError> {
    let init = init_value(history_pointer)?;

    let new_val = {
        // Use multiset diff: remove each item in 'removed' exactly once from 'init'
        let mut result = StateChange::multiset_diff(&init, &changes.removed);
        result.extend(changes.added.clone());
        result
    };

    if new_val.is_empty() && !init.is_empty() {
        // Case 1: All items present in base are removed - remove action
        Ok(remove_action(history_pointer))
    } else if init != new_val {
        // Case 2: Items were updated - update action
        Ok(update_action(history_pointer, new_val))
    } else {
        // Case 3: Error case - no changes
        Err(HistoryError::MergeError(
            "Merging logic error: empty channel change for produce or join when computing trie action."
                .to_string(),
        ))
    }
}
