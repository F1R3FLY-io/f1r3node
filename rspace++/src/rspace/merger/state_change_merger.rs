// See rspace/src/main/scala/coop/rchain/rspace/merger/StateChangeMerger.scala

use shared::rust::ByteVector;

use super::channel_change::ChannelChange;
use super::merging_logic::NumberChannelsDiff;
use super::state_change::StateChange;
use crate::rspace::errors::HistoryError;
use crate::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use crate::rspace::hashing::stable_hash_provider;
use crate::rspace::history::history_reader::HistoryReader;
use crate::rspace::hot_store_trie_action::{
    HotStoreTrieAction, TrieDeleteAction, TrieDeleteConsume, TrieDeleteJoins, TrieDeleteProduce,
    TrieInsertAction, TrieInsertBinaryConsume, TrieInsertBinaryJoins, TrieInsertBinaryProduce,
};

/**
 * This classes are used to compute joins.
 * Consume value pointer that stores continuations on some channel is
 * identified by channels involved in. Therefore when no continuations on
 * some consume is left and the whole consume ponter is removed -
 * no joins with corresponding seq of channels exist in tuple space. So join
 * should be removed.
 */
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
    // Sort continuation changes by hash of consume channels for deterministic
    // ordering
    let mut cont_changes_sorted: Vec<_> = changes
        .cont_changes
        .iter()
        .map(|ref_multi| (ref_multi.key().clone(), ref_multi.value().clone()))
        .collect();
    cont_changes_sorted.sort_by_key(|(consume_channels, _)| {
        stable_hash_provider::hash_from_hashes(consume_channels)
    });

    let consume_with_join_actions: Vec<ConsumeAndJoinActions<C, P, A, K>> = cont_changes_sorted
        .iter()
        .map(|(consume_channels, channel_change)| {
            // Use hash_from_hashes to match EXEC path's hash_from_vec behavior:
            // The EXEC path uses hash_from_vec(&channels) which serializes each channel,
            // hashes each, sorts, concatenates, and hashes again.
            // Since consume_channels here is already Vec<Blake2b256Hash>, we use
            // hash_from_hashes which sorts, concatenates, and hashes - matching
            // the EXEC behavior.
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
                // Konts were updated but consume is present in base state - update konts, no
                // joins.
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

    // OBSERVABILITY — at entry to the produce-action loop, log the high-level
    // shape: how many distinct channels have datum diffs, and how many of
    // those have an entry in `mergeable_chs` (which determines OR-fold vs
    // multiset-fallthrough dispatch in the closure below).
    let in_mergeable_count = datums_changes_sorted
        .iter()
        .filter(|(h, _)| mergeable_chs.contains_key(h))
        .count();
    tracing::debug!(
        target: "f1r3fly.merge.trie_action",
        datum_channels = datums_changes_sorted.len(),
        in_mergeable_chs = in_mergeable_count,
        fall_through_to_multiset = datums_changes_sorted.len() - in_mergeable_count,
        "[TRIE-ACTION] compute_trie_actions dispatch summary",
    );

    let produce_trie_actions = datums_changes_sorted
        .iter()
        .map(|(history_pointer, changes)| {
            // OBSERVABILITY — log per-channel dispatch decision. `in_mergeable`
            // means handle_channel_change will return Some(action) and run
            // the OR-fold / IntegerAdd path; otherwise we fall through to
            // make_trie_action with multiset semantics.
            let in_mergeable = mergeable_chs.contains_key(history_pointer);
            let ch_bytes = history_pointer.bytes();
            let ch_short =
                hex::encode(&ch_bytes[..std::cmp::min(8, ch_bytes.len())]);
            tracing::debug!(
                target: "f1r3fly.merge.trie_action",
                channel_short = %ch_short,
                in_mergeable_chs = in_mergeable,
                added_len = changes.added.len(),
                removed_len = changes.removed.len(),
                "[TRIE-ACTION] per-channel dispatch",
            );
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

    // Sort joins changes by history pointer for deterministic ordering
    let mut joins_changes_sorted: Vec<_> = joins_changes.into_iter().collect();
    joins_changes_sorted.sort_by_key(|(history_pointer, _)| history_pointer.clone());

    let joins_trie_actions = joins_changes_sorted
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

    // OBSERVABILITY — log every multiset-fallthrough trie action so we can
    // trace the orphan's origin. The merger writes here when a channel is
    // NOT in `number_channels_data` (commutative path). For tagged channels
    // whose deploy value is non-numeric this is the load-bearing dispatch
    // for multi-Datum birth: `init - removed + added` can yield >1 element
    // when (a) init already had a stale datum and changes don't consume
    // it, or (b) `added` carries multiple values (cross-branch ChannelChange
    // multiset-union accumulating distinct sibling produces).
    let ch_bytes = history_pointer.bytes();
    let ch_short =
        hex::encode(&ch_bytes[..std::cmp::min(8, ch_bytes.len())]);
    if new_val.len() > 1 {
        tracing::warn!(
            target: "f1r3fly.merge.trie_action",
            channel_short = %ch_short,
            channel_full = %hex::encode(&ch_bytes),
            init_len = init.len(),
            added_len = changes.added.len(),
            removed_len = changes.removed.len(),
            new_val_len = new_val.len(),
            "[TRIE-ACTION-MULTI-DATUM] make_trie_action will write {} datums (init={} added={} removed={})",
            new_val.len(), init.len(), changes.added.len(), changes.removed.len()
        );
    } else if changes.added.len() + changes.removed.len() > 0 {
        tracing::debug!(
            target: "f1r3fly.merge.trie_action",
            channel_short = %ch_short,
            init_len = init.len(),
            added_len = changes.added.len(),
            removed_len = changes.removed.len(),
            new_val_len = new_val.len(),
            "[TRIE-ACTION] make_trie_action multiset path",
        );
    }

    if new_val.is_empty() && !init.is_empty() {
        // Case 1: All items present in base are removed - remove action
        Ok(remove_action(history_pointer))
    } else if init != new_val {
        // Case 2: Items were updated - update action
        Ok(update_action(history_pointer, new_val))
    } else {
        // Case 3: Error case - no changes
        Err(HistoryError::MergeError(
            "Merging logic error: empty channel change for produce or join when computing trie \
             action."
                .to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use dashmap::DashMap;

    use super::*;
    use crate::rspace::history::history_reader::HistoryReaderBase;
    use crate::rspace::internal::{Datum, WaitingContinuation};

    struct StubHistoryReaderBinary {
        data_map: HashMap<Blake2b256Hash, Vec<Vec<u8>>>,
    }

    struct EmptyHistoryReaderBase;

    impl HistoryReaderBase<(), (), (), ()> for EmptyHistoryReaderBase {
        fn get_data_proj(&self, _key: &()) -> Vec<Datum<()>> { vec![] }

        fn get_continuations_proj(&self, _key: &Vec<()>) -> Vec<WaitingContinuation<(), ()>> {
            vec![]
        }

        fn get_joins_proj(&self, _key: &()) -> Vec<Vec<()>> { vec![] }
    }

    impl HistoryReader<Blake2b256Hash, (), (), (), ()> for StubHistoryReaderBinary {
        fn root(&self) -> Blake2b256Hash { Blake2b256Hash::from_bytes(vec![0xff; 32]) }

        fn get_data_proj(&self, _key: &Blake2b256Hash) -> Result<Vec<Datum<()>>, HistoryError> {
            Ok(vec![])
        }

        fn get_data_proj_binary(&self, key: &Blake2b256Hash) -> Result<Vec<Vec<u8>>, HistoryError> {
            Ok(self.data_map.get(key).cloned().unwrap_or_default())
        }

        fn get_continuations_proj(
            &self,
            _key: &Blake2b256Hash,
        ) -> Result<Vec<WaitingContinuation<(), ()>>, HistoryError> {
            Ok(vec![])
        }

        fn get_continuations_proj_binary(
            &self,
            _key: &Blake2b256Hash,
        ) -> Result<Vec<Vec<u8>>, HistoryError> {
            Ok(vec![])
        }

        fn get_joins_proj(&self, _key: &Blake2b256Hash) -> Result<Vec<Vec<()>>, HistoryError> {
            Ok(vec![])
        }

        fn get_joins_proj_binary(
            &self,
            _key: &Blake2b256Hash,
        ) -> Result<Vec<Vec<u8>>, HistoryError> {
            Ok(vec![])
        }

        fn base(&self) -> Box<dyn HistoryReaderBase<(), (), (), ()>> {
            Box::new(EmptyHistoryReaderBase)
        }

        fn get_data_proj_generic(&self, _key: &()) -> Vec<Datum<()>> { vec![] }

        fn get_continuations_proj_generic(
            &self,
            _key: &Vec<()>,
        ) -> Vec<WaitingContinuation<(), ()>> {
            vec![]
        }

        fn get_joins_proj_generic(&self, _key: &()) -> Vec<Vec<()>> { vec![] }
    }

    /// Reproduces the ChannelChange.combine() duplication bug end-to-end
    /// through compute_trie_actions.
    #[test]
    fn compute_trie_actions_should_not_duplicate_data_when_merging_identical_sibling_changes() {
        let datum_a: Vec<u8> = vec![0xaa; 32];
        let datum_b: Vec<u8> = vec![0xbb; 32];
        let channel_hash = Blake2b256Hash::from_bytes(vec![0x01; 32]);

        let base_reader: Box<dyn HistoryReader<Blake2b256Hash, (), (), (), ()>> =
            Box::new(StubHistoryReaderBinary {
                data_map: HashMap::from([(channel_hash.clone(), vec![datum_a.clone()])]),
            });

        // Two sibling blocks both change A -> B on the same channel
        let datums_changes = DashMap::new();
        datums_changes.insert(channel_hash.clone(), ChannelChange {
            added: vec![datum_b.clone()],
            removed: vec![datum_a.clone()],
        });
        let branch_change = StateChange {
            datums_changes,
            cont_changes: DashMap::new(),
            consume_channels_to_join_serialized_map: DashMap::new(),
        };
        let combined = branch_change.clone().combine(branch_change);

        let mergeable_chs: NumberChannelsDiff = BTreeMap::new();
        let no_override =
            |_: &Blake2b256Hash,
             _: &ChannelChange<Vec<u8>>,
             _: &NumberChannelsDiff|
             -> Result<Option<HotStoreTrieAction<(), (), (), ()>>, HistoryError> {
                Ok(None)
            };

        let actions = compute_trie_actions(&combined, &base_reader, &mergeable_chs, no_override)
            .expect("compute_trie_actions should succeed");

        assert_eq!(actions.len(), 1, "expected exactly one trie action");
        match &actions[0] {
            HotStoreTrieAction::TrieInsertAction(TrieInsertAction::TrieInsertBinaryProduce(
                insert,
            )) => {
                assert_eq!(insert.hash, channel_hash);
                assert_eq!(insert.data, vec![datum_b]);
            }
            other => panic!("expected TrieInsertBinaryProduce, got {:?}", other),
        }
    }

    /// A single `ChannelChange` with one add against an empty base state
    /// yields exactly one trie insert with that one datum. Companion of
    /// `compute_trie_actions_should_not_duplicate_data_when_merging_identical_sibling_changes`.
    #[test]
    fn compute_trie_actions_writes_single_datum_for_single_survivor_after_conflict_resolution() {
        let datum_x: Vec<u8> = vec![0x11; 32];
        let channel_hash = Blake2b256Hash::from_bytes(vec![0x01; 32]);

        let base_reader: Box<dyn HistoryReader<Blake2b256Hash, (), (), (), ()>> =
            Box::new(StubHistoryReaderBinary {
                data_map: HashMap::new(),
            });

        let survivor_change = {
            let dc = DashMap::new();
            dc.insert(channel_hash.clone(), ChannelChange {
                added: vec![datum_x.clone()],
                removed: vec![],
            });
            StateChange {
                datums_changes: dc,
                cont_changes: DashMap::new(),
                consume_channels_to_join_serialized_map: DashMap::new(),
            }
        };

        let mergeable_chs: NumberChannelsDiff = BTreeMap::new();
        let no_override =
            |_: &Blake2b256Hash,
             _: &ChannelChange<Vec<u8>>,
             _: &NumberChannelsDiff|
             -> Result<Option<HotStoreTrieAction<(), (), (), ()>>, HistoryError> {
                Ok(None)
            };

        let actions =
            compute_trie_actions(&survivor_change, &base_reader, &mergeable_chs, no_override)
                .expect("compute_trie_actions should succeed");

        assert_eq!(actions.len(), 1, "expected exactly one trie action on the channel");
        match &actions[0] {
            HotStoreTrieAction::TrieInsertAction(TrieInsertAction::TrieInsertBinaryProduce(
                insert,
            )) => {
                assert_eq!(insert.hash, channel_hash);
                assert_eq!(insert.data, vec![datum_x]);
            }
            other => panic!("expected TrieInsertBinaryProduce, got {:?}", other),
        }
    }
}
