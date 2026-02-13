// See casper/src/main/scala/coop/rchain/casper/merging/DeployChainIndex.scala

use prost::bytes::Bytes;
use shared::rust::hashable_set::HashableSet;
use std::{collections::HashSet, sync::Arc};

use rspace_plus_plus::rspace::{
    errors::HistoryError,
    hashing::blake2b256_hash::Blake2b256Hash,
    history::history_repository::HistoryRepository,
    merger::{event_log_index::EventLogIndex, state_change::StateChange},
};

use super::deploy_index::DeployIndex;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeployIdWithCost {
    pub deploy_id: Bytes,
    pub cost: u64,
}

/** index of deploys depending on each other inside a single block (state transition) */
#[derive(Debug, Clone, Hash)]
pub struct DeployChainIndex {
    pub deploys_with_cost: HashableSet<DeployIdWithCost>,
    pre_state_hash: Blake2b256Hash,
    post_state_hash: Blake2b256Hash,
    pub event_log_index: EventLogIndex,
    pub state_changes: StateChange,
    // caching hash code helps a lot to increase performance of computing rejection options
    // TODO mysterious speedup of merging benchmark when setting this to some fixed value - OLD
    hash_code: i32,
}

impl DeployChainIndex {
    pub fn new<C, P, A, K>(
        deploys: &HashableSet<DeployIndex>,
        pre_state_hash: &Blake2b256Hash,
        post_state_hash: &Blake2b256Hash,
        history_repository: Arc<Box<dyn HistoryRepository<C, P, A, K> + Send + Sync + 'static>>,
    ) -> Result<Self, HistoryError>
    where
        C: std::clone::Clone
            + serde::Serialize
            + for<'de> serde::Deserialize<'de>
            + Send
            + Sync
            + 'static,
        P: std::clone::Clone + for<'de> serde::Deserialize<'de> + Send + Sync + 'static,
        A: std::clone::Clone + for<'de> serde::Deserialize<'de> + Send + Sync + 'static,
        K: std::clone::Clone + for<'de> serde::Deserialize<'de> + Send + Sync + 'static,
    {
        let deploys_with_cost: HashSet<DeployIdWithCost> = deploys
            .0
            .iter()
            .map(|deploy| DeployIdWithCost {
                deploy_id: deploy.deploy_id.clone(),
                cost: deploy.cost,
            })
            .collect();

        let event_log_index = deploys
            .into_iter()
            .fold(EventLogIndex::empty(), |acc, deploy| {
                EventLogIndex::combine(&acc, &deploy.event_log_index)
            });

        let pre_history_reader = history_repository.get_history_reader_struct(&pre_state_hash)?;
        let post_history_reader = history_repository.get_history_reader_struct(&post_state_hash)?;

        let state_changes =
            StateChange::new(pre_history_reader, post_history_reader, &event_log_index)?;

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for deploy in &deploys_with_cost {
            std::hash::Hash::hash(&deploy.deploy_id, &mut hasher);
        }
        let hash_code = std::hash::Hasher::finish(&hasher) as i32;

        Ok(Self {
            deploys_with_cost: HashableSet(deploys_with_cost),
            pre_state_hash: pre_state_hash.clone(),
            post_state_hash: post_state_hash.clone(),
            event_log_index,
            state_changes,
            hash_code,
        })
    }
}

impl PartialEq for DeployChainIndex {
    fn eq(&self, other: &Self) -> bool {
        self.deploys_with_cost == other.deploys_with_cost
    }
}

impl Eq for DeployChainIndex {}

impl PartialOrd for DeployChainIndex {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DeployChainIndex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Total ordering for deterministic processing across validators.
        // Primary sort by postStateHash, with tiebreakers by preStateHash and then
        // lexicographic comparison of sorted deploy IDs. This prevents ambiguity
        // when two deploy chains produce the same post-state hash.

        // 1. PRIMARY: post_state_hash
        let post_cmp = self.post_state_hash.cmp(&other.post_state_hash);
        if post_cmp != std::cmp::Ordering::Equal {
            return post_cmp;
        }

        // 2. SECONDARY: pre_state_hash
        let pre_cmp = self.pre_state_hash.cmp(&other.pre_state_hash);
        if pre_cmp != std::cmp::Ordering::Equal {
            return pre_cmp;
        }

        // 3. TERTIARY: Lexicographic comparison of sorted deploy IDs
        let mut a_ids: Vec<_> = self
            .deploys_with_cost
            .0
            .iter()
            .map(|d| &d.deploy_id)
            .collect();
        let mut b_ids: Vec<_> = other
            .deploys_with_cost
            .0
            .iter()
            .map(|d| &d.deploy_id)
            .collect();
        a_ids.sort();
        b_ids.sort();

        let len_cmp = a_ids.len().cmp(&b_ids.len());
        if len_cmp != std::cmp::Ordering::Equal {
            return len_cmp;
        }

        for (ai, bi) in a_ids.iter().zip(b_ids.iter()) {
            let id_cmp = ai.cmp(bi);
            if id_cmp != std::cmp::Ordering::Equal {
                return id_cmp;
            }
        }

        std::cmp::Ordering::Equal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rspace_plus_plus::rspace::{
        hashing::blake2b256_hash::Blake2b256Hash,
        merger::{event_log_index::EventLogIndex, state_change::StateChange},
    };
    use shared::rust::hashable_set::HashableSet;
    use std::collections::HashSet;

    fn mk_hash(byte: u8) -> Blake2b256Hash {
        Blake2b256Hash::from_bytes(vec![byte; 32])
    }

    fn mk_deploy_id(byte: u8) -> Bytes {
        Bytes::from(vec![byte; 64])
    }

    fn mk_index(post_state: u8, pre_state: u8, deploy_ids: &[u8]) -> DeployChainIndex {
        let deploys_with_cost: HashSet<DeployIdWithCost> = deploy_ids
            .iter()
            .map(|&b| DeployIdWithCost {
                deploy_id: mk_deploy_id(b),
                cost: 0,
            })
            .collect();

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for d in &deploys_with_cost {
            std::hash::Hash::hash(&d.deploy_id, &mut hasher);
        }
        let hash_code = std::hash::Hasher::finish(&hasher) as i32;

        DeployChainIndex {
            deploys_with_cost: HashableSet(deploys_with_cost),
            pre_state_hash: mk_hash(pre_state),
            post_state_hash: mk_hash(post_state),
            event_log_index: EventLogIndex::empty(),
            state_changes: StateChange::empty(),
            hash_code,
        }
    }

    #[test]
    fn ordering_by_post_state_hash_primarily() {
        let a = mk_index(1, 0, &[10]);
        let b = mk_index(2, 0, &[10]);
        assert!(
            a.cmp(&b) == std::cmp::Ordering::Less,
            "lower postStateHash should sort before higher"
        );
    }

    #[test]
    fn ordering_uses_pre_state_hash_as_tiebreaker() {
        let a = mk_index(1, 1, &[10]);
        let b = mk_index(1, 2, &[10]);
        assert_ne!(a.cmp(&b), std::cmp::Ordering::Equal);
    }

    #[test]
    fn ordering_uses_deploy_ids_as_final_tiebreaker() {
        let a = mk_index(1, 1, &[10]);
        let b = mk_index(1, 1, &[20]);
        assert_ne!(a.cmp(&b), std::cmp::Ordering::Equal);
    }

    #[test]
    fn ordering_compares_equal_for_identical_instances() {
        let a = mk_index(1, 1, &[10, 20]);
        let b = mk_index(1, 1, &[10, 20]);
        assert_eq!(a.cmp(&b), std::cmp::Ordering::Equal);
    }

    #[test]
    fn ordering_is_antisymmetric() {
        let a = mk_index(1, 1, &[10]);
        let b = mk_index(1, 1, &[20]);
        let cmp = a.cmp(&b);
        assert_eq!(b.cmp(&a), cmp.reverse());
    }

    #[test]
    fn ordering_produces_consistent_sorted_order() {
        let items = vec![
            mk_index(3, 1, &[10]),
            mk_index(1, 2, &[20]),
            mk_index(2, 1, &[30]),
            mk_index(1, 1, &[40]),
            mk_index(1, 1, &[10]),
        ];

        let mut sorted1 = items.clone();
        sorted1.sort();

        let mut sorted2 = items.clone();
        sorted2.reverse();
        sorted2.sort();

        // Both sort orders must produce the same sequence
        for (a, b) in sorted1.iter().zip(sorted2.iter()) {
            assert_eq!(
                a.cmp(b),
                std::cmp::Ordering::Equal,
                "sorted order must be identical regardless of input order"
            );
        }
    }
}
