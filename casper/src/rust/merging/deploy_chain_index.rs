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

    /// Construct a DeployChainIndex directly from its parts (for testing).
    pub fn from_parts(
        deploys_with_cost: HashableSet<DeployIdWithCost>,
        pre_state_hash: Blake2b256Hash,
        post_state_hash: Blake2b256Hash,
        event_log_index: EventLogIndex,
        state_changes: StateChange,
    ) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        deploys_with_cost.hash(&mut hasher);
        let hash_code = hasher.finish() as i32;
        DeployChainIndex {
            deploys_with_cost,
            pre_state_hash,
            post_state_hash,
            event_log_index,
            state_changes,
            hash_code,
        }
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
        // 1. PRIMARY: Highest total cost first (economic incentive)
        //    Higher-paying transactions get priority in conflict resolution
        let self_total_cost: u64 = self.deploys_with_cost.0.iter().map(|d| d.cost).sum();
        let other_total_cost: u64 = other.deploys_with_cost.0.iter().map(|d| d.cost).sum();

        let cost_cmp = self_total_cost.cmp(&other_total_cost).reverse(); // Higher cost first
        if cost_cmp != std::cmp::Ordering::Equal {
            return cost_cmp;
        }

        // 2. SECONDARY: Highest single deploy cost (prioritize high-value individual transactions)
        let self_max_cost = self
            .deploys_with_cost
            .0
            .iter()
            .map(|d| d.cost)
            .max()
            .unwrap_or(0);
        let other_max_cost = other
            .deploys_with_cost
            .0
            .iter()
            .map(|d| d.cost)
            .max()
            .unwrap_or(0);

        let max_cost_cmp = self_max_cost.cmp(&other_max_cost).reverse(); // Higher max cost first
        if max_cost_cmp != std::cmp::Ordering::Equal {
            return max_cost_cmp;
        }

        // 3. TERTIARY: Lexicographically smallest deploy signature (deterministic)
        //    This ensures consistent ordering across all nodes when costs are equal
        let self_min_deploy = self
            .deploys_with_cost
            .0
            .iter()
            .min_by(|a, b| a.deploy_id.cmp(&b.deploy_id));
        let other_min_deploy = other
            .deploys_with_cost
            .0
            .iter()
            .min_by(|a, b| a.deploy_id.cmp(&b.deploy_id));

        let signature_cmp = match (self_min_deploy, other_min_deploy) {
            (Some(self_deploy), Some(other_deploy)) => {
                self_deploy.deploy_id.cmp(&other_deploy.deploy_id)
            }
            (Some(_), None) => std::cmp::Ordering::Greater,
            (None, Some(_)) => std::cmp::Ordering::Less,
            (None, None) => std::cmp::Ordering::Equal,
        };

        if signature_cmp != std::cmp::Ordering::Equal {
            return signature_cmp;
        }

        // 4. QUATERNARY: Post-state hash as final fallback
        //    Ensures total ordering even for identical deploys (should be rare)
        self.post_state_hash.cmp(&other.post_state_hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
    use std::collections::HashSet;

    fn mk_index(deploys: &[(u8, u64)], post_state_seed: u8) -> DeployChainIndex {
        let deploys_with_cost: HashSet<DeployIdWithCost> = deploys
            .iter()
            .map(|(id, cost)| DeployIdWithCost {
                deploy_id: Bytes::from(vec![*id]),
                cost: *cost,
            })
            .collect();

        DeployChainIndex {
            deploys_with_cost: HashableSet(deploys_with_cost),
            pre_state_hash: Blake2b256Hash::from_bytes(vec![0u8; 32]),
            post_state_hash: Blake2b256Hash::from_bytes(vec![post_state_seed; 32]),
            event_log_index: EventLogIndex::empty(),
            state_changes: StateChange::empty(),
            hash_code: 0,
        }
    }

    #[test]
    fn ordering_prefers_higher_total_cost() {
        let high_total = mk_index(&[(1, 10), (2, 1)], 1); // total = 11
        let low_total = mk_index(&[(1, 9), (2, 1)], 2); // total = 10

        assert_eq!(high_total.cmp(&low_total), std::cmp::Ordering::Less);
        assert_eq!(low_total.cmp(&high_total), std::cmp::Ordering::Greater);
    }

    #[test]
    fn ordering_tie_breaks_on_max_cost_then_signature() {
        // Same total (11), different max (7 vs 6)
        let max_seven = mk_index(&[(1, 7), (2, 4)], 1);
        let max_six = mk_index(&[(1, 6), (2, 5)], 2);
        assert_eq!(max_seven.cmp(&max_six), std::cmp::Ordering::Less);

        // Same total/max, tie-break by smallest deploy signature (2 < 3)
        let min_sig_two = mk_index(&[(2, 5), (9, 5)], 1);
        let min_sig_three = mk_index(&[(3, 5), (9, 5)], 2);
        assert_eq!(min_sig_two.cmp(&min_sig_three), std::cmp::Ordering::Less);
    }

    #[test]
    fn ordering_final_tie_breaks_on_post_state_hash() {
        let a = mk_index(&[(1, 5)], 0x01);
        let b = mk_index(&[(1, 5)], 0x02);

        assert_eq!(a.cmp(&b), std::cmp::Ordering::Less);
        assert_eq!(b.cmp(&a), std::cmp::Ordering::Greater);
    }
}
