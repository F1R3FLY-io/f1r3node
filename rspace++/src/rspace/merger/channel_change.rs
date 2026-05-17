// See rspace/src/main/scala/coop/rchain/rspace/merger/ChannelChange.scala

#[derive(Debug, Clone)]
pub struct ChannelChange<A> {
    pub added: Vec<A>,
    pub removed: Vec<A>,
}

impl<A> ChannelChange<A> {
    pub fn empty() -> Self {
        Self {
            added: Vec::new(),
            removed: Vec::new(),
        }
    }

    /// Multiset union: self ++ (other diff self) = max(count_self, count_other)
    /// per element.
    ///
    /// When two sibling blocks (same parent/LCA) execute identical system
    /// deploys, plain concatenation would double entries. Multiset union
    /// prevents this by only appending elements from `other` that are not
    /// already present in `self` (accounting for multiplicities).
    pub fn combine(self, other: Self) -> Self
    where A: PartialEq {
        let self_added_len = self.added.len();
        let self_removed_len = self.removed.len();
        let other_added_len = other.added.len();
        let other_removed_len = other.removed.len();
        let added_only_in_other = Self::vec_diff(other.added, &self.added);
        let removed_only_in_other = Self::vec_diff(other.removed, &self.removed);
        let result = Self {
            added: self.added.into_iter().chain(added_only_in_other).collect(),
            removed: self
                .removed
                .into_iter()
                .chain(removed_only_in_other)
                .collect(),
        };
        // OBSERVABILITY — record the multiset-union outcome. Caller is
        // generic (no channel hash visible here), so callers that want to
        // tag specific channels should log surrounding context with a
        // matching event-id or channel identifier. This emit fires per
        // ChannelChange combine and is the lowest-level signal of the
        // multiset-union behavior responsible for sibling-merge multi-Datum.
        if result.added.len() > 1 || result.removed.len() > 1 {
            tracing::debug!(
                target: "f1r3fly.merge.combine",
                self_added = self_added_len,
                self_removed = self_removed_len,
                other_added = other_added_len,
                other_removed = other_removed_len,
                result_added = result.added.len(),
                result_removed = result.removed.len(),
                "[CHANNEL-CHANGE-COMBINE-MULTI] ChannelChange::combine yielded multi-element result",
            );
        }
        result
    }

    /// Multiset difference: for each element in `to_remove`, removes at most
    /// one matching element from `from`.
    fn vec_diff(mut from: Vec<A>, to_remove: &[A]) -> Vec<A>
    where A: PartialEq {
        for item in to_remove {
            if let Some(pos) = from.iter().position(|x| x == item) {
                from.remove(pos);
            }
        }
        from
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rspace::merger::state_change::StateChange;

    #[test]
    fn combine_should_not_duplicate_when_combining_identical_changes_from_sibling_blocks() {
        // Two sibling blocks both transition channel state: remove A, add B
        let datum_a: Vec<u8> = vec![0xaa; 32];
        let datum_b: Vec<u8> = vec![0xbb; 32];

        let change = ChannelChange {
            added: vec![datum_b.clone()],
            removed: vec![datum_a.clone()],
        };
        let combined = change.clone().combine(change);

        // Applying mkTrieAction formula: (init diff removed) ++ added
        // With init = [A], correct result is [B] (not [B, B])
        let init = vec![datum_a];
        let mut merged_result = StateChange::multiset_diff(&init, &combined.removed);
        merged_result.extend(combined.added);

        assert_eq!(merged_result, vec![datum_b]);
    }
}
