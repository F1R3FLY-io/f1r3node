// See rspace/src/main/scala/coop/rchain/rspace/SpaceMatcher.scala

use std::collections::HashMap;

use super::r#match::Match;
use super::rspace_interface::ISpace;
use crate::rspace::internal::{ConsumeCandidate, Datum, ProduceCandidate, WaitingContinuation};

type MatchingDataCandidate<C, A> = (ConsumeCandidate<C, A>, Vec<(Datum<A>, i32)>);

pub trait SpaceMatcher<C, P, A, K>: ISpace<C, P, A, K>
where
    C: Clone + std::hash::Hash + Eq,
    P: Clone,
    A: Clone,
    K: Clone,
{
    fn find_matching_data_candidate(
        &self,
        matcher: &Box<dyn Match<P, A>>,
        channel: C,
        data: &[(Datum<A>, i32)],
        pattern: &P,
    ) -> Option<MatchingDataCandidate<C, A>> {
        for (idx, (datum, data_index)) in data.iter().enumerate() {
            metrics::counter!("rspace.matcher.get_calls", "source" => "rspace").increment(1);
            let t_clone = std::time::Instant::now();
            let pattern_cloned = pattern.clone();
            let data_cloned = datum.a.clone();
            metrics::counter!("rspace.matcher.clone_ns", "source" => "rspace")
                .increment(t_clone.elapsed().as_nanos() as u64);
            let t_match = std::time::Instant::now();
            let match_result = matcher.get(pattern_cloned, data_cloned);
            metrics::counter!("rspace.matcher.fold_match_ns", "source" => "rspace")
                .increment(t_match.elapsed().as_nanos() as u64);

            if let Some(mat) = match_result {
                let indexed_datums = if datum.persist {
                    data.to_vec()
                } else {
                    let mut remaining = Vec::with_capacity(data.len() - 1);
                    remaining.extend_from_slice(&data[..idx]);
                    remaining.extend_from_slice(&data[idx + 1..]);
                    remaining
                };
                return Some((
                    ConsumeCandidate {
                        channel,
                        datum: Datum {
                            a: mat,
                            persist: datum.persist,
                            source: datum.source.clone(),
                        },
                        removed_datum: datum.a.clone(),
                        datum_index: *data_index,
                    },
                    indexed_datums,
                ));
            }
        }
        None
    }

    /// Attempts to match all channel-pattern pairs against the data map.
    /// Records mutations in `rollback` so the caller can undo them on failure.
    fn extract_data_candidates_rollback(
        &self,
        matcher: &Box<dyn Match<P, A>>,
        channel_pattern_pairs: &[(C, P)],
        channel_to_indexed_data: &mut HashMap<C, Vec<(Datum<A>, i32)>>,
        rollback: &mut Vec<(C, Vec<(Datum<A>, i32)>)>,
    ) -> Vec<Option<ConsumeCandidate<C, A>>> {
        let mut acc = Vec::with_capacity(channel_pattern_pairs.len());

        for (channel, pattern) in channel_pattern_pairs {
            let maybe_tuple: Option<MatchingDataCandidate<C, A>> =
                match channel_to_indexed_data.get(channel) {
                    Some(indexed_data) => self.find_matching_data_candidate(
                        matcher,
                        channel.clone(),
                        indexed_data,
                        pattern,
                    ),
                    None => None,
                };

            match maybe_tuple {
                Some((cand, rem)) => {
                    acc.push(Some(cand));
                    // Save the original before mutating
                    if let Some(original) = channel_to_indexed_data.get(channel) {
                        rollback.push((channel.clone(), original.clone()));
                    }
                    channel_to_indexed_data.insert(channel.clone(), rem);
                }
                None => {
                    acc.push(None);
                }
            }
        }

        acc
    }

    /// Non-rollback version for consume path (no speculative matching).
    fn extract_data_candidates(
        &self,
        matcher: &Box<dyn Match<P, A>>,
        channel_pattern_pairs: &[(C, P)],
        channel_to_indexed_data: &mut HashMap<C, Vec<(Datum<A>, i32)>>,
    ) -> Vec<Option<ConsumeCandidate<C, A>>> {
        let mut rollback = Vec::new();
        self.extract_data_candidates_rollback(
            matcher,
            channel_pattern_pairs,
            channel_to_indexed_data,
            &mut rollback,
        )
    }

    fn extract_first_match(
        &self,
        matcher: &Box<dyn Match<P, A>>,
        channels: Vec<C>,
        match_candidates: Vec<(WaitingContinuation<P, K>, i32)>,
        mut channel_to_index_data: HashMap<C, Vec<(Datum<A>, i32)>>,
    ) -> Option<ProduceCandidate<C, P, A, K>> {
        for (cont, index) in &match_candidates {
            let channel_pattern_pairs: Vec<(C, P)> = channels
                .iter()
                .cloned()
                .zip(cont.patterns.iter().cloned())
                .collect();

            let mut rollback = Vec::new();
            let data_candidates = self.extract_data_candidates_rollback(
                matcher,
                &channel_pattern_pairs,
                &mut channel_to_index_data,
                &mut rollback,
            );

            if data_candidates.iter().all(|x| x.is_some()) {
                return Some(ProduceCandidate {
                    channels,
                    continuation: cont.clone(),
                    continuation_index: *index,
                    data_candidates: data_candidates.into_iter().filter_map(|x| x).collect(),
                });
            }

            // Restore only the channels that were actually mutated
            for (ch, original) in rollback {
                channel_to_index_data.insert(ch, original);
            }
        }
        None
    }
}
