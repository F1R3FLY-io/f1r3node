// See rspace/src/main/scala/coop/rchain/rspace/SpaceMatcher.scala

use dashmap::DashMap;

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
    /** Searches through data, looking for a match with a given pattern.
     *
     * If there is a match, we return the matching [[ConsumeCandidate]],
     * along with the remaining unmatched data. If an illegal state is
     * reached during searching for a match we short circuit and return
     * the state.
     */
    fn find_matching_data_candidate(
        &self,
        matcher: &Box<dyn Match<P, A>>,
        channel: C,
        data: Vec<(Datum<A>, i32)>,
        pattern: P,
        prefix: Vec<(Datum<A>, i32)>,
    ) -> Option<MatchingDataCandidate<C, A>> {
        match data.split_first() {
            Some((
                (
                    datum @ Datum {
                        a: match_candidate,
                        persist,
                        source: produce_ref,
                    },
                    data_index,
                ),
                remaining,
            )) => match matcher.get(pattern.clone(), match_candidate.clone()) {
                Some(mat) => {
                    let indexed_datums = if *persist {
                        data.clone()
                    } else {
                        let mut new_prefix = prefix;
                        new_prefix.extend_from_slice(remaining);
                        new_prefix
                    };
                    Some((
                        ConsumeCandidate {
                            channel,
                            datum: Datum {
                                a: mat,
                                persist: *persist,
                                source: produce_ref.clone(),
                            },
                            removed_datum: match_candidate.clone(),
                            datum_index: *data_index,
                        },
                        indexed_datums,
                    ))
                }
                None => {
                    let mut new_prefix = prefix;
                    new_prefix.push((datum.clone(), *data_index));
                    self.find_matching_data_candidate(
                        matcher,
                        channel,
                        remaining.to_vec(),
                        pattern,
                        new_prefix,
                    )
                }
            },
            None => None,
        }
    }

    /** Iterates through (channel, pattern) pairs looking for matching data.
     *
     * Potential match candidates are supplied by the `channelToIndexedData`
     * cache.
     *
     * After a match is found, we remove the matching datum from the
     * candidate cache for remaining matches. If an illegal state is
     * reached when searching a matching candidate we treat it as if no
     * match was found and append the illegal state to result list.
     */
    fn extract_data_candidates(
        &self,
        matcher: &Box<dyn Match<P, A>>,
        channel_pattern_pairs: Vec<(C, P)>,
        channel_to_indexed_data: DashMap<C, Vec<(Datum<A>, i32)>>,
        acc: Vec<Option<ConsumeCandidate<C, A>>>,
    ) -> Vec<Option<ConsumeCandidate<C, A>>> {
        match channel_pattern_pairs.split_first() {
            Some(((channel, pattern), tail)) => {
                let maybe_tuple: Option<MatchingDataCandidate<C, A>> =
                    match channel_to_indexed_data.get(channel) {
                        Some(indexed_data) => {
                            // TODO: Review — Scala does not sort here but passes at the
                            // Rholang level for duplicate-channel joins. Rust needs this
                            // sort to ensure deterministic datum selection regardless of
                            // whether the datum list was shuffled (play) or sequential
                            // (replay).
                            //
                            // When the same channel appears at multiple positions in a
                            // multi-channel join (e.g., [ch, other, ch]), both positions
                            // share the same entry in channel_to_indexed_data. The first
                            // processed position picks a datum and removes it, affecting
                            // what's available for the second position. Without sorting,
                            // play's shuffle produces a different pick order than replay's
                            // sequential order, causing different datum-to-position
                            // assignments. Since Blake2b512Random::merge is order-dependent,
                            // this cascades into different execution paths.
                            //
                            // Duplicate-channel joins arise at runtime when a Receive node's
                            // bound variables resolve to the same channel through
                            // substitution — the normalizer cannot prevent this.
                            let mut sorted_data = indexed_data.clone();
                            sorted_data.sort_by_key(|(_, idx)| *idx);
                            self.find_matching_data_candidate(
                                &matcher,
                                channel.clone(),
                                sorted_data,
                                pattern.clone(),
                                Vec::new(),
                            )
                        }
                        None => None,
                    };

                match maybe_tuple {
                    Some((cand, rem)) => {
                        let mut new_acc = acc;
                        new_acc.push(Some(cand));
                        let new_data = channel_to_indexed_data;
                        new_data.insert(channel.clone(), rem);
                        self.extract_data_candidates(matcher, tail.to_vec(), new_data, new_acc)
                    }
                    None => {
                        let mut new_acc = acc;
                        new_acc.push(None);
                        self.extract_data_candidates(
                            matcher,
                            tail.to_vec(),
                            channel_to_indexed_data,
                            new_acc,
                        )
                    }
                }
            }
            None => acc,
        }
    }

    fn extract_first_match(
        &self,
        matcher: &Box<dyn Match<P, A>>,
        channels: Vec<C>,
        match_candidates: Vec<(WaitingContinuation<P, K>, i32)>,
        channel_to_index_data: DashMap<C, Vec<(Datum<A>, i32)>>,
    ) -> Option<ProduceCandidate<C, P, A, K>> {
        match match_candidates.split_first() {
            Some(((cont @ WaitingContinuation { patterns, .. }, index), remaining)) => {
                let maybe_data_candidates: Option<Vec<ConsumeCandidate<C, A>>> = {
                    let data_candidates = self.extract_data_candidates(
                        matcher,
                        channels.clone().into_iter().zip(patterns.clone()).collect(),
                        channel_to_index_data.clone(),
                        Vec::new(),
                    );
                    if data_candidates.iter().all(|x| x.is_some()) {
                        Some(data_candidates.into_iter().filter_map(|x| x).collect())
                    } else {
                        None
                    }
                };
                match maybe_data_candidates {
                    Some(data_candidates) => Some(ProduceCandidate {
                        channels,
                        continuation: cont.clone(),
                        continuation_index: *index,
                        data_candidates,
                    }),
                    None => self.extract_first_match(
                        matcher,
                        channels,
                        remaining.to_vec(),
                        channel_to_index_data,
                    ),
                }
            }
            None => None,
        }
    }
}
