// See rspace/src/main/scala/coop/rchain/rspace/SpaceMatcher.scala

use dashmap::DashMap;

use super::r#match::Match;
use super::rspace_interface::ISpace;
use crate::rspace::internal::{ConsumeCandidate, Datum, ProduceCandidate, WaitingContinuation};

type MatchingDataCandidate<C, A> = (ConsumeCandidate<C, A>, Vec<(Datum<A>, i32)>);

pub trait SpaceMatcher<C, P, A, K>: ISpace<C, P, A, K>
where
    C: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    P: Clone + std::fmt::Debug,
    A: Clone + std::fmt::Debug,
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
                    // println!("\npattern: {:?}", pattern);
                    // println!("\nmatch_candidate: {:?}", match_candidate);

                    // println!("\nmatch found! Match: {:?}", mat);

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
                    // println!("\npattern: {:?}", pattern);
                    // println!("\nmatch_candidate: {:?}", match_candidate);
                    // println!("\nno match found!");

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
        // println!("\nHit extract_data_candidates");
        match channel_pattern_pairs.last() {
            Some((channel, pattern)) => {
                let maybe_tuple: Option<MatchingDataCandidate<C, A>> =
                    match channel_to_indexed_data.get(&channel) {
                        Some(indexed_data) => {
                            // println!("\nCalling findMatchingDataCandidate");
                            self.find_matching_data_candidate(
                                &matcher,
                                channel.clone(),
                                indexed_data.clone(),
                                pattern.clone(),
                                Vec::new(),
                            )
                        }
                        None => {
                            // println!("\nHitting None in maybeTuple");
                            None
                        }
                    };

                // println!("\nmaybe_tuple: {:?}", maybe_tuple);

                match maybe_tuple {
                    Some((cand, rem)) => {
                        let mut new_acc = acc;
                        new_acc.push(Some(cand));
                        let mut new_pairs = channel_pattern_pairs.clone();
                        new_pairs.pop();
                        let new_data = channel_to_indexed_data;
                        new_data.insert(channel.clone(), rem);
                        self.extract_data_candidates(matcher, new_pairs, new_data, new_acc)
                    }
                    None => {
                        let mut new_acc = acc;
                        new_acc.push(None);
                        let mut new_pairs = channel_pattern_pairs;
                        new_pairs.pop();
                        self.extract_data_candidates(
                            matcher,
                            new_pairs,
                            channel_to_indexed_data,
                            new_acc,
                        )
                    }
                }
            }
            None => acc.into_iter().rev().collect(),
        }
    }

    fn extract_first_match(
        &self,
        matcher: &Box<dyn Match<P, A>>,
        channels: Vec<C>,
        match_candidates: Vec<(WaitingContinuation<P, K>, i32)>,
        channel_to_index_data: DashMap<C, Vec<(Datum<A>, i32)>>,
    ) -> Option<ProduceCandidate<C, P, A, K>> {
        if tracing::enabled!(target: "f1r3fly.rspace.matcher", tracing::Level::DEBUG) {
            let data_summary: Vec<_> = channel_to_index_data
                .iter()
                .map(|entry| {
                    let datums: Vec<_> = entry.value().iter().map(|(d, idx)| (*idx, format!("{:?}", d.a))).collect();
                    (format!("{:?}", entry.key()), datums)
                })
                .collect();
            tracing::debug!(
                target: "f1r3fly.rspace.matcher",
                channels = ?channels,
                num_candidates = match_candidates.len(),
                data_summary = ?data_summary,
                "extract_first_match: starting with {} candidates",
                match_candidates.len()
            );
        }

        match match_candidates.last() {
            Some((cont @ WaitingContinuation { patterns, .. }, index)) => {
                tracing::debug!(
                    target: "f1r3fly.rspace.matcher",
                    cont_index = *index,
                    num_patterns = patterns.len(),
                    persist = cont.persist,
                    patterns = ?patterns,
                    "extract_first_match: trying continuation #{} ({} patterns, persist={})",
                    index, patterns.len(), cont.persist
                );

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

                let matched = maybe_data_candidates.is_some();
                tracing::debug!(
                    target: "f1r3fly.rspace.matcher",
                    cont_index = *index,
                    matched,
                    "extract_first_match: continuation #{} → {}",
                    index, if matched { "MATCHED" } else { "rejected" }
                );

                match maybe_data_candidates {
                    Some(data_candidates) => Some(ProduceCandidate {
                        channels,
                        continuation: cont.clone(),
                        continuation_index: *index,
                        data_candidates,
                    }),
                    None => {
                        let mut new_candidates = match_candidates;
                        new_candidates.pop();
                        self.extract_first_match(
                            matcher,
                            channels,
                            new_candidates,
                            channel_to_index_data,
                        )
                    }
                }
            }
            None => {
                tracing::debug!(
                    target: "f1r3fly.rspace.matcher",
                    channels = ?channels,
                    "extract_first_match: no candidates matched"
                );
                None
            }
        }
    }
}
