// See models/src/main/scala/coop/rchain/models/rholang/sorter/SendSortMatcher.scala

use crate::rhoapi::{Par, Send};

use super::{
    par_sort_matcher::ParSortMatcher,
    score_tree::{Score, ScoreAtom, ScoredTerm, Tree},
    sortable::Sortable,
};

pub struct SendSortMatcher;

impl Sortable<Send> for SendSortMatcher {
    fn sort_match(s: &Send) -> ScoredTerm<Send> {
        let sorted_chan = ParSortMatcher::sort_match(
            s.chan
                .as_ref()
                .expect("channel field on Send was None, should be Some"),
        );

        let sorted_data: Vec<ScoredTerm<Par>> = s
            .data
            .iter()
            .map(|p| ParSortMatcher::sort_match(p))
            .collect();

        let sorted_send = Send {
            chan: Some(sorted_chan.term),
            data: sorted_data.clone().into_iter().map(|p| p.term).collect(),
            persistent: s.persistent,
            locally_free: s.locally_free.clone(),
            connective_used: s.connective_used,
            hyperparams: s.hyperparams.clone(),  // Hyperparam support
        };

        let persistent_score: i64 = if s.persistent { 1 } else { 0 };
        let connective_used_score: i64 = if s.connective_used { 1 } else { 0 };
        let send_score = Tree::<ScoreAtom>::create_node_from_i32(
            Score::SEND,
            vec![
                Tree::<ScoreAtom>::create_leaf_from_i64(persistent_score),
                sorted_chan.score,
            ]
            .into_iter()
            .chain(sorted_data.into_iter().map(|p| p.score))
            .chain(vec![Tree::<ScoreAtom>::create_leaf_from_i64(
                connective_used_score,
            )])
            .collect(),
        );

        ScoredTerm {
            term: sorted_send,
            score: send_score,
        }
    }
}
