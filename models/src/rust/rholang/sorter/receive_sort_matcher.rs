// See models/src/main/scala/coop/rchain/models/rholang/sorter/ReceiveSortMatcher.scala

use crate::rhoapi::{Par, Receive, ReceiveBind};

use super::{
    par_sort_matcher::ParSortMatcher,
    score_tree::{Score, ScoreAtom, ScoredTerm, Tree},
    sortable::Sortable,
    var_sort_matcher::VarSortMatcher,
};

pub struct ReceiveSortMatcher;

impl ReceiveSortMatcher {
    pub fn sort_bind(bind: ReceiveBind) -> ScoredTerm<ReceiveBind> {
        let patterns = bind.patterns;
        let source = bind
            .source
            .expect("source field on Bind was None, should be Some");

        let sorted_patterns: Vec<ScoredTerm<Par>> = patterns
            .into_iter()
            .map(|p| ParSortMatcher::sort_match(&p))
            .collect();
        let sorted_channel = ParSortMatcher::sort_match(&source);
        let sorted_remainder = match &bind.remainder {
            Some(bind_remainder) => {
                let scored_var = VarSortMatcher::sort_match(&bind_remainder);
                ScoredTerm {
                    term: Some(scored_var.term),
                    score: scored_var.score,
                }
            }
            None => ScoredTerm {
                term: None,
                score: Tree::<ScoreAtom>::create_leaf_from_i64(Score::ABSENT as i64),
            },
        };

        ScoredTerm {
            term: ReceiveBind {
                patterns: sorted_patterns
                    .clone()
                    .into_iter()
                    .map(|p| p.term)
                    .collect(),
                source: Some(sorted_channel.term),
                remainder: bind.remainder,
                free_count: bind.free_count,
                pattern_modifiers: bind.pattern_modifiers.clone(),
            },
            score: Tree::Node(
                vec![sorted_channel.score]
                    .into_iter()
                    .chain(sorted_patterns.into_iter().map(|p| p.score))
                    .chain(vec![sorted_remainder.score].into_iter())
                    .collect(),
            ),
        }
    }
}

impl Sortable<Receive> for ReceiveSortMatcher {
    // The order of the binds must already be presorted by the time this is called.
    // This function will then sort the insides of the preordered binds.
    fn sort_match(r: &Receive) -> ScoredTerm<Receive> {
        let sorted_binds: Vec<ScoredTerm<ReceiveBind>> = r
            .binds
            .clone()
            .into_iter()
            .map(|rb| ReceiveSortMatcher::sort_bind(rb))
            .collect();

        let persistent_score: i64 = if r.persistent { 1 } else { 0 };
        let peek_score: i64 = if r.peek { 1 } else { 0 };
        let connective_used_score: i64 = if r.connective_used { 1 } else { 0 };
        let sorted_body = ParSortMatcher::sort_match(
            r.body
                .as_ref()
                .expect("body field on Receive was None, should be Some"),
        );

        ScoredTerm {
            term: Receive {
                binds: sorted_binds.clone().into_iter().map(|rb| rb.term).collect(),
                body: Some(sorted_body.term),
                persistent: r.persistent,
                peek: r.peek,
                bind_count: r.bind_count,
                locally_free: r.locally_free.clone(),
                connective_used: r.connective_used,
            },
            score: Tree::<ScoreAtom>::create_node_from_i32(
                Score::RECEIVE,
                vec![
                    Tree::<ScoreAtom>::create_leaf_from_i64(persistent_score),
                    Tree::<ScoreAtom>::create_leaf_from_i64(peek_score),
                ]
                .into_iter()
                .chain(sorted_binds.into_iter().map(|rb| rb.score))
                .chain(vec![sorted_body.score])
                .chain(vec![
                    Tree::<ScoreAtom>::create_leaf_from_i64(r.bind_count as i64),
                    Tree::<ScoreAtom>::create_leaf_from_i64(connective_used_score),
                ])
                .collect(),
            ),
        }
    }
}
