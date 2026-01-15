// See models/src/main/scala/coop/rchain/models/rholang/sorter/NewSortMatcher.scala

use crate::{
    rhoapi::{New, Par},
    rust::rholang::sorter::score_tree::Score,
};

use super::{
    par_sort_matcher::ParSortMatcher,
    score_tree::{ScoreAtom, ScoredTerm, Tree},
    sortable::Sortable,
};

pub struct NewSortMatcher;

impl Sortable<New> for NewSortMatcher {
    fn sort_match(n: &New) -> ScoredTerm<New> {
        let sorted_par = ParSortMatcher::sort_match(
            n.p.as_ref()
                .expect("p field on New was None, should be Some"),
        );

        let mut sorted_uri = n.uri.clone();
        sorted_uri.sort();

        let uri_score = if !sorted_uri.is_empty() {
            sorted_uri
                .clone()
                .into_iter()
                .map(|s| Tree::<ScoreAtom>::create_leaf_from_string(s))
                .collect()
        } else {
            vec![Tree::<ScoreAtom>::create_leaf_from_i64(
                Score::ABSENT as i64,
            )]
        };

        let injections_list: Vec<(String, Par)> = n.injections.clone().into_iter().collect();
        let injections_score = if !injections_list.is_empty() {
            injections_list
                .iter()
                .map(|(k, v)| {
                    let scored_term = ParSortMatcher::sort_match(v);
                    Tree::<ScoreAtom>::create_node_from_string(k.clone(), vec![scored_term.score])
                })
                .collect()
        } else {
            vec![Tree::<ScoreAtom>::create_leaf_from_i64(
                Score::ABSENT as i64,
            )]
        };

        ScoredTerm {
            term: New {
                bind_count: n.bind_count,
                p: Some(sorted_par.term),
                uri: sorted_uri,
                injections: n.injections.clone(),
                locally_free: n.locally_free.clone(),
                space_types: n.space_types.clone(),
            },
            score: Tree::Node(
                std::iter::once(Tree::<ScoreAtom>::create_leaf_from_i64(Score::NEW as i64))
                    .chain(std::iter::once(Tree::<ScoreAtom>::create_leaf_from_i64(
                        n.bind_count as i64,
                    )))
                    .chain(uri_score.into_iter())
                    .chain(injections_score.into_iter())
                    .chain(std::iter::once(sorted_par.score))
                    .collect(),
            ),
        }
    }
}
