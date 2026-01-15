// See models/src/main/scala/coop/rchain/models/rholang/sorter/ParSortMatcher.scala

use crate::{
    rhoapi::{Bundle, Connective, Expr, GUnforgeable, Match, New, Par, Receive, Send},
    rust::rholang::sorter::{
        bundle_sort_matcher::BundleSortMatcher,
        connective_sort_matcher::ConnectiveSortMatcher,
        expr_sort_matcher::ExprSortMatcher,
        match_sort_matcher::MatchSortMatcher,
        new_sort_matcher::NewSortMatcher,
        receive_sort_matcher::ReceiveSortMatcher,
        score_tree::{Score, ScoreAtom, Tree},
        unforgeable_sort_matcher::UnforgeableSortMatcher,
    },
};

use super::{score_tree::ScoredTerm, send_sort_matcher::SendSortMatcher, sortable::Sortable};

pub struct ParSortMatcher;

impl Sortable<Par> for ParSortMatcher {
    fn sort_match(par: &Par) -> ScoredTerm<Par> {
        let sends: Vec<ScoredTerm<Send>> = {
            let mut _sends: Vec<ScoredTerm<Send>> = par
                .sends
                .iter()
                .map(|s| SendSortMatcher::sort_match(s))
                .collect();

            ScoredTerm::sort_vec(&mut _sends);
            _sends
        };

        let receives: Vec<ScoredTerm<Receive>> = {
            let mut _receives: Vec<ScoredTerm<Receive>> = par
                .receives
                .iter()
                .map(|r| ReceiveSortMatcher::sort_match(r))
                .collect();

            ScoredTerm::sort_vec(&mut _receives);
            _receives
        };

        let exprs: Vec<ScoredTerm<Expr>> = {
            let mut _exprs: Vec<ScoredTerm<Expr>> = par
                .exprs
                .iter()
                .map(|e| ExprSortMatcher::sort_match(e))
                .collect();

            ScoredTerm::sort_vec(&mut _exprs);
            _exprs
        };

        let news: Vec<ScoredTerm<New>> = {
            let mut _news: Vec<ScoredTerm<New>> = par
                .news
                .iter()
                .map(|n| NewSortMatcher::sort_match(n))
                .collect();

            ScoredTerm::sort_vec(&mut _news);
            _news
        };

        let matches: Vec<ScoredTerm<Match>> = {
            let mut _matches: Vec<ScoredTerm<Match>> = par
                .matches
                .iter()
                .map(|m| MatchSortMatcher::sort_match(m))
                .collect();

            ScoredTerm::sort_vec(&mut _matches);
            _matches
        };

        let bundles: Vec<ScoredTerm<Bundle>> = {
            let mut _bundles: Vec<ScoredTerm<Bundle>> = par
                .bundles
                .iter()
                .map(|b| BundleSortMatcher::sort_match(b))
                .collect();

            ScoredTerm::sort_vec(&mut _bundles);
            _bundles
        };

        let connectives: Vec<ScoredTerm<Connective>> = {
            let mut _connectives: Vec<ScoredTerm<Connective>> = par
                .connectives
                .iter()
                .map(|c| ConnectiveSortMatcher::sort_match(c))
                .collect();

            ScoredTerm::sort_vec(&mut _connectives);
            _connectives
        };

        let unforgeables: Vec<ScoredTerm<GUnforgeable>> = {
            let mut _unforgeables: Vec<ScoredTerm<GUnforgeable>> = par
                .unforgeables
                .iter()
                .map(|gu| UnforgeableSortMatcher::sort_match(gu))
                .collect();

            ScoredTerm::sort_vec(&mut _unforgeables);
            _unforgeables
        };

        let sorted_par = Par {
            sends: sends.clone().into_iter().map(|s| s.term).collect(),
            receives: receives.clone().into_iter().map(|r| r.term).collect(),
            news: news.clone().into_iter().map(|n| n.term).collect(),
            exprs: exprs.clone().into_iter().map(|e| e.term).collect(),
            matches: matches.clone().into_iter().map(|m| m.term).collect(),
            unforgeables: unforgeables.clone().into_iter().map(|gu| gu.term).collect(),
            bundles: bundles.clone().into_iter().map(|b| b.term).collect(),
            connectives: connectives.clone().into_iter().map(|c| c.term).collect(),
            locally_free: par.locally_free.clone(),
            connective_used: par.connective_used,
            use_blocks: par.use_blocks.clone(),  // Reifying RSpaces
        };

        let connective_used_score: i64 = if par.connective_used { 1 } else { 0 };
        let par_score = Tree::<ScoreAtom>::create_node_from_i32(
            Score::PAR as i32,
            sends
                .into_iter()
                .map(|s| s.score)
                .chain(
                    receives
                        .into_iter()
                        .map(|r| r.score)
                        .chain(exprs.into_iter().map(|e| e.score))
                        .chain(news.into_iter().map(|n| n.score))
                        .chain(matches.into_iter().map(|m| m.score))
                        .chain(bundles.into_iter().map(|b| b.score))
                        .chain(connectives.into_iter().map(|c| c.score))
                        .chain(unforgeables.into_iter().map(|gu| gu.score))
                        .chain(vec![Tree::<ScoreAtom>::create_leaf_from_i64(
                            connective_used_score,
                        )]),
                )
                .collect(),
        );

        ScoredTerm {
            term: sorted_par,
            score: par_score,
        }
    }
}
