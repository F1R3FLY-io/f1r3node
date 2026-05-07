use models::rhoapi::var::VarInstance::{FreeVar, Wildcard};
use models::rhoapi::{MatchCase, Par, Var};

use super::has_locally_free::HasLocallyFree;
use super::spatial_matcher::{SpatialMatcher, SpatialMatcherContext};
use crate::rust::interpreter::metrics_constants::{
    RHOLANG_MATCHER_FOLD_MATCH_CALLS_METRIC,
    RHOLANG_MATCHER_FOLD_MATCH_RECURSION_DEPTH_TOTAL_METRIC,
    RHOLANG_MATCHER_FOLD_MATCH_TAIL_CLONE_NS_METRIC, RHOLANG_METRICS_SOURCE,
};

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/matcher/SpatialMatcher.scala - foldMatch
pub trait FoldMatch<T, P> {
    fn fold_match(
        &mut self,
        tlist: Vec<T>,
        plist: Vec<P>,
        remainder: Option<Var>,
    ) -> Option<Vec<T>>;

    fn free_check(&self, trem: &[T], level: i32, acc: Vec<T>) -> Option<Vec<T>>;
}

impl FoldMatch<Par, Par> for SpatialMatcherContext {
    fn fold_match(
        &mut self,
        tlist: Vec<Par>,
        plist: Vec<Par>,
        remainder: Option<Var>,
    ) -> Option<Vec<Par>> {
        // Iterative pair-walk over (tlist, plist) — index into the originals
        // and clone only the per-iteration head pair that `spatial_match`
        // consumes. Total Par clones: O(min(tlist.len(), plist.len())).
        metrics::counter!(RHOLANG_MATCHER_FOLD_MATCH_CALLS_METRIC, "source" => RHOLANG_METRICS_SOURCE)
            .increment(1);
        metrics::counter!(RHOLANG_MATCHER_FOLD_MATCH_RECURSION_DEPTH_TOTAL_METRIC, "source" => RHOLANG_METRICS_SOURCE)
            .increment(tlist.len().max(plist.len()) as u64);

        let n = tlist.len().min(plist.len());
        for i in 0..n {
            let __clone_start = std::time::Instant::now();
            let t_owned = tlist[i].clone();
            let p_owned = plist[i].clone();
            metrics::counter!(RHOLANG_MATCHER_FOLD_MATCH_TAIL_CLONE_NS_METRIC, "source" => RHOLANG_METRICS_SOURCE)
                .increment(__clone_start.elapsed().as_nanos() as u64);
            self.spatial_match(t_owned, p_owned)?;
        }

        if tlist.len() == plist.len() {
            // Exact-length walk consumed both lists.
            Some(Vec::new())
        } else if plist.len() < tlist.len() {
            // Surplus targets — must be absorbed by the remainder var, if any.
            let trem = &tlist[n..];
            match remainder {
                None => None,
                Some(Var {
                    var_instance: Some(FreeVar(level)),
                }) => self.free_check(trem, level, Vec::new()),
                Some(Var {
                    var_instance: Some(Wildcard(_)),
                }) => Some(Vec::new()),
                _ => None,
            }
        } else {
            // Surplus patterns with no targets — no match possible.
            None
        }
    }

    fn free_check(&self, trem: &[Par], level: i32, mut acc: Vec<Par>) -> Option<Vec<Par>> {
        match trem {
            &[] => Some(acc),

            [item, rem @ ..] => {
                if self.locally_free(item.to_owned(), 0).is_empty() {
                    acc.push(item.clone());
                    self.free_check(rem, level, acc)
                } else {
                    None
                }
            }
        }
    }
}

impl FoldMatch<MatchCase, MatchCase> for SpatialMatcherContext {
    fn fold_match(
        &mut self,
        tlist: Vec<MatchCase>,
        plist: Vec<MatchCase>,
        remainder: Option<Var>,
    ) -> Option<Vec<MatchCase>> {
        // Iterative pair-walk; head-pair clone per iteration, no tail-vec clone.
        let n = tlist.len().min(plist.len());
        for i in 0..n {
            self.spatial_match(tlist[i].clone(), plist[i].clone())?;
        }

        if tlist.len() == plist.len() {
            Some(Vec::new())
        } else if plist.len() < tlist.len() {
            let trem = &tlist[n..];
            match remainder {
                None => None,
                Some(Var {
                    var_instance: Some(FreeVar(level)),
                }) => self.free_check(trem, level, Vec::new()),
                Some(Var {
                    var_instance: Some(Wildcard(_)),
                }) => Some(Vec::new()),
                _ => None,
            }
        } else {
            None
        }
    }

    fn free_check(
        &self,
        trem: &[MatchCase],
        level: i32,
        mut acc: Vec<MatchCase>,
    ) -> Option<Vec<MatchCase>> {
        match trem {
            &[] => Some(acc),

            [item, rem @ ..] => {
                if self.locally_free(item.to_owned(), 0).is_empty() {
                    acc.push(item.clone());
                    self.free_check(rem, level, acc)
                } else {
                    None
                }
            }
        }
    }
}
