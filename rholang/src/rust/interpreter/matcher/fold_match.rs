use models::rhoapi::var::VarInstance::{FreeVar, Wildcard};
use models::rhoapi::{MatchCase, Par, Var};

use super::has_locally_free::HasLocallyFree;
use super::spatial_matcher::{SpatialMatcher, SpatialMatcherContext};

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
        match (tlist.as_slice(), plist.as_slice()) {
            (&[], &[]) => {
                Some(Vec::new())
            }

            (&[], _) => {
                tracing::trace!(
                    target: "f1r3fly.rholang.matcher",
                    remaining_patterns = plist.len(),
                    "fold_match: data exhausted but {} patterns remain → no match",
                    plist.len()
                );
                None
            }

            (trem, &[]) => match remainder {
                None => {
                    tracing::trace!(
                        target: "f1r3fly.rholang.matcher",
                        remaining_data = trem.len(),
                        "fold_match: patterns exhausted but {} data remain, no remainder → no match",
                        trem.len()
                    );
                    None
                }

                Some(Var {
                    var_instance: Some(FreeVar(level)),
                }) => self.free_check(trem, level, Vec::new()),

                Some(Var {
                    var_instance: Some(Wildcard(_)),
                }) => Some(Vec::new()),

                _ => None,
            },

            ([t, trem @ ..], [p, prem @ ..]) => {
                let match_result = self.spatial_match(t.to_owned(), p.to_owned());

                if match_result.is_none() {
                    tracing::debug!(
                        target: "f1r3fly.rholang.matcher",
                        target_connective_used = t.connective_used,
                        pattern_connective_used = p.connective_used,
                        target_exprs = t.exprs.len(),
                        pattern_exprs = p.exprs.len(),
                        remaining_pairs = prem.len(),
                        "fold_match: spatial_match FAILED at element {}/{} — \
                         target.exprs={:?}, pattern.exprs={:?}",
                        plist.len() - prem.len(),
                        plist.len(),
                        t.exprs.iter().map(|e| format!("{:?}", e.expr_instance)).collect::<Vec<_>>(),
                        p.exprs.iter().map(|e| format!("{:?}", e.expr_instance)).collect::<Vec<_>>()
                    );
                }

                match_result
                    .and_then(|_| self.fold_match(trem.to_vec(), prem.to_vec(), remainder))
            }
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
        // println!("\nHit fold_match");

        match (tlist.as_slice(), plist.as_slice()) {
            (&[], &[]) => Some(Vec::new()),

            (&[], _) => None,

            (trem, &[]) => match remainder {
                None => None,

                Some(Var {
                    var_instance: Some(FreeVar(level)),
                }) => self.free_check(trem, level, Vec::new()),

                Some(Var {
                    var_instance: Some(Wildcard(_)),
                }) => Some(Vec::new()),

                _ => None,
            },

            ([t, trem @ ..], [p, prem @ ..]) => self
                .spatial_match(t.to_owned(), p.to_owned())
                .and_then(|_| self.fold_match(trem.to_vec(), prem.to_vec(), remainder)),
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
