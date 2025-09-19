use models::rust::{
    rholang::implicits::vector_par,
    utils::{new_elist_expr, to_vec},
};
use rspace_plus_plus::rspace::r#match::Match;
use std::marker::{Send, Sync};

use super::{exports::*, fold_match::FoldMatch, spatial_matcher::SpatialMatcherContext};

#[derive(Clone, Default)]
pub struct Matcher;

// Matcher must implement Send + Sync to satisfy Match trait bounds
unsafe impl Send for Matcher {}
unsafe impl Sync for Matcher {}

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/storage/package.scala - matchListPar
impl Match<BindPattern, ListParWithRandom> for Matcher {
    fn get(&self, pattern: BindPattern, data: ListParWithRandom) -> Option<ListParWithRandom> {
        let mut spatial_matcher = SpatialMatcherContext::new();

        // println!("\npattern in get: {:?}", pattern);
        // println!("\ndata in get: {:?}", data);

        let fold_match_result =
            spatial_matcher.fold_match(data.pars, pattern.patterns, pattern.remainder.clone());
        let match_result = match fold_match_result {
            Some(pars) => Some((spatial_matcher.free_map, pars)),
            None => None,
        };

        // println!("\nmatch_result: {:?}", match_result);

        let result = match match_result {
            Some((mut free_map, caught_rem)) => {
                let remainder_map = match pattern.remainder {
                    Some(Var {
                        var_instance: Some(FreeVar(level)),
                    }) => {
                        free_map.insert(
                            level,
                            vector_par(Vec::new(), false).with_exprs(vec![new_elist_expr(
                                caught_rem,
                                Vec::new(),
                                false,
                                None,
                            )]),
                        );
                        free_map
                    }
                    _ => free_map,
                };
                Some(ListParWithRandom {
                    pars: to_vec(remainder_map, pattern.free_count),
                    random_state: data.random_state,
                })
            }
            None => None,
        };

        // println!("\nresult: {:?}", result);
        result
    }
}
