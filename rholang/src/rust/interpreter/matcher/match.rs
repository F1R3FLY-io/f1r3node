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

/// Produce a concise human-readable summary of a Par for diagnostic logs.
fn summarize_par(par: &Par) -> String {
    // Single expression — most common case for pattern elements
    if par.exprs.len() == 1
        && par.sends.is_empty()
        && par.receives.is_empty()
        && par.news.is_empty()
        && par.unforgeables.is_empty()
        && par.bundles.is_empty()
        && par.connectives.is_empty()
    {
        match &par.exprs[0].expr_instance {
            Some(GString(s)) => return format!("@\"{}\"", s),
            Some(GInt(n)) => return format!("@{}", n),
            Some(GBool(b)) => return format!("@{}", b),
            Some(GUri(u)) => return format!("@`{}`", u),
            Some(GByteArray(bytes)) => return format!("@bytes({})", bytes.len()),
            Some(EVarBody(EVar {
                v:
                    Some(Var {
                        var_instance: Some(FreeVar(level)),
                    }),
            })) => return format!("<free:{}>", level),
            Some(EVarBody(EVar {
                v:
                    Some(Var {
                        var_instance: Some(Wildcard(_)),
                    }),
            })) => return "<wildcard>".to_string(),
            Some(EVarBody(EVar {
                v:
                    Some(Var {
                        var_instance: Some(BoundVar(level)),
                    }),
            })) => return format!("<bound:{}>", level),
            Some(EListBody(_)) => return "<list>".to_string(),
            Some(ETupleBody(_)) => return "<tuple>".to_string(),
            Some(EMapBody(_)) => return "<map>".to_string(),
            Some(ESetBody(_)) => return "<set>".to_string(),
            _ => return format!("<expr:{:?}>", par.exprs[0].expr_instance),
        }
    }

    // Single unforgeable (channel name)
    if par.unforgeables.len() == 1
        && par.exprs.is_empty()
        && par.sends.is_empty()
        && par.receives.is_empty()
    {
        return "<unforgeable>".to_string();
    }

    // Connective pattern
    if !par.connectives.is_empty() && par.connective_used {
        return format!(
            "<connective:conn_used={},exprs={},conn={}>",
            par.connective_used,
            par.exprs.len(),
            par.connectives.len()
        );
    }

    // Empty par
    if par.sends.is_empty()
        && par.receives.is_empty()
        && par.news.is_empty()
        && par.exprs.is_empty()
        && par.unforgeables.is_empty()
        && par.bundles.is_empty()
        && par.connectives.is_empty()
    {
        return "<nil>".to_string();
    }

    format!(
        "<par:sends={},recvs={},news={},exprs={},unforg={},bundles={},conn={},conn_used={}>",
        par.sends.len(),
        par.receives.len(),
        par.news.len(),
        par.exprs.len(),
        par.unforgeables.len(),
        par.bundles.len(),
        par.connectives.len(),
        par.connective_used
    )
}

/// Produce a concise summary of a BindPattern's patterns.
fn summarize_bind_pattern(bp: &BindPattern) -> Vec<String> {
    bp.patterns.iter().map(|p| summarize_par(p)).collect()
}

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/storage/package.scala - matchListPar
impl Match<BindPattern, ListParWithRandom> for Matcher {
    fn get(&self, pattern: BindPattern, data: ListParWithRandom) -> Option<ListParWithRandom> {
        let mut spatial_matcher = SpatialMatcherContext::new();

        if tracing::enabled!(target: "f1r3fly.rholang.matcher", tracing::Level::DEBUG) {
            let pattern_summary = summarize_bind_pattern(&pattern);
            let data_summary: Vec<String> =
                data.pars.iter().map(|p| summarize_par(p)).collect();
            tracing::debug!(
                target: "f1r3fly.rholang.matcher",
                pattern_count = pattern.patterns.len(),
                data_count = data.pars.len(),
                free_count = pattern.free_count,
                has_remainder = pattern.remainder.is_some(),
                pattern_summary = ?pattern_summary,
                data_summary = ?data_summary,
                "Matcher::get: checking BindPattern({} patterns) vs data({} pars)",
                pattern.patterns.len(),
                data.pars.len()
            );
        }

        let fold_match_result =
            spatial_matcher.fold_match(data.pars, pattern.patterns, pattern.remainder.clone());
        let match_result = match fold_match_result {
            Some(pars) => Some((spatial_matcher.free_map, pars)),
            None => None,
        };

        if tracing::enabled!(target: "f1r3fly.rholang.matcher", tracing::Level::DEBUG) {
            tracing::debug!(
                target: "f1r3fly.rholang.matcher",
                matched = match_result.is_some(),
                "Matcher::get: fold_match → {}",
                if match_result.is_some() { "MATCHED" } else { "no match" }
            );
        }

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

        result
    }
}
