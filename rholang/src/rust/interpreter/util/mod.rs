use models::{
    rhoapi::{Bundle, Connective, EList, Expr, ListParWithRandom, Match, New, Par, Receive, Send, UseBlock},
    rhoapi::expr::ExprInstance,
    rust::utils::union,
};

use super::matcher::has_locally_free::HasLocallyFree;

pub mod address_tools;
pub mod base58;
pub mod rev_address;

// Helper enum. This is 'GeneratedMessage' in Scala
#[derive(Clone, Debug)]
pub enum GeneratedMessage {
    Send(Send),
    Receive(Receive),
    New(New),
    Match(Match),
    Bundle(Bundle),
    Expr(Expr),
    UseBlock(UseBlock),
}

// These two functions need to be under 'rholang' dir because of HasLocallyFree Trait.
// This trait should, I think, be moved to models

// See models/src/main/scala/coop/rchain/models/rholang/implicits.scala - prepend
pub fn prepend_connective(mut p: Par, c: Connective, depth: i32) -> Par {
    let mut new_connectives = vec![c.clone()];
    new_connectives.append(&mut p.connectives);

    Par {
        connectives: new_connectives,
        locally_free: c.locally_free(c.clone(), depth),
        connective_used: p.connective_used || c.clone().connective_used(c),
        ..p.clone()
    }
}

pub fn prepend_expr(mut p: Par, e: Expr, depth: i32) -> Par {
    let mut new_exprs = vec![e.clone()];
    new_exprs.append(&mut p.exprs);

    Par {
        exprs: new_exprs,
        locally_free: union(p.locally_free.clone(), e.locally_free(e.clone(), depth)),
        connective_used: p.connective_used || e.clone().connective_used(e),
        ..p.clone()
    }
}

pub fn prepend_new(mut p: Par, n: New) -> Par {
    let mut new_news = vec![n.clone()];
    new_news.append(&mut p.news);

    Par {
        news: new_news,
        locally_free: union(p.locally_free.clone(), n.clone().locally_free),
        connective_used: p.connective_used || n.clone().connective_used(n),
        ..p.clone()
    }
}

pub fn prepend_bundle(mut p: Par, b: Bundle) -> Par {
    let mut new_bundles = vec![b.clone()];
    new_bundles.append(&mut p.bundles);

    Par {
        bundles: new_bundles,
        locally_free: union(p.locally_free.clone(), b.body.unwrap().locally_free),
        ..p.clone()
    }
}

/// Prepend a UseBlock to a Par, updating locally_free and connective_used.
///
/// UseBlocks implement scoped default space selection for Reifying RSpaces.
/// The space expression and body are evaluated within the UseBlock scope.
///
/// Formal Correspondence:
/// - Registry/Invariants.v: inv_use_blocks_valid
/// - GenericRSpace.v: UseBlock scope management
pub fn prepend_use_block(mut p: Par, ub: UseBlock) -> Par {
    let mut new_use_blocks = vec![ub.clone()];
    new_use_blocks.append(&mut p.use_blocks);

    Par {
        use_blocks: new_use_blocks,
        locally_free: union(p.locally_free.clone(), ub.locally_free.clone()),
        connective_used: p.connective_used || ub.connective_used,
        ..p.clone()
    }
}

// for locally_free parameter, in case when we have (bodyResult.par.locallyFree.from(boundCount).map(x => x - boundCount))
pub(crate) fn filter_and_adjust_bitset(bitset: Vec<u8>, bound_count: usize) -> Vec<u8> {
    bitset
        .into_iter()
        .enumerate()
        .filter_map(|(i, _)| {
            if i >= bound_count {
                Some(i as u8 - bound_count as u8)
            } else {
                None
            }
        })
        .collect()
}

/// Wrap data with suffix key for PathMap prefix aggregation semantics.
///
/// Per the "Reifying RSpaces" spec (lines 163-184):
/// - Data at `@[0,1,2]` consumed at prefix `@[0,1]` becomes `[2, data]`
/// - The suffix key elements are prepended to the data as a list
///
/// # Arguments
/// * `data` - The original `ListParWithRandom` data to wrap
/// * `suffix_key` - The path suffix between consume prefix and actual data channel
///
/// # Returns
/// A new `ListParWithRandom` with suffix key prepended:
/// - For single-element suffix `[2]`: creates `[2, original_data...]`
/// - For multi-element suffix `[2,3]`: creates `[[2,3], original_data...]`
/// - For empty suffix (exact match): returns original data unchanged
///
/// # Formal Correspondence
/// - `PathMapStore.v`: `send_visible_from_prefix` theorem
/// - `PathMapQuantale.v`: Path concatenation properties
///
/// # Example
/// ```ignore
/// // Data "hi" at @[0,1,2] consumed at @[0,1] with suffix [2]:
/// // becomes [2, "hi"]
/// let wrapped = wrap_with_suffix_key(data, &vec![2]);
/// ```
pub fn wrap_with_suffix_key(data: ListParWithRandom, suffix_key: &[u8]) -> ListParWithRandom {
    // Empty suffix means exact match - no wrapping needed
    if suffix_key.is_empty() {
        return data;
    }

    // Convert suffix key bytes to a Par
    let suffix_par = suffix_key_to_par(suffix_key);

    // Prepend the suffix Par to the data's pars
    let mut wrapped_pars = vec![suffix_par];
    wrapped_pars.extend(data.pars);

    ListParWithRandom {
        pars: wrapped_pars,
        random_state: data.random_state,
    }
}

/// Convert a suffix key (path bytes) to a Rholang Par representation.
///
/// # Suffix Key Representation
/// - Single byte `[n]`: becomes `GInt(n)` - a simple integer
/// - Multiple bytes `[a, b, c, ...]`: becomes `[a, b, c, ...]` - a list of integers
///
/// This follows the spec where data at `@[0,1,2]` viewed at `@[0,1]` has suffix `[2]`,
/// which becomes the integer `2` prepended to the data. For deeper nesting like
/// `@[0,1,2,3]` viewed at `@[0,1]`, the suffix `[2,3]` becomes the list `[2,3]`.
fn suffix_key_to_par(suffix_key: &[u8]) -> Par {
    if suffix_key.len() == 1 {
        // Single element - return as GInt
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::GInt(suffix_key[0] as i64)),
        }])
    } else {
        // Multiple elements - return as EList of GInt
        let elements: Vec<Par> = suffix_key
            .iter()
            .map(|&byte| {
                Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::GInt(byte as i64)),
                }])
            })
            .collect();

        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EListBody(EList {
                ps: elements,
                locally_free: vec![],
                connective_used: false,
                remainder: None,
            })),
        }])
    }
}

#[cfg(test)]
mod suffix_key_tests {
    use super::*;

    #[test]
    fn test_wrap_empty_suffix_no_change() {
        let data = ListParWithRandom {
            pars: vec![Par::default()],
            random_state: vec![],
        };
        let wrapped = wrap_with_suffix_key(data.clone(), &[]);
        assert_eq!(wrapped.pars.len(), data.pars.len());
    }

    #[test]
    fn test_wrap_single_element_suffix() {
        let data = ListParWithRandom {
            pars: vec![Par::default()],
            random_state: vec![1, 2, 3],
        };
        let wrapped = wrap_with_suffix_key(data, &[2]);

        // Should have 2 elements: [suffix, original_data]
        assert_eq!(wrapped.pars.len(), 2);

        // First element should be GInt(2)
        let suffix_par = &wrapped.pars[0];
        assert!(!suffix_par.exprs.is_empty());
        match &suffix_par.exprs[0].expr_instance {
            Some(ExprInstance::GInt(n)) => assert_eq!(*n, 2),
            _ => panic!("Expected GInt for single-element suffix"),
        }
    }

    #[test]
    fn test_wrap_multi_element_suffix() {
        let data = ListParWithRandom {
            pars: vec![Par::default()],
            random_state: vec![],
        };
        let wrapped = wrap_with_suffix_key(data, &[2, 3, 4]);

        // Should have 2 elements: [suffix_list, original_data]
        assert_eq!(wrapped.pars.len(), 2);

        // First element should be EList of GInts
        let suffix_par = &wrapped.pars[0];
        assert!(!suffix_par.exprs.is_empty());
        match &suffix_par.exprs[0].expr_instance {
            Some(ExprInstance::EListBody(list)) => {
                assert_eq!(list.ps.len(), 3);
                // Verify elements are 2, 3, 4
                for (i, expected) in [2i64, 3, 4].iter().enumerate() {
                    match &list.ps[i].exprs[0].expr_instance {
                        Some(ExprInstance::GInt(n)) => assert_eq!(n, expected),
                        _ => panic!("Expected GInt in suffix list"),
                    }
                }
            }
            _ => panic!("Expected EList for multi-element suffix"),
        }
    }

    #[test]
    fn test_wrap_preserves_random_state() {
        let data = ListParWithRandom {
            pars: vec![Par::default()],
            random_state: vec![42, 43, 44],
        };
        let wrapped = wrap_with_suffix_key(data.clone(), &[5]);

        assert_eq!(wrapped.random_state, data.random_state);
    }
}
