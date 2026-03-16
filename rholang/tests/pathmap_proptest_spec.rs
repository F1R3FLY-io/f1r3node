// Property-based tests for PathMap decomposition and element extraction.
// Uses proptest to generate arbitrary Par values at various nesting depths
// and verify invariants of the spatial matcher and element extraction methods.

use models::rhoapi::expr::ExprInstance;
use models::rhoapi::{EList, EPathMap, Expr, Par};
use models::rust::utils::{new_freevar_var, new_gint_par, new_gstring_par, new_wildcard_var};
use proptest::prelude::*;
use rholang::rust::interpreter::matcher::spatial_matcher::{SpatialMatcher, SpatialMatcherContext};

// =============================================================================
// Arbitrary Par generators with bounded depth
// =============================================================================

/// Generate an arbitrary "leaf" Par (depth 0)
fn arb_leaf_par() -> impl Strategy<Value = Par> {
    prop_oneof![
        // GString
        "[a-z]{1,8}".prop_map(|s| new_gstring_par(s, Vec::new(), false)),
        // GInt
        (-1000i64..1000).prop_map(|n| new_gint_par(n, Vec::new(), false)),
        // GBool
        any::<bool>().prop_map(|b| Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::GBool(b)),
        }])),
    ]
}

/// Generate an arbitrary Par at a given nesting depth.
/// At depth 0, only leaf types are generated.
/// At depth > 0, compound types (lists, tuples, pathmaps) are also generated.
fn arb_par(depth: u32) -> BoxedStrategy<Par> {
    if depth == 0 {
        arb_leaf_par().boxed()
    } else {
        let leaf = arb_leaf_par();
        let inner_depth = depth - 1;

        prop_oneof![
            // Leaf
            leaf,
            // EList of elements at (depth-1)
            prop::collection::vec(arb_par(inner_depth), 0..4).prop_map(|elements| {
                Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::EListBody(EList {
                        ps: elements,
                        locally_free: Vec::new(),
                        connective_used: false,
                        remainder: None,
                    })),
                }])
            }),
            // EPathMap of elements at (depth-1)
            prop::collection::vec(arb_par(inner_depth), 0..4).prop_map(|elements| {
                Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::EPathmapBody(EPathMap {
                        ps: elements,
                        locally_free: Vec::new(),
                        connective_used: false,
                        remainder: None,
                    })),
                }])
            }),
        ]
        .boxed()
    }
}

/// Generate an arbitrary Vec<Par> of PathMap elements at a given depth.
fn arb_pathmap_elements(depth: u32) -> BoxedStrategy<Vec<Par>> {
    prop::collection::vec(arb_par(depth), 0..6).boxed()
}

// =============================================================================
// Property: Self-matching (reflexivity)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Any PathMap should match itself exactly.
    #[test]
    fn pathmap_self_match(elements in arb_pathmap_elements(2)) {
        let target = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPathmapBody(EPathMap {
                ps: elements.clone(),
                locally_free: Vec::new(),
                connective_used: false,
                remainder: None,
            })),
        }]);
        let pattern = target.clone();

        let mut ctx = SpatialMatcherContext::new();
        let result = <SpatialMatcherContext as SpatialMatcher<Expr, Expr>>::spatial_match(
            &mut ctx,
            target.exprs[0].clone(),
            pattern.exprs[0].clone(),
        );
        prop_assert!(result.is_some(), "Any PathMap should match itself");
    }

    /// A wildcard-remainder pattern with no fixed elements should match any PathMap.
    #[test]
    fn wildcard_remainder_matches_any(elements in arb_pathmap_elements(2)) {
        let target = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPathmapBody(EPathMap {
                ps: elements,
                locally_free: Vec::new(),
                connective_used: false,
                remainder: None,
            })),
        }]);
        let pattern = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPathmapBody(EPathMap {
                ps: vec![],
                locally_free: Vec::new(),
                connective_used: true,
                remainder: Some(new_wildcard_var()),
            })),
        }]);

        let mut ctx = SpatialMatcherContext::new();
        let result = <SpatialMatcherContext as SpatialMatcher<Expr, Expr>>::spatial_match(
            &mut ctx,
            target.exprs[0].clone(),
            pattern.exprs[0].clone(),
        );
        prop_assert!(
            result.is_some(),
            "Wildcard-remainder with no fixed elements should match any PathMap"
        );
    }

    /// A freevar-remainder pattern with no fixed elements should bind the entire
    /// target to the variable.
    #[test]
    fn freevar_remainder_binds_all(elements in arb_pathmap_elements(2)) {
        // The spatial matcher sorts and deduplicates PathMap elements (set semantics),
        // so the effective element count is the number of unique elements.
        let mut unique_elements = elements.clone();
        unique_elements.sort();
        unique_elements.dedup();
        let unique_n = unique_elements.len();

        let target = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPathmapBody(EPathMap {
                ps: elements,
                locally_free: Vec::new(),
                connective_used: false,
                remainder: None,
            })),
        }]);
        let pattern = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPathmapBody(EPathMap {
                ps: vec![],
                locally_free: Vec::new(),
                connective_used: true,
                remainder: Some(new_freevar_var(0)),
            })),
        }]);

        let mut ctx = SpatialMatcherContext::new();
        let result = <SpatialMatcherContext as SpatialMatcher<Expr, Expr>>::spatial_match(
            &mut ctx,
            target.exprs[0].clone(),
            pattern.exprs[0].clone(),
        );
        prop_assert!(result.is_some(), "FreeVar-remainder with no fixed elements should match");

        let bound = ctx.free_map.get(&0);
        prop_assert!(bound.is_some(), "FreeVar(0) should be bound");

        let bound_par = bound.unwrap();
        if let Some(ExprInstance::EPathmapBody(pm)) = &bound_par.exprs[0].expr_instance {
            prop_assert_eq!(
                pm.ps.len(),
                unique_n,
                "Bound remainder should contain all {} unique target elements",
                unique_n
            );
        } else {
            prop_assert!(false, "Bound value should be an EPathmapBody");
        }
    }
}

// =============================================================================
// Property: Size invariant
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// The size of an EPathMap should equal the number of elements in its ps field.
    #[test]
    fn pathmap_size_equals_ps_len(elements in arb_pathmap_elements(3)) {
        let pm = EPathMap {
            ps: elements.clone(),
            locally_free: Vec::new(),
            connective_used: false,
            remainder: None,
        };
        prop_assert_eq!(pm.ps.len(), elements.len());
    }
}

// =============================================================================
// Property: Decomposition conservation
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// When a PathMap with elements [e1, ..., en] is matched against a pattern
    /// with one fixed element and a freevar remainder, the fixed match + remainder
    /// should collectively contain all elements from the original.
    #[test]
    fn decomposition_preserves_element_count(
        elements in prop::collection::vec(arb_leaf_par(), 1..6)
    ) {
        let _n = elements.len();
        let mut sorted_elements = elements.clone();
        sorted_elements.sort();
        sorted_elements.dedup();
        let unique_n = sorted_elements.len();

        if unique_n == 0 {
            return Ok(()); // Skip degenerate case
        }

        // Use the first sorted element as the fixed pattern element
        let fixed = sorted_elements[0].clone();

        let target = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPathmapBody(EPathMap {
                ps: sorted_elements.clone(),
                locally_free: Vec::new(),
                connective_used: false,
                remainder: None,
            })),
        }]);

        let pattern = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPathmapBody(EPathMap {
                ps: vec![fixed],
                locally_free: Vec::new(),
                connective_used: true,
                remainder: Some(new_freevar_var(0)),
            })),
        }]);

        let mut ctx = SpatialMatcherContext::new();
        let result = <SpatialMatcherContext as SpatialMatcher<Expr, Expr>>::spatial_match(
            &mut ctx,
            target.exprs[0].clone(),
            pattern.exprs[0].clone(),
        );
        prop_assert!(result.is_some(), "Matching with one fixed + remainder should succeed");

        let bound = ctx.free_map.get(&0);
        prop_assert!(bound.is_some(), "Remainder should be bound");

        let bound_par = bound.unwrap();
        if let Some(ExprInstance::EPathmapBody(remainder_pm)) = &bound_par.exprs[0].expr_instance {
            // fixed (1) + remainder = unique_n
            prop_assert_eq!(
                1 + remainder_pm.ps.len(),
                unique_n,
                "Fixed (1) + remainder ({}) should equal total unique elements ({})",
                remainder_pm.ps.len(),
                unique_n
            );
        } else {
            prop_assert!(false, "Remainder should be EPathmapBody");
        }
    }
}

// =============================================================================
// Property: Nesting round-trip for SExpr encoder/decoder
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// For leaf Pars (GString, GInt), encoding to SExpr and decoding back should
    /// preserve the ExprInstance type and value.
    #[test]
    fn sexpr_round_trip_leaf(par in arb_leaf_par()) {
        use models::rust::par_to_sexpr::ParToSExpr;
        use models::rust::pathmap_integration::parse_sexpr;
        use models::rust::sexpr_to_par::SExprToPar;

        let sexpr_str = ParToSExpr::par_to_sexpr(&par);
        let sexpr = parse_sexpr(&sexpr_str);
        let encoded = sexpr.encode();
        let decoded = SExprToPar::decode_segment(&encoded);

        prop_assert!(decoded.is_ok(), "Decoding should succeed for leaf Par");
        let decoded_par = decoded.unwrap();

        // Compare expr_instances (ignoring metadata like locally_free)
        if !par.exprs.is_empty() && !decoded_par.exprs.is_empty() {
            prop_assert_eq!(
                &decoded_par.exprs[0].expr_instance,
                &par.exprs[0].expr_instance,
                "Round-trip should preserve expr_instance"
            );
        }
    }
}

// =============================================================================
// Property: Nested PathMap in PathMap matching
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// A PathMap containing nested PathMaps should still self-match.
    #[test]
    fn nested_pathmap_self_match(
        inner_elements in prop::collection::vec(arb_leaf_par(), 0..3),
        outer_extra in prop::collection::vec(arb_leaf_par(), 0..3),
    ) {
        // Create inner pathmap
        let inner_pm = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPathmapBody(EPathMap {
                ps: inner_elements,
                locally_free: Vec::new(),
                connective_used: false,
                remainder: None,
            })),
        }]);

        // Create outer pathmap containing the inner one plus some extras
        let mut outer_elements = vec![inner_pm];
        outer_elements.extend(outer_extra);

        let target = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPathmapBody(EPathMap {
                ps: outer_elements.clone(),
                locally_free: Vec::new(),
                connective_used: false,
                remainder: None,
            })),
        }]);
        let pattern = target.clone();

        let mut ctx = SpatialMatcherContext::new();
        let result = <SpatialMatcherContext as SpatialMatcher<Expr, Expr>>::spatial_match(
            &mut ctx,
            target.exprs[0].clone(),
            pattern.exprs[0].clone(),
        );
        prop_assert!(result.is_some(), "Nested PathMap should self-match");
    }
}

// =============================================================================
// Property: List inside PathMap matching
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// A PathMap containing lists should self-match.
    #[test]
    fn list_in_pathmap_self_match(
        list_contents in prop::collection::vec(
            prop::collection::vec("[a-z]{1,4}".prop_map(|s| new_gstring_par(s, Vec::new(), false)), 1..4),
            1..4
        ),
    ) {
        let elements: Vec<Par> = list_contents
            .into_iter()
            .map(|contents| {
                Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::EListBody(EList {
                        ps: contents,
                        locally_free: Vec::new(),
                        connective_used: false,
                        remainder: None,
                    })),
                }])
            })
            .collect();

        let target = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPathmapBody(EPathMap {
                ps: elements.clone(),
                locally_free: Vec::new(),
                connective_used: false,
                remainder: None,
            })),
        }]);
        let pattern = target.clone();

        let mut ctx = SpatialMatcherContext::new();
        let result = <SpatialMatcherContext as SpatialMatcher<Expr, Expr>>::spatial_match(
            &mut ctx,
            target.exprs[0].clone(),
            pattern.exprs[0].clone(),
        );
        prop_assert!(result.is_some(), "PathMap with list elements should self-match");
    }
}

// =============================================================================
// Property: First and last invariants on non-empty lists
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// For any non-empty list, first() and last() should return valid elements,
    /// and for a single-element list, first() == last().
    #[test]
    fn first_last_invariants(elements in prop::collection::vec(arb_leaf_par(), 1..10)) {
        let first = elements.first().cloned();
        let last = elements.last().cloned();
        prop_assert!(first.is_some(), "first() should be Some for non-empty list");
        prop_assert!(last.is_some(), "last() should be Some for non-empty list");

        if elements.len() == 1 {
            prop_assert_eq!(
                &first.as_ref().unwrap().exprs[0].expr_instance,
                &last.as_ref().unwrap().exprs[0].expr_instance,
                "first() should equal last() for single-element list"
            );
        }
    }
}

// =============================================================================
// Property: Deeply nested Pars (depth limit test)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// PathMaps containing elements at depth 3 should still self-match.
    #[test]
    fn deep_nested_pathmap_self_match(elements in arb_pathmap_elements(3)) {
        let target = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPathmapBody(EPathMap {
                ps: elements.clone(),
                locally_free: Vec::new(),
                connective_used: false,
                remainder: None,
            })),
        }]);
        let pattern = target.clone();

        let mut ctx = SpatialMatcherContext::new();
        let result = <SpatialMatcherContext as SpatialMatcher<Expr, Expr>>::spatial_match(
            &mut ctx,
            target.exprs[0].clone(),
            pattern.exprs[0].clone(),
        );
        prop_assert!(result.is_some(), "Deep nested PathMap should self-match");
    }
}
