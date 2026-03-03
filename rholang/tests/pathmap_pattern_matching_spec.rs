// Unit tests for PathMap pattern matching via the spatial matcher.
// Tests EPathmapBody decomposition with exact match, wildcard remainder,
// free variable remainder, empty PathMaps, and mismatches.

use models::rhoapi::expr::ExprInstance;
use models::rhoapi::{EList, EPathMap, Expr, Par, Var};
use models::rust::utils::{new_epathmap_expr, new_freevar_var, new_gstring_par, new_wildcard_var};
use rholang::rust::interpreter::matcher::spatial_matcher::{SpatialMatcher, SpatialMatcherContext};

#[cfg(test)]
mod pathmap_pattern_matching_tests {
    use super::*;

    /// Helper: create a Par containing a GString
    fn gstring_par(s: &str) -> Par {
        new_gstring_par(s.to_string(), Vec::new(), false)
    }

    /// Helper: create a Par containing an EList of GStrings
    fn list_par(strings: &[&str]) -> Par {
        let elements: Vec<Par> = strings.iter().map(|s| gstring_par(s)).collect();
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EListBody(EList {
                ps: elements,
                locally_free: Vec::new(),
                connective_used: false,
                remainder: None,
            })),
        }])
    }

    /// Helper: create a Par wrapping an EPathMap expression
    fn pathmap_par(elements: Vec<Par>, remainder: Option<Var>) -> Par {
        let connective_used = remainder.is_some();
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPathmapBody(EPathMap {
                ps: elements,
                locally_free: Vec::new(),
                connective_used,
                remainder,
            })),
        }])
    }

    /// Helper: run spatial match on two Expr values
    fn match_exprs(target: Expr, pattern: Expr) -> (Option<()>, SpatialMatcherContext) {
        let mut ctx = SpatialMatcherContext::new();
        let result = <SpatialMatcherContext as SpatialMatcher<Expr, Expr>>::spatial_match(
            &mut ctx, target, pattern,
        );
        (result, ctx)
    }

    // =========================================================================
    // Exact match tests
    // =========================================================================

    #[test]
    fn test_exact_match_two_elements() {
        let target = pathmap_par(vec![list_par(&["a"]), list_par(&["b"])], None);
        let pattern = pathmap_par(vec![list_par(&["a"]), list_par(&["b"])], None);

        let (result, _) = match_exprs(target.exprs[0].clone(), pattern.exprs[0].clone());
        assert!(result.is_some(), "Exact match of two identical PathMaps should succeed");
    }

    #[test]
    fn test_exact_match_single_element() {
        let target = pathmap_par(vec![list_par(&["x"])], None);
        let pattern = pathmap_par(vec![list_par(&["x"])], None);

        let (result, _) = match_exprs(target.exprs[0].clone(), pattern.exprs[0].clone());
        assert!(result.is_some(), "Exact match of single-element PathMaps should succeed");
    }

    #[test]
    fn test_empty_pathmap_match() {
        let target = pathmap_par(vec![], None);
        let pattern = pathmap_par(vec![], None);

        let (result, _) = match_exprs(target.exprs[0].clone(), pattern.exprs[0].clone());
        assert!(result.is_some(), "Empty PathMaps should match");
    }

    #[test]
    fn test_three_element_exact_match() {
        let target = pathmap_par(
            vec![list_par(&["x"]), list_par(&["y"]), list_par(&["z"])],
            None,
        );
        let pattern = pathmap_par(
            vec![list_par(&["x"]), list_par(&["y"]), list_par(&["z"])],
            None,
        );

        let (result, _) = match_exprs(target.exprs[0].clone(), pattern.exprs[0].clone());
        assert!(result.is_some(), "Three-element exact match should succeed");
    }

    #[test]
    fn test_nested_list_elements() {
        let target = pathmap_par(vec![list_par(&["a", "b"]), list_par(&["c", "d"])], None);
        let pattern = pathmap_par(vec![list_par(&["a", "b"]), list_par(&["c", "d"])], None);

        let (result, _) = match_exprs(target.exprs[0].clone(), pattern.exprs[0].clone());
        assert!(result.is_some(), "Multi-segment path elements should match exactly");
    }

    #[test]
    fn test_pathmap_with_gstring_elements() {
        let target = pathmap_par(vec![gstring_par("hello"), gstring_par("world")], None);
        let pattern = pathmap_par(vec![gstring_par("hello"), gstring_par("world")], None);

        let (result, _) = match_exprs(target.exprs[0].clone(), pattern.exprs[0].clone());
        assert!(result.is_some(), "PathMap with GString elements should match");
    }

    // =========================================================================
    // Mismatch tests
    // =========================================================================

    #[test]
    fn test_mismatch_different_element_count() {
        let target = pathmap_par(vec![list_par(&["a"]), list_par(&["b"])], None);
        let pattern = pathmap_par(vec![list_par(&["a"])], None);

        let (result, _) = match_exprs(target.exprs[0].clone(), pattern.exprs[0].clone());
        assert!(
            result.is_none(),
            "PathMap with 2 elements should not match pattern with 1 element (no remainder)"
        );
    }

    #[test]
    fn test_mismatch_different_values() {
        let target = pathmap_par(vec![list_par(&["a"])], None);
        let pattern = pathmap_par(vec![list_par(&["b"])], None);

        let (result, _) = match_exprs(target.exprs[0].clone(), pattern.exprs[0].clone());
        assert!(result.is_none(), "PathMaps with different elements should not match");
    }

    #[test]
    fn test_mismatch_empty_vs_nonempty() {
        let target = pathmap_par(vec![], None);
        let pattern = pathmap_par(vec![list_par(&["a"])], None);

        let (result, _) = match_exprs(target.exprs[0].clone(), pattern.exprs[0].clone());
        assert!(result.is_none(), "Empty target should not match non-empty pattern");
    }

    #[test]
    fn test_type_mismatch_pathmap_vs_set() {
        let target_expr = new_epathmap_expr(vec![gstring_par("a")], Vec::new(), false, None);
        let pattern_expr =
            models::rust::utils::new_eset_expr(vec![gstring_par("a")], Vec::new(), false, None);

        let (result, _) = match_exprs(target_expr, pattern_expr);
        assert!(
            result.is_none(),
            "EPathmapBody should not match ESetBody even with same elements"
        );
    }

    // =========================================================================
    // Wildcard remainder tests
    // =========================================================================

    #[test]
    fn test_wildcard_remainder_match() {
        let target = pathmap_par(
            vec![list_par(&["a"]), list_par(&["b"]), list_par(&["c"])],
            None,
        );
        let pattern = pathmap_par(vec![list_par(&["a"])], Some(new_wildcard_var()));

        let (result, _) = match_exprs(target.exprs[0].clone(), pattern.exprs[0].clone());
        assert!(result.is_some(), "Wildcard remainder should match any extra elements");
    }

    #[test]
    fn test_wildcard_remainder_exact_size() {
        let target = pathmap_par(vec![list_par(&["a"])], None);
        let pattern = pathmap_par(vec![list_par(&["a"])], Some(new_wildcard_var()));

        let (result, _) = match_exprs(target.exprs[0].clone(), pattern.exprs[0].clone());
        assert!(
            result.is_some(),
            "Wildcard remainder should match even when no extra elements exist"
        );
    }

    #[test]
    fn test_wildcard_remainder_empty_target() {
        let target = pathmap_par(vec![], None);
        let pattern = pathmap_par(vec![], Some(new_wildcard_var()));

        let (result, _) = match_exprs(target.exprs[0].clone(), pattern.exprs[0].clone());
        assert!(
            result.is_some(),
            "Wildcard remainder on empty pattern should match empty target"
        );
    }

    // =========================================================================
    // Free variable remainder tests (binding)
    // =========================================================================

    #[test]
    fn test_freevar_remainder_binding() {
        let target = pathmap_par(
            vec![list_par(&["a"]), list_par(&["b"]), list_par(&["c"])],
            None,
        );
        let pattern = pathmap_par(vec![list_par(&["a"])], Some(new_freevar_var(0)));

        let (result, ctx) = match_exprs(target.exprs[0].clone(), pattern.exprs[0].clone());
        assert!(result.is_some(), "FreeVar remainder should match and bind extra elements");

        let bound = ctx.free_map.get(&0);
        assert!(bound.is_some(), "FreeVar(0) should be bound in the free_map");

        let bound_par = bound.expect("bound should exist");
        assert!(!bound_par.exprs.is_empty(), "Bound value should have expressions");

        match &bound_par.exprs[0].expr_instance {
            Some(ExprInstance::EPathmapBody(remainder_pm)) => {
                assert_eq!(
                    remainder_pm.ps.len(),
                    2,
                    "Remainder PathMap should have 2 elements (the non-matched ones)"
                );
            }
            other => panic!("Expected EPathmapBody for remainder binding, got: {:?}", other),
        }
    }

    #[test]
    fn test_freevar_remainder_empty_binding() {
        let target = pathmap_par(vec![list_par(&["a"])], None);
        let pattern = pathmap_par(vec![list_par(&["a"])], Some(new_freevar_var(0)));

        let (result, ctx) = match_exprs(target.exprs[0].clone(), pattern.exprs[0].clone());
        assert!(result.is_some(), "FreeVar remainder should match with empty remainder");

        let bound = ctx.free_map.get(&0);
        assert!(bound.is_some(), "FreeVar(0) should be bound");

        let bound_par = bound.expect("bound should exist");
        match &bound_par.exprs[0].expr_instance {
            Some(ExprInstance::EPathmapBody(remainder_pm)) => {
                assert_eq!(
                    remainder_pm.ps.len(),
                    0,
                    "Remainder PathMap should be empty when target matches exactly"
                );
            }
            other => panic!("Expected EPathmapBody, got: {:?}", other),
        }
    }

    #[test]
    fn test_freevar_remainder_all_unmatched() {
        // Pattern has no fixed elements, just ...rest — binds everything
        let target = pathmap_par(vec![gstring_par("x"), gstring_par("y")], None);
        let pattern = pathmap_par(vec![], Some(new_freevar_var(0)));

        let (result, ctx) = match_exprs(target.exprs[0].clone(), pattern.exprs[0].clone());
        assert!(result.is_some(), "Empty pattern with FreeVar remainder should match anything");

        let bound = ctx.free_map.get(&0);
        assert!(bound.is_some(), "FreeVar(0) should be bound");

        let bound_par = bound.expect("bound should exist");
        match &bound_par.exprs[0].expr_instance {
            Some(ExprInstance::EPathmapBody(remainder_pm)) => {
                assert_eq!(
                    remainder_pm.ps.len(),
                    2,
                    "Remainder should contain all target elements"
                );
            }
            other => panic!("Expected EPathmapBody, got: {:?}", other),
        }
    }

    // =========================================================================
    // Order independence tests (PathMap is unordered like set)
    // =========================================================================

    #[test]
    fn test_order_independent_match() {
        // {| ["b"], ["a"] |} matches {| ["a"], ["b"] |} (order doesn't matter)
        let target = pathmap_par(vec![list_par(&["b"]), list_par(&["a"])], None);
        let pattern = pathmap_par(vec![list_par(&["a"]), list_par(&["b"])], None);

        let (result, _) = match_exprs(target.exprs[0].clone(), pattern.exprs[0].clone());
        assert!(result.is_some(), "PathMap matching should be order-independent");
    }
}
