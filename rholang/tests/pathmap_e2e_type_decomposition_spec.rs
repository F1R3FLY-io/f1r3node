// End-to-end integration tests for PathMap type decomposition via the spatial matcher.
// Tests the three-phase pipeline: Compile (Rholang source → Par) → Embed (verify type
// variant in PathMap elements) → Decompose (spatial match with exact, wildcard, freevar).
// Covers all Rholang type variants: scalars, collections, processes, nested structures.

use models::rhoapi::expr::ExprInstance;
use models::rhoapi::{EPathMap, Expr, Par, Var};
use models::rust::utils::{new_freevar_var, new_gbytearray_par, new_wildcard_var};
use rholang::rust::interpreter::compiler::compiler::Compiler;
use rholang::rust::interpreter::matcher::spatial_matcher::{SpatialMatcher, SpatialMatcherContext};

#[cfg(test)]
mod pathmap_e2e_type_decomposition_tests {
    use super::*;

    // =========================================================================
    // Helpers
    // =========================================================================

    /// Phase 1: Compile Rholang source to normalized Par.
    fn compile(rho: &str) -> Par {
        Compiler::source_to_adt(rho).expect("Rholang compilation should succeed")
    }

    /// Construct an EPathmapBody Expr from elements and optional remainder.
    fn make_pathmap_expr(elements: Vec<Par>, remainder: Option<Var>) -> Expr {
        let connective_used = remainder.is_some();
        Expr {
            expr_instance: Some(ExprInstance::EPathmapBody(EPathMap {
                ps: elements,
                locally_free: Vec::new(),
                connective_used,
                remainder,
            })),
        }
    }

    /// Run spatial match on two Expr values, return result and context.
    fn match_exprs(target: Expr, pattern: Expr) -> (Option<()>, SpatialMatcherContext) {
        let mut ctx = SpatialMatcherContext::new();
        let result = <SpatialMatcherContext as SpatialMatcher<Expr, Expr>>::spatial_match(
            &mut ctx, target, pattern,
        );
        (result, ctx)
    }

    /// Extract the EPathMap from a compiled Par.
    fn extract_pathmap(par: &Par) -> &EPathMap {
        par.exprs
            .iter()
            .find_map(|e| match &e.expr_instance {
                Some(ExprInstance::EPathmapBody(pm)) => Some(pm),
                _ => None,
            })
            .expect("Par should contain an EPathmapBody expression")
    }

    /// Extract the EPathmapBody Expr from a compiled Par.
    fn extract_pathmap_expr(par: &Par) -> Expr {
        par.exprs
            .iter()
            .find(|e| matches!(&e.expr_instance, Some(ExprInstance::EPathmapBody(_))))
            .cloned()
            .expect("Par should contain an EPathmapBody expression")
    }

    /// Assert that a compiled PathMap matches itself (self-match).
    fn assert_self_match(compiled: &Par) {
        let target = extract_pathmap_expr(compiled);
        let pattern = target.clone();
        let (result, _) = match_exprs(target, pattern);
        assert!(result.is_some(), "PathMap should match itself");
    }

    /// Assert exact match between two sets of PathMap elements.
    fn assert_exact_match(target_elements: Vec<Par>, pattern_elements: Vec<Par>) {
        let target = make_pathmap_expr(target_elements, None);
        let pattern = make_pathmap_expr(pattern_elements, None);
        let (result, _) = match_exprs(target, pattern);
        assert!(result.is_some(), "Exact PathMap match should succeed");
    }

    /// Assert freevar remainder decomposition: match fixed elements, bind remainder.
    fn assert_freevar_remainder(
        target_elems: Vec<Par>,
        fixed_elems: Vec<Par>,
        expected_remainder_count: usize,
    ) {
        let target = make_pathmap_expr(target_elems, None);
        let pattern = make_pathmap_expr(fixed_elems, Some(new_freevar_var(0)));
        let (result, ctx) = match_exprs(target, pattern);
        assert!(
            result.is_some(),
            "Freevar remainder match should succeed"
        );

        let bound = ctx
            .free_map
            .get(&0)
            .expect("FreeVar(0) should be bound in free_map");
        match &bound.exprs[0].expr_instance {
            Some(ExprInstance::EPathmapBody(remainder)) => {
                assert_eq!(
                    remainder.ps.len(),
                    expected_remainder_count,
                    "Remainder PathMap should have {} elements, got {}",
                    expected_remainder_count,
                    remainder.ps.len()
                );
            }
            other => panic!(
                "Expected EPathmapBody for remainder binding, got: {:?}",
                other
            ),
        }
    }

    /// Check whether any element in a PathMap has an expr matching a predicate.
    fn has_expr_element<F: Fn(&ExprInstance) -> bool>(pm: &EPathMap, pred: F) -> bool {
        pm.ps
            .iter()
            .any(|p| p.exprs.iter().any(|e| e.expr_instance.as_ref().is_some_and(&pred)))
    }

    /// Check whether any element in a PathMap has non-empty sends.
    fn has_send_element(pm: &EPathMap) -> bool {
        pm.ps.iter().any(|p| !p.sends.is_empty())
    }

    /// Check whether any element in a PathMap has non-empty receives.
    fn has_receive_element(pm: &EPathMap) -> bool {
        pm.ps.iter().any(|p| !p.receives.is_empty())
    }

    /// Check whether any element in a PathMap has non-empty news.
    fn has_new_element(pm: &EPathMap) -> bool {
        pm.ps.iter().any(|p| !p.news.is_empty())
    }

    /// Check whether any element in a PathMap has non-empty matches.
    fn has_match_element(pm: &EPathMap) -> bool {
        pm.ps.iter().any(|p| !p.matches.is_empty())
    }

    /// Check whether any element in a PathMap has non-empty bundles.
    fn has_bundle_element(pm: &EPathMap) -> bool {
        pm.ps.iter().any(|p| !p.bundles.is_empty())
    }

    // =========================================================================
    // Category 1: Scalar Types (11 tests)
    // =========================================================================

    #[test]
    fn test_e2e_gbool_true_in_pathmap() {
        let compiled = compile("{| true |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::GBool(true))),
            "Element should be GBool(true)"
        );
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_gbool_false_in_pathmap() {
        let compiled = compile("{| false |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::GBool(false))),
            "Element should be GBool(false)"
        );
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_gint_positive_in_pathmap() {
        let compiled = compile("{| 42 |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::GInt(42))),
            "Element should be GInt(42)"
        );
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_gint_negative_in_pathmap() {
        let compiled = compile("{| -7 |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::GInt(-7))),
            "Element should be GInt(-7)"
        );
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_gint_zero_in_pathmap() {
        let compiled = compile("{| 0 |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::GInt(0))),
            "Element should be GInt(0)"
        );
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_gstring_in_pathmap() {
        let compiled = compile(r#"{| "hello" |}"#);
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::GString(s) if s == "hello")),
            "Element should be GString(\"hello\")"
        );
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_gstring_empty_in_pathmap() {
        let compiled = compile(r#"{| "" |}"#);
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::GString(s) if s.is_empty())),
            "Element should be GString(\"\")"
        );
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_guri_in_pathmap() {
        let compiled = compile("{| `rho:test:uri` |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::GUri(u) if u == "rho:test:uri")),
            "Element should be GUri(\"rho:test:uri\")"
        );
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_gbytearray_in_pathmap_manual() {
        // GByteArray has no Rholang literal; construct manually.
        let byte_par = new_gbytearray_par(vec![0xDE, 0xAD, 0xBE, 0xEF], Vec::new(), false);
        let elements = vec![byte_par.clone()];

        // Verify the type
        assert!(
            byte_par
                .exprs
                .iter()
                .any(|e| matches!(&e.expr_instance, Some(ExprInstance::GByteArray(_)))),
            "Manually constructed Par should contain GByteArray"
        );

        // Exact match
        assert_exact_match(elements.clone(), elements.clone());

        // Wildcard remainder
        let target = make_pathmap_expr(elements.clone(), None);
        let pattern = make_pathmap_expr(elements, Some(new_wildcard_var()));
        let (result, _) = match_exprs(target, pattern);
        assert!(
            result.is_some(),
            "GByteArray PathMap should match with wildcard remainder"
        );
    }

    #[test]
    fn test_e2e_nil_in_pathmap() {
        let compiled = compile("{| Nil |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");

        // Nil compiles to Par::default() (all fields empty)
        let element = &pm.ps[0];
        let is_nil = element.sends.is_empty()
            && element.receives.is_empty()
            && element.news.is_empty()
            && element.exprs.is_empty()
            && element.matches.is_empty()
            && element.bundles.is_empty()
            && element.connectives.is_empty()
            && element.unforgeables.is_empty();
        assert!(is_nil, "Nil element should be an empty Par");

        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_mixed_scalars_in_pathmap() {
        let compiled = compile(r#"{| true, 42, "hello" |}"#);
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 3, "PathMap should have 3 elements");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::GBool(true))),
            "Should contain GBool(true)"
        );
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::GInt(42))),
            "Should contain GInt(42)"
        );
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::GString(s) if s == "hello")),
            "Should contain GString(\"hello\")"
        );
        assert_self_match(&compiled);
    }

    // =========================================================================
    // Category 2: Collection Types (7 tests)
    // =========================================================================

    #[test]
    fn test_e2e_elist_in_pathmap() {
        let compiled = compile("{| [1, 2, 3] |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::EListBody(_))),
            "Element should be EListBody"
        );
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_etuple_in_pathmap() {
        let compiled = compile("{| (1, 2, 3) |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::ETupleBody(_))),
            "Element should be ETupleBody"
        );
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_eset_in_pathmap() {
        let compiled = compile("{| Set(1, 2, 3) |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::ESetBody(_))),
            "Element should be ESetBody"
        );
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_emap_in_pathmap() {
        let compiled = compile(r#"{| {"key": "value"} |}"#);
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::EMapBody(_))),
            "Element should be EMapBody"
        );
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_nested_pathmap_in_pathmap() {
        let compiled = compile("{| {| 1, 2 |} |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::EPathmapBody(_))),
            "Element should be a nested EPathmapBody"
        );
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_empty_collections_in_pathmap() {
        let compiled = compile("{| [], Set() |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 2, "PathMap should have 2 elements");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::EListBody(_))),
            "Should contain EListBody"
        );
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::ESetBody(_))),
            "Should contain ESetBody"
        );
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_mixed_collections_in_pathmap() {
        let compiled = compile(r#"{| [1, 2], (3, 4), {"a": "b"} |}"#);
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 3, "PathMap should have 3 elements");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::EListBody(_))),
            "Should contain EListBody"
        );
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::ETupleBody(_))),
            "Should contain ETupleBody"
        );
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::EMapBody(_))),
            "Should contain EMapBody"
        );
        assert_self_match(&compiled);
    }

    // =========================================================================
    // Category 3: Process Types (6 tests)
    // =========================================================================

    #[test]
    fn test_e2e_send_in_pathmap() {
        let compiled = compile("{| @0!(1) |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(has_send_element(pm), "Element should contain a Send");
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_receive_in_pathmap() {
        let compiled = compile("{| for(@x <- @0) { Nil } |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(has_receive_element(pm), "Element should contain a Receive");
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_new_in_pathmap() {
        let compiled = compile("{| new x in { Nil } |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(has_new_element(pm), "Element should contain a New");
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_match_in_pathmap() {
        let compiled = compile(r#"{| match 42 { 42 => "found" } |}"#);
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(has_match_element(pm), "Element should contain a Match");
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_bundle_in_pathmap() {
        let compiled = compile("{| bundle+{Nil} |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(has_bundle_element(pm), "Element should contain a Bundle");
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_mixed_processes_in_pathmap() {
        let compiled = compile("{| @0!(1), for(@x <- @1){Nil}, new y in {Nil} |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 3, "PathMap should have 3 elements");
        assert!(has_send_element(pm), "Should contain a Send");
        assert!(has_receive_element(pm), "Should contain a Receive");
        assert!(has_new_element(pm), "Should contain a New");
        assert_self_match(&compiled);
    }

    // =========================================================================
    // Category 4: Nested/Compound Structures (7 tests)
    // =========================================================================

    #[test]
    fn test_e2e_list_of_lists_in_pathmap() {
        let compiled = compile("{| [[1, 2], [3, 4]] |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::EListBody(_))),
            "Element should be EListBody"
        );

        // Verify the outer list contains inner lists
        if let Some(ExprInstance::EListBody(outer)) =
            &pm.ps[0].exprs[0].expr_instance
        {
            assert_eq!(outer.ps.len(), 2, "Outer list should have 2 inner lists");
            for inner_par in &outer.ps {
                assert!(
                    inner_par
                        .exprs
                        .iter()
                        .any(|e| matches!(&e.expr_instance, Some(ExprInstance::EListBody(_)))),
                    "Inner element should be EListBody"
                );
            }
        } else {
            panic!("Expected EListBody as PathMap element");
        }
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_map_with_nested_values_in_pathmap() {
        let compiled = compile(r#"{| {"o": {"i": 42}} |}"#);
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::EMapBody(_))),
            "Element should be EMapBody"
        );
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_tuple_containing_set_in_pathmap() {
        let compiled = compile("{| (Set(1, 2), Set(3, 4)) |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::ETupleBody(_))),
            "Element should be ETupleBody"
        );
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_send_and_list_in_pathmap() {
        let compiled = compile("{| @0!(1), [2, 3] |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 2, "PathMap should have 2 elements");
        assert!(has_send_element(pm), "Should contain a Send");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::EListBody(_))),
            "Should contain an EListBody"
        );
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_deeply_nested_pathmap_list() {
        let compiled = compile("{| [{| 1, 2 |}] |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");

        // Outer element should be a list
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::EListBody(_))),
            "Outer element should be EListBody"
        );

        // The list should contain a PathMap
        if let Some(ExprInstance::EListBody(list)) = &pm.ps[0].exprs[0].expr_instance {
            assert_eq!(list.ps.len(), 1, "List should have 1 element");
            assert!(
                list.ps[0]
                    .exprs
                    .iter()
                    .any(|e| matches!(&e.expr_instance, Some(ExprInstance::EPathmapBody(_)))),
                "List element should be a nested EPathmapBody"
            );
        } else {
            panic!("Expected EListBody as PathMap element");
        }
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_all_scalar_types_in_pathmap() {
        let compiled = compile(r#"{| true, 42, "hi", `rho:t` |}"#);
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 4, "PathMap should have 4 elements");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::GBool(true))),
            "Should contain GBool"
        );
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::GInt(42))),
            "Should contain GInt"
        );
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::GString(s) if s == "hi")),
            "Should contain GString"
        );
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::GUri(u) if u == "rho:t")),
            "Should contain GUri"
        );
        assert_self_match(&compiled);
    }

    #[test]
    fn test_e2e_process_inside_collection() {
        let compiled = compile("{| [@0!(1), @1!(2)] |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 1, "PathMap should have 1 element");
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::EListBody(_))),
            "Element should be EListBody"
        );

        // Verify the list elements contain sends
        if let Some(ExprInstance::EListBody(list)) = &pm.ps[0].exprs[0].expr_instance {
            assert_eq!(list.ps.len(), 2, "List should have 2 elements");
            for elem in &list.ps {
                assert!(!elem.sends.is_empty(), "List element should contain a Send");
            }
        } else {
            panic!("Expected EListBody as PathMap element");
        }
        assert_self_match(&compiled);
    }

    // =========================================================================
    // Category 5: Remainder Decomposition (6 tests)
    // =========================================================================

    #[test]
    fn test_e2e_wildcard_remainder_mixed() {
        let compiled = compile(r#"{| true, 42, "hi" |}"#);
        let pm = extract_pathmap(&compiled);
        let elements = pm.ps.clone();

        // Match one element with wildcard remainder — should succeed
        let target = make_pathmap_expr(elements.clone(), None);
        let pattern = make_pathmap_expr(vec![elements[0].clone()], Some(new_wildcard_var()));
        let (result, _) = match_exprs(target, pattern);
        assert!(
            result.is_some(),
            "Wildcard remainder should match any extra elements"
        );
    }

    #[test]
    fn test_e2e_freevar_binds_scalars() {
        let compiled = compile(r#"{| true, 42, "hi" |}"#);
        let pm = extract_pathmap(&compiled);
        let elements = pm.ps.clone();
        assert_eq!(elements.len(), 3);

        // Fix one element, bind the rest
        assert_freevar_remainder(elements.clone(), vec![elements[0].clone()], 2);
    }

    #[test]
    fn test_e2e_freevar_binds_collections() {
        let compiled = compile("{| [1, 2], (3, 4), Set(5) |}");
        let pm = extract_pathmap(&compiled);
        let elements = pm.ps.clone();
        assert_eq!(elements.len(), 3);

        // Fix one collection, bind the other two
        assert_freevar_remainder(elements.clone(), vec![elements[0].clone()], 2);
    }

    #[test]
    fn test_e2e_freevar_binds_processes() {
        let compiled = compile("{| @0!(1), @1!(2), @2!(3) |}");
        let pm = extract_pathmap(&compiled);
        let elements = pm.ps.clone();
        assert_eq!(elements.len(), 3);

        // Fix one send, bind the other two
        assert_freevar_remainder(elements.clone(), vec![elements[0].clone()], 2);
    }

    #[test]
    fn test_e2e_freevar_binds_nested() {
        let compiled = compile("{| {| 1 |}, [2, 3], 42 |}");
        let pm = extract_pathmap(&compiled);
        let elements = pm.ps.clone();
        assert_eq!(elements.len(), 3);

        // Fix one element, bind the other two
        assert_freevar_remainder(elements.clone(), vec![elements[0].clone()], 2);
    }

    #[test]
    fn test_e2e_empty_remainder_exact() {
        let compiled = compile(r#"{| "only" |}"#);
        let pm = extract_pathmap(&compiled);
        let elements = pm.ps.clone();
        assert_eq!(elements.len(), 1);

        // Fix the one element — remainder should be empty
        assert_freevar_remainder(elements.clone(), vec![elements[0].clone()], 0);
    }

    // =========================================================================
    // Category 6: Order Independence (2 tests)
    // =========================================================================

    #[test]
    fn test_e2e_order_independent_compiled() {
        // Compile in different source orders — compiler normalizes ordering
        let compiled_ba = compile(r#"{| "b", "a" |}"#);
        let compiled_ab = compile(r#"{| "a", "b" |}"#);

        let target = extract_pathmap_expr(&compiled_ba);
        let pattern = extract_pathmap_expr(&compiled_ab);
        let (result, _) = match_exprs(target, pattern);
        assert!(
            result.is_some(),
            "PathMap matching should be order-independent"
        );
    }

    #[test]
    fn test_e2e_dedup_compiled() {
        // Compiler preserves duplicates in the AST, but the spatial matcher
        // applies set semantics during matching (implicit dedup).
        let compiled_dup = compile("{| 1, 1, 2 |}");
        let compiled_nodup = compile("{| 1, 2 |}");

        let pm_dup = extract_pathmap(&compiled_dup);
        let pm_nodup = extract_pathmap(&compiled_nodup);

        // Compiler preserves all 3 elements (no compile-time dedup)
        assert_eq!(pm_dup.ps.len(), 3, "Compiler should preserve duplicate elements");
        assert_eq!(pm_nodup.ps.len(), 2, "Non-duplicate should have 2 elements");

        // Spatial matcher applies set semantics: {| 1, 1, 2 |} matches {| 1, 2 |}
        let target = extract_pathmap_expr(&compiled_dup);
        let pattern = extract_pathmap_expr(&compiled_nodup);
        let (result, _) = match_exprs(target, pattern);
        assert!(
            result.is_some(),
            "Spatial matcher should apply set-semantic dedup: {{| 1, 1, 2 |}} matches {{| 1, 2 |}}"
        );

        // Self-match with duplicates also works
        assert_self_match(&compiled_dup);
    }

    // =========================================================================
    // Category 7: Mismatch Tests (3 tests)
    // =========================================================================

    #[test]
    fn test_e2e_mismatch_different_types() {
        // GInt(42) should not match GString("42")
        let target_compiled = compile("{| 42 |}");
        let pattern_compiled = compile(r#"{| "42" |}"#);

        let target = extract_pathmap_expr(&target_compiled);
        let pattern = extract_pathmap_expr(&pattern_compiled);
        let (result, _) = match_exprs(target, pattern);
        assert!(
            result.is_none(),
            "GInt should not match GString even with same textual value"
        );
    }

    #[test]
    fn test_e2e_mismatch_different_collections() {
        // EList should not match ETuple
        let target_compiled = compile("{| [1, 2, 3] |}");
        let pattern_compiled = compile("{| (1, 2, 3) |}");

        let target = extract_pathmap_expr(&target_compiled);
        let pattern = extract_pathmap_expr(&pattern_compiled);
        let (result, _) = match_exprs(target, pattern);
        assert!(
            result.is_none(),
            "EList PathMap should not match ETuple PathMap"
        );
    }

    #[test]
    fn test_e2e_mismatch_count_no_remainder() {
        // 3 elements should not match 2 elements without a remainder
        let target_compiled = compile("{| 1, 2, 3 |}");
        let pattern_compiled = compile("{| 1, 2 |}");

        let target = extract_pathmap_expr(&target_compiled);
        let pattern = extract_pathmap_expr(&pattern_compiled);
        let (result, _) = match_exprs(target, pattern);
        assert!(
            result.is_none(),
            "3-element PathMap should not match 2-element pattern without remainder"
        );
    }

    // =========================================================================
    // Category 8: Comprehensive Round-Trip (3 tests)
    // =========================================================================

    #[test]
    fn test_e2e_round_trip_all_ground_types() {
        let compiled = compile(r#"{| true, 42, "hello", `rho:test` |}"#);
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 4, "Should have 4 ground type elements");

        // Verify all types present
        assert!(has_expr_element(pm, |e| matches!(e, ExprInstance::GBool(_))));
        assert!(has_expr_element(pm, |e| matches!(e, ExprInstance::GInt(_))));
        assert!(has_expr_element(pm, |e| matches!(e, ExprInstance::GString(_))));
        assert!(has_expr_element(pm, |e| matches!(e, ExprInstance::GUri(_))));

        // Round-trip: extract elements, reconstruct, match
        let elements = pm.ps.clone();
        assert_exact_match(elements.clone(), elements);
    }

    #[test]
    fn test_e2e_round_trip_all_collection_types() {
        let compiled = compile(r#"{| [1, 2], (3, 4), Set(5), {"k": "v"}, {| 42 |} |}"#);
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 5, "Should have 5 collection type elements");

        // Verify all collection types present
        assert!(has_expr_element(pm, |e| matches!(e, ExprInstance::EListBody(_))));
        assert!(has_expr_element(pm, |e| matches!(e, ExprInstance::ETupleBody(_))));
        assert!(has_expr_element(pm, |e| matches!(e, ExprInstance::ESetBody(_))));
        assert!(has_expr_element(pm, |e| matches!(e, ExprInstance::EMapBody(_))));
        assert!(has_expr_element(pm, |e| matches!(e, ExprInstance::EPathmapBody(_))));

        // Round-trip: extract, reconstruct, match
        let elements = pm.ps.clone();
        assert_exact_match(elements.clone(), elements);
    }

    #[test]
    fn test_e2e_round_trip_mixed_comprehensive() {
        let compiled = compile("{| true, [1, 2], @0!(1), new x in {Nil} |}");
        let pm = extract_pathmap(&compiled);
        assert_eq!(pm.ps.len(), 4, "Should have 4 mixed elements");

        // Verify: scalar, collection, and process types all present
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::GBool(true))),
            "Should contain GBool"
        );
        assert!(
            has_expr_element(pm, |e| matches!(e, ExprInstance::EListBody(_))),
            "Should contain EListBody"
        );
        assert!(has_send_element(pm), "Should contain a Send");
        assert!(has_new_element(pm), "Should contain a New");

        // Round-trip: extract, reconstruct, match
        let elements = pm.ps.clone();
        assert_exact_match(elements.clone(), elements);
    }
}
