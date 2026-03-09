// Integration tests for PathMap and list method dispatch through the interpreter.
// Unlike the structural tests in pathmap_element_extraction_spec.rs, these exercise
// the actual method dispatch pipeline: EMethodBody -> DebruijnInterpreter::eval_expr_to_par().

use models::rhoapi::expr::ExprInstance;
use models::rhoapi::{BindPattern, EList, Expr, ListParWithRandom, Par, TaggedContinuation};
use models::rust::utils::{
    new_elist_par, new_emethod_expr, new_epathmap_par, new_gint_par, new_gstring_par,
};
use rholang::rust::interpreter::{
    env::Env,
    errors::InterpreterError,
    reduce::DebruijnInterpreter,
    test_utils::persistent_store_tester::create_test_space,
};
use rspace_plus_plus::rspace::rspace::RSpace;

#[cfg(test)]
mod pathmap_method_dispatch_tests {
    use super::*;

    fn gstring(s: &str) -> Par {
        new_gstring_par(s.to_string(), Vec::new(), false)
    }

    fn gint(n: i64) -> Par {
        new_gint_par(n, Vec::new(), false)
    }

    fn pathmap(elements: Vec<Par>) -> Par {
        new_epathmap_par(elements, Vec::new(), false, None, Vec::new(), false)
    }

    fn list(elements: Vec<Par>) -> Par {
        new_elist_par(elements, Vec::new(), false, None, Vec::new(), false)
    }

    fn method_call(method_name: &str, target: Par, args: Vec<Par>) -> Expr {
        new_emethod_expr(method_name.to_string(), target, args, Vec::new())
    }

    async fn make_reducer() -> DebruijnInterpreter {
        let (_, reducer) =
            create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
                .await;
        reducer
    }

    // ===================================================================
    // Test 1a: size on PathMap with 3 elements
    // ===================================================================

    #[tokio::test]
    async fn pathmap_size_returns_element_count() {
        let reducer = make_reducer().await;
        let target = pathmap(vec![gstring("a"), gstring("b"), gstring("c")]);
        let expr = method_call("size", target, vec![]);

        let result = reducer
            .eval_expr_to_par(&expr, &Env::new())
            .expect("size on 3-element PathMap should succeed");
        assert_eq!(gint(3), result);
    }

    // ===================================================================
    // Test 1b: size on empty PathMap
    // ===================================================================

    #[tokio::test]
    async fn pathmap_size_empty_returns_zero() {
        let reducer = make_reducer().await;
        let target = pathmap(vec![]);
        let expr = method_call("size", target, vec![]);

        let result = reducer
            .eval_expr_to_par(&expr, &Env::new())
            .expect("size on empty PathMap should succeed");
        assert_eq!(gint(0), result);
    }

    // ===================================================================
    // Test 1c: length on PathMap
    // ===================================================================

    #[tokio::test]
    async fn pathmap_length_returns_element_count() {
        let reducer = make_reducer().await;
        let target = pathmap(vec![gstring("x"), gstring("y")]);
        let expr = method_call("length", target, vec![]);

        let result = reducer
            .eval_expr_to_par(&expr, &Env::new())
            .expect("length on 2-element PathMap should succeed");
        assert_eq!(gint(2), result);
    }

    // ===================================================================
    // Test 1d: toList on PathMap
    // ===================================================================

    #[tokio::test]
    async fn pathmap_to_list_returns_list_of_elements() {
        let reducer = make_reducer().await;
        let elements = vec![gstring("a"), gstring("b")];
        let target = pathmap(elements.clone());
        let expr = method_call("toList", target, vec![]);

        let result = reducer
            .eval_expr_to_par(&expr, &Env::new())
            .expect("toList on PathMap should succeed");

        let expected = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EListBody(EList {
                ps: elements,
                locally_free: Vec::new(),
                connective_used: false,
                remainder: None,
            })),
        }]);
        assert_eq!(expected, result);
    }

    // ===================================================================
    // Test 1e: toList on empty PathMap
    // ===================================================================

    #[tokio::test]
    async fn pathmap_to_list_empty_returns_empty_list() {
        let reducer = make_reducer().await;
        let target = pathmap(vec![]);
        let expr = method_call("toList", target, vec![]);

        let result = reducer
            .eval_expr_to_par(&expr, &Env::new())
            .expect("toList on empty PathMap should succeed");

        let expected = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EListBody(EList {
                ps: vec![],
                locally_free: Vec::new(),
                connective_used: false,
                remainder: None,
            })),
        }]);
        assert_eq!(expected, result);
    }

    // ===================================================================
    // Test 1f: toSet on PathMap with duplicates
    // ===================================================================

    #[tokio::test]
    async fn pathmap_to_set_deduplicates_elements() {
        let reducer = make_reducer().await;
        let target = pathmap(vec![gstring("a"), gstring("b"), gstring("a")]);
        let expr = method_call("toSet", target, vec![]);

        let result = reducer
            .eval_expr_to_par(&expr, &Env::new())
            .expect("toSet on PathMap should succeed");

        assert_eq!(
            result.exprs.len(),
            1,
            "Result should have exactly one expr"
        );
        match &result.exprs[0].expr_instance {
            Some(ExprInstance::ESetBody(eset)) => {
                assert_eq!(
                    eset.ps.len(),
                    2,
                    "Set should contain 2 unique elements after dedup"
                );
            }
            other => panic!("Expected ESetBody, got {:?}", other),
        }
    }

    // ===================================================================
    // Test 1g: first on list
    // ===================================================================

    #[tokio::test]
    async fn list_first_returns_first_element() {
        let reducer = make_reducer().await;
        let target = list(vec![gstring("a"), gstring("b"), gstring("c")]);
        let expr = method_call("first", target, vec![]);

        let result = reducer
            .eval_expr_to_par(&expr, &Env::new())
            .expect("first on list should succeed");
        assert_eq!(gstring("a"), result);
    }

    // ===================================================================
    // Test 1h: last on list
    // ===================================================================

    #[tokio::test]
    async fn list_last_returns_last_element() {
        let reducer = make_reducer().await;
        let target = list(vec![gstring("a"), gstring("b"), gstring("c")]);
        let expr = method_call("last", target, vec![]);

        let result = reducer
            .eval_expr_to_par(&expr, &Env::new())
            .expect("last on list should succeed");
        assert_eq!(gstring("c"), result);
    }

    // ===================================================================
    // Test 1i: first and last equal for single-element list
    // ===================================================================

    #[tokio::test]
    async fn list_first_and_last_equal_for_single_element() {
        let reducer = make_reducer().await;
        let target = list(vec![gstring("only")]);

        let first_expr = method_call("first", target.clone(), vec![]);
        let last_expr = method_call("last", target, vec![]);

        let first_result = reducer
            .eval_expr_to_par(&first_expr, &Env::new())
            .expect("first on single-element list should succeed");
        let last_result = reducer
            .eval_expr_to_par(&last_expr, &Env::new())
            .expect("last on single-element list should succeed");

        assert_eq!(gstring("only"), first_result);
        assert_eq!(gstring("only"), last_result);
        assert_eq!(first_result, last_result);
    }

    // ===================================================================
    // Test 1j: paths on PathMap
    // ===================================================================

    #[tokio::test]
    async fn pathmap_paths_returns_list_of_paths() {
        let reducer = make_reducer().await;
        let target = pathmap(vec![gstring("a"), gstring("b")]);
        let expr = method_call("paths", target, vec![]);

        let result = reducer
            .eval_expr_to_par(&expr, &Env::new())
            .expect("paths on PathMap should succeed");

        assert_eq!(
            result.exprs.len(),
            1,
            "Result should have exactly one expr"
        );
        match &result.exprs[0].expr_instance {
            Some(ExprInstance::EListBody(elist)) => {
                assert!(
                    !elist.ps.is_empty(),
                    "paths should return a non-empty list for a non-empty PathMap"
                );
            }
            other => panic!("Expected EListBody from paths method, got {:?}", other),
        }
    }

    // ===================================================================
    // Test 1k: size with wrong argument count
    // ===================================================================

    #[tokio::test]
    async fn pathmap_size_with_args_returns_error() {
        let reducer = make_reducer().await;
        let target = pathmap(vec![gstring("a")]);
        let expr = method_call("size", target, vec![gint(1)]);

        let result = reducer.eval_expr_to_par(&expr, &Env::new());

        match result {
            Err(InterpreterError::MethodArgumentNumberMismatch {
                method,
                expected,
                actual,
            }) => {
                assert_eq!(method, "size");
                assert_eq!(expected, 0);
                assert_eq!(actual, 1);
            }
            other => panic!(
                "Expected MethodArgumentNumberMismatch error, got {:?}",
                other
            ),
        }
    }
}
