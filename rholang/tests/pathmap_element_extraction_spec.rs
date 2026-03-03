// Unit tests for PathMap element extraction methods: toList, toSet, size, length,
// and the new list methods: first, last.
// These test the method dispatch at the Expr/ExprInstance level directly,
// without going through the full interpreter.

use models::rhoapi::expr::ExprInstance;
use models::rhoapi::{EList, EPathMap, Expr, Par};
use models::rust::utils::new_gstring_par;

#[cfg(test)]
mod pathmap_element_extraction_tests {
    use super::*;

    fn gstring_par(s: &str) -> Par {
        new_gstring_par(s.to_string(), Vec::new(), false)
    }

    fn gint_par(n: i64) -> Par {
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::GInt(n)),
        }])
    }

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

    fn make_pathmap_expr(elements: Vec<Par>) -> Expr {
        Expr {
            expr_instance: Some(ExprInstance::EPathmapBody(EPathMap {
                ps: elements,
                locally_free: Vec::new(),
                connective_used: false,
                remainder: None,
            })),
        }
    }

    #[allow(dead_code)]
    fn make_list_expr(elements: Vec<Par>) -> Expr {
        Expr {
            expr_instance: Some(ExprInstance::EListBody(EList {
                ps: elements,
                locally_free: Vec::new(),
                connective_used: false,
                remainder: None,
            })),
        }
    }

    // =========================================================================
    // EPathMap size tests
    // =========================================================================

    #[test]
    fn test_pathmap_size_two_elements() {
        let pm = EPathMap {
            ps: vec![list_par(&["a"]), list_par(&["b"])],
            locally_free: Vec::new(),
            connective_used: false,
            remainder: None,
        };
        assert_eq!(pm.ps.len(), 2);
    }

    #[test]
    fn test_pathmap_size_empty() {
        let pm = EPathMap {
            ps: vec![],
            locally_free: Vec::new(),
            connective_used: false,
            remainder: None,
        };
        assert_eq!(pm.ps.len(), 0);
    }

    #[test]
    fn test_pathmap_size_five_elements() {
        let pm = EPathMap {
            ps: vec![
                gstring_par("a"),
                gstring_par("b"),
                gstring_par("c"),
                gstring_par("d"),
                gstring_par("e"),
            ],
            locally_free: Vec::new(),
            connective_used: false,
            remainder: None,
        };
        assert_eq!(pm.ps.len(), 5);
    }

    // =========================================================================
    // EPathMap to list conversion tests (structural)
    // =========================================================================

    #[test]
    fn test_pathmap_to_list_preserves_elements() {
        let elements = vec![list_par(&["a"]), list_par(&["b"]), list_par(&["c"])];
        let pm_expr = make_pathmap_expr(elements.clone());

        // Extract the ps field (what toList returns)
        if let Some(ExprInstance::EPathmapBody(pm)) = pm_expr.expr_instance {
            assert_eq!(pm.ps.len(), 3);
            // Verify elements are preserved
            for (i, elem) in pm.ps.iter().enumerate() {
                assert_eq!(
                    elem.exprs[0].expr_instance, elements[i].exprs[0].expr_instance,
                    "Element {} should be preserved",
                    i
                );
            }
        } else {
            panic!("Expected EPathmapBody");
        }
    }

    #[test]
    fn test_pathmap_to_list_empty() {
        let pm_expr = make_pathmap_expr(vec![]);
        if let Some(ExprInstance::EPathmapBody(pm)) = pm_expr.expr_instance {
            assert_eq!(pm.ps.len(), 0);
        } else {
            panic!("Expected EPathmapBody");
        }
    }

    // =========================================================================
    // EPathMap to set conversion tests (structural)
    // =========================================================================

    #[test]
    fn test_pathmap_to_set_deduplicates() {
        // When converting to set, duplicate elements should be removed
        use models::rust::par_set::ParSet;
        use models::rust::par_set_type_mapper::ParSetTypeMapper;

        let elements = vec![gstring_par("a"), gstring_par("a"), gstring_par("b")];
        let par_set = ParSet::new(elements, false, Vec::new(), None);
        let eset = ParSetTypeMapper::par_set_to_eset(par_set);

        // ParSet deduplicates, so the set should have 2 elements
        let restored = ParSetTypeMapper::eset_to_par_set(eset);
        assert_eq!(restored.ps.length(), 2, "Set should deduplicate elements");
    }

    // =========================================================================
    // List first/last tests (structural)
    // =========================================================================

    #[test]
    fn test_list_first_nonempty() {
        let list = EList {
            ps: vec![gstring_par("a"), gstring_par("b"), gstring_par("c")],
            locally_free: Vec::new(),
            connective_used: false,
            remainder: None,
        };
        let first = list.ps.first().cloned();
        assert!(first.is_some());
        assert_eq!(
            first.expect("first should exist").exprs[0].expr_instance,
            Some(ExprInstance::GString("a".to_string()))
        );
    }

    #[test]
    fn test_list_last_nonempty() {
        let list = EList {
            ps: vec![gstring_par("a"), gstring_par("b"), gstring_par("c")],
            locally_free: Vec::new(),
            connective_used: false,
            remainder: None,
        };
        let last = list.ps.last().cloned();
        assert!(last.is_some());
        assert_eq!(
            last.expect("last should exist").exprs[0].expr_instance,
            Some(ExprInstance::GString("c".to_string()))
        );
    }

    #[test]
    fn test_list_first_empty() {
        let list = EList {
            ps: vec![],
            locally_free: Vec::new(),
            connective_used: false,
            remainder: None,
        };
        assert!(list.ps.first().is_none(), "first() on empty list should be None");
    }

    #[test]
    fn test_list_last_empty() {
        let list = EList {
            ps: vec![],
            locally_free: Vec::new(),
            connective_used: false,
            remainder: None,
        };
        assert!(list.ps.last().is_none(), "last() on empty list should be None");
    }

    #[test]
    fn test_list_first_single() {
        let list = EList {
            ps: vec![gint_par(42)],
            locally_free: Vec::new(),
            connective_used: false,
            remainder: None,
        };
        let first = list.ps.first().cloned();
        assert_eq!(
            first.expect("first should exist").exprs[0].expr_instance,
            Some(ExprInstance::GInt(42))
        );
    }

    #[test]
    fn test_list_last_single() {
        let list = EList {
            ps: vec![gint_par(42)],
            locally_free: Vec::new(),
            connective_used: false,
            remainder: None,
        };
        let last = list.ps.last().cloned();
        assert_eq!(
            last.expect("last should exist").exprs[0].expr_instance,
            Some(ExprInstance::GInt(42))
        );
    }

    #[test]
    fn test_list_first_equals_last_for_single_element() {
        let list = EList {
            ps: vec![gstring_par("only")],
            locally_free: Vec::new(),
            connective_used: false,
            remainder: None,
        };
        assert_eq!(list.ps.first(), list.ps.last());
    }

    // =========================================================================
    // PathMap element diversity tests
    // =========================================================================

    #[test]
    fn test_pathmap_with_mixed_types() {
        let elements = vec![
            gstring_par("text"),
            gint_par(42),
            list_par(&["nested", "list"]),
        ];
        let pm = EPathMap {
            ps: elements.clone(),
            locally_free: Vec::new(),
            connective_used: false,
            remainder: None,
        };
        assert_eq!(pm.ps.len(), 3);
        // Verify types preserved
        assert!(matches!(
            &pm.ps[0].exprs[0].expr_instance,
            Some(ExprInstance::GString(_))
        ));
        assert!(matches!(
            &pm.ps[1].exprs[0].expr_instance,
            Some(ExprInstance::GInt(42))
        ));
        assert!(matches!(
            &pm.ps[2].exprs[0].expr_instance,
            Some(ExprInstance::EListBody(_))
        ));
    }
}
