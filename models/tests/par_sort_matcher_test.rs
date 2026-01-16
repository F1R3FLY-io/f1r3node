// See models/src/test/scala/coop/rchain/models/rholang/SortTest.scala - ParSortMatcherSpec

use models::rhoapi::connective::ConnectiveInstance::{
    ConnAndBody, ConnBool, ConnByteArray, ConnInt, ConnNotBody, ConnOrBody, ConnString, ConnUri,
    VarRefBody,
};
use models::rhoapi::expr::ExprInstance;
use models::rhoapi::{
    Connective, ConnectiveBody, EMap, Expr, KeyValuePair, Match, MatchCase, New, Receive,
    ReceiveBind, VarRef,
};
use models::rust::rholang::sorter::expr_sort_matcher::ExprSortMatcher;
use models::rust::utils::{
    new_boundvar_expr, new_boundvar_par, new_bundle_par, new_ediv_expr_gint, new_eeq_expr_gint,
    new_egt_expr_gbool, new_egte_expr_gbool, new_elt_expr_gint, new_elte_expr_gint, new_emap_par,
    new_emethod_expr, new_eminus_expr_gint, new_emult_expr_gint, new_eneq_expr_gint, new_eor_expr,
    new_eplus_expr_gint, new_eplus_par, new_eplus_par_gint, new_eset_par, new_freevar_expr,
    new_freevar_par, new_gbool_expr, new_gbool_par, new_gint_par, new_gstring_expr, new_guri_expr,
    new_key_value_pair, new_new_par, new_send_par, new_wildcard_par,
};
use models::{
    create_bit_vector,
    rhoapi::Par,
    rhoapi::Send,
    rust::{
        rholang::sorter::{par_sort_matcher::ParSortMatcher, sortable::Sortable},
        utils::new_gint_expr,
    },
};
use std::collections::BTreeMap;

#[test]
fn par_should_sort_so_that_smaller_integers_come_first() {
    let par_ground = Par::default().with_exprs(vec![
        new_gint_expr(2),
        new_gint_expr(1),
        new_gint_expr(-1),
        new_gint_expr(-2),
        new_gint_expr(0),
    ]);

    let sorted_par_ground = Par::default().with_exprs(vec![
        new_gint_expr(-2),
        new_gint_expr(-1),
        new_gint_expr(0),
        new_gint_expr(1),
        new_gint_expr(2),
    ]);

    assert_eq!(
        ParSortMatcher::sort_match(&par_ground).term,
        sorted_par_ground
    );
}

#[test]
fn par_should_sort_in_order_of_boolean_int_string_uri() {
    let par_ground = Par::default().with_exprs(vec![
        new_guri_expr("https://www.rchain.coop/".to_string()),
        new_gint_expr(47),
        new_gstring_expr("Hello".to_string()),
        new_gbool_expr(true),
    ]);

    let sorted_par_ground = Par::default().with_exprs(vec![
        new_gbool_expr(true),
        new_gint_expr(47),
        new_gstring_expr("Hello".to_string()),
        new_guri_expr("https://www.rchain.coop/".to_string()),
    ]);

    assert_eq!(
        ParSortMatcher::sort_match(&par_ground).term,
        sorted_par_ground
    );
}

#[test]
fn par_should_sort_and_deduplicate_sets_insides() {
    let par_ground = new_eset_par(
        vec![
            new_gint_par(2, vec![], false),
            new_gint_par(1, vec![], false),
            new_eset_par(
                vec![
                    new_gint_par(1, vec![], false),
                    new_gint_par(2, vec![], false),
                ],
                vec![],
                false,
                None,
                vec![],
                false,
            ),
            new_eset_par(
                vec![
                    new_gint_par(1, vec![], false),
                    new_gint_par(1, vec![], false),
                ],
                vec![],
                false,
                None,
                vec![],
                false,
            ),
        ],
        vec![],
        false,
        None,
        vec![],
        false,
    );

    let sorted_par_ground = new_eset_par(
        vec![
            new_gint_par(1, vec![], false),
            new_gint_par(2, vec![], false),
            new_eset_par(
                vec![new_gint_par(1, vec![], false)],
                vec![],
                false,
                None,
                vec![],
                false,
            ),
            new_eset_par(
                vec![
                    new_gint_par(1, vec![], false),
                    new_gint_par(2, vec![], false),
                ],
                vec![],
                false,
                None,
                vec![],
                false,
            ),
        ],
        vec![],
        false,
        None,
        vec![],
        false,
    );

    assert_eq!(
        ParSortMatcher::sort_match(&par_ground).term,
        sorted_par_ground
    );
}

#[test]
fn par_should_sort_map_insides_by_key_and_last_write_should_win() {
    let par_ground = new_emap_par(
        vec![
            new_key_value_pair(
                new_gint_par(2, vec![], false),
                new_eset_par(
                    vec![
                        new_gint_par(2, vec![], false),
                        new_gint_par(1, vec![], false),
                    ],
                    vec![],
                    false,
                    None,
                    vec![],
                    false,
                ),
            ),
            new_key_value_pair(
                new_gint_par(2, vec![], false),
                new_gint_par(1, vec![], false),
            ),
            new_key_value_pair(
                new_gint_par(1, vec![], false),
                new_gint_par(1, vec![], false),
            ),
        ],
        vec![],
        false,
        None,
        vec![],
        false,
    );

    let sorted_par_ground = new_emap_par(
        vec![
            new_key_value_pair(
                new_gint_par(1, vec![], false),
                new_gint_par(1, vec![], false),
            ),
            new_key_value_pair(
                new_gint_par(2, vec![], false),
                new_gint_par(1, vec![], false),
            ),
        ],
        vec![],
        false,
        None,
        vec![],
        false,
    );

    assert_eq!(
        ParSortMatcher::sort_match(&par_ground).term,
        sorted_par_ground
    );
}

#[test]
fn par_should_use_sorted_subtrees_and_their_scores_in_results() {
    let s1 = Send {
        chan: Some(Par::default()),
        ..Default::default()
    };

    let s2 = Send {
        chan: Some(Par::default().with_receives(vec![Receive {
            binds: vec![],
            body: Some(Par::default()),
            ..Default::default()
        }])),
        ..Default::default()
    };

    let p21 = Par::default().with_sends(vec![s2.clone(), s1.clone()]);
    let p12 = Par::default().with_sends(vec![s1.clone(), s2.clone()]);

    assert_eq!(
        p12,
        ParSortMatcher::sort_match(&p12).term,
        "p12 is not returned as sorted itself"
    );
    assert_ne!(
        p21,
        ParSortMatcher::sort_match(&p21).term,
        "p21 is returned as sorted itself"
    );

    assert_eq!(
        ParSortMatcher::sort_match(&p12).term,
        ParSortMatcher::sort_match(&p21).term,
        "Sorted p12 and p21 do not have the same structure"
    );

    assert_eq!(
        ParSortMatcher::sort_match(&p12),
        ParSortMatcher::sort_match(&p21),
        "Sorted p12 and p21 do not match in all properties"
    );
}

#[test]
fn par_should_keep_order_when_adding_numbers() {
    let par_expr = new_eplus_par(
        new_eplus_par_gint(1, 3, Vec::new(), false),
        new_gint_par(2, Vec::new(), false),
    );

    let result = ParSortMatcher::sort_match(&par_expr);

    assert_eq!(
        result.term, par_expr,
        "The order of expressions in par_expr is not preserved"
    );
}

#[test]
fn par_should_sort_according_to_pemdas() {
    let par_expr = Par {
        exprs: vec![
            new_eminus_expr_gint(4, 3, Vec::new(), false),
            new_ediv_expr_gint(1, 5, Vec::new(), false),
            new_eplus_expr_gint(1, 3, Vec::new(), false),
            new_emult_expr_gint(6, 3, Vec::new(), false),
        ],
        ..Default::default()
    };

    let sorted_par_expr = Par {
        exprs: vec![
            new_emult_expr_gint(6, 3, Vec::new(), false),
            new_ediv_expr_gint(1, 5, Vec::new(), false),
            new_eplus_expr_gint(1, 3, Vec::new(), false),
            new_eminus_expr_gint(4, 3, Vec::new(), false),
        ],
        ..Default::default()
    };

    let result = ParSortMatcher::sort_match(&par_expr);

    assert_eq!(
        result.term, sorted_par_expr,
        "Expressions were not sorted according to PEMDAS"
    );
}

#[test]
fn par_should_sort_comparisons_in_order_of_lt_lte_gt_gte_eq_neq() {
    let par_expr = Par {
        exprs: vec![
            new_eeq_expr_gint(4, 3, Vec::new(), false),
            new_eneq_expr_gint(1, 5, Vec::new(), false),
            new_elt_expr_gint(1, 5, Vec::new(), false),
            new_egt_expr_gbool(false, true, Vec::new(), false),
            new_elte_expr_gint(1, 5, Vec::new(), false),
            new_egte_expr_gbool(false, true, Vec::new(), false),
        ],
        ..Default::default()
    };

    let sorted_par_expr = Par {
        exprs: vec![
            new_elt_expr_gint(1, 5, Vec::new(), false),
            new_elte_expr_gint(1, 5, Vec::new(), false),
            new_egt_expr_gbool(false, true, Vec::new(), false),
            new_egte_expr_gbool(false, true, Vec::new(), false),
            new_eeq_expr_gint(4, 3, Vec::new(), false),
            new_eneq_expr_gint(1, 5, Vec::new(), false),
        ],
        ..Default::default()
    };

    let result = ParSortMatcher::sort_match(&par_expr);

    assert_eq!(
        result.term, sorted_par_expr,
        "Comparisons were not sorted in the correct order (LT, LTE, GT, GTE, EQ, NEQ)"
    );
}

#[test]
fn par_should_sort_methods_after_other_expressions() {
    let par_expr = Par {
        exprs: vec![
            new_eor_expr(
                new_boundvar_par(0, Vec::new(), false),
                new_boundvar_par(1, Vec::new(), false),
            ),
            new_emethod_expr(
                "nth".to_string(),
                new_boundvar_par(2, create_bit_vector(&vec![2]), false),
                vec![new_gint_par(1, Vec::new(), false)],
                create_bit_vector(&vec![2]),
            ),
            new_eor_expr(
                new_boundvar_par(3, Vec::new(), false),
                new_boundvar_par(4, Vec::new(), false),
            ),
        ],
        ..Default::default()
    };

    let sorted_par_expr = Par {
        exprs: vec![
            new_eor_expr(
                new_boundvar_par(0, Vec::new(), false),
                new_boundvar_par(1, Vec::new(), false),
            ),
            new_eor_expr(
                new_boundvar_par(3, Vec::new(), false),
                new_boundvar_par(4, Vec::new(), false),
            ),
            new_emethod_expr(
                "nth".to_string(),
                new_boundvar_par(2, create_bit_vector(&vec![2]), false),
                vec![new_gint_par(1, Vec::new(), false)],
                create_bit_vector(&vec![2]),
            ),
        ],
        ..Default::default()
    };

    let result = ParSortMatcher::sort_match(&par_expr);

    assert_eq!(
        result.term, sorted_par_expr,
        "Methods were not sorted after other expressions"
    );
}

#[test]
fn par_should_sort_methods_based_on_method_name_target_and_arguments() {
    let par_expr = Par {
        exprs: vec![
            new_emethod_expr(
                "nth".to_string(),
                new_boundvar_par(2, create_bit_vector(&vec![2]), false),
                vec![new_gint_par(1, Vec::new(), false)],
                create_bit_vector(&vec![2]),
            ),
            new_emethod_expr(
                "mth".to_string(),
                new_boundvar_par(2, create_bit_vector(&vec![2]), false),
                vec![new_gint_par(1, Vec::new(), false)],
                create_bit_vector(&vec![2]),
            ),
            new_emethod_expr(
                "nth".to_string(),
                new_boundvar_par(2, create_bit_vector(&vec![2]), false),
                vec![
                    new_gint_par(2, Vec::new(), false),
                    new_gint_par(3, Vec::new(), false),
                ],
                create_bit_vector(&vec![2]),
            ),
            new_emethod_expr(
                "nth".to_string(),
                new_boundvar_par(2, create_bit_vector(&vec![2]), false),
                vec![new_gint_par(2, Vec::new(), false)],
                create_bit_vector(&vec![2]),
            ),
        ],
        ..Default::default()
    };

    let sorted_par_expr = Par {
        exprs: vec![
            new_emethod_expr(
                "mth".to_string(),
                new_boundvar_par(2, create_bit_vector(&vec![2]), false),
                vec![new_gint_par(1, Vec::new(), false)],
                create_bit_vector(&vec![2]),
            ),
            new_emethod_expr(
                "nth".to_string(),
                new_boundvar_par(2, create_bit_vector(&vec![2]), false),
                vec![new_gint_par(1, Vec::new(), false)],
                create_bit_vector(&vec![2]),
            ),
            new_emethod_expr(
                "nth".to_string(),
                new_boundvar_par(2, create_bit_vector(&vec![2]), false),
                vec![new_gint_par(2, Vec::new(), false)],
                create_bit_vector(&vec![2]),
            ),
            new_emethod_expr(
                "nth".to_string(),
                new_boundvar_par(2, create_bit_vector(&vec![2]), false),
                vec![
                    new_gint_par(2, Vec::new(), false),
                    new_gint_par(3, Vec::new(), false),
                ],
                create_bit_vector(&vec![2]),
            ),
        ],
        ..Default::default()
    };

    let result = ParSortMatcher::sort_match(&par_expr);

    assert_eq!(
        result.term, sorted_par_expr,
        "Methods were not sorted based on methodName, target, and arguments"
    );
}

#[test]
fn par_should_sort_sends_based_on_persistence_channel_and_data() {
    let par_expr = Par {
        sends: vec![
            Send {
                chan: Some(new_gint_par(5, Vec::new(), false)),
                data: vec![new_gint_par(3, Vec::new(), false)],
                persistent: false,
                locally_free: Vec::new(),
                connective_used: false,
                hyperparams: vec![]
            },
            Send {
                chan: Some(new_gint_par(5, Vec::new(), false)),
                data: vec![new_gint_par(3, Vec::new(), false)],
                persistent: true,
                locally_free: Vec::new(),
                connective_used: false,
                hyperparams: vec![]
            },
            Send {
                chan: Some(new_gint_par(4, Vec::new(), false)),
                data: vec![new_gint_par(2, Vec::new(), false)],
                persistent: false,
                locally_free: Vec::new(),
                connective_used: false,
                hyperparams: vec![]
            },
            Send {
                chan: Some(new_gint_par(5, Vec::new(), false)),
                data: vec![new_gint_par(2, Vec::new(), false)],
                persistent: false,
                locally_free: Vec::new(),
                connective_used: false,
                hyperparams: vec![]
            },
        ],
        ..Default::default()
    };

    let sorted_par_expr = Par {
        sends: vec![
            Send {
                chan: Some(new_gint_par(4, Vec::new(), false)),
                data: vec![new_gint_par(2, Vec::new(), false)],
                persistent: false,
                locally_free: Vec::new(),
                connective_used: false,
                hyperparams: vec![]
            },
            Send {
                chan: Some(new_gint_par(5, Vec::new(), false)),
                data: vec![new_gint_par(2, Vec::new(), false)],
                persistent: false,
                locally_free: Vec::new(),
                connective_used: false,
                hyperparams: vec![]
            },
            Send {
                chan: Some(new_gint_par(5, Vec::new(), false)),
                data: vec![new_gint_par(3, Vec::new(), false)],
                persistent: false,
                locally_free: Vec::new(),
                connective_used: false,
                hyperparams: vec![]
            },
            Send {
                chan: Some(new_gint_par(5, Vec::new(), false)),
                data: vec![new_gint_par(3, Vec::new(), false)],
                persistent: true,
                locally_free: Vec::new(),
                connective_used: false,
                hyperparams: vec![]
            },
        ],
        ..Default::default()
    };

    let result = ParSortMatcher::sort_match(&par_expr);

    assert_eq!(
        result.term, sorted_par_expr,
        "Sends were not sorted based on persistence, channel, and data"
    );
}

#[test]
fn par_should_sort_receives_based_on_persistence_peek_channels_patterns_and_body() {
    let par_expr = Par {
        receives: vec![
            Receive {
                binds: vec![ReceiveBind {
                    patterns: vec![new_gint_par(1, Vec::new(), false)],
                    source: Some(new_gint_par(3, Vec::new(), false)),
                    remainder: None,
                    free_count: 0,
                    pattern_modifiers: vec![],
                }],
                body: Some(Par::default()),
                persistent: false,
                peek: false,
                bind_count: 0,
                locally_free: Vec::new(),
                connective_used: false,
            },
            Receive {
                binds: vec![ReceiveBind {
                    patterns: vec![new_gint_par(0, Vec::new(), false)],
                    source: Some(new_gint_par(3, Vec::new(), false)),
                    remainder: None,
                    free_count: 0,
                    pattern_modifiers: vec![],
                }],
                body: Some(new_boundvar_par(0, Vec::new(), false)),
                persistent: false,
                peek: false,
                bind_count: 0,
                locally_free: Vec::new(),
                connective_used: false,
            },
            Receive {
                binds: vec![ReceiveBind {
                    patterns: vec![new_gint_par(0, Vec::new(), false)],
                    source: Some(new_gint_par(3, Vec::new(), false)),
                    remainder: None,
                    free_count: 0,
                    pattern_modifiers: vec![],
                }],
                body: Some(Par::default()),
                persistent: false,
                peek: false,
                bind_count: 0,
                locally_free: Vec::new(),
                connective_used: false,
            },
            Receive {
                binds: vec![ReceiveBind {
                    patterns: vec![new_gint_par(0, Vec::new(), false)],
                    source: Some(new_gint_par(3, Vec::new(), false)),
                    remainder: None,
                    free_count: 0,
                    pattern_modifiers: vec![],
                }],
                body: Some(Par::default()),
                persistent: true,
                peek: false,
                bind_count: 0,
                locally_free: Vec::new(),
                connective_used: false,
            },
            Receive {
                binds: vec![ReceiveBind {
                    patterns: vec![new_gint_par(0, Vec::new(), false)],
                    source: Some(new_gint_par(3, Vec::new(), false)),
                    remainder: None,
                    free_count: 0,
                    pattern_modifiers: vec![],
                }],
                body: Some(Par::default()),
                persistent: true,
                peek: true,
                bind_count: 0,
                locally_free: Vec::new(),
                connective_used: false,
            },
            Receive {
                binds: vec![ReceiveBind {
                    patterns: vec![new_gint_par(100, Vec::new(), false)],
                    source: Some(new_gint_par(2, Vec::new(), false)),
                    remainder: None,
                    free_count: 0,
                    pattern_modifiers: vec![],
                }],
                body: Some(Par::default()),
                persistent: false,
                peek: false,
                bind_count: 0,
                locally_free: Vec::new(),
                connective_used: false,
            },
        ],
        ..Default::default()
    };

    let sorted_par_expr = Par {
        receives: vec![
            Receive {
                binds: vec![ReceiveBind {
                    patterns: vec![new_gint_par(100, Vec::new(), false)],
                    source: Some(new_gint_par(2, Vec::new(), false)),
                    remainder: None,
                    free_count: 0,
                    pattern_modifiers: vec![],
                }],
                body: Some(Par::default()),
                persistent: false,
                peek: false,
                bind_count: 0,
                locally_free: Vec::new(),
                connective_used: false,
            },
            Receive {
                binds: vec![ReceiveBind {
                    patterns: vec![new_gint_par(0, Vec::new(), false)],
                    source: Some(new_gint_par(3, Vec::new(), false)),
                    remainder: None,
                    free_count: 0,
                    pattern_modifiers: vec![],
                }],
                body: Some(Par::default()),
                persistent: false,
                peek: false,
                bind_count: 0,
                locally_free: Vec::new(),
                connective_used: false,
            },
            Receive {
                binds: vec![ReceiveBind {
                    patterns: vec![new_gint_par(0, Vec::new(), false)],
                    source: Some(new_gint_par(3, Vec::new(), false)),
                    remainder: None,
                    free_count: 0,
                    pattern_modifiers: vec![],
                }],
                body: Some(new_boundvar_par(0, Vec::new(), false)),
                persistent: false,
                peek: false,
                bind_count: 0,
                locally_free: Vec::new(),
                connective_used: false,
            },
            Receive {
                binds: vec![ReceiveBind {
                    patterns: vec![new_gint_par(1, Vec::new(), false)],
                    source: Some(new_gint_par(3, Vec::new(), false)),
                    remainder: None,
                    free_count: 0,
                    pattern_modifiers: vec![],
                }],
                body: Some(Par::default()),
                persistent: false,
                peek: false,
                bind_count: 0,
                locally_free: Vec::new(),
                connective_used: false,
            },
            Receive {
                binds: vec![ReceiveBind {
                    patterns: vec![new_gint_par(0, Vec::new(), false)],
                    source: Some(new_gint_par(3, Vec::new(), false)),
                    remainder: None,
                    free_count: 0,
                    pattern_modifiers: vec![],
                }],
                body: Some(Par::default()),
                persistent: true,
                peek: false,
                bind_count: 0,
                locally_free: Vec::new(),
                connective_used: false,
            },
            Receive {
                binds: vec![ReceiveBind {
                    patterns: vec![new_gint_par(0, Vec::new(), false)],
                    source: Some(new_gint_par(3, Vec::new(), false)),
                    remainder: None,
                    free_count: 0,
                    pattern_modifiers: vec![],
                }],
                body: Some(Par::default()),
                persistent: true,
                peek: true,
                bind_count: 0,
                locally_free: Vec::new(),
                connective_used: false,
            },
        ],
        ..Default::default()
    };

    let result = ParSortMatcher::sort_match(&par_expr);

    assert_eq!(
        result.term, sorted_par_expr,
        "Receives were not sorted based on persistence, peek, channels, patterns, and body"
    );
}

#[test]
fn par_should_sort_matches_based_on_value_and_cases() {
    let par_match = Par {
        matches: vec![
            Match {
                target: Some(new_gint_par(5, Vec::new(), false)),
                cases: vec![
                    MatchCase {
                        pattern: Some(new_gint_par(5, Vec::new(), false)),
                        source: Some(new_gint_par(5, Vec::new(), false)),
                        free_count: 0,
                    },
                    MatchCase {
                        pattern: Some(new_gint_par(4, Vec::new(), false)),
                        source: Some(new_gint_par(4, Vec::new(), false)),
                        free_count: 0,
                    },
                ],
                locally_free: Vec::new(),
                connective_used: false,
            },
            Match {
                target: Some(new_gbool_par(true, Vec::new(), false)),
                cases: vec![
                    MatchCase {
                        pattern: Some(new_gint_par(5, Vec::new(), false)),
                        source: Some(new_gint_par(5, Vec::new(), false)),
                        free_count: 0,
                    },
                    MatchCase {
                        pattern: Some(new_gint_par(4, Vec::new(), false)),
                        source: Some(new_gint_par(4, Vec::new(), false)),
                        free_count: 0,
                    },
                ],
                locally_free: Vec::new(),
                connective_used: false,
            },
            Match {
                target: Some(new_gbool_par(true, Vec::new(), false)),
                cases: vec![
                    MatchCase {
                        pattern: Some(new_gint_par(4, Vec::new(), false)),
                        source: Some(new_gint_par(4, Vec::new(), false)),
                        free_count: 0,
                    },
                    MatchCase {
                        pattern: Some(new_gint_par(3, Vec::new(), false)),
                        source: Some(new_gint_par(3, Vec::new(), false)),
                        free_count: 0,
                    },
                ],
                locally_free: Vec::new(),
                connective_used: false,
            },
        ],
        ..Default::default()
    };

    let sorted_par_match = Par {
        matches: vec![
            Match {
                target: Some(new_gbool_par(true, Vec::new(), false)),
                cases: vec![
                    MatchCase {
                        pattern: Some(new_gint_par(4, Vec::new(), false)),
                        source: Some(new_gint_par(4, Vec::new(), false)),
                        free_count: 0,
                    },
                    MatchCase {
                        pattern: Some(new_gint_par(3, Vec::new(), false)),
                        source: Some(new_gint_par(3, Vec::new(), false)),
                        free_count: 0,
                    },
                ],
                locally_free: Vec::new(),
                connective_used: false,
            },
            Match {
                target: Some(new_gbool_par(true, Vec::new(), false)),
                cases: vec![
                    MatchCase {
                        pattern: Some(new_gint_par(5, Vec::new(), false)),
                        source: Some(new_gint_par(5, Vec::new(), false)),
                        free_count: 0,
                    },
                    MatchCase {
                        pattern: Some(new_gint_par(4, Vec::new(), false)),
                        source: Some(new_gint_par(4, Vec::new(), false)),
                        free_count: 0,
                    },
                ],
                locally_free: Vec::new(),
                connective_used: false,
            },
            Match {
                target: Some(new_gint_par(5, Vec::new(), false)),
                cases: vec![
                    MatchCase {
                        pattern: Some(new_gint_par(5, Vec::new(), false)),
                        source: Some(new_gint_par(5, Vec::new(), false)),
                        free_count: 0,
                    },
                    MatchCase {
                        pattern: Some(new_gint_par(4, Vec::new(), false)),
                        source: Some(new_gint_par(4, Vec::new(), false)),
                        free_count: 0,
                    },
                ],
                locally_free: Vec::new(),
                connective_used: false,
            },
        ],
        ..Default::default()
    };

    let result = ParSortMatcher::sort_match(&par_match);

    assert_eq!(
        result.term, sorted_par_match,
        "Matches were not sorted based on value and cases"
    );
}

#[test]
fn par_should_sort_news_based_on_bindcount_uris_and_body() {
    let par_new = Par {
        news: vec![
            New {
                bind_count: 2,
                p: Some(Par::default()),
                uri: Vec::new(),
                injections: BTreeMap::new(),
                locally_free: Vec::new(),
                space_types: vec![],
            },
            New {
                bind_count: 2,
                p: Some(Par::default()),
                uri: vec!["rho:io:stderr".to_string()],
                injections: BTreeMap::new(),
                locally_free: Vec::new(),
                space_types: vec![],
            },
            New {
                bind_count: 2,
                p: Some(Par::default()),
                uri: vec!["rho:io:stdout".to_string()],
                injections: BTreeMap::new(),
                locally_free: Vec::new(),
                space_types: vec![],
            },
            New {
                bind_count: 1,
                p: Some(Par::default()),
                uri: Vec::new(),
                injections: BTreeMap::new(),
                locally_free: Vec::new(),
                space_types: vec![],
            },
            New {
                bind_count: 2,
                p: Some(new_gint_par(7, Vec::new(), false)),
                uri: vec!["rho:io:stdout".to_string()],
                injections: BTreeMap::new(),
                locally_free: Vec::new(),
                space_types: vec![],
            },
        ],
        ..Default::default()
    };

    let sorted_par_new = Par {
        news: vec![
            New {
                bind_count: 1,
                p: Some(Par::default()),
                uri: Vec::new(),
                injections: BTreeMap::new(),
                locally_free: Vec::new(),
                space_types: vec![],
            },
            New {
                bind_count: 2,
                p: Some(Par::default()),
                uri: Vec::new(),
                injections: BTreeMap::new(),
                locally_free: Vec::new(),
                space_types: vec![],
            },
            New {
                bind_count: 2,
                p: Some(Par::default()),
                uri: vec!["rho:io:stderr".to_string()],
                injections: BTreeMap::new(),
                locally_free: Vec::new(),
                space_types: vec![],
            },
            New {
                bind_count: 2,
                p: Some(Par::default()),
                uri: vec!["rho:io:stdout".to_string()],
                injections: BTreeMap::new(),
                locally_free: Vec::new(),
                space_types: vec![],
            },
            New {
                bind_count: 2,
                p: Some(new_gint_par(7, Vec::new(), false)),
                uri: vec!["rho:io:stdout".to_string()],
                injections: BTreeMap::new(),
                locally_free: Vec::new(),
                space_types: vec![],
            },
        ],
        ..Default::default()
    };

    let result = ParSortMatcher::sort_match(&par_new);

    assert_eq!(
        result.term, sorted_par_new,
        "News were not sorted based on bindCount, uri, and body"
    );
}

#[test]
fn par_should_sort_uris_in_news() {
    let par_new = Par {
        news: vec![New {
            bind_count: 1,
            p: Some(Par::default()),
            uri: vec!["rho:io:stdout".to_string(), "rho:io:stderr".to_string()],
            injections: BTreeMap::new(),
            locally_free: Vec::new(),
            space_types: vec![],
        }],
        ..Default::default()
    };

    let sorted_par_new = Par {
        news: vec![New {
            bind_count: 1,
            p: Some(Par::default()),
            uri: vec!["rho:io:stderr".to_string(), "rho:io:stdout".to_string()],
            injections: BTreeMap::new(),
            locally_free: Vec::new(),
            space_types: vec![],
        }],
        ..Default::default()
    };

    let result = ParSortMatcher::sort_match(&par_new);

    assert_eq!(
        result.term, sorted_par_new,
        "URIs in News were not sorted correctly"
    );
}

#[test]
fn par_should_sort_evars_based_on_type_and_levels() {
    let par_ground = Par {
        exprs: vec![
            new_freevar_expr(2),
            new_freevar_expr(1),
            new_boundvar_expr(2),
            new_boundvar_expr(1),
        ],
        ..Default::default()
    };

    let sorted_par_ground = Par {
        exprs: vec![
            new_boundvar_expr(1),
            new_boundvar_expr(2),
            new_freevar_expr(1),
            new_freevar_expr(2),
        ],
        ..Default::default()
    };

    let result = ParSortMatcher::sort_match(&par_ground);

    assert_eq!(
        result.term, sorted_par_ground,
        "EVars were not sorted based on their type and levels"
    );
}

#[test]
fn par_should_sort_exprs_in_order_of_ground_vars_arithmetic_comparisons_logical() {
    let par_expr = Par {
        exprs: vec![
            new_eeq_expr_gint(4, 3, Vec::new(), false),
            new_ediv_expr_gint(1, 5, Vec::new(), false),
            new_boundvar_expr(1),
            new_eor_expr(
                new_gbool_par(false, Vec::new(), false),
                new_gbool_par(true, Vec::new(), false),
            ),
            new_gint_expr(1),
        ],
        ..Default::default()
    };

    let sorted_par_expr = Par {
        exprs: vec![
            new_gint_expr(1),
            new_boundvar_expr(1),
            new_ediv_expr_gint(1, 5, Vec::new(), false),
            new_eeq_expr_gint(4, 3, Vec::new(), false),
            new_eor_expr(
                new_gbool_par(false, Vec::new(), false),
                new_gbool_par(true, Vec::new(), false),
            ),
        ],
        ..Default::default()
    };

    let result = ParSortMatcher::sort_match(&par_expr);

    assert_eq!(
    result.term, sorted_par_expr,
    "Expressions were not sorted in the correct order: ground, vars, arithmetic, comparisons, logical"
  );
}

#[test]
fn par_should_sort_expressions_inside_bundle() {
    let par_expr = Par {
        exprs: vec![
            new_eeq_expr_gint(4, 3, Vec::new(), false),
            new_ediv_expr_gint(1, 5, Vec::new(), false),
            new_boundvar_expr(1),
            new_eor_expr(
                new_gbool_par(false, Vec::new(), false),
                new_gbool_par(true, Vec::new(), false),
            ),
            new_gint_expr(1),
        ],
        ..Default::default()
    };

    let sorted_par_expr = Par {
        exprs: vec![
            new_gint_expr(1),
            new_boundvar_expr(1),
            new_ediv_expr_gint(1, 5, Vec::new(), false),
            new_eeq_expr_gint(4, 3, Vec::new(), false),
            new_eor_expr(
                new_gbool_par(false, Vec::new(), false),
                new_gbool_par(true, Vec::new(), false),
            ),
        ],
        ..Default::default()
    };

    let bundle = new_bundle_par(par_expr.clone(), false, false);
    let sorted_bundle = new_bundle_par(sorted_par_expr.clone(), false, false);

    let result = ParSortMatcher::sort_match(&bundle);

    assert_eq!(
        result.term, sorted_bundle,
        "Expressions inside the bundle were not sorted correctly"
    );
}

#[test]
fn par_should_sort_expressions_in_nested_bundles_preserving_polarities() {
    let par_expr = Par {
        exprs: vec![
            new_eeq_expr_gint(4, 3, Vec::new(), false),
            new_ediv_expr_gint(1, 5, Vec::new(), false),
            new_boundvar_expr(1),
            new_eor_expr(
                new_gbool_par(false, Vec::new(), false),
                new_gbool_par(true, Vec::new(), false),
            ),
            new_gint_expr(1),
        ],
        ..Default::default()
    };

    let sorted_par_expr = Par {
        exprs: vec![
            new_gint_expr(1),
            new_boundvar_expr(1),
            new_ediv_expr_gint(1, 5, Vec::new(), false),
            new_eeq_expr_gint(4, 3, Vec::new(), false),
            new_eor_expr(
                new_gbool_par(false, Vec::new(), false),
                new_gbool_par(true, Vec::new(), false),
            ),
        ],
        ..Default::default()
    };

    let nested_bundle = new_bundle_par(
        new_bundle_par(new_bundle_par(par_expr.clone(), true, false), false, true),
        false,
        true,
    );

    let sorted_nested_bundle = new_bundle_par(
        new_bundle_par(
            new_bundle_par(sorted_par_expr.clone(), true, false),
            false,
            true,
        ),
        false,
        true,
    );

    let result = ParSortMatcher::sort_match(&nested_bundle);

    assert_eq!(
        result.term, sorted_nested_bundle,
        "Nested bundles with polarities were not sorted correctly"
    );
}

#[test]
fn par_should_sort_logical_connectives_in_not_and_or_order() {
    let par_expr = Par {
        connectives: vec![
            Connective {
                connective_instance: Some(ConnAndBody(ConnectiveBody {
                    ps: vec![
                        new_freevar_par(0, Vec::new()),
                        new_send_par(
                            new_freevar_par(1, Vec::new()),
                            vec![new_freevar_par(2, Vec::new())],
                            false,
                            Vec::new(),
                            false,
                            Vec::new(),
                            false,
                        ),
                    ],
                })),
            },
            Connective {
                connective_instance: Some(ConnOrBody(ConnectiveBody {
                    ps: vec![
                        new_new_par(
                            1,
                            new_wildcard_par(Vec::new(), false),
                            vec![],
                            BTreeMap::new(),
                            Vec::new(),
                            Vec::new(),
                            false,
                        ),
                        new_new_par(
                            2,
                            new_wildcard_par(Vec::new(), false),
                            vec![],
                            BTreeMap::new(),
                            Vec::new(),
                            Vec::new(),
                            false,
                        ),
                    ],
                })),
            },
            Connective {
                connective_instance: Some(ConnNotBody(Par::default())),
            },
        ],
        connective_used: true,
        ..Default::default()
    };

    let sorted_par_expr = Par {
        connectives: vec![
            Connective {
                connective_instance: Some(ConnNotBody(Par::default())),
            },
            Connective {
                connective_instance: Some(ConnAndBody(ConnectiveBody {
                    ps: vec![
                        new_freevar_par(0, Vec::new()),
                        new_send_par(
                            new_freevar_par(1, Vec::new()),
                            vec![new_freevar_par(2, Vec::new())],
                            false,
                            Vec::new(),
                            false,
                            Vec::new(),
                            false,
                        ),
                    ],
                })),
            },
            Connective {
                connective_instance: Some(ConnOrBody(ConnectiveBody {
                    ps: vec![
                        new_new_par(
                            1,
                            new_wildcard_par(Vec::new(), false),
                            vec![],
                            BTreeMap::new(),
                            Vec::new(),
                            Vec::new(),
                            false,
                        ),
                        new_new_par(
                            2,
                            new_wildcard_par(Vec::new(), false),
                            vec![],
                            BTreeMap::new(),
                            Vec::new(),
                            Vec::new(),
                            false,
                        ),
                    ],
                })),
            },
        ],
        connective_used: true,
        ..Default::default()
    };

    let result = ParSortMatcher::sort_match(&par_expr);

    assert_eq!(
        result.term, sorted_par_expr,
        "Logical connectives were not sorted in 'not', 'and', 'or' order"
    );
}

#[test]
fn par_should_sort_logical_connectives_in_varref_bool_int_string_uri_bytearray_order() {
    let par_expr = Par {
        connectives: vec![
            Connective {
                connective_instance: Some(ConnByteArray(true)),
            },
            Connective {
                connective_instance: Some(ConnUri(true)),
            },
            Connective {
                connective_instance: Some(ConnString(true)),
            },
            Connective {
                connective_instance: Some(ConnInt(true)),
            },
            Connective {
                connective_instance: Some(ConnBool(true)),
            },
            Connective {
                connective_instance: Some(VarRefBody(VarRef::default())),
            },
        ],
        connective_used: true,
        ..Default::default()
    };

    let sorted_par_expr = Par {
        connectives: vec![
            Connective {
                connective_instance: Some(VarRefBody(VarRef::default())),
            },
            Connective {
                connective_instance: Some(ConnBool(true)),
            },
            Connective {
                connective_instance: Some(ConnInt(true)),
            },
            Connective {
                connective_instance: Some(ConnString(true)),
            },
            Connective {
                connective_instance: Some(ConnUri(true)),
            },
            Connective {
                connective_instance: Some(ConnByteArray(true)),
            },
        ],
        connective_used: true,
        ..Default::default()
    };

    let result = ParSortMatcher::sort_match(&par_expr);

    assert_eq!(
    result.term, sorted_par_expr,
    "Logical connectives were not sorted in 'varref', 'bool', 'int', 'string', 'uri', 'bytearray' order"
  );
}

#[test]
fn expr_should_sort_based_on_connective_used_flag() {
    let expr = Expr {
        expr_instance: Some(ExprInstance::EMapBody(EMap {
            kvs: vec![
                KeyValuePair {
                    key: Some(Par {
                        ..Default::default()
                    }),
                    value: Some(Par {
                        exprs: vec![],
                        ..Default::default()
                    }),
                },
                KeyValuePair {
                    key: Some(Par {
                        connective_used: true,
                        ..Default::default()
                    }),
                    value: Some(Par {
                        exprs: vec![],
                        ..Default::default()
                    }),
                },
            ],
            locally_free: vec![],
            connective_used: false,
            remainder: None,
        })),
    };

    //TODO I'm not sure about this test, used ExprSortMatcher to sort Expr
    let result = ExprSortMatcher::sort_match(&expr);

    assert_eq!(
        result.term, expr,
        "Expression was not sorted correctly based on the connective_used flag"
    );
}
