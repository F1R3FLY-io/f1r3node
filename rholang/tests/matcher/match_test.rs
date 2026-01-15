/*
 * See rholang/src/test/scala/coop/rchain/rholang/interpreter/matcher/MatchTest.scala
 *
 * It's important to note that in the Scala tests there are differences in the way they name their tests.
 * For example: 'should "work"', 'should "better be quick"', 'should "succeed"', etc...
 *
 * On the Scala side, there are multiple functions that create RhoTypes and implicitly pass them
 * to 'assertSpatialMatch' even though 'assertSpatialMatch' takes type 'Par'
 * For example: passing an 'Expr', 'Expr' to function that takes type 'Par', 'Par'
 *
 *
 * Might be able to use '::default()' at certain points
*/
use std::collections::BTreeMap;

use connective::ConnectiveInstance::*;
use expr::ExprInstance::*;
use g_unforgeable::UnfInstance::GPrivateBody;
use models::rust::par_map::ParMap;
use models::rust::par_map_type_mapper::ParMapTypeMapper;
use models::rust::par_set::ParSet;
use models::rust::par_set_type_mapper::ParSetTypeMapper;
use models::rust::utils::*;
use models::{rhoapi::*, rust::rholang::implicits::vector_par};
use rholang::rust::interpreter::matcher::spatial_matcher::{SpatialMatcher, SpatialMatcherContext};
use rholang::rust::interpreter::util::{prepend_connective, prepend_expr};

fn assert_spatial_match(
    target: Par,
    pattern: Par,
    expected_captures: Option<FreeMap>,
) -> Result<(), String> {
    // println!("\ntest target: {:?}", target);
    // println!("\ntest pattern: {:?}", pattern);

    let mut spatial_matcher = SpatialMatcherContext::new();
    let spatial_match_result = spatial_matcher.spatial_match(target, pattern);
    let result = spatial_matcher.free_map;

    if spatial_match_result.is_some() && expected_captures.is_some() {
        assert_eq!(result, expected_captures.unwrap());
    } else {
        assert_eq!(None, expected_captures);
    }

    Ok(())
}

#[test]
fn matching_ground_with_var_should_work() {
    let target = new_gint_par(7, Vec::new(), false);
    let pattern = new_freevar_par(0, Vec::new());

    let expected_captures = BTreeMap::from([(0, new_gint_par(7, Vec::new(), false))]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_bound_var_with_free_var_should_fail() {
    let target: Par = new_boundvar_par(0, Vec::new(), false);
    let pattern: Par = new_freevar_par(0, Vec::new());

    assert!(assert_spatial_match(target, pattern, None).is_ok());
}

#[test]
fn matching_bound_var_with_a_wildcard_should_succeed() {
    let target: Par = new_boundvar_par(0, Vec::new(), false);
    let pattern: Par = new_wildcard_par(Vec::new(), true);

    assert!(assert_spatial_match(target, pattern, Some(BTreeMap::new())).is_ok());
}

#[test]
fn matching_lists_of_grounds_with_lists_of_vars_should_work() {
    let target: Par = new_elist_par(
        vec![
            new_gstring_par("add".to_string(), Vec::new(), false),
            new_gint_par(7, Vec::new(), false),
            new_gint_par(8, Vec::new(), false),
        ],
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );

    let pattern: Par = new_elist_par(
        vec![
            new_gstring_par("add".to_string(), Vec::new(), false),
            new_freevar_par(0, Vec::new()),
            new_freevar_par(1, Vec::new()),
        ],
        Vec::new(),
        true,
        None,
        Vec::new(),
        true,
    );

    let expected_captures = BTreeMap::from([
        (0, new_gint_par(7, Vec::new(), false)),
        (1, new_gint_par(8, Vec::new(), false)),
    ]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_two_lists_in_parallel_should_work() {
    let target: Par = prepend_expr(
        new_elist_par(
            vec![
                new_gint_par(7, Vec::new(), false),
                new_gint_par(9, Vec::new(), false),
            ],
            Vec::new(),
            false,
            None,
            Vec::new(),
            false,
        ),
        new_elist_expr(
            vec![
                new_gint_par(7, Vec::new(), false),
                new_gint_par(8, Vec::new(), false),
            ],
            Vec::new(),
            false,
            None,
        ),
        0,
    );

    let pattern: Par = prepend_expr(
        new_elist_par(
            vec![
                new_freevar_par(0, Vec::new()),
                new_gint_par(9, Vec::new(), false),
            ],
            Vec::new(),
            true,
            None,
            Vec::new(),
            false,
        ),
        new_elist_expr(
            vec![
                new_gint_par(7, Vec::new(), false),
                new_freevar_par(1, Vec::new()),
            ],
            Vec::new(),
            true,
            None,
        ),
        1,
    );

    let expected_captures = BTreeMap::from([
        (0, new_gint_par(7, Vec::new(), false)),
        (1, new_gint_par(8, Vec::new(), false)),
    ]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_huge_list_of_targets_with_no_patterns_and_a_remainder_should_better_be_quick() {
    let target: Par = vector_par(Vec::new(), false).with_exprs(vec![new_gint_expr(1); 1000]);

    let pattern: Par = new_freevar_par(0, Vec::new());

    let expected_captures = BTreeMap::from([(0, target.clone())]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_a_sends_channel_should_work() {
    let target = new_send_par(
        vector_par(Vec::new(), false).with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(GPrivateBody(GPrivate {
                id: "unforgeable".into(),
            })),
        }]),
        vec![
            new_gint_par(7, Vec::new(), false),
            new_gint_par(8, Vec::new(), false),
        ],
        false,
        Vec::new(),
        false,
        Vec::new(),
        false,
    );

    let pattern = new_send_par(
        new_freevar_par(0, Vec::new()),
        vec![
            new_wildcard_par(Vec::new(), true),
            new_gint_par(8, Vec::new(), false),
        ],
        false,
        Vec::new(),
        true,
        Vec::new(),
        true,
    );

    let expected_captures = BTreeMap::from([(
        0,
        vector_par(Vec::new(), false).with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(GPrivateBody(GPrivate {
                id: "unforgeable".into(),
            })),
        }]),
    )]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_a_sends_body_should_work() {
    let target = new_send_par(
        vector_par(Vec::new(), false).with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(GPrivateBody(GPrivate {
                id: "unforgeable".into(),
            })),
        }]),
        vec![
            new_gint_par(7, Vec::new(), false),
            new_gint_par(8, Vec::new(), false),
        ],
        false,
        Vec::new(),
        false,
        Vec::new(),
        false,
    );

    let pattern = new_send_par(
        new_wildcard_par(Vec::new(), true),
        vec![
            new_freevar_par(0, Vec::new()),
            new_gint_par(8, Vec::new(), false),
        ],
        false,
        Vec::new(),
        true,
        Vec::new(),
        true,
    );

    let expected_captures = BTreeMap::from([(0, new_gint_par(7, Vec::new(), false))]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_a_send_should_require_arity_matching() {
    let target = new_send_par(
        vector_par(Vec::new(), false).with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(GPrivateBody(GPrivate {
                id: "unforgeable".into(),
            })),
        }]),
        vec![
            new_gint_par(7, Vec::new(), false),
            new_gint_par(8, Vec::new(), false),
            new_gint_par(9, Vec::new(), false),
        ],
        false,
        Vec::new(),
        false,
        Vec::new(),
        false,
    );

    let pattern = new_send_par(
        new_wildcard_par(Vec::new(), true),
        vec![
            new_freevar_par(0, Vec::new()),
            new_wildcard_par(Vec::new(), true),
        ],
        false,
        Vec::new(),
        true,
        Vec::new(),
        true,
    );

    assert!(assert_spatial_match(target, pattern, None).is_ok());
}

#[test]
fn matching_extras_with_free_variable_should_work() {
    let target: Par = prepend_expr(
        prepend_expr(new_gint_par(9, Vec::new(), false), new_gint_expr(8), 0),
        new_gint_expr(7),
        0,
    );

    let pattern: Par = prepend_expr(new_freevar_par(0, Vec::new()), new_gint_expr(8), 1);

    let expected_captures = BTreeMap::from([(
        0,
        prepend_expr(new_gint_par(9, Vec::new(), false), new_gint_expr(7), 0),
    )]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_a_singleton_list_should_work() {
    let target: Par = new_elist_par(
        vec![new_gint_par(1, Vec::new(), false)],
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );

    let pattern: Par = new_elist_par(
        vec![new_freevar_par(0, Vec::new())],
        Vec::new(),
        true,
        None,
        Vec::new(),
        true,
    );

    let expected_captures = BTreeMap::from([(0, new_gint_par(1, Vec::new(), false))]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_patterns_to_equal_targets_should_work() {
    let target: Par = prepend_expr(new_gint_par(1, Vec::new(), false), new_gint_expr(1), 0);
    let pattern: Par = prepend_expr(new_freevar_par(1, Vec::new()), new_freevar_expr(0), 0);

    let expected_captures = BTreeMap::from([(0, target.clone())]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_multiple_reminders_should_assign_captures_to_remainders_greedily() {
    let target: Par = prepend_expr(
        prepend_expr(new_gint_par(3, Vec::new(), false), new_gint_expr(2), 0),
        new_gint_expr(1),
        0,
    );

    let pattern: Par = prepend_expr(
        prepend_expr(new_wildcard_par(Vec::new(), false), new_freevar_expr(1), 0),
        new_freevar_expr(0),
        0,
    );

    let expected_captures = BTreeMap::from([(0, target.clone())]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_that_requires_revision_of_prior_matches_should_work() {
    let pattern = prepend_expr(
        new_elist_par(
            vec![
                new_freevar_par(1, Vec::new()),
                new_elist_par(
                    vec![new_gint_par(2, Vec::new(), false)],
                    Vec::new(),
                    false,
                    None,
                    Vec::new(),
                    false,
                ),
            ],
            Vec::new(),
            true,
            None,
            Vec::new(),
            true,
        ),
        new_elist_expr(
            vec![
                new_gint_par(1, Vec::new(), false),
                new_elist_par(
                    vec![new_freevar_par(0, Vec::new())],
                    Vec::new(),
                    true,
                    None,
                    Vec::new(),
                    true,
                ),
            ],
            Vec::new(),
            true,
            None,
        ),
        0,
    );

    let target: Par = prepend_expr(
        new_elist_par(
            vec![
                new_gint_par(1, Vec::new(), false),
                new_elist_par(
                    vec![new_gint_par(3, Vec::new(), false)],
                    Vec::new(),
                    false,
                    None,
                    Vec::new(),
                    false,
                ),
            ],
            Vec::new(),
            false,
            None,
            Vec::new(),
            false,
        ),
        new_elist_expr(
            vec![
                new_gint_par(1, Vec::new(), false),
                new_elist_par(
                    vec![new_gint_par(2, Vec::new(), false)],
                    Vec::new(),
                    false,
                    None,
                    Vec::new(),
                    false,
                ),
            ],
            Vec::new(),
            false,
            None,
        ),
        0,
    );

    let expected_captures = BTreeMap::from([
        (0, new_gint_par(3, Vec::new(), false)),
        (1, new_gint_par(1, Vec::new(), false)),
    ]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_that_requires_revision_of_prior_matches_should_work_with_remainders() {
    let pattern: Par = prepend_expr(
        new_elist_par(
            vec![
                new_freevar_par(1, Vec::new()),
                new_elist_par(
                    vec![new_gint_par(2, Vec::new(), false)],
                    Vec::new(),
                    false,
                    None,
                    Vec::new(),
                    false,
                ),
            ],
            Vec::new(),
            true,
            None,
            Vec::new(),
            true,
        ),
        new_elist_expr(
            vec![
                new_gint_par(1, Vec::new(), false),
                new_elist_par(
                    Vec::new(),
                    Vec::new(),
                    true,
                    Some(new_freevar_var(0)),
                    Vec::new(),
                    true,
                ),
            ],
            Vec::new(),
            true,
            None,
        ),
        0,
    );

    let target: Par = prepend_expr(
        new_elist_par(
            vec![
                new_gint_par(1, Vec::new(), false),
                new_elist_par(
                    vec![new_gint_par(3, Vec::new(), false)],
                    Vec::new(),
                    false,
                    None,
                    Vec::new(),
                    false,
                ),
            ],
            Vec::new(),
            false,
            None,
            Vec::new(),
            false,
        ),
        new_elist_expr(
            vec![
                new_gint_par(1, Vec::new(), false),
                new_elist_par(
                    vec![new_gint_par(2, Vec::new(), false)],
                    Vec::new(),
                    false,
                    None,
                    Vec::new(),
                    false,
                ),
            ],
            Vec::new(),
            false,
            None,
        ),
        0,
    );

    let expected_captures = BTreeMap::from([
        (
            0,
            new_elist_par(
                vec![new_gint_par(3, Vec::new(), false)],
                Vec::new(),
                false,
                None,
                Vec::new(),
                false,
            ),
        ),
        (1, new_gint_par(1, Vec::new(), false)),
    ]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_that_requires_revision_of_prior_matches_should_work_despite_having_other_terms_in_the_same_list(
) {
    let pattern: Par = prepend_expr(
        new_elist_par(
            vec![
                new_freevar_par(1, Vec::new()),
                new_elist_par(
                    vec![
                        new_gint_par(2, Vec::new(), false),
                        new_gint_par(2, Vec::new(), false),
                    ],
                    Vec::new(),
                    false,
                    None,
                    Vec::new(),
                    false,
                ),
            ],
            Vec::new(),
            true,
            None,
            Vec::new(),
            true,
        ),
        new_elist_expr(
            vec![
                new_gint_par(1, Vec::new(), false),
                new_elist_par(
                    vec![
                        new_freevar_par(0, Vec::new()),
                        new_gint_par(2, Vec::new(), false),
                    ],
                    Vec::new(),
                    true,
                    None,
                    Vec::new(),
                    true,
                ),
            ],
            Vec::new(),
            true,
            None,
        ),
        0,
    );

    let target: Par = prepend_expr(
        new_elist_par(
            vec![
                new_gint_par(1, Vec::new(), false),
                new_elist_par(
                    vec![
                        new_gint_par(3, Vec::new(), false),
                        new_gint_par(2, Vec::new(), false),
                    ],
                    Vec::new(),
                    false,
                    None,
                    Vec::new(),
                    false,
                ),
            ],
            Vec::new(),
            false,
            None,
            Vec::new(),
            false,
        ),
        new_elist_expr(
            vec![
                new_gint_par(1, Vec::new(), false),
                new_elist_par(
                    vec![
                        new_gint_par(2, Vec::new(), false),
                        new_gint_par(2, Vec::new(), false),
                    ],
                    Vec::new(),
                    false,
                    None,
                    Vec::new(),
                    false,
                ),
            ],
            Vec::new(),
            false,
            None,
        ),
        0,
    );

    let expected_captures = BTreeMap::from([
        (0, new_gint_par(3, Vec::new(), false)),
        (1, new_gint_par(1, Vec::new(), false)),
    ]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_that_requires_revision_of_prior_matches_should_work_with_remainders_and_wildcards() {
    let pattern: Par = prepend_expr(
        vector_par(Vec::new(), true).with_exprs(vec![new_freevar_expr(0)]),
        new_elist_expr(
            vec![new_wildcard_par(Vec::new(), true)],
            Vec::new(),
            true,
            None,
        ),
        0,
    );

    let target: Par = prepend_expr(
        vector_par(vec![0], false).with_exprs(vec![new_elist_expr(
            vec![new_boundvar_par(0, vec![0], false)],
            vec![0],
            false,
            None,
        )]),
        new_elist_expr(
            vec![new_gint_par(1, Vec::new(), false)],
            Vec::new(),
            false,
            None,
        ),
        0,
    );

    let expected_captures = BTreeMap::from([(
        0,
        new_elist_par(
            vec![new_gint_par(1, Vec::new(), false)],
            Vec::new(),
            false,
            None,
            Vec::new(),
            false,
        ),
    )]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_extras_with_wildcard_should_work() {
    let target: Par = prepend_expr(
        prepend_expr(
            vector_par(Vec::new(), false).with_exprs(vec![new_gint_expr(9)]),
            new_gint_expr(8),
            0,
        ),
        new_gint_expr(7),
        0,
    );

    let pattern: Par = prepend_expr(
        vector_par(Vec::new(), true).with_exprs(vec![new_wildcard_expr()]),
        new_gint_expr(8),
        1,
    );

    assert!(assert_spatial_match(target, pattern, Some(BTreeMap::new())).is_ok());
}

#[test]
fn matching_extras_with_wildcard_and_free_variable_should_capture_in_the_free_variable() {
    let target: Par = prepend_expr(
        prepend_expr(
            vector_par(Vec::new(), false).with_exprs(vec![new_gint_expr(9)]),
            new_gint_expr(8),
            0,
        ),
        new_gint_expr(7),
        0,
    );

    let pattern: Par = prepend_expr(
        prepend_expr(
            vector_par(Vec::new(), true).with_exprs(vec![new_wildcard_expr()]),
            new_freevar_expr(0),
            1,
        ),
        new_gint_expr(8),
        1,
    );

    let expected_captures = BTreeMap::from([(
        0,
        prepend_expr(new_gint_par(9, Vec::new(), false), new_gint_expr(7), 0),
    )]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_send_with_free_variable_in_channel_and_variable_position_should_capture_both_values() {
    let target: Par = new_send_par(
        vector_par(Vec::new(), false).with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(GPrivateBody(GPrivate { id: "zero".into() })),
        }]),
        vec![
            new_gint_par(7, Vec::new(), false),
            vector_par(Vec::new(), false).with_unforgeables(vec![GUnforgeable {
                unf_instance: Some(GPrivateBody(GPrivate { id: "one".into() })),
            }]),
        ],
        false,
        Vec::new(),
        false,
        Vec::new(),
        false,
    );

    let pattern: Par = new_send_par(
        vector_par(Vec::new(), true).with_exprs(vec![new_freevar_expr(0)]),
        vec![
            new_gint_par(7, Vec::new(), false),
            new_freevar_par(1, Vec::new()),
        ],
        false,
        Vec::new(),
        true,
        Vec::new(),
        true,
    );

    let expected_captures = BTreeMap::from([
        (
            0,
            vector_par(Vec::new(), false).with_unforgeables(vec![GUnforgeable {
                unf_instance: Some(GPrivateBody(GPrivate { id: "zero".into() })),
            }]),
        ),
        (
            1,
            vector_par(Vec::new(), false).with_unforgeables(vec![GUnforgeable {
                unf_instance: Some(GPrivateBody(GPrivate { id: "one".into() })),
            }]),
        ),
    ]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_a_receive_with_a_free_variable_in_the_channel_and_a_free_variable_in_the_body_should_capture_for_both_variables(
) {
    let target: Par = new_receive_par(
        vec![
            ReceiveBind {
                patterns: vec![
                    new_freevar_par(0, Vec::new()),
                    new_freevar_par(1, Vec::new()),
                ],
                source: Some(new_gint_par(7, Vec::new(), false)),
                remainder: None,
                free_count: 0,
            pattern_modifiers: vec![],
            },
            ReceiveBind {
                patterns: vec![
                    new_freevar_par(0, Vec::new()),
                    new_freevar_par(1, Vec::new()),
                ],
                source: Some(new_gint_par(8, Vec::new(), false)),
                remainder: None,
                free_count: 0,
            pattern_modifiers: vec![],
            },
        ],
        new_send_par(
            vector_par(Vec::new(), false).with_unforgeables(vec![GUnforgeable {
                unf_instance: Some(GPrivateBody(GPrivate {
                    id: "unforgeable".into(),
                })),
            }]),
            vec![
                new_gint_par(9, Vec::new(), false),
                new_gint_par(10, Vec::new(), false),
            ],
            false,
            Vec::new(),
            false,
            Vec::new(),
            false,
        ),
        false,
        false,
        4,
        Vec::new(),
        false,
        Vec::new(),
        false,
    );

    let pattern: Par = new_receive_par(
        vec![
            ReceiveBind {
                patterns: vec![
                    new_freevar_par(0, Vec::new()),
                    new_freevar_par(1, Vec::new()),
                ],
                source: Some(new_gint_par(7, Vec::new(), false)),
                remainder: None,
                free_count: 0,
            pattern_modifiers: vec![],
            },
            ReceiveBind {
                patterns: vec![
                    new_freevar_par(0, Vec::new()),
                    new_freevar_par(1, Vec::new()),
                ],
                source: Some(new_freevar_par(0, Vec::new())),
                remainder: None,
                free_count: 0,
            pattern_modifiers: vec![],
            },
        ],
        new_freevar_par(1, Vec::new()),
        false,
        false,
        4,
        Vec::new(),
        true,
        Vec::new(),
        true,
    );

    let expected_captures = BTreeMap::from([
        (0, new_gint_par(8, Vec::new(), false)),
        (
            1,
            new_send_par(
                vector_par(Vec::new(), false).with_unforgeables(vec![GUnforgeable {
                    unf_instance: Some(GPrivateBody(GPrivate {
                        id: "unforgeable".into(),
                    })),
                }]),
                vec![
                    new_gint_par(9, Vec::new(), false),
                    new_gint_par(10, Vec::new(), false),
                ],
                false,
                Vec::new(),
                false,
                Vec::new(),
                false,
            ),
        ),
    ]);
    assert!(assert_spatial_match(target, pattern.clone(), Some(expected_captures)).is_ok());
}

#[test]
fn matching_an_eval_with_no_free_variables_should_succeed_but_not_capture_anything() {
    let target: Par = new_boundvar_par(0, vec![0], false);
    let pattern: Par = new_boundvar_par(0, vec![0], false);

    assert!(assert_spatial_match(target, pattern.clone(), Some(BTreeMap::new())).is_ok());
}

#[test]
fn matching_between_news_should_match_the_bodies_if_the_new_count_is_the_same() {
    let target: Par = new_new_par(
        2,
        vector_par(Vec::new(), false)
            .prepend_send(new_send(
                new_boundvar_par(0, vec![0], false),
                vec![new_gint_par(43, Vec::new(), false)],
                false,
                vec![0],
                false,
            ))
            .prepend_send(new_send(
                new_gint_par(7, Vec::new(), false),
                vec![new_gint_par(42, Vec::new(), false)],
                false,
                Vec::new(),
                false,
            )),
        Vec::new(),
        BTreeMap::new(),
        Vec::new(),
        Vec::new(),
        false,
    );

    let pattern: Par = new_new_par(
        2,
        prepend_expr(
            vector_par(Vec::new(), false).prepend_send(new_send(
                new_gint_par(7, Vec::new(), false),
                vec![new_freevar_par(0, Vec::new())],
                false,
                Vec::new(),
                true,
            )),
            new_wildcard_expr(),
            1,
        ),
        Vec::new(),
        BTreeMap::new(),
        Vec::new(),
        Vec::new(),
        true,
    );

    let expected_captures = BTreeMap::from([(0, new_gint_par(42, Vec::new(), false))]);
    assert!(assert_spatial_match(target, pattern.clone(), Some(expected_captures)).is_ok());
}

#[test]
fn matching_between_matches_should_require_equality_of_cases_but_match_targets_and_inside_bodies() {
    let target: Par = new_match_par(
        new_elist_par(
            vec![
                new_gint_par(4, Vec::new(), false),
                new_gint_par(20, Vec::new(), false),
            ],
            Vec::new(),
            false,
            None,
            Vec::new(),
            false,
        ),
        vec![
            MatchCase {
                pattern: Some(new_elist_par(
                    vec![
                        new_freevar_par(0, Vec::new()),
                        new_freevar_par(1, Vec::new()),
                    ],
                    Vec::new(),
                    false,
                    None,
                    Vec::new(),
                    false,
                )),
                source: Some(new_send_par(
                    new_boundvar_par(1, vec![1], false),
                    vec![new_boundvar_par(0, vec![0], false)],
                    false,
                    vec![0, 1],
                    false,
                    vec![0, 1],
                    false,
                )),
                free_count: 0,
            },
            MatchCase {
                pattern: Some(new_wildcard_par(Vec::new(), true)),
                source: Some(vector_par(Vec::new(), false)),
                free_count: 0,
            },
        ],
        Vec::new(),
        false,
        Vec::new(),
        false,
    );

    let pattern: Par = new_match_par(
        new_freevar_par(0, Vec::new()),
        vec![
            MatchCase {
                pattern: Some(new_elist_par(
                    vec![
                        new_freevar_par(0, Vec::new()),
                        new_freevar_par(1, Vec::new()),
                    ],
                    Vec::new(),
                    false,
                    None,
                    Vec::new(),
                    false,
                )),
                source: Some(new_wildcard_par(Vec::new(), true)),
                free_count: 0,
            },
            MatchCase {
                pattern: Some(new_wildcard_par(Vec::new(), true)),
                source: Some(new_freevar_par(1, Vec::new())),
                free_count: 0,
            },
        ],
        Vec::new(),
        true,
        Vec::new(),
        true,
    );

    let expected_captures = BTreeMap::from([
        (
            0,
            new_elist_par(
                vec![
                    new_gint_par(4, Vec::new(), false),
                    new_gint_par(20, Vec::new(), false),
                ],
                Vec::new(),
                false,
                None,
                Vec::new(),
                false,
            ),
        ),
        (1, Par::default()),
    ]);
    assert!(assert_spatial_match(target, pattern.clone(), Some(expected_captures)).is_ok());
}

#[test]
fn matching_a_list_with_a_remainder_should_capture_the_remainder() {
    let target: Par = new_elist_par(
        vec![
            new_gint_par(1, Vec::new(), false),
            new_gint_par(2, Vec::new(), false),
        ],
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );

    let pattern: Par = new_elist_par(
        vec![new_gint_par(1, Vec::new(), false)],
        Vec::new(),
        true,
        Some(new_freevar_var(0)),
        Vec::new(),
        true,
    );

    let expected_captures = BTreeMap::from([(
        0,
        new_elist_par(
            vec![new_gint_par(2, Vec::new(), false)],
            Vec::new(),
            false,
            None,
            Vec::new(),
            false,
        ),
    )]);
    assert!(assert_spatial_match(target, pattern.clone(), Some(expected_captures)).is_ok());
}

#[test]
fn matching_sets_should_work_for_concrete_sets() {
    let target: Par = new_eset_par(
        vec![
            new_gint_par(1, Vec::new(), false),
            new_gint_par(2, Vec::new(), false),
            new_gint_par(3, Vec::new(), false),
        ],
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );

    let pattern: Par = target.clone();

    assert!(assert_spatial_match(target, pattern.clone(), Some(BTreeMap::new())).is_ok());
}

#[test]
fn matching_sets_should_work_with_free_variables_and_wildcards() {
    let target: Par = new_eset_par(
        vec![
            new_gint_par(1, Vec::new(), false),
            new_gint_par(2, Vec::new(), false),
            new_gint_par(3, Vec::new(), false),
            new_gint_par(4, Vec::new(), false),
            new_gint_par(5, Vec::new(), false),
        ],
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );

    let pattern: Par = new_eset_par(
        vec![
            new_gint_par(2, Vec::new(), false),
            new_gint_par(5, Vec::new(), false),
            new_freevar_par(0, Vec::new()),
            new_freevar_par(1, Vec::new()),
            new_wildcard_par(Vec::new(), true),
        ],
        Vec::new(),
        true,
        None,
        Vec::new(),
        true,
    );

    let expected_captures = BTreeMap::from([
        (0, new_gint_par(4, Vec::new(), false)),
        (1, new_gint_par(3, Vec::new(), false)),
    ]);
    assert!(assert_spatial_match(target.clone(), pattern, Some(expected_captures)).is_ok());

    let no_matching_pattern = new_eset_par(
        vec![
            new_gint_par(2, Vec::new(), false),
            new_gint_par(5, Vec::new(), false),
            new_freevar_par(0, Vec::new()),
            new_wildcard_par(Vec::new(), true),
        ],
        Vec::new(),
        true,
        None,
        Vec::new(),
        true,
    );

    assert!(assert_spatial_match(target, no_matching_pattern, None).is_ok());
}

#[test]
fn matching_sets_should_work_with_wildcard_remainders() {
    let target_elements = vec![
        new_gint_par(1, Vec::new(), false),
        new_gint_par(2, Vec::new(), false),
        new_gint_par(3, Vec::new(), false),
        new_gint_par(4, Vec::new(), false),
        new_gint_par(5, Vec::new(), false),
    ];

    let target: Par = new_eset_par(
        target_elements.clone(),
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );
    let pattern: Par = new_eset_par(
        vec![
            new_gint_par(1, Vec::new(), false),
            new_gint_par(4, Vec::new(), false),
        ],
        Vec::new(),
        true,
        Some(new_wildcard_var()),
        Vec::new(),
        true,
    );

    assert!(assert_spatial_match(target.clone(), pattern.clone(), Some(BTreeMap::new())).is_ok());

    let all_elements_and_wildcard: Par = new_eset_par(
        target_elements,
        Vec::new(),
        true,
        Some(new_wildcard_var()),
        Vec::new(),
        true,
    );

    assert!(assert_spatial_match(
        target.clone(),
        all_elements_and_wildcard,
        Some(BTreeMap::new())
    )
    .is_ok());

    let just_wildcard = new_eset_par(
        Vec::new(),
        Vec::new(),
        true,
        Some(new_wildcard_var()),
        Vec::new(),
        true,
    );

    assert!(assert_spatial_match(target, just_wildcard, Some(BTreeMap::new())).is_ok());
}

#[test]
fn matching_sets_should_work_with_var_remainders() {
    let target_elements = vec![
        new_gint_par(1, Vec::new(), false),
        new_gint_par(2, Vec::new(), false),
        new_gint_par(3, Vec::new(), false),
        new_gint_par(4, Vec::new(), false),
        new_gint_par(5, Vec::new(), false),
    ];

    let target: Par = new_eset_par(
        target_elements.clone(),
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );
    let pattern: Par = new_eset_par(
        vec![
            new_gint_par(1, Vec::new(), false),
            new_gint_par(4, Vec::new(), false),
            new_freevar_par(0, Vec::new()),
        ],
        Vec::new(),
        true,
        Some(new_freevar_var(1)),
        Vec::new(),
        true,
    );

    let expected_captures = BTreeMap::from([
        (0, new_gint_par(2, Vec::new(), false)),
        (
            1,
            new_eset_par(
                vec![
                    new_gint_par(3, Vec::new(), false),
                    new_gint_par(5, Vec::new(), false),
                ],
                Vec::new(),
                false,
                None,
                Vec::new(),
                false,
            ),
        ),
    ]);
    assert!(assert_spatial_match(target.clone(), pattern, Some(expected_captures)).is_ok());

    let all_elements_and_remainder: Par = new_eset_par(
        target_elements.clone(),
        Vec::new(),
        true,
        Some(new_freevar_var(0)),
        Vec::new(),
        true,
    );

    assert!(assert_spatial_match(
        target.clone(),
        all_elements_and_remainder,
        Some(BTreeMap::from([(
            0,
            new_eset_par(vec![], Vec::new(), false, None, Vec::new(), false)
        )]))
    )
    .is_ok());

    let just_remainder = new_eset_par(
        Vec::new(),
        Vec::new(),
        true,
        Some(new_freevar_var(0)),
        Vec::new(),
        true,
    );

    assert!(assert_spatial_match(
        target,
        just_remainder,
        Some(BTreeMap::from([(
            0,
            new_eset_par(target_elements, Vec::new(), false, None, Vec::new(), false)
        )]))
    )
    .is_ok());
}

#[test]
fn matching_maps_should_work_for_concrete_maps() {
    let target: Par = new_emap_par(
        vec![
            new_key_value_pair(
                new_gint_par(1, Vec::new(), false),
                new_gint_par(2, Vec::new(), false),
            ),
            new_key_value_pair(
                new_gint_par(3, Vec::new(), false),
                new_gint_par(4, Vec::new(), false),
            ),
        ],
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );

    let pattern: Par = target.clone();
    assert!(assert_spatial_match(target, pattern, Some(BTreeMap::new())).is_ok());
}

#[test]
fn matching_maps_should_work_with_free_variables_and_wildcards() {
    let target: Par = new_emap_par(
        vec![
            new_key_value_pair(
                new_gint_par(1, Vec::new(), false),
                new_gint_par(2, Vec::new(), false),
            ),
            new_key_value_pair(
                new_gint_par(3, Vec::new(), false),
                new_gint_par(4, Vec::new(), false),
            ),
            new_key_value_pair(
                new_gint_par(5, Vec::new(), false),
                new_gint_par(6, Vec::new(), false),
            ),
        ],
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );

    let pattern: Par = new_emap_par(
        vec![
            new_key_value_pair(
                new_gint_par(1, Vec::new(), false),
                new_freevar_par(1, Vec::new()),
            ),
            new_key_value_pair(
                new_gint_par(3, Vec::new(), false),
                new_gint_par(4, Vec::new(), false),
            ),
            new_key_value_pair(
                new_freevar_par(0, Vec::new()),
                new_wildcard_par(Vec::new(), true),
            ),
        ],
        vec![0, 1],
        true,
        None,
        vec![0, 1],
        true,
    );

    let expected_captures = BTreeMap::from([
        (0, new_gint_par(5, Vec::new(), false)),
        (1, new_gint_par(2, Vec::new(), false)),
    ]);
    assert!(assert_spatial_match(target.clone(), pattern, Some(expected_captures)).is_ok());

    let non_matching_pattern = new_emap_par(
        vec![
            new_key_value_pair(
                new_gint_par(3, Vec::new(), false),
                new_freevar_par(1, Vec::new()),
            ),
            new_key_value_pair(
                new_freevar_par(0, Vec::new()),
                new_wildcard_par(Vec::new(), true),
            ),
            new_key_value_pair(
                new_wildcard_par(Vec::new(), true),
                new_gint_par(4, Vec::new(), false),
            ),
        ],
        vec![0, 1],
        true,
        None,
        vec![0, 1],
        true,
    );

    assert!(assert_spatial_match(target, non_matching_pattern, None).is_ok());
}

#[test]
fn matching_maps_should_work_with_wildcard_remainders() {
    let target: Par = new_emap_par(
        vec![
            new_key_value_pair(
                new_gint_par(1, Vec::new(), false),
                new_gint_par(2, Vec::new(), false),
            ),
            new_key_value_pair(
                new_gint_par(3, Vec::new(), false),
                new_gint_par(4, Vec::new(), false),
            ),
            new_key_value_pair(
                new_gint_par(5, Vec::new(), false),
                new_gint_par(6, Vec::new(), false),
            ),
        ],
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );

    let pattern: Par = new_emap_par(
        vec![new_key_value_pair(
            new_gint_par(3, Vec::new(), false),
            new_gint_par(4, Vec::new(), false),
        )],
        vec![],
        true,
        Some(new_wildcard_var()),
        vec![],
        true,
    );

    assert!(assert_spatial_match(target.clone(), pattern, Some(BTreeMap::new())).is_ok());

    let all_elements_and_wildcard = new_emap_par(
        vec![
            new_key_value_pair(
                new_gint_par(1, Vec::new(), false),
                new_gint_par(2, Vec::new(), false),
            ),
            new_key_value_pair(
                new_gint_par(3, Vec::new(), false),
                new_gint_par(4, Vec::new(), false),
            ),
            new_key_value_pair(
                new_gint_par(5, Vec::new(), false),
                new_gint_par(6, Vec::new(), false),
            ),
        ],
        vec![],
        true,
        Some(new_wildcard_var()),
        vec![],
        true,
    );

    assert!(assert_spatial_match(
        target.clone(),
        all_elements_and_wildcard,
        Some(BTreeMap::new())
    )
    .is_ok());

    let just_wildcard = new_emap_par(vec![], vec![], true, Some(new_wildcard_var()), vec![], true);

    assert!(assert_spatial_match(target.clone(), just_wildcard, Some(BTreeMap::new())).is_ok());
}

#[test]
fn matching_maps_should_work_with_var_remainders() {
    let target_elements = vec![
        new_key_value_pair(
            new_gint_par(1, Vec::new(), false),
            new_gint_par(2, Vec::new(), false),
        ),
        new_key_value_pair(
            new_gint_par(3, Vec::new(), false),
            new_gint_par(4, Vec::new(), false),
        ),
        new_key_value_pair(
            new_gint_par(5, Vec::new(), false),
            new_gint_par(6, Vec::new(), false),
        ),
    ];
    let target: Par = new_emap_par(
        target_elements.clone(),
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );

    let pattern: Par = new_emap_par(
        vec![new_key_value_pair(
            new_freevar_par(0, Vec::new()),
            new_gint_par(4, Vec::new(), false),
        )],
        vec![],
        true,
        Some(new_freevar_var(1)),
        vec![],
        true,
    );

    let expected_captures = BTreeMap::from([
        (0, new_gint_par(3, Vec::new(), false)),
        (
            1,
            new_emap_par(
                vec![
                    new_key_value_pair(
                        new_gint_par(1, Vec::new(), false),
                        new_gint_par(2, Vec::new(), false),
                    ),
                    new_key_value_pair(
                        new_gint_par(5, Vec::new(), false),
                        new_gint_par(6, Vec::new(), false),
                    ),
                ],
                Vec::new(),
                false,
                None,
                Vec::new(),
                false,
            ),
        ),
    ]);

    assert!(assert_spatial_match(target.clone(), pattern.clone(), Some(expected_captures)).is_ok());

    let all_elements_and_remainder = new_emap_par(
        target_elements.clone(),
        vec![],
        true,
        Some(new_freevar_var(0)),
        vec![],
        true,
    );

    let expected_captures = BTreeMap::from([(
        0,
        new_emap_par(vec![], Vec::new(), false, None, Vec::new(), false),
    )]);
    assert!(assert_spatial_match(
        target.clone(),
        all_elements_and_remainder,
        Some(expected_captures)
    )
    .is_ok());

    let just_remainder = new_emap_par(vec![], vec![], true, Some(new_freevar_var(0)), vec![], true);

    let expected_captures = BTreeMap::from([(
        0,
        new_emap_par(target_elements, Vec::new(), false, None, Vec::new(), false),
    )]);
    assert!(assert_spatial_match(target.clone(), just_remainder, Some(expected_captures)).is_ok());
}

#[test]
fn matching_a_whole_list_with_a_remainder_should_capture_the_list() {
    let target: Par = new_elist_par(
        vec![
            new_gint_par(1, Vec::new(), false),
            new_gint_par(2, Vec::new(), false),
            new_gint_par(3, Vec::new(), false),
        ],
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );

    let pattern = new_elist_par(
        vec![],
        Vec::new(),
        true,
        Some(new_freevar_var(0)),
        Vec::new(),
        true,
    );

    let expected_captures = BTreeMap::from([(0, target.clone())]);
    assert!(assert_spatial_match(target, pattern.clone(), Some(expected_captures)).is_ok());
}

#[test]
fn matching_inside_bundles_should_not_be_possible() {
    let target: Par = vector_par(Vec::new(), false).with_bundles(vec![Bundle {
        body: Some(
            vector_par(Vec::new(), false)
                .prepend_send(new_send(
                    vector_par(Vec::new(), false).with_unforgeables(vec![GUnforgeable {
                        unf_instance: Some(GPrivateBody(GPrivate { id: "0".into() })),
                    }]),
                    vec![new_gint_par(43, Vec::new(), false)],
                    false,
                    Vec::new(),
                    false,
                ))
                .prepend_send(new_send(
                    new_gint_par(7, Vec::new(), false),
                    vec![new_gint_par(42, Vec::new(), false)],
                    false,
                    Vec::new(),
                    false,
                )),
        ),
        write_flag: false,
        read_flag: false,
    }]);

    let pattern: Par = vector_par(Vec::new(), false).with_bundles(vec![Bundle {
        body: Some(prepend_expr(
            vector_par(Vec::new(), false).prepend_send(new_send(
                new_gint_par(7, Vec::new(), false),
                vec![new_freevar_par(0, Vec::new())],
                false,
                Vec::new(),
                false,
            )),
            new_wildcard_expr(),
            1,
        )),
        write_flag: false,
        read_flag: false,
    }]);

    assert!(assert_spatial_match(target, pattern.clone(), None).is_ok());
}

#[test]
fn matching_inside_bundles_should_be_possible_to_match_on_entire_bundle() {
    let target: Par = vector_par(Vec::new(), false).with_bundles(vec![Bundle {
        body: Some(
            vector_par(Vec::new(), false)
                .prepend_send(new_send(
                    vector_par(Vec::new(), false).with_unforgeables(vec![GUnforgeable {
                        unf_instance: Some(GPrivateBody(GPrivate { id: "0".into() })),
                    }]),
                    vec![new_gint_par(43, Vec::new(), false)],
                    false,
                    Vec::new(),
                    false,
                ))
                .prepend_send(new_send(
                    new_gint_par(7, Vec::new(), false),
                    vec![new_gint_par(42, Vec::new(), false)],
                    false,
                    Vec::new(),
                    false,
                )),
        ),
        write_flag: false,
        read_flag: false,
    }]);

    let pattern: Par = new_freevar_par(0, Vec::new());

    let expected_captures = BTreeMap::from([(0, target.clone())]);
    assert!(assert_spatial_match(target, pattern.clone(), Some(expected_captures)).is_ok());
}

#[test]
fn matching_a_single_and_should_match_both_sides() {
    let target: Par = new_send_par(
        new_gint_par(7, Vec::new(), false),
        vec![new_gint_par(8, Vec::new(), false)],
        false,
        Vec::new(),
        false,
        Vec::new(),
        false,
    );

    let fail_target = new_send_par(
        new_gint_par(7, Vec::new(), false),
        vec![new_gint_par(9, Vec::new(), false)],
        false,
        Vec::new(),
        false,
        Vec::new(),
        false,
    );

    let pattern: Par = new_conn_and_body_par(
        vec![
            new_send_par(
                new_gint_par(7, Vec::new(), false),
                vec![new_freevar_par(0, Vec::new())],
                false,
                Vec::new(),
                true,
                Vec::new(),
                true,
            ),
            new_send_par(
                new_freevar_par(1, Vec::new()),
                vec![new_gint_par(8, Vec::new(), false)],
                false,
                Vec::new(),
                true,
                Vec::new(),
                true,
            ),
        ],
        Vec::new(),
        true,
    );

    let expected_captures = BTreeMap::from([
        (0, new_gint_par(8, Vec::new(), false)),
        (1, new_gint_par(7, Vec::new(), false)),
    ]);
    assert!(assert_spatial_match(target, pattern.clone(), Some(expected_captures)).is_ok());
    assert!(assert_spatial_match(fail_target, pattern, None).is_ok());
}

#[test]
fn matching_a_single_or_should_match_some_side() {
    let target: Par = new_send_par(
        new_gint_par(7, Vec::new(), false),
        vec![new_gint_par(8, Vec::new(), false)],
        false,
        Vec::new(),
        false,
        Vec::new(),
        false,
    );

    let fail_target: Par = new_send_par(
        new_gint_par(7, Vec::new(), false),
        vec![new_gint_par(9, Vec::new(), false)],
        false,
        Vec::new(),
        false,
        Vec::new(),
        false,
    );

    let pattern: Par = new_conn_or_body_par(
        vec![
            new_send_par(
                new_gint_par(9, Vec::new(), false),
                vec![new_freevar_par(0, Vec::new())],
                false,
                Vec::new(),
                true,
                Vec::new(),
                true,
            ),
            new_send_par(
                new_freevar_par(0, Vec::new()),
                vec![new_gint_par(8, Vec::new(), false)],
                false,
                Vec::new(),
                true,
                Vec::new(),
                true,
            ),
        ],
        Vec::new(),
        true,
    );

    assert!(assert_spatial_match(target, pattern.clone(), Some(BTreeMap::new())).is_ok());
    assert!(assert_spatial_match(fail_target, pattern, None).is_ok());
}

#[test]
fn matching_negation_should_work() {
    let target: Par = vector_par(Vec::new(), false).with_sends(vec![
        new_send(
            new_gint_par(1, Vec::new(), false),
            vec![new_gint_par(2, Vec::new(), false)],
            false,
            Vec::new(),
            false,
        ),
        new_send(
            new_gint_par(2, Vec::new(), false),
            vec![new_gint_par(3, Vec::new(), false)],
            false,
            Vec::new(),
            false,
        ),
        new_send(
            new_gint_par(3, Vec::new(), false),
            vec![new_gint_par(4, Vec::new(), false)],
            false,
            Vec::new(),
            false,
        ),
    ]);

    let pattern_connective: Connective = Connective {
        connective_instance: Some(ConnNotBody(vector_par(Vec::new(), false))),
    };

    let pattern: Par =
        vector_par(Vec::new(), true).with_connectives(vec![pattern_connective.clone()]);

    assert!(assert_spatial_match(target.clone(), pattern.clone(), Some(BTreeMap::new())).is_ok());

    let double_pattern: Par =
        pattern.with_connectives(vec![pattern_connective.clone(), pattern_connective.clone()]);

    assert!(assert_spatial_match(target.clone(), double_pattern, Some(BTreeMap::new())).is_ok());

    let triple_pattern = pattern.with_connectives(vec![
        pattern_connective.clone(),
        pattern_connective.clone(),
        pattern_connective.clone(),
    ]);

    assert!(assert_spatial_match(target.clone(), triple_pattern, Some(BTreeMap::new())).is_ok());

    let quadruple_pattern = pattern.with_connectives(vec![
        pattern_connective.clone(),
        pattern_connective.clone(),
        pattern_connective.clone(),
        pattern_connective.clone(),
    ]);

    assert!(assert_spatial_match(target, quadruple_pattern, None).is_ok());
}

#[test]
fn matching_a_complicated_connective_should_work() {
    let target: Par = vector_par(Vec::new(), false).with_sends(vec![
        new_send(
            new_gint_par(1, Vec::new(), false),
            vec![new_gint_par(6, Vec::new(), false)],
            false,
            Vec::new(),
            false,
        ),
        new_send(
            new_gint_par(2, Vec::new(), false),
            vec![new_gint_par(7, Vec::new(), false)],
            false,
            Vec::new(),
            false,
        ),
        new_send(
            new_gint_par(3, Vec::new(), false),
            vec![new_gint_par(8, Vec::new(), false)],
            false,
            Vec::new(),
            false,
        ),
    ]);

    let fail_target: Par = vector_par(Vec::new(), false).with_sends(vec![
        new_send(
            new_gint_par(1, Vec::new(), false),
            vec![new_gint_par(6, Vec::new(), false)],
            false,
            Vec::new(),
            false,
        ),
        new_send(
            new_gint_par(2, Vec::new(), false),
            vec![new_gint_par(9, Vec::new(), false)],
            false,
            Vec::new(),
            false,
        ),
        new_send(
            new_gint_par(3, Vec::new(), false),
            vec![new_gint_par(8, Vec::new(), false)],
            false,
            Vec::new(),
            false,
        ),
    ]);

    let non_null_connective: Connective = Connective {
        connective_instance: Some(ConnNotBody(vector_par(Vec::new(), false))),
    };

    let non_null_par: Par =
        vector_par(Vec::new(), true).with_connectives(vec![non_null_connective]);

    let single_factor: Connective = Connective {
        connective_instance: Some(ConnNotBody(vector_par(Vec::new(), true).with_connectives(
            vec![
                Connective {
                    connective_instance: Some(ConnNotBody(vector_par(Vec::new(), false))),
                },
                Connective {
                    connective_instance: Some(ConnNotBody(vector_par(Vec::new(), false))),
                },
            ],
        ))),
    };

    let single_factor_par: Par = vector_par(Vec::new(), true).with_connectives(vec![single_factor]);

    let capture: Connective = Connective {
        connective_instance: Some(ConnAndBody(ConnectiveBody {
            ps: vec![
                new_freevar_par(0, Vec::new()),
                new_send_par(
                    new_freevar_par(1, Vec::new()),
                    vec![new_gint_par(7, Vec::new(), false)],
                    false,
                    Vec::new(),
                    true,
                    Vec::new(),
                    true,
                ),
            ],
        })),
    };

    let prime: Connective = Connective {
        connective_instance: Some(ConnAndBody(ConnectiveBody {
            ps: vec![non_null_par.clone(), single_factor_par.clone()],
        })),
    };

    let alternative: Connective = Connective {
        connective_instance: Some(ConnOrBody(ConnectiveBody {
            ps: vec![
                new_send_par(
                    new_freevar_par(0, Vec::new()),
                    vec![new_gint_par(7, Vec::new(), false)],
                    false,
                    Vec::new(),
                    true,
                    Vec::new(),
                    true,
                ),
                new_send_par(
                    new_freevar_par(0, Vec::new()),
                    vec![new_gint_par(8, Vec::new(), false)],
                    false,
                    Vec::new(),
                    true,
                    Vec::new(),
                    true,
                ),
            ],
        })),
    };

    let pattern = vector_par(Vec::new(), true).with_connectives(vec![capture, prime, alternative]);

    let expected_captures = BTreeMap::from([
        (
            0,
            new_send_par(
                new_gint_par(2, Vec::new(), false),
                vec![new_gint_par(7, Vec::new(), false)],
                false,
                Vec::new(),
                false,
                Vec::new(),
                false,
            ),
        ),
        (1, new_gint_par(2, Vec::new(), false)),
    ]);
    assert!(assert_spatial_match(target, pattern.clone(), Some(expected_captures)).is_ok());
    assert!(assert_spatial_match(fail_target, pattern, None).is_ok());
}

#[test]
fn matching_a_target_with_var_ref_and_a_pattern_with_a_var_ref_should_ignore_locally_free() {
    let target: Par = new_new_par(
        1,
        new_receive_par(
            vec![ReceiveBind {
                patterns: vec![prepend_connective(
                    vector_par(Vec::new(), false),
                    Connective {
                        connective_instance: Some(VarRefBody(VarRef { index: 0, depth: 1 })),
                    },
                    1,
                )],
                source: Some(vector_par(Vec::new(), false)),
                remainder: None,
                free_count: 0,
                pattern_modifiers: vec![],
            }],
            vector_par(Vec::new(), false),
            false,
            false,
            0,
            vec![0],
            false,
            vec![0],
            false,
        ),
        Vec::new(),
        BTreeMap::new(),
        Vec::new(),
        Vec::new(),
        false,
    );

    let pattern = new_new_par(
        1,
        new_receive_par(
            vec![ReceiveBind {
                patterns: vec![prepend_connective(
                    vector_par(Vec::new(), false),
                    Connective {
                        connective_instance: Some(VarRefBody(VarRef { index: 0, depth: 1 })),
                    },
                    0,
                )],
                source: Some(vector_par(Vec::new(), false)),
                remainder: None,
                free_count: 0,
                pattern_modifiers: vec![],
            }],
            vector_par(Vec::new(), false),
            false,
            false,
            0,
            Vec::new(),
            false,
            Vec::new(),
            false,
        ),
        Vec::new(),
        BTreeMap::new(),
        Vec::new(),
        Vec::new(),
        false,
    );

    assert!(assert_spatial_match(target, pattern, Some(BTreeMap::new())).is_ok());
}

#[test]
fn matching_plus_plus_should_work() {
    let target: Par = vector_par(Vec::new(), false).with_exprs(vec![Expr {
        expr_instance: Some(EPlusPlusBody(EPlusPlus {
            p1: Some(new_gstring_par("abc".to_string(), Vec::new(), false)),
            p2: Some(new_gstring_par("def".to_string(), Vec::new(), false)),
        })),
    }]);

    let pattern = vector_par(Vec::new(), true).with_exprs(vec![Expr {
        expr_instance: Some(EPlusPlusBody(EPlusPlus {
            p1: Some(new_freevar_par(0, Vec::new())),
            p2: Some(new_freevar_par(1, Vec::new())),
        })),
    }]);

    let expected_captures = BTreeMap::from([
        (0, new_gstring_par("abc".to_string(), Vec::new(), false)),
        (1, new_gstring_par("def".to_string(), Vec::new(), false)),
    ]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_percent_percent_should_work() {
    let map: Par = vector_par(Vec::new(), false).with_exprs(vec![Expr {
        expr_instance: Some(EMapBody(ParMapTypeMapper::par_map_to_emap(
            ParMap::create_from_vec(vec![(
                new_gstring_par("name".to_string(), Vec::new(), false),
                new_gstring_par("a".to_string(), Vec::new(), false),
            )]),
        ))),
    }]);

    let target = vector_par(Vec::new(), false).with_exprs(vec![Expr {
        expr_instance: Some(EPercentPercentBody(EPercentPercent {
            p1: Some(new_gstring_par("${name}".to_string(), Vec::new(), false)),
            p2: Some(map.clone()),
        })),
    }]);

    let pattern = vector_par(Vec::new(), true).with_exprs(vec![Expr {
        expr_instance: Some(EPercentPercentBody(EPercentPercent {
            p1: Some(new_freevar_par(0, Vec::new())),
            p2: Some(new_freevar_par(1, Vec::new())),
        })),
    }]);

    let expected_captures = BTreeMap::from([
        (0, new_gstring_par("${name}".to_string(), Vec::new(), false)),
        (1, map),
    ]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_minus_minus_should_work() {
    let lhs_set = ESetBody(ParSetTypeMapper::par_set_to_eset(ParSet::new(
        vec![
            new_gint_par(1, Vec::new(), false),
            new_gint_par(2, Vec::new(), false),
            new_gint_par(3, Vec::new(), false),
        ],
        false,
        Vec::new(),
        None,
    )));
    let lhs_set_par: Par = vector_par(Vec::new(), false).with_exprs(vec![Expr {
        expr_instance: Some(ESetBody(ParSetTypeMapper::par_set_to_eset(ParSet::new(
            vec![
                new_gint_par(1, Vec::new(), false),
                new_gint_par(2, Vec::new(), false),
                new_gint_par(3, Vec::new(), false),
            ],
            false,
            Vec::new(),
            None,
        )))),
    }]);

    let rhs_set = ESetBody(ParSetTypeMapper::par_set_to_eset(ParSet::new(
        vec![
            new_gint_par(1, Vec::new(), false),
            new_gint_par(2, Vec::new(), false),
        ],
        false,
        Vec::new(),
        None,
    )));
    let rhs_set_par: Par = vector_par(Vec::new(), false).with_exprs(vec![Expr {
        expr_instance: Some(ESetBody(ParSetTypeMapper::par_set_to_eset(ParSet::new(
            vec![
                new_gint_par(1, Vec::new(), false),
                new_gint_par(2, Vec::new(), false),
            ],
            false,
            Vec::new(),
            None,
        )))),
    }]);

    let target = vector_par(Vec::new(), false).with_exprs(vec![Expr {
        expr_instance: Some(EMinusMinusBody(EMinusMinus {
            p1: Some(vector_par(Vec::new(), false).with_exprs(vec![Expr {
                expr_instance: Some(lhs_set),
            }])),
            p2: Some(vector_par(Vec::new(), false).with_exprs(vec![Expr {
                expr_instance: Some(rhs_set),
            }])),
        })),
    }]);

    let pattern = vector_par(Vec::new(), true).with_exprs(vec![Expr {
        expr_instance: Some(EMinusMinusBody(EMinusMinus {
            p1: Some(new_freevar_par(0, Vec::new())),
            p2: Some(new_freevar_par(1, Vec::new())),
        })),
    }]);

    let expected_captures = BTreeMap::from([(0, lhs_set_par), (1, rhs_set_par)]);
    assert!(assert_spatial_match(target, pattern, Some(expected_captures)).is_ok());
}

#[test]
fn matching_bool_should_work() {
    let success_target = vector_par(Vec::new(), false).with_exprs(vec![Expr {
        expr_instance: Some(GBool(false)),
    }]);

    let fail_target = new_gstring_par("Fail".to_string(), Vec::new(), false);
    let pattern = vector_par(Vec::new(), true).with_connectives(vec![Connective {
        connective_instance: Some(ConnBool(true)),
    }]);

    assert!(assert_spatial_match(success_target, pattern.clone(), Some(BTreeMap::new())).is_ok());
    assert!(assert_spatial_match(fail_target, pattern, None).is_ok());
}

#[test]
fn matching_int_should_work() {
    let success_target = new_gint_par(7, Vec::new(), false);
    let fail_target = new_gstring_par("Fail".to_string(), Vec::new(), false);
    let pattern = vector_par(Vec::new(), true).with_connectives(vec![Connective {
        connective_instance: Some(ConnInt(true)),
    }]);

    assert!(assert_spatial_match(success_target, pattern.clone(), Some(BTreeMap::new())).is_ok());
    assert!(assert_spatial_match(fail_target, pattern, None).is_ok());
}

#[test]
fn matching_string_should_work() {
    let success_target = new_gstring_par("Match me!".to_string(), Vec::new(), false);
    let fail_target = new_gint_par(42, Vec::new(), false);
    let pattern = vector_par(Vec::new(), true).with_connectives(vec![Connective {
        connective_instance: Some(ConnString(true)),
    }]);

    assert!(assert_spatial_match(success_target, pattern.clone(), Some(BTreeMap::new())).is_ok());
    assert!(assert_spatial_match(fail_target, pattern, None).is_ok());
}

#[test]
fn matching_uri_should_work() {
    let success_target = vector_par(Vec::new(), false).with_exprs(vec![Expr {
        expr_instance: Some(GUri("rho:io:stdout".to_string())),
    }]);

    let fail_target = new_gstring_par("Fail".to_string(), Vec::new(), false);
    let pattern = vector_par(Vec::new(), true).with_connectives(vec![Connective {
        connective_instance: Some(ConnUri(true)),
    }]);

    assert!(assert_spatial_match(success_target, pattern.clone(), Some(BTreeMap::new())).is_ok());
    assert!(assert_spatial_match(fail_target, pattern, None).is_ok());
}

#[test]
fn matching_byte_array_should_work() {
    let success_target = vector_par(Vec::new(), false).with_exprs(vec![Expr {
        expr_instance: Some(GByteArray(vec![74, 75])),
    }]);

    let fail_target = new_gstring_par("Fail".to_string(), Vec::new(), false);
    let pattern = vector_par(Vec::new(), true).with_connectives(vec![Connective {
        connective_instance: Some(ConnByteArray(true)),
    }]);

    assert!(assert_spatial_match(success_target, pattern.clone(), Some(BTreeMap::new())).is_ok());
    assert!(assert_spatial_match(fail_target, pattern, None).is_ok());
}
