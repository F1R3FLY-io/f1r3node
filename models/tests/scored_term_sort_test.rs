// See models/src/test/scala/coop/rchain/models/rholang/SortTest.scala - ScoredTermSpec

use models::rhoapi::{
    EList, EMethod, New, Par,
};
use models::rust::par_map::ParMap;
use models::rust::par_map_type_mapper::ParMapTypeMapper;
use models::rust::rholang::sorter::bundle_sort_matcher::BundleSortMatcher;
use models::rust::rholang::sorter::connective_sort_matcher::ConnectiveSortMatcher;
use models::rust::rholang::sorter::match_sort_matcher::MatchSortMatcher;
use models::rust::rholang::sorter::new_sort_matcher::NewSortMatcher;
use models::rust::rholang::sorter::par_sort_matcher::ParSortMatcher;
use models::rust::rholang::sorter::receive_sort_matcher::ReceiveSortMatcher;
use models::rust::rholang::sorter::send_sort_matcher::SendSortMatcher;
use models::rust::rholang::sorter::var_sort_matcher::VarSortMatcher;
use models::rust::test_utils::test_utils::{
    generate_bundle, generate_connective, generate_expr, generate_match,
    generate_new, generate_par, generate_receive, generate_send, generate_var,
};
use models::{
    rhoapi::{expr::ExprInstance, Expr, Var},
    rust::{
        par_set::ParSet,
        par_set_type_mapper::ParSetTypeMapper,
        rholang::sorter::{
            expr_sort_matcher::ExprSortMatcher,
            score_tree::{ScoreAtom, ScoredTerm, Tree},
            sortable::Sortable,
        },
        sorted_par_hash_set::SortedParHashSet,
    },
};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;

fn global_proptest_config() -> ProptestConfig {
    ProptestConfig {
        failure_persistence: Some(Box::new(FileFailurePersistence::WithSource("regressions"))),
        ..ProptestConfig::default()
    }
}

#[test]
fn scored_term_should_sort_so_that_shorter_nodes_come_first() {
    let mut unsorted_terms = vec![
        ScoredTerm {
            term: "foo".to_string(),
            score: Tree::<ScoreAtom>::create_node_from_i64s(vec![1, 2, 2, 3]),
        },
        ScoredTerm {
            term: "bar".to_string(),
            score: Tree::<ScoreAtom>::create_node_from_i64s(vec![1, 2, 2]),
        },
    ];

    let sorted_terms = vec![
        ScoredTerm {
            term: "bar".to_string(),
            score: Tree::<ScoreAtom>::create_node_from_i64s(vec![1, 2, 2]),
        },
        ScoredTerm {
            term: "foo".to_string(),
            score: Tree::<ScoreAtom>::create_node_from_i64s(vec![1, 2, 2, 3]),
        },
    ];

    ScoredTerm::sort_vec(&mut unsorted_terms);
    assert_eq!(unsorted_terms, sorted_terms);
}

#[test]
fn scored_term_should_sort_so_that_smaller_leaves_stay_first() {
    let mut unsorted_terms = vec![
        ScoredTerm {
            term: "foo".to_string(),
            score: Tree::<ScoreAtom>::create_leaf_from_i64(1),
        },
        ScoredTerm {
            term: "bar".to_string(),
            score: Tree::<ScoreAtom>::create_leaf_from_i64(2),
        },
    ];

    let sorted_terms = vec![
        ScoredTerm {
            term: "foo".to_string(),
            score: Tree::<ScoreAtom>::create_leaf_from_i64(1),
        },
        ScoredTerm {
            term: "bar".to_string(),
            score: Tree::<ScoreAtom>::create_leaf_from_i64(2),
        },
    ];

    ScoredTerm::sort_vec(&mut unsorted_terms);
    assert_eq!(unsorted_terms, sorted_terms);
}

#[test]
fn scored_term_should_sort_so_that_smaller_leaves_are_put_first() {
    let mut unsorted_terms = vec![
        ScoredTerm {
            term: "foo".to_string(),
            score: Tree::<ScoreAtom>::create_leaf_from_i64(2),
        },
        ScoredTerm {
            term: "bar".to_string(),
            score: Tree::<ScoreAtom>::create_leaf_from_i64(1),
        },
    ];

    let sorted_terms = vec![
        ScoredTerm {
            term: "bar".to_string(),
            score: Tree::<ScoreAtom>::create_leaf_from_i64(1),
        },
        ScoredTerm {
            term: "foo".to_string(),
            score: Tree::<ScoreAtom>::create_leaf_from_i64(2),
        },
    ];

    ScoredTerm::sort_vec(&mut unsorted_terms);
    assert_eq!(unsorted_terms, sorted_terms);
}

#[test]
fn scored_term_should_sort_so_that_smaller_nodes_are_put_first() {
    let mut unsorted_terms = vec![
        ScoredTerm {
            term: "foo".to_string(),
            score: Tree::Node(vec![
                Tree::<ScoreAtom>::create_node_from_i64s(vec![1, 2]),
                Tree::<ScoreAtom>::create_node_from_i64s(vec![2, 2]),
            ]),
        },
        ScoredTerm {
            term: "bar".to_string(),
            score: Tree::Node(vec![
                Tree::<ScoreAtom>::create_node_from_i64s(vec![1, 1]),
                Tree::<ScoreAtom>::create_node_from_i64s(vec![2, 2]),
            ]),
        },
    ];

    let sorted_terms = vec![
        ScoredTerm {
            term: "bar".to_string(),
            score: Tree::Node(vec![
                Tree::<ScoreAtom>::create_node_from_i64s(vec![1, 1]),
                Tree::<ScoreAtom>::create_node_from_i64s(vec![2, 2]),
            ]),
        },
        ScoredTerm {
            term: "foo".to_string(),
            score: Tree::Node(vec![
                Tree::<ScoreAtom>::create_node_from_i64s(vec![1, 2]),
                Tree::<ScoreAtom>::create_node_from_i64s(vec![2, 2]),
            ]),
        },
    ];

    ScoredTerm::sort_vec(&mut unsorted_terms);
    assert_eq!(unsorted_terms, sorted_terms);
}

#[test]
fn scored_term_should_sort_so_that_whenever_scores_differ_then_result_terms_have_to_differ_and_the_other_way_around(
) {
    fn check<T, F>(generator: impl Strategy<Value = Vec<T>>, sort_fn: F)
    where
        T: Clone + PartialEq + std::fmt::Debug,
        F: Fn(&T) -> T,
    {
        proptest!(global_proptest_config(), |(values in generator)| {
            for x in &values {
                for y in &values {
                    let x_sorted = sort_fn(x);
                    let y_sorted = sort_fn(y);

                    assert_eq!(
                        x_sorted == y_sorted,
                        x == y,
                        "Sorted terms and scores do not align for {:?} and {:?}",
                        x,
                        y
                    );
                }
            }
        });
    }

    check(prop::collection::vec(generate_bundle(2), 5), |x| {
        BundleSortMatcher::sort_match(x).term
    });

    check(prop::collection::vec(generate_connective(2), 5), |x| {
        ConnectiveSortMatcher::sort_match(x).term
    });

    check(prop::collection::vec(generate_expr(2), 5), |x| {
        ExprSortMatcher::sort_match(x).term
    });

    check(prop::collection::vec(generate_match(2), 5), |x| {
        MatchSortMatcher::sort_match(x).term
    });

    check(prop::collection::vec(generate_new(2), 5), |x| {
        NewSortMatcher::sort_match(x).term
    });

    check(prop::collection::vec(generate_par(2), 5), |x| {
        ParSortMatcher::sort_match(x).term
    });

    check(prop::collection::vec(generate_receive(2), 5), |x| {
        ReceiveSortMatcher::sort_match(x).term
    });

    check(prop::collection::vec(generate_send(2), 5), |x| {
        SendSortMatcher::sort_match(x).term
    });

    check(prop::collection::vec(generate_var(2), 5), |x| {
        VarSortMatcher::sort_match(x).term
    });
}

#[test]
fn scored_term_should_sort_so_that_whenever_scores_or_result_terms_differ_then_the_initial_terms_differ_and_the_other_way_around(
) {
    fn check<T, F>(generator: impl Strategy<Value = Vec<T>>, sort_fn: F)
    where
        T: Clone + PartialEq + std::fmt::Debug,
        F: Fn(&T) -> T,
    {
        proptest!(global_proptest_config(), |(values in generator)| {
            for x in &values {
                for y in &values {
                    let x_sorted = sort_fn(x);
                    let y_sorted = sort_fn(y);

                    if x_sorted != y_sorted {
                        assert_ne!(x, y,"Initial terms should differ when sorted terms or scores differ:\n\
                            x: {:?}\n\
                            y: {:?}\n\
                            x_sorted: {:?}\n\
                            y_sorted: {:?}", x, y, x_sorted, y_sorted);
                    } else {
                        assert_eq!(x, y,"Initial terms should be equal when sorted terms and scores are equal:\n\
                            x: {:?}\n\
                            y: {:?}\n\
                            x_sorted: {:?}\n\
                            y_sorted: {:?}", x, y, x_sorted, y_sorted);
                    }
                }
            }
        });
    }

    check(prop::collection::vec(generate_bundle(2), 5), |x| {
        BundleSortMatcher::sort_match(x).term
    });

    check(prop::collection::vec(generate_connective(2), 5), |x| {
        ConnectiveSortMatcher::sort_match(x).term
    });

    check(prop::collection::vec(generate_expr(2), 5), |x| {
        ExprSortMatcher::sort_match(x).term
    });

    check(prop::collection::vec(generate_match(2), 5), |x| {
        MatchSortMatcher::sort_match(x).term
    });

    check(prop::collection::vec(generate_new(2), 5), |x| {
        NewSortMatcher::sort_match(x).term
    });

    check(prop::collection::vec(generate_par(2), 5), |x| {
        ParSortMatcher::sort_match(x).term
    });

    check(prop::collection::vec(generate_receive(2), 5), |x| {
        ReceiveSortMatcher::sort_match(x).term
    });

    check(prop::collection::vec(generate_send(2), 5), |x| {
        SendSortMatcher::sort_match(x).term
    });

    check(prop::collection::vec(generate_var(2), 5), |x| {
        VarSortMatcher::sort_match(x).term
    });
}

#[test]
fn scored_term_should_sort_so_that_unequal_terms_have_unequal_scores_and_the_other_way_around() {
    fn check_score_equality<T, F>(generator: impl Strategy<Value = Vec<T>>, sort_fn: F)
    where
        T: Clone + PartialEq + std::fmt::Debug,
        F: Fn(&T) -> T,
    {
        proptest!(global_proptest_config(), |(values in generator)| {
            for x in &values {
                for y in &values {
                    if x != y {
                        assert_ne!(sort_fn(x), sort_fn(y),"Unequal terms should have unequal scores:\n\
                            x: {:?}\n\
                            y: {:?}\n\
                            x_sorted: {:?}\n\
                            y_sorted: {:?}", x, y, sort_fn(x), sort_fn(y));
                    } else {
                        assert_eq!(sort_fn(x), sort_fn(y),"Equal terms should have equal scores:\n\
                            x: {:?}\n\
                            y: {:?}\n\
                            x_sorted: {:?}\n\
                            y_sorted: {:?}", x, y, sort_fn(x), sort_fn(y));
                    }
                }
            }
        });
    }

    check_score_equality(prop::collection::vec(generate_bundle(2), 5), |x| {
        BundleSortMatcher::sort_match(x).term
    });

    check_score_equality(prop::collection::vec(generate_connective(2), 5), |x| {
        ConnectiveSortMatcher::sort_match(x).term
    });

    check_score_equality(prop::collection::vec(generate_expr(2), 5), |x| {
        ExprSortMatcher::sort_match(x).term
    });

    check_score_equality(prop::collection::vec(generate_match(2), 5), |x| {
        MatchSortMatcher::sort_match(x).term
    });

    check_score_equality(prop::collection::vec(generate_new(2), 5), |x| {
        NewSortMatcher::sort_match(x).term
    });

    check_score_equality(prop::collection::vec(generate_par(2), 5), |x| {
        ParSortMatcher::sort_match(x).term
    });

    check_score_equality(prop::collection::vec(generate_receive(2), 5), |x| {
        ReceiveSortMatcher::sort_match(x).term
    });

    check_score_equality(prop::collection::vec(generate_send(2), 5), |x| {
        SendSortMatcher::sort_match(x).term
    });

    check_score_equality(prop::collection::vec(generate_var(2), 5), |x| {
        VarSortMatcher::sort_match(x).term
    });
}

#[test]
fn scored_term_should_score_in_different_byte_string_should_be_unequal() {
    let a = Expr {
        expr_instance: Some(ExprInstance::GByteArray(hex::decode("80").unwrap())),
    };

    let b = Expr {
        expr_instance: Some(ExprInstance::GByteArray(hex::decode("D9").unwrap())),
    };

    assert_ne!(
        ExprSortMatcher::sort_match(&a).score,
        ExprSortMatcher::sort_match(&b).score,
        "Scores for different ByteStrings should be unequal"
    );
}

#[test]
fn scored_term_should_sort_so_that_unequal_new_have_unequal_scores() {
    // based on new_sort_matcher.rs -> p field on New should be Some
    let new_1 = New {
        p: Some(Par::default()),
        bind_count: 1,
        injections: {
            let mut injections = std::collections::BTreeMap::new();
            injections.insert(
                "key".to_string(),
                Par {
                    bundles: vec![],
                    sends: vec![],
                    receives: vec![],
                    ..Default::default()
                },
            );
            injections
        },
        ..Default::default()
    };

    let new_2 = New {
        p: Some(Par::default()),
        bind_count: 1,
        ..Default::default()
    };
    assert_ne!(new_1, new_2);

    assert_ne!(
        NewSortMatcher::sort_match(&new_1).score,
        NewSortMatcher::sort_match(&new_2).score
    );
}

#[test]
fn scored_term_should_sort_so_that_unequal_emethod_have_unequal_scores() {
    // based on expr_sort_matcher.rs -> target field on EMethod should be Some
    let method_1 = Expr {
        expr_instance: Some(ExprInstance::EMethodBody(EMethod {
            target: Some(Par::default()),
            connective_used: true,
            ..Default::default()
        })),
    };

    let method_2 = Expr {
        expr_instance: Some(ExprInstance::EMethodBody(EMethod {
            target: Some(Par::default()),
            connective_used: false,
            ..Default::default()
        })),
    };

    assert_ne!(
        ExprSortMatcher::sort_match(&method_1).score,
        ExprSortMatcher::sort_match(&method_2).score
    );
}

#[test]
fn scored_term_should_sort_so_that_unequal_par_map_have_unequal_scores() {
    let map_1 = Expr {
        expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
            ParMap::new(vec![], true, vec![], None),
        ))),
    };

    let map_2 = Expr {
        expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
            ParMap::new(vec![], false, vec![], None),
        ))),
    };

    assert_ne!(
        ExprSortMatcher::sort_match(&map_1).score,
        ExprSortMatcher::sort_match(&map_2).score
    );

    let map_3 = Expr {
        expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
            ParMap::new(vec![], true, vec![], None),
        ))),
    };

    let map_4 = Expr {
        expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
            ParMap::new(vec![], true, vec![], Some(Var::default())),
        ))),
    };

    assert_ne!(
        ExprSortMatcher::sort_match(&map_3).score,
        ExprSortMatcher::sort_match(&map_4).score
    );
}

#[test]
fn scored_term_should_sort_so_that_unequal_par_set_have_unequal_scores() {
    let set_1 = Expr {
        expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
            ParSet {
                ps: SortedParHashSet::create_from_empty(),
                connective_used: true,
                locally_free: vec![],
                remainder: None,
            },
        ))),
    };

    let set_2 = Expr {
        expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
            ParSet {
                ps: SortedParHashSet::create_from_empty(),
                connective_used: false,
                locally_free: vec![],
                remainder: None,
            },
        ))),
    };

    assert!(ExprSortMatcher::sort_match(&set_1).score != ExprSortMatcher::sort_match(&set_2).score);

    let set_3 = Expr {
        expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
            ParSet {
                ps: SortedParHashSet::create_from_empty(),
                connective_used: true,
                locally_free: vec![],
                remainder: None,
            },
        ))),
    };

    let set_4 = Expr {
        expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
            ParSet {
                ps: SortedParHashSet::create_from_empty(),
                connective_used: true,
                locally_free: vec![],
                remainder: Some(Var::default()),
            },
        ))),
    };

    assert_ne!(
        ExprSortMatcher::sort_match(&set_3).score,
        ExprSortMatcher::sort_match(&set_4).score
    );
}

#[test]
fn scored_term_should_sort_so_that_unequal_list_have_unequal_scores() {
    let list_1 = Expr {
        expr_instance: Some(ExprInstance::EListBody(EList {
            ps: vec![],
            locally_free: vec![],
            connective_used: true,
            remainder: None,
        })),
    };

    let list_2 = Expr {
        expr_instance: Some(ExprInstance::EListBody(EList {
            ps: vec![],
            locally_free: vec![],
            connective_used: false,
            remainder: None,
        })),
    };

    assert_ne!(
        ExprSortMatcher::sort_match(&list_1).score,
        ExprSortMatcher::sort_match(&list_2).score
    );

    let list_3 = Expr {
        expr_instance: Some(ExprInstance::EListBody(EList {
            ps: vec![],
            locally_free: vec![],
            connective_used: true,
            remainder: None,
        })),
    };

    let list_4 = Expr {
        expr_instance: Some(ExprInstance::EListBody(EList {
            ps: vec![],
            locally_free: vec![],
            connective_used: true,
            remainder: Some(Var::default()),
        })),
    };

    assert_ne!(
        ExprSortMatcher::sort_match(&list_3).score,
        ExprSortMatcher::sort_match(&list_4).score
    );
}
