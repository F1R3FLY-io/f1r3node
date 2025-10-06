use crate::rust::interpreter::compiler::exports::{FreeMap, ProcVisitInputs, ProcVisitOutputs};
use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::util::filter_and_adjust_bitset;
use models::rhoapi::{Match, MatchCase, Par};
use models::rust::utils::union;
use std::collections::HashMap;

use rholang_parser::ast::{AnnProc, Case};

pub fn normalize_p_match<'ast>(
    expression: &'ast AnnProc<'ast>,
    cases: &'ast [Case<'ast>],
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    //We don't have any CaseImpl inside Rust AST, so we should work with simple Case struct
    fn lift_case<'ast>(
        case: &'ast Case<'ast>,
    ) -> Result<(&'ast AnnProc<'ast>, &'ast AnnProc<'ast>), InterpreterError> {
        Ok((&case.pattern, &case.proc))
    }

    let target_result = normalize_ann_proc(
        expression,
        ProcVisitInputs {
            par: Par::default(),
            ..input.clone()
        },
        env,
        parser,
    )?;

    let mut init_acc = (vec![], target_result.free_map.clone(), Vec::new(), false);

    for case in cases {
        let (pattern, case_body) = lift_case(case)?;
        let pattern_result = normalize_ann_proc(
            pattern,
            ProcVisitInputs {
                par: Par::default(),
                bound_map_chain: input.bound_map_chain.push(),
                free_map: FreeMap::default(),
            },
            env,
            parser,
        )?;

        let case_env = input
            .bound_map_chain
            .absorb_free_span(&pattern_result.free_map);
        let bound_count = pattern_result.free_map.count_no_wildcards();

        let case_body_result = normalize_ann_proc(
            case_body,
            ProcVisitInputs {
                par: Par::default(),
                bound_map_chain: case_env.clone(),
                free_map: init_acc.1.clone(),
            },
            env,
            parser,
        )?;

        init_acc.0.insert(
            0,
            MatchCase {
                pattern: Some(pattern_result.par.clone()),
                source: Some(case_body_result.par.clone()),
                free_count: bound_count as i32,
            },
        );
        init_acc.1 = case_body_result.free_map;
        init_acc.2 = union(
            union(init_acc.2.clone(), pattern_result.par.locally_free.clone()),
            filter_and_adjust_bitset(case_body_result.par.locally_free.clone(), bound_count),
        );
        init_acc.3 = init_acc.3 || case_body_result.par.connective_used;
    }

    let result_match = Match {
        target: Some(target_result.par.clone()),
        cases: init_acc.0.into_iter().rev().collect(),
        locally_free: union(init_acc.2, target_result.par.locally_free.clone()),
        connective_used: init_acc.3 || target_result.par.connective_used.clone(),
    };

    Ok(ProcVisitOutputs {
        par: input.par.clone().prepend_match(result_match.clone()),
        free_map: init_acc.1,
    })
}

// See rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use models::{
        create_bit_vector,
        rhoapi::{Match, MatchCase, Par, Receive, ReceiveBind},
        rust::utils::{
            new_boundvar_par, new_elist_par, new_freevar_expr, new_freevar_par, new_gint_par,
            new_send, new_wildcard_par,
        },
    };

    use crate::rust::interpreter::{
        compiler::{exports::ProcVisitInputs, normalize::VarSort},
        errors::InterpreterError,
        test_utils::utils::proc_visit_inputs_and_env,
        util::prepend_expr,
    };

    #[test]
    fn p_match_should_fail_if_a_free_variable_is_used_twice_in_the_target() {
        // match 47 { case (y | y) => Nil }
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{AnnProc, Case, Id, Proc, Var};
        use rholang_parser::{SourcePos, SourceSpan};

        let p_match = AnnProc {
            proc: Box::leak(Box::new(Proc::Match {
                expression: AnnProc {
                    proc: Box::leak(Box::new(Proc::LongLiteral(47))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                cases: vec![Case {
                    pattern: AnnProc {
                        proc: Box::leak(Box::new(Proc::Par {
                            left: AnnProc {
                                proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                                    name: "y",
                                    pos: SourcePos { line: 0, col: 0 },
                                })))),
                                span: SourceSpan {
                                    start: SourcePos { line: 0, col: 0 },
                                    end: SourcePos { line: 0, col: 0 },
                                },
                            },
                            right: AnnProc {
                                proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                                    name: "y",
                                    pos: SourcePos { line: 0, col: 0 },
                                })))),
                                span: SourceSpan {
                                    start: SourcePos { line: 0, col: 0 },
                                    end: SourcePos { line: 0, col: 0 },
                                },
                            },
                        })),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    proc: AnnProc {
                        proc: Box::leak(Box::new(Proc::Nil)),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                }],
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(
            &p_match,
            proc_visit_inputs_and_env().0,
            &proc_visit_inputs_and_env().1,
            &parser,
        );
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InterpreterError::UnexpectedReuseOfProcContextFreeSpan {
                var_name,
                first_use: _,
                second_use: _
            }) if var_name == "y"
        ));
    }

    #[test]
    fn p_match_should_have_a_free_count_of_1_if_the_case_contains_a_wildcard_and_a_free_variable() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{AnnProc, Case, Collection, Id, Proc, Var};
        use rholang_parser::{SourcePos, SourceSpan};

        let (mut inputs, env) = proc_visit_inputs_and_env();
        inputs.bound_map_chain = inputs.bound_map_chain.put_pos((
            "x".to_string(),
            VarSort::ProcSort,
            SourcePos { line: 0, col: 0 },
        ));

        // Create match x { case [y, _] => Nil ; case _ => Nil } using new AST
        let p_match = AnnProc {
            proc: Box::leak(Box::new(Proc::Match {
                expression: AnnProc {
                    proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                        name: "x",
                        pos: SourcePos { line: 0, col: 0 },
                    })))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                cases: vec![
                    Case {
                        pattern: AnnProc {
                            proc: Box::leak(Box::new(Proc::Collection(Collection::List {
                                elements: vec![
                                    AnnProc {
                                        proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                                            name: "y",
                                            pos: SourcePos { line: 0, col: 0 },
                                        })))),
                                        span: SourceSpan {
                                            start: SourcePos { line: 0, col: 0 },
                                            end: SourcePos { line: 0, col: 0 },
                                        },
                                    },
                                    AnnProc {
                                        proc: Box::leak(Box::new(Proc::ProcVar(Var::Wildcard))),
                                        span: SourceSpan {
                                            start: SourcePos { line: 0, col: 0 },
                                            end: SourcePos { line: 0, col: 0 },
                                        },
                                    },
                                ],
                                remainder: None,
                            }))),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                        proc: AnnProc {
                            proc: Box::leak(Box::new(Proc::Nil)),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                    },
                    Case {
                        pattern: AnnProc {
                            proc: Box::leak(Box::new(Proc::ProcVar(Var::Wildcard))),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                        proc: AnnProc {
                            proc: Box::leak(Box::new(Proc::Nil)),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                    },
                ],
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&p_match, inputs, &env, &parser);
        assert!(result.is_ok());

        let expected_result = Par::default().prepend_match(Match {
            target: Some(new_boundvar_par(0, create_bit_vector(&vec![0]), false)),
            cases: vec![
                MatchCase {
                    pattern: Some(new_elist_par(
                        vec![
                            new_freevar_par(0, Vec::new()),
                            new_wildcard_par(Vec::new(), true),
                        ],
                        Vec::new(),
                        true,
                        None,
                        Vec::new(),
                        true,
                    )),
                    source: Some(Par::default()),
                    free_count: 1,
                },
                MatchCase {
                    pattern: Some(new_wildcard_par(Vec::new(), true)),
                    source: Some(Par::default()),
                    free_count: 0,
                },
            ],
            locally_free: create_bit_vector(&vec![0]),
            connective_used: false,
        });

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.unwrap().par.matches[0].cases[0].free_count, 1);
    }

    #[test]
    fn p_match_should_handle_a_match_inside_a_for_comprehension() {
        // for (@x <- @Nil) { match x { case 42 => Nil ; case y => Nil } | @Nil!(47)
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{
            AnnName, AnnProc, Bind, Case, Id, Name, Names, Proc, SendType, Source, Var,
        };
        use rholang_parser::{SourcePos, SourceSpan};

        // Create the complete Par structure: for (@x <- @Nil) { match x { case 42 => Nil ; case y => Nil } } | @Nil!(47)
        let p_par = AnnProc {
            proc: Box::leak(Box::new(Proc::Par {
                left: AnnProc {
                    proc: Box::leak(Box::new(Proc::ForComprehension {
                        receipts: smallvec::SmallVec::from_vec(vec![smallvec::SmallVec::from_vec(
                            vec![Bind::Linear {
                                lhs: Names {
                                    names: smallvec::SmallVec::from_vec(vec![AnnName {
                                        name: Name::Quote(Box::leak(Box::new(Proc::ProcVar(
                                            Var::Id(Id {
                                                name: "x",
                                                pos: SourcePos { line: 0, col: 0 },
                                            }),
                                        )))),
                                        span: SourceSpan {
                                            start: SourcePos { line: 0, col: 0 },
                                            end: SourcePos { line: 0, col: 0 },
                                        },
                                    }]),
                                    remainder: None,
                                },
                                rhs: Source::Simple {
                                    name: AnnName {
                                        name: Name::Quote(Box::leak(Box::new(Proc::Nil))),
                                        span: SourceSpan {
                                            start: SourcePos { line: 0, col: 0 },
                                            end: SourcePos { line: 0, col: 0 },
                                        },
                                    },
                                },
                            }],
                        )]),
                        proc: AnnProc {
                            proc: Box::leak(Box::new(Proc::Match {
                                expression: AnnProc {
                                    proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                                        name: "x",
                                        pos: SourcePos { line: 0, col: 0 },
                                    })))),
                                    span: SourceSpan {
                                        start: SourcePos { line: 0, col: 0 },
                                        end: SourcePos { line: 0, col: 0 },
                                    },
                                },
                                cases: vec![
                                    Case {
                                        pattern: AnnProc {
                                            proc: Box::leak(Box::new(Proc::LongLiteral(42))),
                                            span: SourceSpan {
                                                start: SourcePos { line: 0, col: 0 },
                                                end: SourcePos { line: 0, col: 0 },
                                            },
                                        },
                                        proc: AnnProc {
                                            proc: Box::leak(Box::new(Proc::Nil)),
                                            span: SourceSpan {
                                                start: SourcePos { line: 0, col: 0 },
                                                end: SourcePos { line: 0, col: 0 },
                                            },
                                        },
                                    },
                                    Case {
                                        pattern: AnnProc {
                                            proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                                                name: "y",
                                                pos: SourcePos { line: 0, col: 0 },
                                            })))),
                                            span: SourceSpan {
                                                start: SourcePos { line: 0, col: 0 },
                                                end: SourcePos { line: 0, col: 0 },
                                            },
                                        },
                                        proc: AnnProc {
                                            proc: Box::leak(Box::new(Proc::Nil)),
                                            span: SourceSpan {
                                                start: SourcePos { line: 0, col: 0 },
                                                end: SourcePos { line: 0, col: 0 },
                                            },
                                        },
                                    },
                                ],
                            })),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                    })),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                right: AnnProc {
                    proc: Box::leak(Box::new(Proc::Send {
                        channel: AnnName {
                            name: Name::Quote(Box::leak(Box::new(Proc::Nil))),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                        send_type: SendType::Single,
                        inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                            proc: Box::leak(Box::new(Proc::LongLiteral(47))),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        }]),
                    })),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&p_par, ProcVisitInputs::new(), &HashMap::new(), &parser);
        assert!(result.is_ok());

        let expected_result = Par::default()
            .prepend_send(new_send(
                Par::default(),
                vec![new_gint_par(47, Vec::new(), false)],
                false,
                Vec::new(),
                false,
            ))
            .prepend_receive(Receive {
                binds: vec![ReceiveBind {
                    patterns: vec![new_freevar_par(0, Vec::new())],
                    source: Some(Par::default()),
                    remainder: None,
                    free_count: 1,
                }],
                body: Some(Par::default().prepend_match(Match {
                    target: Some(new_boundvar_par(0, create_bit_vector(&vec![0]), false)),
                    cases: vec![
                        MatchCase {
                            pattern: Some(new_gint_par(42, Vec::new(), false)),
                            source: Some(Par::default()),
                            free_count: 0,
                        },
                        MatchCase {
                            pattern: Some(new_freevar_par(0, Vec::new())),
                            source: Some(Par::default()),
                            free_count: 1,
                        },
                    ],
                    locally_free: create_bit_vector(&vec![0]),
                    connective_used: false,
                })),
                persistent: false,
                peek: false,
                bind_count: 1,
                locally_free: Vec::new(),
                connective_used: false,
            });

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.unwrap().free_map, ProcVisitInputs::new().free_map);
    }

    #[test]
    fn p_match_should_handle_a_match_inside_a_for_pattern() {
        // for (@{match {x | y} { 47 => Nil }} <- @Nil) { Nil }
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{
            AnnName, AnnProc, Bind, Case, Id, Name, Names, Proc, Source, Var,
        };
        use rholang_parser::{SourcePos, SourceSpan};

        // Create for (@{match {x | y} { 47 => Nil }} <- @Nil) { Nil } using new AST
        let input = AnnProc {
            proc: Box::leak(Box::new(Proc::ForComprehension {
                receipts: smallvec::SmallVec::from_vec(vec![smallvec::SmallVec::from_vec(vec![
                    Bind::Linear {
                        lhs: Names {
                            names: smallvec::SmallVec::from_vec(vec![AnnName {
                                name: Name::Quote(Box::leak(Box::new(Proc::Match {
                                    expression: AnnProc {
                                        proc: Box::leak(Box::new(Proc::Par {
                                            left: AnnProc {
                                                proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(
                                                    Id {
                                                        name: "x",
                                                        pos: SourcePos { line: 0, col: 0 },
                                                    },
                                                )))),
                                                span: SourceSpan {
                                                    start: SourcePos { line: 0, col: 0 },
                                                    end: SourcePos { line: 0, col: 0 },
                                                },
                                            },
                                            right: AnnProc {
                                                proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(
                                                    Id {
                                                        name: "y",
                                                        pos: SourcePos { line: 0, col: 0 },
                                                    },
                                                )))),
                                                span: SourceSpan {
                                                    start: SourcePos { line: 0, col: 0 },
                                                    end: SourcePos { line: 0, col: 0 },
                                                },
                                            },
                                        })),
                                        span: SourceSpan {
                                            start: SourcePos { line: 0, col: 0 },
                                            end: SourcePos { line: 0, col: 0 },
                                        },
                                    },
                                    cases: vec![Case {
                                        pattern: AnnProc {
                                            proc: Box::leak(Box::new(Proc::LongLiteral(47))),
                                            span: SourceSpan {
                                                start: SourcePos { line: 0, col: 0 },
                                                end: SourcePos { line: 0, col: 0 },
                                            },
                                        },
                                        proc: AnnProc {
                                            proc: Box::leak(Box::new(Proc::Nil)),
                                            span: SourceSpan {
                                                start: SourcePos { line: 0, col: 0 },
                                                end: SourcePos { line: 0, col: 0 },
                                            },
                                        },
                                    }],
                                }))),
                                span: SourceSpan {
                                    start: SourcePos { line: 0, col: 0 },
                                    end: SourcePos { line: 0, col: 0 },
                                },
                            }]),
                            remainder: None,
                        },
                        rhs: Source::Simple {
                            name: AnnName {
                                name: Name::Quote(Box::leak(Box::new(Proc::Nil))),
                                span: SourceSpan {
                                    start: SourcePos { line: 0, col: 0 },
                                    end: SourcePos { line: 0, col: 0 },
                                },
                            },
                        },
                    },
                ])]),
                proc: AnnProc {
                    proc: Box::leak(Box::new(Proc::Nil)),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let (inputs, env) = proc_visit_inputs_and_env();

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&input, inputs.clone(), &env, &parser);
        assert!(result.is_ok());

        let expected_result = Par::default().prepend_receive(Receive {
            binds: vec![ReceiveBind {
                patterns: vec![{
                    let mut par = Par::default().with_matches(vec![Match {
                        target: Some(prepend_expr(
                            new_freevar_par(1, Vec::new()),
                            new_freevar_expr(0),
                            0,
                        )),
                        cases: vec![MatchCase {
                            pattern: Some(new_gint_par(47, Vec::new(), false)),
                            source: Some(Par::default()),
                            free_count: 0,
                        }],
                        locally_free: Vec::new(),
                        connective_used: true,
                    }]);
                    par.connective_used = true;
                    par
                }],
                source: Some(Par::default()),
                remainder: None,
                free_count: 2,
            }],
            body: Some(Par::default()),
            persistent: false,
            peek: false,
            bind_count: 2,
            locally_free: Vec::new(),
            connective_used: false,
        });

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }
}
