use crate::rust::interpreter::compiler::exports::{ProcVisitInputsSpan, ProcVisitOutputsSpan};
use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::{Match, MatchCase, Par};
use models::rust::utils::{new_gbool_par, union};
use std::collections::HashMap;

use rholang_parser::ast::AnnProc;

pub fn normalize_p_if_new_ast<'ast>(
    condition: &'ast AnnProc<'ast>,
    if_true: &'ast AnnProc<'ast>,
    if_false: Option<&'ast AnnProc<'ast>>,
    mut input: ProcVisitInputsSpan,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputsSpan, InterpreterError> {
    let target_result = normalize_ann_proc(
        &condition,
        ProcVisitInputsSpan { ..input.clone() },
        env,
        parser,
    )?;

    let true_case_body = normalize_ann_proc(
        &if_true,
        ProcVisitInputsSpan {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: target_result.free_map.clone(),
        },
        env,
        parser,
    )?;

    let false_case_body = match if_false {
        Some(false_proc) => normalize_ann_proc(
            false_proc,
            ProcVisitInputsSpan {
                par: Par::default(),
                bound_map_chain: input.bound_map_chain.clone(),
                free_map: true_case_body.free_map.clone(),
            },
            env,
            parser,
        )?,
        None => {
            let nil_proc_ref = parser.ast_builder().const_nil();
            let nil_ann_proc = rholang_parser::ast::AnnProc {
                proc: nil_proc_ref,
                span: rholang_parser::SourceSpan {
                    start: rholang_parser::SourcePos { line: 0, col: 0 },
                    end: rholang_parser::SourcePos { line: 0, col: 0 },
                },
            };
            normalize_ann_proc(
                &nil_ann_proc,
                ProcVisitInputsSpan {
                    par: Par::default(),
                    bound_map_chain: input.bound_map_chain.clone(),
                    free_map: true_case_body.free_map.clone(),
                },
                env,
                parser,
            )?
        }
    };

    // Construct the desugared if as a Match
    let desugared_if = Match {
        target: Some(target_result.par.clone()),
        cases: vec![
            MatchCase {
                pattern: Some(new_gbool_par(true, vec![], false)),
                source: Some(true_case_body.par.clone()),
                free_count: 0,
            },
            MatchCase {
                pattern: Some(new_gbool_par(false, vec![], false)),
                source: Some(false_case_body.par.clone()),
                free_count: 0,
            },
        ],
        locally_free: union(
            union(
                target_result.par.locally_free.clone(),
                true_case_body.par.locally_free.clone(),
            ),
            false_case_body.par.locally_free.clone(),
        ),
        connective_used: target_result.par.connective_used
            || true_case_body.par.connective_used
            || false_case_body.par.connective_used,
    };

    // Update the input par by prepending the desugared if statement
    let updated_par = input.par.prepend_match(desugared_if);

    Ok(ProcVisitOutputsSpan {
        par: updated_par,
        free_map: false_case_body.free_map,
    })
}

// See rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use models::{
        create_bit_vector,
        rhoapi::{expr::ExprInstance, EEq, Expr, Match, MatchCase, Par, Send},
        rust::utils::{
            new_boundvar_par, new_gbool_par, new_gint_expr, new_gint_par, new_new_par, new_send_par,
        },
    };

    use crate::rust::interpreter::test_utils::utils::proc_visit_inputs_and_env_span;

    #[test]
    fn new_ast_p_if_else_should_desugar_to_match_with_true_false_cases() {
        // if (true) { @Nil!(47) }
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{AnnName, AnnProc, Name, Proc, SendType};
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env_span();

        let if_then_else = AnnProc {
            proc: Box::leak(Box::new(Proc::IfThenElse {
                condition: AnnProc {
                    proc: Box::leak(Box::new(Proc::BoolLiteral(true))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                if_true: AnnProc {
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
                if_false: None,
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&if_then_else, inputs.clone(), &env, &parser);
        assert!(result.is_ok());

        let expected_result = Par::default().prepend_match(Match {
            target: Some(new_gbool_par(true, Vec::new(), false)),
            cases: vec![
                MatchCase {
                    pattern: Some(new_gbool_par(true, Vec::new(), false)),
                    source: Some(Par::default().with_sends(vec![Send {
                        chan: Some(Par::default()),
                        data: vec![new_gint_par(47, Vec::new(), false)],
                        persistent: false,
                        locally_free: Vec::new(),
                        connective_used: false,
                    }])),
                    free_count: 0,
                },
                MatchCase {
                    pattern: Some(new_gbool_par(false, Vec::new(), false)),
                    source: Some(Par::default()),
                    free_count: 0,
                },
            ],
            locally_free: Vec::new(),
            connective_used: false,
        });

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }

    #[test]
    fn new_ast_p_if_else_should_not_mix_par_from_the_input_with_normalized_one() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{AnnProc, Proc};
        use rholang_parser::{SourcePos, SourceSpan};

        let (mut inputs, env) = proc_visit_inputs_and_env_span();
        inputs.par = Par::default().with_exprs(vec![new_gint_expr(7)]);

        // if (true) { 10 }
        let condition = AnnProc {
            proc: Box::leak(Box::new(Proc::BoolLiteral(true))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let if_true = AnnProc {
            proc: Box::leak(Box::new(Proc::LongLiteral(10))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let if_then_else = AnnProc {
            proc: Box::leak(Box::new(Proc::IfThenElse {
                condition: condition.clone(),
                if_true: if_true.clone(),
                if_false: None,
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };
        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&if_then_else, inputs.clone(), &env, &parser);
        assert!(result.is_ok());

        let expected_result = Par::default()
            .with_matches(vec![Match {
                target: Some(new_gbool_par(true, Vec::new(), false)),
                cases: vec![
                    MatchCase {
                        pattern: Some(new_gbool_par(true, Vec::new(), false)),
                        source: Some(new_gint_par(10, Vec::new(), false)),
                        free_count: 0,
                    },
                    MatchCase {
                        pattern: Some(new_gbool_par(false, Vec::new(), false)),
                        source: Some(Par::default()),
                        free_count: 0,
                    },
                ],
                locally_free: Vec::new(),
                connective_used: false,
            }])
            .with_exprs(vec![new_gint_expr(7)]);

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }

    #[test]
    fn new_ast_p_if_else_should_handle_a_more_complicated_if_statement_with_an_else_clause() {
        // if (47 == 47) { new x in { x!(47) } } else { new y in { y!(47) } }
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{
            AnnName, AnnProc, BinaryExpOp, Id, Name, NameDecl, Proc, SendType, Var,
        };
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env_span();

        let if_then_else = AnnProc {
            proc: Box::leak(Box::new(Proc::IfThenElse {
                condition: AnnProc {
                    proc: Box::leak(Box::new(Proc::BinaryExp {
                        op: BinaryExpOp::Eq,
                        left: AnnProc {
                            proc: Box::leak(Box::new(Proc::LongLiteral(47))),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                        right: AnnProc {
                            proc: Box::leak(Box::new(Proc::LongLiteral(47))),
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
                if_true: AnnProc {
                    proc: Box::leak(Box::new(Proc::New {
                        decls: vec![NameDecl {
                            id: Id {
                                name: "x",
                                pos: SourcePos { line: 0, col: 0 },
                            },
                            uri: None,
                        }],
                        proc: AnnProc {
                            proc: Box::leak(Box::new(Proc::Send {
                                channel: AnnName {
                                    name: Name::ProcVar(Var::Id(Id {
                                        name: "x",
                                        pos: SourcePos { line: 0, col: 0 },
                                    })),
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
                },
                if_false: Some(AnnProc {
                    proc: Box::leak(Box::new(Proc::New {
                        decls: vec![NameDecl {
                            id: Id {
                                name: "y",
                                pos: SourcePos { line: 0, col: 0 },
                            },
                            uri: None,
                        }],
                        proc: AnnProc {
                            proc: Box::leak(Box::new(Proc::Send {
                                channel: AnnName {
                                    name: Name::ProcVar(Var::Id(Id {
                                        name: "y",
                                        pos: SourcePos { line: 0, col: 0 },
                                    })),
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
                }),
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&if_then_else, inputs.clone(), &env, &parser);
        assert!(result.is_ok());

        let expected_result = Par::default().with_matches(vec![Match {
            target: Some(Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::EEqBody(EEq {
                    p1: Some(new_gint_par(47, Vec::new(), false)),
                    p2: Some(new_gint_par(47, Vec::new(), false)),
                })),
            }])),
            cases: vec![
                MatchCase {
                    pattern: Some(new_gbool_par(true, Vec::new(), false)),
                    source: Some(new_new_par(
                        1,
                        new_send_par(
                            new_boundvar_par(0, create_bit_vector(&vec![0]), false),
                            vec![new_gint_par(47, Vec::new(), false)],
                            false,
                            create_bit_vector(&vec![0]),
                            false,
                            create_bit_vector(&vec![0]),
                            false,
                        ),
                        vec![],
                        BTreeMap::new(),
                        Vec::new(),
                        Vec::new(),
                        false,
                    )),
                    free_count: 0,
                },
                MatchCase {
                    pattern: Some(new_gbool_par(false, Vec::new(), false)),
                    source: Some(new_new_par(
                        1,
                        new_send_par(
                            new_boundvar_par(0, create_bit_vector(&vec![0]), false),
                            vec![new_gint_par(47, Vec::new(), false)],
                            false,
                            create_bit_vector(&vec![0]),
                            false,
                            create_bit_vector(&vec![0]),
                            false,
                        ),
                        vec![],
                        BTreeMap::new(),
                        Vec::new(),
                        Vec::new(),
                        false,
                    )),
                    free_count: 0,
                },
            ],
            locally_free: Vec::new(),
            connective_used: false,
        }]);

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }
}
