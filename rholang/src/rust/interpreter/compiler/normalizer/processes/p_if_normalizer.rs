use super::exports::*;
use crate::rust::interpreter::compiler::normalize::{
    normalize_match_proc, normalize_ann_proc, ProcVisitInputs, ProcVisitOutputs,
};
use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::{Match, MatchCase, Par};
use models::rust::utils::{new_gbool_par, union};
use std::collections::HashMap;

// New AST imports
use rholang_parser::ast::AnnProc;

pub fn normalize_p_if(
    value_proc: &Proc,
    true_body_proc: &Proc,
    false_body_proc: &Proc,
    mut input: ProcVisitInputs,
    env: &HashMap<String, Par>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let target_result =
        normalize_match_proc(&value_proc, ProcVisitInputs { ..input.clone() }, env)?;

    let true_case_body = normalize_match_proc(
        &true_body_proc,
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: target_result.free_map.clone(),
            source_span: input.source_span,
        },
        env,
    )?;

    let false_case_body = normalize_match_proc(
        &false_body_proc,
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: true_case_body.free_map.clone(),
            source_span: input.source_span,
        },
        env,
    )?;

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

    Ok(ProcVisitOutputs {
        par: updated_par,
        free_map: false_case_body.free_map,
    })
}

/// Parallel version of normalize_p_if for new AST IfThenElse
pub fn normalize_p_if_new_ast<'ast>(
    condition: &'ast AnnProc<'ast>,
    if_true: &'ast AnnProc<'ast>,
    if_false: Option<&'ast AnnProc<'ast>>,
    mut input: ProcVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let target_result =
        normalize_ann_proc(&condition, ProcVisitInputs { ..input.clone() }, env, parser)?;

    let true_case_body = normalize_ann_proc(
        &if_true,
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: target_result.free_map.clone(),
            source_span: input.source_span,
        },
        env,
        parser,
    )?;

    let false_case_body = match if_false {
        Some(false_proc) => normalize_ann_proc(
            false_proc,
            ProcVisitInputs {
                par: Par::default(),
                bound_map_chain: input.bound_map_chain.clone(),
                free_map: true_case_body.free_map.clone(),
                source_span: input.source_span,
            },
            env,
            parser,
        )?,
        None => {
            // Create nil AnnProc using parser's const_nil
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
                ProcVisitInputs {
                    par: Par::default(),
                    bound_map_chain: input.bound_map_chain.clone(),
                    free_map: true_case_body.free_map.clone(),
                    source_span: input.source_span,
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

    Ok(ProcVisitOutputs {
        par: updated_par,
        free_map: false_case_body.free_map,
    })
}

// See rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use models::{
        create_bit_vector,
        rhoapi::{expr::ExprInstance, EEq, Expr, Match, MatchCase, Par, Send},
        rust::utils::{
            new_boundvar_par, new_gbool_par, new_gint_expr, new_gint_par, new_new_par, new_send_par,
        },
    };

    use crate::rust::interpreter::{
        compiler::{
            normalize::{normalize_match_proc, ProcVisitInputs},
            rholang_ast::{Decls, Name, NameDecl, Proc, ProcList, SendType},
        },
        test_utils::utils::proc_visit_inputs_and_env,
    };

    #[test]
    fn p_if_else_should_desugar_to_match_with_true_false_cases() {
        // if (true) { @Nil!(47) }

        let p_if = Proc::IfElse {
            condition: Box::new(Proc::BoolLiteral {
                value: true,
                line_num: 0,
                col_num: 0,
            }),
            if_true: Box::new(Proc::Send {
                name: Name::new_name_quote_nil(),
                send_type: SendType::new_single(),
                inputs: ProcList::new(vec![Proc::new_proc_int(47)]),
                line_num: 0,
                col_num: 0,
            }),
            alternative: None,
            line_num: 0,
            col_num: 0,
        };

        let result = normalize_match_proc(&p_if, ProcVisitInputs::new(), &HashMap::new());
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
        assert_eq!(result.unwrap().free_map, ProcVisitInputs::new().free_map);
    }

    #[test]
    fn p_if_else_should_not_mix_par_from_the_input_with_normalized_one() {
        let p_if = Proc::IfElse {
            condition: Box::new(Proc::BoolLiteral {
                value: true,
                line_num: 0,
                col_num: 0,
            }),
            if_true: Box::new(Proc::new_proc_int(10)),
            alternative: None,
            line_num: 0,
            col_num: 0,
        };

        let (mut inputs, env) = proc_visit_inputs_and_env();
        inputs.par = Par::default().with_exprs(vec![new_gint_expr(7)]);

        let result = normalize_match_proc(&p_if, inputs.clone(), &env);
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
    fn p_if_else_should_handle_a_more_complicated_if_statement_with_an_else_clause() {
        // if (47 == 47) { new x in { x!(47) } } else { new y in { y!(47) } }
        let condition = Proc::Eq {
            left: Box::new(Proc::new_proc_int(47)),
            right: Box::new(Proc::new_proc_int(47)),
            line_num: 0,
            col_num: 0,
        };

        let p_new_if = Proc::New {
            decls: Decls {
                decls: vec![NameDecl::new("x", None)],
                line_num: 0,
                col_num: 0,
            },
            proc: Box::new(Proc::Send {
                name: Name::new_name_var("x"),
                send_type: SendType::new_single(),
                inputs: ProcList::new(vec![Proc::new_proc_int(47)]),
                line_num: 0,
                col_num: 0,
            }),
            line_num: 0,
            col_num: 0,
        };

        let p_new_else = Proc::New {
            decls: Decls {
                decls: vec![NameDecl::new("y", None)],
                line_num: 0,
                col_num: 0,
            },
            proc: Box::new(Proc::Send {
                name: Name::new_name_var("y"),
                send_type: SendType::new_single(),
                inputs: ProcList::new(vec![Proc::new_proc_int(47)]),
                line_num: 0,
                col_num: 0,
            }),
            line_num: 0,
            col_num: 0,
        };

        let p_if = Proc::IfElse {
            condition: Box::new(condition),
            if_true: Box::new(p_new_if),
            alternative: Some(Box::new(p_new_else)),
            line_num: 0,
            col_num: 0,
        };

        let (inputs, env) = proc_visit_inputs_and_env();
        let result = normalize_match_proc(&p_if, inputs.clone(), &env);
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

    // ============================================================================
    // NEW AST PARALLEL TESTS - EXACT MAPPING TO ORIGINAL TESTS
    // ============================================================================

    #[test]
    fn new_ast_p_if_else_should_desugar_to_match_with_true_false_cases() {
        // Maps to original: p_if_else_should_desugar_to_match_with_true_false_cases
        // Test: if (true) { @Nil!(47) }
        use rholang_parser::ast::{AnnProc, AnnName, Name as NewName, Proc as NewProc, SendType as NewSendType};
        use rholang_parser::{SourcePos, SourceSpan};
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;

        let (inputs, env) = proc_visit_inputs_and_env();
        
        // Create the complete IfThenElse AST node and normalize through normalize_ann_proc 
        // to test the complete pipeline (same as how the original test works)
        let if_then_else = AnnProc {
            proc: Box::leak(Box::new(NewProc::IfThenElse {
                condition: AnnProc {
                    proc: Box::leak(Box::new(NewProc::BoolLiteral(true))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                if_true: AnnProc {
                    proc: Box::leak(Box::new(NewProc::Send {
                        channel: AnnName {
                            name: NewName::Quote(Box::leak(Box::new(NewProc::Nil))),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                        send_type: NewSendType::Single,
                        inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                            proc: Box::leak(Box::new(NewProc::LongLiteral(47))),
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
                if_false: None, // Test the None case (same as original)
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
        // Maps to original: p_if_else_should_not_mix_par_from_the_input_with_normalized_one
        use rholang_parser::ast::{AnnProc, Proc as NewProc};
        use rholang_parser::{SourcePos, SourceSpan};
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;

        let (mut inputs, env) = proc_visit_inputs_and_env();
        inputs.par = Par::default().with_exprs(vec![new_gint_expr(7)]);

        // Create if (true) { 10 } - simple condition and if_true, no else
        let condition = AnnProc {
            proc: Box::leak(Box::new(NewProc::BoolLiteral(true))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let if_true = AnnProc {
            proc: Box::leak(Box::new(NewProc::LongLiteral(10))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        // Note: We no longer create nil_proc separately since we're testing through IfThenElse AST node
        // Create the complete IfThenElse AST node and normalize through normalize_ann_proc 
        // to test the complete pipeline (same as how the original test works)
        let if_then_else = AnnProc {
            proc: Box::leak(Box::new(NewProc::IfThenElse {
                condition: condition.clone(),
                if_true: if_true.clone(),
                if_false: None, // Test the None case 
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
        // Maps to original: p_if_else_should_handle_a_more_complicated_if_statement_with_an_else_clause
        // Test: if (47 == 47) { new x in { x!(47) } } else { new y in { y!(47) } }
        use rholang_parser::ast::{AnnProc, AnnName, BinaryExpOp, Id, Name as NewName, NameDecl as NewNameDecl, Proc as NewProc, SendType as NewSendType, Var as NewVar};
        use rholang_parser::{SourcePos, SourceSpan};
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;

        let (inputs, env) = proc_visit_inputs_and_env();

        // Create the complete IfThenElse AST node and normalize through normalize_ann_proc 
        // to test the complete pipeline (same as how the original test works)
        let if_then_else = AnnProc {
            proc: Box::leak(Box::new(NewProc::IfThenElse {
                condition: AnnProc {
                    proc: Box::leak(Box::new(NewProc::BinaryExp {
                        op: BinaryExpOp::Eq,
                        left: AnnProc {
                            proc: Box::leak(Box::new(NewProc::LongLiteral(47))),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                        right: AnnProc {
                            proc: Box::leak(Box::new(NewProc::LongLiteral(47))),
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
                    proc: Box::leak(Box::new(NewProc::New {
                        decls: vec![NewNameDecl {
                            id: Id {
                                name: "x",
                                pos: SourcePos { line: 0, col: 0 },
                            },
                            uri: None,
                        }],
                        proc: AnnProc {
                            proc: Box::leak(Box::new(NewProc::Send {
                                channel: AnnName {
                                    name: NewName::ProcVar(NewVar::Id(Id {
                                        name: "x",
                                        pos: SourcePos { line: 0, col: 0 },
                                    })),
                                    span: SourceSpan {
                                        start: SourcePos { line: 0, col: 0 },
                                        end: SourcePos { line: 0, col: 0 },
                                    },
                                },
                                send_type: NewSendType::Single,
                                inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                                    proc: Box::leak(Box::new(NewProc::LongLiteral(47))),
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
                    proc: Box::leak(Box::new(NewProc::New {
                        decls: vec![NewNameDecl {
                            id: Id {
                                name: "y",
                                pos: SourcePos { line: 0, col: 0 },
                            },
                            uri: None,
                        }],
                        proc: AnnProc {
                            proc: Box::leak(Box::new(NewProc::Send {
                                channel: AnnName {
                                    name: NewName::ProcVar(NewVar::Id(Id {
                                        name: "y",
                                        pos: SourcePos { line: 0, col: 0 },
                                    })),
                                    span: SourceSpan {
                                        start: SourcePos { line: 0, col: 0 },
                                        end: SourcePos { line: 0, col: 0 },
                                    },
                                },
                                send_type: NewSendType::Single,
                                inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                                    proc: Box::leak(Box::new(NewProc::LongLiteral(47))),
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
