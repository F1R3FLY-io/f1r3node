use super::exports::{FreeMap, InterpreterError, Proc, ProcVisitInputs, ProcVisitOutputs};
use crate::rust::interpreter::{
    compiler::{
        exports::{FreeMapSpan, ProcVisitInputsSpan, ProcVisitOutputsSpan},
        normalize::{normalize_ann_proc, normalize_match_proc},
    },
    util::prepend_expr,
};
use models::rhoapi::{expr, EMatches, Expr, Par};
use std::collections::HashMap;

// New AST imports
use rholang_parser::ast::AnnProc;

pub fn normalize_p_matches(
    left_proc: &Proc,
    right_proc: &Proc,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let left_result = normalize_match_proc(
        left_proc,
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: input.free_map.clone(),
        },
        env,
    )?;

    let right_result = normalize_match_proc(
        right_proc,
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone().push(),
            free_map: FreeMap::default(),
        },
        env,
    )?;

    let new_expr = Expr {
        expr_instance: Some(expr::ExprInstance::EMatchesBody(EMatches {
            target: Some(left_result.par.clone()),
            pattern: Some(right_result.par.clone()),
        })),
    };

    let prepend_par = prepend_expr(input.par, new_expr, input.bound_map_chain.depth() as i32);

    Ok(ProcVisitOutputs {
        par: prepend_par,
        free_map: left_result.free_map,
    })
}

/// Parallel version of normalize_p_matches for new AST BinaryExp with Matches op
pub fn normalize_p_matches_new_ast<'ast>(
    left: &'ast AnnProc<'ast>,
    right: &'ast AnnProc<'ast>,
    input: ProcVisitInputsSpan,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputsSpan, InterpreterError> {
    let left_result = normalize_ann_proc(
        left,
        ProcVisitInputsSpan {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: input.free_map.clone(),
        },
        env,
        parser,
    )?;

    let right_result = normalize_ann_proc(
        right,
        ProcVisitInputsSpan {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone().push(),
            free_map: FreeMapSpan::default(),
        },
        env,
        parser,
    )?;

    let new_expr = Expr {
        expr_instance: Some(expr::ExprInstance::EMatchesBody(EMatches {
            target: Some(left_result.par.clone()),
            pattern: Some(right_result.par.clone()),
        })),
    };

    let prepend_par = prepend_expr(input.par, new_expr, input.bound_map_chain.depth() as i32);

    Ok(ProcVisitOutputsSpan {
        par: prepend_par,
        free_map: left_result.free_map,
    })
}

//rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use crate::rust::interpreter::compiler::normalize::normalize_match_proc;
    use crate::rust::interpreter::compiler::rholang_ast::{Negation, Proc};
    use crate::rust::interpreter::test_utils::utils::{
        proc_visit_inputs_and_env, proc_visit_inputs_and_env_span,
    };
    use models::rhoapi::connective::ConnectiveInstance::ConnNotBody;

    use crate::rust::interpreter::util::prepend_expr;
    use models::rhoapi::{expr, Connective, EMatches, Expr, Par};
    use models::rust::utils::{new_gint_par, new_wildcard_par};
    use pretty_assertions::assert_eq;

    //1 matches _
    #[test]
    fn p_matches_should_normalize_one_matches_wildcard() {
        let (inputs, env) = proc_visit_inputs_and_env();
        let proc = Proc::Matches {
            left: Box::new(Proc::new_proc_int(1)),
            right: Box::new(Proc::new_proc_wildcard()),
            line_num: 0,
            col_num: 0,
        };

        let result = normalize_match_proc(&proc, inputs.clone(), &env);

        let expected_par = prepend_expr(
            inputs.par.clone(),
            Expr {
                expr_instance: Some(expr::ExprInstance::EMatchesBody(EMatches {
                    target: Some(new_gint_par(1, Vec::new(), false)),
                    pattern: Some(new_wildcard_par(Vec::new(), true)),
                })),
            },
            0,
        );

        assert_eq!(result.clone().unwrap().par, expected_par);
        assert_eq!(result.unwrap().par.connective_used, false);
    }

    //1 matches 2
    #[test]
    fn p_matches_should_normalize_correctly_one_matches_two() {
        let (inputs, env) = proc_visit_inputs_and_env();
        let proc = Proc::Matches {
            left: Box::new(Proc::new_proc_int(1)),
            right: Box::new(Proc::new_proc_int(2)),
            line_num: 0,
            col_num: 0,
        };

        let result = normalize_match_proc(&proc, inputs.clone(), &env);

        let expected_par = prepend_expr(
            inputs.par.clone(),
            Expr {
                expr_instance: Some(expr::ExprInstance::EMatchesBody(EMatches {
                    target: Some(new_gint_par(1, Vec::new(), false)),
                    pattern: Some(new_gint_par(2, Vec::new(), false)),
                })),
            },
            0,
        );

        assert_eq!(result.clone().unwrap().par, expected_par);
        assert_eq!(result.unwrap().par.connective_used, false);
    }

    //1 matches ~1
    #[test]
    fn p_matches_should_normalize_one_matches_tilda_with_connective_used_false() {
        let (inputs, env) = proc_visit_inputs_and_env();
        let proc = Proc::Matches {
            left: Box::new(Proc::new_proc_int(1)),
            right: Box::new(Negation::new_negation_int(1)),
            line_num: 0,
            col_num: 0,
        };

        let result = normalize_match_proc(&proc, inputs.clone(), &env);

        let expected_par = prepend_expr(
            inputs.par.clone(),
            Expr {
                expr_instance: Some(expr::ExprInstance::EMatchesBody(EMatches {
                    target: Some(new_gint_par(1, Vec::new(), false)),
                    pattern: Some(Par {
                        connectives: vec![Connective {
                            connective_instance: Some(ConnNotBody(new_gint_par(
                                1,
                                Vec::new(),
                                false,
                            ))),
                        }],
                        connective_used: true,
                        ..Par::default().clone()
                    }),
                })),
            },
            0,
        );

        assert_eq!(result.clone().unwrap().par, expected_par);
        assert_eq!(result.unwrap().par.connective_used, false);
    }

    //~1 matches 1
    #[test]
    fn p_matches_should_normalize_tilda_one_matches_one_with_connective_used_true() {
        let (inputs, env) = proc_visit_inputs_and_env();
        let proc = Proc::Matches {
            left: Box::new(Negation::new_negation_int(1)),
            right: Box::new(Proc::new_proc_int(1)),
            line_num: 0,
            col_num: 0,
        };

        let result = normalize_match_proc(&proc, inputs.clone(), &env);

        let expected_par = prepend_expr(
            inputs.par.clone(),
            Expr {
                expr_instance: Some(expr::ExprInstance::EMatchesBody(EMatches {
                    target: Some(Par {
                        connectives: vec![Connective {
                            connective_instance: Some(ConnNotBody(new_gint_par(
                                1,
                                Vec::new(),
                                false,
                            ))),
                        }],
                        connective_used: true,
                        ..Par::default().clone()
                    }),
                    pattern: Some(new_gint_par(1, Vec::new(), false)),
                })),
            },
            0,
        );

        assert_eq!(result.clone().unwrap().par, expected_par);
        assert_eq!(result.unwrap().par.connective_used, true)
    }

    // ============================================================================
    // NEW AST PARALLEL TESTS - EXACT MAPPING TO ORIGINAL TESTS
    // ============================================================================

    #[test]
    fn new_ast_p_matches_should_normalize_one_matches_wildcard() {
        // Maps to original: p_matches_should_normalize_one_matches_wildcard
        // Test: 1 matches _
        use super::normalize_p_matches_new_ast;
        use rholang_parser::ast::{AnnProc, Proc as NewProc, Var as NewVar};
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env_span();

        // Create "1 matches _" - LongLiteral matches Wildcard
        let left_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::LongLiteral(1))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let right_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::ProcVar(NewVar::Wildcard))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result =
            normalize_p_matches_new_ast(&left_proc, &right_proc, inputs.clone(), &env, &parser);

        let expected_par = prepend_expr(
            inputs.par.clone(),
            Expr {
                expr_instance: Some(expr::ExprInstance::EMatchesBody(EMatches {
                    target: Some(new_gint_par(1, Vec::new(), false)),
                    pattern: Some(new_wildcard_par(Vec::new(), true)),
                })),
            },
            0,
        );

        assert_eq!(result.clone().unwrap().par, expected_par);
        assert_eq!(result.unwrap().par.connective_used, false);
    }

    #[test]
    fn new_ast_p_matches_should_normalize_correctly_one_matches_two() {
        // Maps to original: p_matches_should_normalize_correctly_one_matches_two
        // Test: 1 matches 2
        use super::normalize_p_matches_new_ast;
        use rholang_parser::ast::{AnnProc, Proc as NewProc};
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env_span();

        // Create "1 matches 2" - LongLiteral matches LongLiteral
        let left_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::LongLiteral(1))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let right_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::LongLiteral(2))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result =
            normalize_p_matches_new_ast(&left_proc, &right_proc, inputs.clone(), &env, &parser);

        let expected_par = prepend_expr(
            inputs.par.clone(),
            Expr {
                expr_instance: Some(expr::ExprInstance::EMatchesBody(EMatches {
                    target: Some(new_gint_par(1, Vec::new(), false)),
                    pattern: Some(new_gint_par(2, Vec::new(), false)),
                })),
            },
            0,
        );

        assert_eq!(result.clone().unwrap().par, expected_par);
        assert_eq!(result.unwrap().par.connective_used, false);
    }

    #[test]
    fn new_ast_p_matches_should_normalize_one_matches_tilda_with_connective_used_false() {
        // Maps to original: p_matches_should_normalize_one_matches_tilda_with_connective_used_false
        // Test: 1 matches ~1
        use super::normalize_p_matches_new_ast;
        use rholang_parser::ast::{AnnProc, Proc as NewProc, UnaryExpOp};
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env_span();

        // Create "1 matches ~1" - LongLiteral matches (Negation LongLiteral)
        let left_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::LongLiteral(1))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let right_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::UnaryExp {
                op: UnaryExpOp::Negation,
                arg: Box::leak(Box::new(NewProc::LongLiteral(1))),
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result =
            normalize_p_matches_new_ast(&left_proc, &right_proc, inputs.clone(), &env, &parser);

        let expected_par = prepend_expr(
            inputs.par.clone(),
            Expr {
                expr_instance: Some(expr::ExprInstance::EMatchesBody(EMatches {
                    target: Some(new_gint_par(1, Vec::new(), false)),
                    pattern: Some(Par {
                        connectives: vec![Connective {
                            connective_instance: Some(ConnNotBody(new_gint_par(
                                1,
                                Vec::new(),
                                false,
                            ))),
                        }],
                        connective_used: true,
                        ..Par::default().clone()
                    }),
                })),
            },
            0,
        );

        assert_eq!(result.clone().unwrap().par, expected_par);
        assert_eq!(result.unwrap().par.connective_used, false);
    }

    #[test]
    fn new_ast_p_matches_should_normalize_tilda_one_matches_one_with_connective_used_true() {
        // Maps to original: p_matches_should_normalize_tilda_one_matches_one_with_connective_used_true
        // Test: ~1 matches 1
        use super::normalize_p_matches_new_ast;
        use rholang_parser::ast::{AnnProc, Proc as NewProc, UnaryExpOp};
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env_span();

        // Create "~1 matches 1" - (Negation LongLiteral) matches LongLiteral
        let left_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::UnaryExp {
                op: UnaryExpOp::Negation,
                arg: Box::leak(Box::new(NewProc::LongLiteral(1))),
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let right_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::LongLiteral(1))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result =
            normalize_p_matches_new_ast(&left_proc, &right_proc, inputs.clone(), &env, &parser);

        let expected_par = prepend_expr(
            inputs.par.clone(),
            Expr {
                expr_instance: Some(expr::ExprInstance::EMatchesBody(EMatches {
                    target: Some(Par {
                        connectives: vec![Connective {
                            connective_instance: Some(ConnNotBody(new_gint_par(
                                1,
                                Vec::new(),
                                false,
                            ))),
                        }],
                        connective_used: true,
                        ..Par::default().clone()
                    }),
                    pattern: Some(new_gint_par(1, Vec::new(), false)),
                })),
            },
            0,
        );

        assert_eq!(result.clone().unwrap().par, expected_par);
        assert_eq!(result.unwrap().par.connective_used, true)
    }
}
