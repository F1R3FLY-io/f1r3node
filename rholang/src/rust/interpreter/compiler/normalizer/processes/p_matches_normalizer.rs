use super::exports::InterpreterError;
use crate::rust::interpreter::{
    compiler::{
        exports::{FreeMap, ProcVisitInputs, ProcVisitOutputs},
        normalize::normalize_ann_proc,
    },
    util::prepend_expr,
};
use models::rhoapi::{expr, EMatches, Expr, Par};
use std::collections::HashMap;

use rholang_parser::ast::AnnProc;

pub fn normalize_p_matches<'ast>(
    left: &'ast AnnProc<'ast>,
    right: &'ast AnnProc<'ast>,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let left_result = normalize_ann_proc(
        left,
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: input.free_map.clone(),
        },
        env,
        parser,
    )?;

    let right_result = normalize_ann_proc(
        right,
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone().push(),
            free_map: FreeMap::default(),
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

    Ok(ProcVisitOutputs {
        par: prepend_par,
        free_map: left_result.free_map,
    })
}

//rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use crate::rust::interpreter::test_utils::utils::proc_visit_inputs_and_env;
    use models::rhoapi::connective::ConnectiveInstance::ConnNotBody;

    use crate::rust::interpreter::util::prepend_expr;
    use models::rhoapi::{expr, Connective, EMatches, Expr, Par};
    use models::rust::utils::{new_gint_par, new_wildcard_par};
    use pretty_assertions::assert_eq;

    #[test]
    fn p_matches_should_normalize_one_matches_wildcard() {
        // Test: 1 matches _
        use super::normalize_p_matches;
        use rholang_parser::ast::{AnnProc, Proc, Var};
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env();

        // Create "1 matches _" - LongLiteral matches Wildcard
        let left_proc = AnnProc {
            proc: Box::leak(Box::new(Proc::LongLiteral(1))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let right_proc = AnnProc {
            proc: Box::leak(Box::new(Proc::ProcVar(Var::Wildcard))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_p_matches(&left_proc, &right_proc, inputs.clone(), &env, &parser);

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
    fn p_matches_should_normalize_correctly_one_matches_two() {
        // Test: 1 matches 2
        use super::normalize_p_matches;
        use rholang_parser::ast::{AnnProc, Proc};
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env();

        // Create "1 matches 2" - LongLiteral matches LongLiteral
        let left_proc = AnnProc {
            proc: Box::leak(Box::new(Proc::LongLiteral(1))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let right_proc = AnnProc {
            proc: Box::leak(Box::new(Proc::LongLiteral(2))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_p_matches(&left_proc, &right_proc, inputs.clone(), &env, &parser);

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
    fn p_matches_should_normalize_one_matches_tilda_with_connective_used_false() {
        // Test: 1 matches ~1
        use super::normalize_p_matches;
        use rholang_parser::ast::{AnnProc, Proc, UnaryExpOp};
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env();

        // Create "1 matches ~1" - LongLiteral matches (Negation LongLiteral)
        let left_proc = AnnProc {
            proc: Box::leak(Box::new(Proc::LongLiteral(1))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let right_proc = AnnProc {
            proc: Box::leak(Box::new(Proc::UnaryExp {
                op: UnaryExpOp::Negation,
                arg: AnnProc {
                    proc: Box::leak(Box::new(Proc::LongLiteral(1))),
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
        let result = normalize_p_matches(&left_proc, &right_proc, inputs.clone(), &env, &parser);

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
    fn p_matches_should_normalize_tilda_one_matches_one_with_connective_used_true() {
        // Test: ~1 matches 1
        use super::normalize_p_matches;
        use rholang_parser::ast::{AnnProc, Proc, UnaryExpOp};
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env();

        // Create "~1 matches 1" - (Negation LongLiteral) matches LongLiteral
        let left_proc = AnnProc {
            proc: Box::leak(Box::new(Proc::UnaryExp {
                op: UnaryExpOp::Negation,
                arg: AnnProc {
                    proc: Box::leak(Box::new(Proc::LongLiteral(1))),
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

        let right_proc = AnnProc {
            proc: Box::leak(Box::new(Proc::LongLiteral(1))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_p_matches(&left_proc, &right_proc, inputs.clone(), &env, &parser);

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
