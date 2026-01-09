use crate::rust::interpreter::compiler::exports::{ProcVisitInputs, ProcVisitOutputs};
use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::{Match, MatchCase, Par};
use models::rust::utils::{new_gbool_par, union};
use std::collections::HashMap;

use rholang_parser::ast::AnnProc;

pub fn normalize_p_if<'ast>(
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
                ProcVisitInputs {
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

    Ok(ProcVisitOutputs {
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

    use crate::rust::interpreter::test_utils::utils::proc_visit_inputs_and_env;

    #[test]
    fn p_if_else_should_desugar_to_match_with_true_false_cases() {
        // if (true) { @Nil!(47) }
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;
        use rholang_parser::ast::SendType;

        let (inputs, env) = proc_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();

        let condition = ParBuilderUtil::create_ast_bool_literal(true, &parser);
        let nil_proc = ParBuilderUtil::create_ast_nil(&parser);
        let channel = ParBuilderUtil::create_ast_quote_name(nil_proc);
        let input_47 = ParBuilderUtil::create_ast_long_literal(47, &parser);
        let if_true =
            ParBuilderUtil::create_ast_send(channel, SendType::Single, vec![input_47], &parser);
        let if_then_else =
            ParBuilderUtil::create_ast_if_then_else(condition, if_true, None, &parser);

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
    fn p_if_else_should_not_mix_par_from_the_input_with_normalized_one() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;

        let (mut inputs, env) = proc_visit_inputs_and_env();
        inputs.par = Par::default().with_exprs(vec![new_gint_expr(7)]);

        let parser = rholang_parser::RholangParser::new();

        // if (true) { 10 }
        let condition = ParBuilderUtil::create_ast_bool_literal(true, &parser);
        let if_true = ParBuilderUtil::create_ast_long_literal(10, &parser);
        let if_then_else =
            ParBuilderUtil::create_ast_if_then_else(condition, if_true, None, &parser);

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
    fn p_if_else_should_handle_a_more_complicated_if_statement_with_an_else_clause() {
        // if (47 == 47) { new x in { x!(47) } } else { new y in { y!(47) } }
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;
        use rholang_parser::ast::{BinaryExpOp, Id, Name, SendType, Var};
        use rholang_parser::SourcePos;

        let (inputs, env) = proc_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();

        // Condition: 47 == 47
        let left_47 = ParBuilderUtil::create_ast_long_literal(47, &parser);
        let right_47 = ParBuilderUtil::create_ast_long_literal(47, &parser);
        let condition =
            ParBuilderUtil::create_ast_binary_exp(BinaryExpOp::Eq, left_47, right_47, &parser);

        // If true: new x in { x!(47) }
        let x_var = Var::Id(Id {
            name: "x",
            pos: SourcePos { line: 0, col: 0 },
        });
        let x_channel = Name::NameVar(x_var);
        let x_input_47 = ParBuilderUtil::create_ast_long_literal(47, &parser);
        let x_send =
            ParBuilderUtil::create_ast_send(x_channel, SendType::Single, vec![x_input_47], &parser);
        let if_true = ParBuilderUtil::create_ast_new(vec![x_var], x_send, &parser);

        // If false: new y in { y!(47) }
        let y_var = Var::Id(Id {
            name: "y",
            pos: SourcePos { line: 0, col: 0 },
        });
        let y_channel = Name::NameVar(y_var);
        let y_input_47 = ParBuilderUtil::create_ast_long_literal(47, &parser);
        let y_send =
            ParBuilderUtil::create_ast_send(y_channel, SendType::Single, vec![y_input_47], &parser);
        let if_false = ParBuilderUtil::create_ast_new(vec![y_var], y_send, &parser);

        let if_then_else =
            ParBuilderUtil::create_ast_if_then_else(condition, if_true, Some(if_false), &parser);

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
