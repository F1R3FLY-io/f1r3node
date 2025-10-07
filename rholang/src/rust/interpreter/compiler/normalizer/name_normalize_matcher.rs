use crate::rust::interpreter::compiler::exports::{
    BoundContext, FreeContext, NameVisitInputs, NameVisitOutputs, ProcVisitInputs,
};
use crate::rust::interpreter::compiler::normalize::{normalize_ann_proc, VarSort};
use crate::rust::interpreter::compiler::normalizer::remainder_normalizer_matcher::normalize_match_name;
use crate::rust::interpreter::compiler::span_utils::SpanContext;
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::util::prepend_expr;
use models::rhoapi::{expr, var, EVar, Expr, Par, Var as model_var};
use models::rust::utils::union;
use std::collections::HashMap;

use rholang_parser::ast::{Name, Names, Var};

pub fn normalize_name<'ast>(
    name: &Name<'ast>,
    input: NameVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<NameVisitOutputs, InterpreterError> {
    match name {
        Name::ProcVar(var) => {
            match var {
                Var::Wildcard => {
                    let wildcard_span = SpanContext::wildcard_span();
                    let wildcard_bind_result = input.free_map.add_wildcard(wildcard_span);

                    let new_expr = Expr {
                        expr_instance: Some(expr::ExprInstance::EVarBody(EVar {
                            v: Some(model_var {
                                var_instance: Some(var::VarInstance::Wildcard(var::WildcardMsg {})),
                            }),
                        })),
                    };

                    Ok(NameVisitOutputs {
                        par: prepend_expr(
                            Par::default(),
                            new_expr,
                            input.bound_map_chain.depth() as i32,
                        ),
                        free_map: wildcard_bind_result,
                    })
                }

                Var::Id(id) => {
                    let name = id.name;
                    // Extract proper source position from Id
                    let source_pos = id.pos;

                    match input.bound_map_chain.get(name) {
                        Some(bound_context) => match bound_context {
                            BoundContext {
                                index: level,
                                typ: VarSort::NameSort,
                                ..
                            } => {
                                let new_expr = Expr {
                                    expr_instance: Some(expr::ExprInstance::EVarBody(EVar {
                                        v: Some(model_var {
                                            var_instance: Some(var::VarInstance::BoundVar(
                                                level as i32,
                                            )),
                                        }),
                                    })),
                                };

                                Ok(NameVisitOutputs {
                                    par: prepend_expr(
                                        Par::default(),
                                        new_expr,
                                        input.bound_map_chain.depth() as i32,
                                    ),
                                    free_map: input.free_map,
                                })
                            }

                            BoundContext {
                                typ: VarSort::ProcSort,
                                source_span: proc_var_span,
                                ..
                            } => Err(InterpreterError::UnexpectedNameContext {
                                var_name: name.to_string(),
                                proc_var_source_span: proc_var_span,
                                name_source_span: SpanContext::pos_to_span(source_pos),
                            }),
                        },

                        None => match input.free_map.get(name) {
                            None => {
                                let updated_free_map = input.free_map.put_pos((
                                    name.to_string(),
                                    VarSort::NameSort,
                                    source_pos,
                                ));
                                let new_expr = Expr {
                                    expr_instance: Some(expr::ExprInstance::EVarBody(EVar {
                                        v: Some(model_var {
                                            var_instance: Some(var::VarInstance::FreeVar(
                                                input.free_map.next_level as i32,
                                            )),
                                        }),
                                    })),
                                };

                                Ok(NameVisitOutputs {
                                    par: prepend_expr(
                                        Par::default(),
                                        new_expr,
                                        input.bound_map_chain.depth() as i32,
                                    ),
                                    free_map: updated_free_map,
                                })
                            }
                            Some(FreeContext {
                                source_span: first_span,
                                ..
                            }) => Err(InterpreterError::UnexpectedReuseOfNameContextFree {
                                var_name: name.to_string(),
                                first_use: first_span,
                                second_use: SpanContext::pos_to_span(source_pos),
                            }),
                        },
                    }
                }
            }
        }

        Name::Quote(proc) => {
            use rholang_parser::ast::AnnProc;

            // Create a synthetic SourceSpan for the quoted process since quotes don't have inherent spans
            // Use wildcard span for semantic clarity that this is compiler-generated
            let span = SpanContext::wildcard_span();

            // Wrap the quoted process in an AnnProc for normalization
            let ann_proc = AnnProc { proc: *proc, span };

            // Call normalize_ann_proc with proper span-based inputs
            let proc_visit_result = normalize_ann_proc(
                &ann_proc,
                ProcVisitInputs {
                    par: Par::default(),
                    bound_map_chain: input.bound_map_chain.clone(),
                    free_map: input.free_map.clone(),
                },
                env,
                parser,
            )?;

            // Return the normalized result
            Ok(NameVisitOutputs {
                par: proc_visit_result.par,
                free_map: proc_visit_result.free_map,
            })
        }
    }
}

pub fn normalize_names<'ast>(
    names: &Names<'ast>,
    input: NameVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<NameVisitOutputs, InterpreterError> {
    let mut current_input = input;
    let mut accumulated_par = Par::default();

    // Process each name in the names vector
    for ann_name in &names.names {
        let name_result = normalize_name(&ann_name.name, current_input.clone(), env, parser)?;

        // Accumulate results using prepend_expr for proper Par composition
        accumulated_par = Par {
            exprs: [accumulated_par.exprs, name_result.par.exprs].concat(),
            locally_free: union(accumulated_par.locally_free, name_result.par.locally_free),
            connective_used: accumulated_par.connective_used || name_result.par.connective_used,
            ..Par::default()
        };

        // Update free map for next iteration
        current_input.free_map = name_result.free_map;
    }

    // Handle remainder if present
    if let Some(remainder_var) = &names.remainder {
        let (remainder_model_var, updated_free_map) =
            normalize_match_name(&Some(remainder_var.clone()), current_input.free_map)?;

        // If there's a remainder variable, add it to the expressions
        if let Some(var) = remainder_model_var {
            let remainder_expr = Expr {
                expr_instance: Some(expr::ExprInstance::EVarBody(EVar { v: Some(var) })),
            };

            accumulated_par = prepend_expr(
                accumulated_par,
                remainder_expr,
                current_input.bound_map_chain.depth() as i32,
            );
        }

        current_input.free_map = updated_free_map;
    }

    Ok(NameVisitOutputs {
        par: accumulated_par,
        free_map: current_input.free_map,
    })
}

//rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/NameMatcherSpec.scala
#[cfg(test)]
mod tests {
    use super::*;
    use crate::rust::interpreter::test_utils::utils::name_visit_inputs_and_env;
    use models::create_bit_vector;
    use models::rust::utils::{new_boundvar_par, new_freevar_par, new_gint_par, new_wildcard_par};
    use rholang_parser::ast::Proc;

    fn bound_name_inputs_with_bound_map_chain_span(
        input: NameVisitInputs,
        name: &str,
        v_type: VarSort,
        line_num: usize,
        col_num: usize,
    ) -> NameVisitInputs {
        use rholang_parser::SourcePos;
        let source_pos = SourcePos {
            line: line_num,
            col: col_num,
        };
        NameVisitInputs {
            bound_map_chain: {
                let updated_bound_map_chain =
                    input
                        .bound_map_chain
                        .put_pos((name.to_string(), v_type, source_pos));
                updated_bound_map_chain
            },
            ..input.clone()
        }
    }

    fn bound_name_inputs_with_free_map_span(
        input: NameVisitInputs,
        name: &str,
        v_type: VarSort,
        line_num: usize,
        col_num: usize,
    ) -> NameVisitInputs {
        use rholang_parser::SourcePos;
        let source_pos = SourcePos {
            line: line_num,
            col: col_num,
        };
        NameVisitInputs {
            free_map: {
                let updated_free_map =
                    input
                        .clone()
                        .free_map
                        .put_pos((name.to_string(), v_type, source_pos));
                updated_free_map
            },
            ..input.clone()
        }
    }

    fn create_wildcard<'ast>() -> Name<'ast> {
        Name::ProcVar(Var::Wildcard)
    }

    fn create_id_var<'ast>(name: &'ast str) -> Name<'ast> {
        use rholang_parser::{ast::Id, SourcePos};
        Name::ProcVar(Var::Id(Id {
            name,
            pos: SourcePos { line: 1, col: 1 },
        }))
    }

    fn create_quote_ground<'ast>() -> Name<'ast> {
        Name::Quote(&Proc::LongLiteral(7))
    }

    #[test]
    fn name_wildcard_should_add_a_wildcard_count_to_known_free() {
        let nw = create_wildcard();
        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();

        let result = normalize_name(&nw, input, &env, &parser);
        let expected_result = new_wildcard_par(Vec::new(), true);

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.par, expected_result);
        assert_eq!(unwrap_result.free_map.count(), 1);
    }

    #[test]
    fn name_var_should_compile_as_bound_var_if_its_in_env() {
        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        let n_var = create_id_var("x");
        let bound_inputs = bound_name_inputs_with_bound_map_chain_span(
            input.clone(),
            "x",
            VarSort::NameSort,
            1,
            1,
        );

        let result = normalize_name(&n_var, bound_inputs.clone(), &env, &parser);
        let expected_result = new_boundvar_par(0, create_bit_vector(&vec![0]), false);

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.par, expected_result);
        assert_eq!(unwrap_result.free_map, bound_inputs.free_map);
    }

    #[test]
    fn name_var_should_compile_as_free_var_if_its_not_in_env() {
        let n_var = create_id_var("x");
        let parser = rholang_parser::RholangParser::new();
        let (input, env) = name_visit_inputs_and_env();

        let result = normalize_name(&n_var, input.clone(), &env, &parser);
        let expected_result = new_freevar_par(0, Vec::new());

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.par, expected_result);
        let bound_inputs =
            bound_name_inputs_with_free_map_span(input.clone(), "x", VarSort::NameSort, 1, 1);
        assert_eq!(result.unwrap().free_map, bound_inputs.free_map);
    }

    #[test]
    fn name_var_should_not_compile_if_its_in_env_of_wrong_sort() {
        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        let n_var = create_id_var("x");
        let bound_inputs = bound_name_inputs_with_bound_map_chain_span(
            input.clone(),
            "x",
            VarSort::ProcSort,
            1,
            1,
        );

        let result = normalize_name(&n_var, bound_inputs, &env, &parser);
        assert!(matches!(
            result,
            Err(InterpreterError::UnexpectedNameContext { .. })
        ));
    }

    #[test]
    fn name_var_should_not_compile_if_used_free_somewhere_else() {
        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        let n_var = create_id_var("x");
        let bound_inputs =
            bound_name_inputs_with_free_map_span(input.clone(), "x", VarSort::NameSort, 1, 1);

        let result = normalize_name(&n_var, bound_inputs, &env, &parser);
        assert!(matches!(
            result,
            Err(InterpreterError::UnexpectedReuseOfNameContextFree { .. })
        ));
    }

    #[test]
    fn name_quote_should_compile_to_a_ground() {
        let n_q_ground = create_quote_ground();
        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();

        let result = normalize_name(&n_q_ground, input.clone(), &env, &parser);
        let expected_result = new_gint_par(7, Vec::new(), false);

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.par, expected_result);
        assert_eq!(unwrap_result.free_map, input.free_map);
    }

    #[test]
    fn name_quote_should_compile_to_bound_var() {
        use rholang_parser::ast::{AnnName, Id, Proc};
        use rholang_parser::{SourcePos, SourceSpan};

        let quoted_proc = Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
            name: "x",
            pos: SourcePos { line: 1, col: 1 },
        }))));

        let quote_name = AnnName {
            name: Name::Quote(quoted_proc),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        let bound_inputs = bound_name_inputs_with_bound_map_chain_span(
            input.clone(),
            "x",
            VarSort::ProcSort,
            1,
            1,
        );

        let result = normalize_name(&quote_name.name, bound_inputs.clone(), &env, &parser);
        let expected_result: Par = new_boundvar_par(0, create_bit_vector(&vec![0]), false);

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.par, expected_result);
        assert_eq!(unwrap_result.free_map, bound_inputs.free_map);
    }

    #[test]
    fn name_quote_should_return_a_free_use_if_the_quoted_proc_has_a_free_var() {
        use rholang_parser::ast::{AnnName, Id, Proc};
        use rholang_parser::{SourcePos, SourceSpan};

        let quoted_proc = Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
            name: "x",
            pos: SourcePos { line: 1, col: 1 },
        }))));

        let quote_name = AnnName {
            name: Name::Quote(quoted_proc),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();

        let result = normalize_name(&quote_name.name, input.clone(), &env, &parser);
        let expected_result = new_freevar_par(0, Vec::new());

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.par, expected_result);

        let bound_inputs =
            bound_name_inputs_with_free_map_span(input.clone(), "x", VarSort::ProcSort, 1, 1);
        assert_eq!(unwrap_result.free_map, bound_inputs.free_map);
    }

    #[test]
    fn name_quote_should_collapse_an_eval() {
        use rholang_parser::ast::{AnnName, Id, Proc};
        use rholang_parser::{SourcePos, SourceSpan};

        let eval_name = AnnName {
            name: Name::ProcVar(Var::Id(Id {
                name: "x",
                pos: SourcePos { line: 1, col: 1 },
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let quoted_proc = Box::leak(Box::new(Proc::Eval { name: eval_name }));

        let quote_name = AnnName {
            name: Name::Quote(quoted_proc),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let (input, env) = name_visit_inputs_and_env();
        let bound_inputs = bound_name_inputs_with_bound_map_chain_span(
            input.clone(),
            "x",
            VarSort::NameSort,
            1,
            1,
        );
        let parser = rholang_parser::RholangParser::new();

        let result = normalize_name(&quote_name.name, bound_inputs.clone(), &env, &parser);
        let expected_result = new_boundvar_par(0, create_bit_vector(&vec![0]), false);

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.par, expected_result);
        assert_eq!(unwrap_result.free_map, bound_inputs.free_map);
    }

    #[test]
    fn name_quote_should_not_collapse_an_eval_eval() {
        use rholang_parser::ast::{AnnName, AnnProc, Id, Proc};
        use rholang_parser::{SourcePos, SourceSpan};

        let eval_name_left = AnnName {
            name: Name::ProcVar(Var::Id(Id {
                name: "x",
                pos: SourcePos { line: 1, col: 1 },
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let eval_name_right = AnnName {
            name: Name::ProcVar(Var::Id(Id {
                name: "x",
                pos: SourcePos { line: 1, col: 1 },
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let left_eval = AnnProc {
            proc: Box::leak(Box::new(Proc::Eval {
                name: eval_name_left,
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let right_eval = AnnProc {
            proc: Box::leak(Box::new(Proc::Eval {
                name: eval_name_right,
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let quoted_proc = Box::leak(Box::new(Proc::Par {
            left: left_eval,
            right: right_eval,
        }));

        let quote_name = AnnName {
            name: Name::Quote(quoted_proc),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        let bound_inputs = bound_name_inputs_with_bound_map_chain_span(
            input.clone(),
            "x",
            VarSort::NameSort,
            1,
            1,
        );

        let result = normalize_name(&quote_name.name, bound_inputs.clone(), &env, &parser);

        let bound_var_expr = new_boundvar_par(0, create_bit_vector(&vec![0]), false);
        let expected_result =
            prepend_expr(bound_var_expr.clone(), bound_var_expr.exprs[0].clone(), 0);

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.par, expected_result);
        assert_eq!(unwrap_result.free_map, bound_inputs.free_map);
    }

    #[test]
    fn normalize_names_single_var() {
        use rholang_parser::ast::{AnnName, Id, Names};
        use rholang_parser::{SourcePos, SourceSpan};

        let ann_name = AnnName {
            name: Name::ProcVar(Var::Id(Id {
                name: "x",
                pos: SourcePos { line: 1, col: 1 },
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let mut names_vec = smallvec::SmallVec::new();
        names_vec.push(ann_name);

        let names = Names {
            names: names_vec,
            remainder: None,
        };

        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        let bound_inputs = bound_name_inputs_with_bound_map_chain_span(
            input.clone(),
            "x",
            VarSort::NameSort,
            1,
            1,
        );

        let result = normalize_names(&names, bound_inputs.clone(), &env, &parser);
        assert!(result.is_ok());

        let unwrap_result = result.unwrap();
        let expected_result = new_boundvar_par(0, create_bit_vector(&vec![0]), false);
        assert_eq!(unwrap_result.par, expected_result);
        assert_eq!(unwrap_result.free_map, bound_inputs.free_map);
    }

    #[test]
    fn normalize_names_multiple_vars() {
        use rholang_parser::ast::{AnnName, Id, Names};
        use rholang_parser::{SourcePos, SourceSpan};

        let ann_name_x = AnnName {
            name: Name::ProcVar(Var::Id(Id {
                name: "x",
                pos: SourcePos { line: 1, col: 1 },
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let ann_name_y = AnnName {
            name: Name::ProcVar(Var::Id(Id {
                name: "y",
                pos: SourcePos { line: 1, col: 1 },
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let mut names_vec = smallvec::SmallVec::new();
        names_vec.push(ann_name_x);
        names_vec.push(ann_name_y);

        let names = Names {
            names: names_vec,
            remainder: None,
        };

        let (mut input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        input.bound_map_chain = input
            .bound_map_chain
            .put_pos((
                "x".to_string(),
                VarSort::NameSort,
                rholang_parser::SourcePos { line: 1, col: 1 },
            ))
            .put_pos((
                "y".to_string(),
                VarSort::NameSort,
                rholang_parser::SourcePos { line: 1, col: 1 },
            ));

        let result = normalize_names(&names, input.clone(), &env, &parser);
        assert!(result.is_ok());

        let unwrap_result = result.unwrap();
        assert!(!unwrap_result.par.exprs.is_empty());
        assert_eq!(unwrap_result.free_map.count(), input.free_map.count());
    }

    #[test]
    fn normalize_names_with_remainder() {
        use rholang_parser::ast::{AnnName, Id, Names};
        use rholang_parser::{SourcePos, SourceSpan};

        let ann_name = AnnName {
            name: Name::ProcVar(Var::Id(Id {
                name: "x",
                pos: SourcePos { line: 1, col: 1 },
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let remainder_var = Var::Id(Id {
            name: "rest",
            pos: SourcePos { line: 1, col: 1 },
        });

        let mut names_vec = smallvec::SmallVec::new();
        names_vec.push(ann_name);

        let names = Names {
            names: names_vec,
            remainder: Some(remainder_var),
        };

        let (mut input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        input.bound_map_chain = input.bound_map_chain.put_pos((
            "x".to_string(),
            VarSort::NameSort,
            rholang_parser::SourcePos { line: 1, col: 1 },
        ));

        let result = normalize_names(&names, input.clone(), &env, &parser);
        assert!(result.is_ok());

        let unwrap_result = result.unwrap();
        assert!(!unwrap_result.par.exprs.is_empty());
        assert!(unwrap_result.free_map.level_bindings.contains_key("rest"));
    }

    #[test]
    fn normalize_names_wildcard_remainder() {
        use rholang_parser::ast::{AnnName, Id, Names};
        use rholang_parser::{SourcePos, SourceSpan};

        let ann_name = AnnName {
            name: Name::ProcVar(Var::Id(Id {
                name: "x",
                pos: SourcePos { line: 1, col: 1 },
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let remainder_wildcard = Var::Wildcard;

        let mut names_vec = smallvec::SmallVec::new();
        names_vec.push(ann_name);

        let names = Names {
            names: names_vec,
            remainder: Some(remainder_wildcard),
        };

        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        let bound_inputs = bound_name_inputs_with_bound_map_chain_span(
            input.clone(),
            "x",
            VarSort::NameSort,
            1,
            1,
        );

        let result = normalize_names(&names, bound_inputs.clone(), &env, &parser);
        assert!(result.is_ok());

        let unwrap_result = result.unwrap();
        assert!(!unwrap_result.par.exprs.is_empty());
        assert!(!unwrap_result.free_map.wildcards.is_empty());
    }

    #[test]
    fn normalize_names_empty_list() {
        use rholang_parser::ast::{Id, Names};
        use rholang_parser::SourcePos;

        let remainder_var = Var::Id(Id {
            name: "rest",
            pos: SourcePos { line: 1, col: 1 },
        });

        let names = Names {
            names: smallvec::SmallVec::new(),
            remainder: Some(remainder_var),
        };

        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();

        let result = normalize_names(&names, input.clone(), &env, &parser);
        assert!(result.is_ok());

        let unwrap_result = result.unwrap();
        assert!(unwrap_result.free_map.level_bindings.contains_key("rest"));
    }
}
