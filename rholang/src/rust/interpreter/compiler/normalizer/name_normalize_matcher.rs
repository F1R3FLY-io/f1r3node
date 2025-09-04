use super::exports::*;
use crate::rust::interpreter::compiler::bound_context::BoundContext;
use crate::rust::interpreter::compiler::exports::FreeContext;
use crate::rust::interpreter::compiler::normalize::{normalize_match_proc, VarSort};
use crate::rust::interpreter::compiler::rholang_ast::{Name, Proc, Quote, Var};
use crate::rust::interpreter::compiler::source_position::SourcePosition;
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::util::prepend_expr;
use models::rhoapi::{expr, var, EVar, Expr, Par, Var as model_var};
use models::rust::utils::union;
use std::collections::HashMap;

// New AST imports for parallel functions
use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
use rholang_parser::ast::{Name as NewName, Names as NewNames, Var as NewVar};

pub fn normalize_name(
    proc: &Name,
    input: NameVisitInputs,
    env: &HashMap<String, Par>,
) -> Result<NameVisitOutputs, InterpreterError> {
    match proc {
        Name::ProcVar(boxed_proc) => match *boxed_proc.clone() {
            Proc::Wildcard { line_num, col_num } => {
                let wildcard_bind_result = input.free_map.add_wildcard(SourcePosition {
                    row: line_num,
                    column: col_num,
                });

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

            Proc::Var(Var {
                name,
                line_num,
                col_num,
            }) => match input.bound_map_chain.get(&name) {
                Some(bound_context) => match bound_context {
                    BoundContext {
                        index: level,
                        typ: VarSort::NameSort,
                        ..
                    } => {
                        let new_expr = Expr {
                            expr_instance: Some(expr::ExprInstance::EVarBody(EVar {
                                v: Some(model_var {
                                    var_instance: Some(var::VarInstance::BoundVar(level as i32)),
                                }),
                            })),
                        };

                        return Ok(NameVisitOutputs {
                            par: prepend_expr(
                                Par::default(),
                                new_expr,
                                input.bound_map_chain.depth() as i32,
                            ),
                            free_map: input.free_map.clone(),
                        });
                    }
                    BoundContext {
                        typ: VarSort::ProcSort,
                        source_position,
                        ..
                    } => {
                        return Err(InterpreterError::UnexpectedNameContext {
                            var_name: name.to_string(),
                            proc_var_source_position: source_position.to_string(),
                            name_source_position: SourcePosition {
                                row: line_num,
                                column: col_num,
                            }
                            .to_string(),
                        }
                        .into());
                    }
                },
                None => match input.free_map.get(&name) {
                    None => {
                        let updated_free_map = input.free_map.put((
                            name.to_string(),
                            VarSort::NameSort,
                            SourcePosition {
                                row: line_num,
                                column: col_num,
                            },
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
                        source_position, ..
                    }) => Err(InterpreterError::UnexpectedReuseOfNameContextFree {
                        var_name: name.to_string(),
                        first_use: source_position.to_string(),
                        second_use: SourcePosition {
                            row: line_num,
                            column: col_num,
                        }
                        .to_string(),
                    }
                    .into()),
                },
            },

            _ => Err(InterpreterError::BugFoundError(format!(
                "Expected Proc::Var or Proc::Wildcard, found {:?}",
                boxed_proc
            ))),
        },

        Name::Quote(boxed_quote) => {
            let Quote { ref quotable, .. } = *boxed_quote.clone();
            let proc_visit_result = normalize_match_proc(
                &quotable,
                ProcVisitInputs {
                    par: Par::default(),
                    bound_map_chain: input.bound_map_chain.clone(),
                    free_map: input.free_map.clone(),
                    source_span: input.source_span,
                },
                env,
            )?;

            Ok(NameVisitOutputs {
                par: proc_visit_result.par,
                free_map: proc_visit_result.free_map,
            })
        }
    }
}

// ============================================================================
// NEW AST PARALLEL FUNCTIONS
// ============================================================================

/// Parallel version of normalize_name for new AST Name types
/// Handles the direct Name<'ast> instead of old Name enum
pub fn normalize_name_new_ast<'ast>(
    name: &NewName<'ast>,
    input: NameVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<NameVisitOutputs, InterpreterError> {
    match name {
        NewName::ProcVar(var) => {
            match var {
                NewVar::Wildcard => {
                    // TODO: Convert SourcePos from new AST to SourcePosition
                    // For now, use placeholder position - this will be fixed in SourceSpan migration
                    let wildcard_bind_result = input
                        .free_map
                        .add_wildcard(SourcePosition { row: 0, column: 0 });

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

                NewVar::Id(id) => {
                    let name = id.name;
                    // TODO: Convert SourcePos from new AST to SourcePosition
                    // For now, use placeholder position - this will be fixed in SourceSpan migration
                    let source_pos = SourcePosition { row: 0, column: 0 };

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
                                source_position,
                                ..
                            } => Err(InterpreterError::UnexpectedNameContext {
                                var_name: name.to_string(),
                                proc_var_source_position: source_position.to_string(),
                                name_source_position: source_pos.to_string(),
                            }),
                        },

                        None => match input.free_map.get(name) {
                            None => {
                                let updated_free_map = input.free_map.put((
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
                                source_position, ..
                            }) => Err(InterpreterError::UnexpectedReuseOfNameContextFree {
                                var_name: name.to_string(),
                                first_use: source_position.to_string(),
                                second_use: source_pos.to_string(),
                            }
                            .into()),
                        },
                    }
                }
            }
        }

        NewName::Quote(proc) => {
            // TODO: Review quote wrapping here
            // For quotes, we need to normalize the quoted process
            // Quote contains &'ast Proc<'ast>, so we need to wrap it in AnnProc
            use rholang_parser::ast::AnnProc;
            use rholang_parser::{SourcePos, SourceSpan};

            // Create a minimal SourceSpan for the quoted process
            let span = SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            };

            let ann_proc = AnnProc { proc: *proc, span };

            let proc_visit_result = normalize_ann_proc(
                &ann_proc,
                ProcVisitInputs {
                    par: Par::default(),
                    bound_map_chain: input.bound_map_chain.clone(),
                    free_map: input.free_map.clone(),
                    source_span: input.source_span,
                },
                env,
                parser,
            )?;

            Ok(NameVisitOutputs {
                par: proc_visit_result.par,
                free_map: proc_visit_result.free_map,
            })
        }
    }
}

/// Parallel version of normalize_names for new AST Names structure
/// Handles the Names<'ast> with SmallVec of AnnName and optional remainder
pub fn normalize_names_new_ast<'ast>(
    names: &NewNames<'ast>,
    input: NameVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<NameVisitOutputs, InterpreterError> {
    let mut current_input = input;
    let mut accumulated_par = Par::default();

    // Process each name in the names vector
    for ann_name in &names.names {
        let name_result = normalize_name_new_ast(&ann_name.name, current_input.clone(), env, parser)?;

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
        // Use the remainder normalizer we created earlier
        use crate::rust::interpreter::compiler::normalizer::remainder_normalizer_matcher::normalize_match_name_new_ast;

        let (remainder_model_var, updated_free_map) =
            normalize_match_name_new_ast(&Some(remainder_var.clone()), current_input.free_map)?;

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
    use crate::rust::interpreter::compiler::rholang_ast::Name;
    use crate::rust::interpreter::test_utils::utils::name_visit_inputs_and_env;
    use models::create_bit_vector;
    use models::rust::utils::{new_boundvar_par, new_freevar_par, new_gint_par, new_wildcard_par};

    fn bound_name_inputs_with_bound_map_chain(
        input: NameVisitInputs,
        name: &str,
        v_type: VarSort,
        line_num: usize,
        col_num: usize,
    ) -> NameVisitInputs {
        NameVisitInputs {
            bound_map_chain: {
                let updated_bound_map_chain = input.bound_map_chain.put((
                    name.to_string(),
                    v_type,
                    SourcePosition {
                        row: line_num,
                        column: col_num,
                    },
                ));
                updated_bound_map_chain
            },
            ..input.clone()
        }
    }

    fn bound_name_inputs_with_free_map(
        input: NameVisitInputs,
        name: &str,
        v_type: VarSort,
        line_num: usize,
        col_num: usize,
    ) -> NameVisitInputs {
        NameVisitInputs {
            free_map: {
                let updated_free_map = input.clone().free_map.put((
                    name.to_string(),
                    v_type,
                    SourcePosition {
                        row: line_num,
                        column: col_num,
                    },
                ));
                updated_free_map
            },
            ..input.clone()
        }
    }

    #[test]
    fn name_wildcard_should_add_a_wildcard_count_to_known_free() {
        let nw = Name::new_name_wildcard();
        let (input, env) = name_visit_inputs_and_env();

        let result = normalize_name(&nw, input, &env);
        let expected_result = new_wildcard_par(Vec::new(), true);

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.clone().par, expected_result);
        assert_eq!(unwrap_result.clone().free_map.count(), 1);
    }

    #[test]
    fn name_var_should_compile_as_bound_var_if_its_in_env() {
        let (input, env) = name_visit_inputs_and_env();
        let n_var: Name = Name::new_name_var("x");
        let bound_inputs =
            bound_name_inputs_with_bound_map_chain(input.clone(), "x", VarSort::NameSort, 0, 0);

        let result = normalize_name(&n_var, bound_inputs.clone(), &env);
        let expected_result = new_boundvar_par(0, create_bit_vector(&vec![0]), false);

        let unwrap_result: NameVisitOutputs = result.clone().unwrap();

        assert_eq!(unwrap_result.par, expected_result);
        assert_eq!(
            unwrap_result.clone().free_map,
            bound_inputs.clone().free_map
        );
    }

    #[test]
    fn name_var_should_compile_as_free_var_if_its_not_in_env() {
        let n_var: Name = Name::new_name_var("x");
        let (input, env) = name_visit_inputs_and_env();

        let result = normalize_name(&n_var, input.clone(), &env);
        let expected_result = new_freevar_par(0, Vec::new());

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.par, expected_result);
        let bound_inputs =
            bound_name_inputs_with_free_map(input.clone(), "x", VarSort::NameSort, 0, 0);
        assert_eq!(result.unwrap().free_map, bound_inputs.free_map);
    }

    #[test]
    fn name_var_should_not_compile_if_its_in_env_of_wrong_sort() {
        let (input, env) = name_visit_inputs_and_env();
        let n_var: Name = Name::new_name_var("x");
        let bound_inputs =
            bound_name_inputs_with_bound_map_chain(input.clone(), "x", VarSort::ProcSort, 0, 0);

        let result = normalize_name(&n_var, bound_inputs, &env);
        assert!(matches!(
            result,
            Err(InterpreterError::UnexpectedNameContext { .. })
        ));
    }

    #[test]
    fn name_var_should_not_compile_if_used_free_somewhere_else() {
        let (input, env) = name_visit_inputs_and_env();
        let n_var: Name = Name::new_name_var("x");
        let bound_inputs =
            bound_name_inputs_with_free_map(input.clone(), "x", VarSort::NameSort, 0, 0);

        let result = normalize_name(&n_var, bound_inputs, &env);
        assert!(matches!(
            result,
            Err(InterpreterError::UnexpectedReuseOfNameContextFree { .. })
        ));
    }

    #[test]
    fn name_quote_should_compile_to_bound_var() {
        let n_q_var = Name::new_name_quote_var("x");
        let (input, env) = name_visit_inputs_and_env();
        let bound_inputs =
            bound_name_inputs_with_bound_map_chain(input.clone(), "x", VarSort::ProcSort, 0, 0);

        let result = normalize_name(&n_q_var, bound_inputs.clone(), &env);
        let expected_result: Par = new_boundvar_par(0, create_bit_vector(&vec![0]), false);

        let unwrap_result = result.clone().unwrap();

        assert_eq!(unwrap_result.clone().par, expected_result);
        assert_eq!(unwrap_result.clone().free_map, bound_inputs.free_map);
    }

    #[test]
    fn name_quote_should_return_a_free_use_if_the_quoted_proc_has_a_free_var() {
        let n_q_var = Name::new_name_quote_var("x");
        let (input, env) = name_visit_inputs_and_env();

        let result = normalize_name(&n_q_var, input.clone(), &env);
        let expected_result = new_freevar_par(0, Vec::new());

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.clone().par, expected_result);

        let bound_inputs =
            bound_name_inputs_with_free_map(input.clone(), "x", VarSort::ProcSort, 0, 0);
        assert_eq!(unwrap_result.clone().free_map, bound_inputs.free_map);
    }

    #[test]
    fn name_quote_should_compile_to_a_ground() {
        let n_q_ground = Name::new_name_quote_ground_long_literal(7);
        let (input, env) = name_visit_inputs_and_env();

        let result = normalize_name(&n_q_ground, input.clone(), &env);
        let expected_result = new_gint_par(7, Vec::new(), false);

        let unwrap_result = result.clone().unwrap();

        assert_eq!(unwrap_result.clone().par, expected_result);
        assert_eq!(unwrap_result.clone().free_map, input.free_map);
    }

    #[test]
    fn name_quote_should_collapse_an_eval() {
        let n_q_eval = Name::new_name_quote_eval("x");
        let (input, env) = name_visit_inputs_and_env();
        let bound_inputs =
            bound_name_inputs_with_bound_map_chain(input.clone(), "x", VarSort::NameSort, 0, 0);

        let result = normalize_name(&n_q_eval, bound_inputs.clone(), &env);
        let expected_result = new_boundvar_par(0, create_bit_vector(&vec![0]), false);

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.clone().par, expected_result);
        assert_eq!(unwrap_result.clone().free_map, bound_inputs.free_map);
    }

    #[test]
    fn name_quote_should_not_collapse_an_eval_eval() {
        let n_q_eval = Name::new_name_quote_par_of_evals("x");
        let (input, env) = name_visit_inputs_and_env();
        let bound_inputs =
            bound_name_inputs_with_bound_map_chain(input.clone(), "x", VarSort::NameSort, 0, 0);

        let result = normalize_name(&n_q_eval, bound_inputs.clone(), &env);

        let bound_var_expr = new_boundvar_par(0, create_bit_vector(&vec![0]), false);
        let expected_result =
            prepend_expr(bound_var_expr.clone(), bound_var_expr.exprs[0].clone(), 0);

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.clone().par, expected_result);
        assert_eq!(unwrap_result.clone().free_map, bound_inputs.free_map);
}

// ============================================================================
    // NEW AST TESTS - Parallel tests for normalize_name_new_ast
// ============================================================================

    // Helper functions for creating new AST test data
    fn create_new_ast_wildcard<'ast>() -> NewName<'ast> {
        NewName::ProcVar(NewVar::Wildcard)
    }

    fn create_new_ast_id_var<'ast>(name: &'ast str) -> NewName<'ast> {
        use rholang_parser::{ast::Id, SourcePos};
        NewName::ProcVar(NewVar::Id(Id {
            name,
            pos: SourcePos { line: 1, col: 1 },
        }))
    }

    fn create_new_ast_quote_ground<'ast>() -> NewName<'ast> {
        use rholang_parser::ast::Proc as NewProc;
        NewName::Quote(&NewProc::LongLiteral(7))
    }

    #[test]
    fn new_ast_name_wildcard_should_add_a_wildcard_count_to_known_free() {
        let nw = create_new_ast_wildcard();
        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();

        let result = normalize_name_new_ast(&nw, input, &env, &parser);
        let expected_result = new_wildcard_par(Vec::new(), true);

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.par, expected_result);
        assert_eq!(unwrap_result.free_map.count(), 1);
    }

    #[test]
    fn new_ast_name_var_should_compile_as_bound_var_if_its_in_env() {
        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        let n_var = create_new_ast_id_var("x");
        let bound_inputs =
            bound_name_inputs_with_bound_map_chain(input.clone(), "x", VarSort::NameSort, 0, 0);

        let result = normalize_name_new_ast(&n_var, bound_inputs.clone(), &env, &parser);
        let expected_result = new_boundvar_par(0, create_bit_vector(&vec![0]), false);

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.par, expected_result);
        assert_eq!(unwrap_result.free_map, bound_inputs.free_map);
    }

    #[test]
    fn new_ast_name_var_should_compile_as_free_var_if_its_not_in_env() {
        let n_var = create_new_ast_id_var("x");
        let parser = rholang_parser::RholangParser::new();
        let (input, env) = name_visit_inputs_and_env();

        let result = normalize_name_new_ast(&n_var, input.clone(), &env, &parser);
        let expected_result = new_freevar_par(0, Vec::new());

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.par, expected_result);
        let bound_inputs =
            bound_name_inputs_with_free_map(input.clone(), "x", VarSort::NameSort, 0, 0);
        assert_eq!(result.unwrap().free_map, bound_inputs.free_map);
    }

    #[test]
    fn new_ast_name_var_should_not_compile_if_its_in_env_of_wrong_sort() {
        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        let n_var = create_new_ast_id_var("x");
        let bound_inputs =
            bound_name_inputs_with_bound_map_chain(input.clone(), "x", VarSort::ProcSort, 0, 0);

        let result = normalize_name_new_ast(&n_var, bound_inputs, &env, &parser);
        assert!(matches!(
            result,
            Err(InterpreterError::UnexpectedNameContext { .. })
        ));
    }

    #[test]
    fn new_ast_name_var_should_not_compile_if_used_free_somewhere_else() {
        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        let n_var = create_new_ast_id_var("x");
        let bound_inputs =
            bound_name_inputs_with_free_map(input.clone(), "x", VarSort::NameSort, 0, 0);

        let result = normalize_name_new_ast(&n_var, bound_inputs, &env, &parser);
        assert!(matches!(
            result,
            Err(InterpreterError::UnexpectedReuseOfNameContextFree { .. })
        ));
    }

    // ============================================================================
    // NEW AST QUOTE TESTS - EXACT MAPPING TO ORIGINAL TESTS
    // ============================================================================

    #[test]
    fn new_ast_name_quote_should_compile_to_a_ground() {
        // Maps to original: name_quote_should_compile_to_a_ground (line 326)
        let n_q_ground = create_new_ast_quote_ground();
        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();

        let result = normalize_name_new_ast(&n_q_ground, input.clone(), &env, &parser);
        let expected_result = new_gint_par(7, Vec::new(), false);

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.par, expected_result);
        assert_eq!(unwrap_result.free_map, input.free_map);
    }

    #[test]
    fn new_ast_name_quote_should_compile_to_bound_var() {
        // Maps to original: name_quote_should_compile_to_bound_var (line 506)
        use rholang_parser::ast::{AnnName, Id, Proc as NewProc};
        use rholang_parser::{SourcePos, SourceSpan};
        
        // Create @{x} where x is a ProcVar
        let quoted_proc = Box::leak(Box::new(NewProc::ProcVar(NewVar::Id(Id {
            name: "x",
            pos: SourcePos { line: 1, col: 1 },
        }))));

        let quote_name = AnnName {
            name: NewName::Quote(quoted_proc),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        let bound_inputs =
            bound_name_inputs_with_bound_map_chain(input.clone(), "x", VarSort::ProcSort, 0, 0);

        let result = normalize_name_new_ast(&quote_name.name, bound_inputs.clone(), &env, &parser);
        let expected_result: Par = new_boundvar_par(0, create_bit_vector(&vec![0]), false);

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.par, expected_result);
        assert_eq!(unwrap_result.free_map, bound_inputs.free_map);
    }

    #[test]
    fn new_ast_name_quote_should_return_a_free_use_if_the_quoted_proc_has_a_free_var() {
        // Maps to original: name_quote_should_return_a_free_use_if_the_quoted_proc_has_a_free_var (line 522)
        use rholang_parser::ast::{AnnName, Id, Proc as NewProc};
        use rholang_parser::{SourcePos, SourceSpan};

        // Create @{x} where x is a ProcVar (unbound)
        let quoted_proc = Box::leak(Box::new(NewProc::ProcVar(NewVar::Id(Id {
            name: "x",
            pos: SourcePos { line: 1, col: 1 },
        }))));

        let quote_name = AnnName {
            name: NewName::Quote(quoted_proc),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();

        let result = normalize_name_new_ast(&quote_name.name, input.clone(), &env, &parser);
        let expected_result = new_freevar_par(0, Vec::new());

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.par, expected_result);

        let bound_inputs =
            bound_name_inputs_with_free_map(input.clone(), "x", VarSort::ProcSort, 1, 1);
        assert_eq!(unwrap_result.free_map, bound_inputs.free_map);
    }

    #[test]
    fn new_ast_name_quote_should_collapse_an_eval() {
        // Maps to original: name_quote_should_collapse_an_eval (line 552)
        use rholang_parser::ast::{AnnName, Id, Proc as NewProc};
        use rholang_parser::{SourcePos, SourceSpan};

        // Create @{*x} where x is a NameSort variable
        let eval_name = AnnName {
            name: NewName::ProcVar(NewVar::Id(Id {
                name: "x", 
                pos: SourcePos { line: 1, col: 1 },
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let quoted_proc = Box::leak(Box::new(NewProc::Eval { name: eval_name }));

        let quote_name = AnnName {
            name: NewName::Quote(quoted_proc),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let (input, env) = name_visit_inputs_and_env();
        let bound_inputs =
            bound_name_inputs_with_bound_map_chain(input.clone(), "x", VarSort::NameSort, 0, 0);
        let parser = rholang_parser::RholangParser::new();

        let result = normalize_name_new_ast(&quote_name.name, bound_inputs.clone(), &env, &parser);
        let expected_result = new_boundvar_par(0, create_bit_vector(&vec![0]), false);

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.par, expected_result);
        assert_eq!(unwrap_result.free_map, bound_inputs.free_map);
    }

    #[test]
    fn new_ast_name_quote_should_not_collapse_an_eval_eval() {
        // Maps to original: name_quote_should_not_collapse_an_eval_eval (line 567)
        use rholang_parser::ast::{AnnName, AnnProc, Id, Proc as NewProc};
        use rholang_parser::{SourcePos, SourceSpan};

        // Create @{*x | *x} where x is a NameSort variable
        let eval_name_left = AnnName {
            name: NewName::ProcVar(NewVar::Id(Id {
                name: "x",
                pos: SourcePos { line: 1, col: 1 },
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let eval_name_right = AnnName {
            name: NewName::ProcVar(NewVar::Id(Id {
                name: "x",
                pos: SourcePos { line: 1, col: 1 },
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let left_eval = AnnProc {
            proc: Box::leak(Box::new(NewProc::Eval { name: eval_name_left })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let right_eval = AnnProc {
            proc: Box::leak(Box::new(NewProc::Eval { name: eval_name_right })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let quoted_proc = Box::leak(Box::new(NewProc::Par {
            left: left_eval,
            right: right_eval,
        }));

        let quote_name = AnnName {
            name: NewName::Quote(quoted_proc),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        let bound_inputs =
            bound_name_inputs_with_bound_map_chain(input.clone(), "x", VarSort::NameSort, 0, 0);

        let result = normalize_name_new_ast(&quote_name.name, bound_inputs.clone(), &env, &parser);

        let bound_var_expr = new_boundvar_par(0, create_bit_vector(&vec![0]), false);
        let expected_result =
            prepend_expr(bound_var_expr.clone(), bound_var_expr.exprs[0].clone(), 0);

        let unwrap_result = result.clone().unwrap();
        assert_eq!(unwrap_result.par, expected_result);
        assert_eq!(unwrap_result.free_map, bound_inputs.free_map);
    }

    // ============================================================================
    // TESTS FOR normalize_names_new_ast
    // ============================================================================

    #[test]
    fn new_ast_normalize_names_single_var() {
        use rholang_parser::ast::{AnnName, Id, Names as NewNames};
        use rholang_parser::{SourcePos, SourceSpan};

        // Create Names with single variable "x"
        let ann_name = AnnName {
            name: NewName::ProcVar(NewVar::Id(Id {
                name: "x",
                pos: SourcePos { line: 1, col: 1 },
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        // Create a SmallVec manually
        let mut names_vec = smallvec::SmallVec::new();
        names_vec.push(ann_name);

        let names = NewNames {
            names: names_vec,
            remainder: None,
        };

        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        let bound_inputs =
            bound_name_inputs_with_bound_map_chain(input.clone(), "x", VarSort::NameSort, 0, 0);

        let result = normalize_names_new_ast(&names, bound_inputs.clone(), &env, &parser);
        assert!(result.is_ok());

        let unwrap_result = result.unwrap();
        let expected_result = new_boundvar_par(0, create_bit_vector(&vec![0]), false);
        assert_eq!(unwrap_result.par, expected_result);
        assert_eq!(unwrap_result.free_map, bound_inputs.free_map);
    }

    #[test]
    fn new_ast_normalize_names_multiple_vars() {
        use rholang_parser::ast::{AnnName, Id, Names as NewNames};
        use rholang_parser::{SourcePos, SourceSpan};

        // Create Names with variables "x" and "y"
        let ann_name_x = AnnName {
            name: NewName::ProcVar(NewVar::Id(Id {
                name: "x",
                pos: SourcePos { line: 1, col: 1 },
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let ann_name_y = AnnName {
            name: NewName::ProcVar(NewVar::Id(Id {
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

        let names = NewNames {
            names: names_vec,
            remainder: None,
        };

        let (mut input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        input.bound_map_chain = input
            .bound_map_chain
            .put(("x".to_string(), VarSort::NameSort, SourcePosition::new(0, 0)))
            .put(("y".to_string(), VarSort::NameSort, SourcePosition::new(0, 0)));

        let result = normalize_names_new_ast(&names, input.clone(), &env, &parser);
        assert!(result.is_ok());

        // Should have both variables in the result
        let unwrap_result = result.unwrap();
        assert!(!unwrap_result.par.exprs.is_empty());
        assert_eq!(unwrap_result.free_map, input.free_map);
    }

    #[test]
    fn new_ast_normalize_names_with_remainder() {
        use rholang_parser::ast::{AnnName, Id, Names as NewNames};
        use rholang_parser::{SourcePos, SourceSpan};

        // Create Names with variable "x" and remainder "rest"
        let ann_name = AnnName {
            name: NewName::ProcVar(NewVar::Id(Id {
                name: "x",
                pos: SourcePos { line: 1, col: 1 },
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let remainder_var = NewVar::Id(Id {
            name: "rest",
            pos: SourcePos { line: 1, col: 1 },
        });

        let mut names_vec = smallvec::SmallVec::new();
        names_vec.push(ann_name);

        let names = NewNames {
            names: names_vec,
            remainder: Some(remainder_var),
        };

        let (mut input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        input.bound_map_chain = input.bound_map_chain.put((
            "x".to_string(),
            VarSort::NameSort,
            SourcePosition::new(0, 0),
        ));

        let result = normalize_names_new_ast(&names, input.clone(), &env, &parser);
        assert!(result.is_ok());

        let unwrap_result = result.unwrap();
        assert!(!unwrap_result.par.exprs.is_empty());
        // Should have "rest" in free map
        assert!(unwrap_result.free_map.level_bindings.contains_key("rest"));
    }

    #[test]
    fn new_ast_normalize_names_wildcard_remainder() {
        use rholang_parser::ast::{AnnName, Id, Names as NewNames};
        use rholang_parser::{SourcePos, SourceSpan};

        // Create Names with variable "x" and wildcard remainder
        let ann_name = AnnName {
            name: NewName::ProcVar(NewVar::Id(Id {
                name: "x",
                pos: SourcePos { line: 1, col: 1 },
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let remainder_wildcard = NewVar::Wildcard;

        let mut names_vec = smallvec::SmallVec::new();
        names_vec.push(ann_name);

        let names = NewNames {
            names: names_vec,
            remainder: Some(remainder_wildcard),
        };

        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        let bound_inputs =
            bound_name_inputs_with_bound_map_chain(input.clone(), "x", VarSort::NameSort, 0, 0);

        let result = normalize_names_new_ast(&names, bound_inputs.clone(), &env, &parser);
        assert!(result.is_ok());

        let unwrap_result = result.unwrap();
        assert!(!unwrap_result.par.exprs.is_empty());
        // Wildcard remainder should be in wildcards list
        assert!(!unwrap_result.free_map.wildcards.is_empty());
    }

    #[test]
    fn new_ast_normalize_names_empty_list() {
        use rholang_parser::ast::{Id, Names as NewNames};
        use rholang_parser::SourcePos;

        // Create Names with empty list and remainder "rest"
        let remainder_var = NewVar::Id(Id {
            name: "rest",
            pos: SourcePos { line: 1, col: 1 },
        });

        let names = NewNames {
            names: smallvec::SmallVec::new(), // Empty names list
            remainder: Some(remainder_var),
        };

        let (input, env) = name_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();

        let result = normalize_names_new_ast(&names, input.clone(), &env, &parser	);
        assert!(result.is_ok());

        let unwrap_result = result.unwrap();
        // Should have "rest" in free map
        assert!(unwrap_result.free_map.level_bindings.contains_key("rest"));
    }
}
