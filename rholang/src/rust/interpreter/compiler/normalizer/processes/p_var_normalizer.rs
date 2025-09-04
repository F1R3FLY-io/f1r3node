use models::rust::utils::{new_boundvar_expr, new_freevar_expr, new_wildcard_expr};

use crate::rust::interpreter::compiler::exports::{BoundContext, FreeContext};
use crate::rust::interpreter::compiler::normalize::VarSort;
use crate::rust::interpreter::compiler::span_utils::SpanContext;

use super::exports::*;
use std::result::Result;

// New AST imports for parallel functions
use rholang_parser::ast::Var as NewVar;

pub fn normalize_p_var(
    p: &Proc,
    input: ProcVisitInputs,
) -> Result<ProcVisitOutputs, InterpreterError> {
    match p {
        Proc::Var(var) => {
            let var_name = var.name.clone();
            let row = var.line_num;
            let column = var.col_num;

            match input.bound_map_chain.get(&var_name) {
                Some(BoundContext {
                    index,
                    typ,
                    source_position,
                }) => match typ {
                    VarSort::ProcSort => Ok(ProcVisitOutputs {
                        par: prepend_expr(
                            input.par,
                            new_boundvar_expr(index as i32),
                            input.bound_map_chain.depth() as i32,
                        ),
                        free_map: input.free_map,
                    }),
                    VarSort::NameSort => Err(InterpreterError::UnexpectedProcContext {
                        var_name,
                        name_var_source_position: source_position,
                        process_source_position: SourcePosition { row, column },
                    }),
                },

                None => match input.free_map.get(&var_name) {
                    Some(FreeContext {
                        source_position, ..
                    }) => Err(InterpreterError::UnexpectedReuseOfProcContextFree {
                        var_name,
                        first_use: source_position,
                        second_use: SourcePosition { row, column },
                    }),

                    None => {
                        let new_bindings_pair = input.free_map.put((
                            var_name,
                            VarSort::ProcSort,
                            SourcePosition { row, column },
                        ));

                        Ok(ProcVisitOutputs {
                            par: prepend_expr(
                                input.par,
                                new_freevar_expr(input.free_map.next_level as i32),
                                input.bound_map_chain.depth() as i32,
                            ),
                            free_map: new_bindings_pair,
                        })
                    }
                },
            }
        }

        Proc::Wildcard { line_num, col_num } => Ok(ProcVisitOutputs {
            par: {
                let mut par = prepend_expr(
                    input.par,
                    new_wildcard_expr(),
                    input.bound_map_chain.depth() as i32,
                );
                par.connective_used = true;
                par
            },
            free_map: input.free_map.add_wildcard(SourcePosition {
                row: *line_num,
                column: *col_num,
            }),
        }),

        _ => Err(InterpreterError::NormalizerError(format!(
            "Expected Proc::Var or Proc::Wildcard, found {:?}",
            p,
        ))),
    }
}

// ============================================================================
// NEW AST PARALLEL FUNCTIONS
// ============================================================================

/// Parallel version of normalize_p_var for new AST Var types
/// Handles Var<'ast> enum (Wildcard | Id) instead of separate Proc variants
pub fn normalize_p_var_new_ast(
    var: &NewVar,
    input: ProcVisitInputs,
) -> Result<ProcVisitOutputs, InterpreterError> {
    match var {
        NewVar::Id(id) => {
            let var_name = id.name;
            // Use span context for accurate error reporting
            let error_position = SpanContext::to_legacy_source_position(input.source_span);

            match input.bound_map_chain.get(var_name) {
                Some(BoundContext {
                    index,
                    typ,
                    source_position,
                }) => match typ {
                    VarSort::ProcSort => Ok(ProcVisitOutputs {
                        par: prepend_expr(
                            input.par,
                            new_boundvar_expr(index as i32),
                            input.bound_map_chain.depth() as i32,
                        ),
                        free_map: input.free_map,
                    }),
                    VarSort::NameSort => Err(InterpreterError::UnexpectedProcContext {
                        var_name: var_name.to_string(),
                        name_var_source_position: source_position.clone(),
                        process_source_position: error_position,
                    }),
                },

                None => match input.free_map.get(var_name) {
                    Some(FreeContext {
                        source_position, ..
                    }) => Err(InterpreterError::UnexpectedReuseOfProcContextFree {
                        var_name: var_name.to_string(),
                        first_use: source_position.clone(),
                        second_use: error_position,
                    }),

                    None => {
                        let new_bindings_pair = input.free_map.put((
                            var_name.to_string(),
                            VarSort::ProcSort,
                            error_position,
                        ));

                        Ok(ProcVisitOutputs {
                            par: prepend_expr(
                                input.par,
                                new_freevar_expr(input.free_map.next_level as i32),
                                input.bound_map_chain.depth() as i32,
                            ),
                            free_map: new_bindings_pair,
                        })
                    }
                },
            }
        }

        NewVar::Wildcard => {
            // TODO: Extract source position from context when SourceSpan migration is complete
            // For now, use placeholder position
            Ok(ProcVisitOutputs {
                par: {
                    let mut par = prepend_expr(
                        input.par,
                        new_wildcard_expr(),
                        input.bound_map_chain.depth() as i32,
                    );
                    par.connective_used = true;
                    par
                },
                free_map: input.free_map.add_wildcard(SourcePosition {
                    row: 0,
                    column: 0,
                }),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::rust::interpreter::compiler::exports::BoundMapChain;
    use crate::rust::interpreter::compiler::rholang_ast::Var;

    use super::*;
    use models::create_bit_vector;
    use models::rhoapi::Par;

    // New AST test imports
    use super::normalize_p_var_new_ast;
    use rholang_parser::ast::{Id, Var as NewVar};
    use rholang_parser::{SourcePos};

    fn inputs() -> ProcVisitInputs {
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: BoundMapChain::new(),
            free_map: FreeMap::new(),
            source_span: SpanContext::zero_span(),
        }
    }

    fn p_var() -> Proc {
        Proc::Var(Var {
            name: "x".to_string(),
            line_num: 0,
            col_num: 0,
        })
    }

    #[test]
    fn p_var_should_compile_as_bound_var_if_its_in_env() {
        let bound_inputs = {
            let mut inputs = inputs();
            inputs.bound_map_chain = inputs.bound_map_chain.put((
                "x".to_string(),
                VarSort::ProcSort,
                SourcePosition::new(0, 0),
            ));
            inputs
        };

        let result = normalize_p_var(&p_var(), bound_inputs);
        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            prepend_expr(inputs().par, new_boundvar_expr(0), 0)
        );

        assert_eq!(result.clone().unwrap().free_map, inputs().free_map);
        assert_eq!(
            result.unwrap().par.locally_free,
            create_bit_vector(&vec![0])
        );
    }

    #[test]
    fn p_var_should_compile_as_free_var_if_its_not_in_env() {
        let result = normalize_p_var(&p_var(), inputs());
        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            prepend_expr(inputs().par, new_freevar_expr(0), 0)
        );

        assert_eq!(
            result.clone().unwrap().free_map,
            inputs().free_map.put((
                "x".to_string(),
                VarSort::ProcSort,
                SourcePosition::new(0, 0)
            ))
        );
    }

    #[test]
    fn p_var_should_not_compile_if_its_in_env_of_the_wrong_sort() {
        let bound_inputs = {
            let mut inputs = inputs();
            inputs.bound_map_chain = inputs.bound_map_chain.put((
                "x".to_string(),
                VarSort::NameSort,
                SourcePosition::new(0, 0),
            ));
            inputs
        };

        let result = normalize_p_var(&p_var(), bound_inputs);
        assert!(result.is_err());
        assert_eq!(
            result,
            Err(InterpreterError::UnexpectedProcContext {
                var_name: "x".to_string(),
                name_var_source_position: SourcePosition::new(0, 0),
                process_source_position: SourcePosition::new(0, 0),
            })
        )
    }

    #[test]
    fn p_var_should_not_compile_if_its_used_free_somewhere_else() {
        let bound_inputs = {
            let mut inputs = inputs();
            inputs.free_map = inputs.free_map.put((
                "x".to_string(),
                VarSort::ProcSort,
                SourcePosition::new(0, 0),
            ));
            inputs
        };

        let result = normalize_p_var(&p_var(), bound_inputs);
        assert!(result.is_err());
        assert_eq!(
            result,
            Err(InterpreterError::UnexpectedReuseOfProcContextFree {
                var_name: "x".to_string(),
                first_use: SourcePosition::new(0, 0),
                second_use: SourcePosition::new(0, 0)
            })
        )
    }

    // ============================================================================
    // NEW AST PARALLEL TESTS - EXACT MAPPING TO ORIGINAL TESTS
    // ============================================================================
    //
    // Original tests (4 total):
    // 1. p_var_should_compile_as_bound_var_if_its_in_env → ✅ IMPLEMENTED
    // 2. p_var_should_compile_as_free_var_if_its_not_in_env → ✅ IMPLEMENTED  
    // 3. p_var_should_not_compile_if_its_in_env_of_the_wrong_sort → ✅ IMPLEMENTED
    // 4. p_var_should_not_compile_if_its_used_free_somewhere_else → ✅ IMPLEMENTED
    // 5. NEW: wildcard test → ✅ IMPLEMENTED
    //
    // NOTE: These tests directly test the new AST Var enum variants

    /// Helper function to create a new AST Id variable
    fn create_new_ast_id_var(name: &'static str) -> NewVar<'static> {
        NewVar::Id(Id {
            name,
            pos: SourcePos { line: 1, col: 1 },
        })
    }

    /// Helper function to create a new AST wildcard variable
    fn create_new_ast_wildcard_var() -> NewVar<'static> {
        NewVar::Wildcard
    }

    #[test]
    fn new_ast_p_var_should_compile_as_bound_var_if_its_in_env() {
        let new_var = create_new_ast_id_var("x");

        let bound_inputs = {
            let mut inputs = inputs();
            inputs.bound_map_chain = inputs.bound_map_chain.put((
                "x".to_string(),
                VarSort::ProcSort,
                SourcePosition::new(0, 0),
            ));
            inputs
        };

        let result = normalize_p_var_new_ast(&new_var, bound_inputs);
        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            prepend_expr(inputs().par, new_boundvar_expr(0), 0)
        );

        assert_eq!(result.clone().unwrap().free_map, inputs().free_map);
        assert_eq!(
            result.unwrap().par.locally_free,
            create_bit_vector(&vec![0])
        );
    }

    #[test]
    fn new_ast_p_var_should_compile_as_free_var_if_its_not_in_env() {
        let new_var = create_new_ast_id_var("x");
        
        // Create inputs with the correct source span from the variable
        let var_span = match &new_var {
            NewVar::Id(id) => rholang_parser::SourceSpan {
                start: id.pos,
                end: id.pos,
            },
            NewVar::Wildcard => SpanContext::zero_span(),
        };
        let mut test_inputs = inputs();
        test_inputs.source_span = var_span;

        let result = normalize_p_var_new_ast(&new_var, test_inputs);
        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            prepend_expr(inputs().par, new_freevar_expr(0), 0)
        );

        assert_eq!(
            result.clone().unwrap().free_map,
            inputs().free_map.put((
                "x".to_string(),
                VarSort::ProcSort,
                SourcePosition::new(1, 1) // Note: Uses new AST source position
            ))
        );
    }

    #[test]
    fn new_ast_p_var_should_not_compile_if_its_in_env_of_the_wrong_sort() {
        let new_var = create_new_ast_id_var("x");

        let bound_inputs = {
            let mut inputs = inputs();
            // Set the correct source span from the variable
            let var_span = match &new_var {
                NewVar::Id(id) => rholang_parser::SourceSpan {
                    start: id.pos,
                    end: id.pos,
                },
                NewVar::Wildcard => SpanContext::zero_span(),
            };
            inputs.source_span = var_span;
            inputs.bound_map_chain = inputs.bound_map_chain.put((
                "x".to_string(),
                VarSort::NameSort,
                SourcePosition::new(0, 0),
            ));
            inputs
        };

        let result = normalize_p_var_new_ast(&new_var, bound_inputs);
        assert!(result.is_err());
        assert_eq!(
            result,
            Err(InterpreterError::UnexpectedProcContext {
                var_name: "x".to_string(),
                name_var_source_position: SourcePosition::new(0, 0),
                process_source_position: SourcePosition::new(1, 1), // Note: Uses new AST source position
            })
        )
    }

    #[test]
    fn new_ast_p_var_should_not_compile_if_its_used_free_somewhere_else() {
        let new_var = create_new_ast_id_var("x");

        let bound_inputs = {
            let mut inputs = inputs();
            // Set the correct source span from the variable
            let var_span = match &new_var {
                NewVar::Id(id) => rholang_parser::SourceSpan {
                    start: id.pos,
                    end: id.pos,
                },
                NewVar::Wildcard => SpanContext::zero_span(),
            };
            inputs.source_span = var_span;
            inputs.free_map = inputs.free_map.put((
                "x".to_string(),
                VarSort::ProcSort,
                SourcePosition::new(0, 0),
            ));
            inputs
        };

        let result = normalize_p_var_new_ast(&new_var, bound_inputs);
        assert!(result.is_err());
        assert_eq!(
            result,
            Err(InterpreterError::UnexpectedReuseOfProcContextFree {
                var_name: "x".to_string(),
                first_use: SourcePosition::new(0, 0),
                second_use: SourcePosition::new(1, 1) // Note: Uses new AST source position
            })
        )
    }

    #[test]
    fn new_ast_p_var_should_handle_wildcard() {
        let wildcard_var = create_new_ast_wildcard_var();

        let result = normalize_p_var_new_ast(&wildcard_var, inputs());
        assert!(result.is_ok());

        // Wildcard should create a wildcard expression and mark connective_used as true
        let unwrap_result = result.unwrap();
        assert!(unwrap_result.par.connective_used);
        assert!(!unwrap_result.par.exprs.is_empty());
        
        // Free map should have one wildcard added
        assert_eq!(unwrap_result.free_map.wildcards.len(), 1);
    }
}
