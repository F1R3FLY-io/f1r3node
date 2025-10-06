use models::rust::utils::{new_boundvar_expr, new_freevar_expr, new_wildcard_expr};

use super::exports::*;
use crate::rust::interpreter::compiler::exports::{BoundContext, FreeContext};
use crate::rust::interpreter::compiler::exports::{ProcVisitInputs, ProcVisitOutputs};
use crate::rust::interpreter::compiler::normalize::VarSort;
use crate::rust::interpreter::compiler::span_utils::SpanContext;
use std::result::Result;

use rholang_parser::ast::Var;

pub fn normalize_p_var<'ast>(
    var: &Var,
    input: ProcVisitInputs,
    var_span: rholang_parser::SourceSpan,
) -> Result<ProcVisitOutputs, InterpreterError> {
    match var {
        Var::Id(id) => {
            let var_name = id.name;

            match input.bound_map_chain.get(var_name) {
                Some(BoundContext {
                    index,
                    typ,
                    source_span,
                }) => match typ {
                    VarSort::ProcSort => Ok(ProcVisitOutputs {
                        par: prepend_expr(
                            input.par,
                            new_boundvar_expr(index as i32),
                            input.bound_map_chain.depth() as i32,
                        ),
                        free_map: input.free_map,
                    }),
                    VarSort::NameSort => Err(InterpreterError::UnexpectedProcContextSpan {
                        var_name: var_name.to_string(),
                        name_var_source_span: source_span,
                        process_source_span: var_span,
                    }),
                },

                None => match input.free_map.get(var_name) {
                    Some(FreeContext { source_span, .. }) => {
                        Err(InterpreterError::UnexpectedReuseOfProcContextFreeSpan {
                            var_name: var_name.to_string(),
                            first_use: source_span,
                            second_use: var_span,
                        })
                    }

                    None => {
                        let new_bindings_pair = input.free_map.put_span((
                            var_name.to_string(),
                            VarSort::ProcSort,
                            var_span,
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

        Var::Wildcard => {
            // Use wildcard span for context
            let wildcard_span = SpanContext::wildcard_span();

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
                free_map: input.free_map.add_wildcard(wildcard_span),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use models::create_bit_vector;

    use super::normalize_p_var;
    use crate::rust::interpreter::test_utils::utils::proc_visit_inputs_and_env;
    use rholang_parser::ast::{Id, Var};
    use rholang_parser::{SourcePos, SourceSpan};

    fn inputs_span() -> ProcVisitInputs {
        let (inputs_data, _env) = proc_visit_inputs_and_env();
        inputs_data
    }

    fn create_new_ast_id_var(name: &'static str) -> Var<'static> {
        Var::Id(Id {
            name,
            pos: SourcePos { line: 1, col: 1 },
        })
    }

    fn create_new_ast_wildcard_var() -> Var<'static> {
        Var::Wildcard
    }

    #[test]
    fn p_var_should_compile_as_bound_var_if_its_in_env() {
        let new_var = create_new_ast_id_var("x");
        let test_span = SourceSpan {
            start: SourcePos { line: 1, col: 1 },
            end: SourcePos { line: 1, col: 2 },
        };

        let bound_inputs = {
            let mut inputs = inputs_span();
            inputs.bound_map_chain =
                inputs
                    .bound_map_chain
                    .put_span(("x".to_string(), VarSort::ProcSort, test_span));
            inputs
        };

        let result = normalize_p_var(&new_var, bound_inputs, test_span);
        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            prepend_expr(inputs_span().par, new_boundvar_expr(0), 0)
        );

        assert_eq!(result.clone().unwrap().free_map, inputs_span().free_map);
        assert_eq!(
            result.unwrap().par.locally_free,
            create_bit_vector(&vec![0])
        );
    }

    #[test]
    fn p_var_should_compile_as_free_var_if_its_not_in_env() {
        let new_var = create_new_ast_id_var("x");
        let test_span = SourceSpan {
            start: SourcePos { line: 1, col: 1 },
            end: SourcePos { line: 1, col: 2 },
        };
        let test_inputs = inputs_span();

        let result = normalize_p_var(&new_var, test_inputs, test_span);
        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            prepend_expr(inputs_span().par, new_freevar_expr(0), 0)
        );

        assert_eq!(
            result.clone().unwrap().free_map,
            inputs_span()
                .free_map
                .put_span(("x".to_string(), VarSort::ProcSort, test_span,))
        );
    }

    #[test]
    fn new_ast_p_var_should_not_compile_if_its_in_env_of_the_wrong_sort() {
        let new_var = create_new_ast_id_var("x");
        let bound_span = SourceSpan {
            start: SourcePos { line: 0, col: 0 },
            end: SourcePos { line: 0, col: 1 },
        };
        let process_span = SourceSpan {
            start: SourcePos { line: 1, col: 1 },
            end: SourcePos { line: 1, col: 2 },
        };

        let bound_inputs = {
            let mut inputs = inputs_span();
            inputs.bound_map_chain =
                inputs
                    .bound_map_chain
                    .put_span(("x".to_string(), VarSort::NameSort, bound_span));
            inputs
        };

        let result = normalize_p_var(&new_var, bound_inputs, process_span);
        assert!(result.is_err());
        assert_eq!(
            result,
            Err(InterpreterError::UnexpectedProcContextSpan {
                var_name: "x".to_string(),
                name_var_source_span: bound_span,
                process_source_span: process_span,
            })
        )
    }

    #[test]
    fn new_ast_p_var_should_not_compile_if_its_used_free_somewhere_else() {
        let new_var = create_new_ast_id_var("x");
        let first_use_span = SourceSpan {
            start: SourcePos { line: 0, col: 0 },
            end: SourcePos { line: 0, col: 1 },
        };
        let second_use_span = SourceSpan {
            start: SourcePos { line: 1, col: 1 },
            end: SourcePos { line: 1, col: 2 },
        };

        let bound_inputs = {
            let mut inputs = inputs_span();
            inputs.free_map =
                inputs
                    .free_map
                    .put_span(("x".to_string(), VarSort::ProcSort, first_use_span));
            inputs
        };

        let result = normalize_p_var(&new_var, bound_inputs, second_use_span);
        assert!(result.is_err());
        assert_eq!(
            result,
            Err(InterpreterError::UnexpectedReuseOfProcContextFreeSpan {
                var_name: "x".to_string(),
                first_use: first_use_span,
                second_use: second_use_span,
            })
        )
    }

    #[test]
    fn new_ast_p_var_should_handle_wildcard() {
        let wildcard_var = create_new_ast_wildcard_var();
        let test_span = SourceSpan {
            start: SourcePos { line: 1, col: 1 },
            end: SourcePos { line: 1, col: 2 },
        };

        let result = normalize_p_var(&wildcard_var, inputs_span(), test_span);
        assert!(result.is_ok());

        let unwrap_result = result.unwrap();
        assert!(unwrap_result.par.connective_used);
        assert!(!unwrap_result.par.exprs.is_empty());

        assert_eq!(unwrap_result.free_map.wildcards.len(), 1);
    }
}
