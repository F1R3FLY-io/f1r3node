//! Normalizer for built-in function calls like getSpaceAgent(space).
//!
//! This module handles the normalization of function call expressions,
//! converting them from the parser AST to the protobuf-based normalized form.

use crate::rust::interpreter::compiler::exports::{ProcVisitInputs, ProcVisitOutputs};
use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::util::prepend_expr;
use models::rhoapi::{expr, EFunction, Expr, Par};
use models::rust::utils::union;
use std::collections::HashMap;

use rholang_parser::ast::Id;

/// Known built-in functions that are supported
const BUILTIN_FUNCTIONS: &[&str] = &["getSpaceAgent"];

/// Check if a name is a built-in function
pub fn is_builtin_function(name: &str) -> bool {
    BUILTIN_FUNCTIONS.contains(&name)
}

/// Normalize a function call expression.
///
/// This handles built-in function calls like `getSpaceAgent(space)`.
/// Unlike method calls, function calls do not have a receiver - they are
/// standalone expressions with just a function name and arguments.
pub fn normalize_p_function<'ast>(
    name_id: &'ast Id<'ast>,
    args: &'ast rholang_parser::ast::ProcList<'ast>,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let function_name = name_id.name;

    // Validate that this is a known built-in function
    if !is_builtin_function(function_name) {
        return Err(InterpreterError::NormalizerError(format!(
            "Unknown function: {}. Known functions: {:?}",
            function_name, BUILTIN_FUNCTIONS
        )));
    }

    // Process arguments
    let init_acc = (
        Vec::new(),
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: input.free_map.clone(),
        },
        Vec::new(),
        false,
    );

    let arg_results = args.iter().rev().try_fold(init_acc, |acc, arg| {
        normalize_ann_proc(arg, acc.1.clone(), env, parser).map(|proc_match_result| {
            (
                {
                    let mut acc_0 = acc.0.clone();
                    acc_0.insert(0, proc_match_result.par.clone());
                    acc_0
                },
                ProcVisitInputs {
                    par: Par::default(),
                    bound_map_chain: input.bound_map_chain.clone(),
                    free_map: proc_match_result.free_map.clone(),
                },
                union(acc.2.clone(), proc_match_result.par.locally_free.clone()),
                acc.3 || proc_match_result.par.connective_used,
            )
        })
    })?;

    let function = EFunction {
        function_name: function_name.to_string(),
        arguments: arg_results.0,
        locally_free: arg_results.2,
        connective_used: arg_results.3,
    };

    let updated_par = prepend_expr(
        input.par,
        Expr {
            expr_instance: Some(expr::ExprInstance::EFunctionBody(function)),
        },
        input.bound_map_chain.depth() as i32,
    );

    Ok(ProcVisitOutputs {
        par: updated_par,
        free_map: arg_results.1.free_map,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use models::{
        create_bit_vector,
        rhoapi::{expr::ExprInstance, EFunction, Expr, Par},
        rust::utils::new_boundvar_par,
    };

    use crate::rust::interpreter::{
        compiler::normalize::VarSort, test_utils::utils::proc_visit_inputs_and_env,
        util::prepend_expr,
    };

    #[test]
    fn test_is_builtin_function() {
        assert!(is_builtin_function("getSpaceAgent"));
        assert!(!is_builtin_function("unknownFunction"));
        assert!(!is_builtin_function(""));
    }

    #[test]
    fn p_function_should_produce_proper_function_call() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;
        use rholang_parser::ast::{Id, Var};
        use rholang_parser::SourcePos;

        let parser = rholang_parser::RholangParser::new();
        let (mut inputs, env) = proc_visit_inputs_and_env();
        inputs.bound_map_chain = inputs.bound_map_chain.put_pos((
            "space".to_string(),
            VarSort::ProcSort,
            SourcePos { line: 0, col: 0 },
        ));

        // Create argument: space (ProcVar)
        let arg = ParBuilderUtil::create_ast_proc_var_from_var(
            Var::Id(Id {
                name: "space",
                pos: SourcePos { line: 0, col: 0 },
            }),
            &parser,
        );

        // Create function name
        let function_id = Id {
            name: "getSpaceAgent",
            pos: SourcePos { line: 0, col: 0 },
        };

        // Create function call
        let function_call =
            ParBuilderUtil::create_ast_function_call(function_id, vec![arg], &parser);

        let result = normalize_ann_proc(&function_call, inputs.clone(), &env, &parser);
        assert!(result.is_ok());

        let expected_result = prepend_expr(
            Par::default(),
            Expr {
                expr_instance: Some(ExprInstance::EFunctionBody(EFunction {
                    function_name: "getSpaceAgent".to_string(),
                    arguments: vec![new_boundvar_par(0, create_bit_vector(&vec![0]), false)],
                    locally_free: create_bit_vector(&vec![0]),
                    connective_used: false,
                })),
            },
            0,
        );

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }

    #[test]
    fn p_function_should_reject_unknown_functions() {
        use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;
        use rholang_parser::ast::Id;
        use rholang_parser::SourcePos;

        let parser = rholang_parser::RholangParser::new();
        let (inputs, env) = proc_visit_inputs_and_env();

        // Create function name for unknown function
        let function_id = Id {
            name: "unknownFunction",
            pos: SourcePos { line: 0, col: 0 },
        };

        // Create function call with no args
        let _function_call =
            ParBuilderUtil::create_ast_function_call(function_id, vec![], &parser);

        let result = normalize_p_function(
            &function_id,
            &smallvec::smallvec![],
            inputs,
            &env,
            &parser,
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, InterpreterError::NormalizerError(_)));
    }
}
