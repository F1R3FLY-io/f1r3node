use super::exports::*;
use crate::rust::interpreter::compiler::exports::{
    NameVisitInputsSpan, ProcVisitInputsSpan, ProcVisitOutputsSpan,
};
use crate::rust::interpreter::compiler::normalize::{
    NameVisitInputs, ProcVisitInputs, ProcVisitOutputs,
};
use crate::rust::interpreter::compiler::rholang_ast::Eval;
use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::Par;
use std::collections::HashMap;

// New AST imports for parallel functions
use crate::rust::interpreter::compiler::normalizer::name_normalize_matcher::normalize_name_new_ast;
use rholang_parser::ast::AnnName;

pub fn normalize_p_eval(
    proc: &Eval,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let name_match_result = normalize_name(
        &proc.name,
        NameVisitInputs {
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: input.free_map.clone(),
        },
        env,
    )?;

    let updated_par = input.par.append(name_match_result.par.clone());

    Ok(ProcVisitOutputs {
        par: updated_par,
        free_map: name_match_result.free_map,
    })
}

// ============================================================================
// NEW AST PARALLEL FUNCTIONS
// ============================================================================

/// Parallel version of normalize_p_eval for new AST Eval structure
/// Handles AnnName<'ast> instead of old Name enum
pub fn normalize_p_eval_new_ast<'ast>(
    eval_name: &AnnName<'ast>,
    input: ProcVisitInputsSpan,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputsSpan, InterpreterError> {
    let name_match_result = normalize_name_new_ast(
        &eval_name.name,
        NameVisitInputsSpan {
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: input.free_map.clone(),
        },
        env,
        parser,
    )?;

    let updated_par = input.par.append(name_match_result.par.clone());

    Ok(ProcVisitOutputsSpan {
        par: updated_par,
        free_map: name_match_result.free_map,
    })
}

// See rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use models::rust::utils::new_boundvar_expr;

    use crate::rust::interpreter::{
        compiler::{
            normalize::{normalize_match_proc, VarSort},
            rholang_ast::{Eval, Name, Quote},
        },
        test_utils::utils::{proc_visit_inputs_and_env, proc_visit_inputs_and_env_span},
        util::prepend_expr,
    };

    use super::{Proc, SourcePosition};

    // New AST test imports
    use super::normalize_p_eval_new_ast;
    use rholang_parser::ast::{AnnName, AnnProc, Id, Name as NewName, Var as NewVar};
    use rholang_parser::{SourcePos, SourceSpan};

    #[test]
    fn p_eval_should_handle_a_bound_name_variable() {
        let p_eval = Proc::Eval(Eval {
            name: Name::new_name_var("x"),
            line_num: 0,
            col_num: 0,
        });

        let (mut inputs, env) = proc_visit_inputs_and_env();
        inputs.bound_map_chain = inputs.bound_map_chain.put((
            "x".to_string(),
            VarSort::NameSort,
            SourcePosition::new(0, 0),
        ));

        let result = normalize_match_proc(&p_eval, inputs.clone(), &env);
        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            prepend_expr(inputs.par, new_boundvar_expr(0), 0)
        );
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }

    #[test]
    fn p_eval_should_collapse_a_quote() {
        let p_eval = Proc::Eval(Eval {
            name: Name::Quote(Box::new(Quote {
                quotable: Box::new(Proc::Par {
                    left: Box::new(Proc::new_proc_var("x")),
                    right: Box::new(Proc::new_proc_var("x")),
                    line_num: 0,
                    col_num: 0,
                }),
                line_num: 0,
                col_num: 0,
            })),
            line_num: 0,
            col_num: 0,
        });

        let (mut inputs, env) = proc_visit_inputs_and_env();
        inputs.bound_map_chain = inputs.bound_map_chain.put((
            "x".to_string(),
            VarSort::ProcSort,
            SourcePosition::new(0, 0),
        ));

        let result = normalize_match_proc(&p_eval, inputs.clone(), &env);
        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            prepend_expr(
                prepend_expr(inputs.par, new_boundvar_expr(0), 0),
                new_boundvar_expr(0),
                0
            )
        );
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }

    // ============================================================================
    // NEW AST PARALLEL TESTS - EXACT MAPPING TO ORIGINAL TESTS
    // ============================================================================
    //
    // NOTE: These tests use Box::leak() to create static references for test data
    // This is acceptable in test contexts where memory cleanup isn't critical

    /// Helper function to create a new AST AnnName with Id variable
    fn create_new_ast_ann_name_id<'ast>(name: &'ast str) -> AnnName<'ast> {
        AnnName {
            name: NewName::ProcVar(NewVar::Id(Id {
                name,
                pos: SourcePos { line: 1, col: 1 },
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        }
    }

    #[test]
    fn new_ast_p_eval_should_handle_a_bound_name_variable() {
        let eval_name = create_new_ast_ann_name_id("x");
        let parser = rholang_parser::RholangParser::new();
        let (mut inputs, env) = proc_visit_inputs_and_env_span();
        inputs.bound_map_chain = inputs.bound_map_chain.put_pos((
            "x".to_string(),
            VarSort::NameSort,
            SourcePos { line: 0, col: 0 },
        ));

        let result = normalize_p_eval_new_ast(&eval_name, inputs.clone(), &env, &parser);
        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            prepend_expr(inputs.par, new_boundvar_expr(0), 0)
        );
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }

    #[test]
    fn new_ast_p_eval_should_collapse_a_simple_quote() {
        use rholang_parser::ast::Proc as NewProc;

        // Create a quote with Par { left: ProcVar("x"), right: ProcVar("x") }
        // This matches the original test exactly: @{x | x} where x is bound as ProcSort
        let left_var = AnnProc {
            proc: Box::leak(Box::new(NewProc::ProcVar(NewVar::Id(Id {
                name: "x",
                pos: SourcePos { line: 1, col: 1 },
            })))),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let right_var = AnnProc {
            proc: Box::leak(Box::new(NewProc::ProcVar(NewVar::Id(Id {
                name: "x",
                pos: SourcePos { line: 1, col: 1 },
            })))),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let quoted_proc = Box::leak(Box::new(NewProc::Par {
            left: left_var,
            right: right_var,
        }));

        let quote_name = AnnName {
            name: NewName::Quote(quoted_proc),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let (mut inputs, env) = proc_visit_inputs_and_env_span();
        inputs.bound_map_chain = inputs.bound_map_chain.put_pos((
            "x".to_string(),
            VarSort::ProcSort,
            SourcePos { line: 0, col: 0 },
        ));

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_p_eval_new_ast(&quote_name, inputs.clone(), &env, &parser);
        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            prepend_expr(
                prepend_expr(inputs.par, new_boundvar_expr(0), 0),
                new_boundvar_expr(0),
                0
            )
        );
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }
}
