use crate::rust::interpreter::{
    compiler::exports::{ProcVisitInputsSpan, ProcVisitOutputsSpan},
    util::prepend_expr,
};

use super::exports::*;

// New AST imports for parallel functions
use crate::rust::interpreter::compiler::normalizer::ground_normalize_matcher::normalize_ground_new_ast;
use rholang_parser::ast::Proc as NewProc;

pub fn normalize_p_ground(
    proc: &Proc,
    input: ProcVisitInputs,
) -> Result<ProcVisitOutputs, InterpreterError> {
    normalize_ground(proc).map(|expr| {
        let new_par = prepend_expr(
            input.par.clone(),
            expr,
            input.bound_map_chain.depth() as i32,
        );
        ProcVisitOutputs {
            par: new_par,
            free_map: input.free_map.clone(),
        }
    })
}

/// Parallel normalizer for new AST ground types from rholang-rs parser
/// This preserves the exact same logic as normalize_p_ground but works directly with new AST
pub fn normalize_p_ground_new_ast<'ast>(
    proc: &NewProc<'ast>,
    input: ProcVisitInputsSpan,
) -> Result<ProcVisitOutputsSpan, InterpreterError> {
    normalize_ground_new_ast(proc).map(|expr| {
        let new_par = prepend_expr(
            input.par.clone(),
            expr,
            input.bound_map_chain.depth() as i32,
        );
        ProcVisitOutputsSpan {
            par: new_par,
            free_map: input.free_map.clone(),
        }
    })
}
