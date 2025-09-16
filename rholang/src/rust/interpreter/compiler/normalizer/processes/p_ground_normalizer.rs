use crate::rust::interpreter::{
    compiler::exports::{ProcVisitInputsSpan, ProcVisitOutputsSpan},
    util::prepend_expr,
};

use super::exports::*;
use crate::rust::interpreter::compiler::normalizer::ground_normalize_matcher::normalize_ground_new_ast;

use rholang_parser::ast::Proc;

pub fn normalize_p_ground_new_ast<'ast>(
    proc: &Proc<'ast>,
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
