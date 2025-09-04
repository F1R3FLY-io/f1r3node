use crate::rust::interpreter::compiler::normalize::{
    CollectVisitInputs, ProcVisitInputs, ProcVisitOutputs,
};
use crate::rust::interpreter::compiler::normalizer::collection_normalize_matcher::{normalize_collection, normalize_collection_new_ast};
use crate::rust::interpreter::compiler::rholang_ast::Collection;
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::util::prepend_expr;
use models::rhoapi::Par;
use rholang_parser::ast::Collection as NewCollection;
use std::collections::HashMap;

pub fn normalize_p_collect(
    proc: &Collection,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let collection_result = normalize_collection(
        proc,
        CollectVisitInputs {
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: input.free_map.clone(),
            source_span: input.source_span,
        },
        env,
    )?;

    let updated_par = prepend_expr(
        input.par,
        collection_result.expr,
        input.bound_map_chain.depth() as i32,
    );

    Ok(ProcVisitOutputs {
        par: updated_par,
        free_map: collection_result.free_map,
    })
}

/// Parallel version of normalize_p_collect for new AST Collection
pub fn normalize_p_collect_new_ast<'ast>(
    proc: &'ast NewCollection<'ast>,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let collection_result = normalize_collection_new_ast(
        proc,
        CollectVisitInputs {
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: input.free_map.clone(),
            source_span: input.source_span,
        },
        env,
        parser,
    )?;

    let updated_par = prepend_expr(
        input.par,
        collection_result.expr,
        input.bound_map_chain.depth() as i32,
    );

    Ok(ProcVisitOutputs {
        par: updated_par,
        free_map: collection_result.free_map,
    })
}
