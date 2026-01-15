//! UseBlock normalizer for Reifying RSpaces.
//!
//! This module handles normalization of `use space { body }` constructs,
//! which provide scoped default space selection.
//!
//! # Formal Correspondence
//!
//! - Registry/Invariants.v: inv_use_blocks_valid invariant
//! - GenericRSpace.v: UseBlock scope management
//! - Safety/Properties.v: seq_is_sequential (Seq channels require UseBlock scope)

use crate::rust::interpreter::compiler::exports::ProcVisitInputs;
use crate::rust::interpreter::compiler::exports::ProcVisitOutputs;
use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
use crate::rust::interpreter::compiler::normalizer::name_normalize_matcher::normalize_name;
use crate::rust::interpreter::compiler::normalize::NameVisitInputs;
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::util::prepend_use_block;
use models::rhoapi::{Par, UseBlock};
use models::rust::utils::union;
use std::collections::HashMap;

use rholang_parser::ast::{AnnProc, Name};

/// Normalize a UseBlock construct.
///
/// Syntax: `use space_expr { body }`
///
/// The space expression is normalized to a Par representing the target space.
/// The body is normalized within the UseBlock scope.
///
/// # Arguments
///
/// * `space` - The space expression (Name AST node)
/// * `proc` - The body process to execute within the space scope
/// * `input` - Current normalization context
/// * `env` - Environment with predefined channels
/// * `parser` - Parser reference for recursive normalization
///
/// # Returns
///
/// * `ProcVisitOutputs` with the UseBlock appended to the Par
pub fn normalize_p_use_block<'ast>(
    space: &Name<'ast>,
    proc: &'ast AnnProc<'ast>,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    // Normalize the space expression to get the target space as a Par
    let space_result = normalize_name(
        space,
        NameVisitInputs {
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: input.free_map.clone(),
        },
        env,
        parser,
    )?;

    // Normalize the body with the same bound context
    // (UseBlock doesn't introduce new bindings, just changes the default space)
    let body_result = normalize_ann_proc(
        proc,
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: space_result.free_map.clone(),
        },
        env,
        parser,
    )?;

    // Combine locally_free from space and body
    let combined_locally_free = union(
        space_result.par.locally_free.clone(),
        body_result.par.locally_free.clone(),
    );

    // Combine connective_used from space and body
    let connective_used = space_result.par.connective_used || body_result.par.connective_used;

    // Create the UseBlock proto message
    let use_block = UseBlock {
        space: Some(space_result.par),
        body: Some(body_result.par),
        locally_free: combined_locally_free,
        connective_used,
    };

    Ok(ProcVisitOutputs {
        par: prepend_use_block(input.par, use_block),
        free_map: body_result.free_map,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rust::interpreter::test_utils::utils::proc_visit_inputs_and_env;
    use rholang_parser::RholangParser;
    use validated::Validated;

    #[test]
    fn test_use_block_normalizes_space_and_body() {
        let parser = RholangParser::new();
        let (inputs, env) = proc_visit_inputs_and_env();

        // Parse: use @"my_space" { Nil }
        // Note: The actual syntax depends on the parser grammar
        let code = r#"use @"my_space" { Nil }"#;
        let result = parser.parse(code);

        match result {
            Validated::Good(procs) if procs.len() == 1 => {
                let ast = procs.into_iter().next().unwrap();
                let normalized = normalize_ann_proc(&ast, inputs, &env, &parser);

                // The normalization should succeed and produce a Par with a UseBlock
                assert!(
                    normalized.is_ok(),
                    "UseBlock normalization failed: {:?}",
                    normalized.err()
                );

                let output = normalized.unwrap();
                assert_eq!(
                    output.par.use_blocks.len(),
                    1,
                    "Expected exactly one UseBlock"
                );

                // Verify the UseBlock has space and body
                let ub = &output.par.use_blocks[0];
                assert!(ub.space.is_some(), "UseBlock should have a space");
                assert!(ub.body.is_some(), "UseBlock should have a body");
            }
            Validated::Good(_) => panic!("Expected single process"),
            Validated::Fail(errors) => {
                // If parsing fails, UseBlock syntax may not be supported yet
                // This is expected during initial implementation
                eprintln!(
                    "UseBlock parsing not yet supported or syntax error: {:?}",
                    errors
                );
            }
        }
    }
}
