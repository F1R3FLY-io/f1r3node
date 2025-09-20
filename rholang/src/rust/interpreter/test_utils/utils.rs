use crate::rust::interpreter::compiler::exports::{
    BoundMapChainSpan, FreeMapSpan, IdContextPos, NameVisitInputsSpan, ProcVisitInputsSpan,
};
use crate::rust::interpreter::compiler::normalize::VarSort;
use crate::rust::interpreter::compiler::normalize::VarSort::{NameSort, ProcSort};
use models::rhoapi::Par;
use std::collections::HashMap;

use rholang_parser::SourcePos;

pub fn name_visit_inputs_and_env_span() -> (NameVisitInputsSpan, HashMap<String, Par>) {
    let input: NameVisitInputsSpan = NameVisitInputsSpan {
        bound_map_chain: BoundMapChainSpan::default(),
        free_map: FreeMapSpan::default(),
    };
    let env: HashMap<String, Par> = HashMap::new();

    (input, env)
}

pub fn proc_visit_inputs_and_env_span() -> (ProcVisitInputsSpan, HashMap<String, Par>) {
    let proc_inputs = ProcVisitInputsSpan {
        par: Default::default(),
        bound_map_chain: BoundMapChainSpan::new(),
        free_map: Default::default(),
    };
    let env: HashMap<String, Par> = HashMap::new();

    (proc_inputs, env)
}

pub fn collection_proc_visit_inputs_and_env_span() -> (ProcVisitInputsSpan, HashMap<String, Par>) {
    let proc_inputs = ProcVisitInputsSpan {
        par: Default::default(),
        bound_map_chain: {
            let bound_map_chain = BoundMapChainSpan::new();
            bound_map_chain.put_all_pos(vec![
                (
                    "P".to_string(),
                    ProcSort,
                    SourcePos { line: 1, col: 1 }, // Use 1-based indexing consistent with rholang-rs
                ),
                ("x".to_string(), NameSort, SourcePos { line: 1, col: 1 }),
            ])
        },
        free_map: Default::default(),
    };
    let env: HashMap<String, Par> = HashMap::new();

    (proc_inputs, env)
}

pub fn proc_visit_inputs_with_updated_bound_map_chain_span(
    input: ProcVisitInputsSpan,
    name: &str,
    vs_type: VarSort,
) -> ProcVisitInputsSpan {
    ProcVisitInputsSpan {
        bound_map_chain: {
            let updated_bound_map_chain = input.bound_map_chain.put_pos((
                name.to_string(),
                vs_type,
                SourcePos { line: 1, col: 1 }, // Use 1-based indexing
            ));
            updated_bound_map_chain
        },
        ..input.clone()
    }
}

pub fn proc_visit_inputs_with_updated_vec_bound_map_chain_span(
    input: ProcVisitInputsSpan,
    new_bindings: Vec<(String, VarSort)>,
) -> ProcVisitInputsSpan {
    let bindings_with_default_positions: Vec<IdContextPos<VarSort>> = new_bindings
        .into_iter()
        .map(|(name, var_sort)| (name, var_sort, SourcePos { line: 1, col: 1 }))
        .collect();

    ProcVisitInputsSpan {
        bound_map_chain: {
            let updated_bound_map_chain = input
                .bound_map_chain
                .put_all_pos(bindings_with_default_positions);
            updated_bound_map_chain
        },
        ..input.clone()
    }
}
