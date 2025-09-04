use crate::rust::interpreter::compiler::bound_map_chain::BoundMapChain;
use crate::rust::interpreter::compiler::exports::IdContext;
use crate::rust::interpreter::compiler::free_map::FreeMap;
use crate::rust::interpreter::compiler::normalize::VarSort::{NameSort, ProcSort};
use crate::rust::interpreter::compiler::normalize::{NameVisitInputs, ProcVisitInputs, VarSort};
use crate::rust::interpreter::compiler::source_position::SourcePosition;
use crate::rust::interpreter::compiler::span_utils::SpanContext;
use models::rhoapi::Par;
use std::collections::HashMap;

pub fn name_visit_inputs_and_env() -> (NameVisitInputs, HashMap<String, Par>) {
    let input: NameVisitInputs = NameVisitInputs {
        bound_map_chain: BoundMapChain::default(),
        free_map: FreeMap::default(),
        source_span: SpanContext::zero_span(),
    };
    let env: HashMap<String, Par> = HashMap::new();

    (input, env)
}

pub fn proc_visit_inputs_and_env() -> (ProcVisitInputs, HashMap<String, Par>) {
    let proc_inputs = ProcVisitInputs {
        par: Default::default(),
        bound_map_chain: BoundMapChain::new(),
        free_map: Default::default(),
        source_span: SpanContext::zero_span(),
    };
    let env: HashMap<String, Par> = HashMap::new();

    (proc_inputs, env)
}

pub fn collection_proc_visit_inputs_and_env() -> (ProcVisitInputs, HashMap<String, Par>) {
    let proc_inputs = ProcVisitInputs {
        par: Default::default(),
        bound_map_chain: {
            let bound_map_chain = BoundMapChain::new();
            bound_map_chain.put_all(vec![
                (
                    "P".to_string(),
                    ProcSort,
                    SourcePosition { row: 0, column: 0 },
                ),
                (
                    "x".to_string(),
                    NameSort,
                    SourcePosition { row: 0, column: 0 },
                ),
            ])
        },
        free_map: Default::default(),
        source_span: SpanContext::zero_span(),
    };
    let env: HashMap<String, Par> = HashMap::new();

    (proc_inputs, env)
}

pub fn proc_visit_inputs_with_updated_bound_map_chain(
    input: ProcVisitInputs,
    name: &str,
    vs_type: VarSort,
) -> ProcVisitInputs {
    ProcVisitInputs {
        bound_map_chain: {
            let updated_bound_map_chain = input.bound_map_chain.put((
                name.to_string(),
                vs_type,
                SourcePosition { row: 0, column: 0 },
            ));
            updated_bound_map_chain
        },
        ..input.clone()
    }
}

pub fn proc_visit_inputs_with_updated_vec_bound_map_chain(
    input: ProcVisitInputs,
    new_bindings: Vec<(String, VarSort)>,
) -> ProcVisitInputs {
    let bindings_with_default_positions: Vec<IdContext<VarSort>> = new_bindings
        .into_iter()
        .map(|(name, var_sort)| (name, var_sort, SourcePosition { row: 0, column: 0 }))
        .collect();

    ProcVisitInputs {
        bound_map_chain: {
            let updated_bound_map_chain = input
                .bound_map_chain
                .put_all(bindings_with_default_positions);
            updated_bound_map_chain
        },
        ..input.clone()
    }
}
