// Legacy types - preserved for backward compatibility
pub use crate::rust::interpreter::compiler::bound_context::BoundContext;
pub use crate::rust::interpreter::compiler::bound_map::BoundMap;
pub use crate::rust::interpreter::compiler::bound_map_chain::BoundMapChain;
pub use crate::rust::interpreter::compiler::free_context::FreeContext;
pub use crate::rust::interpreter::compiler::free_map::FreeMap;
pub use crate::rust::interpreter::compiler::id_context::IdContext;
pub use crate::rust::interpreter::compiler::source_position::SourcePosition;

// SourceSpan-based parallel types for rholang-rs parser integration
pub use crate::rust::interpreter::compiler::bound_context::BoundContextSpan;
pub use crate::rust::interpreter::compiler::bound_map::BoundMapSpan;
pub use crate::rust::interpreter::compiler::bound_map_chain::BoundMapChainSpan;
pub use crate::rust::interpreter::compiler::free_context::FreeContextSpan;
pub use crate::rust::interpreter::compiler::free_map::FreeMapSpan;
pub use crate::rust::interpreter::compiler::id_context::{IdContextSpan, IdContextPos};
pub use crate::rust::interpreter::compiler::normalize::{
    ProcVisitInputsSpan, ProcVisitOutputsSpan, NameVisitInputsSpan, NameVisitOutputsSpan,
    CollectVisitInputsSpan, CollectVisitOutputsSpan
};

pub use models::rhoapi::connective::ConnectiveInstance;

pub use crate::rust::interpreter::compiler::normalizer::processes::p_bundle_normalizer::normalize_p_bundle;
pub use crate::rust::interpreter::compiler::normalizer::processes::p_collect_normalizer::normalize_p_collect;
pub use crate::rust::interpreter::compiler::normalizer::processes::p_conjunction_normalizer::normalize_p_conjunction;
pub use crate::rust::interpreter::compiler::normalizer::processes::p_contr_normalizer::normalize_p_contr;
pub use crate::rust::interpreter::compiler::normalizer::processes::p_disjunction_normalizer::normalize_p_disjunction;
pub use crate::rust::interpreter::compiler::normalizer::processes::p_eval_normalizer::normalize_p_eval;
pub use crate::rust::interpreter::compiler::normalizer::processes::p_ground_normalizer::normalize_p_ground;
pub use crate::rust::interpreter::compiler::normalizer::processes::p_if_normalizer::normalize_p_if;
pub use crate::rust::interpreter::compiler::normalizer::processes::p_match_normalizer::normalize_p_match;
pub use crate::rust::interpreter::compiler::normalizer::processes::p_matches_normalizer::normalize_p_matches;
pub use crate::rust::interpreter::compiler::normalizer::processes::p_method_normalizer::normalize_p_method;
pub use crate::rust::interpreter::compiler::normalizer::processes::p_negation_normalizer::normalize_p_negation;
pub use crate::rust::interpreter::compiler::normalizer::processes::p_new_normalizer::normalize_p_new;
pub use crate::rust::interpreter::compiler::normalizer::processes::p_par_normalizer::normalize_p_par;
pub use crate::rust::interpreter::compiler::normalizer::processes::p_send_normalizer::normalize_p_send;
pub use crate::rust::interpreter::compiler::normalizer::processes::p_send_sync_normalizer::normalize_p_send_sync;
pub use crate::rust::interpreter::compiler::normalizer::processes::p_simple_type_normalizer::normalize_simple_type;
