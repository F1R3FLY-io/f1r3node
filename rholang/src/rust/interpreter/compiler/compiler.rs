// See rholang/src/main/scala/coop/rchain/rholang/interpreter/compiler/Compiler.scala

use models::{
    rhoapi::{connective::ConnectiveInstance, Par},
    rust::rholang::sorter::{par_sort_matcher::ParSortMatcher, sortable::Sortable},
};
use std::collections::HashMap;

use crate::rust::interpreter::{
    compiler::normalizer::parser::parse_rholang_code_to_proc, errors::InterpreterError,
};

use super::{
    normalize::{normalize_match_proc, ProcVisitInputs},
    rholang_ast::Proc,
};

pub struct Compiler;

impl Compiler {
    pub fn source_to_adt(source: &str) -> Result<Par, InterpreterError> {
        Self::source_to_adt_with_normalizer_env(source, HashMap::new())
    }

    pub fn source_to_adt_with_normalizer_env(
        source: &str,
        normalizer_env: HashMap<String, Par>,
    ) -> Result<Par, InterpreterError> {
        let proc = Self::source_to_ast(source)?;
        Self::ast_to_adt_with_normalizer_env(proc, normalizer_env)
    }

    pub fn ast_to_adt(proc: Proc) -> Result<Par, InterpreterError> {
        Self::ast_to_adt_with_normalizer_env(proc, HashMap::new())
    }

    pub fn ast_to_adt_with_normalizer_env(
        proc: Proc,
        normalizer_env: HashMap<String, Par>,
    ) -> Result<Par, InterpreterError> {
        let par = Self::normalize_term(proc, normalizer_env)?;
        let sorted_par = ParSortMatcher::sort_match(&par);
        Ok(sorted_par.term)
    }

    pub fn source_to_ast(source: &str) -> Result<Proc, InterpreterError> {
        parse_rholang_code_to_proc(source)
    }

    fn normalize_term(
        term: Proc,
        normalizer_env: HashMap<String, Par>,
    ) -> Result<Par, InterpreterError> {
        // println!("\nhit normalize_term");
        normalize_match_proc(&term, ProcVisitInputs::new(), &normalizer_env).map(
            |normalized_term| {
                // println!("\nnormalized term: {:?}", normalized_term);
                if normalized_term.free_map.count() > 0 {
                    if normalized_term.free_map.wildcards.is_empty()
                        && normalized_term.free_map.connectives.is_empty()
                    {
                        let top_level_free_list: Vec<String> = normalized_term
                            .free_map
                            .level_bindings
                            .into_iter()
                            .map(|(name, free_context)| {
                                format!("{} at {}", name, free_context.source_position)
                            })
                            .collect();

                        Err(InterpreterError::TopLevelFreeVariablesNotAllowedError(
                            top_level_free_list.join(", "),
                        ))
                    } else if !normalized_term.free_map.connectives.is_empty() {
                        fn connective_instance_to_string(conn: ConnectiveInstance) -> String {
                            match conn {
                                ConnectiveInstance::ConnAndBody(_) => {
                                    String::from("/\\ (conjunction)")
                                }

                                ConnectiveInstance::ConnOrBody(_) => {
                                    String::from("\\/ (disjunction)")
                                }

                                ConnectiveInstance::ConnNotBody(_) => String::from("~ (negation)"),

                                _ => format!("{:?}", conn),
                            }
                        }

                        let connectives: Vec<String> = normalized_term
                            .free_map
                            .connectives
                            .into_iter()
                            .map(|(conn_type, source_position)| {
                                format!(
                                    "{} at {}",
                                    connective_instance_to_string(conn_type),
                                    source_position
                                )
                            })
                            .collect();

                        Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(
                            connectives.join(", "),
                        ))
                    } else {
                        let top_level_wildcard_list: Vec<String> = normalized_term
                            .free_map
                            .wildcards
                            .into_iter()
                            .map(|source_position| format!("_ (wildcard) at {}", source_position))
                            .collect();

                        Err(InterpreterError::TopLevelWildcardsNotAllowedError(
                            top_level_wildcard_list.join(", "),
                        ))
                    }
                } else {
                    Ok(normalized_term.par)
                }
            },
        )?
    }
}
