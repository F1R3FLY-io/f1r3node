// See rholang/src/main/scala/coop/rchain/rholang/interpreter/compiler/Compiler.scala

use super::normalize::normalize_ann_proc;
use crate::rust::interpreter::{compiler::exports::ProcVisitInputsSpan, errors::InterpreterError};
use models::{
    rhoapi::{connective::ConnectiveInstance, Par},
    rust::rholang::sorter::{par_sort_matcher::ParSortMatcher, sortable::Sortable},
};
use std::collections::HashMap;

pub struct Compiler;

impl Compiler {
    pub fn source_to_adt(source: &str) -> Result<Par, InterpreterError> {
        Self::source_to_adt_with_normalizer_env(source, HashMap::new())
    }

    pub fn source_to_adt_with_normalizer_env(
        source: &str,
        normalizer_env: HashMap<String, Par>,
    ) -> Result<Par, InterpreterError> {
        let parser = rholang_parser::RholangParser::new();
        let result = parser.parse(source);

        match result {
            validated::Validated::Good(procs) => {
                if procs.len() == 1 {
                    let proc = procs.into_iter().next().unwrap();
                    Self::normalize_term(proc, normalizer_env, &parser)
                } else {
                    Err(InterpreterError::ParserError(format!(
                        "Expected single process, got {}",
                        procs.len()
                    )))
                }
            }
            validated::Validated::Fail(failures) => {
                // Convert parsing failures to InterpreterError
                let error_messages: Vec<String> = failures
                    .iter()
                    .flat_map(|failure| {
                        failure
                            .errors
                            .iter()
                            .map(|error| format!("{:?} at {:?}", error.error, error.span))
                    })
                    .collect();
                Err(InterpreterError::ParserError(format!(
                    "Parse failed: {}",
                    error_messages.join(", ")
                )))
            }
        }
    }

    fn normalize_term<'a>(
        ast: rholang_parser::ast::AnnProc<'a>,
        normalizer_env: HashMap<String, Par>,
        parser: &'a rholang_parser::RholangParser<'a>,
    ) -> Result<Par, InterpreterError> {
        let normalized_result =
            normalize_ann_proc(&ast, ProcVisitInputsSpan::new(), &normalizer_env, parser)?;

        if normalized_result.free_map.count() > 0 {
            if !normalized_result.free_map.connectives.is_empty() {
                fn connective_instance_to_string(conn: ConnectiveInstance) -> String {
                    match conn {
                        ConnectiveInstance::ConnAndBody(_) => String::from("/\\ (conjunction)"),
                        ConnectiveInstance::ConnOrBody(_) => String::from("\\/ (disjunction)"),
                        ConnectiveInstance::ConnNotBody(_) => String::from("~ (negation)"),
                        _ => format!("{:?}", conn),
                    }
                }

                let connectives: Vec<String> = normalized_result
                    .free_map
                    .connectives
                    .into_iter()
                    .map(|(conn_type, source_position)| {
                        format!(
                            "{} at {:?}",
                            connective_instance_to_string(conn_type),
                            source_position
                        )
                    })
                    .collect();

                return Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(
                    connectives.join(", "),
                ));
            } else if !normalized_result.free_map.wildcards.is_empty() {
                let top_level_wildcard_list: Vec<String> = normalized_result
                    .free_map
                    .wildcards
                    .into_iter()
                    .map(|source_position| format!("_ (wildcard) at {:?}", source_position))
                    .collect();

                return Err(InterpreterError::TopLevelWildcardsNotAllowedError(
                    top_level_wildcard_list.join(", "),
                ));
            } else {
                let free_variable_list: Vec<String> = normalized_result
                    .free_map
                    .level_bindings
                    .into_iter()
                    .map(|(var_name, var_sort)| {
                        format!("{} at {:?}", var_name, var_sort.source_span)
                    })
                    .collect();

                return Err(InterpreterError::TopLevelFreeVariablesNotAllowedError(
                    free_variable_list.join(", "),
                ));
            }
        }

        let sorted_par = ParSortMatcher::sort_match(&normalized_result.par);
        Ok(sorted_par.term)
    }
}
