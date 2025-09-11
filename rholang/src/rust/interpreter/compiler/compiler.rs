// See rholang/src/main/scala/coop/rchain/rholang/interpreter/compiler/Compiler.scala

use models::{
    rhoapi::{connective::ConnectiveInstance, Par},
    rust::rholang::sorter::{par_sort_matcher::ParSortMatcher, sortable::Sortable},
};
use std::collections::HashMap;

use crate::rust::interpreter::{
    compiler::{exports::ProcVisitInputsSpan, normalizer::parser::parse_rholang_code_to_proc},
    errors::InterpreterError,
};

use super::{
    normalize::{normalize_ann_proc, normalize_match_proc, ProcVisitInputs},
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
                                format!("{} at {:?}", name, free_context.source_position)
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
                                    "{} at {:?}",
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
                            .map(|source_position| format!("_ (wildcard) at {:?}", source_position))
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

    // New parser integration methods
    pub fn new_source_to_adt(source: &str) -> Result<Par, InterpreterError> {
        Self::new_source_to_adt_with_normalizer_env(source, HashMap::new())
    }

    pub fn new_source_to_adt_with_normalizer_env(
        source: &str,
        normalizer_env: HashMap<String, Par>,
    ) -> Result<Par, InterpreterError> {
        // Use arena-based approach - parser owns the arena, we work with its lifetimes
        let parser = rholang_parser::RholangParser::new();
        let result = parser.parse(source);

        match result {
            validated::Validated::Good(procs) => {
                if procs.len() == 1 {
                    let proc = procs.into_iter().next().unwrap();
                    // Work directly with arena-allocated AST - no unsafe transmute needed
                    Self::arena_normalize_term(proc, normalizer_env, &parser)
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

    // Arena-based normalization - works with parser's natural lifetimes
    fn arena_normalize_term<'a>(
        ast: rholang_parser::ast::AnnProc<'a>,
        normalizer_env: HashMap<String, Par>,
        parser: &'a rholang_parser::RholangParser<'a>,
    ) -> Result<Par, InterpreterError> {
        let normalized_result =
            normalize_ann_proc(&ast, ProcVisitInputsSpan::new(), &normalizer_env, parser)?;

        // Reuse existing validation logic from normalize_term
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

    // Legacy method - kept for backward compatibility but now deprecated
    pub fn new_source_to_ast(
        source: &str,
    ) -> Result<rholang_parser::ast::AnnProc<'static>, InterpreterError> {
        let parser = rholang_parser::RholangParser::new();
        let result = parser.parse(source);

        match result {
            validated::Validated::Good(procs) => {
                if procs.len() == 1 {
                    // Convert to 'static lifetime using Box::leak
                    let proc = procs.into_iter().next().unwrap();
                    let static_proc = unsafe {
                        std::mem::transmute::<
                            rholang_parser::ast::AnnProc<'_>,
                            rholang_parser::ast::AnnProc<'static>,
                        >(proc)
                    };
                    Ok(static_proc)
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

    pub fn new_ast_to_adt_with_normalizer_env(
        ast: rholang_parser::ast::AnnProc<'static>,
        normalizer_env: HashMap<String, Par>,
        parser: &'static rholang_parser::RholangParser<'static>,
    ) -> Result<Par, InterpreterError> {
        let par = Self::new_normalize_term(ast, normalizer_env, parser)?;
        let sorted_par = ParSortMatcher::sort_match(&par);
        Ok(sorted_par.term)
    }

    fn new_normalize_term(
        ast: rholang_parser::ast::AnnProc<'static>,
        normalizer_env: HashMap<String, Par>,
        parser: &'static rholang_parser::RholangParser<'static>,
    ) -> Result<Par, InterpreterError> {
        let normalized_result =
            normalize_ann_proc(&ast, ProcVisitInputsSpan::new(), &normalizer_env, parser)?;

        // Reuse existing validation logic from normalize_term
        if normalized_result.free_map.count() > 0 {
            if normalized_result.free_map.wildcards.is_empty()
                && normalized_result.free_map.connectives.is_empty()
            {
                let top_level_free_list: Vec<String> = normalized_result
                    .free_map
                    .level_bindings
                    .into_iter()
                    .map(|(name, free_context)| {
                        format!("{} at {:?}", name, free_context.source_span)
                    })
                    .collect();

                Err(InterpreterError::TopLevelFreeVariablesNotAllowedError(
                    top_level_free_list.join(", "),
                ))
            } else if !normalized_result.free_map.connectives.is_empty() {
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

                Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(
                    connectives.join(", "),
                ))
            } else {
                let top_level_wildcard_list: Vec<String> = normalized_result
                    .free_map
                    .wildcards
                    .into_iter()
                    .map(|source_position| format!("_ (wildcard) at {:?}", source_position))
                    .collect();

                Err(InterpreterError::TopLevelWildcardsNotAllowedError(
                    top_level_wildcard_list.join(", "),
                ))
            }
        } else {
            Ok(normalized_result.par)
        }
    }
}
