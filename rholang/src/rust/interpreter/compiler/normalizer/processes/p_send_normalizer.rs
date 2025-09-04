use super::exports::*;
use crate::rust::interpreter::compiler::normalize::{
    normalize_ann_proc, normalize_match_proc, NameVisitInputs, ProcVisitInputs, ProcVisitOutputs,
};
use crate::rust::interpreter::compiler::normalizer::name_normalize_matcher::normalize_name_new_ast;
use crate::rust::interpreter::compiler::rholang_ast::{Name, ProcList, SendType};
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::matcher::has_locally_free::HasLocallyFree;
use models::rhoapi::{Par, Send};
use models::rust::utils::union;
use rholang_parser::ast::AnnName;
use std::collections::HashMap;

pub fn normalize_p_send(
    name: &Name,
    send_type: &SendType,
    inputs: &ProcList,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let name_match_result = normalize_name(
        name,
        NameVisitInputs {
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: input.free_map.clone(),
            source_span: input.source_span,
        },
        env,
    )?;

    let mut acc = (
        Vec::new(),
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: name_match_result.free_map.clone(),
            source_span: input.source_span,
        },
        Vec::new(),
        false,
    );

    for proc in inputs.procs.clone() {
        let proc_match_result = normalize_match_proc(&proc, acc.1.clone(), env)?;

        acc.0.push(proc_match_result.par.clone());
        acc.1 = ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: proc_match_result.free_map.clone(),
            source_span: input.source_span,
        };
        acc.2 = union(acc.2.clone(), proc_match_result.par.locally_free.clone());
        acc.3 = acc.3 || proc_match_result.par.connective_used;
    }

    let persistent = match send_type {
        SendType::Single { .. } => false,
        SendType::Multiple { .. } => true,
    };

    let send = Send {
        chan: Some(name_match_result.par.clone()),
        data: acc.0,
        persistent,
        locally_free: union(
            name_match_result.par.clone().locally_free(
                name_match_result.par.clone(),
                input.bound_map_chain.depth() as i32,
            ),
            acc.2,
        ),
        connective_used: name_match_result
            .par
            .connective_used(name_match_result.par.clone())
            || acc.3,
    };

    let updated_par = input.par.clone().prepend_send(send);

    Ok(ProcVisitOutputs {
        par: updated_par,
        free_map: acc.1.free_map,
    })
}

/// Parallel version of normalize_p_send for new AST Send
pub fn normalize_p_send_new_ast<'ast>(
    channel: &'ast AnnName<'ast>,
    send_type: &rholang_parser::ast::SendType,
    inputs: &'ast rholang_parser::ast::ProcList<'ast>,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let name_match_result = normalize_name_new_ast(
        &channel.name,
        NameVisitInputs {
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: input.free_map.clone(),
            source_span: input.source_span,
        },
        env,
        parser,
    )?;

    let mut acc = (
        Vec::new(),
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: name_match_result.free_map.clone(),
            source_span: input.source_span,
        },
        Vec::new(),
        false,
    );

    for proc in inputs.iter() {
        let proc_match_result = normalize_ann_proc(proc, acc.1.clone(), env, parser)?;

        acc.0.push(proc_match_result.par.clone());
        acc.1 = ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: proc_match_result.free_map.clone(),
            source_span: input.source_span,
        };
        acc.2 = union(acc.2.clone(), proc_match_result.par.locally_free.clone());
        acc.3 = acc.3 || proc_match_result.par.connective_used;
    }

    let persistent = match send_type {
        rholang_parser::ast::SendType::Single => false,
        rholang_parser::ast::SendType::Multiple => true,
    };

    let send = Send {
        chan: Some(name_match_result.par.clone()),
        data: acc.0,
        persistent,
        locally_free: union(
            name_match_result.par.clone().locally_free(
                name_match_result.par.clone(),
                input.bound_map_chain.depth() as i32,
            ),
            acc.2,
        ),
        connective_used: name_match_result
            .par
            .connective_used(name_match_result.par.clone())
            || acc.3,
    };

    let updated_par = input.par.clone().prepend_send(send);

    Ok(ProcVisitOutputs {
        par: updated_par,
        free_map: acc.1.free_map,
    })
}

// See rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use models::{
        create_bit_vector,
        rhoapi::Par,
        rust::utils::{new_boundvar_par, new_gint_par, new_send},
    };

    use crate::rust::interpreter::{
        compiler::{
            compiler::Compiler,
            normalize::{normalize_match_proc, ProcVisitInputs, VarSort},
            rholang_ast::{Name, ProcList, SendType},
        },
        errors::InterpreterError,
        test_utils::utils::proc_visit_inputs_and_env,
    };

    use super::{Proc, SourcePosition};

    #[test]
    fn p_send_should_handle_a_basic_send() {
        let p_send = Proc::Send {
            name: Name::new_name_quote_nil(),
            send_type: SendType::new_single(),
            inputs: ProcList::new(vec![Proc::new_proc_int(7), Proc::new_proc_int(8)]),
            line_num: 0,
            col_num: 0,
        };

        let (mut inputs, env) = proc_visit_inputs_and_env();

        let result = normalize_match_proc(&p_send, inputs.clone(), &env);
        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            inputs.par.prepend_send(new_send(
                Par::default(),
                vec![
                    new_gint_par(7, Vec::new(), false),
                    new_gint_par(8, Vec::new(), false)
                ],
                false,
                Vec::new(),
                false
            ))
        );
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }

    #[test]
    fn p_send_should_handle_a_name_var() {
        let p_send = Proc::Send {
            name: Name::new_name_var("x"),
            send_type: SendType::new_single(),
            inputs: ProcList::new(vec![Proc::new_proc_int(7), Proc::new_proc_int(8)]),
            line_num: 0,
            col_num: 0,
        };

        let (mut inputs, env) = proc_visit_inputs_and_env();
        inputs.bound_map_chain = inputs.bound_map_chain.put((
            "x".to_string(),
            VarSort::NameSort,
            SourcePosition::new(0, 0),
        ));

        let result = normalize_match_proc(&p_send, inputs.clone(), &env);
        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            inputs.par.prepend_send(new_send(
                new_boundvar_par(0, create_bit_vector(&vec![0]), false),
                vec![
                    new_gint_par(7, Vec::new(), false),
                    new_gint_par(8, Vec::new(), false)
                ],
                false,
                create_bit_vector(&vec![0]),
                false
            ))
        );
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }

    #[test]
    fn p_send_should_propagate_known_free() {
        let p_send = Proc::Send {
            name: Name::new_name_quote_var("x"),
            send_type: SendType::new_single(),
            inputs: ProcList::new(vec![Proc::new_proc_int(7), Proc::new_proc_var("x")]),
            line_num: 0,
            col_num: 0,
        };

        let result = normalize_match_proc(&p_send, ProcVisitInputs::new(), &HashMap::new());
        assert!(result.is_err());
        assert_eq!(
            result,
            Err(InterpreterError::UnexpectedReuseOfProcContextFree {
                var_name: "x".to_string(),
                first_use: SourcePosition::new(0, 0),
                second_use: SourcePosition::new(0, 0)
            })
        );
    }

    #[test]
    fn p_send_should_not_compile_if_data_contains_negation() {
        let result = Compiler::source_to_adt(r#"new x in { x!(~1) }"#);
        assert!(result.is_err());
        assert_eq!(
            result,
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(
                format!("~ (negation) at {:?}", SourcePosition::new(0, 14))
            ))
        )
    }

    #[test]
    fn p_send_should_not_compile_if_data_contains_conjuction() {
        let result = Compiler::source_to_adt(r#"new x in { x!(1 /\ 2) }"#);
        assert!(result.is_err());
        assert_eq!(
            result,
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(
                format!("/\\ (conjunction) at {:?}", SourcePosition::new(0, 14))
            ))
        )
    }

    #[test]
    fn p_send_should_not_compile_if_data_contains_disjunction() {
        let result = Compiler::source_to_adt(r#"new x in { x!(1 \/ 2) }"#);
        assert!(result.is_err());
        assert_eq!(
            result,
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(
                format!("\\/ (disjunction) at {:?}", SourcePosition::new(0, 14))
            ))
        )
    }

    #[test]
    fn p_send_should_not_compile_if_data_contains_wildcard() {
        let result = Compiler::source_to_adt(r#"@"x"!(_)"#);
        assert!(result.is_err());
        assert_eq!(
            result,
            Err(InterpreterError::TopLevelWildcardsNotAllowedError(format!(
                "_ (wildcard) at {:?}",
                SourcePosition::new(0, 6)
            )))
        )
    }

    #[test]
    fn p_send_should_not_compile_if_data_contains_free_variable() {
        let result = Compiler::source_to_adt(r#"@"x"!(y)"#);
        assert!(result.is_err());
        assert_eq!(
            result,
            Err(InterpreterError::TopLevelFreeVariablesNotAllowedError(
                format!("y at {:?}", SourcePosition::new(0, 6))
            ))
        )
    }

    #[test]
    fn p_send_should_not_compile_if_name_contains_connectives() {
        let result1 = Compiler::source_to_adt(r#"@{Nil /\ Nil}!(1)"#);
        assert!(result1.is_err());
        assert_eq!(
            result1,
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(
                format!(
                    "/\\ (conjunction) at {:?}",
                    SourcePosition { row: 0, column: 2 }
                )
            ))
        );

        let result2 = Compiler::source_to_adt(r#"@{Nil \/ Nil}!(1)"#);
        assert!(result2.is_err());
        assert_eq!(
            result2,
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(
                format!(
                    "\\/ (disjunction) at {:?}",
                    SourcePosition { row: 0, column: 2 }
                )
            ))
        );

        let result3 = Compiler::source_to_adt(r#"@{~Nil}!(1)"#);
        assert!(result3.is_err());
        assert_eq!(
            result3,
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(
                format!("~ (negation) at {:?}", SourcePosition { row: 0, column: 2 })
            ))
        );
    }

    #[test]
    fn new_ast_p_send_should_handle_a_basic_send() {
        // Maps to original: p_send_should_handle_a_basic_send
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{
            AnnName, AnnProc, Name as NewName, Proc as NewProc, SendType as NewSendType,
        };
        use rholang_parser::{SourcePos, SourceSpan};

        let (mut inputs, env) = proc_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();

        // Create @Nil!(7, 8) using new AST
        let send_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::Send {
                channel: AnnName {
                    name: NewName::Quote(Box::leak(Box::new(NewProc::Nil))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                send_type: NewSendType::Single,
                inputs: smallvec::SmallVec::from_vec(vec![
                    AnnProc {
                        proc: Box::leak(Box::new(NewProc::LongLiteral(7))),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    AnnProc {
                        proc: Box::leak(Box::new(NewProc::LongLiteral(8))),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                ]),
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let result = normalize_ann_proc(&send_proc, inputs.clone(), &env, &parser);
        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            inputs.par.prepend_send(new_send(
                Par::default(),
                vec![
                    new_gint_par(7, Vec::new(), false),
                    new_gint_par(8, Vec::new(), false)
                ],
                false,
                Vec::new(),
                false
            ))
        );
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }

    #[test]
    fn new_ast_p_send_should_handle_a_name_var() {
        // Maps to original: p_send_should_handle_a_name_var
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{
            AnnName, AnnProc, Id, Name as NewName, Proc as NewProc, SendType as NewSendType,
            Var as NewVar,
        };
        use rholang_parser::{SourcePos, SourceSpan};

        let (mut inputs, env) = proc_visit_inputs_and_env();
        inputs.bound_map_chain = inputs.bound_map_chain.put((
            "x".to_string(),
            VarSort::NameSort,
            SourcePosition::new(0, 0),
        ));
        let parser = rholang_parser::RholangParser::new();

        // Create x!(7, 8) using new AST (name variable send)
        let send_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::Send {
                channel: AnnName {
                    name: NewName::ProcVar(NewVar::Id(Id {
                        name: "x",
                        pos: SourcePos { line: 0, col: 0 },
                    })),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                send_type: NewSendType::Single,
                inputs: smallvec::SmallVec::from_vec(vec![
                    AnnProc {
                        proc: Box::leak(Box::new(NewProc::LongLiteral(7))),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    AnnProc {
                        proc: Box::leak(Box::new(NewProc::LongLiteral(8))),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                ]),
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let result = normalize_ann_proc(&send_proc, inputs.clone(), &env, &parser);
        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            inputs.par.prepend_send(new_send(
                new_boundvar_par(0, create_bit_vector(&vec![0]), false),
                vec![
                    new_gint_par(7, Vec::new(), false),
                    new_gint_par(8, Vec::new(), false)
                ],
                false,
                create_bit_vector(&vec![0]),
                false
            ))
        );
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }

    #[test]
    fn new_ast_p_send_should_propagate_known_free() {
        // Maps to original: p_send_should_propagate_known_free
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{
            AnnName, AnnProc, Id, Name as NewName, Proc as NewProc, SendType as NewSendType,
            Var as NewVar,
        };
        use rholang_parser::{SourcePos, SourceSpan};

        // Create @{x}!(7, x) using new AST - reuse of variable x should cause error
        let send_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::Send {
                channel: AnnName {
                    name: NewName::Quote(Box::leak(Box::new(NewProc::ProcVar(NewVar::Id(Id {
                        name: "x",
                        pos: SourcePos { line: 0, col: 0 },
                    }))))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                send_type: NewSendType::Single,
                inputs: smallvec::SmallVec::from_vec(vec![
                    AnnProc {
                        proc: Box::leak(Box::new(NewProc::LongLiteral(7))),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    AnnProc {
                        proc: Box::leak(Box::new(NewProc::ProcVar(NewVar::Id(Id {
                            name: "x",
                            pos: SourcePos { line: 0, col: 0 },
                        })))),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                ]),
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result =
            normalize_ann_proc(&send_proc, ProcVisitInputs::new(), &HashMap::new(), &parser);
        assert!(result.is_err());
        assert_eq!(
            result,
            Err(InterpreterError::UnexpectedReuseOfProcContextFree {
                var_name: "x".to_string(),
                first_use: SourcePosition::new(1, 1),
                second_use: SourcePosition::new(0, 0)
            })
        );
    }

    #[test]
    fn new_ast_p_send_should_not_compile_if_data_contains_negation() {
        let result = Compiler::new_source_to_adt(r#"new x in { x!(~1) }"#);
        assert!(result.is_err());
        match result {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("~ (negation)"));
            }
            other => panic!(
                "Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}",
                other
            ),
        }
    }

    #[test]
    fn new_ast_p_send_should_not_compile_if_data_contains_conjuction() {
        let result = Compiler::new_source_to_adt(r#"new x in { x!(1 /\ 2) }"#);
        assert!(result.is_err());
        match result {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("/\\ (conjunction)"));
            }
            other => panic!(
                "Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}",
                other
            ),
        }
    }

    #[test]
    fn new_ast_p_send_should_not_compile_if_data_contains_disjunction() {
        let result = Compiler::new_source_to_adt(r#"new x in { x!(1 \/ 2) }"#);
        assert!(result.is_err());
        match result {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("\\/ (disjunction)"));
            }
            other => panic!(
                "Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}",
                other
            ),
        }
    }

    #[test]
    fn new_ast_p_send_should_not_compile_if_data_contains_wildcard() {
        let result = Compiler::new_source_to_adt(r#"@"x"!(_)"#);
        assert!(result.is_err());
        match result {
            Err(InterpreterError::TopLevelWildcardsNotAllowedError(msg)) => {
                assert!(msg.contains("_ (wildcard)"));
            }
            other => panic!(
                "Expected TopLevelWildcardsNotAllowedError, got: {:?}",
                other
            ),
        }
    }

    #[test]
    fn new_ast_p_send_should_not_compile_if_data_contains_free_variable() {
        let result = Compiler::new_source_to_adt(r#"@"x"!(y)"#);
        assert!(result.is_err());
        match result {
            Err(InterpreterError::TopLevelFreeVariablesNotAllowedError(msg)) => {
                assert!(msg.contains("y"));
            }
            other => panic!(
                "Expected TopLevelFreeVariablesNotAllowedError, got: {:?}",
                other
            ),
        }
    }

    #[test]
    fn new_ast_p_send_should_not_compile_if_name_contains_connectives() {
        // Test conjunction in channel name
        let result1 = Compiler::new_source_to_adt(r#"@{Nil /\ Nil}!(1)"#);
        assert!(result1.is_err());
        match result1 {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("/\\ (conjunction)"));
            }
            other => panic!(
                "Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}",
                other
            ),
        }

        // Test disjunction in channel name
        let result2 = Compiler::new_source_to_adt(r#"@{Nil \/ Nil}!(1)"#);
        assert!(result2.is_err());
        match result2 {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("\\/ (disjunction)"));
            }
            other => panic!(
                "Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}",
                other
            ),
        }

        // Test negation in channel name
        let result3 = Compiler::new_source_to_adt(r#"@{~Nil}!(1)"#);
        assert!(result3.is_err());
        match result3 {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("~ (negation)"));
            }
            other => panic!(
                "Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}",
                other
            ),
        }
    }
}
