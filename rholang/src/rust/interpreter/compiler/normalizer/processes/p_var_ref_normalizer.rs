use crate::rust::interpreter::compiler::exports::{BoundContext, BoundContextSpan};
use crate::rust::interpreter::compiler::normalize::VarSort;
use crate::rust::interpreter::compiler::rholang_ast::{VarRef as PVarRef, VarRefKind};
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::util::prepend_connective;

use super::exports::*;
// Additional exports needed for span-based types
use crate::rust::interpreter::compiler::exports::{ProcVisitInputsSpan, ProcVisitOutputsSpan};
use models::rhoapi::connective::ConnectiveInstance;
use models::rhoapi::{Connective, VarRef};
use std::result::Result;

// New AST imports for parallel functions
use rholang_parser::ast::{Id, VarRefKind as NewVarRefKind};

pub fn normalize_p_var_ref(
    p: &PVarRef,
    input: ProcVisitInputs,
) -> Result<ProcVisitOutputs, InterpreterError> {
    match input.bound_map_chain.find(&p.var.name) {
        Some((
            BoundContext {
                index,
                typ,
                source_position,
            },
            depth,
        )) => match typ {
            VarSort::ProcSort => match p.var_ref_kind {
                VarRefKind::Proc => Ok(ProcVisitOutputs {
                    par: prepend_connective(
                        input.par,
                        Connective {
                            connective_instance: Some(ConnectiveInstance::VarRefBody(VarRef {
                                index: index as i32,
                                depth: depth as i32,
                            })),
                        },
                        input.bound_map_chain.depth() as i32,
                    ),
                    free_map: input.free_map,
                }),

                _ => Err(InterpreterError::UnexpectedProcContext {
                    var_name: p.var.name.clone(),
                    name_var_source_position: source_position,
                    process_source_position: SourcePosition {
                        row: p.line_num,
                        column: p.col_num,
                    },
                }),
            },
            VarSort::NameSort => match p.var_ref_kind {
                VarRefKind::Name => Ok(ProcVisitOutputs {
                    par: prepend_connective(
                        input.par,
                        Connective {
                            connective_instance: Some(ConnectiveInstance::VarRefBody(VarRef {
                                index: index as i32,
                                depth: depth as i32,
                            })),
                        },
                        input.bound_map_chain.depth() as i32,
                    ),
                    free_map: input.free_map,
                }),

                _ => Err(InterpreterError::UnexpectedProcContext {
                    var_name: p.var.name.clone(),
                    name_var_source_position: source_position,
                    process_source_position: SourcePosition {
                        row: p.line_num,
                        column: p.col_num,
                    },
                }),
            },
        },

        None => Err(InterpreterError::UnboundVariableRef {
            var_name: p.var.name.clone(),
            line: p.line_num,
            col: p.col_num,
        }),
    }
}

// ============================================================================
// NEW AST PARALLEL FUNCTIONS
// ============================================================================

/// Parallel version of normalize_p_var_ref for new AST VarRef structure
/// Handles VarRef { kind: VarRefKind, var: Id } instead of old VarRef struct
pub fn normalize_p_var_ref_new_ast(
    var_ref_kind: NewVarRefKind,
    var_id: &Id,
    input: ProcVisitInputsSpan,
    var_ref_span: rholang_parser::SourceSpan,
) -> Result<ProcVisitOutputsSpan, InterpreterError> {
    match input.bound_map_chain.find(var_id.name) {
        Some((
            BoundContextSpan {
                index,
                typ,
                source_span,
            },
            depth,
        )) => match typ {
            VarSort::ProcSort => match var_ref_kind {
                NewVarRefKind::Proc => Ok(ProcVisitOutputsSpan {
                    par: prepend_connective(
                        input.par,
                        Connective {
                            connective_instance: Some(ConnectiveInstance::VarRefBody(VarRef {
                                index: index as i32,
                                depth: depth as i32,
                            })),
                        },
                        input.bound_map_chain.depth() as i32,
                    ),
                    free_map: input.free_map,
                }),

                _ => Err(InterpreterError::UnexpectedProcContextSpan {
                    var_name: var_id.name.to_string(),
                    name_var_source_span: source_span,
                    process_source_span: var_ref_span,
                }),
            },
            VarSort::NameSort => match var_ref_kind {
                NewVarRefKind::Name => Ok(ProcVisitOutputsSpan {
                    par: prepend_connective(
                        input.par,
                        Connective {
                            connective_instance: Some(ConnectiveInstance::VarRefBody(VarRef {
                                index: index as i32,
                                depth: depth as i32,
                            })),
                        },
                        input.bound_map_chain.depth() as i32,
                    ),
                    free_map: input.free_map,
                }),

                _ => Err(InterpreterError::UnexpectedNameContextSpan {
                    var_name: var_id.name.to_string(),
                    proc_var_source_span: source_span,
                    name_source_span: var_ref_span,
                }),
            },
        },

        None => Err(InterpreterError::UnboundVariableRefSpan {
            var_name: var_id.name.to_string(),
            source_span: var_ref_span,
        }),
    }
}

#[cfg(test)]
mod tests {
    use crate::rust::interpreter::compiler::normalize::normalize_match_proc;
    use crate::rust::interpreter::compiler::normalize::VarSort::{NameSort, ProcSort};
    use crate::rust::interpreter::compiler::rholang_ast::Proc::Match;
    use crate::rust::interpreter::compiler::rholang_ast::{
        Block, Case, LinearBind, Name, Names, Proc, Quote, Receipt, Receipts, Source, Var, VarRef,
        VarRefKind,
    };
    use crate::rust::interpreter::test_utils::utils::{
        proc_visit_inputs_and_env, proc_visit_inputs_and_env_span,
        proc_visit_inputs_with_updated_bound_map_chain,
        proc_visit_inputs_with_updated_bound_map_chain_span,
    };
    use models::create_bit_vector;
    use models::rhoapi::connective::ConnectiveInstance::VarRefBody;
    use models::rhoapi::{Connective, Match as model_match, MatchCase, Par, ReceiveBind};
    use models::rhoapi::{Receive, VarRef as model_VarRef};
    use models::rust::utils::new_gint_par;
    use pretty_assertions::assert_eq;

    // New AST test imports (for future blocked tests)
    // use super::normalize_p_var_ref_new_ast;
    // use crate::rust::interpreter::errors::InterpreterError;
    // use models::rhoapi::connective::ConnectiveInstance;
    // use rholang_parser::ast::{Id, VarRefKind as NewVarRefKind};
    // use rholang_parser::SourcePos;

    #[test]
    fn p_var_ref_should_do_deep_lookup_in_match_case() {
        // assuming `x` is bound
        // example: @7!(10) | for (@x <- @7) { … }
        // match 7 { =x => Nil }

        let (inputs, env) = proc_visit_inputs_and_env();
        let bound_inputs =
            proc_visit_inputs_with_updated_bound_map_chain(inputs.clone(), "x", ProcSort);

        let proc = Match {
            expression: Box::new(Proc::new_proc_int(7)),
            cases: vec![Case::new(
                Proc::VarRef(VarRef {
                    var_ref_kind: VarRefKind::Proc,
                    var: Var::new("x".to_string()),
                    line_num: 0,
                    col_num: 0,
                }),
                Proc::new_proc_nil(),
            )],
            line_num: 0,
            col_num: 0,
        };

        let result = normalize_match_proc(&proc, bound_inputs.clone(), &env);
        let expected_result = bound_inputs
            .par
            .clone()
            .with_matches(vec![
                (model_match {
                    target: Some(new_gint_par(7, Vec::new(), false)),

                    cases: vec![MatchCase {
                        pattern: Some(
                            Par {
                                connectives: vec![Connective {
                                    connective_instance: Some(VarRefBody(model_VarRef {
                                        index: 0,
                                        depth: 1,
                                    })),
                                }],
                                ..Par::default().clone()
                            }
                            .with_locally_free(create_bit_vector(&vec![0])),
                        ),
                        source: Some(Par::default()),
                        free_count: 0,
                    }],

                    locally_free: create_bit_vector(&vec![0]),
                    connective_used: false,
                }),
            ])
            .with_locally_free(create_bit_vector(&vec![0]));
        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.clone().unwrap().free_map, inputs.free_map);
        // Make sure that variable references in patterns are reflected
        // BitSet(0) == create_bit_vector(&vec![0])
        assert_eq!(
            result.clone().unwrap().par.locally_free,
            create_bit_vector(&vec![0])
        );
    }

    #[test]
    fn p_var_ref_should_do_deep_lookup_in_receive_case() {
        // assuming `x` is bound:
        // example : new x in { … }
        // for(@{=*x} <- @Nil) { Nil }

        let (inputs, env) = proc_visit_inputs_and_env();
        let bound_inputs =
            proc_visit_inputs_with_updated_bound_map_chain(inputs.clone(), "x", NameSort);

        let proc = Proc::Input {
            formals: Receipts::new(vec![Receipt::LinearBinds(LinearBind::new_linear_bind(
                Names {
                    names: vec![Name::new_name_quote_var_ref("x")],
                    cont: None,
                    line_num: 0,
                    col_num: 0,
                },
                Source::new_simple_source(Name::Quote(Box::new(Quote {
                    quotable: Box::new(Proc::new_proc_nil()),
                    line_num: 0,
                    col_num: 0,
                }))),
            ))]),
            proc: Box::new(Block::new_block_nil()),
            line_num: 0,
            col_num: 0,
        };

        let result = normalize_match_proc(&proc, bound_inputs.clone(), &env);
        let expected_result = inputs
            .par
            .clone()
            .with_receives(vec![Receive {
                binds: vec![ReceiveBind {
                    patterns: vec![Par {
                        connectives: vec![Connective {
                            connective_instance: Some(VarRefBody(model_VarRef {
                                index: 0,
                                depth: 1,
                            })),
                        }],
                        ..Par::default().clone()
                    }
                    .with_locally_free(create_bit_vector(&vec![0]))],
                    source: Some(Par::default()),
                    remainder: None,
                    free_count: 0,
                }],
                body: Some(Par::default()),
                persistent: false,
                peek: false,
                bind_count: 0,
                locally_free: create_bit_vector(&vec![0]),
                connective_used: false,
            }])
            .with_locally_free(create_bit_vector(&vec![0]));

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.clone().unwrap().free_map, inputs.free_map);
        assert_eq!(
            result.unwrap().par.locally_free,
            create_bit_vector(&vec![0])
        )
    }

    // ============================================================================
    // NEW AST PARALLEL TESTS - EXACT MAPPING TO ORIGINAL TESTS
    // ============================================================================

    use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
    use rholang_parser::ast::{
        AnnName as NewAnnName, AnnProc as NewAnnProc, Bind as NewBind, Case as NewCase, Id,
        Name as NewName, Names as NewNames, Proc as NewProc, Source as NewSource,
        VarRefKind as NewVarRefKind,
    };
    use rholang_parser::{SourcePos, SourceSpan};

    #[test]
    fn new_ast_p_var_ref_should_do_deep_lookup_in_match_case() {
        // Maps to: p_var_ref_should_do_deep_lookup_in_match_case (line 191)
        // Purpose: Tests VarRef lookup in Match case patterns
        // Logic: match 7 { =x => Nil } with x bound as ProcSort
        // Expected: VarRef connective with index=0, depth=1

        let (inputs, env) = proc_visit_inputs_and_env_span();
        let bound_inputs =
            proc_visit_inputs_with_updated_bound_map_chain_span(inputs.clone(), "x", ProcSort);
        let parser = rholang_parser::RholangParser::new();

        // Create: match 7 { =x => Nil }
        let match_proc = NewAnnProc {
            proc: Box::leak(Box::new(NewProc::Match {
                expression: NewAnnProc {
                    proc: Box::leak(Box::new(NewProc::LongLiteral(7))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                cases: vec![NewCase {
                    pattern: NewAnnProc {
                        proc: Box::leak(Box::new(NewProc::VarRef {
                            kind: NewVarRefKind::Proc,
                            var: Id {
                                name: "x",
                                pos: SourcePos { line: 0, col: 0 },
                            },
                        })),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    proc: NewAnnProc {
                        proc: Box::leak(Box::new(NewProc::Nil)),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                }],
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let result = normalize_ann_proc(&match_proc, bound_inputs.clone(), &env, &parser);
        let expected_result = bound_inputs
            .par
            .clone()
            .with_matches(vec![
                (model_match {
                    target: Some(new_gint_par(7, Vec::new(), false)),

                    cases: vec![MatchCase {
                        pattern: Some(
                            Par {
                                connectives: vec![Connective {
                                    connective_instance: Some(VarRefBody(model_VarRef {
                                        index: 0,
                                        depth: 1,
                                    })),
                                }],
                                ..Par::default().clone()
                            }
                            .with_locally_free(create_bit_vector(&vec![0])),
                        ),
                        source: Some(Par::default()),
                        free_count: 0,
                    }],

                    locally_free: create_bit_vector(&vec![0]),
                    connective_used: false,
                }),
            ])
            .with_locally_free(create_bit_vector(&vec![0]));

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.clone().unwrap().free_map, inputs.free_map);
        // Make sure that variable references in patterns are reflected
        // BitSet(0) == create_bit_vector(&vec![0])
        assert_eq!(
            result.clone().unwrap().par.locally_free,
            create_bit_vector(&vec![0])
        );
    }

    #[test]
    fn new_ast_p_var_ref_should_do_deep_lookup_in_receive_case() {
        // Maps to: p_var_ref_should_do_deep_lookup_in_receive_case (line 256)
        // Purpose: Tests VarRef lookup in receive pattern with quoted name variable reference
        // Logic: for(@{=*x} <- @Nil) { Nil } with x bound as NameSort
        // Expected: VarRef connective with index=0, depth=1

        let (inputs, env) = proc_visit_inputs_and_env_span();
        let bound_inputs =
            proc_visit_inputs_with_updated_bound_map_chain_span(inputs.clone(), "x", NameSort);
        let parser = rholang_parser::RholangParser::new();

        // Create: for(@{=*x} <- @Nil) { Nil }
        // This is a complex structure: quoted name with VarRef inside
        let for_comprehension = NewAnnProc {
            proc: Box::leak(Box::new(NewProc::ForComprehension {
                receipts: smallvec::SmallVec::from_vec(vec![smallvec::SmallVec::from_vec(vec![
                    NewBind::Linear {
                        lhs: NewNames {
                            names: smallvec::SmallVec::from_vec(vec![NewAnnName {
                                name: NewName::Quote(Box::leak(Box::new(NewProc::VarRef {
                                    kind: NewVarRefKind::Name,
                                    var: Id {
                                        name: "x",
                                        pos: SourcePos { line: 0, col: 0 },
                                    },
                                }))),
                                span: SourceSpan {
                                    start: SourcePos { line: 0, col: 0 },
                                    end: SourcePos { line: 0, col: 0 },
                                },
                            }]),
                            remainder: None,
                        },
                        rhs: NewSource::Simple {
                            name: NewAnnName {
                                name: NewName::Quote(Box::leak(Box::new(NewProc::Nil))),
                                span: SourceSpan {
                                    start: SourcePos { line: 0, col: 0 },
                                    end: SourcePos { line: 0, col: 0 },
                                },
                            },
                        },
                    },
                ])]),
                proc: NewAnnProc {
                    proc: Box::leak(Box::new(NewProc::Nil)),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let result = normalize_ann_proc(&for_comprehension, bound_inputs.clone(), &env, &parser);
        let expected_result = inputs
            .par
            .clone()
            .with_receives(vec![Receive {
                binds: vec![ReceiveBind {
                    patterns: vec![Par {
                        connectives: vec![Connective {
                            connective_instance: Some(VarRefBody(model_VarRef {
                                index: 0,
                                depth: 1,
                            })),
                        }],
                        ..Par::default().clone()
                    }
                    .with_locally_free(create_bit_vector(&vec![0]))],
                    source: Some(Par::default()),
                    remainder: None,
                    free_count: 0,
                }],
                body: Some(Par::default()),
                persistent: false,
                peek: false,
                bind_count: 0,
                locally_free: create_bit_vector(&vec![0]),
                connective_used: false,
            }])
            .with_locally_free(create_bit_vector(&vec![0]));

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.clone().unwrap().free_map, inputs.free_map);
        assert_eq!(
            result.unwrap().par.locally_free,
            create_bit_vector(&vec![0])
        )
    }
}
