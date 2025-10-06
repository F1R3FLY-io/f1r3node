use crate::rust::interpreter::compiler::exports::BoundContextSpan;
use crate::rust::interpreter::compiler::exports::{ProcVisitInputsSpan, ProcVisitOutputsSpan};
use crate::rust::interpreter::compiler::normalize::VarSort;
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::util::prepend_connective;
use models::rhoapi::connective::ConnectiveInstance;
use models::rhoapi::{Connective, VarRef};
use std::result::Result;

use rholang_parser::ast::{Id, VarRefKind};

pub fn normalize_p_var_ref_new_ast(
    var_ref_kind: VarRefKind,
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
                VarRefKind::Proc => Ok(ProcVisitOutputsSpan {
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
                VarRefKind::Name => Ok(ProcVisitOutputsSpan {
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
    use crate::rust::interpreter::compiler::normalize::VarSort::{NameSort, ProcSort};
    use crate::rust::interpreter::test_utils::utils::{
        proc_visit_inputs_and_env_span, proc_visit_inputs_with_updated_bound_map_chain_span,
    };
    use models::create_bit_vector;
    use models::rhoapi::connective::ConnectiveInstance::VarRefBody;
    use models::rhoapi::{Connective, Match as model_match, MatchCase, Par, ReceiveBind};
    use models::rhoapi::{Receive, VarRef as model_VarRef};
    use models::rust::utils::new_gint_par;
    use pretty_assertions::assert_eq;

    use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
    use rholang_parser::ast::{
        AnnName, AnnProc, Bind, Case, Id, Name, Names, Proc, Source, VarRefKind,
    };
    use rholang_parser::{SourcePos, SourceSpan};

    #[test]
    fn p_var_ref_should_do_deep_lookup_in_match_case() {
        let (inputs, env) = proc_visit_inputs_and_env_span();
        let bound_inputs =
            proc_visit_inputs_with_updated_bound_map_chain_span(inputs.clone(), "x", ProcSort);
        let parser = rholang_parser::RholangParser::new();

        let match_proc = AnnProc {
            proc: Box::leak(Box::new(Proc::Match {
                expression: AnnProc {
                    proc: Box::leak(Box::new(Proc::LongLiteral(7))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                cases: vec![Case {
                    pattern: AnnProc {
                        proc: Box::leak(Box::new(Proc::VarRef {
                            kind: VarRefKind::Proc,
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
                    proc: AnnProc {
                        proc: Box::leak(Box::new(Proc::Nil)),
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
        assert_eq!(
            result.clone().unwrap().par.locally_free,
            create_bit_vector(&vec![0])
        );
    }

    #[test]
    fn p_var_ref_should_do_deep_lookup_in_receive_case() {
        let (inputs, env) = proc_visit_inputs_and_env_span();
        let bound_inputs =
            proc_visit_inputs_with_updated_bound_map_chain_span(inputs.clone(), "x", NameSort);
        let parser = rholang_parser::RholangParser::new();

        let for_comprehension = AnnProc {
            proc: Box::leak(Box::new(Proc::ForComprehension {
                receipts: smallvec::SmallVec::from_vec(vec![smallvec::SmallVec::from_vec(vec![
                    Bind::Linear {
                        lhs: Names {
                            names: smallvec::SmallVec::from_vec(vec![AnnName {
                                name: Name::Quote(Box::leak(Box::new(Proc::VarRef {
                                    kind: VarRefKind::Name,
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
                        rhs: Source::Simple {
                            name: AnnName {
                                name: Name::Quote(Box::leak(Box::new(Proc::Nil))),
                                span: SourceSpan {
                                    start: SourcePos { line: 0, col: 0 },
                                    end: SourcePos { line: 0, col: 0 },
                                },
                            },
                        },
                    },
                ])]),
                proc: AnnProc {
                    proc: Box::leak(Box::new(Proc::Nil)),
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
