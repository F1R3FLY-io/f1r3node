use crate::rust::interpreter::compiler::exports::{
    BoundMapChainSpan, IdContextPos, ProcVisitInputsSpan, ProcVisitOutputsSpan,
};
use crate::rust::interpreter::compiler::normalize::{normalize_ann_proc, VarSort};
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::util::filter_and_adjust_bitset;
use crate::rust::interpreter::util::prepend_new;
use models::rhoapi::{New, Par};
use std::collections::{BTreeMap, HashMap};

use rholang_parser::ast::{AnnProc, NameDecl};
use rholang_parser::SourcePos;

pub fn normalize_p_new_new_ast<'ast>(
    decls: &[NameDecl<'ast>],
    proc: &'ast AnnProc<'ast>,
    input: ProcVisitInputsSpan,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputsSpan, InterpreterError> {
    // TODO: bindings within a single new shouldn't have overlapping names. - OLD
    let new_tagged_bindings: Vec<(Option<String>, String, VarSort, usize, usize)> = decls
        .iter()
        .map(|decl| match decl {
            NameDecl { id, uri: None } => Ok((
                None,
                id.name.to_string(),
                VarSort::NameSort,
                id.pos.line,
                id.pos.col,
            )),
            NameDecl { id, uri: Some(urn) } => Ok((
                Some((**urn).to_string()), // Dereference Uri to get the inner &str
                id.name.to_string(),
                VarSort::NameSort,
                id.pos.line,
                id.pos.col,
            )),
        })
        .collect::<Result<Vec<_>, InterpreterError>>()?;

    // Sort bindings: None's first, then URI's lexicographically
    let mut sorted_bindings: Vec<(Option<String>, String, VarSort, usize, usize)> =
        new_tagged_bindings;
    sorted_bindings.sort_by(|a, b| a.0.cmp(&b.0));

    let new_bindings: Vec<IdContextPos<VarSort>> = sorted_bindings
        .iter()
        .map(|row| {
            (
                row.1.clone(),
                row.2.clone(),
                SourcePos {
                    line: row.3,
                    col: row.4,
                },
            )
        })
        .collect();

    let uris: Vec<String> = sorted_bindings
        .iter()
        .filter_map(|row| row.0.clone())
        .collect();

    let new_env: BoundMapChainSpan<VarSort> = input.bound_map_chain.put_all_pos(new_bindings);
    let new_count: usize = new_env.get_count() - input.bound_map_chain.get_count();

    let body_result = normalize_ann_proc(
        proc,
        ProcVisitInputsSpan {
            par: Par::default(),
            bound_map_chain: new_env.clone(),
            free_map: input.free_map.clone(),
        },
        env,
        parser,
    )?;

    // TODO: we should build btree_map with real values, not a copied references from env: ref &HashMap
    let btree_map: BTreeMap<String, Par> =
        env.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

    let result_new = New {
        bind_count: new_count as i32,
        p: Some(body_result.par.clone()),
        uri: uris,
        injections: btree_map,
        locally_free: filter_and_adjust_bitset(body_result.par.clone().locally_free, new_count),
    };

    Ok(ProcVisitOutputsSpan {
        par: prepend_new(input.par.clone(), result_new),
        free_map: body_result.free_map.clone(),
    })
}

// See rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use models::{
        create_bit_vector,
        rhoapi::{New, Par},
        rust::utils::{new_boundvar_par, new_gint_par, new_send},
    };

    use crate::rust::interpreter::{
        test_utils::utils::proc_visit_inputs_and_env_span, util::prepend_new,
    };

    #[test]
    fn p_new_should_bind_new_variables() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{AnnName, AnnProc, Id, Name, NameDecl, Proc, SendType, Var};
        use rholang_parser::{SourcePos, SourceSpan};

        let p_new = AnnProc {
            proc: Box::leak(Box::new(Proc::New {
                decls: vec![
                    NameDecl {
                        id: Id {
                            name: "x",
                            pos: SourcePos { line: 0, col: 0 },
                        },
                        uri: None,
                    },
                    NameDecl {
                        id: Id {
                            name: "y",
                            pos: SourcePos { line: 0, col: 0 },
                        },
                        uri: None,
                    },
                    NameDecl {
                        id: Id {
                            name: "z",
                            pos: SourcePos { line: 0, col: 0 },
                        },
                        uri: None,
                    },
                ],
                proc: AnnProc {
                    proc: Box::leak(Box::new(Proc::Par {
                        left: AnnProc {
                            proc: Box::leak(Box::new(Proc::Par {
                                left: AnnProc {
                                    proc: Box::leak(Box::new(Proc::Send {
                                        channel: AnnName {
                                            name: Name::ProcVar(Var::Id(Id {
                                                name: "x",
                                                pos: SourcePos { line: 0, col: 0 },
                                            })),
                                            span: SourceSpan {
                                                start: SourcePos { line: 0, col: 0 },
                                                end: SourcePos { line: 0, col: 0 },
                                            },
                                        },
                                        send_type: SendType::Single,
                                        inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                                            proc: Box::leak(Box::new(Proc::LongLiteral(7))),
                                            span: SourceSpan {
                                                start: SourcePos { line: 0, col: 0 },
                                                end: SourcePos { line: 0, col: 0 },
                                            },
                                        }]),
                                    })),
                                    span: SourceSpan {
                                        start: SourcePos { line: 0, col: 0 },
                                        end: SourcePos { line: 0, col: 0 },
                                    },
                                },
                                right: AnnProc {
                                    proc: Box::leak(Box::new(Proc::Send {
                                        channel: AnnName {
                                            name: Name::ProcVar(Var::Id(Id {
                                                name: "y",
                                                pos: SourcePos { line: 0, col: 0 },
                                            })),
                                            span: SourceSpan {
                                                start: SourcePos { line: 0, col: 0 },
                                                end: SourcePos { line: 0, col: 0 },
                                            },
                                        },
                                        send_type: SendType::Single,
                                        inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                                            proc: Box::leak(Box::new(Proc::LongLiteral(8))),
                                            span: SourceSpan {
                                                start: SourcePos { line: 0, col: 0 },
                                                end: SourcePos { line: 0, col: 0 },
                                            },
                                        }]),
                                    })),
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
                        },
                        right: AnnProc {
                            proc: Box::leak(Box::new(Proc::Send {
                                channel: AnnName {
                                    name: Name::ProcVar(Var::Id(Id {
                                        name: "z",
                                        pos: SourcePos { line: 0, col: 0 },
                                    })),
                                    span: SourceSpan {
                                        start: SourcePos { line: 0, col: 0 },
                                        end: SourcePos { line: 0, col: 0 },
                                    },
                                },
                                send_type: SendType::Single,
                                inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                                    proc: Box::leak(Box::new(Proc::LongLiteral(9))),
                                    span: SourceSpan {
                                        start: SourcePos { line: 0, col: 0 },
                                        end: SourcePos { line: 0, col: 0 },
                                    },
                                }]),
                            })),
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
                },
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(
            &p_new,
            proc_visit_inputs_and_env_span().0,
            &proc_visit_inputs_and_env_span().1,
            &parser,
        );
        assert!(result.is_ok());

        let expected_result = prepend_new(
            Par::default(),
            New {
                bind_count: 3,
                p: Some(
                    Par::default()
                        .prepend_send(new_send(
                            new_boundvar_par(2, create_bit_vector(&vec![2]), false),
                            vec![new_gint_par(7, Vec::new(), false)],
                            false,
                            create_bit_vector(&vec![2]),
                            false,
                        ))
                        .prepend_send(new_send(
                            new_boundvar_par(1, create_bit_vector(&vec![1]), false),
                            vec![new_gint_par(8, Vec::new(), false)],
                            false,
                            create_bit_vector(&vec![1]),
                            false,
                        ))
                        .prepend_send(new_send(
                            new_boundvar_par(0, create_bit_vector(&vec![0]), false),
                            vec![new_gint_par(9, Vec::new(), false)],
                            false,
                            create_bit_vector(&vec![0]),
                            false,
                        )),
                ),
                uri: Vec::new(),
                injections: BTreeMap::new(),
                locally_free: Vec::new(),
            },
        );

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(
            result.unwrap().free_map,
            proc_visit_inputs_and_env_span().0.free_map
        );
    }

    #[test]
    fn p_new_should_sort_uris_and_place_them_at_the_end() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{AnnName, AnnProc, Id, Name, NameDecl, Proc, SendType, Uri, Var};
        use rholang_parser::{SourcePos, SourceSpan};

        let p_new = AnnProc {
            proc: Box::leak(Box::new(Proc::New {
                decls: vec![
                    NameDecl {
                        id: Id {
                            name: "x",
                            pos: SourcePos { line: 0, col: 0 },
                        },
                        uri: None,
                    },
                    NameDecl {
                        id: Id {
                            name: "y",
                            pos: SourcePos { line: 0, col: 0 },
                        },
                        uri: None,
                    },
                    NameDecl {
                        id: Id {
                            name: "r",
                            pos: SourcePos { line: 0, col: 0 },
                        },
                        uri: Some(Uri::from("rho:registry")),
                    },
                    NameDecl {
                        id: Id {
                            name: "out",
                            pos: SourcePos { line: 0, col: 0 },
                        },
                        uri: Some(Uri::from("rho:stdout")),
                    },
                    NameDecl {
                        id: Id {
                            name: "z",
                            pos: SourcePos { line: 0, col: 0 },
                        },
                        uri: None,
                    },
                ],
                proc: AnnProc {
                    proc: Box::leak(Box::new(Proc::Par {
                        left: AnnProc {
                            proc: Box::leak(Box::new(Proc::Par {
                                left: AnnProc {
                                    proc: Box::leak(Box::new(Proc::Par {
                                        left: AnnProc {
                                            proc: Box::leak(Box::new(Proc::Par {
                                                left: AnnProc {
                                                    proc: Box::leak(Box::new(Proc::Send {
                                                        channel: AnnName {
                                                            name: Name::ProcVar(Var::Id(Id {
                                                                name: "x",
                                                                pos: SourcePos { line: 0, col: 0 },
                                                            })),
                                                            span: SourceSpan {
                                                                start: SourcePos {
                                                                    line: 0,
                                                                    col: 0,
                                                                },
                                                                end: SourcePos { line: 0, col: 0 },
                                                            },
                                                        },
                                                        send_type: SendType::Single,
                                                        inputs: smallvec::SmallVec::from_vec(vec![
                                                            AnnProc {
                                                                proc: Box::leak(Box::new(
                                                                    Proc::LongLiteral(7),
                                                                )),
                                                                span: SourceSpan {
                                                                    start: SourcePos {
                                                                        line: 0,
                                                                        col: 0,
                                                                    },
                                                                    end: SourcePos {
                                                                        line: 0,
                                                                        col: 0,
                                                                    },
                                                                },
                                                            },
                                                        ]),
                                                    })),
                                                    span: SourceSpan {
                                                        start: SourcePos { line: 0, col: 0 },
                                                        end: SourcePos { line: 0, col: 0 },
                                                    },
                                                },
                                                right: AnnProc {
                                                    proc: Box::leak(Box::new(Proc::Send {
                                                        channel: AnnName {
                                                            name: Name::ProcVar(Var::Id(Id {
                                                                name: "y",
                                                                pos: SourcePos { line: 0, col: 0 },
                                                            })),
                                                            span: SourceSpan {
                                                                start: SourcePos {
                                                                    line: 0,
                                                                    col: 0,
                                                                },
                                                                end: SourcePos { line: 0, col: 0 },
                                                            },
                                                        },
                                                        send_type: SendType::Single,
                                                        inputs: smallvec::SmallVec::from_vec(vec![
                                                            AnnProc {
                                                                proc: Box::leak(Box::new(
                                                                    Proc::LongLiteral(8),
                                                                )),
                                                                span: SourceSpan {
                                                                    start: SourcePos {
                                                                        line: 0,
                                                                        col: 0,
                                                                    },
                                                                    end: SourcePos {
                                                                        line: 0,
                                                                        col: 0,
                                                                    },
                                                                },
                                                            },
                                                        ]),
                                                    })),
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
                                        },
                                        right: AnnProc {
                                            proc: Box::leak(Box::new(Proc::Send {
                                                channel: AnnName {
                                                    name: Name::ProcVar(Var::Id(Id {
                                                        name: "r",
                                                        pos: SourcePos { line: 0, col: 0 },
                                                    })),
                                                    span: SourceSpan {
                                                        start: SourcePos { line: 0, col: 0 },
                                                        end: SourcePos { line: 0, col: 0 },
                                                    },
                                                },
                                                send_type: SendType::Single,
                                                inputs: smallvec::SmallVec::from_vec(vec![
                                                    AnnProc {
                                                        proc: Box::leak(Box::new(
                                                            Proc::LongLiteral(9),
                                                        )),
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
                                        },
                                    })),
                                    span: SourceSpan {
                                        start: SourcePos { line: 0, col: 0 },
                                        end: SourcePos { line: 0, col: 0 },
                                    },
                                },
                                right: AnnProc {
                                    proc: Box::leak(Box::new(Proc::Send {
                                        channel: AnnName {
                                            name: Name::ProcVar(Var::Id(Id {
                                                name: "out",
                                                pos: SourcePos { line: 0, col: 0 },
                                            })),
                                            span: SourceSpan {
                                                start: SourcePos { line: 0, col: 0 },
                                                end: SourcePos { line: 0, col: 0 },
                                            },
                                        },
                                        send_type: SendType::Single,
                                        inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                                            proc: Box::leak(Box::new(Proc::LongLiteral(10))),
                                            span: SourceSpan {
                                                start: SourcePos { line: 0, col: 0 },
                                                end: SourcePos { line: 0, col: 0 },
                                            },
                                        }]),
                                    })),
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
                        },
                        right: AnnProc {
                            proc: Box::leak(Box::new(Proc::Send {
                                channel: AnnName {
                                    name: Name::ProcVar(Var::Id(Id {
                                        name: "z",
                                        pos: SourcePos { line: 0, col: 0 },
                                    })),
                                    span: SourceSpan {
                                        start: SourcePos { line: 0, col: 0 },
                                        end: SourcePos { line: 0, col: 0 },
                                    },
                                },
                                send_type: SendType::Single,
                                inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                                    proc: Box::leak(Box::new(Proc::LongLiteral(11))),
                                    span: SourceSpan {
                                        start: SourcePos { line: 0, col: 0 },
                                        end: SourcePos { line: 0, col: 0 },
                                    },
                                }]),
                            })),
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
                },
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(
            &p_new,
            proc_visit_inputs_and_env_span().0,
            &proc_visit_inputs_and_env_span().1,
            &parser,
        );
        assert!(result.is_ok());

        let expected_result = prepend_new(
            Par::default(),
            New {
                bind_count: 5,
                p: Some(
                    Par::default()
                        .prepend_send(new_send(
                            new_boundvar_par(4, create_bit_vector(&vec![4]), false),
                            vec![new_gint_par(7, Vec::new(), false)],
                            false,
                            create_bit_vector(&vec![4]),
                            false,
                        ))
                        .prepend_send(new_send(
                            new_boundvar_par(3, create_bit_vector(&vec![3]), false),
                            vec![new_gint_par(8, Vec::new(), false)],
                            false,
                            create_bit_vector(&vec![3]),
                            false,
                        ))
                        .prepend_send(new_send(
                            new_boundvar_par(1, create_bit_vector(&vec![1]), false),
                            vec![new_gint_par(9, Vec::new(), false)],
                            false,
                            create_bit_vector(&vec![1]),
                            false,
                        ))
                        .prepend_send(new_send(
                            new_boundvar_par(0, create_bit_vector(&vec![0]), false),
                            vec![new_gint_par(10, Vec::new(), false)],
                            false,
                            create_bit_vector(&vec![0]),
                            false,
                        ))
                        .prepend_send(new_send(
                            new_boundvar_par(2, create_bit_vector(&vec![2]), false),
                            vec![new_gint_par(11, Vec::new(), false)],
                            false,
                            create_bit_vector(&vec![2]),
                            false,
                        )),
                ),
                uri: vec!["rho:registry".to_string(), "rho:stdout".to_string()],
                injections: BTreeMap::new(),
                locally_free: Vec::new(),
            },
        );

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(
            result.clone().unwrap().par.news[0]
                .p
                .clone()
                .unwrap()
                .sends
                .into_iter()
                .map(|x| x.locally_free)
                .collect::<Vec<Vec<u8>>>(),
            vec![
                create_bit_vector(&vec![2]),
                create_bit_vector(&vec![0]),
                create_bit_vector(&vec![1]),
                create_bit_vector(&vec![3]),
                create_bit_vector(&vec![4])
            ]
        );
        assert_eq!(
            result.unwrap().par.news[0].p.clone().unwrap().locally_free,
            create_bit_vector(&vec![0, 1, 2, 3, 4])
        );
    }
}
