use crate::rust::interpreter::compiler::exports::{
    BoundMapChain, BoundMapChainSpan, IdContext, IdContextPos, ProcVisitInputsSpan,
    ProcVisitOutputsSpan, SourcePosition,
};
use crate::rust::interpreter::compiler::normalize::{
    normalize_ann_proc, normalize_match_proc, ProcVisitInputs, ProcVisitOutputs, VarSort,
};
use crate::rust::interpreter::compiler::rholang_ast::{Decls, NameDecl, Proc};
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::util::filter_and_adjust_bitset;
use crate::rust::interpreter::util::prepend_new;
use models::rhoapi::{New, Par};
use rholang_parser::ast::AnnProc;
use rholang_parser::SourcePos;
use std::collections::{BTreeMap, HashMap};

pub fn normalize_p_new(
    decls: &Decls,
    proc: &Box<Proc>,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    // TODO: bindings within a single new shouldn't have overlapping names. - OLD
    let new_tagged_bindings: Vec<(Option<String>, String, VarSort, usize, usize)> = decls
        .decls
        .iter()
        .map(|decl| match decl {
            NameDecl {
                var,
                uri: None,
                line_num,
                col_num,
            } => Ok((
                None,
                var.name.clone(),
                VarSort::NameSort,
                *line_num,
                *col_num,
            )),
            NameDecl {
                var,
                uri: Some(urn),
                line_num,
                col_num,
            } => Ok((
                Some(urn.value.clone()),
                var.name.clone(),
                VarSort::NameSort,
                *line_num,
                *col_num,
            )),
        })
        .collect::<Result<Vec<_>, InterpreterError>>()?;

    // Sort bindings: None's first, then URI's lexicographically
    let mut sorted_bindings: Vec<(Option<String>, String, VarSort, usize, usize)> =
        new_tagged_bindings;
    sorted_bindings.sort_by(|a, b| a.0.cmp(&b.0));

    let new_bindings: Vec<IdContext<VarSort>> = sorted_bindings
        .iter()
        .map(|row| {
            (
                row.1.clone(),
                row.2.clone(),
                SourcePosition::new(row.3, row.4),
            )
        })
        .collect();

    let uris: Vec<String> = sorted_bindings
        .iter()
        .filter_map(|row| row.0.clone())
        .collect();

    let new_env: BoundMapChain<VarSort> = input.bound_map_chain.put_all(new_bindings);
    let new_count: usize = new_env.get_count() - input.bound_map_chain.get_count();

    let body_result = normalize_match_proc(
        &proc,
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: new_env.clone(),
            free_map: input.free_map.clone(),
        },
        env,
    )?;

    //we should build btree_map with real values, not a copied references from env: ref &HashMap
    let btree_map: BTreeMap<String, Par> =
        env.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

    let result_new = New {
        bind_count: new_count as i32,
        p: Some(body_result.par.clone()),
        uri: uris,
        injections: btree_map,
        locally_free: filter_and_adjust_bitset(body_result.par.clone().locally_free, new_count),
    };

    Ok(ProcVisitOutputs {
        par: prepend_new(input.par.clone(), result_new),
        free_map: body_result.free_map.clone(),
    })
}

/// Parallel version of normalize_p_new for new AST New
pub fn normalize_p_new_new_ast<'ast>(
    decls: &[rholang_parser::ast::NameDecl<'ast>],
    proc: &'ast AnnProc<'ast>,
    input: ProcVisitInputsSpan,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputsSpan, InterpreterError> {
    // TODO: bindings within a single new shouldn't have overlapping names. - OLD
    let new_tagged_bindings: Vec<(Option<String>, String, VarSort, usize, usize)> = decls
        .iter()
        .map(|decl| match decl {
            rholang_parser::ast::NameDecl { id, uri: None } => Ok((
                None,
                id.name.to_string(),
                VarSort::NameSort,
                id.pos.line,
                id.pos.col,
            )),
            rholang_parser::ast::NameDecl { id, uri: Some(urn) } => Ok((
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
        compiler::{
            normalize::normalize_match_proc,
            rholang_ast::{Decls, Name, NameDecl, Proc, ProcList, SendType},
        },
        test_utils::utils::{proc_visit_inputs_and_env, proc_visit_inputs_and_env_span},
        util::prepend_new,
    };

    #[test]
    fn p_new_should_bind_new_variables() {
        let p_new = Proc::New {
            decls: Decls {
                decls: vec![
                    NameDecl::new("x", None),
                    NameDecl::new("y", None),
                    NameDecl::new("z", None),
                ],
                line_num: 0,
                col_num: 0,
            },
            proc: Box::new(Proc::Par {
                left: Box::new(Proc::Par {
                    left: Box::new(Proc::Send {
                        name: Name::new_name_var("x"),
                        send_type: SendType::new_single(),
                        inputs: ProcList::new(vec![Proc::new_proc_int(7)]),
                        line_num: 0,
                        col_num: 0,
                    }),
                    right: Box::new(Proc::Send {
                        name: Name::new_name_var("y"),
                        send_type: SendType::new_single(),
                        inputs: ProcList::new(vec![Proc::new_proc_int(8)]),
                        line_num: 0,
                        col_num: 0,
                    }),
                    line_num: 0,
                    col_num: 0,
                }),
                right: Box::new(Proc::Send {
                    name: Name::new_name_var("z"),
                    send_type: SendType::new_single(),
                    inputs: ProcList::new(vec![Proc::new_proc_int(9)]),
                    line_num: 0,
                    col_num: 0,
                }),
                line_num: 0,
                col_num: 0,
            }),
            line_num: 0,
            col_num: 0,
        };

        let result = normalize_match_proc(
            &p_new,
            proc_visit_inputs_and_env().0,
            &proc_visit_inputs_and_env().1,
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
            proc_visit_inputs_and_env().0.free_map
        )
    }

    #[test]
    fn p_new_should_sort_uris_and_place_them_at_the_end() {
        let p_new = Proc::New {
            decls: Decls {
                decls: vec![
                    NameDecl::new("x", None),
                    NameDecl::new("y", None),
                    NameDecl::new("r", Some("rho:registry")),
                    NameDecl::new("out", Some("rho:stdout")),
                    NameDecl::new("z", None),
                ],
                line_num: 0,
                col_num: 0,
            },
            proc: Box::new(Proc::Par {
                left: Box::new(Proc::Par {
                    left: Box::new(Proc::Par {
                        left: Box::new(Proc::Par {
                            left: Box::new(Proc::Send {
                                name: Name::new_name_var("x"),
                                send_type: SendType::new_single(),
                                inputs: ProcList::new(vec![Proc::new_proc_int(7)]),
                                line_num: 0,
                                col_num: 0,
                            }),
                            right: Box::new(Proc::Send {
                                name: Name::new_name_var("y"),
                                send_type: SendType::new_single(),
                                inputs: ProcList::new(vec![Proc::new_proc_int(8)]),
                                line_num: 0,
                                col_num: 0,
                            }),
                            line_num: 0,
                            col_num: 0,
                        }),
                        right: Box::new(Proc::Send {
                            name: Name::new_name_var("r"),
                            send_type: SendType::new_single(),
                            inputs: ProcList::new(vec![Proc::new_proc_int(9)]),
                            line_num: 0,
                            col_num: 0,
                        }),
                        line_num: 0,
                        col_num: 0,
                    }),
                    right: Box::new(Proc::Send {
                        name: Name::new_name_var("out"),
                        send_type: SendType::new_single(),
                        inputs: ProcList::new(vec![Proc::new_proc_int(10)]),
                        line_num: 0,
                        col_num: 0,
                    }),
                    line_num: 0,
                    col_num: 0,
                }),
                right: Box::new(Proc::Send {
                    name: Name::new_name_var("z"),
                    send_type: SendType::new_single(),
                    inputs: ProcList::new(vec![Proc::new_proc_int(11)]),
                    line_num: 0,
                    col_num: 0,
                }),
                line_num: 0,
                col_num: 0,
            }),
            line_num: 0,
            col_num: 0,
        };

        let result = normalize_match_proc(
            &p_new,
            proc_visit_inputs_and_env().0,
            &proc_visit_inputs_and_env().1,
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

    #[test]
    fn new_ast_p_new_should_bind_new_variables() {
        // Maps to original: p_new_should_bind_new_variables
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{
            AnnName, AnnProc, Id, Name as NewName, NameDecl as NewNameDecl, Proc as NewProc,
            SendType as NewSendType, Var as NewVar,
        };
        use rholang_parser::{SourcePos, SourceSpan};

        // Create new x, y, z in { x!(7) | y!(8) | z!(9) } using new AST
        let p_new = AnnProc {
            proc: Box::leak(Box::new(NewProc::New {
                decls: vec![
                    NewNameDecl {
                        id: Id {
                            name: "x",
                            pos: SourcePos { line: 0, col: 0 },
                        },
                        uri: None,
                    },
                    NewNameDecl {
                        id: Id {
                            name: "y",
                            pos: SourcePos { line: 0, col: 0 },
                        },
                        uri: None,
                    },
                    NewNameDecl {
                        id: Id {
                            name: "z",
                            pos: SourcePos { line: 0, col: 0 },
                        },
                        uri: None,
                    },
                ],
                proc: AnnProc {
                    proc: Box::leak(Box::new(NewProc::Par {
                        left: AnnProc {
                            proc: Box::leak(Box::new(NewProc::Par {
                                left: AnnProc {
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
                                        inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                                            proc: Box::leak(Box::new(NewProc::LongLiteral(7))),
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
                                    proc: Box::leak(Box::new(NewProc::Send {
                                        channel: AnnName {
                                            name: NewName::ProcVar(NewVar::Id(Id {
                                                name: "y",
                                                pos: SourcePos { line: 0, col: 0 },
                                            })),
                                            span: SourceSpan {
                                                start: SourcePos { line: 0, col: 0 },
                                                end: SourcePos { line: 0, col: 0 },
                                            },
                                        },
                                        send_type: NewSendType::Single,
                                        inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                                            proc: Box::leak(Box::new(NewProc::LongLiteral(8))),
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
                            proc: Box::leak(Box::new(NewProc::Send {
                                channel: AnnName {
                                    name: NewName::ProcVar(NewVar::Id(Id {
                                        name: "z",
                                        pos: SourcePos { line: 0, col: 0 },
                                    })),
                                    span: SourceSpan {
                                        start: SourcePos { line: 0, col: 0 },
                                        end: SourcePos { line: 0, col: 0 },
                                    },
                                },
                                send_type: NewSendType::Single,
                                inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                                    proc: Box::leak(Box::new(NewProc::LongLiteral(9))),
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
    fn new_ast_p_new_should_sort_uris_and_place_them_at_the_end() {
        // Maps to original: p_new_should_sort_uris_and_place_them_at_the_end
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{
            AnnName, AnnProc, Id, Name as NewName, NameDecl as NewNameDecl, Proc as NewProc,
            SendType as NewSendType, Uri as NewUri, Var as NewVar,
        };
        use rholang_parser::{SourcePos, SourceSpan};

        // Create new x, y, r `rho:registry`, out `rho:stdout`, z in { x!(7) | y!(8) | r!(9) | out!(10) | z!(11) } using new AST
        let p_new = AnnProc {
            proc: Box::leak(Box::new(NewProc::New {
                decls: vec![
                    NewNameDecl {
                        id: Id {
                            name: "x",
                            pos: SourcePos { line: 0, col: 0 },
                        },
                        uri: None,
                    },
                    NewNameDecl {
                        id: Id {
                            name: "y",
                            pos: SourcePos { line: 0, col: 0 },
                        },
                        uri: None,
                    },
                    NewNameDecl {
                        id: Id {
                            name: "r",
                            pos: SourcePos { line: 0, col: 0 },
                        },
                        uri: Some(NewUri::from("rho:registry")),
                    },
                    NewNameDecl {
                        id: Id {
                            name: "out",
                            pos: SourcePos { line: 0, col: 0 },
                        },
                        uri: Some(NewUri::from("rho:stdout")),
                    },
                    NewNameDecl {
                        id: Id {
                            name: "z",
                            pos: SourcePos { line: 0, col: 0 },
                        },
                        uri: None,
                    },
                ],
                proc: AnnProc {
                    proc: Box::leak(Box::new(NewProc::Par {
                        left: AnnProc {
                            proc: Box::leak(Box::new(NewProc::Par {
                                left: AnnProc {
                                    proc: Box::leak(Box::new(NewProc::Par {
                                        left: AnnProc {
                                            proc: Box::leak(Box::new(NewProc::Par {
                                                left: AnnProc {
                                                    proc: Box::leak(Box::new(NewProc::Send {
                                                        channel: AnnName {
                                                            name: NewName::ProcVar(NewVar::Id(
                                                                Id {
                                                                    name: "x",
                                                                    pos: SourcePos {
                                                                        line: 0,
                                                                        col: 0,
                                                                    },
                                                                },
                                                            )),
                                                            span: SourceSpan {
                                                                start: SourcePos {
                                                                    line: 0,
                                                                    col: 0,
                                                                },
                                                                end: SourcePos { line: 0, col: 0 },
                                                            },
                                                        },
                                                        send_type: NewSendType::Single,
                                                        inputs: smallvec::SmallVec::from_vec(vec![
                                                            AnnProc {
                                                                proc: Box::leak(Box::new(
                                                                    NewProc::LongLiteral(7),
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
                                                    proc: Box::leak(Box::new(NewProc::Send {
                                                        channel: AnnName {
                                                            name: NewName::ProcVar(NewVar::Id(
                                                                Id {
                                                                    name: "y",
                                                                    pos: SourcePos {
                                                                        line: 0,
                                                                        col: 0,
                                                                    },
                                                                },
                                                            )),
                                                            span: SourceSpan {
                                                                start: SourcePos {
                                                                    line: 0,
                                                                    col: 0,
                                                                },
                                                                end: SourcePos { line: 0, col: 0 },
                                                            },
                                                        },
                                                        send_type: NewSendType::Single,
                                                        inputs: smallvec::SmallVec::from_vec(vec![
                                                            AnnProc {
                                                                proc: Box::leak(Box::new(
                                                                    NewProc::LongLiteral(8),
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
                                            proc: Box::leak(Box::new(NewProc::Send {
                                                channel: AnnName {
                                                    name: NewName::ProcVar(NewVar::Id(Id {
                                                        name: "r",
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
                                                        proc: Box::leak(Box::new(
                                                            NewProc::LongLiteral(9),
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
                                    proc: Box::leak(Box::new(NewProc::Send {
                                        channel: AnnName {
                                            name: NewName::ProcVar(NewVar::Id(Id {
                                                name: "out",
                                                pos: SourcePos { line: 0, col: 0 },
                                            })),
                                            span: SourceSpan {
                                                start: SourcePos { line: 0, col: 0 },
                                                end: SourcePos { line: 0, col: 0 },
                                            },
                                        },
                                        send_type: NewSendType::Single,
                                        inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                                            proc: Box::leak(Box::new(NewProc::LongLiteral(10))),
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
                            proc: Box::leak(Box::new(NewProc::Send {
                                channel: AnnName {
                                    name: NewName::ProcVar(NewVar::Id(Id {
                                        name: "z",
                                        pos: SourcePos { line: 0, col: 0 },
                                    })),
                                    span: SourceSpan {
                                        start: SourcePos { line: 0, col: 0 },
                                        end: SourcePos { line: 0, col: 0 },
                                    },
                                },
                                send_type: NewSendType::Single,
                                inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                                    proc: Box::leak(Box::new(NewProc::LongLiteral(11))),
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
