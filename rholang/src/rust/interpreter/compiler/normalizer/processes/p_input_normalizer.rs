// See rholang/src/main/scala/coop/rchain/rholang/interpreter/compiler/normalizer/processes/PInputNormalizer.scala

use crate::rust::interpreter::{
    compiler::{
        exports::{
            FreeMapSpan, NameVisitInputsSpan, NameVisitOutputsSpan, ProcVisitInputsSpan,
            ProcVisitOutputsSpan,
        },
        normalize::{normalize_ann_proc, VarSort},
        normalizer::{
            name_normalize_matcher::normalize_name_new_ast,
            processes::utils::fail_on_invalid_connective_span,
            remainder_normalizer_matcher::normalize_match_name_new_ast,
        },
        receive_binds_sort_matcher::pre_sort_binds_span,
        span_utils::SpanContext,
    },
    errors::InterpreterError,
    matcher::has_locally_free::HasLocallyFree,
    unwrap_option_safe,
    util::filter_and_adjust_bitset,
};
use models::{
    rhoapi::{Par, Receive, ReceiveBind},
    rust::utils::union,
};
use shared::rust::BitSet;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use rholang_parser::SourceSpan;
use rholang_parser::{
    ast::{AnnName, AnnProc, Bind, Name, Proc, Source},
    SourcePos,
};

pub fn normalize_p_input_new_ast<'ast>(
    receipts: &'ast smallvec::SmallVec<[smallvec::SmallVec<[Bind<'ast>; 1]>; 1]>,
    body: &'ast AnnProc<'ast>,
    input: ProcVisitInputsSpan,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputsSpan, InterpreterError> {
    fn create_ann_proc_with_span<'ast>(proc: &'ast Proc<'ast>, span: SourceSpan) -> AnnProc<'ast> {
        AnnProc { proc, span }
    }

    fn create_ann_name_with_span<'ast>(name: Name<'ast>, span: SourceSpan) -> AnnName<'ast> {
        AnnName { name, span }
    }

    if receipts.is_empty() || receipts[0].is_empty() {
        return Err(InterpreterError::BugFoundError(
            "Expected at least one receipt".to_string(),
        ));
    }

    let head_receipt = &receipts[0][0];

    let receipt_contains_complex_source = match head_receipt {
        Bind::Linear { rhs, .. } => match rhs {
            Source::Simple { .. } => false,
            _ => true,
        },
        _ => false,
    };

    if receipt_contains_complex_source {
        let mut list_linear_bind: Vec<Bind<'ast>> = Vec::new();
        let mut list_name_decl: Vec<rholang_parser::ast::NameDecl<'ast>> = Vec::new();

        let (sends_proc, continuation_proc): (AnnProc<'ast>, AnnProc<'ast>) = receipts
            .iter()
            .flat_map(|receipt_group| receipt_group.iter())
            .try_fold(
                (
                    // Initial sends (Nil) - inherit span from for-comprehension
                    // TODO: Update zero span
                    create_ann_proc_with_span(
                        parser.ast_builder().const_nil(),
                        SpanContext::zero_span(), // Inherit from for-comprehension
                    ),
                    // Initial continuation (original body)
                    *body,
                ),
                |(sends, continuation), bind| {
                    match bind {
                        Bind::Linear { lhs, rhs } => {
                            let identifier = Uuid::new_v4().to_string();
                            // Create temporary variable - point to binding site
                            // TODO: Update zero span
                            let binding_span = SpanContext::zero_span();
                            let temp_var = Name::ProcVar(rholang_parser::ast::Var::Id(
                                rholang_parser::ast::Id {
                                    name: parser.ast_builder().alloc_str(&identifier),
                                    pos: binding_span.start, // Point to binding declaration
                                },
                            ));

                            match rhs {
                                Source::Simple { .. } => {
                                    // Simple source - just add to list
                                    list_linear_bind.push(bind.clone());
                                    Ok((sends, continuation))
                                }

                                Source::ReceiveSend { name, .. } => {
                                    // ReceiveSend desugaring: x <- name?() becomes x, temp <- name & temp!()
                                    let mut new_names = lhs.names.clone();
                                    new_names.push(create_ann_name_with_span(
                                        temp_var.clone(),
                                        binding_span, // Use derived binding span
                                    ));

                                    list_linear_bind.push(Bind::Linear {
                                        lhs: rholang_parser::ast::Names {
                                            names: new_names,
                                            remainder: lhs.remainder.clone(),
                                        },
                                        rhs: Source::Simple { name: *name },
                                    });

                                    // Add send: temp!()
                                    // TODO: Update zero span
                                    let temp_send = create_ann_proc_with_span(
                                        parser.ast_builder().alloc_send(
                                            rholang_parser::ast::SendType::Single,
                                            create_ann_name_with_span(
                                                temp_var,
                                                binding_span, // Use derived binding span
                                            ),
                                            &[],
                                        ),
                                        SpanContext::zero_span(), // Inherit from for-comprehension
                                    );

                                    let new_continuation = AnnProc {
                                        proc: parser
                                            .ast_builder()
                                            .alloc_par(temp_send, continuation),
                                        span: continuation.span,
                                    };

                                    Ok((sends, new_continuation))
                                }

                                Source::SendReceive { name, inputs, .. } => {
                                    // SendReceive desugaring: x <- name!(args) becomes new temp in { name!(temp, args) | x <- temp }
                                    list_name_decl.push(rholang_parser::ast::NameDecl {
                                        id: rholang_parser::ast::Id {
                                            name: parser.ast_builder().alloc_str(&identifier),
                                            pos: SourcePos { line: 0, col: 0 },
                                        },
                                        uri: None,
                                    });

                                    list_linear_bind.push(Bind::Linear {
                                        lhs: lhs.clone(),
                                        rhs: Source::Simple {
                                            name: create_ann_name_with_span(
                                                temp_var.clone(),
                                                binding_span, // Use derived binding span
                                            ),
                                        },
                                    });

                                    // Prepend temp variable to inputs
                                    let mut new_inputs = Vec::new();
                                    // TODO: Update zero span
                                    new_inputs.push(create_ann_proc_with_span(
                                        parser.ast_builder().alloc_eval(create_ann_name_with_span(
                                            temp_var,
                                            binding_span, // Use derived binding span
                                        )),
                                        SpanContext::zero_span(), // Inherit from for-comprehension
                                    ));
                                    new_inputs.extend(inputs.iter().cloned());

                                    // Create new send
                                    let new_send = AnnProc {
                                        proc: parser.ast_builder().alloc_send(
                                            rholang_parser::ast::SendType::Single,
                                            *name,
                                            &new_inputs,
                                        ),
                                        span: SourceSpan {
                                            start: SourcePos { line: 0, col: 0 },
                                            end: SourcePos { line: 0, col: 0 },
                                        },
                                    };

                                    let new_sends = AnnProc {
                                        proc: parser.ast_builder().alloc_par(new_send, sends),
                                        span: sends.span,
                                    };

                                    Ok((new_sends, continuation))
                                }
                            }
                        }
                        _ => Err(InterpreterError::BugFoundError(format!(
                            "Expected Linear bind in complex source desugaring, found {:?}",
                            bind
                        ))),
                    }
                },
            )?;

        // Create the desugared ForComprehension
        let desugared_for_comprehension = AnnProc {
            proc: parser
                .ast_builder()
                .alloc_for(vec![list_linear_bind], continuation_proc),
            span: body.span,
        };

        // Create final process (New + Par if needed)
        let final_proc = if list_name_decl.is_empty() {
            desugared_for_comprehension
        } else {
            let par_proc = AnnProc {
                proc: parser
                    .ast_builder()
                    .alloc_par(sends_proc, desugared_for_comprehension),
                span: body.span,
            };

            AnnProc {
                proc: parser.ast_builder().alloc_new(par_proc, list_name_decl),
                span: body.span,
            }
        };

        // Recursively normalize the desugared process
        normalize_ann_proc(&final_proc, input, env, parser)
    } else {
        // Simple source handling - similar to original's else branch

        // Convert receipts to the format expected by processing functions
        // Note: We flatten the nested SmallVec structure since input normalizer expects a flat list
        let flat_receipts: Vec<&Bind<'ast>> = receipts
            .iter()
            .flat_map(|receipt_group| receipt_group.iter())
            .collect();

        let processed_receipts: Result<Vec<_>, InterpreterError> = flat_receipts
            .iter()
            .map(|receipt| match receipt {
                Bind::Linear { lhs, rhs } => {
                    let names: Vec<_> = lhs.names.iter().collect();
                    let remainder = &lhs.remainder;

                    let source_name = match rhs {
                        Source::Simple { name } => &name.name,
                        _ => {
                            return Err(InterpreterError::ParserError(
                                "Only simple sources supported in current implementation"
                                    .to_string(),
                            ))
                        }
                    };

                    Ok(((names, remainder), source_name))
                }
                Bind::Repeated { lhs, rhs } => {
                    let names: Vec<_> = lhs.names.iter().collect();
                    let remainder = &lhs.remainder;
                    Ok(((names, remainder), &rhs.name))
                }
                Bind::Peek { lhs, rhs } => {
                    let names: Vec<_> = lhs.names.iter().collect();
                    let remainder = &lhs.remainder;
                    Ok(((names, remainder), &rhs.name))
                }
            })
            .collect();

        let processed = processed_receipts?;

        // Determine bind characteristics from first receipt
        let (persistent, peek) = match head_receipt {
            Bind::Linear { .. } => (false, false),
            Bind::Repeated { .. } => (true, false),
            Bind::Peek { .. } => (false, true),
        };

        // Extract patterns and sources
        let (patterns, sources): (Vec<_>, Vec<_>) = processed.into_iter().unzip();

        // Process sources using new AST name normalizer
        fn process_sources_new_ast<'ast>(
            sources: Vec<&'ast rholang_parser::ast::Name<'ast>>,
            input: ProcVisitInputsSpan,
            env: &HashMap<String, Par>,
            parser: &'ast rholang_parser::RholangParser<'ast>,
        ) -> Result<(Vec<Par>, FreeMapSpan<VarSort>, BitSet, bool), InterpreterError> {
            let mut vector_par = Vec::new();
            let mut current_known_free = input.free_map;
            let mut locally_free = Vec::new();
            let mut connective_used = false;

            for name in sources {
                let NameVisitOutputsSpan {
                    par,
                    free_map: updated_known_free,
                } = normalize_name_new_ast(
                    name,
                    NameVisitInputsSpan {
                        bound_map_chain: input.bound_map_chain.clone(),
                        free_map: current_known_free,
                    },
                    env,
                    parser,
                )?;

                vector_par.push(par.clone());
                current_known_free = updated_known_free;
                locally_free = union(
                    locally_free,
                    par.locally_free(par.clone(), input.bound_map_chain.depth() as i32),
                );
                connective_used = connective_used || par.clone().connective_used(par);
            }

            Ok((
                vector_par,
                current_known_free,
                locally_free,
                connective_used,
            ))
        }

        fn process_patterns_new_ast<'ast>(
            patterns: Vec<(
                Vec<&'ast AnnName<'ast>>,
                &Option<rholang_parser::ast::Var<'ast>>,
            )>,
            input: ProcVisitInputsSpan,
            env: &HashMap<String, Par>,
            parser: &'ast rholang_parser::RholangParser<'ast>,
        ) -> Result<
            Vec<(
                Vec<Par>,
                Option<models::rhoapi::Var>,
                FreeMapSpan<VarSort>,
                BitSet,
            )>,
            InterpreterError,
        > {
            patterns
                .into_iter()
                .map(|(names, name_remainder)| {
                    let mut vector_par = Vec::new();
                    let mut current_known_free = FreeMapSpan::new();
                    let mut locally_free = Vec::new();

                    for ann_name in names {
                        let NameVisitOutputsSpan {
                            par,
                            free_map: updated_known_free,
                        } = normalize_name_new_ast(
                            &ann_name.name,
                            NameVisitInputsSpan {
                                bound_map_chain: input.bound_map_chain.push(),
                                free_map: current_known_free,
                            },
                            env,
                            parser,
                        )?;

                        fail_on_invalid_connective_span(
                            &input,
                            &NameVisitOutputsSpan {
                                par: par.clone(),
                                free_map: updated_known_free.clone(),
                            },
                        )?;

                        vector_par.push(par.clone());
                        current_known_free = updated_known_free;
                        locally_free = union(
                            locally_free,
                            par.locally_free(par.clone(), input.bound_map_chain.depth() as i32 + 1),
                        );
                    }

                    let (optional_var, known_free) =
                        normalize_match_name_new_ast(name_remainder, current_known_free)?;

                    Ok((vector_par, optional_var, known_free, locally_free))
                })
                .collect()
        }

        let processed_patterns = process_patterns_new_ast(patterns, input.clone(), env, parser)?;
        let processed_sources = process_sources_new_ast(sources, input.clone(), env, parser)?;
        let (sources_par, sources_free, sources_locally_free, sources_connective_used) =
            processed_sources;

        // Pre-sort binds using span-aware version
        let receive_binds_and_free_maps = pre_sort_binds_span(
            processed_patterns
                .clone()
                .into_iter()
                .zip(sources_par)
                .into_iter()
                .map(|((a, b, c, _), e)| (a, b, e, c))
                .collect(),
        )?;

        let (receive_binds, receive_bind_free_maps): (Vec<ReceiveBind>, Vec<FreeMapSpan<VarSort>>) =
            receive_binds_and_free_maps.into_iter().unzip();

        // Channel duplicate check
        let channels: Vec<Par> = receive_binds
            .clone()
            .into_iter()
            .map(|rb| rb.source.unwrap())
            .collect();

        let channels_set: HashSet<Par> = channels.clone().into_iter().collect();
        let has_same_channels = channels.len() > channels_set.len();

        if has_same_channels {
            // TODO: Review
            return Err(InterpreterError::ReceiveOnSameChannelsErrorSpan {
                source_span: body.span,
            });
        }

        // Merge receive bind free maps
        let receive_binds_free_map = receive_bind_free_maps.into_iter().try_fold(
            FreeMapSpan::new(),
            |known_free, receive_bind_free_map| {
                let (updated_known_free, conflicts) = known_free.merge(receive_bind_free_map);

                if conflicts.is_empty() {
                    Ok(updated_known_free)
                } else {
                    let (shadowing_var, source_span) = &conflicts[0];
                    let original_span =
                        unwrap_option_safe(known_free.get(shadowing_var))?.source_span;
                    Err(InterpreterError::UnexpectedReuseOfNameContextFreeSpan {
                        var_name: shadowing_var.to_string(),
                        first_use: original_span,
                        second_use: *source_span,
                    })
                }
            },
        )?;

        // Process body
        let proc_visit_outputs = normalize_ann_proc(
            body,
            ProcVisitInputsSpan {
                par: Par::default(),
                bound_map_chain: input
                    .bound_map_chain
                    .absorb_free_span(&receive_binds_free_map),
                free_map: sources_free,
            },
            env,
            parser,
        )?;

        let bind_count = receive_binds_free_map.count_no_wildcards();

        Ok(ProcVisitOutputsSpan {
            par: input.par.clone().prepend_receive(Receive {
                binds: receive_binds,
                body: Some(proc_visit_outputs.clone().par),
                persistent,
                peek,
                bind_count: bind_count as i32,
                locally_free: {
                    union(
                        sources_locally_free,
                        union(
                            processed_patterns
                                .into_iter()
                                .map(|pattern| pattern.3)
                                .fold(Vec::new(), |locally_free1, locally_free2| {
                                    union(locally_free1, locally_free2)
                                }),
                            filter_and_adjust_bitset(
                                proc_visit_outputs.par.locally_free,
                                bind_count,
                            ),
                        ),
                    )
                },
                connective_used: sources_connective_used || proc_visit_outputs.par.connective_used,
            }),
            free_map: proc_visit_outputs.free_map,
        })
    }
}

// See rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use models::{
        create_bit_vector,
        rhoapi::Receive,
        rust::utils::{
            new_boundvar_par, new_elist_par, new_freevar_par, new_freevar_var, new_gint_par,
            new_send, new_send_par,
        },
    };

    use crate::rust::interpreter::compiler::{compiler::Compiler, exports::BoundMapChainSpan};

    use super::*;

    fn inputs_span() -> ProcVisitInputsSpan {
        ProcVisitInputsSpan {
            par: Par::default(),
            bound_map_chain: BoundMapChainSpan::new(),
            free_map: FreeMapSpan::new(),
        }
    }

    #[test]
    fn new_ast_p_input_should_handle_a_simple_receive() {
        // for ( x, y <- @Nil ) { x!(*y) }
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{
            AnnName, AnnProc, Bind, Id, Name, Names, Proc, SendType, Source, Var,
        };
        use rholang_parser::{SourcePos, SourceSpan};

        let (mut inputs_data, env) = (inputs_span(), HashMap::new());

        let bind = Bind::Linear {
            lhs: Names {
                names: smallvec::SmallVec::from_vec(vec![
                    AnnName {
                        name: Name::ProcVar(Var::Id(Id {
                            name: "x",
                            pos: SourcePos { line: 0, col: 0 },
                        })),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    AnnName {
                        name: Name::ProcVar(Var::Id(Id {
                            name: "y",
                            pos: SourcePos { line: 0, col: 0 },
                        })),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                ]),
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
        };

        // Create body: x!(*y)
        let body = AnnProc {
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
                    proc: Box::leak(Box::new(Proc::Eval {
                        name: AnnName {
                            name: Name::ProcVar(Var::Id(Id {
                                name: "y",
                                pos: SourcePos { line: 0, col: 0 },
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
                }]),
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        // Create ForComprehension
        let for_comprehension = AnnProc {
            proc: Box::leak(Box::new(Proc::ForComprehension {
                receipts: smallvec::SmallVec::from_vec(vec![smallvec::SmallVec::from_vec(vec![
                    bind,
                ])]),
                proc: body,
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&for_comprehension, inputs_data.clone(), &env, &parser);
        assert!(result.is_ok());

        let bind_count = 2;
        let expected_result = inputs_data.par.prepend_receive(Receive {
            binds: vec![ReceiveBind {
                patterns: vec![
                    new_freevar_par(0, Vec::new()),
                    new_freevar_par(1, Vec::new()),
                ],
                source: Some(Par::default()),
                remainder: None,
                free_count: 2,
            }],
            body: Some(new_send_par(
                new_boundvar_par(1, create_bit_vector(&vec![1]), false),
                vec![new_boundvar_par(0, create_bit_vector(&vec![0]), false)],
                false,
                create_bit_vector(&vec![0, 1]),
                false,
                create_bit_vector(&vec![0, 1]),
                false,
            )),
            persistent: false,
            peek: false,
            bind_count,
            locally_free: Vec::new(),
            connective_used: false,
        });

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.unwrap().free_map, inputs_data.free_map);
    }

    #[test]
    fn new_ast_p_input_should_handle_peek() {
        // for ( x, y <<- @Nil ) { x!(*y) }
        use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;

        let result = ParBuilderUtil::mk_term_new_ast(r#"for ( x, y <<- @Nil ) { x!(*y) }"#);

        assert!(
            result.is_ok(),
            "Failed to parse and normalize the Rholang code"
        );
        let normalized = result.unwrap();

        assert!(
            !normalized.receives.is_empty(),
            "Should have at least one receive"
        );
        assert_eq!(
            normalized.receives[0].peek, true,
            "Peek should be true for <<- operator"
        );
    }

    #[test]
    fn new_ast_p_input_should_bind_whole_list_to_the_list_remainder() {
        // for (@[...a] <- @0) { Nil }
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{AnnName, AnnProc, Bind, Id, Name, Names, Proc, Source, Var};
        use rholang_parser::{SourcePos, SourceSpan};

        let (mut inputs_data, env) = (inputs_span(), HashMap::new());

        // Create bind for the pattern: @[...a] <- @0 (list remainder)
        let bind = Bind::Linear {
            lhs: Names {
                names: smallvec::SmallVec::from_vec(vec![AnnName {
                    name: Name::Quote(Box::leak(Box::new(Proc::Collection(
                        rholang_parser::ast::Collection::List {
                            elements: Vec::new(),
                            remainder: Some(Var::Id(Id {
                                name: "a",
                                pos: SourcePos { line: 0, col: 0 },
                            })),
                        },
                    )))),
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
        };

        // Create body: Nil
        let body = AnnProc {
            proc: Box::leak(Box::new(Proc::Nil)),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        // Create ForComprehension
        let for_comprehension = AnnProc {
            proc: Box::leak(Box::new(Proc::ForComprehension {
                receipts: smallvec::SmallVec::from_vec(vec![smallvec::SmallVec::from_vec(vec![
                    bind,
                ])]),
                proc: body,
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&for_comprehension, inputs_data.clone(), &env, &parser);
        assert!(result.is_ok());

        let bind_count = 1;
        let expected_result = inputs_data.par.prepend_receive(Receive {
            binds: vec![ReceiveBind {
                patterns: vec![new_elist_par(
                    Vec::new(),
                    Vec::new(),
                    true,
                    Some(new_freevar_var(0)),
                    Vec::new(),
                    true,
                )],
                source: Some(Par::default()),
                remainder: None,
                free_count: 1,
            }],
            body: Some(Par::default()),
            persistent: false,
            peek: false,
            bind_count,
            locally_free: Vec::new(),
            connective_used: false,
        });

        assert_eq!(result.unwrap().par, expected_result);
    }

    #[test]
    fn new_ast_p_input_should_handle_a_more_complicated_receive() {
        // for ( (x1, @y1) <- @Nil  & (x2, @y2) <- @1) { x1!(y2) | x2!(y1) }
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{
            AnnName, AnnProc, Bind, Id, Name, Names, Proc, SendType, Source, Var,
        };
        use rholang_parser::{SourcePos, SourceSpan};

        let (mut inputs_data, env) = (inputs_span(), HashMap::new());

        // Create first bind: x1, @y1 <- @Nil
        let bind1 = Bind::Linear {
            lhs: Names {
                names: smallvec::SmallVec::from_vec(vec![
                    AnnName {
                        name: Name::ProcVar(Var::Id(Id {
                            name: "x1",
                            pos: SourcePos { line: 0, col: 0 },
                        })),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    AnnName {
                        name: Name::Quote(Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                            name: "y1",
                            pos: SourcePos { line: 0, col: 0 },
                        }))))),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                ]),
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
        };

        // Create second bind: x2, @y2 <- @1
        let bind2 = Bind::Linear {
            lhs: Names {
                names: smallvec::SmallVec::from_vec(vec![
                    AnnName {
                        name: Name::ProcVar(Var::Id(Id {
                            name: "x2",
                            pos: SourcePos { line: 0, col: 0 },
                        })),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    AnnName {
                        name: Name::Quote(Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                            name: "y2",
                            pos: SourcePos { line: 0, col: 0 },
                        }))))),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                ]),
                remainder: None,
            },
            rhs: Source::Simple {
                name: AnnName {
                    name: Name::Quote(Box::leak(Box::new(Proc::LongLiteral(1)))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
            },
        };

        // Create body: x1!(y2) | x2!(y1)
        let send1 = AnnProc {
            proc: Box::leak(Box::new(Proc::Send {
                channel: AnnName {
                    name: Name::ProcVar(Var::Id(Id {
                        name: "x1",
                        pos: SourcePos { line: 0, col: 0 },
                    })),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                send_type: SendType::Single,
                inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                    proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                        name: "y2",
                        pos: SourcePos { line: 0, col: 0 },
                    })))),
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
        };

        let send2 = AnnProc {
            proc: Box::leak(Box::new(Proc::Send {
                channel: AnnName {
                    name: Name::ProcVar(Var::Id(Id {
                        name: "x2",
                        pos: SourcePos { line: 0, col: 0 },
                    })),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                send_type: SendType::Single,
                inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                    proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                        name: "y1",
                        pos: SourcePos { line: 0, col: 0 },
                    })))),
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
        };

        let body = AnnProc {
            proc: Box::leak(Box::new(Proc::Par {
                left: send1,
                right: send2,
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        // Create ForComprehension - Two separate receipt groups for & operation
        let for_comprehension = AnnProc {
            proc: Box::leak(Box::new(Proc::ForComprehension {
                receipts: smallvec::SmallVec::from_vec(vec![
                    smallvec::SmallVec::from_vec(vec![bind1]),
                    smallvec::SmallVec::from_vec(vec![bind2]),
                ]),
                proc: body,
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&for_comprehension, inputs_data.clone(), &env, &parser);
        assert!(result.is_ok());

        let bind_count = 4;
        let expected_result = inputs_data.par.prepend_receive(Receive {
            binds: vec![
                ReceiveBind {
                    patterns: vec![
                        new_freevar_par(0, Vec::new()),
                        new_freevar_par(1, Vec::new()),
                    ],
                    source: Some(Par::default()),
                    remainder: None,
                    free_count: 2,
                },
                ReceiveBind {
                    patterns: vec![
                        new_freevar_par(0, Vec::new()),
                        new_freevar_par(1, Vec::new()),
                    ],
                    source: Some(new_gint_par(1, Vec::new(), false)),
                    remainder: None,
                    free_count: 2,
                },
            ],
            body: Some({
                let mut par = Par::default().with_sends(vec![
                    new_send(
                        new_boundvar_par(1, create_bit_vector(&vec![1]), false),
                        vec![new_boundvar_par(2, create_bit_vector(&vec![2]), false)],
                        false,
                        create_bit_vector(&vec![1, 2]),
                        false,
                    ),
                    new_send(
                        new_boundvar_par(3, create_bit_vector(&vec![3]), false),
                        vec![new_boundvar_par(0, create_bit_vector(&vec![0]), false)],
                        false,
                        create_bit_vector(&vec![0, 3]),
                        false,
                    ),
                ]);
                par.locally_free = create_bit_vector(&vec![0, 1, 2, 3]);
                par
            }),
            persistent: false,
            peek: false,
            bind_count,
            locally_free: Vec::new(),
            connective_used: false,
        });

        assert_eq!(result.unwrap().par, expected_result);
    }

    #[test]
    fn new_ast_p_input_should_fail_if_a_free_variable_is_used_in_two_different_receives() {
        // for ( (x1, @y1) <- @Nil  & (x2, @y1) <- @1) { Nil }
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{AnnName, AnnProc, Bind, Id, Name, Names, Proc, Source, Var};
        use rholang_parser::{SourcePos, SourceSpan};

        // Create first bind: x1, @y1 <- @Nil
        let bind1 = Bind::Linear {
            lhs: Names {
                names: smallvec::SmallVec::from_vec(vec![
                    AnnName {
                        name: Name::ProcVar(Var::Id(Id {
                            name: "x1",
                            pos: SourcePos { line: 0, col: 0 },
                        })),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    AnnName {
                        name: Name::Quote(Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                            name: "y1",
                            pos: SourcePos { line: 0, col: 0 },
                        }))))),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                ]),
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
        };

        // Create second bind: x2, @y1 <- @1 (reusing y1!)
        let bind2 = Bind::Linear {
            lhs: Names {
                names: smallvec::SmallVec::from_vec(vec![
                    AnnName {
                        name: Name::ProcVar(Var::Id(Id {
                            name: "x2",
                            pos: SourcePos { line: 0, col: 0 },
                        })),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    AnnName {
                        name: Name::Quote(Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                            name: "y1", // Reusing same variable!
                            pos: SourcePos { line: 0, col: 0 },
                        }))))),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                ]),
                remainder: None,
            },
            rhs: Source::Simple {
                name: AnnName {
                    name: Name::Quote(Box::leak(Box::new(Proc::LongLiteral(1)))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
            },
        };

        // Create body: Nil
        let body = AnnProc {
            proc: Box::leak(Box::new(Proc::Nil)),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        // Create ForComprehension
        let for_comprehension = AnnProc {
            proc: Box::leak(Box::new(Proc::ForComprehension {
                receipts: smallvec::SmallVec::from_vec(vec![
                    smallvec::SmallVec::from_vec(vec![bind1]),
                    smallvec::SmallVec::from_vec(vec![bind2]),
                ]),
                proc: body,
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result =
            normalize_ann_proc(&for_comprehension, inputs_span(), &HashMap::new(), &parser);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InterpreterError::UnexpectedReuseOfNameContextFreeSpan {
                var_name,
                first_use: _,
                second_use: _
            }) if var_name == "y1"
        ));
    }

    #[test]
    fn new_ast_p_input_should_not_compile_when_connectives_are_used_in_the_channel() {
        // Test disjunction in channel
        let result1 = Compiler::source_to_adt(r#"for(x <- @{Nil \/ Nil}){ Nil }"#);
        assert!(result1.is_err());
        match result1 {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("\\/ (disjunction)"));
            }
            other => panic!(
                "Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}",
                other
            ),
        }

        // Test conjunction in channel
        let result2 = Compiler::source_to_adt(r#"for(x <- @{Nil /\ Nil}){ Nil }"#);
        assert!(result2.is_err());
        match result2 {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("/\\ (conjunction)"));
            }
            other => panic!(
                "Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}",
                other
            ),
        }

        // Test negation in channel
        let result3 = Compiler::source_to_adt(r#"for(x <- @{~Nil}){ Nil }"#);
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

    #[test]
    fn new_ast_p_input_should_not_compile_when_connectives_are_at_the_top_level_expression_in_the_body(
    ) {
        // Test conjunction in body
        let result1 = Compiler::source_to_adt(r#"for(x <- @Nil){ 1 /\ 2 }"#);
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

        // Test disjunction in body
        let result2 = Compiler::source_to_adt(r#"for(x <- @Nil){ 1 \/ 2 }"#);
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

        // Test negation in body
        let result3 = Compiler::source_to_adt(r#"for(x <- @Nil){ ~1 }"#);
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

    #[test]
    fn new_ast_p_input_should_not_compile_when_logical_or_or_not_is_used_in_pattern_of_receive() {
        // Test disjunction in pattern
        let result1 =
            Compiler::source_to_adt(r#"new x in { for(@{Nil \/ Nil} <- x) { Nil } }"#);
        assert!(result1.is_err());
        match result1 {
            Err(InterpreterError::PatternReceiveError(msg)) => {
                assert!(msg.contains("\\/ (disjunction)"));
            }
            other => panic!("Expected PatternReceiveError, got: {:?}", other),
        }

        // Test negation in pattern
        let result2 = Compiler::source_to_adt(r#"new x in { for(@{~Nil} <- x) { Nil } }"#);
        assert!(result2.is_err());
        match result2 {
            Err(InterpreterError::PatternReceiveError(msg)) => {
                assert!(msg.contains("~ (negation)"));
            }
            other => panic!("Expected PatternReceiveError, got: {:?}", other),
        }
    }

    #[test]
    fn new_ast_p_input_should_compile_when_logical_and_is_used_in_pattern_of_receive() {
        // Test that conjunction in pattern is allowed
        let result1 =
            Compiler::source_to_adt(r#"new x in { for(@{Nil /\ Nil} <- x) { Nil } }"#);
        assert!(
            result1.is_ok(),
            "Conjunction in pattern should be allowed, but got error: {:?}",
            result1
        );
    }
}
