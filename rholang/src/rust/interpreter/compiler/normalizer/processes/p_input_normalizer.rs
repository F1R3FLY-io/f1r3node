// See rholang/src/main/scala/coop/rchain/rholang/interpreter/compiler/normalizer/processes/PInputNormalizer.scala

use std::collections::{HashMap, HashSet};

use models::{
    rhoapi::{Par, Receive, ReceiveBind},
    rust::utils::union,
    BitSet,
};
use uuid::Uuid;

use crate::rust::interpreter::{
    compiler::{
        exports::FreeMap,
        normalize::{normalize_match_proc, NameVisitInputs, NameVisitOutputs, VarSort},
        span_utils::{SpanContext, SpanOffset},
        normalizer::{
            name_normalize_matcher::normalize_name, processes::utils::fail_on_invalid_connective,
            remainder_normalizer_matcher::normalize_match_name,
        },
        receive_binds_sort_matcher::pre_sort_binds,
        rholang_ast::{
            Block, Decls, Eval, LinearBind, Name, NameDecl, Names, ProcList, Receipt, Receipts,
            SendType, Source, Var,
        },
    },
    matcher::has_locally_free::HasLocallyFree,
    unwrap_option_safe,
    util::filter_and_adjust_bitset,
};

use super::exports::{InterpreterError, Proc, ProcVisitInputs, ProcVisitOutputs};

pub fn normalize_p_input(
    formals: &Receipts,
    body: &Block,
    line_num: usize,
    col_num: usize,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    // Do I return an error for this case?
    if formals.receipts.is_empty() {
        return Err(InterpreterError::BugFoundError(
            "Exepected at least one receipt".to_string(),
        ));
    } else {
        let head_receipt = &formals.receipts[0];

        let receipt_contains_complex_source = match head_receipt {
            Receipt::LinearBinds(linear_bind) => match linear_bind.input {
                Source::Simple { .. } => false,
                _ => true,
            },
            _ => false,
        };

        // println!(
        //     "\nreceipt_contains_complex_source: {:?}",
        //     receipt_contains_complex_source,
        // );

        if receipt_contains_complex_source {
            match head_receipt {
                Receipt::LinearBinds(linear_bind) => match &linear_bind.input {
                    Source::Simple { .. } => {
                        let list_receipt = Receipts {
                            receipts: Vec::new(),
                            line_num: 0,
                            col_num: 0,
                        };
                        let mut list_linear_bind: Vec<Receipt> = Vec::new();
                        let mut list_name_decl = Decls {
                            decls: Vec::new(),
                            line_num: 0,
                            col_num: 0,
                        };

                        let (sends, continuation): (Proc, Proc) =
                            formals.clone().receipts.into_iter().try_fold(
                                (
                                    Proc::Nil {
                                        line_num: 0,
                                        col_num: 0,
                                    },
                                    body.proc.clone(),
                                ),
                                |(sends, continuation), lb| match lb {
                                    Receipt::LinearBinds(linear_bind) => {
                                        let identifier = Uuid::new_v4().to_string();
                                        let r = Proc::Var(Var {
                                            name: identifier.clone(),
                                            line_num: 0,
                                            col_num: 0,
                                        });

                                        match linear_bind.input {
                                            Source::Simple { .. } => {
                                                list_linear_bind
                                                    .push(Receipt::LinearBinds(linear_bind));
                                                Ok((sends, continuation))
                                            }

                                            Source::ReceiveSend { name, .. } => {
                                                let mut list_name = linear_bind.names.names;
                                                list_name.push(Name::ProcVar(Box::new(r.clone())));

                                                list_linear_bind.push(Receipt::LinearBinds(
                                                    LinearBind {
                                                        names: Names {
                                                            names: list_name,
                                                            cont: linear_bind.names.cont,
                                                            line_num: 0,
                                                            col_num: 0,
                                                        },
                                                        input: Source::Simple {
                                                            name,
                                                            line_num: 0,
                                                            col_num: 0,
                                                        },
                                                        line_num: 0,
                                                        col_num: 0,
                                                    },
                                                ));

                                                Ok((
                                                    sends,
                                                    Proc::Par {
                                                        left: Box::new(Proc::Send {
                                                            name: Name::ProcVar(Box::new(r)),
                                                            send_type: SendType::Single {
                                                                line_num: 0,
                                                                col_num: 0,
                                                            },
                                                            inputs: ProcList {
                                                                procs: Vec::new(),
                                                                line_num: 0,
                                                                col_num: 0,
                                                            },
                                                            line_num: 0,
                                                            col_num: 0,
                                                        }),
                                                        right: Box::new(continuation),
                                                        line_num: 0,
                                                        col_num: 0,
                                                    },
                                                ))
                                            }

                                            Source::SendReceive { name, inputs, .. } => {
                                                list_name_decl.decls.push(NameDecl {
                                                    var: Var {
                                                        name: identifier,
                                                        line_num: 0,
                                                        col_num: 0,
                                                    },
                                                    uri: None,
                                                    line_num: 0,
                                                    col_num: 0,
                                                });

                                                list_linear_bind.push(Receipt::LinearBinds(
                                                    LinearBind {
                                                        names: Names {
                                                            names: linear_bind.names.names,
                                                            cont: linear_bind.names.cont,
                                                            line_num: 0,
                                                            col_num: 0,
                                                        },
                                                        input: Source::Simple {
                                                            name: Name::ProcVar(Box::new(
                                                                r.clone(),
                                                            )),
                                                            line_num: 0,
                                                            col_num: 0,
                                                        },
                                                        line_num: 0,
                                                        col_num: 0,
                                                    },
                                                ));

                                                let mut list_proc = inputs.procs;
                                                list_proc.insert(
                                                    0,
                                                    Proc::Eval(Eval {
                                                        name: Name::ProcVar(Box::new(r)),
                                                        line_num: 0,
                                                        col_num: 0,
                                                    }),
                                                );

                                                Ok((
                                                    Proc::Par {
                                                        left: Box::new(Proc::Send {
                                                            name,
                                                            send_type: SendType::Single {
                                                                line_num: 0,
                                                                col_num: 0,
                                                            },
                                                            inputs: ProcList {
                                                                procs: list_proc,
                                                                line_num: 0,
                                                                col_num: 0,
                                                            },
                                                            line_num: 0,
                                                            col_num: 0,
                                                        }),
                                                        right: Box::new(sends),
                                                        line_num: 0,
                                                        col_num: 0,
                                                    },
                                                    continuation,
                                                ))
                                            }
                                        }
                                    }

                                    _ => Err(InterpreterError::BugFoundError(format!(
                                        "Expected LinearBinds, found {:?}",
                                        &lb
                                    ))),
                                },
                            )?;

                        let p_input = Proc::Input {
                            formals: list_receipt,
                            proc: Box::new(Block {
                                proc: continuation,
                                line_num: 0,
                                col_num: 0,
                            }),
                            line_num: 0,
                            col_num: 0,
                        };

                        let p_new = Proc::New {
                            decls: list_name_decl.clone(),
                            proc: Box::new(Proc::Par {
                                left: Box::new(sends),
                                right: Box::new(p_input.clone()),
                                line_num: 0,
                                col_num: 0,
                            }),
                            line_num: 0,
                            col_num: 0,
                        };

                        normalize_match_proc(
                            {
                                if list_name_decl.decls.is_empty() {
                                    &p_input
                                } else {
                                    &p_new
                                }
                            },
                            input,
                            env,
                        )
                    }

                    _ => {
                        return Err(InterpreterError::BugFoundError(format!(
                            "Expected SimpleSource, found {:?}",
                            &linear_bind.input
                        )))
                    }
                },
                _ => {
                    return Err(InterpreterError::BugFoundError(format!(
                        "Expected LinearBinds, found {:?}",
                        &formals.receipts[0]
                    )))
                }
            }
        } else {
            // To handle the most common case where we can sort the binds because
            // they're from different sources, Each channel's list of patterns starts its free variables at 0.
            // We check for overlap at the end after sorting. We could check before, but it'd be an extra step.
            // We split this into parts. First we process all the sources, then we process all the bindings.
            fn process_sources(
                sources: Vec<Name>,
                input: ProcVisitInputs,
                env: &HashMap<String, Par>,
            ) -> Result<(Vec<Par>, FreeMap<VarSort>, BitSet, bool), InterpreterError> {
                let mut vector_par = Vec::new();
                let mut current_known_free = input.free_map;
                let mut locally_free = Vec::new();
                let mut connective_used = false;

                for name in sources {
                    let NameVisitOutputs {
                        par,
                        free_map: updated_known_free,
                    } = normalize_name(
                        &name,
                        NameVisitInputs {
                            bound_map_chain: input.bound_map_chain.clone(),
                            free_map: current_known_free,
                            source_span: input.source_span,
                        },
                        env,
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

            fn process_patterns(
                patterns: Vec<(Vec<Name>, Option<Box<Proc>>)>,
                input: ProcVisitInputs,
                env: &HashMap<String, Par>,
            ) -> Result<
                Vec<(
                    Vec<Par>,
                    Option<models::rhoapi::Var>,
                    FreeMap<VarSort>,
                    BitSet,
                )>,
                InterpreterError,
            > {
                patterns
                    .into_iter()
                    .map(|(names, name_remainder)| {
                        let mut vector_par = Vec::new();
                        let mut current_known_free = FreeMap::new();
                        let mut locally_free = Vec::new();

                        for name in names {
                            let NameVisitOutputs {
                                par,
                                free_map: updated_known_free,
                            } = {
                                let input = NameVisitInputs {
                                    bound_map_chain: input.bound_map_chain.push(),
                                    free_map: current_known_free,
                                    source_span: input.source_span,
                                };
                                // println!("\ninput: {:?}", input);
                                // println!("\nname: {:?}", name);
                                normalize_name(&name, input, env)?
                            };

                            // println!("\npar: {:?}", par);

                            fail_on_invalid_connective(
                                &input,
                                &NameVisitOutputs {
                                    par: par.clone(),
                                    free_map: updated_known_free.clone(),
                                },
                            )?;

                            vector_par.push(par.clone());
                            current_known_free = updated_known_free;
                            locally_free = union(
                                locally_free,
                                par.locally_free(
                                    par.clone(),
                                    input.bound_map_chain.depth() as i32 + 1,
                                ),
                            );
                        }

                        // println!("\ncurrent_known_free: {:?}", current_known_free);

                        let (optional_var, known_free) =
                            normalize_match_name(&name_remainder, current_known_free)?;

                        // println!("\noptional_var: {:?}", optional_var);
                        // println!("\nknown_free: {:?}", known_free);

                        Ok((vector_par, optional_var, known_free, locally_free))
                    })
                    .collect()
            }

            let (consumes, persistent, peek): (
                Vec<((Vec<Name>, Option<Box<Proc>>), Name)>,
                bool,
                bool,
            ) = {
                let consumes: Vec<((Vec<Name>, Option<Box<Proc>>), Name)> = formals
                    .receipts
                    .clone()
                    .into_iter()
                    .map(|receipt| match receipt {
                        Receipt::LinearBinds(linear_bind) => {
                            ((linear_bind.names.names, linear_bind.names.cont), {
                                match linear_bind.input {
                                    Source::Simple { name, .. } => name,
                                    Source::ReceiveSend { name, .. } => name,
                                    Source::SendReceive { name, .. } => name,
                                }
                            })
                        }

                        Receipt::RepeatedBinds(repeated_bind) => (
                            (repeated_bind.names.names, repeated_bind.names.cont),
                            repeated_bind.input,
                        ),

                        Receipt::PeekBinds(peek_bind) => (
                            (peek_bind.names.names, peek_bind.names.cont),
                            peek_bind.input,
                        ),
                    })
                    .collect();

                match head_receipt {
                    Receipt::LinearBinds(_) => (consumes, false, false),
                    Receipt::RepeatedBinds(_) => (consumes, true, false),
                    Receipt::PeekBinds(_) => (consumes, false, true),
                }
            };

            let (patterns, names): (Vec<(Vec<Name>, Option<Box<Proc>>)>, Vec<Name>) =
                consumes.into_iter().unzip();

            // println!("\npatterns: {:#?}", patterns);

            let processed_patterns = process_patterns(patterns, input.clone(), env)?;
            // println!("\nprocessed_patterns: {:#?}", processed_patterns);
            let processed_sources = process_sources(names, input.clone(), env)?;
            let (sources, sources_free, sources_locally_free, sources_connective_used) =
                processed_sources;

            // println!("\nsources: {:?}", sources);

            let receive_binds_and_free_maps = pre_sort_binds(
                processed_patterns
                    .clone()
                    .into_iter()
                    .zip(sources)
                    .into_iter()
                    .map(|((a, b, c, _), e)| (a, b, e, c))
                    .collect(),
            )?;

            let (receive_binds, receive_bind_free_maps): (Vec<ReceiveBind>, Vec<FreeMap<VarSort>>) =
                receive_binds_and_free_maps.into_iter().unzip();

            let channels: Vec<Par> = receive_binds
                .clone()
                .into_iter()
                .map(|rb| rb.source.unwrap())
                .collect();

            let channels_set: HashSet<Par> = channels.clone().into_iter().collect();
            let has_same_channels = channels.len() > channels_set.len();

            if has_same_channels {
                return Err(InterpreterError::ReceiveOnSameChannelsError {
                    line: line_num,
                    col: col_num,
                });
            }

            // println!("\nreceive_binds_free_maps: {:?}", receive_bind_free_maps);

            let receive_binds_free_map = receive_bind_free_maps.into_iter().try_fold(
                FreeMap::new(),
                |known_free, receive_bind_free_map| {
                    let (updated_known_free, conflicts) = known_free.merge(receive_bind_free_map);

                    if conflicts.is_empty() {
                        Ok(updated_known_free)
                    } else {
                        let (shadowing_var, source_position) = &conflicts[0];
                        let original_position =
                            unwrap_option_safe(known_free.get(shadowing_var))?.source_position;
                        Err(InterpreterError::UnexpectedReuseOfNameContextFree {
                            var_name: shadowing_var.to_string(),
                            first_use: original_position.to_string(),
                            second_use: source_position.to_string(),
                        })
                    }
                },
            )?;

            // println!("\nreceive_binds_free_map: {:?}", receive_binds_free_map);
            // println!(
            //     "\nfree_map: {:?}",
            //     input
            //         .bound_map_chain
            //         .absorb_free(receive_binds_free_map.clone()),
            // );
            // println!("\nsources_free: {:?}", sources_free);

            let proc_visit_outputs = normalize_match_proc(
                &body.proc,
                ProcVisitInputs {
                    par: Par::default(),
                    bound_map_chain: input
                        .bound_map_chain
                        .absorb_free(receive_binds_free_map.clone()),
                    free_map: sources_free,
                    source_span: input.source_span, // Use input span for old AST
                },
                env,
            )?;

            let bind_count = receive_binds_free_map.count_no_wildcards();

            Ok(ProcVisitOutputs {
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
                    connective_used: sources_connective_used
                        || proc_visit_outputs.par.connective_used,
                }),
                free_map: proc_visit_outputs.free_map,
            })
        }
    }
}

// ============================================================================
// NEW AST PARALLEL FUNCTIONS
// ============================================================================

/// Parallel version of normalize_p_input for new AST ForComprehension
/// Maps Input { formals: Receipts, proc: Block } to ForComprehension { receipts, proc }
pub fn normalize_p_input_new_ast<'ast>(
    receipts: &'ast smallvec::SmallVec<[smallvec::SmallVec<[rholang_parser::ast::Bind<'ast>; 1]>; 1]>,
    body: &'ast rholang_parser::ast::AnnProc<'ast>,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    use crate::rust::interpreter::compiler::{
        normalize::normalize_ann_proc,
        normalizer::{
            name_normalize_matcher::normalize_name_new_ast,
            remainder_normalizer_matcher::normalize_match_name_new_ast,
        },
    };

    // Local helper functions for span-aware AST construction
    fn create_ann_proc_with_span<'ast>(
        proc: &'ast rholang_parser::ast::Proc<'ast>,
        span: rholang_parser::SourceSpan,
    ) -> rholang_parser::ast::AnnProc<'ast> {
        rholang_parser::ast::AnnProc { proc, span }
    }

    fn create_ann_name_with_span<'ast>(
        name: rholang_parser::ast::Name<'ast>,
        span: rholang_parser::SourceSpan,
    ) -> rholang_parser::ast::AnnName<'ast> {
        rholang_parser::ast::AnnName { name, span }
    }

    // Ensure we have at least one receipt group (same validation as original)
    if receipts.is_empty() || receipts[0].is_empty() {
        return Err(InterpreterError::BugFoundError(
            "Expected at least one receipt".to_string(),
        ));
    }

    let head_receipt = &receipts[0][0];

    // Check if receipt contains complex source (same logic as original)
    let receipt_contains_complex_source = match head_receipt {
        rholang_parser::ast::Bind::Linear { rhs, .. } => match rhs {
            rholang_parser::ast::Source::Simple { .. } => false,
            _ => true,
        },
        _ => false,
    };

    if receipt_contains_complex_source {
        // Complex source handling - desugar complex sources into simple ones
        // This follows the same logic as the original normalize_p_input
        
        let mut list_linear_bind: Vec<rholang_parser::ast::Bind<'ast>> = Vec::new();
        let mut list_name_decl: Vec<rholang_parser::ast::NameDecl<'ast>> = Vec::new();

        let (sends_proc, continuation_proc): (rholang_parser::ast::AnnProc<'ast>, rholang_parser::ast::AnnProc<'ast>) =
            receipts.iter().flat_map(|receipt_group| receipt_group.iter()).try_fold(
                (
                    // Initial sends (Nil) - inherit span from for-comprehension
                    create_ann_proc_with_span(
                        parser.ast_builder().const_nil(),
                        input.source_span, // Inherit from for-comprehension
                    ),
                    // Initial continuation (original body)
                    *body,
                ),
                |(sends, continuation), bind| {
                    match bind {
                        rholang_parser::ast::Bind::Linear { lhs, rhs } => {
                            let identifier = Uuid::new_v4().to_string();
                            // TODO: Replace Box::leak with proper arena allocation for strings
                            let identifier_leaked = Box::leak(identifier.into_boxed_str());
                            
                            // Create temporary variable - point to binding site
                            let binding_span = SpanContext::derive_synthetic_span(
                                input.source_span, 
                                SpanOffset::StartPosition
                            );
                            let temp_var = rholang_parser::ast::Name::ProcVar(
                                rholang_parser::ast::Var::Id(rholang_parser::ast::Id {
                                    name: identifier_leaked,
                                    pos: binding_span.start, // Point to binding declaration
                                })
                            );

                            match rhs {
                                rholang_parser::ast::Source::Simple { .. } => {
                                    // Simple source - just add to list
                                    list_linear_bind.push(bind.clone());
                                    Ok((sends, continuation))
                                }

                                rholang_parser::ast::Source::ReceiveSend { name, .. } => {
                                    // ReceiveSend desugaring: x <- name?() becomes x, temp <- name & temp!()
                                    let mut new_names = lhs.names.clone();
                                    new_names.push(create_ann_name_with_span(
                                        temp_var.clone(),
                                        binding_span, // Use derived binding span
                                    ));

                                    list_linear_bind.push(rholang_parser::ast::Bind::Linear {
                                        lhs: rholang_parser::ast::Names {
                                            names: new_names,
                                            remainder: lhs.remainder.clone(),
                                        },
                                        rhs: rholang_parser::ast::Source::Simple {
                                            name: *name,
                                        },
                                    });

                                    // Add send: temp!()
                                    let temp_send = create_ann_proc_with_span(
                                        parser.ast_builder().alloc_send(
                                            rholang_parser::ast::SendType::Single,
                                            create_ann_name_with_span(
                                                temp_var,
                                                binding_span, // Use derived binding span
                                            ),
                                            &[],
                                        ),
                                        input.source_span, // Inherit from for-comprehension
                                    );
                                    
                                    let new_continuation = rholang_parser::ast::AnnProc {
                                        proc: parser.ast_builder().alloc_par(temp_send, continuation),
                                        span: continuation.span,
                                    };

                                    Ok((sends, new_continuation))
                                }

                                rholang_parser::ast::Source::SendReceive { name, inputs, .. } => {
                                    // SendReceive desugaring: x <- name!(args) becomes new temp in { name!(temp, args) | x <- temp }
                                    list_name_decl.push(rholang_parser::ast::NameDecl {
                                        id: rholang_parser::ast::Id {
                                            name: identifier_leaked,
                                            pos: rholang_parser::SourcePos { line: 0, col: 0 },
                                        },
                                        uri: None,
                                    });

                                    list_linear_bind.push(rholang_parser::ast::Bind::Linear {
                                        lhs: lhs.clone(),
                                        rhs: rholang_parser::ast::Source::Simple {
                                            name: create_ann_name_with_span(
                                                temp_var.clone(),
                                                binding_span, // Use derived binding span
                                            ),
                                        },
                                    });

                                    // Prepend temp variable to inputs
                                    let mut new_inputs = Vec::new();
                                    new_inputs.push(create_ann_proc_with_span(
                                        parser.ast_builder().alloc_eval(
                                            create_ann_name_with_span(
                                                temp_var,
                                                binding_span, // Use derived binding span
                                            )
                                        ),
                                        input.source_span, // Inherit from for-comprehension
                                    ));
                                    new_inputs.extend(inputs.iter().cloned());

                                    // Create new send
                                    let new_send = rholang_parser::ast::AnnProc {
                                        proc: parser.ast_builder().alloc_send(
                                            rholang_parser::ast::SendType::Single,
                                            *name,
                                            &new_inputs,
                                        ),
                                        span: rholang_parser::SourceSpan {
                                            start: rholang_parser::SourcePos { line: 0, col: 0 },
                                            end: rholang_parser::SourcePos { line: 0, col: 0 },
                                        },
                                    };
                                    
                                    let new_sends = rholang_parser::ast::AnnProc {
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
        let desugared_for_comprehension = rholang_parser::ast::AnnProc {
            proc: parser.ast_builder().alloc_for(
                vec![list_linear_bind],
                continuation_proc,
            ),
            span: body.span,
        };

        // Create final process (New + Par if needed)
        let final_proc = if list_name_decl.is_empty() {
            desugared_for_comprehension
        } else {
            let par_proc = rholang_parser::ast::AnnProc {
                proc: parser.ast_builder().alloc_par(sends_proc, desugared_for_comprehension),
                span: body.span,
            };
            
            rholang_parser::ast::AnnProc {
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
        let flat_receipts: Vec<&rholang_parser::ast::Bind<'ast>> = receipts
            .iter()
            .flat_map(|receipt_group| receipt_group.iter())
            .collect();
            
        let processed_receipts: Result<Vec<_>, InterpreterError> = flat_receipts
            .iter()
            .map(|receipt| {
                match receipt {
                    rholang_parser::ast::Bind::Linear { lhs, rhs } => {
                        let names: Vec<_> = lhs.names.iter().collect();
                        let remainder = &lhs.remainder;
                        
                        let source_name = match rhs {
                            rholang_parser::ast::Source::Simple { name } => &name.name,
                            _ => return Err(InterpreterError::ParserError(
                                "Only simple sources supported in current implementation".to_string()
                            )),
                        };
                        
                        Ok(((names, remainder), source_name))
                    }
                    rholang_parser::ast::Bind::Repeated { lhs, rhs } => {
                        let names: Vec<_> = lhs.names.iter().collect();
                        let remainder = &lhs.remainder;
                        Ok(((names, remainder), &rhs.name))
                    }
                    rholang_parser::ast::Bind::Peek { lhs, rhs } => {
                        let names: Vec<_> = lhs.names.iter().collect();
                        let remainder = &lhs.remainder;
                        Ok(((names, remainder), &rhs.name))
                    }
                }
            })
            .collect();
            
        let processed = processed_receipts?;
        
        // Determine bind characteristics from first receipt
        let (persistent, peek) = match head_receipt {
            rholang_parser::ast::Bind::Linear { .. } => (false, false),
            rholang_parser::ast::Bind::Repeated { .. } => (true, false),
            rholang_parser::ast::Bind::Peek { .. } => (false, true),
        };

        // Extract patterns and sources
        let (patterns, sources): (Vec<_>, Vec<_>) = processed.into_iter().unzip();

        // Process sources using new AST name normalizer
        fn process_sources_new_ast<'ast>(
            sources: Vec<&'ast rholang_parser::ast::Name<'ast>>,
            input: ProcVisitInputs,
            env: &HashMap<String, Par>,
            parser: &'ast rholang_parser::RholangParser<'ast>,
        ) -> Result<(Vec<Par>, FreeMap<VarSort>, BitSet, bool), InterpreterError> {
            let mut vector_par = Vec::new();
            let mut current_known_free = input.free_map;
            let mut locally_free = Vec::new();
            let mut connective_used = false;

            for name in sources {
                let NameVisitOutputs {
                    par,
                    free_map: updated_known_free,
                } = normalize_name_new_ast(
                    name,
                    NameVisitInputs {
                        bound_map_chain: input.bound_map_chain.clone(),
                        free_map: current_known_free,
                        source_span: input.source_span,
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

        // Process patterns using new AST
        fn process_patterns_new_ast<'ast>(
            patterns: Vec<(Vec<&'ast rholang_parser::ast::AnnName<'ast>>, &Option<rholang_parser::ast::Var<'ast>>)>,
            input: ProcVisitInputs,
            env: &HashMap<String, Par>,
            parser: &'ast rholang_parser::RholangParser<'ast>,
        ) -> Result<
            Vec<(
                Vec<Par>,
                Option<models::rhoapi::Var>,
                FreeMap<VarSort>,
                BitSet,
            )>,
            InterpreterError,
        > {
            patterns
                .into_iter()
                .map(|(names, name_remainder)| {
                    let mut vector_par = Vec::new();
                    let mut current_known_free = FreeMap::new();
                    let mut locally_free = Vec::new();

                    for ann_name in names {
                        let NameVisitOutputs {
                            par,
                            free_map: updated_known_free,
                        } = normalize_name_new_ast(
                            &ann_name.name,
                            NameVisitInputs {
                                bound_map_chain: input.bound_map_chain.push(),
                                free_map: current_known_free,
                                source_span: input.source_span,
                            },
                            env,
                            parser,
                        )?;

                        fail_on_invalid_connective(
                            &input,
                            &NameVisitOutputs {
                                par: par.clone(),
                                free_map: updated_known_free.clone(),
                            },
                        )?;

                        vector_par.push(par.clone());
                        current_known_free = updated_known_free;
                        locally_free = union(
                            locally_free,
                            par.locally_free(
                                par.clone(),
                                input.bound_map_chain.depth() as i32 + 1,
                            ),
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

        // Pre-sort binds (reuse existing logic)
        let receive_binds_and_free_maps = pre_sort_binds(
            processed_patterns
                .clone()
                .into_iter()
                .zip(sources_par)
                .into_iter()
                .map(|((a, b, c, _), e)| (a, b, e, c))
                .collect(),
        )?;

        let (receive_binds, receive_bind_free_maps): (Vec<ReceiveBind>, Vec<FreeMap<VarSort>>) =
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
            return Err(InterpreterError::ReceiveOnSameChannelsError {
                line: 0, // TODO: extract from span
                col: 0,
            });
        }

        // Merge receive bind free maps
        let receive_binds_free_map = receive_bind_free_maps.into_iter().try_fold(
            FreeMap::new(),
            |known_free, receive_bind_free_map| {
                let (updated_known_free, conflicts) = known_free.merge(receive_bind_free_map);

                if conflicts.is_empty() {
                    Ok(updated_known_free)
                } else {
                    let (shadowing_var, source_position) = &conflicts[0];
                    let original_position =
                        unwrap_option_safe(known_free.get(shadowing_var))?.source_position;
                    Err(InterpreterError::UnexpectedReuseOfNameContextFree {
                        var_name: shadowing_var.to_string(),
                        first_use: original_position.to_string(),
                        second_use: source_position.to_string(),
                    })
                }
            },
        )?;

        // Process body
        let proc_visit_outputs = normalize_ann_proc(
            body,
            ProcVisitInputs {
                par: Par::default(),
                bound_map_chain: input
                    .bound_map_chain
                    .absorb_free(receive_binds_free_map.clone()),
                free_map: sources_free,
                source_span: body.span, // Use body span for normalization
            },
            env,
            parser,
        )?;

        let bind_count = receive_binds_free_map.count_no_wildcards();

        Ok(ProcVisitOutputs {
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
                connective_used: sources_connective_used
                    || proc_visit_outputs.par.connective_used,
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

    use crate::rust::interpreter::compiler::{
        compiler::Compiler,
        exports::{BoundMapChain, SourcePosition},
        normalizer::parser::parse_rholang_code_to_proc,
        rholang_ast::{Collection, Quote},
    };

    use super::*;

    fn inputs() -> ProcVisitInputs {
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: BoundMapChain::new(),
            free_map: FreeMap::new(),
            source_span: SpanContext::zero_span(),
        }
    }

    #[test]
    fn p_input_should_handle_a_simple_receive() {
        // for ( x, y <- @Nil ) { x!(*y) }
        let mut list_bindings: Vec<Name> = Vec::new();
        list_bindings.push(Name::new_name_var("x"));
        list_bindings.push(Name::new_name_var("y"));

        let mut list_linear_binds: Vec<Receipt> = Vec::new();
        list_linear_binds.push(Receipt::LinearBinds(LinearBind {
            names: Names::new(list_bindings, None),
            input: Source::new_simple_source(Name::new_name_quote_nil()),
            line_num: 0,
            col_num: 0,
        }));

        let body = Proc::Send {
            name: Name::new_name_var("x"),
            send_type: SendType::Single {
                line_num: 0,
                col_num: 0,
            },
            inputs: ProcList::new(vec![Proc::Eval(Eval {
                name: Name::new_name_var("y"),
                line_num: 0,
                col_num: 0,
            })]),
            line_num: 0,
            col_num: 0,
        };

        let basic_input = Proc::Input {
            formals: Receipts {
                receipts: list_linear_binds,
                line_num: 0,
                col_num: 0,
            },
            proc: Box::new(Block {
                proc: body,
                line_num: 0,
                col_num: 0,
            }),
            line_num: 0,
            col_num: 0,
        };

        let bind_count = 2;

        let result = normalize_match_proc(&basic_input, inputs(), &HashMap::new());
        assert!(result.is_ok());

        let expected_result = inputs().par.prepend_receive(Receive {
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

        // println!("\nresult: {:#?}", result.clone().unwrap().par);
        // println!("\nexpected_result: {:#?}", expected_result);

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.unwrap().free_map, inputs().free_map);
    }

    #[test]
    fn p_input_should_handle_peek() {
        let basic_input = parse_rholang_code_to_proc(r#"for ( x, y <<- @Nil ) { x!(*y) }"#);
        assert!(basic_input.is_ok());

        let result = normalize_match_proc(&basic_input.unwrap(), inputs(), &HashMap::new());
        assert!(result.is_ok());
        assert_eq!(result.unwrap().par.receives[0].peek, true);
    }

    #[test]
    fn p_input_should_handle_a_more_complicated_receive() {
        // for ( (x1, @y1) <- @Nil  & (x2, @y2) <- @1) { x1!(y2) | x2!(y1) }
        let mut list_bindings1: Vec<Name> = Vec::new();
        list_bindings1.push(Name::new_name_var("x1"));
        list_bindings1.push(Name::new_name_quote_var("y1"));

        let mut list_bindings2: Vec<Name> = Vec::new();
        list_bindings2.push(Name::new_name_var("x2"));
        list_bindings2.push(Name::new_name_quote_var("y2"));

        let list_receipt = vec![
            Receipt::new_linear_bind_receipt(
                Names {
                    names: list_bindings1,
                    cont: None,
                    line_num: 0,
                    col_num: 0,
                },
                Source::new_simple_source(Name::new_name_quote_nil()),
            ),
            Receipt::new_linear_bind_receipt(
                Names {
                    names: list_bindings2,
                    cont: None,
                    line_num: 0,
                    col_num: 0,
                },
                Source::new_simple_source(Name::Quote(Box::new(Quote {
                    quotable: Box::new(Proc::new_proc_int(1)),
                    line_num: 0,
                    col_num: 0,
                }))),
            ),
        ];

        let list_send1 = ProcList {
            procs: vec![Proc::Var(Var {
                name: "y2".to_string(),
                line_num: 0,
                col_num: 0,
            })],
            line_num: 0,
            col_num: 0,
        };
        let list_send2 = ProcList {
            procs: vec![Proc::Var(Var {
                name: "y1".to_string(),
                line_num: 0,
                col_num: 0,
            })],
            line_num: 0,
            col_num: 0,
        };

        let body = Block {
            proc: Proc::Par {
                left: Box::new(Proc::Send {
                    name: Name::new_name_var("x1"),
                    send_type: SendType::Single {
                        line_num: 0,
                        col_num: 0,
                    },
                    inputs: list_send1,
                    line_num: 0,
                    col_num: 0,
                }),
                right: Box::new(Proc::Send {
                    name: Name::new_name_var("x2"),
                    send_type: SendType::Single {
                        line_num: 0,
                        col_num: 0,
                    },
                    inputs: list_send2,
                    line_num: 0,
                    col_num: 0,
                }),
                line_num: 0,
                col_num: 0,
            },
            line_num: 0,
            col_num: 0,
        };

        let p_input = Proc::Input {
            formals: Receipts {
                receipts: list_receipt,
                line_num: 0,
                col_num: 0,
            },
            proc: Box::new(body),
            line_num: 0,
            col_num: 0,
        };

        let bind_count = 4;

        let result = normalize_match_proc(&p_input, inputs(), &HashMap::new());
        assert!(result.is_ok());

        let expected_result = inputs().par.prepend_receive(Receive {
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

        // println!("\nresult: {:#?}", result.clone().unwrap().par);
        // println!("\nexpected_result: {:#?}", expected_result);

        assert_eq!(result.unwrap().par, expected_result)
    }

    #[test]
    fn p_input_should_bind_whole_list_to_the_list_remainder() {
        // for (@[...a] <- @0) { … }
        let list_bindings = vec![Name::Quote(Box::new(Quote {
            quotable: Box::new(Proc::Collection(Collection::List {
                elements: vec![],
                cont: Some(Box::new(Proc::new_proc_var("a"))),
                line_num: 0,
                col_num: 0,
            })),
            line_num: 0,
            col_num: 0,
        }))];

        let bind_count = 1;
        let p_input = Proc::Input {
            formals: Receipts {
                receipts: vec![Receipt::LinearBinds(LinearBind {
                    names: Names {
                        names: list_bindings,
                        cont: None,
                        line_num: 0,
                        col_num: 0,
                    },
                    input: Source::Simple {
                        name: Name::new_name_quote_nil(),
                        line_num: 0,
                        col_num: 0,
                    },
                    line_num: 0,
                    col_num: 0,
                })],
                line_num: 0,
                col_num: 0,
            },
            proc: Box::new(Block::new_block_nil()),
            line_num: 0,
            col_num: 0,
        };

        let result = normalize_match_proc(&p_input, inputs(), &HashMap::new());
        assert!(result.is_ok());
        let expected_result = inputs().par.prepend_receive(Receive {
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
    fn p_input_should_fail_if_a_free_variable_is_used_in_two_different_receives() {
        // for ( (x1, @y1) <- @Nil  & (x2, @y1) <- @1) { Nil }
        let mut list_bindings1: Vec<Name> = Vec::new();
        list_bindings1.push(Name::new_name_var("x1"));
        list_bindings1.push(Name::new_name_quote_var("y1"));

        let mut list_bindings2: Vec<Name> = Vec::new();
        list_bindings2.push(Name::new_name_var("x2"));
        list_bindings2.push(Name::new_name_quote_var("y1"));

        let list_receipt = vec![
            Receipt::new_linear_bind_receipt(
                Names {
                    names: list_bindings1,
                    cont: None,
                    line_num: 0,
                    col_num: 0,
                },
                Source::new_simple_source(Name::new_name_quote_nil()),
            ),
            Receipt::new_linear_bind_receipt(
                Names {
                    names: list_bindings2,
                    cont: None,
                    line_num: 0,
                    col_num: 0,
                },
                Source::new_simple_source(Name::Quote(Box::new(Quote {
                    quotable: Box::new(Proc::new_proc_int(1)),
                    line_num: 0,
                    col_num: 0,
                }))),
            ),
        ];

        let p_input = Proc::Input {
            formals: Receipts {
                receipts: list_receipt,
                line_num: 0,
                col_num: 0,
            },
            proc: Box::new(Block {
                proc: Proc::Nil {
                    line_num: 0,
                    col_num: 0,
                },
                line_num: 0,
                col_num: 0,
            }),
            line_num: 0,
            col_num: 0,
        };

        let result = normalize_match_proc(&p_input, inputs(), &HashMap::new());
        assert!(result.is_err());
        assert_eq!(
            result,
            Err(InterpreterError::UnexpectedReuseOfNameContextFree {
                var_name: "y1".to_string(),
                first_use: "0:0".to_string(),
                second_use: "0:0".to_string(),
            })
        )
    }

    #[test]
    fn p_input_should_not_compile_when_connectives_are_used_in_the_cahnnel() {
        let result1 = Compiler::source_to_adt(r#"for(x <- @{Nil \/ Nil}){ Nil }"#);
        assert!(result1.is_err());
        assert_eq!(
            result1,
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(
                format!(
                    "\\/ (disjunction) at {:?}",
                    SourcePosition { row: 0, column: 11 }
                )
            ))
        );

        let result2 = Compiler::source_to_adt(r#"for(x <- @{Nil /\ Nil}){ Nil }"#);
        assert!(result2.is_err());
        assert_eq!(
            result2,
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(
                format!(
                    "/\\ (conjunction) at {:?}",
                    SourcePosition { row: 0, column: 11 }
                )
            ))
        );

        let result3 = Compiler::source_to_adt(r#"for(x <- @{~Nil}){ Nil }"#);
        assert!(result3.is_err());
        assert_eq!(
            result3,
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(
                format!(
                    "~ (negation) at {:?}",
                    SourcePosition { row: 0, column: 11 }
                )
            ))
        );
    }

    #[test]
    fn p_input_should_not_compile_when_connectives_are_at_the_top_level_expression_in_the_body() {
        let result1 = Compiler::source_to_adt(r#"for(x <- @Nil){ 1 /\ 2 }"#);
        assert!(result1.is_err());
        assert_eq!(
            result1,
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(
                format!(
                    "/\\ (conjunction) at {:?}",
                    SourcePosition { row: 0, column: 16 }
                )
            ))
        );

        let result2 = Compiler::source_to_adt(r#"for(x <- @Nil){ 1 \/ 2 }"#);
        assert!(result2.is_err());
        assert_eq!(
            result2,
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(
                format!(
                    "\\/ (disjunction) at {:?}",
                    SourcePosition { row: 0, column: 16 }
                )
            ))
        );

        let result3 = Compiler::source_to_adt(r#"for(x <- @Nil){ ~1 }"#);
        assert!(result3.is_err());
        assert_eq!(
            result3,
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(
                format!(
                    "~ (negation) at {:?}",
                    SourcePosition { row: 0, column: 16 }
                )
            ))
        );
    }

    #[test]
    fn p_input_should_not_compile_when_logical_or_or_not_is_used_in_pattern_of_receive() {
        let result1 = Compiler::source_to_adt(r#"new x in { for(@{Nil \/ Nil} <- x) { Nil } }"#);
        assert!(result1.is_err());
        assert_eq!(
            result1,
            Err(InterpreterError::PatternReceiveError(format!(
                "\\/ (disjunction) at {:?}",
                SourcePosition { row: 0, column: 17 }
            )))
        );

        let result2 = Compiler::source_to_adt(r#"new x in { for(@{~Nil} <- x) { Nil } }"#);
        assert!(result2.is_err());
        assert_eq!(
            result2,
            Err(InterpreterError::PatternReceiveError(format!(
                "~ (negation) at {:?}",
                SourcePosition { row: 0, column: 17 }
            )))
        );
    }

    #[test]
    fn p_input_should_compile_when_logical_and_is_used_in_pattern_of_receive() {
        let result1 = Compiler::source_to_adt(r#"new x in { for(@{Nil /\ Nil} <- x) { Nil } }"#);
        assert!(result1.is_ok());
    }

    // ============================================================================
    // NEW AST PARALLEL TESTS
    // ============================================================================

    #[test]
    fn new_ast_p_input_should_handle_a_simple_receive() {
        // Maps to original: p_input_should_handle_a_simple_receive
        // for ( x, y <- @Nil ) { x!(*y) }
        use rholang_parser::ast::{AnnProc, AnnName, Bind, Id, Name as NewName, Names as NewNames, Proc as NewProc, Source, Var as NewVar};
        use rholang_parser::{SourcePos, SourceSpan};
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;

        let (mut inputs_data, env) = (inputs(), HashMap::new());

        // Create ForComprehension: for (x, y <- @Nil) { x!(*y) }
        let bind = Bind::Linear {
            lhs: NewNames {
                names: smallvec::SmallVec::from_vec(vec![
                    AnnName {
                        name: NewName::ProcVar(NewVar::Id(Id {
                            name: "x",
                            pos: SourcePos { line: 0, col: 0 },
                        })),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    AnnName {
                        name: NewName::ProcVar(NewVar::Id(Id {
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
                    name: NewName::Quote(Box::leak(Box::new(NewProc::Nil))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
            },
        };

        // Create body: x!(*y)
        let body = AnnProc {
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
                send_type: rholang_parser::ast::SendType::Single,
                inputs: smallvec::SmallVec::from_vec(vec![
                    AnnProc {
                        proc: Box::leak(Box::new(NewProc::Eval {
                            name: AnnName {
                                name: NewName::ProcVar(NewVar::Id(Id {
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
                    },
                ]),
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        // Create ForComprehension
        let for_comprehension = AnnProc {
            proc: Box::leak(Box::new(NewProc::ForComprehension {
                receipts: smallvec::SmallVec::from_vec(vec![
                    smallvec::SmallVec::from_vec(vec![bind])
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
        // Maps to original: p_input_should_handle_peek
        // for ( x, y <<- @Nil ) { x!(*y) }
        use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;

        // Use the high-level parser method to create the AST from source code
        let result = ParBuilderUtil::mk_term_new_ast(r#"for ( x, y <<- @Nil ) { x!(*y) }"#);
        
        assert!(result.is_ok(), "Failed to parse and normalize the Rholang code");
        let normalized = result.unwrap();
        
        // Check that peek is set to true for the <<- operator
        assert!(!normalized.receives.is_empty(), "Should have at least one receive");
        assert_eq!(normalized.receives[0].peek, true, "Peek should be true for <<- operator");
    }

    #[test]
    fn new_ast_p_input_should_bind_whole_list_to_the_list_remainder() {
        // Maps to original: p_input_should_bind_whole_list_to_the_list_remainder
        // for (@[...a] <- @0) { Nil }
        use rholang_parser::ast::{AnnProc, AnnName, Bind, Id, Name as NewName, Names as NewNames, Proc as NewProc, Source, Var as NewVar};
        use rholang_parser::{SourcePos, SourceSpan};
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;

        let (mut inputs_data, env) = (inputs(), HashMap::new());

        // Create bind for the pattern: @[...a] <- @0 (list remainder)
        let bind = Bind::Linear {
            lhs: NewNames {
                names: smallvec::SmallVec::from_vec(vec![
                    AnnName {
                        name: NewName::Quote(Box::leak(Box::new(NewProc::Collection(
                            rholang_parser::ast::Collection::List {
                                elements: Vec::new(),
                                remainder: Some(NewVar::Id(Id {
                                    name: "a", 
                                    pos: SourcePos { line: 0, col: 0 },
                                })),
                            }
                        )))),
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
                    name: NewName::Quote(Box::leak(Box::new(NewProc::Nil))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
            },
        };

        // Create body: Nil
        let body = AnnProc {
            proc: Box::leak(Box::new(NewProc::Nil)),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        // Create ForComprehension
        let for_comprehension = AnnProc {
            proc: Box::leak(Box::new(NewProc::ForComprehension {
                receipts: smallvec::SmallVec::from_vec(vec![
                    smallvec::SmallVec::from_vec(vec![bind])
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
        // Maps to original: p_input_should_handle_a_more_complicated_receive
        // for ( (x1, @y1) <- @Nil  & (x2, @y2) <- @1) { x1!(y2) | x2!(y1) }
        use rholang_parser::ast::{AnnProc, AnnName, Bind, Id, Name as NewName, Names as NewNames, Proc as NewProc, Source, Var as NewVar};
        use rholang_parser::{SourcePos, SourceSpan};
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;

        let (mut inputs_data, env) = (inputs(), HashMap::new());

        // Create first bind: x1, @y1 <- @Nil
        let bind1 = Bind::Linear {
            lhs: NewNames {
                names: smallvec::SmallVec::from_vec(vec![
                    AnnName {
                        name: NewName::ProcVar(NewVar::Id(Id {
                            name: "x1",
                            pos: SourcePos { line: 0, col: 0 },
                        })),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    AnnName {
                        name: NewName::Quote(Box::leak(Box::new(NewProc::ProcVar(NewVar::Id(Id {
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
                    name: NewName::Quote(Box::leak(Box::new(NewProc::Nil))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
            },
        };

        // Create second bind: x2, @y2 <- @1  
        let bind2 = Bind::Linear {
            lhs: NewNames {
                names: smallvec::SmallVec::from_vec(vec![
                    AnnName {
                        name: NewName::ProcVar(NewVar::Id(Id {
                            name: "x2",
                            pos: SourcePos { line: 0, col: 0 },
                        })),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    AnnName {
                        name: NewName::Quote(Box::leak(Box::new(NewProc::ProcVar(NewVar::Id(Id {
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
                    name: NewName::Quote(Box::leak(Box::new(NewProc::LongLiteral(1)))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
            },
        };

        // Create body: x1!(y2) | x2!(y1)
        let send1 = AnnProc {
            proc: Box::leak(Box::new(NewProc::Send {
                channel: AnnName {
                    name: NewName::ProcVar(NewVar::Id(Id {
                        name: "x1",
                        pos: SourcePos { line: 0, col: 0 },
                    })),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                send_type: rholang_parser::ast::SendType::Single,
                inputs: smallvec::SmallVec::from_vec(vec![
                    AnnProc {
                        proc: Box::leak(Box::new(NewProc::ProcVar(NewVar::Id(Id {
                            name: "y2",
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
        
        let send2 = AnnProc {
            proc: Box::leak(Box::new(NewProc::Send {
                channel: AnnName {
                    name: NewName::ProcVar(NewVar::Id(Id {
                        name: "x2",
                        pos: SourcePos { line: 0, col: 0 },
                    })),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                send_type: rholang_parser::ast::SendType::Single,
                inputs: smallvec::SmallVec::from_vec(vec![
                    AnnProc {
                        proc: Box::leak(Box::new(NewProc::ProcVar(NewVar::Id(Id {
                            name: "y1",
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

        let body = AnnProc {
            proc: Box::leak(Box::new(NewProc::Par {
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
            proc: Box::leak(Box::new(NewProc::ForComprehension {
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
        // Maps to original: p_input_should_fail_if_a_free_variable_is_used_in_two_different_receives
        // for ( (x1, @y1) <- @Nil  & (x2, @y1) <- @1) { Nil }
        use rholang_parser::ast::{AnnProc, AnnName, Bind, Id, Name as NewName, Names as NewNames, Proc as NewProc, Source, Var as NewVar};
        use rholang_parser::{SourcePos, SourceSpan};
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;

        // Create first bind: x1, @y1 <- @Nil
        let bind1 = Bind::Linear {
            lhs: NewNames {
                names: smallvec::SmallVec::from_vec(vec![
                    AnnName {
                        name: NewName::ProcVar(NewVar::Id(Id {
                            name: "x1",
                            pos: SourcePos { line: 0, col: 0 },
                        })),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    AnnName {
                        name: NewName::Quote(Box::leak(Box::new(NewProc::ProcVar(NewVar::Id(Id {
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
                    name: NewName::Quote(Box::leak(Box::new(NewProc::Nil))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
            },
        };

        // Create second bind: x2, @y1 <- @1 (reusing y1!)
        let bind2 = Bind::Linear {
            lhs: NewNames {
                names: smallvec::SmallVec::from_vec(vec![
                    AnnName {
                        name: NewName::ProcVar(NewVar::Id(Id {
                            name: "x2",
                            pos: SourcePos { line: 0, col: 0 },
                        })),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    AnnName {
                        name: NewName::Quote(Box::leak(Box::new(NewProc::ProcVar(NewVar::Id(Id {
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
                    name: NewName::Quote(Box::leak(Box::new(NewProc::LongLiteral(1)))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
            },
        };

        // Create body: Nil
        let body = AnnProc {
            proc: Box::leak(Box::new(NewProc::Nil)),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        // Create ForComprehension
        let for_comprehension = AnnProc {
            proc: Box::leak(Box::new(NewProc::ForComprehension {
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
        let result = normalize_ann_proc(&for_comprehension, inputs(), &HashMap::new(), &parser);
        assert!(result.is_err());
        assert_eq!(
            result,
            Err(InterpreterError::UnexpectedReuseOfNameContextFree {
                var_name: "y1".to_string(),
                first_use: "1:1".to_string(),
                second_use: "1:1".to_string(),
            })
        );
    }

    #[test]
    fn new_ast_p_input_should_not_compile_when_connectives_are_used_in_the_channel() {
        // Test disjunction in channel
        let result1 = Compiler::new_source_to_adt(r#"for(x <- @{Nil \/ Nil}){ Nil }"#);
        assert!(result1.is_err());
        match result1 {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("\\/ (disjunction)"));
            }
            other => panic!("Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}", other),
        }

        // Test conjunction in channel
        let result2 = Compiler::new_source_to_adt(r#"for(x <- @{Nil /\ Nil}){ Nil }"#);
        assert!(result2.is_err());
        match result2 {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("/\\ (conjunction)"));
            }
            other => panic!("Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}", other),
        }

        // Test negation in channel
        let result3 = Compiler::new_source_to_adt(r#"for(x <- @{~Nil}){ Nil }"#);
        assert!(result3.is_err());
        match result3 {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("~ (negation)"));
            }
            other => panic!("Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}", other),
        }
    }

    #[test]
    fn new_ast_p_input_should_not_compile_when_connectives_are_at_the_top_level_expression_in_the_body() {
        // Test conjunction in body
        let result1 = Compiler::new_source_to_adt(r#"for(x <- @Nil){ 1 /\ 2 }"#);
        assert!(result1.is_err());
        match result1 {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("/\\ (conjunction)"));
            }
            other => panic!("Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}", other),
        }

        // Test disjunction in body
        let result2 = Compiler::new_source_to_adt(r#"for(x <- @Nil){ 1 \/ 2 }"#);
        assert!(result2.is_err());
        match result2 {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("\\/ (disjunction)"));
            }
            other => panic!("Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}", other),
        }

        // Test negation in body
        let result3 = Compiler::new_source_to_adt(r#"for(x <- @Nil){ ~1 }"#);
        assert!(result3.is_err());
        match result3 {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("~ (negation)"));
            }
            other => panic!("Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}", other),
        }
    }

    #[test]
    fn new_ast_p_input_should_not_compile_when_logical_or_or_not_is_used_in_pattern_of_receive() {
        // Test disjunction in pattern
        let result1 = Compiler::new_source_to_adt(r#"new x in { for(@{Nil \/ Nil} <- x) { Nil } }"#);
        assert!(result1.is_err());
        match result1 {
            Err(InterpreterError::PatternReceiveError(msg)) => {
                assert!(msg.contains("\\/ (disjunction)"));
            }
            other => panic!("Expected PatternReceiveError, got: {:?}", other),
        }

        // Test negation in pattern
        let result2 = Compiler::new_source_to_adt(r#"new x in { for(@{~Nil} <- x) { Nil } }"#);
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
        // Test that conjunction in pattern is allowed (should compile successfully)
        let result1 = Compiler::new_source_to_adt(r#"new x in { for(@{Nil /\ Nil} <- x) { Nil } }"#);
        assert!(result1.is_ok(), "Conjunction in pattern should be allowed, but got error: {:?}", result1);
    }
}
