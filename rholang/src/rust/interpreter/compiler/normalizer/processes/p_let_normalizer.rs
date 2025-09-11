// See rholang/src/main/scala/coop/rchain/rholang/interpreter/compiler/normalizer/processes/PLetNormalizer.scala

use std::collections::HashMap;

use models::{
    rhoapi::{MatchCase, Par},
    rust::utils::{new_elist_par, new_match_par, union},
};
use uuid::Uuid;

use crate::rust::interpreter::{
    compiler::{
        exports::{FreeMap, ProcVisitInputsSpan, ProcVisitOutputsSpan},
        normalize::{normalize_match_proc, NameVisitInputs, NameVisitOutputs, VarSort},
        normalizer::{
            name_normalize_matcher::normalize_name,
            remainder_normalizer_matcher::normalize_match_name,
        },
        rholang_ast::{
            Block, Decl, Decls, DeclsChoice, LinearBind, Name, NameDecl, Names, Proc, ProcList,
            Receipt, Receipts, SendType, Source, Var,
        },
    },
    matcher::has_locally_free::HasLocallyFree,
    util::filter_and_adjust_bitset,
};

use super::exports::{InterpreterError, ProcVisitInputs, ProcVisitOutputs};

// TODO: This file is going to need a review because we do not have a single 'decl' on our 'let' rule
pub fn normalize_p_let(
    decls_choice: &DeclsChoice,
    body: &Block,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    type ListName = Vec<Name>;
    type NameRemainder = Option<Box<Proc>>;
    type ListProc = Vec<Proc>;

    match decls_choice.clone() {
        DeclsChoice::ConcDecls { decls, .. } => {
            fn extract_names_and_procs(decl: Decl) -> (ListName, NameRemainder, ListProc) {
                (decl.names.names, decl.names.cont, decl.procs)
            }

            // We don't have a single 'decl' on 'let' rule in grammar, so I assume we just map over 'decls'
            let (list_names, list_name_remainders, list_procs): (
                Vec<ListName>,
                Vec<NameRemainder>,
                Vec<ListProc>,
            ) = {
                let tuples: Vec<(ListName, NameRemainder, ListProc)> = decls
                    .into_iter()
                    .map(|conc_decl_impl| extract_names_and_procs(conc_decl_impl))
                    .collect();
                unzip3(tuples)
            };

            /*
            It is not necessary to use UUIDs to achieve concurrent let declarations.
            While there is the possibility for collisions with either variables declared by the user
            or variables declared within this translation, the chances for collision are astronomically
            small (see analysis here: https://towardsdatascience.com/are-uuids-really-unique-57eb80fc2a87).
            A strictly correct approach would be one that performs a ADT rather than an AST translation, which
            was not done here due to time constraints. - OLD
            */
            let variable_names: Vec<String> = (0..list_names.len())
                .map(|_| Uuid::new_v4().to_string())
                .collect();

            let p_sends: Vec<Proc> = variable_names
                .clone()
                .into_iter()
                .zip(list_procs)
                .map(|(variable_name, list_proc)| Proc::Send {
                    name: Name::ProcVar(Box::new(Proc::Var(Var {
                        name: variable_name,
                        line_num: 0,
                        col_num: 0,
                    }))),
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
                })
                .collect();

            let p_input = {
                let list_linear_bind: Vec<Receipt> = variable_names
                    .clone()
                    .into_iter()
                    .zip(list_names)
                    .zip(list_name_remainders)
                    .map(|((variable_name, list_name), name_remainder)| {
                        Receipt::LinearBinds(LinearBind {
                            names: Names {
                                names: list_name,
                                cont: name_remainder,
                                line_num: 0,
                                col_num: 0,
                            },
                            input: Source::Simple {
                                name: Name::ProcVar(Box::new(Proc::Var(Var {
                                    name: variable_name,
                                    line_num: 0,
                                    col_num: 0,
                                }))),
                                line_num: 0,
                                col_num: 0,
                            },
                            line_num: 0,
                            col_num: 0,
                        })
                    })
                    .collect();

                let list_receipt = Receipts {
                    receipts: list_linear_bind,
                    line_num: 0,
                    col_num: 0,
                };

                Proc::Input {
                    formals: list_receipt,
                    proc: Box::new(body.clone()),
                    line_num: 0,
                    col_num: 0,
                }
            };

            let p_par = {
                let procs = {
                    let mut combined = p_sends;
                    combined.push(p_input);
                    combined
                };

                procs.clone().into_iter().skip(2).fold(
                    Proc::Par {
                        left: Box::new(procs[0].clone()),
                        right: Box::new(procs[1].clone()),
                        line_num: 0,
                        col_num: 0,
                    },
                    |p_par, proc| Proc::Par {
                        left: Box::new(p_par),
                        right: Box::new(proc),
                        line_num: 0,
                        col_num: 0,
                    },
                )
            };

            let p_new = Proc::New {
                decls: Decls {
                    decls: variable_names
                        .into_iter()
                        .map(|name| NameDecl {
                            var: Var {
                                name,
                                line_num: 0,
                                col_num: 0,
                            },
                            uri: None,
                            line_num: 0,
                            col_num: 0,
                        })
                        .collect(),
                    line_num: 0,
                    col_num: 0,
                },
                proc: Box::new(p_par),
                line_num: 0,
                col_num: 0,
            };

            normalize_match_proc(&p_new, input, env)
        }

        /*
        Let processes with a single bind or with sequential binds ";" are converted into match processes rather
        than input processes, so that each sequential bind doesn't add a new unforgeable name to the tuplespace.
        The Rholang 1.1 spec defines them as the latter. Because the Rholang 1.1 spec defines let processes in terms
        of a output process in concurrent composition with an input process, the let process appears to quote the
        process on the RHS of "<-" and bind it to the pattern on LHS. For example, in
            let x <- 1 in { Nil }
        the process (value) "1" is quoted and bound to "x" as a name. There is no way to perform an AST transformation
        of sequential let into a match process and still preserve these semantics, so we have to do an ADT transformation.
         */
        DeclsChoice::LinearDecls { decls, .. } => {
            let new_continuation = if decls.is_empty() {
                body.proc.clone()
            } else {
                // Similarly here, we don't have a single 'decl' field on `let' rule.
                let new_decls: Vec<Decl> = if decls.len() == 1 {
                    decls.clone()
                } else {
                    let mut new_linear_decls = Vec::new();
                    for decl in decls.clone().into_iter().skip(1) {
                        new_linear_decls.push(decl);
                    }
                    new_linear_decls
                };

                Proc::Let {
                    decls: DeclsChoice::LinearDecls {
                        decls: new_decls,
                        line_num: 0,
                        col_num: 0,
                    },
                    body: Box::new(body.clone()),
                    line_num: 0,
                    col_num: 0,
                }
            };

            fn list_proc_to_elist(
                list_proc: Vec<Proc>,
                known_free: FreeMap<VarSort>,
                input: ProcVisitInputs,
                env: &HashMap<String, Par>,
            ) -> Result<ProcVisitOutputs, InterpreterError> {
                let mut vector_par = Vec::new();
                let mut current_known_free = known_free;
                let mut locally_free = Vec::new();
                let mut connective_used = false;

                for proc in list_proc {
                    let ProcVisitOutputs {
                        par,
                        free_map: updated_known_free,
                    } = normalize_match_proc(
                        &proc,
                        ProcVisitInputs {
                            par: Par::default(),
                            bound_map_chain: input.bound_map_chain.clone(),
                            free_map: current_known_free,
                        },
                        env,
                    )?;

                    vector_par.insert(0, par.clone());
                    current_known_free = updated_known_free;
                    locally_free = union(locally_free, par.locally_free);
                    connective_used |= par.connective_used;
                }

                Ok(ProcVisitOutputs {
                    par: new_elist_par(
                        vector_par.into_iter().rev().collect(),
                        locally_free,
                        connective_used,
                        None,
                        Vec::new(),
                        false,
                    ),
                    free_map: current_known_free,
                })
            }

            fn list_name_to_elist(
                list_name: Vec<Name>,
                name_remainder: &NameRemainder,
                input: ProcVisitInputs,
                env: &HashMap<String, Par>,
            ) -> Result<ProcVisitOutputs, InterpreterError> {
                let (optional_var, remainder_known_free) =
                    normalize_match_name(name_remainder, FreeMap::new())?;

                let mut vector_par = Vec::new();
                let mut current_known_free = remainder_known_free;
                let mut locally_free = Vec::new();

                for name in list_name {
                    let NameVisitOutputs {
                        par,
                        free_map: updated_known_free,
                    } = normalize_name(
                        &name,
                        NameVisitInputs {
                            bound_map_chain: input.bound_map_chain.push(),
                            free_map: current_known_free,
                        },
                        env,
                    )?;

                    vector_par.insert(0, par.clone());
                    current_known_free = updated_known_free;
                    // Use input.env.depth + 1 because the pattern was evaluated w.r.t input.env.push,
                    // and more generally because locally free variables become binders in the pattern position
                    locally_free = union(
                        locally_free,
                        par.clone()
                            .locally_free(par, input.bound_map_chain.depth() as i32 + 1),
                    );
                }

                Ok(ProcVisitOutputs {
                    par: new_elist_par(
                        vector_par.into_iter().rev().collect(),
                        locally_free,
                        true,
                        optional_var,
                        Vec::new(),
                        false,
                    ),
                    free_map: current_known_free,
                })
            }

            // Again, we don't have a single 'decl' field on 'let' I am using the first 'decl' element
            let decl = decls[0].clone();

            list_proc_to_elist(decl.procs, input.clone().free_map, input.clone(), env).and_then(
                |ProcVisitOutputs {
                     par: value_list_par,
                     free_map: value_known_free,
                 }| {
                    list_name_to_elist(decl.names.names, &decl.names.cont, input.clone(), env)
                        .and_then(
                            |ProcVisitOutputs {
                                 par: pattern_list_par,
                                 free_map: pattern_known_free,
                             }| {
                                normalize_match_proc(
                                    &new_continuation,
                                    ProcVisitInputs {
                                        par: Par::default(),
                                        bound_map_chain: input
                                            .bound_map_chain
                                            .absorb_free(pattern_known_free.clone()),
                                        free_map: value_known_free,
                                    },
                                    env,
                                )
                                .map(
                                    |ProcVisitOutputs {
                                         par: continuation_par,
                                         free_map: continuation_known_free,
                                     }| ProcVisitOutputs {
                                        par: new_match_par(
                                            value_list_par.clone(),
                                            vec![MatchCase {
                                                pattern: Some(pattern_list_par.clone()),
                                                source: Some(continuation_par.clone()),
                                                free_count: pattern_known_free.count_no_wildcards()
                                                    as i32,
                                            }],
                                            union(
                                                value_list_par.locally_free,
                                                union(
                                                    pattern_list_par.locally_free,
                                                    filter_and_adjust_bitset(
                                                        continuation_par.locally_free,
                                                        pattern_known_free.count_no_wildcards(),
                                                    ),
                                                ),
                                            ),
                                            value_list_par.connective_used
                                                || continuation_par.connective_used,
                                            Vec::new(),
                                            false,
                                        ),
                                        free_map: continuation_known_free,
                                    },
                                )
                            },
                        )
                },
            )
        }
    }
}

pub fn normalize_p_let_new_ast<'ast>(
    bindings: &'ast smallvec::SmallVec<[rholang_parser::ast::LetBinding<'ast>; 1]>,
    body: &'ast rholang_parser::ast::AnnProc<'ast>,
    concurrent: bool,
    let_span: rholang_parser::SourceSpan,
    input: ProcVisitInputsSpan,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputsSpan, InterpreterError> {
    use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
    use crate::rust::interpreter::compiler::span_utils::SpanContext;

    if concurrent {
        // Concurrent let declarations - similar to ConcDecls in original
        // Transform into new declarations with sends and input process

        // RHOLANG-RS IMPROVEMENT: Could use semantic naming based on actual variable names
        // e.g., "__let_x_0_L5C10" for variable 'x' at binding index 0, line 5, col 10
        // This would extract name hints from lhs.name for Single bindings and lhs for Multiple
        let variable_names: Vec<String> = (0..bindings.len())
            .map(|_| Uuid::new_v4().to_string())
            .collect();

        // Create send processes for each binding
        let mut send_processes = Vec::new();

        for (i, binding) in bindings.iter().enumerate() {
            let variable_name = &variable_names[i];

            match binding {
                rholang_parser::ast::LetBinding::Single { rhs, .. } => {
                    // Derive spans from actual rhs location
                    let rhs_span = rhs.span;
                    let variable_span = SpanContext::variable_span_from_binding(rhs_span, i);
                    let send_span = SpanContext::synthetic_construct_span(rhs_span, 10); // Offset to mark as send

                    // Create send: variable_name!(rhs)
                    let send_proc = rholang_parser::ast::AnnProc {
                        proc: parser.ast_builder().alloc_send(
                            rholang_parser::ast::SendType::Single,
                            rholang_parser::ast::AnnName {
                                name: rholang_parser::ast::Name::ProcVar(
                                    rholang_parser::ast::Var::Id(rholang_parser::ast::Id {
                                        name: parser.ast_builder().alloc_str(&variable_name),
                                        pos: variable_span.start,
                                    }),
                                ),
                                span: variable_span,
                            },
                            &[*rhs],
                        ),
                        span: send_span,
                    };
                    send_processes.push(send_proc);
                }

                rholang_parser::ast::LetBinding::Multiple { rhs, .. } => {
                    // Derive span from range of all rhs expressions
                    let rhs_span = if rhs.len() > 1 {
                        SpanContext::merge_two_spans(rhs[0].span, rhs[rhs.len() - 1].span)
                    } else if rhs.len() == 1 {
                        rhs[0].span
                    } else {
                        let_span // Fallback to let construct span
                    };
                    let variable_span = SpanContext::variable_span_from_binding(rhs_span, i);
                    // RHOLANG-RS IMPROVEMENT: Could use SpanContext::send_span_from_binding for better accuracy
                    let send_span = SpanContext::synthetic_construct_span(rhs_span, 10); // Offset to mark as send

                    // Create send: variable_name!(rhs[0], rhs[1], ...)
                    let send_proc = rholang_parser::ast::AnnProc {
                        proc: parser.ast_builder().alloc_send(
                            rholang_parser::ast::SendType::Single,
                            rholang_parser::ast::AnnName {
                                name: rholang_parser::ast::Name::ProcVar(
                                    rholang_parser::ast::Var::Id(rholang_parser::ast::Id {
                                        // Using rholang-rs arena allocation instead of Box::leak
                                        name: parser.ast_builder().alloc_str(&variable_name),
                                        pos: variable_span.start,
                                    }),
                                ),
                                span: variable_span,
                            },
                            rhs,
                        ),
                        span: send_span,
                    };
                    send_processes.push(send_proc);
                }
            }
        }

        // Create input process binds for each binding
        let mut input_binds: Vec<smallvec::SmallVec<[rholang_parser::ast::Bind<'ast>; 1]>> =
            Vec::new();

        for (i, binding) in bindings.iter().enumerate() {
            let variable_name = &variable_names[i];

            match binding {
                rholang_parser::ast::LetBinding::Single { lhs, .. } => {
                    // Derive spans from actual lhs location
                    let lhs_span = lhs.span;
                    let variable_span = SpanContext::variable_span_from_binding(lhs_span, i);

                    // Create bind: lhs <- variable_name
                    let bind = rholang_parser::ast::Bind::Linear {
                        lhs: rholang_parser::ast::Names {
                            names: smallvec::SmallVec::from_vec(vec![*lhs]),
                            remainder: None,
                        },
                        rhs: rholang_parser::ast::Source::Simple {
                            name: rholang_parser::ast::AnnName {
                                name: rholang_parser::ast::Name::ProcVar(
                                    rholang_parser::ast::Var::Id(rholang_parser::ast::Id {
                                        // Using rholang-rs arena allocation instead of Box::leak
                                        name: parser.ast_builder().alloc_str(&variable_name),
                                        pos: variable_span.start,
                                    }),
                                ),
                                span: variable_span,
                            },
                        },
                    };
                    input_binds.push(smallvec::SmallVec::from_vec(vec![bind]));
                }

                rholang_parser::ast::LetBinding::Multiple { lhs, rhs, .. } => {
                    // RHOLANG-RS IMPROVEMENT: For Multiple bindings, lhs is Var<'ast>, not AnnName<'ast>
                    // Could extract precise position from Var::Id(id) => id.pos, vs Var::Wildcard (no position)
                    // Currently deriving from first rhs, but should distinguish between these cases
                    let lhs_span = rhs.get(0).map(|r| r.span).unwrap_or(let_span); // Use first rhs or let span
                    let variable_span = SpanContext::variable_span_from_binding(lhs_span, i);

                    // Create bind: lhs, _, _, ... <- variable_name (with wildcards for extra values)
                    let mut names = vec![rholang_parser::ast::AnnName {
                        name: rholang_parser::ast::Name::ProcVar(*lhs),
                        span: lhs_span, // Use derived span
                    }];

                    // Add wildcards for remaining values
                    // RHOLANG-RS LIMITATION: Var::Wildcard has no position data in rholang-rs
                    // Our wildcard_span_with_context approach is actually optimal given this constraint
                    for _ in 1..rhs.len() {
                        let wildcard_span = SpanContext::wildcard_span_with_context(lhs_span);
                        names.push(rholang_parser::ast::AnnName {
                            name: rholang_parser::ast::Name::ProcVar(
                                rholang_parser::ast::Var::Wildcard,
                            ),
                            span: wildcard_span,
                        });
                    }

                    let bind = rholang_parser::ast::Bind::Linear {
                        lhs: rholang_parser::ast::Names {
                            names: smallvec::SmallVec::from_vec(names),
                            remainder: None,
                        },
                        rhs: rholang_parser::ast::Source::Simple {
                            name: rholang_parser::ast::AnnName {
                                name: rholang_parser::ast::Name::ProcVar(
                                    rholang_parser::ast::Var::Id(rholang_parser::ast::Id {
                                        // Using rholang-rs arena allocation instead of Box::leak
                                        name: parser.ast_builder().alloc_str(&variable_name),
                                        pos: variable_span.start,
                                    }),
                                ),
                                span: variable_span,
                            },
                        },
                    };
                    input_binds.push(smallvec::SmallVec::from_vec(vec![bind]));
                }
            }
        }

        // Create the for-comprehension (input process)
        // Use body span as this is the primary process being executed
        let for_comprehension = rholang_parser::ast::AnnProc {
            proc: parser.ast_builder().alloc_for(input_binds, *body),
            span: body.span, // Use actual body span for accurate debugging
        };

        // Create parallel composition of all sends and the for-comprehension
        let mut all_processes = send_processes;
        all_processes.push(for_comprehension);

        // Build parallel composition
        let par_proc = if all_processes.len() == 1 {
            all_processes[0]
        } else {
            // Create initial parallel composition with meaningful span
            let first_span = all_processes[0].span;
            let second_span = all_processes[1].span;
            let initial_par_span = SpanContext::merge_two_spans(first_span, second_span);

            let mut result = rholang_parser::ast::AnnProc {
                proc: parser
                    .ast_builder()
                    .alloc_par(all_processes[0], all_processes[1]),
                span: initial_par_span,
            };

            // Add remaining processes, expanding span to cover all
            for proc in all_processes.iter().skip(2) {
                let expanded_span = SpanContext::merge_two_spans(result.span, proc.span);
                result = rholang_parser::ast::AnnProc {
                    proc: parser.ast_builder().alloc_par(result, *proc),
                    span: expanded_span,
                };
            }
            result
        };

        // Create new declaration with all variable names
        let name_decls: Vec<rholang_parser::ast::NameDecl> = variable_names
            .into_iter()
            .enumerate()
            .map(|(idx, name)| {
                let decl_span = SpanContext::variable_span_from_binding(let_span, idx);
                rholang_parser::ast::NameDecl {
                    id: rholang_parser::ast::Id {
                        // Using rholang-rs arena allocation instead of Box::leak
                        name: parser.ast_builder().alloc_str(&name),
                        pos: decl_span.start,
                    },
                    uri: None,
                }
            })
            .collect();

        // The new process spans the entire let construct
        let new_proc = rholang_parser::ast::AnnProc {
            proc: parser.ast_builder().alloc_new(par_proc, name_decls),
            span: let_span, // Use the original let span for the entire construct
        };

        // Normalize the constructed new process
        normalize_ann_proc(&new_proc, input, env, parser)
    } else {
        // Sequential let declarations - similar to LinearDecls in original
        // Transform into match process

        if bindings.is_empty() {
            // Empty bindings - just normalize the body
            return normalize_ann_proc(body, input, env, parser);
        }

        // For sequential let, we process one binding at a time
        // let x <- rhs in body becomes match rhs { x => body }

        let first_binding = &bindings[0];

        match first_binding {
            rholang_parser::ast::LetBinding::Single { lhs, rhs } => {
                // RHOLANG-RS STRENGTH: Single bindings have rich AnnName<'ast> with full span info
                // lhs.name provides precise Name enum (ProcVar or Quote) and lhs.span gives full range
                let lhs_span = lhs.span;
                let rhs_span = rhs.span;
                let pattern_span = SpanContext::synthetic_construct_span(lhs_span, 5); // Offset for pattern

                // Create match case
                let match_case = rholang_parser::ast::Case {
                    pattern: rholang_parser::ast::AnnProc {
                        proc: parser
                            .ast_builder()
                            .alloc_list(&[rholang_parser::ast::AnnProc {
                                proc: parser.ast_builder().alloc_eval(*lhs),
                                span: lhs_span, // Use actual lhs span
                            }]),
                        span: pattern_span, // Use synthetic pattern span
                    },
                    proc: if bindings.len() > 1 {
                        // More bindings - create nested let
                        let remaining_bindings: smallvec::SmallVec<
                            [rholang_parser::ast::LetBinding<'ast>; 1],
                        > = smallvec::SmallVec::from_vec(bindings[1..].to_vec());
                        let nested_span = SpanContext::merge_two_spans(body.span, let_span);
                        rholang_parser::ast::AnnProc {
                            proc: parser
                                .ast_builder()
                                .alloc_let(remaining_bindings, *body, false),
                            span: nested_span, // Use merged span for nested let
                        }
                    } else {
                        // Last binding - use body directly
                        *body
                    },
                };

                // Create match expression from rhs
                let match_expr_span = rhs_span;
                let match_expr = rholang_parser::ast::AnnProc {
                    proc: parser.ast_builder().alloc_list(&[*rhs]),
                    span: match_expr_span, // Use actual rhs span
                };

                // Create match process spanning from rhs to body
                let match_span = SpanContext::merge_two_spans(rhs_span, body.span);
                let match_proc = rholang_parser::ast::AnnProc {
                    proc: parser
                        .ast_builder()
                        .alloc_match(match_expr, &[match_case.pattern, match_case.proc]),
                    span: match_span, // Use derived match span
                };

                normalize_ann_proc(&match_proc, input, env, parser)
            }

            rholang_parser::ast::LetBinding::Multiple { lhs, rhs } => {
                // Multiple binding: let x <- (rhs1, rhs2, ...) in body
                // becomes: match [rhs1, rhs2, ...] { [x, _, _, ...] => body }

                // RHOLANG-RS IMPROVEMENT: Could leverage lhs position data more precisely
                // For Var::Id(id), use id.pos directly; for Var::Wildcard, no position available
                // Currently using first rhs as context, but could be more semantic
                let lhs_span = rhs.get(0).map(|r| r.span).unwrap_or(let_span); // Use first rhs or let span
                let rhs_list_span = if rhs.len() > 1 {
                    SpanContext::merge_two_spans(rhs[0].span, rhs[rhs.len() - 1].span)
                } else if rhs.len() == 1 {
                    rhs[0].span
                } else {
                    let_span // Fallback
                };

                // Create pattern elements with proper spans
                let lhs_name_span = SpanContext::synthetic_construct_span(lhs_span, 0);
                let mut pattern_elements = vec![rholang_parser::ast::AnnProc {
                    proc: parser
                        .ast_builder()
                        .alloc_eval(rholang_parser::ast::AnnName {
                            name: rholang_parser::ast::Name::ProcVar(*lhs),
                            span: lhs_name_span,
                        }),
                    span: lhs_name_span,
                }];

                // Add wildcards for remaining values
                for _ in 1..rhs.len() {
                    let wildcard_span = SpanContext::wildcard_span_with_context(lhs_span);
                    pattern_elements.push(rholang_parser::ast::AnnProc {
                        proc: parser.ast_builder().const_wild(),
                        span: wildcard_span,
                    });
                }

                let pattern_list_span = SpanContext::synthetic_construct_span(lhs_span, 10);
                let match_case = rholang_parser::ast::Case {
                    pattern: rholang_parser::ast::AnnProc {
                        proc: parser.ast_builder().alloc_list(&pattern_elements),
                        span: pattern_list_span,
                    },
                    proc: if bindings.len() > 1 {
                        // More bindings - create nested let
                        let remaining_bindings: smallvec::SmallVec<
                            [rholang_parser::ast::LetBinding<'ast>; 1],
                        > = smallvec::SmallVec::from_vec(bindings[1..].to_vec());
                        let nested_span = SpanContext::merge_two_spans(body.span, let_span);
                        rholang_parser::ast::AnnProc {
                            proc: parser
                                .ast_builder()
                                .alloc_let(remaining_bindings, *body, false),
                            span: nested_span,
                        }
                    } else {
                        // Last binding - use body directly
                        *body
                    },
                };

                // Create match expression from rhs list
                let match_expr = rholang_parser::ast::AnnProc {
                    proc: parser.ast_builder().alloc_list(rhs),
                    span: rhs_list_span, // Use span covering all rhs expressions
                };

                // Create match process
                let match_span = SpanContext::merge_two_spans(rhs_list_span, body.span);
                let match_proc = rholang_parser::ast::AnnProc {
                    proc: parser
                        .ast_builder()
                        .alloc_match(match_expr, &[match_case.pattern, match_case.proc]),
                    span: match_span, // Use span from rhs to body
                };

                normalize_ann_proc(&match_proc, input, env, parser)
            }
        }
    }
}

fn unzip3<T, U, V>(tuples: Vec<(Vec<T>, U, Vec<V>)>) -> (Vec<Vec<T>>, Vec<U>, Vec<Vec<V>>) {
    let mut vec_t = Vec::with_capacity(tuples.len());
    let mut vec_u = Vec::with_capacity(tuples.len());
    let mut vec_v = Vec::with_capacity(tuples.len());

    for (t, u, v) in tuples {
        vec_t.push(t);
        vec_u.push(u);
        vec_v.push(v);
    }

    (vec_t, vec_u, vec_v)
}

//rholang/src/test/scala/coop/rchain/rholang/interpreter/LetSpec.scala
#[cfg(test)]
mod tests {
    use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
    use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;
    use crate::rust::interpreter::test_utils::utils::proc_visit_inputs_and_env_span;
    use models::rhoapi::Par;
    use pretty_assertions::assert_eq;
    use rholang_parser::ast::{
        AnnName as NewAnnName, AnnProc as NewAnnProc, Id, LetBinding as NewLetBinding,
        Name as NewName, Proc as NewProc, Var as NewVar,
    };
    use rholang_parser::{SourcePos, SourceSpan};

    // Helper functions for future string-based tests when parser supports let syntax
    #[allow(dead_code)]
    fn get_normalized_par(rho: &str) -> Par {
        ParBuilderUtil::mk_term(rho).expect("Compilation failed to normalize Par")
    }

    #[allow(dead_code)]
    pub fn assert_equal_normalized(rho1: &str, rho2: &str) {
        assert_eq!(
            get_normalized_par(rho1),
            get_normalized_par(rho2),
            "Normalized Par values are not equal"
        );
    }

    // New AST tests - mapped from original LetSpec.scala tests
    #[test]
    fn new_ast_translate_single_declaration_into_match_process() {
        // Maps to: "translate a single declaration of multiple variables into a list match process"
        let (inputs, env) = proc_visit_inputs_and_env_span();

        // Create: let x <- 42 in { @x!("result") }
        let let_proc = NewAnnProc {
            proc: Box::leak(Box::new(NewProc::Let {
                bindings: smallvec::SmallVec::from_vec(vec![NewLetBinding::Single {
                    lhs: NewAnnName {
                        name: NewName::ProcVar(NewVar::Id(Id {
                            name: "x",
                            pos: SourcePos { line: 0, col: 0 },
                        })),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    rhs: NewAnnProc {
                        proc: Box::leak(Box::new(NewProc::LongLiteral(42))),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                }]),
                body: NewAnnProc {
                    proc: Box::leak(Box::new(NewProc::Send {
                        channel: NewAnnName {
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
                        inputs: smallvec::SmallVec::from_vec(vec![NewAnnProc {
                            proc: Box::leak(Box::new(NewProc::StringLiteral("result"))),
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
                concurrent: false,
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&let_proc, inputs.clone(), &env, &parser);
        assert!(result.is_ok());

        // Should transform into a match process
        let normalized = result.unwrap();
        assert!(normalized.par.matches.len() > 0);
    }

    #[test]
    fn new_ast_translate_concurrent_declarations_into_comm() {
        // Maps to: "translate multiple concurrent let declarations into a COMM"
        let (inputs, env) = proc_visit_inputs_and_env_span();

        // Create: let x <- 1, y <- 2 in { @x!(@y) } (concurrent)
        let let_proc = NewAnnProc {
            proc: Box::leak(Box::new(NewProc::Let {
                bindings: smallvec::SmallVec::from_vec(vec![
                    NewLetBinding::Single {
                        lhs: NewAnnName {
                            name: NewName::ProcVar(NewVar::Id(Id {
                                name: "x",
                                pos: SourcePos { line: 0, col: 0 },
                            })),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                        rhs: NewAnnProc {
                            proc: Box::leak(Box::new(NewProc::LongLiteral(1))),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                    },
                    NewLetBinding::Single {
                        lhs: NewAnnName {
                            name: NewName::ProcVar(NewVar::Id(Id {
                                name: "y",
                                pos: SourcePos { line: 0, col: 0 },
                            })),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                        rhs: NewAnnProc {
                            proc: Box::leak(Box::new(NewProc::LongLiteral(2))),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                    },
                ]),
                body: NewAnnProc {
                    proc: Box::leak(Box::new(NewProc::Send {
                        channel: NewAnnName {
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
                        inputs: smallvec::SmallVec::from_vec(vec![NewAnnProc {
                            proc: Box::leak(Box::new(NewProc::Eval {
                                name: NewAnnName {
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
                        }]),
                    })),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                concurrent: true,
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&let_proc, inputs.clone(), &env, &parser);
        assert!(result.is_ok());

        // Should transform into a new process with sends and receives
        let normalized = result.unwrap();
        assert!(normalized.par.news.len() > 0); // Should have new declarations
    }

    #[test]
    fn new_ast_handle_multiple_variable_declaration() {
        // Maps to: "translate a single declaration of multiple variables into a list match process"
        let (inputs, env) = proc_visit_inputs_and_env_span();

        // Create: let x <- (1, 2, 3) in { @x!("got first") }
        let let_proc = NewAnnProc {
            proc: Box::leak(Box::new(NewProc::Let {
                bindings: smallvec::SmallVec::from_vec(vec![NewLetBinding::Multiple {
                    lhs: NewVar::Id(Id {
                        name: "x",
                        pos: SourcePos { line: 0, col: 0 },
                    }),
                    rhs: vec![
                        NewAnnProc {
                            proc: Box::leak(Box::new(NewProc::LongLiteral(1))),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                        NewAnnProc {
                            proc: Box::leak(Box::new(NewProc::LongLiteral(2))),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                        NewAnnProc {
                            proc: Box::leak(Box::new(NewProc::LongLiteral(3))),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                    ],
                }]),
                body: NewAnnProc {
                    proc: Box::leak(Box::new(NewProc::Send {
                        channel: NewAnnName {
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
                        inputs: smallvec::SmallVec::from_vec(vec![NewAnnProc {
                            proc: Box::leak(Box::new(NewProc::StringLiteral("got first"))),
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
                concurrent: false,
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&let_proc, inputs.clone(), &env, &parser);
        assert!(result.is_ok());

        // Should transform into a match process with list pattern
        let normalized = result.unwrap();
        assert!(normalized.par.matches.len() > 0);
    }

    #[test]
    fn new_ast_handle_empty_bindings() {
        // Edge case: empty bindings should just normalize the body
        let (inputs, env) = proc_visit_inputs_and_env_span();

        // Create: let in { @"stdout"!("hello") }
        let let_proc = NewAnnProc {
            proc: Box::leak(Box::new(NewProc::Let {
                bindings: smallvec::SmallVec::new(),
                body: NewAnnProc {
                    proc: Box::leak(Box::new(NewProc::Send {
                        channel: NewAnnName {
                            name: NewName::Quote(Box::leak(Box::new(NewProc::StringLiteral(
                                "stdout",
                            )))),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                        send_type: rholang_parser::ast::SendType::Single,
                        inputs: smallvec::SmallVec::from_vec(vec![NewAnnProc {
                            proc: Box::leak(Box::new(NewProc::StringLiteral("hello"))),
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
                concurrent: false,
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&let_proc, inputs.clone(), &env, &parser);
        assert!(result.is_ok());

        // Should just normalize the body directly
        let normalized = result.unwrap();
        assert!(normalized.par.sends.len() > 0);
    }

    #[test]
    fn new_ast_translate_sequential_declarations_into_nested_matches() {
        // Maps to: "translate multiple sequential let declarations into nested match processes"
        let (inputs, env) = proc_visit_inputs_and_env_span();

        // Create: let x <- 1 in { let y <- 2 in { @x!(@y) } }
        let inner_let = NewAnnProc {
            proc: Box::leak(Box::new(NewProc::Let {
                bindings: smallvec::SmallVec::from_vec(vec![NewLetBinding::Single {
                    lhs: NewAnnName {
                        name: NewName::ProcVar(NewVar::Id(Id {
                            name: "y",
                            pos: SourcePos { line: 0, col: 0 },
                        })),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    rhs: NewAnnProc {
                        proc: Box::leak(Box::new(NewProc::LongLiteral(2))),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                }]),
                body: NewAnnProc {
                    proc: Box::leak(Box::new(NewProc::Send {
                        channel: NewAnnName {
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
                        inputs: smallvec::SmallVec::from_vec(vec![NewAnnProc {
                            proc: Box::leak(Box::new(NewProc::Eval {
                                name: NewAnnName {
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
                        }]),
                    })),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                concurrent: false,
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let outer_let = NewAnnProc {
            proc: Box::leak(Box::new(NewProc::Let {
                bindings: smallvec::SmallVec::from_vec(vec![NewLetBinding::Single {
                    lhs: NewAnnName {
                        name: NewName::ProcVar(NewVar::Id(Id {
                            name: "x",
                            pos: SourcePos { line: 0, col: 0 },
                        })),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    rhs: NewAnnProc {
                        proc: Box::leak(Box::new(NewProc::LongLiteral(1))),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                }]),
                body: inner_let,
                concurrent: false,
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&outer_let, inputs.clone(), &env, &parser);
        assert!(result.is_ok());

        // Should transform into nested match processes
        let normalized = result.unwrap();
        assert!(normalized.par.matches.len() > 0);
    }
}
