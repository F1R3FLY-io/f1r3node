// See rholang/src/main/scala/coop/rchain/rholang/interpreter/compiler/normalizer/processes/PLetNormalizer.scala

use super::exports::InterpreterError;
use crate::rust::interpreter::compiler::{
    exports::{ProcVisitInputsSpan, ProcVisitOutputsSpan},
    normalize::normalize_ann_proc,
    span_utils::SpanContext,
};
use models::rhoapi::Par;
use std::collections::HashMap;
use uuid::Uuid;

use rholang_parser::ast::{
    AnnName, AnnProc, Bind, Id, LetBinding, Name, NameDecl, Names, SendType, Source, Var,
};
use rholang_parser::SourceSpan;

pub fn normalize_p_let<'ast>(
    bindings: &'ast smallvec::SmallVec<[LetBinding<'ast>; 1]>,
    body: &'ast AnnProc<'ast>,
    concurrent: bool,
    let_span: SourceSpan,
    input: ProcVisitInputsSpan,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputsSpan, InterpreterError> {
    if concurrent {
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
                LetBinding::Single { rhs, .. } => {
                    // Derive spans from actual rhs location
                    let rhs_span = rhs.span;
                    let variable_span = SpanContext::variable_span_from_binding(rhs_span, i);
                    let send_span = SpanContext::synthetic_construct_span(rhs_span, 10); // Offset to mark as send

                    // Create send: variable_name!(rhs)
                    let send_proc = AnnProc {
                        proc: parser.ast_builder().alloc_send(
                            SendType::Single,
                            AnnName {
                                name: Name::ProcVar(Var::Id(Id {
                                    name: parser.ast_builder().alloc_str(&variable_name),
                                    pos: variable_span.start,
                                })),
                                span: variable_span,
                            },
                            &[*rhs],
                        ),
                        span: send_span,
                    };
                    send_processes.push(send_proc);
                }

                LetBinding::Multiple { rhs, .. } => {
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
                    let send_proc = AnnProc {
                        proc: parser.ast_builder().alloc_send(
                            SendType::Single,
                            AnnName {
                                name: Name::ProcVar(Var::Id(Id {
                                    name: parser.ast_builder().alloc_str(&variable_name),
                                    pos: variable_span.start,
                                })),
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
        let mut input_binds: Vec<smallvec::SmallVec<[Bind<'ast>; 1]>> = Vec::new();

        for (i, binding) in bindings.iter().enumerate() {
            let variable_name = &variable_names[i];

            match binding {
                LetBinding::Single { lhs, .. } => {
                    // Derive spans from actual lhs location
                    let lhs_span = lhs.span;
                    let variable_span = SpanContext::variable_span_from_binding(lhs_span, i);

                    // Create bind: lhs <- variable_name
                    let bind = Bind::Linear {
                        lhs: Names {
                            names: smallvec::SmallVec::from_vec(vec![*lhs]),
                            remainder: None,
                        },
                        rhs: Source::Simple {
                            name: AnnName {
                                name: Name::ProcVar(Var::Id(Id {
                                    name: parser.ast_builder().alloc_str(&variable_name),
                                    pos: variable_span.start,
                                })),
                                span: variable_span,
                            },
                        },
                    };
                    input_binds.push(smallvec::SmallVec::from_vec(vec![bind]));
                }

                LetBinding::Multiple { lhs, rhs, .. } => {
                    // RHOLANG-RS IMPROVEMENT: For Multiple bindings, lhs is Var<'ast>, not AnnName<'ast>
                    // Could extract precise position from Var::Id(id) => id.pos, vs Var::Wildcard (no position)
                    // Currently deriving from first rhs, but should distinguish between these cases
                    let lhs_span = rhs.get(0).map(|r| r.span).unwrap_or(let_span); // Use first rhs or let span
                    let variable_span = SpanContext::variable_span_from_binding(lhs_span, i);

                    // Create bind: lhs, _, _, ... <- variable_name (with wildcards for extra values)
                    let mut names = vec![AnnName {
                        name: Name::ProcVar(*lhs),
                        span: lhs_span, // Use derived span
                    }];

                    // Add wildcards for remaining values
                    // RHOLANG-RS LIMITATION: Var::Wildcard has no position data in rholang-rs
                    // Our wildcard_span_with_context approach is actually optimal given this constraint
                    for _ in 1..rhs.len() {
                        let wildcard_span = SpanContext::wildcard_span_with_context(lhs_span);
                        names.push(AnnName {
                            name: Name::ProcVar(Var::Wildcard),
                            span: wildcard_span,
                        });
                    }

                    let bind = Bind::Linear {
                        lhs: Names {
                            names: smallvec::SmallVec::from_vec(names),
                            remainder: None,
                        },
                        rhs: Source::Simple {
                            name: AnnName {
                                name: Name::ProcVar(Var::Id(Id {
                                    name: parser.ast_builder().alloc_str(&variable_name),
                                    pos: variable_span.start,
                                })),
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
        let for_comprehension = AnnProc {
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

            let mut result = AnnProc {
                proc: parser
                    .ast_builder()
                    .alloc_par(all_processes[0], all_processes[1]),
                span: initial_par_span,
            };

            // Add remaining processes, expanding span to cover all
            for proc in all_processes.iter().skip(2) {
                let expanded_span = SpanContext::merge_two_spans(result.span, proc.span);
                result = AnnProc {
                    proc: parser.ast_builder().alloc_par(result, *proc),
                    span: expanded_span,
                };
            }
            result
        };

        // Create new declaration with all variable names
        let name_decls: Vec<NameDecl> = variable_names
            .into_iter()
            .enumerate()
            .map(|(idx, name)| {
                let decl_span = SpanContext::variable_span_from_binding(let_span, idx);
                NameDecl {
                    id: Id {
                        name: parser.ast_builder().alloc_str(&name),
                        pos: decl_span.start,
                    },
                    uri: None,
                }
            })
            .collect();

        // The new process spans the entire let construct
        let new_proc = AnnProc {
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
            LetBinding::Single { lhs, rhs } => {
                // RHOLANG-RS STRENGTH: Single bindings have rich AnnName<'ast> with full span info
                // lhs.name provides precise Name enum (ProcVar or Quote) and lhs.span gives full range
                let lhs_span = lhs.span;
                let rhs_span = rhs.span;
                let pattern_span = SpanContext::synthetic_construct_span(lhs_span, 5); // Offset for pattern

                // Create match case
                let match_case = rholang_parser::ast::Case {
                    pattern: AnnProc {
                        proc: parser.ast_builder().alloc_list(&[AnnProc {
                            proc: parser.ast_builder().alloc_eval(*lhs),
                            span: lhs_span, // Use actual lhs span
                        }]),
                        span: pattern_span, // Use synthetic pattern span
                    },
                    proc: if bindings.len() > 1 {
                        // More bindings - create nested let
                        let remaining_bindings: smallvec::SmallVec<[LetBinding<'ast>; 1]> =
                            smallvec::SmallVec::from_vec(bindings[1..].to_vec());
                        let nested_span = SpanContext::merge_two_spans(body.span, let_span);
                        AnnProc {
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
                let match_expr = AnnProc {
                    proc: parser.ast_builder().alloc_list(&[*rhs]),
                    span: match_expr_span, // Use actual rhs span
                };

                // Create match process spanning from rhs to body
                let match_span = SpanContext::merge_two_spans(rhs_span, body.span);
                let match_proc = AnnProc {
                    proc: parser
                        .ast_builder()
                        .alloc_match(match_expr, &[match_case.pattern, match_case.proc]),
                    span: match_span, // Use derived match span
                };

                normalize_ann_proc(&match_proc, input, env, parser)
            }

            LetBinding::Multiple { lhs, rhs } => {
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
                let mut pattern_elements = vec![AnnProc {
                    proc: parser.ast_builder().alloc_eval(AnnName {
                        name: Name::ProcVar(*lhs),
                        span: lhs_name_span,
                    }),
                    span: lhs_name_span,
                }];

                // Add wildcards for remaining values
                for _ in 1..rhs.len() {
                    let wildcard_span = SpanContext::wildcard_span_with_context(lhs_span);
                    pattern_elements.push(AnnProc {
                        proc: parser.ast_builder().const_wild(),
                        span: wildcard_span,
                    });
                }

                let pattern_list_span = SpanContext::synthetic_construct_span(lhs_span, 10);
                let match_case = rholang_parser::ast::Case {
                    pattern: AnnProc {
                        proc: parser.ast_builder().alloc_list(&pattern_elements),
                        span: pattern_list_span,
                    },
                    proc: if bindings.len() > 1 {
                        // More bindings - create nested let
                        let remaining_bindings: smallvec::SmallVec<[LetBinding<'ast>; 1]> =
                            smallvec::SmallVec::from_vec(bindings[1..].to_vec());
                        let nested_span = SpanContext::merge_two_spans(body.span, let_span);
                        AnnProc {
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
                let match_expr = AnnProc {
                    proc: parser.ast_builder().alloc_list(rhs),
                    span: rhs_list_span, // Use span covering all rhs expressions
                };

                // Create match process
                let match_span = SpanContext::merge_two_spans(rhs_list_span, body.span);
                let match_proc = AnnProc {
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

//rholang/src/test/scala/coop/rchain/rholang/interpreter/LetSpec.scala
#[cfg(test)]
mod tests {
    use crate::rust::interpreter::test_utils::utils::proc_visit_inputs_and_env_span;
    use rholang_parser::ast::Proc;
    use rholang_parser::SourcePos;

    #[test]
    fn test_translate_single_declaration_into_match_process() {
        use super::*;

        let (inputs, env) = proc_visit_inputs_and_env_span();

        // Create: let x <- 42 in { @x!("result") }
        let bindings = smallvec::SmallVec::from_vec(vec![LetBinding::Single {
            lhs: AnnName {
                name: Name::ProcVar(Var::Id(Id {
                    name: "x",
                    pos: SourcePos { line: 0, col: 0 },
                })),
                span: SourceSpan {
                    start: SourcePos { line: 0, col: 0 },
                    end: SourcePos { line: 0, col: 0 },
                },
            },
            rhs: AnnProc {
                proc: Box::leak(Box::new(Proc::LongLiteral(42))),
                span: SourceSpan {
                    start: SourcePos { line: 0, col: 0 },
                    end: SourcePos { line: 0, col: 0 },
                },
            },
        }]);

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
                    proc: Box::leak(Box::new(Proc::StringLiteral("result"))),
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

        let concurrent = false;
        let let_span = SourceSpan {
            start: SourcePos { line: 0, col: 0 },
            end: SourcePos { line: 0, col: 0 },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_p_let(
            &bindings,
            &body,
            concurrent,
            let_span,
            inputs.clone(),
            &env,
            &parser,
        );
        assert!(result.is_ok());

        // Should transform into a match process
        let normalized = result.unwrap();
        assert!(normalized.par.matches.len() > 0);
    }

    #[test]
    fn test_translate_concurrent_declarations_into_comm() {
        use super::*;

        let (inputs, env) = proc_visit_inputs_and_env_span();

        // Create: let x <- 1, y <- 2 in { @x!(@y) } (concurrent)
        let bindings = smallvec::SmallVec::from_vec(vec![
            LetBinding::Single {
                lhs: AnnName {
                    name: Name::ProcVar(Var::Id(Id {
                        name: "x",
                        pos: SourcePos { line: 0, col: 0 },
                    })),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                rhs: AnnProc {
                    proc: Box::leak(Box::new(Proc::LongLiteral(1))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
            },
            LetBinding::Single {
                lhs: AnnName {
                    name: Name::ProcVar(Var::Id(Id {
                        name: "y",
                        pos: SourcePos { line: 0, col: 0 },
                    })),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                rhs: AnnProc {
                    proc: Box::leak(Box::new(Proc::LongLiteral(2))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
            },
        ]);

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

        let concurrent = true;
        let let_span = SourceSpan {
            start: SourcePos { line: 0, col: 0 },
            end: SourcePos { line: 0, col: 0 },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_p_let(
            &bindings,
            &body,
            concurrent,
            let_span,
            inputs.clone(),
            &env,
            &parser,
        );
        assert!(result.is_ok());

        // Should transform into a new process with sends and receives
        let normalized = result.unwrap();
        assert!(normalized.par.news.len() > 0); // Should have new declarations
    }

    #[test]
    fn test_handle_multiple_variable_declaration() {
        use super::*;

        let (inputs, env) = proc_visit_inputs_and_env_span();

        // Create: let x <- (1, 2, 3) in { @x!("got first") }
        let bindings = smallvec::SmallVec::from_vec(vec![LetBinding::Multiple {
            lhs: Var::Id(Id {
                name: "x",
                pos: SourcePos { line: 0, col: 0 },
            }),
            rhs: vec![
                AnnProc {
                    proc: Box::leak(Box::new(Proc::LongLiteral(1))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                AnnProc {
                    proc: Box::leak(Box::new(Proc::LongLiteral(2))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                AnnProc {
                    proc: Box::leak(Box::new(Proc::LongLiteral(3))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
            ],
        }]);

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
                    proc: Box::leak(Box::new(Proc::StringLiteral("got first"))),
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

        let concurrent = false;
        let let_span = SourceSpan {
            start: SourcePos { line: 0, col: 0 },
            end: SourcePos { line: 0, col: 0 },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_p_let(
            &bindings,
            &body,
            concurrent,
            let_span,
            inputs.clone(),
            &env,
            &parser,
        );
        assert!(result.is_ok());

        // Should transform into a match process with list pattern
        let normalized = result.unwrap();
        assert!(normalized.par.matches.len() > 0);
    }

    #[test]
    fn test_handle_empty_bindings() {
        use super::*;

        let (inputs, env) = proc_visit_inputs_and_env_span();

        // Create: let in { @"stdout"!("hello") }
        let bindings = smallvec::SmallVec::new();

        let body = AnnProc {
            proc: Box::leak(Box::new(Proc::Send {
                channel: AnnName {
                    name: Name::Quote(Box::leak(Box::new(Proc::StringLiteral("stdout")))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                send_type: SendType::Single,
                inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                    proc: Box::leak(Box::new(Proc::StringLiteral("hello"))),
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

        let concurrent = false;
        let let_span = SourceSpan {
            start: SourcePos { line: 0, col: 0 },
            end: SourcePos { line: 0, col: 0 },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_p_let(
            &bindings,
            &body,
            concurrent,
            let_span,
            inputs.clone(),
            &env,
            &parser,
        );
        assert!(result.is_ok());

        // Should just normalize the body directly
        let normalized = result.unwrap();
        assert!(normalized.par.sends.len() > 0);
    }

    #[test]
    fn test_translate_sequential_declarations_into_nested_matches() {
        use super::*;

        let (inputs, env) = proc_visit_inputs_and_env_span();

        // Create: let x <- 1 in { let y <- 2 in { @x!(@y) } }
        let inner_let = AnnProc {
            proc: Box::leak(Box::new(Proc::Let {
                bindings: smallvec::SmallVec::from_vec(vec![LetBinding::Single {
                    lhs: AnnName {
                        name: Name::ProcVar(Var::Id(Id {
                            name: "y",
                            pos: SourcePos { line: 0, col: 0 },
                        })),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    rhs: AnnProc {
                        proc: Box::leak(Box::new(Proc::LongLiteral(2))),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                }]),
                body: AnnProc {
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
                },
                concurrent: false,
            })),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let bindings = smallvec::SmallVec::from_vec(vec![LetBinding::Single {
            lhs: AnnName {
                name: Name::ProcVar(Var::Id(Id {
                    name: "x",
                    pos: SourcePos { line: 0, col: 0 },
                })),
                span: SourceSpan {
                    start: SourcePos { line: 0, col: 0 },
                    end: SourcePos { line: 0, col: 0 },
                },
            },
            rhs: AnnProc {
                proc: Box::leak(Box::new(Proc::LongLiteral(1))),
                span: SourceSpan {
                    start: SourcePos { line: 0, col: 0 },
                    end: SourcePos { line: 0, col: 0 },
                },
            },
        }]);

        let body = inner_let;
        let concurrent = false;
        let let_span = SourceSpan {
            start: SourcePos { line: 0, col: 0 },
            end: SourcePos { line: 0, col: 0 },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_p_let(
            &bindings,
            &body,
            concurrent,
            let_span,
            inputs.clone(),
            &env,
            &parser,
        );
        assert!(result.is_ok());

        // Should transform into nested match processes
        let normalized = result.unwrap();
        assert!(normalized.par.matches.len() > 0);
    }
}
