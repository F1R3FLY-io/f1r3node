use super::exports::*;
use crate::rust::interpreter::compiler::exports::{ProcVisitInputsSpan, ProcVisitOutputsSpan};
use crate::rust::interpreter::compiler::normalize::{normalize_ann_proc, normalize_match_proc};
use crate::rust::interpreter::compiler::rholang_ast::{Block, BundleType};
use crate::rust::interpreter::util::prepend_bundle;
use models::rhoapi::{Bundle, Par};
use models::rust::bundle_ops::BundleOps;
use rholang_parser::ast::AnnProc;
use rholang_parser::SourceSpan;
use std::collections::HashMap;
use std::result::Result;

pub fn normalize_p_bundle(
    bundle_type: &BundleType,
    block: &Box<Block>,
    input: ProcVisitInputs,
    line_num: usize,
    column_num: usize,
    env: &HashMap<String, Par>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    fn error(target_result: ProcVisitOutputs) -> Result<ProcVisitOutputs, InterpreterError> {
        let err_msg = {
            let at = |variable: &str, source_position: &SourcePosition| {
                format!(
                    "{} at line {}, column {}",
                    variable, source_position.row, source_position.column
                )
            };

            let wildcards_positions: Vec<String> = target_result
                .free_map
                .wildcards
                .iter()
                .map(|pos| at("", pos))
                .collect();

            let free_vars_positions: Vec<String> = target_result
                .free_map
                .level_bindings
                .iter()
                .map(|(name, context)| at(&format!("`{}`", name), &context.source_position))
                .collect();

            let err_msg_wildcards = if !wildcards_positions.is_empty() {
                format!(" Wildcards positions: {}", wildcards_positions.join(", "))
            } else {
                String::new()
            };

            let err_msg_free_vars = if !free_vars_positions.is_empty() {
                format!(
                    " Free variables positions: {}",
                    free_vars_positions.join(", ")
                )
            } else {
                String::new()
            };

            format!(
                "Bundle's content must not have free variables or wildcards.{}{}",
                err_msg_wildcards, err_msg_free_vars
            )
        };

        Err(InterpreterError::UnexpectedBundleContent(format!(
            "Bundle's content must not have free variables or wildcards. {}",
            err_msg
        )))
    }
    let target_result = normalize_match_proc(
        &block.proc,
        ProcVisitInputs {
            par: Par::default(),
            ..input.clone()
        },
        env,
    )?;
    // println!("\ntarget_result: {:?}", target_result);

    let outermost_bundle = match bundle_type {
        BundleType::BundleReadWrite { .. } => Bundle {
            body: Some(target_result.par.clone()),
            write_flag: true,
            read_flag: true,
        },
        BundleType::BundleRead { .. } => Bundle {
            body: Some(target_result.par.clone()),
            write_flag: false,
            read_flag: true,
        },
        BundleType::BundleWrite { .. } => Bundle {
            body: Some(target_result.par.clone()),
            write_flag: true,
            read_flag: false,
        },
        BundleType::BundleEquiv { .. } => Bundle {
            body: Some(target_result.par.clone()),
            write_flag: false,
            read_flag: false,
        },
    };

    let res = if !target_result.clone().par.connectives.is_empty() {
        Err(InterpreterError::UnexpectedBundleContent(format!(
            "Illegal top-level connective in bundle at line {}, column {}.",
            line_num, column_num
        )))
    } else if !target_result.clone().free_map.wildcards.is_empty()
        || !target_result.free_map.level_bindings.is_empty()
    {
        error(target_result)
    } else {
        let new_bundle = match target_result.par.single_bundle() {
            Some(single) => BundleOps::merge(&outermost_bundle, &single),
            None => outermost_bundle,
        };

        Ok(ProcVisitOutputs {
            par: {
                let updated_bundle = prepend_bundle(input.par.clone(), new_bundle);
                updated_bundle
            },
            free_map: input.free_map.clone(),
        })
    };

    res
}

/// Parallel version of normalize_p_bundle for new AST Bundle
pub fn normalize_p_bundle_new_ast<'ast>(
    bundle_type: &rholang_parser::ast::BundleType,
    proc: &'ast AnnProc<'ast>,
    input: ProcVisitInputsSpan,
    span: &rholang_parser::SourceSpan,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputsSpan, InterpreterError> {
    fn error(
        target_result: ProcVisitOutputsSpan,
    ) -> Result<ProcVisitOutputsSpan, InterpreterError> {
        let err_msg = {
            let at = |variable: &str, source_position: &SourceSpan| {
                format!(
                    "{} at line {}, column {}",
                    variable, source_position.start.line, source_position.start.col
                )
            };

            let wildcards_positions: Vec<String> = target_result
                .free_map
                .wildcards
                .iter()
                .map(|pos| at("", pos))
                .collect();

            let free_vars_positions: Vec<String> = target_result
                .free_map
                .level_bindings
                .iter()
                .map(|(name, context)| at(&format!("`{}`", name), &context.source_span))
                .collect();

            let err_msg_wildcards = if !wildcards_positions.is_empty() {
                format!(" Wildcards positions: {}", wildcards_positions.join(", "))
            } else {
                String::new()
            };

            let err_msg_free_vars = if !free_vars_positions.is_empty() {
                format!(
                    " Free variables positions: {}",
                    free_vars_positions.join(", ")
                )
            } else {
                String::new()
            };

            format!(
                "Bundle's content must not have free variables or wildcards.{}{}",
                err_msg_wildcards, err_msg_free_vars
            )
        };

        Err(InterpreterError::UnexpectedBundleContent(format!(
            "Bundle's content must not have free variables or wildcards. {}",
            err_msg
        )))
    }

    let target_result = normalize_ann_proc(
        proc,
        ProcVisitInputsSpan {
            par: Par::default(),
            ..input.clone()
        },
        env,
        parser,
    )?;

    let outermost_bundle = match bundle_type {
        rholang_parser::ast::BundleType::BundleReadWrite => Bundle {
            body: Some(target_result.par.clone()),
            write_flag: true,
            read_flag: true,
        },
        rholang_parser::ast::BundleType::BundleRead => Bundle {
            body: Some(target_result.par.clone()),
            write_flag: false,
            read_flag: true,
        },
        rholang_parser::ast::BundleType::BundleWrite => Bundle {
            body: Some(target_result.par.clone()),
            write_flag: true,
            read_flag: false,
        },
        rholang_parser::ast::BundleType::BundleEquiv => Bundle {
            body: Some(target_result.par.clone()),
            write_flag: false,
            read_flag: false,
        },
    };

    let res = if !target_result.clone().par.connectives.is_empty() {
        Err(InterpreterError::UnexpectedBundleContent(format!(
            "Illegal top-level connective in bundle at line {}, column {}.",
            span.start.line, span.start.col
        )))
    } else if !target_result.clone().free_map.wildcards.is_empty()
        || !target_result.free_map.level_bindings.is_empty()
    {
        error(target_result)
    } else {
        let new_bundle = match target_result.par.single_bundle() {
            Some(single) => BundleOps::merge(&outermost_bundle, &single),
            None => outermost_bundle,
        };

        Ok(ProcVisitOutputsSpan {
            par: {
                let updated_bundle = prepend_bundle(input.par.clone(), new_bundle);
                updated_bundle
            },
            free_map: input.free_map.clone(),
        })
    };

    res
}

#[cfg(test)]
mod tests {
    use crate::rust::interpreter::compiler::exports::ProcVisitInputsSpan;
    use crate::rust::interpreter::compiler::normalize::{
        normalize_match_proc, ProcVisitInputs, VarSort,
    };
    use crate::rust::interpreter::compiler::rholang_ast::{
        Block, BundleType, Name, Proc, ProcList, SendType, SimpleType,
    };
    use crate::rust::interpreter::errors::InterpreterError;
    use crate::rust::interpreter::test_utils::utils::{
        proc_visit_inputs_and_env, proc_visit_inputs_with_updated_bound_map_chain_span,
    };
    use crate::rust::interpreter::test_utils::utils::{
        proc_visit_inputs_and_env_span, proc_visit_inputs_with_updated_bound_map_chain,
    };
    use models::create_bit_vector;
    use models::rhoapi::{Bundle, Par};
    use models::rust::utils::new_boundvar_par;
    use pretty_assertions::assert_eq;

    #[test]
    fn p_bundle_should_normalize_terms_inside() {
        let (inputs, env) = proc_visit_inputs_and_env();
        let proc = Proc::Bundle {
            bundle_type: BundleType::new_bundle_read_write(),
            proc: Box::new(Block::new(Proc::new_proc_var("x"))),
            line_num: 0,
            col_num: 0,
        };
        let bound_inputs =
            proc_visit_inputs_with_updated_bound_map_chain(inputs.clone(), "x", VarSort::ProcSort);
        // println!("\nbound_inputs: {:?}", bound_inputs);
        let result = normalize_match_proc(&proc, bound_inputs.clone(), &env);
        // println!("\nresult: {:?}", result);
        let expected_result = inputs
            .par
            .with_bundles(vec![Bundle {
                body: Some(new_boundvar_par(0, create_bit_vector(&vec![0]), false)),
                write_flag: true,
                read_flag: true,
            }])
            .with_locally_free(create_bit_vector(&vec![0]));

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.clone().unwrap().free_map, bound_inputs.free_map);
    }

    /** Example:
     * bundle { _ | x }
     */
    #[test]
    fn p_bundle_should_throw_an_error_when_wildcard_or_free_variable_is_found_inside_body_of_bundle(
    ) {
        let (inputs, env) = proc_visit_inputs_and_env();
        let proc = Proc::Bundle {
            bundle_type: BundleType::new_bundle_read_write(),
            proc: Box::new(Block::new(Proc::new_proc_par_with_wildcard_and_var("x"))),
            line_num: 0,
            col_num: 0,
        };
        let result = normalize_match_proc(&proc, inputs.clone(), &env);
        assert!(matches!(
            result,
            Err(InterpreterError::UnexpectedBundleContent { .. })
        ));
    }

    /** Example:
     * bundle { Uri }
     */
    #[test]
    fn p_bundle_should_throw_an_error_when_connective_is_used_at_top_level_of_body_of_bundle() {
        let (inputs, env) = proc_visit_inputs_and_env();
        let proc = Proc::Bundle {
            bundle_type: BundleType::new_bundle_read_write(),
            proc: Box::new(Block::new(Proc::SimpleType(SimpleType::new_uri()))),
            line_num: 0,
            col_num: 0,
        };
        let result = normalize_match_proc(&proc, inputs.clone(), &env);

        assert!(matches!(
            result,
            Err(InterpreterError::UnexpectedBundleContent { .. })
        ));
    }

    /** Example:
     * bundle { @Nil!(Uri) }
     */
    #[test]
    fn p_bundle_should_not_throw_an_error_when_connective_is_used_outside_of_top_level_of_body_of_bundle(
    ) {
        let (inputs, env) = proc_visit_inputs_and_env();

        let proc = Proc::Bundle {
            bundle_type: BundleType::new_bundle_read_write(),
            proc: Box::new(Block::new(Proc::Send {
                name: Name::new_name_quote_nil(),
                send_type: SendType::new_single(),
                inputs: ProcList::new(vec![Proc::SimpleType(SimpleType::new_uri())]),
                line_num: 0,
                col_num: 0,
            })),
            line_num: 0,
            col_num: 0,
        };
        let result = normalize_match_proc(&proc, inputs.clone(), &env);

        assert!(result.is_ok());
    }

    #[test]
    fn p_bundle_should_interpret_bundle_polarization() {
        let (inputs, env) = proc_visit_inputs_and_env();

        pub fn new_bundle(proc: Proc, read_only: bool, write_only: bool) -> Proc {
            let bundle_type = match (read_only, write_only) {
                (true, true) => BundleType::new_bundle_read_write(),
                (true, false) => BundleType::new_bundle_read(),
                (false, true) => BundleType::new_bundle_write(),
                (false, false) => BundleType::new_bundle_equiv(),
            };

            Proc::Bundle {
                bundle_type,
                proc: Box::new(Block::new(proc)),
                line_num: 0,
                col_num: 0,
            }
        }

        let proc = Proc::new_proc_var("x");
        let bound_inputs =
            proc_visit_inputs_with_updated_bound_map_chain(inputs.clone(), "x", VarSort::ProcSort);

        fn expected_results(write_flag: bool, read_flag: bool, inputs: &ProcVisitInputs) -> Par {
            inputs
                .clone()
                .par
                .with_bundles(vec![Bundle {
                    body: Some(new_boundvar_par(0, create_bit_vector(&vec![0]), false)),
                    write_flag,
                    read_flag,
                }])
                .with_locally_free(create_bit_vector(&vec![0]))
        }

        let test = |read_only: bool, write_only: bool| {
            // println!(
            //     "Testing bundle with flags read_only={}, write_only={}",
            //     read_only, write_only
            // );
            let bundle_proc = new_bundle(proc.clone(), read_only, write_only);
            let result = normalize_match_proc(&bundle_proc, bound_inputs.clone(), &env);
            let expected = expected_results(write_only, read_only, &bound_inputs);

            assert_eq!(
                result.clone().unwrap().par,
                expected,
                "Resulting `Par` did not match expected"
            );
            assert_eq!(
                result.unwrap().free_map,
                inputs.free_map,
                "Resulting `FreeMap` did not match expected"
            );
        };

        test(true, true);
        test(true, false);
        test(false, true);
        test(false, false);
    }

    #[test]
    fn p_bundle_should_collapse_nested_bundles_merging_their_polarizations() {
        let (inputs, env) = proc_visit_inputs_and_env();

        let proc = Proc::Bundle {
            bundle_type: BundleType::new_bundle_read_write(),
            proc: Box::new(Block::new(Proc::Bundle {
                bundle_type: BundleType::new_bundle_read(),
                proc: Box::new(Block::new(Proc::new_proc_var("x"))),
                line_num: 0,
                col_num: 0,
            })),
            line_num: 0,
            col_num: 0,
        };

        let bound_inputs =
            proc_visit_inputs_with_updated_bound_map_chain(inputs.clone(), "x", VarSort::ProcSort);

        let expected_result = inputs
            .par
            .with_bundles(vec![Bundle {
                body: Some(new_boundvar_par(0, create_bit_vector(&vec![0]), false)),
                write_flag: false,
                read_flag: true,
            }])
            .with_locally_free(create_bit_vector(&vec![0]));

        let result = normalize_match_proc(&proc, bound_inputs.clone(), &env);

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.unwrap().free_map, bound_inputs.free_map);
    }

    #[test]
    fn new_ast_p_bundle_should_normalize_terms_inside() {
        // Maps to original: p_bundle_should_normalize_terms_inside
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{
            AnnProc, BundleType as NewBundleType, Id, Proc as NewProc, Var as NewVar,
        };
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env_span();
        let bound_inputs = proc_visit_inputs_with_updated_bound_map_chain_span(
            inputs.clone(),
            "x",
            VarSort::ProcSort,
        );

        // Create bundle+- { x } using new AST
        let bundle_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::Bundle {
                bundle_type: NewBundleType::BundleReadWrite,
                proc: AnnProc {
                    proc: Box::leak(Box::new(NewProc::ProcVar(NewVar::Id(Id {
                        name: "x",
                        pos: SourcePos { line: 0, col: 0 },
                    })))),
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
        let result = normalize_ann_proc(&bundle_proc, bound_inputs.clone(), &env, &parser);
        let expected_result = inputs
            .par
            .with_bundles(vec![Bundle {
                body: Some(new_boundvar_par(0, create_bit_vector(&vec![0]), false)),
                write_flag: true,
                read_flag: true,
            }])
            .with_locally_free(create_bit_vector(&vec![0]));

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.clone().unwrap().free_map, bound_inputs.free_map);
    }

    #[test]
    fn new_ast_p_bundle_should_throw_an_error_when_wildcard_or_free_variable_is_found_inside_body_of_bundle(
    ) {
        // Maps to original: p_bundle_should_throw_an_error_when_wildcard_or_free_variable_is_found_inside_body_of_bundle
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{
            AnnProc, BundleType as NewBundleType, Id, Proc as NewProc, Var as NewVar,
        };
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env_span();

        // Create bundle+- { _ | x } using new AST (Par with wildcard and free variable)
        let bundle_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::Bundle {
                bundle_type: NewBundleType::BundleReadWrite,
                proc: AnnProc {
                    proc: Box::leak(Box::new(NewProc::Par {
                        left: AnnProc {
                            proc: Box::leak(Box::new(NewProc::ProcVar(NewVar::Wildcard))),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                        right: AnnProc {
                            proc: Box::leak(Box::new(NewProc::ProcVar(NewVar::Id(Id {
                                name: "x",
                                pos: SourcePos { line: 0, col: 0 },
                            })))),
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
        let result = normalize_ann_proc(&bundle_proc, inputs.clone(), &env, &parser);
        assert!(matches!(
            result,
            Err(InterpreterError::UnexpectedBundleContent { .. })
        ));
    }

    #[test]
    fn new_ast_p_bundle_should_throw_an_error_when_connective_is_used_at_top_level_of_body_of_bundle(
    ) {
        // Maps to original: p_bundle_should_throw_an_error_when_connective_is_used_at_top_level_of_body_of_bundle
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{
            AnnProc, BundleType as NewBundleType, Proc as NewProc, SimpleType as NewSimpleType,
        };
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env_span();

        // Create bundle+- { Uri } using new AST
        let bundle_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::Bundle {
                bundle_type: NewBundleType::BundleReadWrite,
                proc: AnnProc {
                    proc: Box::leak(Box::new(NewProc::SimpleType(NewSimpleType::Uri))),
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
        let result = normalize_ann_proc(&bundle_proc, inputs.clone(), &env, &parser);

        assert!(matches!(
            result,
            Err(InterpreterError::UnexpectedBundleContent { .. })
        ));
    }

    #[test]
    fn new_ast_p_bundle_should_not_throw_an_error_when_connective_is_used_outside_of_top_level_of_body_of_bundle(
    ) {
        // Maps to original: p_bundle_should_not_throw_an_error_when_connective_is_used_outside_of_top_level_of_body_of_bundle
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{
            AnnName, AnnProc, BundleType as NewBundleType, Name as NewName, Proc as NewProc,
            SendType as NewSendType, SimpleType as NewSimpleType,
        };
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env_span();

        // Create bundle+- { @Nil!(Uri) } using new AST
        let bundle_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::Bundle {
                bundle_type: NewBundleType::BundleReadWrite,
                proc: AnnProc {
                    proc: Box::leak(Box::new(NewProc::Send {
                        channel: AnnName {
                            name: NewName::Quote(Box::leak(Box::new(NewProc::Nil))),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                        send_type: NewSendType::Single,
                        inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                            proc: Box::leak(Box::new(NewProc::SimpleType(NewSimpleType::Uri))),
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
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&bundle_proc, inputs.clone(), &env, &parser);

        assert!(result.is_ok());
    }

    #[test]
    fn new_ast_p_bundle_should_interpret_bundle_polarization() {
        // Maps to original: p_bundle_should_interpret_bundle_polarization
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{
            AnnProc, BundleType as NewBundleType, Id, Proc as NewProc, Var as NewVar,
        };
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env_span();
        let bound_inputs = proc_visit_inputs_with_updated_bound_map_chain_span(
            inputs.clone(),
            "x",
            VarSort::ProcSort,
        );

        fn new_bundle<'ast>(bundle_type: NewBundleType) -> AnnProc<'ast> {
            AnnProc {
                proc: Box::leak(Box::new(NewProc::Bundle {
                    bundle_type,
                    proc: AnnProc {
                        proc: Box::leak(Box::new(NewProc::ProcVar(NewVar::Id(Id {
                            name: "x",
                            pos: SourcePos { line: 0, col: 0 },
                        })))),
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
            }
        }

        fn expected_results(
            write_flag: bool,
            read_flag: bool,
            inputs: &ProcVisitInputsSpan,
        ) -> Par {
            inputs
                .clone()
                .par
                .with_bundles(vec![Bundle {
                    body: Some(new_boundvar_par(0, create_bit_vector(&vec![0]), false)),
                    write_flag,
                    read_flag,
                }])
                .with_locally_free(create_bit_vector(&vec![0]))
        }

        let test = |bundle_type: NewBundleType, write_flag: bool, read_flag: bool| {
            let parser = rholang_parser::RholangParser::new();
            let bundle_proc = new_bundle(bundle_type);
            let result = normalize_ann_proc(&bundle_proc, bound_inputs.clone(), &env, &parser);
            let expected = expected_results(write_flag, read_flag, &bound_inputs);

            assert_eq!(
                result.clone().unwrap().par,
                expected,
                "Resulting `Par` did not match expected"
            );
            assert_eq!(
                result.unwrap().free_map,
                inputs.free_map,
                "Resulting `FreeMap` did not match expected"
            );
        };

        test(NewBundleType::BundleReadWrite, true, true);
        test(NewBundleType::BundleRead, false, true);
        test(NewBundleType::BundleWrite, true, false);
        test(NewBundleType::BundleEquiv, false, false);
    }

    #[test]
    fn new_ast_p_bundle_should_collapse_nested_bundles_merging_their_polarizations() {
        // Maps to original: p_bundle_should_collapse_nested_bundles_merging_their_polarizations
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{
            AnnProc, BundleType as NewBundleType, Id, Proc as NewProc, Var as NewVar,
        };
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env_span();
        let bound_inputs = proc_visit_inputs_with_updated_bound_map_chain_span(
            inputs.clone(),
            "x",
            VarSort::ProcSort,
        );

        // Create bundle+- { bundle+ { x } } using new AST (nested bundles)
        let bundle_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::Bundle {
                bundle_type: NewBundleType::BundleReadWrite,
                proc: AnnProc {
                    proc: Box::leak(Box::new(NewProc::Bundle {
                        bundle_type: NewBundleType::BundleRead,
                        proc: AnnProc {
                            proc: Box::leak(Box::new(NewProc::ProcVar(NewVar::Id(Id {
                                name: "x",
                                pos: SourcePos { line: 0, col: 0 },
                            })))),
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

        let expected_result = inputs
            .par
            .with_bundles(vec![Bundle {
                body: Some(new_boundvar_par(0, create_bit_vector(&vec![0]), false)),
                write_flag: false, // Read-only because ReadWrite AND Read = Read
                read_flag: true,
            }])
            .with_locally_free(create_bit_vector(&vec![0]));

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&bundle_proc, bound_inputs.clone(), &env, &parser);

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.unwrap().free_map, bound_inputs.free_map);
    }
}
