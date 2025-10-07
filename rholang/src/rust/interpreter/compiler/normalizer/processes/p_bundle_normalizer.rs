use super::exports::*;
use crate::rust::interpreter::compiler::exports::{ProcVisitInputs, ProcVisitOutputs};
use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
use crate::rust::interpreter::util::prepend_bundle;
use models::rhoapi::{Bundle, Par};
use models::rust::bundle_ops::BundleOps;
use std::collections::HashMap;
use std::result::Result;

use rholang_parser::ast::{AnnProc, BundleType};
use rholang_parser::{RholangParser, SourceSpan};

pub fn normalize_p_bundle<'ast>(
    bundle_type: &BundleType,
    proc: &'ast AnnProc<'ast>,
    input: ProcVisitInputs,
    span: &SourceSpan,
    env: &HashMap<String, Par>,
    parser: &'ast RholangParser<'ast>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    fn error(target_result: ProcVisitOutputs) -> Result<ProcVisitOutputs, InterpreterError> {
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
        ProcVisitInputs {
            par: Par::default(),
            ..input.clone()
        },
        env,
        parser,
    )?;

    let outermost_bundle = match bundle_type {
        BundleType::BundleReadWrite => Bundle {
            body: Some(target_result.par.clone()),
            write_flag: true,
            read_flag: true,
        },
        BundleType::BundleRead => Bundle {
            body: Some(target_result.par.clone()),
            write_flag: false,
            read_flag: true,
        },
        BundleType::BundleWrite => Bundle {
            body: Some(target_result.par.clone()),
            write_flag: true,
            read_flag: false,
        },
        BundleType::BundleEquiv => Bundle {
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

#[cfg(test)]
mod tests {
    use crate::rust::interpreter::compiler::exports::ProcVisitInputs;
    use crate::rust::interpreter::compiler::normalize::VarSort;
    use crate::rust::interpreter::errors::InterpreterError;
    use crate::rust::interpreter::test_utils::utils::{
        proc_visit_inputs_and_env, proc_visit_inputs_with_updated_bound_map_chain,
    };
    use models::create_bit_vector;
    use models::rhoapi::{Bundle, Par};
    use models::rust::utils::new_boundvar_par;
    use pretty_assertions::assert_eq;

    #[test]
    fn p_bundle_should_normalize_terms_inside() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{AnnProc, BundleType, Id, Proc, Var};
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env();
        let bound_inputs =
            proc_visit_inputs_with_updated_bound_map_chain(inputs.clone(), "x", VarSort::ProcSort);

        let bundle_proc = AnnProc {
            proc: Box::leak(Box::new(Proc::Bundle {
                bundle_type: BundleType::BundleReadWrite,
                proc: AnnProc {
                    proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
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

    /** Example:
     * bundle { _ | x }
     */
    #[test]
    fn p_bundle_should_throw_an_error_when_wildcard_or_free_variable_is_found_inside_body_of_bundle(
    ) {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{AnnProc, BundleType, Id, Proc, Var};
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env();

        let bundle_proc = AnnProc {
            proc: Box::leak(Box::new(Proc::Bundle {
                bundle_type: BundleType::BundleReadWrite,
                proc: AnnProc {
                    proc: Box::leak(Box::new(Proc::Par {
                        left: AnnProc {
                            proc: Box::leak(Box::new(Proc::ProcVar(Var::Wildcard))),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                        right: AnnProc {
                            proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
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

    /** Example:
     * bundle { Uri }
     */
    #[test]
    fn p_bundle_should_throw_an_error_when_connective_is_used_at_top_level_of_body_of_bundle() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{AnnProc, BundleType, Proc, SimpleType};
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env();

        let bundle_proc = AnnProc {
            proc: Box::leak(Box::new(Proc::Bundle {
                bundle_type: BundleType::BundleReadWrite,
                proc: AnnProc {
                    proc: Box::leak(Box::new(Proc::SimpleType(SimpleType::Uri))),
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

    /** Example:
     * bundle { @Nil!(Uri) }
     */
    #[test]
    fn p_bundle_should_not_throw_an_error_when_connective_is_used_outside_of_top_level_of_body_of_bundle(
    ) {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{AnnProc, BundleType, Name, Proc, SendType, SimpleType};
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env();

        let bundle_proc = AnnProc {
            proc: Box::leak(Box::new(Proc::Bundle {
                bundle_type: BundleType::BundleReadWrite,
                proc: AnnProc {
                    proc: Box::leak(Box::new(Proc::Send {
                        channel: Name::Quote(AnnProc {
                            proc: Box::leak(Box::new(Proc::Nil)),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        }),
                        send_type: SendType::Single,
                        inputs: smallvec::SmallVec::from_vec(vec![AnnProc {
                            proc: Box::leak(Box::new(Proc::SimpleType(SimpleType::Uri))),
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
    fn p_bundle_should_interpret_bundle_polarization() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{AnnProc, BundleType, Id, Proc, Var};
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env();
        let bound_inputs =
            proc_visit_inputs_with_updated_bound_map_chain(inputs.clone(), "x", VarSort::ProcSort);

        fn new_bundle<'ast>(bundle_type: BundleType) -> AnnProc<'ast> {
            AnnProc {
                proc: Box::leak(Box::new(Proc::Bundle {
                    bundle_type,
                    proc: AnnProc {
                        proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
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

        let test = |bundle_type: BundleType, write_flag: bool, read_flag: bool| {
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

        test(BundleType::BundleReadWrite, true, true);
        test(BundleType::BundleRead, false, true);
        test(BundleType::BundleWrite, true, false);
        test(BundleType::BundleEquiv, false, false);
    }

    #[test]
    fn p_bundle_should_collapse_nested_bundles_merging_their_polarizations() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{AnnProc, BundleType, Id, Proc, Var};
        use rholang_parser::{SourcePos, SourceSpan};

        let (inputs, env) = proc_visit_inputs_and_env();
        let bound_inputs =
            proc_visit_inputs_with_updated_bound_map_chain(inputs.clone(), "x", VarSort::ProcSort);

        let bundle_proc = AnnProc {
            proc: Box::leak(Box::new(Proc::Bundle {
                bundle_type: BundleType::BundleReadWrite,
                proc: AnnProc {
                    proc: Box::leak(Box::new(Proc::Bundle {
                        bundle_type: BundleType::BundleRead,
                        proc: AnnProc {
                            proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
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
