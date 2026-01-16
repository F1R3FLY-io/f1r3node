use crate::rust::interpreter::compiler::exports::BoundContext;
use crate::rust::interpreter::compiler::exports::{ProcVisitInputs, ProcVisitOutputs};
use crate::rust::interpreter::compiler::normalize::VarSort;
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::util::prepend_connective;
use models::rhoapi::connective::ConnectiveInstance;
use models::rhoapi::{Connective, VarRef};
use std::result::Result;

use rholang_parser::ast::{Id, VarRefKind};

pub fn normalize_p_var_ref(
    var_ref_kind: VarRefKind,
    var_id: &Id,
    input: ProcVisitInputs,
    var_ref_span: rholang_parser::SourceSpan,
) -> Result<ProcVisitOutputs, InterpreterError> {
    match input.bound_map_chain.find(var_id.name) {
        Some((
            BoundContext {
                index,
                typ,
                source_span,
            },
            depth,
        )) => match typ {
            VarSort::ProcSort => match var_ref_kind {
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
                    var_name: var_id.name.to_string(),
                    name_var_source_span: source_span,
                    process_source_span: var_ref_span,
                }),
            },
            VarSort::NameSort => match var_ref_kind {
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

                _ => Err(InterpreterError::UnexpectedNameContext {
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
        proc_visit_inputs_and_env, proc_visit_inputs_with_updated_bound_map_chain,
    };
    use models::create_bit_vector;
    use models::rhoapi::connective::ConnectiveInstance::VarRefBody;
    use models::rhoapi::{Connective, Match as model_match, MatchCase, Par, ReceiveBind};
    use models::rhoapi::{Receive, VarRef as model_VarRef};
    use models::rust::utils::new_gint_par;
    use pretty_assertions::assert_eq;

    use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
    use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;
    use rholang_parser::ast::{Bind, Case, Id, Names, Source, VarRefKind};
    use rholang_parser::SourcePos;

    #[test]
    fn p_var_ref_should_do_deep_lookup_in_match_case() {
        let (inputs, env) = proc_visit_inputs_and_env();
        let bound_inputs =
            proc_visit_inputs_with_updated_bound_map_chain(inputs.clone(), "x", ProcSort);
        let parser = rholang_parser::RholangParser::new();

        // Create pattern: =x (VarRef)
        let pattern = ParBuilderUtil::create_ast_var_ref(
            VarRefKind::Proc,
            Id {
                name: "x",
                pos: SourcePos { line: 0, col: 0 },
            },
            &parser,
        );

        // Create case body: Nil
        let case_body = ParBuilderUtil::create_ast_nil(&parser);

        // Create match expression: 7
        let expression = ParBuilderUtil::create_ast_long_literal(7, &parser);

        // Create match: match 7 { case =x => Nil }
        let match_proc = ParBuilderUtil::create_ast_match(
            expression,
            vec![Case {
                pattern,
                proc: case_body,
            }],
            &parser,
        );

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
        let (inputs, env) = proc_visit_inputs_and_env();
        let bound_inputs =
            proc_visit_inputs_with_updated_bound_map_chain(inputs.clone(), "x", NameSort);
        let parser = rholang_parser::RholangParser::new();

        // Create pattern: @=x (Quote of VarRef)
        let var_ref = ParBuilderUtil::create_ast_var_ref(
            VarRefKind::Name,
            Id {
                name: "x",
                pos: SourcePos { line: 0, col: 0 },
            },
            &parser,
        );
        let pattern_name = ParBuilderUtil::create_ast_quote_name(var_ref);

        // Create channel: @Nil
        let nil_proc = ParBuilderUtil::create_ast_nil(&parser);
        let channel_name = ParBuilderUtil::create_ast_quote_name(nil_proc);

        // Create bind: @=x <- @Nil
        let bind = Bind::Linear {
            lhs: Names {
                names: smallvec::SmallVec::from_vec(vec![pattern_name]),
                remainder: None,
            },
            rhs: Source::Simple { name: channel_name },
            pattern_match: None,
        };

        // Create continuation body: Nil
        let cont_body = ParBuilderUtil::create_ast_nil(&parser);

        // Create for comprehension: for (@=x <- @Nil) { Nil }
        let for_comprehension =
            ParBuilderUtil::create_ast_for_comprehension(vec![vec![bind]], cont_body, &parser);

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
                    pattern_modifiers: vec![],
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
