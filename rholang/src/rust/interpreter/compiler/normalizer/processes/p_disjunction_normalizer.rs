use crate::rust::interpreter::compiler::exports::SourcePosition;
use crate::rust::interpreter::compiler::free_map::FreeMap;
use crate::rust::interpreter::compiler::normalize::{
    normalize_match_proc, ProcVisitInputs, ProcVisitOutputs, normalize_ann_proc,
};
use crate::rust::interpreter::compiler::rholang_ast::Disjunction;
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::util::prepend_connective;
use models::rhoapi::connective::ConnectiveInstance;
use models::rhoapi::{Connective, ConnectiveBody, Par};
use std::collections::HashMap;

// New AST imports
use rholang_parser::ast::AnnProc;

pub fn normalize_p_disjunction(
    proc: &Disjunction,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let left_result = normalize_match_proc(
        &proc.left,
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: FreeMap::default(),
            source_span: input.source_span,
        },
        env,
    )?;

    let right_result = normalize_match_proc(
        &proc.right,
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: FreeMap::default(),
            source_span: input.source_span,
        },
        env,
    )?;

    let lp = left_result.par;
    let result_connective = match lp.single_connective() {
        Some(Connective {
            connective_instance: Some(ConnectiveInstance::ConnOrBody(conn_body)),
        }) => Connective {
            connective_instance: Some(ConnectiveInstance::ConnOrBody(ConnectiveBody {
                ps: {
                    let mut ps = conn_body.ps.clone();
                    ps.push(right_result.par);
                    ps
                },
            })),
        },
        _ => Connective {
            connective_instance: Some(ConnectiveInstance::ConnOrBody(ConnectiveBody {
                ps: vec![lp, right_result.par],
            })),
        },
    };

    let result_par = prepend_connective(
        input.par.clone(),
        result_connective.clone(),
        input.bound_map_chain.depth() as i32,
    );

    let updated_free_map = input.free_map.add_connective(
        result_connective.connective_instance.unwrap(),
        SourcePosition {
            row: proc.line_num,
            column: proc.col_num,
        },
    );

    Ok(ProcVisitOutputs {
        par: result_par,
        free_map: updated_free_map,
    })
}

/// Parallel version of normalize_p_disjunction for new AST BinaryExp with Disjunction op
pub fn normalize_p_disjunction_new_ast<'ast>(
    left: &'ast AnnProc<'ast>,
    right: &'ast AnnProc<'ast>,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let left_result = normalize_ann_proc(
        left,
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: FreeMap::default(),
            source_span: input.source_span,
        },
        env,
        parser,
    )?;

    let right_result = normalize_ann_proc(
        right,
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: FreeMap::default(),
            source_span: input.source_span,
        },
        env,
        parser,
    )?;

    let lp = left_result.par;
    let result_connective = match lp.single_connective() {
        Some(Connective {
            connective_instance: Some(ConnectiveInstance::ConnOrBody(conn_body)),
        }) => Connective {
            connective_instance: Some(ConnectiveInstance::ConnOrBody(ConnectiveBody {
                ps: {
                    let mut ps = conn_body.ps.clone();
                    ps.push(right_result.par);
                    ps
                },
            })),
        },
        _ => Connective {
            connective_instance: Some(ConnectiveInstance::ConnOrBody(ConnectiveBody {
                ps: vec![lp, right_result.par],
            })),
        },
    };

    let result_par = prepend_connective(
        input.par.clone(),
        result_connective.clone(),
        input.bound_map_chain.depth() as i32,
    );

		// TODO: Review this src position
    let updated_free_map = input.free_map.add_connective(
        result_connective.connective_instance.unwrap(),
        SourcePosition {
            row: 0, // Default position for new AST
            column: 0,
        },
    );

    Ok(ProcVisitOutputs {
        par: result_par,
        free_map: updated_free_map,
    })
}

//rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use crate::rust::interpreter::compiler::normalize::normalize_match_proc;
    use crate::rust::interpreter::compiler::rholang_ast::Disjunction;
    use crate::rust::interpreter::test_utils::utils::proc_visit_inputs_and_env;
    use models::rhoapi::connective::ConnectiveInstance;
    use models::rhoapi::{Connective, ConnectiveBody};
    use models::rust::utils::new_freevar_par;
    use pretty_assertions::assert_eq;

    #[test]
    fn p_disjunction_should_delegate_but_not_count_any_free_variables_inside() {
        let (inputs, env) = proc_visit_inputs_and_env();
        let proc = Disjunction::new_disjunction_with_par_of_var("x", "x");

        let result = normalize_match_proc(&proc, inputs.clone(), &env);
        let expected_result = inputs
            .par
            .with_connectives(vec![Connective {
                connective_instance: Some(ConnectiveInstance::ConnOrBody(ConnectiveBody {
                    ps: vec![
                        new_freevar_par(0, Vec::new()),
                        new_freevar_par(0, Vec::new()),
                    ],
                })),
            }])
            .with_connective_used(true);

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(
            result.clone().unwrap().free_map.level_bindings,
            inputs.free_map.level_bindings
        );
        assert_eq!(
            result.clone().unwrap().free_map.next_level,
            inputs.free_map.next_level
        );
    }

    // ============================================================================
    // NEW AST PARALLEL TESTS - EXACT MAPPING TO ORIGINAL TESTS
    // ============================================================================

    #[test]
    fn new_ast_p_disjunction_should_delegate_but_not_count_any_free_variables_inside() {
        // Maps to original: p_disjunction_should_delegate_but_not_count_any_free_variables_inside
        use rholang_parser::ast::{AnnProc, Id, Proc as NewProc, Var as NewVar};
        use rholang_parser::{SourcePos, SourceSpan};
        use super::normalize_p_disjunction_new_ast;

        let (inputs, env) = proc_visit_inputs_and_env();
        
        // Create x || x where x is a free variable (ProcVar) - same variable both sides
        let left_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::ProcVar(NewVar::Id(Id {
                name: "x",
                pos: SourcePos { line: 0, col: 0 },
            })))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let right_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::ProcVar(NewVar::Id(Id {
                name: "x",
                pos: SourcePos { line: 0, col: 0 },
            })))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_p_disjunction_new_ast(&left_proc, &right_proc, inputs.clone(), &env, &parser);
        let expected_result = inputs
            .par
            .with_connectives(vec![Connective {
                connective_instance: Some(ConnectiveInstance::ConnOrBody(ConnectiveBody {
                    ps: vec![
                        new_freevar_par(0, Vec::new()),
                        new_freevar_par(0, Vec::new()),
                    ],
                })),
            }])
            .with_connective_used(true);

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(
            result.clone().unwrap().free_map.level_bindings,
            inputs.free_map.level_bindings
        );
        assert_eq!(
            result.clone().unwrap().free_map.next_level,
            inputs.free_map.next_level
        );
    }
}
