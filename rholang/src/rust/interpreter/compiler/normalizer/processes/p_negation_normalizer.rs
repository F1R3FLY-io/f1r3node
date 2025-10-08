use crate::rust::interpreter::compiler::exports::{FreeMap, ProcVisitInputs, ProcVisitOutputs};
use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::util::prepend_connective;
use models::rhoapi::{connective, Connective, Par};
use std::collections::HashMap;

use rholang_parser::ast::{AnnProc, Proc};

pub fn normalize_p_negation<'ast>(
    arg: &'ast Proc<'ast>,
    unary_expr_span: rholang_parser::SourceSpan,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    // Use the actual span of the entire UnaryExp (~<expr>) for accurate source location
    let ann_proc = AnnProc {
        proc: arg,
        span: unary_expr_span,
    };

    let body_result = normalize_ann_proc(
        &ann_proc,
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: FreeMap::default(),
        },
        env,
        parser,
    )?;

    // Create Connective with ConnNotBody
    let connective = Connective {
        connective_instance: Some(connective::ConnectiveInstance::ConnNotBody(
            body_result.par.clone(),
        )),
    };

    let updated_par = prepend_connective(
        input.par,
        connective.clone(),
        input.bound_map_chain.clone().depth() as i32,
    );

    Ok(ProcVisitOutputs {
        par: updated_par,
        free_map: input.free_map.add_connective(
            connective.connective_instance.unwrap(),
            unary_expr_span, // Use the actual span of the entire negation operation
        ),
    })
}

//rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use crate::rust::interpreter::test_utils::utils::proc_visit_inputs_and_env;
    use models::rhoapi::connective::ConnectiveInstance;
    use models::rhoapi::Connective;
    use models::rust::utils::new_freevar_par;
    use pretty_assertions::assert_eq;

    #[test]
    fn p_negation_should_delegate_but_not_count_any_free_variables_inside() {
        use super::normalize_p_negation;
        use rholang_parser::ast::{Id, Proc, Var};
        use rholang_parser::SourcePos;
        use rholang_parser::SourceSpan;

        let (inputs, env) = proc_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();
        let var_proc = Proc::ProcVar(Var::Id(Id {
            name: "x",
            pos: SourcePos { line: 1, col: 1 },
        }));

        let test_span = SourceSpan {
            start: SourcePos { line: 1, col: 1 },
            end: SourcePos { line: 1, col: 2 },
        };
        let result = normalize_p_negation(&var_proc, test_span, inputs.clone(), &env, &parser);
        let expected_result = inputs
            .par
            .with_connectives(vec![Connective {
                connective_instance: Some(ConnectiveInstance::ConnNotBody(new_freevar_par(
                    0,
                    Vec::new(),
                ))),
            }])
            .with_connective_used(true);

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(
            result.clone().unwrap().free_map.level_bindings,
            inputs.free_map.level_bindings
        );
        assert_eq!(
            result.unwrap().free_map.next_level,
            inputs.free_map.next_level
        )
    }
}
