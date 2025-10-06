use crate::rust::interpreter::compiler::exports::{
    NameVisitInputsSpan, ProcVisitInputsSpan, ProcVisitOutputsSpan,
};
use crate::rust::interpreter::compiler::normalizer::name_normalize_matcher::normalize_name;
use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::Par;
use std::collections::HashMap;

use rholang_parser::ast::AnnName;

pub fn normalize_p_eval<'ast>(
    eval_name: &AnnName<'ast>,
    input: ProcVisitInputsSpan,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputsSpan, InterpreterError> {
    let name_match_result = normalize_name(
        &eval_name.name,
        NameVisitInputsSpan {
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: input.free_map.clone(),
        },
        env,
        parser,
    )?;

    let updated_par = input.par.append(name_match_result.par.clone());

    Ok(ProcVisitOutputsSpan {
        par: updated_par,
        free_map: name_match_result.free_map,
    })
}

// See rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use models::rust::utils::new_boundvar_expr;

    use crate::rust::interpreter::{
        compiler::normalize::VarSort, test_utils::utils::proc_visit_inputs_and_env_span,
        util::prepend_expr,
    };

    use super::normalize_p_eval;
    use rholang_parser::ast::{AnnName, AnnProc, Id, Name, Var};
    use rholang_parser::{SourcePos, SourceSpan};

    fn create_new_ast_ann_name_id<'ast>(name: &'ast str) -> AnnName<'ast> {
        AnnName {
            name: Name::ProcVar(Var::Id(Id {
                name,
                pos: SourcePos { line: 1, col: 1 },
            })),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        }
    }

    #[test]
    fn p_eval_should_handle_a_bound_name_variable() {
        let eval_name = create_new_ast_ann_name_id("x");
        let parser = rholang_parser::RholangParser::new();
        let (mut inputs, env) = proc_visit_inputs_and_env_span();
        inputs.bound_map_chain = inputs.bound_map_chain.put_pos((
            "x".to_string(),
            VarSort::NameSort,
            SourcePos { line: 0, col: 0 },
        ));

        let result = normalize_p_eval(&eval_name, inputs.clone(), &env, &parser);
        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            prepend_expr(inputs.par, new_boundvar_expr(0), 0)
        );
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }

    #[test]
    fn p_eval_should_collapse_a_simple_quote() {
        use rholang_parser::ast::Proc;

        let left_var = AnnProc {
            proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                name: "x",
                pos: SourcePos { line: 1, col: 1 },
            })))),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let right_var = AnnProc {
            proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                name: "x",
                pos: SourcePos { line: 1, col: 1 },
            })))),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let quoted_proc = Box::leak(Box::new(Proc::Par {
            left: left_var,
            right: right_var,
        }));

        let quote_name = AnnName {
            name: Name::Quote(quoted_proc),
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        };

        let (mut inputs, env) = proc_visit_inputs_and_env_span();
        inputs.bound_map_chain = inputs.bound_map_chain.put_pos((
            "x".to_string(),
            VarSort::ProcSort,
            SourcePos { line: 0, col: 0 },
        ));

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_p_eval(&quote_name, inputs.clone(), &env, &parser);
        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            prepend_expr(
                prepend_expr(inputs.par, new_boundvar_expr(0), 0),
                new_boundvar_expr(0),
                0
            )
        );
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }
}
