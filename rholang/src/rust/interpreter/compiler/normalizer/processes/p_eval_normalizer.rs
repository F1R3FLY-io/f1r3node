use crate::rust::interpreter::compiler::exports::{
    NameVisitInputs, ProcVisitInputs, ProcVisitOutputs,
};
use crate::rust::interpreter::compiler::normalizer::name_normalize_matcher::normalize_name;
use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::Par;
use std::collections::HashMap;

use rholang_parser::ast::Name;

pub fn normalize_p_eval<'ast>(
    eval_name: &Name<'ast>,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let name_match_result = normalize_name(
        eval_name,
        NameVisitInputs {
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: input.free_map.clone(),
        },
        env,
        parser,
    )?;

    let updated_par = input.par.append(name_match_result.par.clone());

    Ok(ProcVisitOutputs {
        par: updated_par,
        free_map: name_match_result.free_map,
    })
}

// See rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use models::rust::utils::new_boundvar_expr;

    use crate::rust::interpreter::{
        compiler::normalize::VarSort, test_utils::utils::proc_visit_inputs_and_env,
        util::prepend_expr,
    };

    use super::normalize_p_eval;
    use rholang_parser::ast::{Id, Name, Var};
    use rholang_parser::SourcePos;

    fn create_name_id<'ast>(name: &'ast str) -> Name<'ast> {
        Name::NameVar(Var::Id(Id {
            name,
            pos: SourcePos { line: 1, col: 1 },
        }))
    }

    #[test]
    fn p_eval_should_handle_a_bound_name_variable() {
        let eval_name = create_name_id("x");
        let parser = rholang_parser::RholangParser::new();
        let (mut inputs, env) = proc_visit_inputs_and_env();
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
        use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;

        let (mut inputs, env) = proc_visit_inputs_and_env();
        inputs.bound_map_chain = inputs.bound_map_chain.put_pos((
            "x".to_string(),
            VarSort::ProcSort,
            SourcePos { line: 0, col: 0 },
        ));

        let parser = rholang_parser::RholangParser::new();

        let left_var = ParBuilderUtil::create_ast_proc_var("x", &parser);
        let right_var = ParBuilderUtil::create_ast_proc_var("x", &parser);
        let quoted_proc = ParBuilderUtil::create_ast_par(left_var, right_var, &parser);
        let quote_name = ParBuilderUtil::create_ast_quote_name(quoted_proc);

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
