use crate::rust::interpreter::compiler::exports::{ProcVisitInputs, ProcVisitOutputs};
use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::matcher::has_locally_free::HasLocallyFree;
use crate::rust::interpreter::util::prepend_expr;
use models::rhoapi::{expr, EMethod, Expr, Par};
use models::rust::utils::union;
use std::collections::HashMap;

use rholang_parser::ast::{AnnProc, Id};

pub fn normalize_p_method<'ast>(
    receiver: &'ast AnnProc<'ast>,
    name_id: &'ast Id<'ast>,
    args: &'ast rholang_parser::ast::ProcList<'ast>,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let target_result = normalize_ann_proc(
        receiver,
        ProcVisitInputs {
            par: Par::default(),
            ..input.clone()
        },
        env,
        parser,
    )?;

    let target = target_result.par;

    let init_acc = (
        Vec::new(),
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: target_result.free_map.clone(),
        },
        Vec::new(),
        false,
    );

    let arg_results = args.iter().rev().try_fold(init_acc, |acc, arg| {
        normalize_ann_proc(arg, acc.1.clone(), env, parser).map(|proc_match_result| {
            (
                {
                    let mut acc_0 = acc.0.clone();
                    acc_0.insert(0, proc_match_result.par.clone());
                    acc_0
                },
                ProcVisitInputs {
                    par: Par::default(),
                    bound_map_chain: input.bound_map_chain.clone(),
                    free_map: proc_match_result.free_map.clone(),
                },
                union(acc.2.clone(), proc_match_result.par.locally_free.clone()),
                acc.3 || proc_match_result.par.connective_used,
            )
        })
    })?;

    let method = EMethod {
        method_name: name_id.name.to_string(),
        target: Some(target.clone()),
        arguments: arg_results.0,
        locally_free: union(
            target.locally_free(target.clone(), input.bound_map_chain.depth() as i32),
            arg_results.2,
        ),
        connective_used: target.connective_used(target.clone()) || arg_results.3,
    };

    let updated_par = prepend_expr(
        input.par,
        Expr {
            expr_instance: Some(expr::ExprInstance::EMethodBody(method)),
        },
        input.bound_map_chain.depth() as i32,
    );

    Ok(ProcVisitOutputs {
        par: updated_par,
        free_map: arg_results.1.free_map,
    })
}

// See rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use models::{
        create_bit_vector,
        rhoapi::{expr::ExprInstance, EMethod, Expr, Par},
        rust::utils::{new_boundvar_par, new_gint_par},
    };

    use crate::rust::interpreter::{
        compiler::normalize::VarSort, test_utils::utils::proc_visit_inputs_and_env,
        util::prepend_expr,
    };

    #[test]
    fn p_method_should_produce_proper_method_call() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;
        use rholang_parser::ast::{Id, Var};
        use rholang_parser::SourcePos;

        let methods = vec![String::from("nth"), String::from("toByteArray")];

        fn test(method_name: String) {
            let parser = rholang_parser::RholangParser::new();
            let (mut inputs, env) = proc_visit_inputs_and_env();
            inputs.bound_map_chain = inputs.bound_map_chain.put_pos((
                "x".to_string(),
                VarSort::ProcSort,
                SourcePos { line: 0, col: 0 },
            ));

            // Create receiver: x (ProcVar)
            let receiver = ParBuilderUtil::create_ast_proc_var_from_var(
                Var::Id(Id {
                    name: "x",
                    pos: SourcePos { line: 0, col: 0 },
                }),
                &parser,
            );

            // Create method name
            let method_id = Id {
                name: &method_name,
                pos: SourcePos { line: 0, col: 0 },
            };

            // Create args: [0]
            let arg = ParBuilderUtil::create_ast_long_literal(0, &parser);

            // Create method call
            let method_call =
                ParBuilderUtil::create_ast_method(method_id, receiver, vec![arg], &parser);

            let result = normalize_ann_proc(&method_call, inputs.clone(), &env, &parser);
            assert!(result.is_ok());

            let expected_result = prepend_expr(
                Par::default(),
                Expr {
                    expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                        method_name,
                        target: Some(new_boundvar_par(0, create_bit_vector(&vec![0]), false)),
                        arguments: vec![new_gint_par(0, Vec::new(), false)],
                        locally_free: create_bit_vector(&vec![0]),
                        connective_used: false,
                    })),
                },
                0,
            );

            assert_eq!(result.clone().unwrap().par, expected_result);
            assert_eq!(result.unwrap().free_map, inputs.free_map);
        }

        test(methods[0].clone());
        test(methods[1].clone());
    }
}
