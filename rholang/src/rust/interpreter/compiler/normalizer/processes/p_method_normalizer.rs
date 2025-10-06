use crate::rust::interpreter::compiler::exports::{ProcVisitInputsSpan, ProcVisitOutputsSpan};
use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::matcher::has_locally_free::HasLocallyFree;
use crate::rust::interpreter::util::prepend_expr;
use models::rhoapi::{expr, EMethod, Expr, Par};
use models::rust::utils::union;
use std::collections::HashMap;

use rholang_parser::ast::{AnnProc, Id};

pub fn normalize_p_method_new_ast<'ast>(
    receiver: &'ast AnnProc<'ast>,
    name_id: &'ast Id<'ast>,
    args: &'ast rholang_parser::ast::ProcList<'ast>,
    input: ProcVisitInputsSpan,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputsSpan, InterpreterError> {
    let target_result = normalize_ann_proc(
        receiver,
        ProcVisitInputsSpan {
            par: Par::default(),
            ..input.clone()
        },
        env,
        parser,
    )?;

    let target = target_result.par;

    let init_acc = (
        Vec::new(),
        ProcVisitInputsSpan {
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
                ProcVisitInputsSpan {
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

    Ok(ProcVisitOutputsSpan {
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
        compiler::normalize::VarSort, test_utils::utils::proc_visit_inputs_and_env_span,
        util::prepend_expr,
    };

    #[test]
    fn p_method_should_produce_proper_method_call() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{AnnProc, Id, Proc, Var};
        use rholang_parser::{SourcePos, SourceSpan};

        let methods = vec![String::from("nth"), String::from("toByteArray")];

        fn test(method_name: String) {
            let parser = rholang_parser::RholangParser::new();
            let (mut inputs, env) = proc_visit_inputs_and_env_span();
            inputs.bound_map_chain = inputs.bound_map_chain.put_pos((
                "x".to_string(),
                VarSort::ProcSort,
                SourcePos { line: 0, col: 0 },
            ));

            let method_call = AnnProc {
                proc: Box::leak(Box::new(Proc::Method {
                    receiver: AnnProc {
                        proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                            name: "x",
                            pos: SourcePos { line: 0, col: 0 },
                        })))),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    name: Id {
                        name: &method_name,
                        pos: SourcePos { line: 0, col: 0 },
                    },
                    args: smallvec::SmallVec::from_vec(vec![AnnProc {
                        proc: Box::leak(Box::new(Proc::LongLiteral(0))),
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
