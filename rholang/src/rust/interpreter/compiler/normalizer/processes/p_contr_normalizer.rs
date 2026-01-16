use crate::rust::interpreter::compiler::exports::{
    FreeMap, NameVisitInputs, ProcVisitInputs, ProcVisitOutputs,
};
use crate::rust::interpreter::compiler::normalize::{normalize_ann_proc, VarSort};
use crate::rust::interpreter::compiler::normalizer::name_normalize_matcher::normalize_name;
use crate::rust::interpreter::compiler::normalizer::processes::utils::fail_on_invalid_connective;
use crate::rust::interpreter::compiler::normalizer::remainder_normalizer_matcher::normalize_match_name;
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::matcher::has_locally_free::HasLocallyFree;
use crate::rust::interpreter::util::filter_and_adjust_bitset;
use models::rhoapi::{Par, Receive, ReceiveBind};
use models::rust::utils::union;
use std::collections::HashMap;

use rholang_parser::ast::{AnnProc, Name};

pub fn normalize_p_contr<'ast>(
    name: &'ast Name<'ast>,
    formals: &rholang_parser::ast::Names<'ast>,
    body: &'ast AnnProc<'ast>,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let name_match_result = normalize_name(
        name,
        NameVisitInputs {
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: input.free_map.clone(),
        },
        env,
        parser,
    )?;

    let mut init_acc = (vec![], FreeMap::<VarSort>::default(), Vec::new());

    for name in formals.names.iter() {
        let res = normalize_name(
            name,
            NameVisitInputs {
                bound_map_chain: input.clone().bound_map_chain.push(),
                free_map: init_acc.1.clone(),
            },
            env,
            parser,
        )?;

        let result = fail_on_invalid_connective(&input, &res)?;

        // Accumulate the result
        init_acc.0.insert(0, result.par.clone());
        init_acc.1 = result.free_map.clone();
        init_acc.2 = union(
            init_acc.clone().2,
            result.par.locally_free(
                result.par.clone(),
                (input.bound_map_chain.depth() + 1) as i32,
            ),
        );
    }

    let remainder_result = normalize_match_name(&formals.remainder, init_acc.1.clone())?;

    let new_enw = input.bound_map_chain.absorb_free_span(&remainder_result.1);
    let bound_count = remainder_result.1.count_no_wildcards();

    let body_result = normalize_ann_proc(
        body,
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: new_enw,
            free_map: name_match_result.free_map.clone(),
        },
        env,
        parser,
    )?;

    let receive = Receive {
        binds: vec![ReceiveBind {
            patterns: init_acc.0.clone().into_iter().rev().collect(),
            source: Some(name_match_result.par.clone()),
            remainder: remainder_result.0.clone(),
            free_count: bound_count as i32,
            pattern_modifiers: vec![],
        }],
        body: Some(body_result.par.clone()),
        persistent: true,
        peek: false,
        bind_count: bound_count as i32,
        locally_free: union(
            name_match_result.par.locally_free(
                name_match_result.par.clone(),
                input.bound_map_chain.depth() as i32,
            ),
            union(
                init_acc.2,
                filter_and_adjust_bitset(body_result.par.clone().locally_free, bound_count),
            ),
        ),
        connective_used: name_match_result
            .par
            .connective_used(name_match_result.par.clone())
            || body_result.par.connective_used(body_result.par.clone()),
    };
    //TODO: I should create new Expr for prepend_expr and provide it instead of receive.clone().into
    let updated_par = input.clone().par.prepend_receive(receive);
    Ok(ProcVisitOutputs {
        par: updated_par,
        free_map: body_result.free_map,
    })
}

// See rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use models::{
        create_bit_vector,
        rhoapi::{expr::ExprInstance, EPlus, Expr, Par, Receive, ReceiveBind},
        rust::utils::{new_boundvar_par, new_freevar_par, new_gint_par, new_send_par},
    };

    use crate::rust::interpreter::{
        compiler::normalize::VarSort, errors::InterpreterError,
        test_utils::utils::proc_visit_inputs_and_env,
    };

    #[test]
    fn p_contr_should_handle_a_basic_contract() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;
        use rholang_parser::ast::SendType;
        use rholang_parser::SourcePos;

        /*  new add in {
             contract add(ret, @x, @y) = {
               ret!(x + y)
             }
           }
           // new is simulated by bindings.
        */

        let (mut inputs, env) = proc_visit_inputs_and_env();
        inputs.bound_map_chain = inputs.bound_map_chain.put_pos((
            "add".to_string(),
            VarSort::NameSort,
            SourcePos { line: 0, col: 0 },
        ));

        let parser = rholang_parser::RholangParser::new();

        // Build formals: ret, @x, @y
        let ret_name = ParBuilderUtil::create_ast_name_from_var("ret");
        let x_proc = ParBuilderUtil::create_ast_proc_var("x", &parser);
        let x_quote = ParBuilderUtil::create_ast_quote_name(x_proc);
        let y_proc = ParBuilderUtil::create_ast_proc_var("y", &parser);
        let y_quote = ParBuilderUtil::create_ast_quote_name(y_proc);
        let formals = ParBuilderUtil::create_ast_names(vec![ret_name, x_quote, y_quote], None);

        // Build body: ret!(x + y)
        let x_var = ParBuilderUtil::create_ast_proc_var("x", &parser);
        let y_var = ParBuilderUtil::create_ast_proc_var("y", &parser);
        let add_expr = ParBuilderUtil::create_ast_add(x_var, y_var, &parser);
        let ret_channel = ParBuilderUtil::create_ast_name_from_var("ret");
        let body =
            ParBuilderUtil::create_ast_send(ret_channel, SendType::Single, vec![add_expr], &parser);

        // Build contract
        let add_name = ParBuilderUtil::create_ast_name_from_var("add");
        let p_contract = ParBuilderUtil::create_ast_contract(add_name, formals, body, &parser);

        let result = normalize_ann_proc(&p_contract, inputs.clone(), &env, &parser);
        assert!(result.is_ok());

        let expected_result = inputs.par.prepend_receive(Receive {
            binds: vec![ReceiveBind {
                patterns: vec![
                    new_freevar_par(0, Vec::new()),
                    new_freevar_par(1, Vec::new()),
                    new_freevar_par(2, Vec::new()),
                ],
                source: Some(new_boundvar_par(0, create_bit_vector(&vec![0]), false)),
                remainder: None,
                free_count: 3,
                pattern_modifiers: vec![],
            }],
            body: Some(new_send_par(
                new_boundvar_par(2, create_bit_vector(&vec![2]), false),
                vec![{
                    let mut par = Par::default().with_exprs(vec![Expr {
                        expr_instance: Some(ExprInstance::EPlusBody(EPlus {
                            p1: Some(new_boundvar_par(1, create_bit_vector(&vec![1]), false)),
                            p2: Some(new_boundvar_par(0, create_bit_vector(&vec![0]), false)),
                        })),
                    }]);
                    par.locally_free = create_bit_vector(&vec![0, 1]);
                    par
                }],
                false,
                create_bit_vector(&vec![0, 1, 2]),
                false,
                create_bit_vector(&vec![0, 1, 2]),
                false,
            )),
            persistent: true,
            peek: false,
            bind_count: 3,
            locally_free: create_bit_vector(&vec![0]),
            connective_used: false,
        });

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }

    #[test]
    fn p_contr_should_not_count_ground_values_in_the_formals_towards_the_bind_count() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;
        use rholang_parser::ast::SendType;
        use rholang_parser::SourcePos;

        /*  new ret5 in {
             contract ret5(ret, @5) = {
               ret!(5)
             }
           }
           // new is simulated by bindings.
        */

        let (mut inputs, env) = proc_visit_inputs_and_env();
        inputs.bound_map_chain = inputs.bound_map_chain.put_pos((
            "ret5".to_string(),
            VarSort::NameSort,
            SourcePos { line: 0, col: 0 },
        ));

        let parser = rholang_parser::RholangParser::new();

        // Build formals: ret, @5
        let ret_name = ParBuilderUtil::create_ast_name_from_var("ret");
        let five_proc = ParBuilderUtil::create_ast_int(5, &parser);
        let five_quote = ParBuilderUtil::create_ast_quote_name(five_proc);
        let formals = ParBuilderUtil::create_ast_names(vec![ret_name, five_quote], None);

        // Build body: ret!(5)
        let five_literal = ParBuilderUtil::create_ast_int(5, &parser);
        let ret_channel = ParBuilderUtil::create_ast_name_from_var("ret");
        let body = ParBuilderUtil::create_ast_send(
            ret_channel,
            SendType::Single,
            vec![five_literal],
            &parser,
        );

        // Build contract
        let ret5_name = ParBuilderUtil::create_ast_name_from_var("ret5");
        let p_contract = ParBuilderUtil::create_ast_contract(ret5_name, formals, body, &parser);

        let result = normalize_ann_proc(&p_contract, inputs.clone(), &env, &parser);
        assert!(result.is_ok());

        let expected_result = inputs.par.prepend_receive(Receive {
            binds: vec![ReceiveBind {
                patterns: vec![
                    new_freevar_par(0, Vec::new()),
                    new_gint_par(5, Vec::new(), false),
                ],
                source: Some(new_boundvar_par(0, create_bit_vector(&vec![0]), false)),
                remainder: None,
                free_count: 1,
                pattern_modifiers: vec![],
            }],
            body: Some(new_send_par(
                new_boundvar_par(0, create_bit_vector(&vec![0]), false),
                vec![new_gint_par(5, Vec::new(), false)],
                false,
                create_bit_vector(&vec![0]),
                false,
                create_bit_vector(&vec![0]),
                false,
            )),
            persistent: true,
            peek: false,
            bind_count: 1,
            locally_free: create_bit_vector(&vec![0]),
            connective_used: false,
        });

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }

    #[test]
    fn p_contr_should_not_compile_when_logical_or_or_not_is_used_in_the_pattern_of_the_receive() {
        use crate::rust::interpreter::compiler::compiler::Compiler;

        // Test disjunction in contract pattern
        let result1 =
            Compiler::source_to_adt(r#"new x in { contract x(@{ y /\ {Nil \/ Nil}}) = { Nil } }"#);
        assert!(result1.is_err());
        match result1 {
            Err(InterpreterError::PatternReceiveError(msg)) => {
                assert!(msg.contains("\\/ (disjunction)"));
            }
            other => panic!("Expected PatternReceiveError, got: {:?}", other),
        }

        // Test negation in contract pattern
        let result2 =
            Compiler::source_to_adt(r#"new x in { contract x(@{ y /\ ~Nil}) = { Nil } }"#);
        assert!(result2.is_err());
        match result2 {
            Err(InterpreterError::PatternReceiveError(msg)) => {
                assert!(msg.contains("~ (negation)"));
            }
            other => panic!("Expected PatternReceiveError, got: {:?}", other),
        }
    }

    #[test]
    fn p_contr_should_compile_when_logical_and_is_used_in_the_pattern_of_the_receive() {
        use crate::rust::interpreter::compiler::compiler::Compiler;

        let result1 =
            Compiler::source_to_adt(r#"new x in { contract x(@{ y /\ {Nil /\ Nil}}) = { Nil } }"#);
        assert!(
            result1.is_ok(),
            "Conjunction in contract pattern should be allowed, but got error: {:?}",
            result1
        );
    }
}
