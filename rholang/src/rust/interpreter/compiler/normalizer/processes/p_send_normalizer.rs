use crate::rust::interpreter::compiler::exports::{
    NameVisitInputs, ProcVisitInputs, ProcVisitOutputs,
};
use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
use crate::rust::interpreter::compiler::normalizer::name_normalize_matcher::normalize_name;
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::matcher::has_locally_free::HasLocallyFree;
use models::rhoapi::{Par, Send};
use models::rust::utils::union;
use std::collections::HashMap;

use rholang_parser::ast::{Name, SendType};

pub fn normalize_p_send<'ast>(
    channel: &'ast Name<'ast>,
    send_type: &SendType,
    inputs: &'ast rholang_parser::ast::ProcList<'ast>,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let name_match_result = normalize_name(
        channel,
        NameVisitInputs {
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: input.free_map.clone(),
        },
        env,
        parser,
    )?;

    let mut acc = (
        Vec::new(),
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: name_match_result.free_map.clone(),
        },
        Vec::new(),
        false,
    );

    for proc in inputs.iter() {
        let proc_match_result = normalize_ann_proc(proc, acc.1.clone(), env, parser)?;

        acc.0.push(proc_match_result.par.clone());
        acc.1 = ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: input.bound_map_chain.clone(),
            free_map: proc_match_result.free_map.clone(),
        };
        acc.2 = union(acc.2.clone(), proc_match_result.par.locally_free.clone());
        acc.3 = acc.3 || proc_match_result.par.connective_used;
    }

    let persistent = match send_type {
        rholang_parser::ast::SendType::Single => false,
        rholang_parser::ast::SendType::Multiple => true,
    };

    let send = Send {
        chan: Some(name_match_result.par.clone()),
        data: acc.0,
        persistent,
        locally_free: union(
            name_match_result.par.clone().locally_free(
                name_match_result.par.clone(),
                input.bound_map_chain.depth() as i32,
            ),
            acc.2,
        ),
        connective_used: name_match_result
            .par
            .connective_used(name_match_result.par.clone())
            || acc.3,
    };

    let updated_par = input.par.clone().prepend_send(send);

    Ok(ProcVisitOutputs {
        par: updated_par,
        free_map: acc.1.free_map,
    })
}

// See rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use models::{
        create_bit_vector,
        rhoapi::Par,
        rust::utils::{new_boundvar_par, new_gint_par, new_send},
    };

    use crate::rust::interpreter::{
        compiler::{compiler::Compiler, exports::ProcVisitInputs, normalize::VarSort},
        errors::InterpreterError,
        test_utils::utils::proc_visit_inputs_and_env,
    };

    #[test]
    fn p_send_should_handle_a_basic_send() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;
        use rholang_parser::ast::SendType;

        let (mut inputs, env) = proc_visit_inputs_and_env();
        let parser = rholang_parser::RholangParser::new();

        // Create channel: @Nil
        let nil_proc = ParBuilderUtil::create_ast_nil(&parser);
        let channel = ParBuilderUtil::create_ast_quote_name(nil_proc);

        // Create inputs: 7, 8
        let input1 = ParBuilderUtil::create_ast_long_literal(7, &parser);
        let input2 = ParBuilderUtil::create_ast_long_literal(8, &parser);

        // Create send: @Nil!(7, 8)
        let send_proc = ParBuilderUtil::create_ast_send(
            channel,
            SendType::Single,
            vec![input1, input2],
            &parser,
        );

        let result = normalize_ann_proc(&send_proc, inputs.clone(), &env, &parser);
        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            inputs.par.prepend_send(new_send(
                Par::default(),
                vec![
                    new_gint_par(7, Vec::new(), false),
                    new_gint_par(8, Vec::new(), false)
                ],
                false,
                Vec::new(),
                false
            ))
        );
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }

    #[test]
    fn p_send_should_handle_a_name_var() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;
        use rholang_parser::ast::SendType;
        use rholang_parser::SourcePos;

        let (mut inputs, env) = proc_visit_inputs_and_env();
        inputs.bound_map_chain = inputs.bound_map_chain.put_pos((
            "x".to_string(),
            VarSort::NameSort,
            SourcePos { line: 0, col: 0 },
        ));
        let parser = rholang_parser::RholangParser::new();

        // Create channel: x (NameVar)
        let channel = ParBuilderUtil::create_ast_name_var("x");

        // Create inputs: 7, 8
        let input1 = ParBuilderUtil::create_ast_long_literal(7, &parser);
        let input2 = ParBuilderUtil::create_ast_long_literal(8, &parser);

        // Create send: x!(7, 8)
        let send_proc = ParBuilderUtil::create_ast_send(
            channel,
            SendType::Single,
            vec![input1, input2],
            &parser,
        );

        let result = normalize_ann_proc(&send_proc, inputs.clone(), &env, &parser);
        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            inputs.par.prepend_send(new_send(
                new_boundvar_par(0, create_bit_vector(&vec![0]), false),
                vec![
                    new_gint_par(7, Vec::new(), false),
                    new_gint_par(8, Vec::new(), false)
                ],
                false,
                create_bit_vector(&vec![0]),
                false
            ))
        );
        assert_eq!(result.unwrap().free_map, inputs.free_map);
    }

    #[test]
    fn p_send_should_propagate_known_free() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;
        use rholang_parser::ast::{Id, SendType, Var};
        use rholang_parser::SourcePos;

        let parser = rholang_parser::RholangParser::new();

        // Create channel: @*x (Quote of Eval of NameVar)
        let name_var = ParBuilderUtil::create_ast_name_var("x");
        let eval_proc = ParBuilderUtil::create_ast_eval(name_var, &parser);
        let channel = ParBuilderUtil::create_ast_quote_name(eval_proc);

        // Create inputs: 7, x (ProcVar)
        let input1 = ParBuilderUtil::create_ast_long_literal(7, &parser);
        let input2 = ParBuilderUtil::create_ast_proc_var_from_var(
            Var::Id(Id {
                name: "x",
                pos: SourcePos { line: 0, col: 0 },
            }),
            &parser,
        );

        // Create send: @*x!(7, x)
        let send_proc = ParBuilderUtil::create_ast_send(
            channel,
            SendType::Single,
            vec![input1, input2],
            &parser,
        );

        let result =
            normalize_ann_proc(&send_proc, ProcVisitInputs::new(), &HashMap::new(), &parser);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InterpreterError::UnexpectedReuseOfProcContextFree {
                var_name,
                first_use: _,
                second_use: _
            }) if var_name == "x"
        ));
    }

    #[test]
    fn p_send_should_not_compile_if_data_contains_negation() {
        let result = Compiler::source_to_adt(r#"new x in { x!(~1) }"#);
        assert!(result.is_err());
        match result {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("~ (negation)"));
            }
            other => panic!(
                "Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}",
                other
            ),
        }
    }

    #[test]
    fn p_send_should_not_compile_if_data_contains_conjuction() {
        let result = Compiler::source_to_adt(r#"new x in { x!(1 /\ 2) }"#);
        assert!(result.is_err());
        match result {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("/\\ (conjunction)"));
            }
            other => panic!(
                "Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}",
                other
            ),
        }
    }

    #[test]
    fn p_send_should_not_compile_if_data_contains_disjunction() {
        let result = Compiler::source_to_adt(r#"new x in { x!(1 \/ 2) }"#);
        assert!(result.is_err());
        match result {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("\\/ (disjunction)"));
            }
            other => panic!(
                "Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}",
                other
            ),
        }
    }

    #[test]
    fn p_send_should_not_compile_if_data_contains_wildcard() {
        let result = Compiler::source_to_adt(r#"@"x"!(_)"#);
        assert!(result.is_err());
        match result {
            Err(InterpreterError::TopLevelWildcardsNotAllowedError(msg)) => {
                assert!(msg.contains("_ (wildcard)"));
            }
            other => panic!(
                "Expected TopLevelWildcardsNotAllowedError, got: {:?}",
                other
            ),
        }
    }

    #[test]
    fn p_send_should_not_compile_if_data_contains_free_variable() {
        let result = Compiler::source_to_adt(r#"@"x"!(y)"#);
        assert!(result.is_err());
        match result {
            Err(InterpreterError::TopLevelFreeVariablesNotAllowedError(msg)) => {
                assert!(msg.contains("y"));
            }
            other => panic!(
                "Expected TopLevelFreeVariablesNotAllowedError, got: {:?}",
                other
            ),
        }
    }

    #[test]
    fn p_send_should_not_compile_if_name_contains_connectives() {
        // Test conjunction in channel name
        let result1 = Compiler::source_to_adt(r#"@{Nil /\ Nil}!(1)"#);
        assert!(result1.is_err());
        match result1 {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("/\\ (conjunction)"));
            }
            other => panic!(
                "Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}",
                other
            ),
        }

        // Test disjunction in channel name
        let result2 = Compiler::source_to_adt(r#"@{Nil \/ Nil}!(1)"#);
        assert!(result2.is_err());
        match result2 {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("\\/ (disjunction)"));
            }
            other => panic!(
                "Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}",
                other
            ),
        }

        // Test negation in channel name
        let result3 = Compiler::source_to_adt(r#"@{~Nil}!(1)"#);
        assert!(result3.is_err());
        match result3 {
            Err(InterpreterError::TopLevelLogicalConnectivesNotAllowedError(msg)) => {
                assert!(msg.contains("~ (negation)"));
            }
            other => panic!(
                "Expected TopLevelLogicalConnectivesNotAllowedError, got: {:?}",
                other
            ),
        }
    }
}
