use crate::rust::interpreter::compiler::exports::{ProcVisitInputsSpan, ProcVisitOutputsSpan};
use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::Par;
use std::collections::HashMap;

use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
use rholang_parser::ast::AnnProc;

pub fn normalize_p_par_new_ast<'ast>(
    left: &AnnProc<'ast>,
    right: &AnnProc<'ast>,
    input: ProcVisitInputsSpan,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputsSpan, InterpreterError> {
    let result = normalize_ann_proc(left, input.clone(), env, parser)?;
    let chained_input = ProcVisitInputsSpan {
        par: result.par.clone(),
        free_map: result.free_map.clone(),
        ..input.clone()
    };

    let chained_res = normalize_ann_proc(right, chained_input, env, parser)?;
    Ok(chained_res)
}

// See rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use models::{
        create_bit_vector,
        rhoapi::Par,
        rust::utils::{new_boundvar_expr, new_freevar_expr, new_gint_expr},
    };

    use crate::rust::interpreter::{
        compiler::{exports::ProcVisitInputsSpan, normalize::VarSort},
        errors::InterpreterError,
        test_utils::utils::proc_visit_inputs_and_env_span,
    };

    use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
    use rholang_parser::ast::{AnnProc, Id, Proc, Var};
    use rholang_parser::{SourcePos, SourceSpan};

    fn create_new_ast_long_proc<'ast>(value: i64) -> AnnProc<'ast> {
        let proc = Box::leak(Box::new(Proc::LongLiteral(value)));
        AnnProc {
            proc,
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        }
    }

    fn create_new_ast_var_proc<'ast>(name: &'ast str) -> AnnProc<'ast> {
        let proc = Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
            name,
            pos: SourcePos { line: 1, col: 1 },
        }))));
        AnnProc {
            proc,
            span: SourceSpan {
                start: SourcePos { line: 1, col: 1 },
                end: SourcePos { line: 1, col: 1 },
            },
        }
    }

    #[test]
    fn new_ast_p_par_should_compile_both_branches_into_a_par_object() {
        let left_proc = create_new_ast_long_proc(7);
        let right_proc = create_new_ast_long_proc(8);

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(
            &AnnProc {
                proc: Box::leak(Box::new(Proc::Par {
                    left: left_proc,
                    right: right_proc,
                })),
                span: SourceSpan {
                    start: SourcePos { line: 1, col: 1 },
                    end: SourcePos { line: 1, col: 1 },
                },
            },
            ProcVisitInputsSpan::new(),
            &HashMap::new(),
            &parser,
        );

        assert!(result.is_ok());
        assert_eq!(
            result.clone().unwrap().par,
            Par::default().with_exprs(vec![new_gint_expr(8), new_gint_expr(7)])
        );
        assert_eq!(
            result.unwrap().free_map,
            ProcVisitInputsSpan::new().free_map
        );
    }

    #[test]
    fn new_ast_p_par_should_compile_both_branches_with_the_same_environment() {
        let left_proc = create_new_ast_var_proc("x");
        let right_proc = create_new_ast_var_proc("x");

        let (mut inputs, env) = proc_visit_inputs_and_env_span();
        inputs.bound_map_chain = inputs.bound_map_chain.put_pos((
            "x".to_string(),
            VarSort::ProcSort,
            SourcePos { line: 0, col: 0 },
        ));

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(
            &AnnProc {
                proc: Box::leak(Box::new(Proc::Par {
                    left: left_proc,
                    right: right_proc,
                })),
                span: SourceSpan {
                    start: SourcePos { line: 1, col: 1 },
                    end: SourcePos { line: 1, col: 1 },
                },
            },
            inputs,
            &env,
            &parser,
        );

        assert!(result.is_ok());
        assert_eq!(result.clone().unwrap().par, {
            let mut par =
                Par::default().with_exprs(vec![new_boundvar_expr(0), new_boundvar_expr(0)]);
            par.locally_free = create_bit_vector(&vec![0]);
            par
        });
        assert_eq!(
            result.unwrap().free_map,
            ProcVisitInputsSpan::new().free_map
        );
    }

    #[test]
    fn new_ast_p_par_should_not_compile_if_both_branches_use_the_same_free_variable() {
        let left_proc = create_new_ast_var_proc("x");
        let right_proc = create_new_ast_var_proc("x");

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(
            &AnnProc {
                proc: Box::leak(Box::new(Proc::Par {
                    left: left_proc,
                    right: right_proc,
                })),
                span: SourceSpan {
                    start: SourcePos { line: 1, col: 1 },
                    end: SourcePos { line: 1, col: 1 },
                },
            },
            ProcVisitInputsSpan::new(),
            &HashMap::new(),
            &parser,
        );

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InterpreterError::UnexpectedReuseOfProcContextFreeSpan {
                var_name,
                first_use: _,
                second_use: _
            }) if var_name == "x"
        ));
    }

    #[test]
    fn new_ast_p_par_should_accumulate_free_counts_from_both_branches() {
        let left_proc = create_new_ast_var_proc("x");
        let right_proc = create_new_ast_var_proc("y");

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(
            &AnnProc {
                proc: Box::leak(Box::new(Proc::Par {
                    left: left_proc,
                    right: right_proc,
                })),
                span: SourceSpan {
                    start: SourcePos { line: 1, col: 1 },
                    end: SourcePos { line: 1, col: 1 },
                },
            },
            ProcVisitInputsSpan::new(),
            &HashMap::new(),
            &parser,
        );

        assert!(result.is_ok());
        assert_eq!(result.clone().unwrap().par, {
            let mut par = Par::default().with_exprs(vec![new_freevar_expr(1), new_freevar_expr(0)]);
            par.connective_used = true;
            par
        });
        assert_eq!(
            result.unwrap().free_map,
            ProcVisitInputsSpan::new().free_map.put_all_pos(vec![
                (
                    "x".to_owned(),
                    VarSort::ProcSort,
                    SourcePos { line: 1, col: 1 }
                ), // Note: Uses new AST source positions
                (
                    "y".to_owned(),
                    VarSort::ProcSort,
                    SourcePos { line: 1, col: 1 }
                )
            ])
        )
    }

    /*
     * TODO:
     * In this test case, 'huge_par' should iterate up to '50000'
     * Without passing 'RUST_MIN_STACK' env variable, this test case will fail with StackOverflowError
     * To test this correctly, change '50' to '50000' and run test with this command: 'env RUST_MIN_STACK=2147483648 cargo test'
     *
     * 'RUST_MIN_STACK=2147483648' sets stack size to 2GB for rust program
     * 'RUST_MIN_STACK=1073741824' sets stack size to 1GB
     * 'RUST_MIN_STACK=536870912' sets stack size to 512MB
     */
    #[test]
    fn new_ast_p_par_should_normalize_without_stack_overflow_error_even_for_huge_program() {
        // Create a huge nested Par with integers 1..50 using new AST
        fn create_huge_par<'ast>(range: std::ops::RangeInclusive<i64>) -> AnnProc<'ast> {
            let mut iter = range.into_iter();
            let first = iter.next().unwrap();
            let first_proc = create_new_ast_long_proc(first);

            iter.fold(first_proc, |acc, n| {
                let next_proc = create_new_ast_long_proc(n);
                AnnProc {
                    proc: Box::leak(Box::new(Proc::Par {
                        left: acc,
                        right: next_proc,
                    })),
                    span: SourceSpan {
                        start: SourcePos { line: 1, col: 1 },
                        end: SourcePos { line: 1, col: 1 },
                    },
                }
            })
        }

        let huge_par = create_huge_par(1..=50);

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(
            &huge_par,
            ProcVisitInputsSpan::new(),
            &HashMap::new(),
            &parser,
        );
        assert!(result.is_ok());
    }
}
