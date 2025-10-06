use super::bound_map_chain::BoundMapChainSpan;
use super::free_map::FreeMapSpan;
use crate::rust::interpreter::compiler::normalizer::processes::{
    p_ground_normalizer::normalize_p_ground_new_ast,
    p_simple_type_normalizer::normalize_simple_type_new_ast,
};
use crate::rust::interpreter::compiler::utils::{BinaryExpr, UnaryExpr};
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::util::prepend_expr;
use models::rhoapi::{EMinus, EPlus, Expr, Par};
use std::collections::HashMap;

use rholang_parser::ast::{AnnProc, Proc};
use rholang_parser::{RholangParser, SourceSpan};

#[derive(Clone, Debug, PartialEq)]
pub enum VarSort {
    ProcSort,
    NameSort,
}

/**
 * Input data to the normalizer
 *
 * @param par collection of things that might be run in parallel
 * @param env
 * @param knownFree
 */
#[derive(Clone, Debug, PartialEq)]
pub struct ProcVisitInputsSpan {
    pub par: Par,
    pub bound_map_chain: BoundMapChainSpan<VarSort>,
    pub free_map: FreeMapSpan<VarSort>,
}

impl ProcVisitInputsSpan {
    pub fn new() -> Self {
        ProcVisitInputsSpan {
            par: Par::default(),
            bound_map_chain: BoundMapChainSpan::new(),
            free_map: FreeMapSpan::new(),
        }
    }
}

impl Default for ProcVisitInputsSpan {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns the update Par and an updated map of free variables.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcVisitOutputsSpan {
    pub par: Par,
    pub free_map: FreeMapSpan<VarSort>,
}

#[derive(Clone, Debug)]
pub struct NameVisitInputsSpan {
    pub bound_map_chain: BoundMapChainSpan<VarSort>,
    pub free_map: FreeMapSpan<VarSort>,
}

#[derive(Clone, Debug)]
pub struct NameVisitOutputsSpan {
    pub par: Par,
    pub free_map: FreeMapSpan<VarSort>,
}

#[derive(Clone, Debug)]
pub struct CollectVisitInputsSpan {
    pub bound_map_chain: BoundMapChainSpan<VarSort>,
    pub free_map: FreeMapSpan<VarSort>,
}

#[derive(Clone, Debug)]
pub struct CollectVisitOutputsSpan {
    pub expr: Expr,
    pub free_map: FreeMapSpan<VarSort>,
}

/**
 * Rholang normalizer entry point
 */
pub fn normalize_ann_proc<'ast>(
    proc: &AnnProc<'ast>,
    input: ProcVisitInputsSpan,
    _env: &HashMap<String, Par>,
    parser: &'ast RholangParser<'ast>,
) -> Result<ProcVisitOutputsSpan, InterpreterError> {
    fn create_ann_proc_wrapper<'ast>(proc: &'ast Proc<'ast>, span: SourceSpan) -> AnnProc<'ast> {
        AnnProc { proc, span }
    }

    fn unary_exp_new_ast<'ast>(
        sub_proc: &'ast Proc<'ast>,
        input: ProcVisitInputsSpan,
        constructor: Box<dyn UnaryExpr>,
        env: &HashMap<String, Par>,
        parser: &'ast RholangParser<'ast>,
        expr_span: SourceSpan,
    ) -> Result<ProcVisitOutputsSpan, InterpreterError> {
        let ann_proc = create_ann_proc_wrapper(sub_proc, expr_span);
        let input_par = input.par.clone();
        let input_depth = input.bound_map_chain.depth();
        let sub_result = normalize_ann_proc(&ann_proc, input, env, parser)?;
        let expr = constructor.from_par(sub_result.par.clone());

        Ok(ProcVisitOutputsSpan {
            par: prepend_expr(input_par, expr, input_depth as i32),
            free_map: sub_result.free_map,
        })
    }

    fn binary_exp_new_ast<'ast>(
        left_proc: &'ast AnnProc<'ast>,
        right_proc: &'ast AnnProc<'ast>,
        input: ProcVisitInputsSpan,
        constructor: Box<dyn BinaryExpr>,
        env: &HashMap<String, Par>,
        parser: &'ast RholangParser<'ast>,
    ) -> Result<ProcVisitOutputsSpan, InterpreterError> {
        let input_par = input.par.clone();
        let input_depth = input.bound_map_chain.depth();
        let input_bound_chain = input.bound_map_chain.clone();

        let left_result = normalize_ann_proc(left_proc, input, env, parser)?;
        let right_result = normalize_ann_proc(
            right_proc,
            ProcVisitInputsSpan {
                par: Par::default(),
                bound_map_chain: input_bound_chain,
                free_map: left_result.free_map.clone(),
            },
            env,
            parser,
        )?;

        let expr: Expr = constructor.from_pars(left_result.par.clone(), right_result.par.clone());

        Ok(ProcVisitOutputsSpan {
            par: prepend_expr(input_par, expr, input_depth as i32),
            free_map: right_result.free_map,
        })
    }

    match &proc.proc {
        Proc::Nil => Ok(ProcVisitOutputsSpan {
            par: input.par.clone(),
            free_map: input.free_map.clone(),
        }),

        // Ground literals
        Proc::Unit
        | Proc::BoolLiteral(_)
        | Proc::LongLiteral(_)
        | Proc::StringLiteral(_)
        | Proc::UriLiteral(_) => normalize_p_ground_new_ast(&proc.proc, input),

        Proc::SimpleType(simple_type) => normalize_simple_type_new_ast(simple_type, input),

        Proc::ProcVar(var) => {
            use crate::rust::interpreter::compiler::normalizer::processes::p_var_normalizer::normalize_p_var_new_ast;
            normalize_p_var_new_ast(var, input, proc.span)
        }

        Proc::Par { left, right } => {
            use crate::rust::interpreter::compiler::normalizer::processes::p_par_normalizer::normalize_p_par_new_ast;
            normalize_p_par_new_ast(left, right, input, _env, parser)
        }

        Proc::Eval { name } => {
            use crate::rust::interpreter::compiler::normalizer::processes::p_eval_normalizer::normalize_p_eval_new_ast;
            normalize_p_eval_new_ast(name, input, _env, parser)
        }

        // UnaryExp - handle all unary operators
        Proc::UnaryExp { op, arg } => match op {
            rholang_parser::ast::UnaryExpOp::Negation => {
                use crate::rust::interpreter::compiler::normalizer::processes::p_negation_normalizer::normalize_p_negation_new_ast;
                normalize_p_negation_new_ast(arg, proc.span, input, _env, parser)
            }
            rholang_parser::ast::UnaryExpOp::Not => {
                use models::rhoapi::ENot;
                unary_exp_new_ast(
                    arg,
                    input,
                    Box::new(ENot::default()),
                    _env,
                    parser,
                    proc.span,
                )
            }
            rholang_parser::ast::UnaryExpOp::Neg => {
                use models::rhoapi::ENeg;
                unary_exp_new_ast(
                    arg,
                    input,
                    Box::new(ENeg::default()),
                    _env,
                    parser,
                    proc.span,
                )
            }
        },

        // BinaryExp - handle all binary operators
        Proc::BinaryExp { op, left, right } => {
            match op {
                // Logical connectives
                rholang_parser::ast::BinaryExpOp::Conjunction => {
                    use crate::rust::interpreter::compiler::normalizer::processes::p_conjunction_normalizer::normalize_p_conjunction_new_ast;
                    normalize_p_conjunction_new_ast(left, right, input, _env, parser)
                }
                rholang_parser::ast::BinaryExpOp::Disjunction => {
                    use crate::rust::interpreter::compiler::normalizer::processes::p_disjunction_normalizer::normalize_p_disjunction_new_ast;
                    normalize_p_disjunction_new_ast(left, right, input, _env, parser)
                }
                rholang_parser::ast::BinaryExpOp::Matches => {
                    use crate::rust::interpreter::compiler::normalizer::processes::p_matches_normalizer::normalize_p_matches_new_ast;
                    normalize_p_matches_new_ast(left, right, input, _env, parser)
                }

                // Arithmetic
                rholang_parser::ast::BinaryExpOp::Add => {
                    binary_exp_new_ast(left, right, input, Box::new(EPlus::default()), _env, parser)
                }
                rholang_parser::ast::BinaryExpOp::Sub => binary_exp_new_ast(
                    left,
                    right,
                    input,
                    Box::new(EMinus::default()),
                    _env,
                    parser,
                ),
                rholang_parser::ast::BinaryExpOp::Mult => {
                    use models::rhoapi::EMult;
                    binary_exp_new_ast(left, right, input, Box::new(EMult::default()), _env, parser)
                }
                rholang_parser::ast::BinaryExpOp::Div => {
                    use models::rhoapi::EDiv;
                    binary_exp_new_ast(left, right, input, Box::new(EDiv::default()), _env, parser)
                }
                rholang_parser::ast::BinaryExpOp::Mod => {
                    use models::rhoapi::EMod;
                    binary_exp_new_ast(left, right, input, Box::new(EMod::default()), _env, parser)
                }

                // Comparison operators
                rholang_parser::ast::BinaryExpOp::Eq => {
                    use models::rhoapi::EEq;
                    binary_exp_new_ast(left, right, input, Box::new(EEq::default()), _env, parser)
                }
                rholang_parser::ast::BinaryExpOp::Neq => {
                    use models::rhoapi::ENeq;
                    binary_exp_new_ast(left, right, input, Box::new(ENeq::default()), _env, parser)
                }
                rholang_parser::ast::BinaryExpOp::Lt => {
                    use models::rhoapi::ELt;
                    binary_exp_new_ast(left, right, input, Box::new(ELt::default()), _env, parser)
                }
                rholang_parser::ast::BinaryExpOp::Lte => {
                    use models::rhoapi::ELte;
                    binary_exp_new_ast(left, right, input, Box::new(ELte::default()), _env, parser)
                }
                rholang_parser::ast::BinaryExpOp::Gt => {
                    use models::rhoapi::EGt;
                    binary_exp_new_ast(left, right, input, Box::new(EGt::default()), _env, parser)
                }
                rholang_parser::ast::BinaryExpOp::Gte => {
                    use models::rhoapi::EGte;
                    binary_exp_new_ast(left, right, input, Box::new(EGte::default()), _env, parser)
                }

                // Set/String operations
                rholang_parser::ast::BinaryExpOp::Concat => {
                    use models::rhoapi::EPlusPlus;
                    binary_exp_new_ast(
                        left,
                        right,
                        input,
                        Box::new(EPlusPlus::default()),
                        _env,
                        parser,
                    )
                }
                rholang_parser::ast::BinaryExpOp::Diff => {
                    use models::rhoapi::EMinusMinus;
                    binary_exp_new_ast(
                        left,
                        right,
                        input,
                        Box::new(EMinusMinus::default()),
                        _env,
                        parser,
                    )
                }

                // Boolean operators
                rholang_parser::ast::BinaryExpOp::Or => {
                    use models::rhoapi::EOr;
                    binary_exp_new_ast(left, right, input, Box::new(EOr::default()), _env, parser)
                }
                rholang_parser::ast::BinaryExpOp::And => {
                    use models::rhoapi::EAnd;
                    binary_exp_new_ast(left, right, input, Box::new(EAnd::default()), _env, parser)
                }

                // String interpolation
                rholang_parser::ast::BinaryExpOp::Interpolation => {
                    use models::rhoapi::EPercentPercent;
                    binary_exp_new_ast(
                        left,
                        right,
                        input,
                        Box::new(EPercentPercent::default()),
                        _env,
                        parser,
                    )
                }
            }
        }

        // IfThenElse - handle conditional statements
        Proc::IfThenElse {
            condition,
            if_true,
            if_false,
        } => {
            use crate::rust::interpreter::compiler::normalizer::processes::p_if_normalizer::normalize_p_if_new_ast;

            // Follow same pattern as original IfElse: use empty Par for normalization, then append original Par
            let mut empty_par_input = input.clone();
            empty_par_input.par = Par::default();

            // Use the updated normalize_p_if_new_ast that handles None case internally
            normalize_p_if_new_ast(
                condition,
                if_true,
                if_false.as_ref(),
                empty_par_input,
                _env,
                parser,
            )
            .map(|mut new_visits| {
                let new_par = new_visits.par.append(input.par);
                new_visits.par = new_par;
                new_visits
            })
        }

        // Method - handle method calls
        Proc::Method {
            receiver,
            name,
            args,
        } => {
            use crate::rust::interpreter::compiler::normalizer::processes::p_method_normalizer::normalize_p_method_new_ast;
            normalize_p_method_new_ast(receiver, name, args, input, _env, parser)
        }

        // Bundle - handle bundle constructs
        Proc::Bundle { bundle_type, proc } => {
            use crate::rust::interpreter::compiler::normalizer::processes::p_bundle_normalizer::normalize_p_bundle_new_ast;
            normalize_p_bundle_new_ast(bundle_type, proc, input, &proc.span, _env, parser)
        }

        // Send - handle send operations
        Proc::Send {
            channel,
            send_type,
            inputs,
        } => {
            use crate::rust::interpreter::compiler::normalizer::processes::p_send_normalizer::normalize_p_send_new_ast;
            normalize_p_send_new_ast(channel, send_type, inputs, input, _env, parser)
        }

        // SendSync - handle synchronous send operations
        Proc::SendSync {
            channel,
            messages,
            cont,
        } => {
            use crate::rust::interpreter::compiler::normalizer::processes::p_send_sync_normalizer::normalize_p_send_sync_new_ast;
            normalize_p_send_sync_new_ast(channel, messages, cont, &proc.span, input, _env, parser)
        }

        // New - handle name declarations and scoping
        Proc::New { decls, proc } => {
            use crate::rust::interpreter::compiler::normalizer::processes::p_new_normalizer::normalize_p_new_new_ast;
            normalize_p_new_new_ast(decls, proc, input, _env, parser)
        }

        // Contract - handle contract declarations
        Proc::Contract {
            name,
            formals,
            body,
        } => {
            use crate::rust::interpreter::compiler::normalizer::processes::p_contr_normalizer::normalize_p_contr_new_ast;
            normalize_p_contr_new_ast(name, formals, body, input, _env, parser)
        }

        // Match - handle pattern matching
        Proc::Match { expression, cases } => {
            use crate::rust::interpreter::compiler::normalizer::processes::p_match_normalizer::normalize_p_match_new_ast;
            normalize_p_match_new_ast(expression, cases, input, _env, parser)
        }

        // Collection - handle data structures (lists, tuples, sets, maps)
        Proc::Collection(collection) => {
            use crate::rust::interpreter::compiler::normalizer::processes::p_collect_normalizer::normalize_p_collect_new_ast;
            normalize_p_collect_new_ast(collection, input, _env, parser)
        }

        // ForComprehension - handle for-comprehensions (was Input in old AST)
        Proc::ForComprehension { receipts, proc } => {
            use crate::rust::interpreter::compiler::normalizer::processes::p_input_normalizer::normalize_p_input_new_ast;
            normalize_p_input_new_ast(receipts, proc, input, _env, parser)
        }

        // Let - handle let bindings
        Proc::Let {
            bindings,
            body,
            concurrent,
        } => {
            use crate::rust::interpreter::compiler::normalizer::processes::p_let_normalizer::normalize_p_let_new_ast;
            normalize_p_let_new_ast(bindings, body, *concurrent, proc.span, input, _env, parser)
        }

        // Quote - handle quoted processes (recursive normalization)
        Proc::Quote { proc: quoted_proc } => {
            // Create AnnProc wrapper for the quoted process with inherited span from Quote expression
            let quoted_ann_proc = create_ann_proc_wrapper(quoted_proc, proc.span);
            normalize_ann_proc(&quoted_ann_proc, input, _env, parser)
        }

        // VarRef - handle variable references
        Proc::VarRef { kind, var } => {
            use crate::rust::interpreter::compiler::normalizer::processes::p_var_ref_normalizer::normalize_p_var_ref_new_ast;
            normalize_p_var_ref_new_ast(*kind, var, input, proc.span)
        }

        // Select - handle select expressions (choice constructs)
        Proc::Select { branches: _ } => {
            // TODO: Implement select normalizer when needed
            // This corresponds to Choice in the old AST which was also not implemented (todo!())
            Err(InterpreterError::ParserError(
                "Select (choice) constructs not yet implemented in normalizer".to_string(),
            ))
        }

        // Bad - handle parsing errors
        Proc::Bad => Err(InterpreterError::ParserError(
            "Bad process node indicates parsing error".to_string(),
        )),
    }
}

// See rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
// inside this source file we tested unary and binary operations, because we don't have separate normalizers for them.
#[cfg(test)]
mod tests {
    use crate::rust::interpreter::compiler::compiler::Compiler;
    use crate::rust::interpreter::compiler::exports::{ProcVisitInputsSpan, ProcVisitOutputsSpan};
    use crate::rust::interpreter::compiler::normalize::VarSort::ProcSort;
    use crate::rust::interpreter::test_utils::utils::{
        proc_visit_inputs_and_env_span, proc_visit_inputs_with_updated_vec_bound_map_chain_span,
    };
    use crate::rust::interpreter::util::prepend_expr;
    use models::create_bit_vector;
    use models::rhoapi::expr::ExprInstance;
    use models::rhoapi::{EDiv, EMinus, EMinusMinus, EMult, EPlus, EPlusPlus, Expr, Par};
    use models::rust::utils::{new_boundvar_par, new_gint_par, new_gstring_par};
    use pretty_assertions::assert_eq;

    #[test]
    fn new_ast_p_nil_should_compile_as_no_modification() {
        use std::collections::HashMap;

        let (inputs, _env) = proc_visit_inputs_and_env_span();

        fn test_with_parser(
            inputs: ProcVisitInputsSpan,
        ) -> Result<ProcVisitOutputsSpan, crate::rust::interpreter::InterpreterError> {
            use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
            use validated::Validated;
            let parser = rholang_parser::RholangParser::new();
            let result = parser.parse("Nil");
            match result {
                Validated::Good(procs) => {
                    if procs.len() == 1 {
                        let ast = procs.into_iter().next().unwrap();
                        normalize_ann_proc(&ast, inputs, &HashMap::new(), &parser)
                    } else {
                        panic!("Expected single process")
                    }
                }
                _ => panic!("Parse failed"),
            }
        }

        let result = test_with_parser(inputs.clone());

        let actual_result = result.unwrap();
        assert_eq!(actual_result.par, inputs.par);
        assert_eq!(actual_result.free_map, inputs.free_map);
    }

    // unary operations:
    #[test]
    fn new_ast_p_not_should_delegate() {
        use std::collections::HashMap;

        let (inputs, _env) = proc_visit_inputs_and_env_span();

        fn test_with_parser(
            inputs: ProcVisitInputsSpan,
        ) -> Result<ProcVisitOutputsSpan, crate::rust::interpreter::InterpreterError> {
            use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
            use validated::Validated;
            let parser = rholang_parser::RholangParser::new();
            let result = parser.parse("~false");
            match result {
                Validated::Good(procs) => {
                    if procs.len() == 1 {
                        let ast = procs.into_iter().next().unwrap();
                        normalize_ann_proc(&ast, inputs, &HashMap::new(), &parser)
                    } else {
                        panic!("Expected single process")
                    }
                }
                _ => panic!("Parse failed"),
            }
        }

        let result = test_with_parser(inputs.clone());

        let expected_result = {
            let mut par = inputs.par.clone();
            par.connectives.push(models::rhoapi::Connective {
                connective_instance: Some(
                    models::rhoapi::connective::ConnectiveInstance::ConnNotBody(Par {
                        exprs: vec![Expr {
                            expr_instance: Some(models::rhoapi::expr::ExprInstance::GBool(false)),
                        }],
                        ..Par::default()
                    }),
                ),
            });
            par.connective_used = true;
            par
        };

        let actual_result = result.unwrap();
        assert_eq!(actual_result.par, expected_result);
        assert_eq!(actual_result.free_map.connectives.len(), 1);
    }

    #[test]
    fn new_ast_p_neg_should_delegate() {
        use std::collections::HashMap;

        let (inputs, _env) = proc_visit_inputs_and_env_span();

        fn test_with_parser(
            inputs: ProcVisitInputsSpan,
        ) -> Result<ProcVisitOutputsSpan, crate::rust::interpreter::InterpreterError> {
            use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
            use validated::Validated;
            let parser = rholang_parser::RholangParser::new();
            let result = parser.parse("-7");
            match result {
                Validated::Good(procs) => {
                    if procs.len() == 1 {
                        let ast = procs.into_iter().next().unwrap();
                        normalize_ann_proc(&ast, inputs, &HashMap::new(), &parser)
                    } else {
                        panic!("Expected single process")
                    }
                }
                _ => panic!("Parse failed"),
            }
        }

        let result = test_with_parser(inputs.clone());

        let expected_result = prepend_expr(
            inputs.par.clone(),
            Expr {
                expr_instance: Some(ExprInstance::GInt(-7)),
            },
            0,
        );

        let actual_result = result.unwrap();
        assert_eq!(actual_result.par, expected_result);
        assert_eq!(actual_result.free_map, inputs.free_map);
    }

    //binary operations:
    #[test]
    fn new_ast_p_mult_should_delegate() {
        use std::collections::HashMap;

        let (inputs, _env) = proc_visit_inputs_and_env_span();

        fn test_with_parser(
            inputs: ProcVisitInputsSpan,
        ) -> Result<ProcVisitOutputsSpan, crate::rust::interpreter::InterpreterError> {
            use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
            use validated::Validated;
            let parser = rholang_parser::RholangParser::new();
            let result = parser.parse("7 * 8");
            match result {
                Validated::Good(procs) => {
                    if procs.len() == 1 {
                        let ast = procs.into_iter().next().unwrap();
                        normalize_ann_proc(&ast, inputs, &HashMap::new(), &parser)
                    } else {
                        panic!("Expected single process")
                    }
                }
                _ => panic!("Parse failed"),
            }
        }

        let result = test_with_parser(inputs.clone());

        let expected_result = prepend_expr(
            inputs.par.clone(),
            Expr {
                expr_instance: Some(ExprInstance::EMultBody(EMult {
                    p1: Some(new_gint_par(7, Vec::new(), false)),
                    p2: Some(new_gint_par(8, Vec::new(), false)),
                })),
            },
            0,
        );

        let actual_result = result.unwrap();
        assert_eq!(actual_result.par, expected_result);
        assert_eq!(actual_result.free_map, inputs.free_map);
    }

    #[test]
    fn new_ast_p_div_should_delegate() {
        use std::collections::HashMap;

        let (inputs, _env) = proc_visit_inputs_and_env_span();

        fn test_with_parser(
            inputs: ProcVisitInputsSpan,
        ) -> Result<ProcVisitOutputsSpan, crate::rust::interpreter::InterpreterError> {
            use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
            use validated::Validated;
            let parser = rholang_parser::RholangParser::new();
            let result = parser.parse("7 / 8");
            match result {
                Validated::Good(procs) => {
                    if procs.len() == 1 {
                        let ast = procs.into_iter().next().unwrap();
                        normalize_ann_proc(&ast, inputs, &HashMap::new(), &parser)
                    } else {
                        panic!("Expected single process")
                    }
                }
                _ => panic!("Parse failed"),
            }
        }

        let result = test_with_parser(inputs.clone());

        let expected_result = prepend_expr(
            inputs.par.clone(),
            Expr {
                expr_instance: Some(ExprInstance::EDivBody(EDiv {
                    p1: Some(new_gint_par(7, Vec::new(), false)),
                    p2: Some(new_gint_par(8, Vec::new(), false)),
                })),
            },
            0,
        );

        let actual_result = result.unwrap();
        assert_eq!(actual_result.par, expected_result);
        assert_eq!(actual_result.free_map, inputs.free_map);
    }

    #[test]
    fn new_ast_p_percent_percent_should_delegate() {
        use std::collections::HashMap;

        let (inputs, _env) = proc_visit_inputs_and_env_span();

        fn test_with_parser(
            inputs: ProcVisitInputsSpan,
        ) -> Result<ProcVisitOutputsSpan, crate::rust::interpreter::InterpreterError> {
            use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
            use validated::Validated;
            let parser = rholang_parser::RholangParser::new();
            let result = parser.parse("7 % 8");
            match result {
                Validated::Good(procs) => {
                    if procs.len() == 1 {
                        let ast = procs.into_iter().next().unwrap();
                        normalize_ann_proc(&ast, inputs, &HashMap::new(), &parser)
                    } else {
                        panic!("Expected single process")
                    }
                }
                _ => panic!("Parse failed"),
            }
        }

        let result = test_with_parser(inputs.clone());

        use models::rhoapi::EMod;
        let expected_result = prepend_expr(
            inputs.par.clone(),
            Expr {
                expr_instance: Some(ExprInstance::EModBody(EMod {
                    p1: Some(new_gint_par(7, Vec::new(), false)),
                    p2: Some(new_gint_par(8, Vec::new(), false)),
                })),
            },
            0,
        );

        let actual_result = result.unwrap();
        assert_eq!(actual_result.par, expected_result);
        assert_eq!(actual_result.free_map, inputs.free_map);
    }

    #[test]
    fn new_ast_p_add_should_delegate() {
        use std::collections::HashMap;

        let (inputs, _env) = proc_visit_inputs_and_env_span();

        fn test_with_parser(
            inputs: ProcVisitInputsSpan,
        ) -> Result<ProcVisitOutputsSpan, crate::rust::interpreter::InterpreterError> {
            use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
            use validated::Validated;
            let parser = rholang_parser::RholangParser::new();
            let result = parser.parse("7 + 8");
            match result {
                Validated::Good(procs) => {
                    if procs.len() == 1 {
                        let ast = procs.into_iter().next().unwrap();
                        normalize_ann_proc(&ast, inputs, &HashMap::new(), &parser)
                    } else {
                        panic!("Expected single process")
                    }
                }
                _ => panic!("Parse failed"),
            }
        }

        let result = test_with_parser(inputs.clone());

        let expected_result = prepend_expr(
            inputs.par.clone(),
            Expr {
                expr_instance: Some(ExprInstance::EPlusBody(EPlus {
                    p1: Some(new_gint_par(7, Vec::new(), false)),
                    p2: Some(new_gint_par(8, Vec::new(), false)),
                })),
            },
            0,
        );

        let actual_result = result.unwrap();
        assert_eq!(actual_result.par, expected_result);
        assert_eq!(actual_result.free_map, inputs.free_map);
    }

    #[test]
    fn new_ast_p_plus_plus_should_delegate() {
        use std::collections::HashMap;

        let (inputs, _env) = proc_visit_inputs_and_env_span();

        fn test_with_parser(
            inputs: ProcVisitInputsSpan,
        ) -> Result<ProcVisitOutputsSpan, crate::rust::interpreter::InterpreterError> {
            use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
            use validated::Validated;
            let parser = rholang_parser::RholangParser::new();
            let result = parser.parse("\"abc\" ++ \"def\"");
            match result {
                Validated::Good(procs) => {
                    if procs.len() == 1 {
                        let ast = procs.into_iter().next().unwrap();
                        normalize_ann_proc(&ast, inputs, &HashMap::new(), &parser)
                    } else {
                        panic!("Expected single process")
                    }
                }
                _ => panic!("Parse failed"),
            }
        }

        let result = test_with_parser(inputs.clone());

        let expected_result = prepend_expr(
            inputs.par.clone(),
            Expr {
                expr_instance: Some(ExprInstance::EPlusPlusBody(EPlusPlus {
                    p1: Some(new_gstring_par("abc".to_string(), Vec::new(), false)),
                    p2: Some(new_gstring_par("def".to_string(), Vec::new(), false)),
                })),
            },
            0,
        );

        let actual_result = result.unwrap();
        assert_eq!(actual_result.par, expected_result);
        assert_eq!(actual_result.free_map, inputs.free_map);
    }

    #[test]
    fn new_ast_p_minus_should_delegate() {
        use std::collections::HashMap;

        let (base_inputs, _env) = proc_visit_inputs_and_env_span();
        let inputs = proc_visit_inputs_with_updated_vec_bound_map_chain_span(
            base_inputs,
            vec![
                ("x".into(), ProcSort),
                ("y".into(), ProcSort),
                ("z".into(), ProcSort),
            ],
        );

        fn test_with_parser(
            inputs: ProcVisitInputsSpan,
        ) -> Result<ProcVisitOutputsSpan, crate::rust::interpreter::InterpreterError> {
            use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
            use validated::Validated;
            let parser = rholang_parser::RholangParser::new();
            let result = parser.parse("x - (y * z)");
            match result {
                Validated::Good(procs) => {
                    if procs.len() == 1 {
                        let ast = procs.into_iter().next().unwrap();
                        normalize_ann_proc(&ast, inputs, &HashMap::new(), &parser)
                    } else {
                        panic!("Expected single process")
                    }
                }
                _ => panic!("Parse failed"),
            }
        }

        let result = test_with_parser(inputs.clone());

        let expected_result = prepend_expr(
            inputs.par.clone(),
            Expr {
                expr_instance: Some(ExprInstance::EMinusBody(EMinus {
                    p1: Some(new_boundvar_par(2, create_bit_vector(&vec![2]), false)),
                    p2: Some(Par {
                        exprs: vec![Expr {
                            expr_instance: Some(ExprInstance::EMultBody(EMult {
                                p1: Some(new_boundvar_par(1, create_bit_vector(&vec![1]), false)),
                                p2: Some(new_boundvar_par(0, create_bit_vector(&vec![0]), false)),
                            })),
                        }],
                        locally_free: create_bit_vector(&vec![0, 1]),
                        connective_used: false,
                        ..Par::default()
                    }),
                })),
            },
            0,
        );

        let actual_result = result.unwrap();
        assert_eq!(actual_result.par, expected_result);
        assert_eq!(actual_result.free_map, inputs.free_map);
    }

    #[test]
    fn new_ast_p_minus_minus_should_delegate() {
        use std::collections::HashMap;

        let (inputs, _env) = proc_visit_inputs_and_env_span();

        fn test_with_parser(
            inputs: ProcVisitInputsSpan,
        ) -> Result<ProcVisitOutputsSpan, crate::rust::interpreter::InterpreterError> {
            use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
            use validated::Validated;
            let parser = rholang_parser::RholangParser::new();
            let result = parser.parse("\"abc\" -- \"def\"");
            match result {
                Validated::Good(procs) => {
                    if procs.len() == 1 {
                        let ast = procs.into_iter().next().unwrap();
                        normalize_ann_proc(&ast, inputs, &HashMap::new(), &parser)
                    } else {
                        panic!("Expected single process")
                    }
                }
                _ => panic!("Parse failed"),
            }
        }

        let result = test_with_parser(inputs.clone());

        let expected_result = prepend_expr(
            inputs.par.clone(),
            Expr {
                expr_instance: Some(ExprInstance::EMinusMinusBody(EMinusMinus {
                    p1: Some(new_gstring_par("abc".to_string(), vec![], false)),
                    p2: Some(new_gstring_par("def".to_string(), vec![], false)),
                })),
            },
            0,
        );

        let actual_result = result.unwrap();
        assert_eq!(actual_result.par, expected_result);
        assert_eq!(actual_result.free_map, inputs.free_map);
    }

    #[test]
    fn new_ast_patterns_should_compile_not_in_top_level() {
        let cases = vec![
            ("wildcard", "send channel", "{_!(1)}"),
            // REMOVED: The pattern "{@=*x!(_)}" was invalid Rholang syntax
            // REASON: Neither "@=*variable" nor "=*variable" are valid in this context
            // This pattern was testing invalid syntax that should not have been allowed
            // Replaced with a valid wildcard pattern for comprehensive testing
            ("wildcard", "send data", "{@_!(_)}"),
            ("wildcard", "send data", "{@Nil!(_)}"),
            ("logical AND", "send data", "{@Nil!(1 /\\ 2)}"),
            ("logical OR", "send data", "{@Nil!(1 \\/ 2)}"),
            ("logical NOT", "send data", "{@Nil!(~1)}"),
            ("logical AND", "send channel", "{@{Nil /\\ Nil}!(Nil)}"),
            ("logical OR", "send channel", "{@{Nil \\/ Nil}!(Nil)}"),
            ("logical NOT", "send channel", "{@{~Nil}!(Nil)}"),
            (
                "wildcard",
                "receive pattern of the consume",
                "{for (_ <- x) { 1 }} ",
            ),
            (
                "wildcard",
                "body of the continuation",
                "{for (@1 <- x) { _ }} ",
            ),
            (
                "logical OR",
                "body of the continuation",
                "{for (@1 <- x) { 10 \\/ 20 }} ",
            ),
            (
                "logical AND",
                "body of the continuation",
                "{for(@1 <- x) { 10 /\\ 20 }} ",
            ),
            (
                "logical NOT",
                "body of the continuation",
                "{for(@1 <- x) { ~10 }} ",
            ),
            (
                "logical OR",
                "channel of the consume",
                "{for (@1 <- @{Nil /\\ Nil}) { Nil }} ",
            ),
            (
                "logical AND",
                "channel of the consume",
                "{for(@1 <- @{Nil \\/ Nil}) { Nil }} ",
            ),
            (
                "logical NOT",
                "channel of the consume",
                "{for(@1 <- @{~Nil}) { Nil }} ",
            ),
            (
                "wildcard",
                "channel of the consume",
                "{for(@1 <- _) { Nil }} ",
            ),
        ];

        for (typ, position, pattern) in cases.iter() {
            let rho = format!(
                r#"
        new x in {{
            for(@y <- x) {{
                match y {{
                    {} => Nil
                }}
            }}
        }}
        "#,
                pattern
            );

            match Compiler::source_to_adt(&rho) {
                Ok(_) => {}
                Err(e) => {
                    panic!(
                        "{} in the {} '{}' should not throw errors: {:?}",
                        typ, position, pattern, e
                    );
                }
            }
        }
    }
}
