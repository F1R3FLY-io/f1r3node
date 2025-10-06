use crate::rust::interpreter::compiler::exports::{
    CollectVisitInputs, CollectVisitOutputs, FreeMap, ProcVisitInputs,
};
use crate::rust::interpreter::compiler::normalize::{normalize_ann_proc, VarSort};
use crate::rust::interpreter::compiler::normalizer::remainder_normalizer_matcher::normalize_remainder;
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::matcher::has_locally_free::HasLocallyFree;
use models::rhoapi::expr::ExprInstance;
use models::rhoapi::{EList, ETuple, Expr, Par, Var};
use models::rust::par_map::ParMap;
use models::rust::par_map_type_mapper::ParMapTypeMapper;
use models::rust::par_set::ParSet;
use models::rust::par_set_type_mapper::ParSetTypeMapper;
use models::rust::sorted_par_hash_set::SortedParHashSet;
use models::rust::sorted_par_map::SortedParMap;
use models::rust::utils::union;
use std::collections::HashMap;
use std::result::Result;

use rholang_parser::ast::{AnnProc, Collection, KeyValuePair};

pub fn normalize_collection<'ast>(
    proc: &'ast Collection<'ast>,
    input: CollectVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<CollectVisitOutputs, InterpreterError> {
    pub fn fold_match<'ast, F>(
        known_free: FreeMap<VarSort>,
        elements: &[AnnProc<'ast>],
        constructor: F,
        input: CollectVisitInputs,
        env: &HashMap<String, Par>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> Result<CollectVisitOutputs, InterpreterError>
    where
        F: Fn(Vec<Par>, Vec<u8>, bool) -> Expr,
    {
        let init = (vec![], known_free.clone(), Vec::new(), false);
        let (mut acc_pars, mut result_known_free, mut locally_free, mut connective_used) = init;

        for element in elements {
            let result = normalize_ann_proc(
                element,
                ProcVisitInputs {
                    par: Par::default(),
                    bound_map_chain: input.bound_map_chain.clone(),
                    free_map: result_known_free.clone(),
                },
                env,
                parser,
            )?;

            acc_pars.push(result.par.clone());
            result_known_free = result.free_map.clone();
            locally_free = union(locally_free, result.par.locally_free);
            connective_used = connective_used || result.par.connective_used;
        }

        let constructed_expr: Expr = constructor(acc_pars, locally_free, connective_used);
        let expr: Expr = constructed_expr.into();

        Ok(CollectVisitOutputs {
            expr,
            free_map: result_known_free,
        })
    }

    pub fn fold_match_map<'ast>(
        known_free: FreeMap<VarSort>,
        remainder: Option<Var>,
        pairs: &[KeyValuePair<'ast>],
        input: CollectVisitInputs,
        env: &HashMap<String, Par>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> Result<CollectVisitOutputs, InterpreterError> {
        let init = (vec![], known_free.clone(), Vec::new(), false);

        let (mut acc_pairs, mut result_known_free, mut locally_free, mut connective_used) = init;

        for key_value_pair in pairs {
            let key_result = normalize_ann_proc(
                &key_value_pair.0,
                ProcVisitInputs {
                    par: Par::default(),
                    bound_map_chain: input.bound_map_chain.clone(),
                    free_map: result_known_free.clone(),
                },
                env,
                parser,
            )?;

            let value_result = normalize_ann_proc(
                &key_value_pair.1,
                ProcVisitInputs {
                    par: Par::default(),
                    bound_map_chain: input.bound_map_chain.clone(),
                    free_map: key_result.free_map.clone(),
                },
                env,
                parser,
            )?;

            acc_pairs.push((key_result.par.clone(), value_result.par.clone()));
            result_known_free = value_result.free_map.clone();
            locally_free = union(
                locally_free,
                union(key_result.par.locally_free, value_result.par.locally_free),
            );
            connective_used = connective_used
                || key_result.par.connective_used
                || value_result.par.connective_used;
        }

        let remainder_connective_used = match remainder {
            Some(ref var) => var.connective_used(var.clone()),
            None => false,
        };

        let remainder_locally_free = match remainder {
            Some(ref var) => var.locally_free(var.clone(), 0),
            None => Vec::new(),
        };

        let expr = Expr {
            expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
                ParMap {
                    ps: SortedParMap::create_from_vec(
                        acc_pairs.clone().into_iter().rev().collect(),
                    ),
                    connective_used: connective_used || remainder_connective_used,
                    locally_free: union(locally_free, remainder_locally_free),
                    remainder: remainder.clone(),
                },
            ))),
        };

        Ok(CollectVisitOutputs {
            expr,
            free_map: result_known_free,
        })
    }

    match proc {
        Collection::List {
            elements,
            remainder,
        } => {
            let (optional_remainder, known_free) =
                normalize_remainder(remainder, input.free_map.clone())?;

            let constructor =
                |ps: Vec<Par>, locally_free: Vec<u8>, connective_used: bool| -> Expr {
                    let mut tmp_e_list = EList {
                        ps,
                        locally_free,
                        connective_used,
                        remainder: optional_remainder.clone(),
                    };

                    tmp_e_list.connective_used =
                        tmp_e_list.connective_used || optional_remainder.is_some();
                    Expr {
                        expr_instance: Some(ExprInstance::EListBody(tmp_e_list)),
                    }
                };

            fold_match(known_free, elements, constructor, input, env, parser)
        }

        Collection::Tuple(elements) => {
            let constructor =
                |ps: Vec<Par>, locally_free: Vec<u8>, connective_used: bool| -> Expr {
                    let tmp_tuple = ETuple {
                        ps,
                        locally_free,
                        connective_used,
                    };

                    Expr {
                        expr_instance: Some(ExprInstance::ETupleBody(tmp_tuple)),
                    }
                };

            fold_match(
                input.free_map.clone(),
                elements,
                constructor,
                input,
                env,
                parser,
            )
        }

        Collection::Set {
            elements,
            remainder,
        } => {
            let (optional_remainder, known_free) =
                normalize_remainder(remainder, input.free_map.clone())?;

            let constructor =
                |pars: Vec<Par>, locally_free: Vec<u8>, connective_used: bool| -> Expr {
                    let mut tmp_par_set = ParSet {
                        ps: SortedParHashSet::create_from_vec(pars),
                        locally_free,
                        connective_used,
                        remainder: optional_remainder.clone(),
                    };

                    tmp_par_set.connective_used =
                        tmp_par_set.connective_used || optional_remainder.is_some();

                    let eset = ParSetTypeMapper::par_set_to_eset(tmp_par_set);

                    Expr {
                        expr_instance: Some(ExprInstance::ESetBody(eset)),
                    }
                };

            fold_match(known_free, elements, constructor, input, env, parser)
        }

        Collection::Map {
            elements,
            remainder,
        } => {
            let (optional_remainder, known_free) =
                normalize_remainder(remainder, input.free_map.clone())?;

            fold_match_map(known_free, optional_remainder, elements, input, env, parser)
        }
    }
}

//rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/CollectMatcherSpec.scala
#[cfg(test)]
mod tests {
    use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
    use crate::rust::interpreter::compiler::normalize::VarSort::{NameSort, ProcSort};
    use crate::rust::interpreter::errors::InterpreterError;
    use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;
    use crate::rust::interpreter::test_utils::utils::collection_proc_visit_inputs_and_env;
    use crate::rust::interpreter::util::prepend_expr;
    use models::create_bit_vector;
    use models::rhoapi::{KeyValuePair, Par};
    use models::rust::utils::{
        new_boundvar_par, new_elist_expr, new_emap_expr, new_eplus_par, new_eset_expr,
        new_etuple_expr, new_freevar_expr, new_freevar_par, new_freevar_var, new_gint_par,
        new_gstring_par,
    };
    use pretty_assertions::assert_eq;
    use rholang_parser::ast::{AnnName, AnnProc, Collection, Id, Name, Proc, Var};
    use rholang_parser::{SourcePos, SourceSpan};

    fn get_normalized_par(rho: &str) -> Par {
        ParBuilderUtil::mk_term(rho).expect("Compilation failed to normalize Par")
    }

    pub fn assert_equal_normalized(rho1: &str, rho2: &str) {
        assert_eq!(
            get_normalized_par(rho1),
            get_normalized_par(rho2),
            "Normalized Par values are not equal"
        );
    }

    #[test]
    fn list_should_delegate() {
        let (inputs, env) = collection_proc_visit_inputs_and_env();

        let proc = AnnProc {
            proc: Box::leak(Box::new(Proc::Collection(Collection::List {
                elements: vec![
                    AnnProc {
                        proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                            name: "P",
                            pos: SourcePos { line: 0, col: 0 },
                        })))),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    AnnProc {
                        proc: Box::leak(Box::new(Proc::Eval {
                            name: AnnName {
                                name: Name::ProcVar(Var::Id(Id {
                                    name: "x",
                                    pos: SourcePos { line: 0, col: 0 },
                                })),
                                span: SourceSpan {
                                    start: SourcePos { line: 0, col: 0 },
                                    end: SourcePos { line: 0, col: 0 },
                                },
                            },
                        })),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                    AnnProc {
                        proc: Box::leak(Box::new(Proc::LongLiteral(7))),
                        span: SourceSpan {
                            start: SourcePos { line: 0, col: 0 },
                            end: SourcePos { line: 0, col: 0 },
                        },
                    },
                ],
                remainder: None,
            }))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&proc, inputs.clone(), &env, &parser);
        let expected_result = prepend_expr(
            inputs.par.clone(),
            new_elist_expr(
                vec![
                    new_boundvar_par(1, create_bit_vector(&vec![1]), false),
                    new_boundvar_par(0, create_bit_vector(&vec![0]), false),
                    new_gint_par(7, Vec::new(), false),
                ],
                create_bit_vector(&vec![0, 1]),
                false,
                None,
            ),
            0,
        );

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(result.clone().unwrap().free_map, inputs.free_map);
    }

    #[test]
    fn list_should_sort_the_insides_of_their_elements() {
        assert_equal_normalized("@0!([{1 | 2}])", "@0!([{2 | 1}])");
    }

    #[test]
    fn list_should_sort_the_insides_of_send_encoded_as_byte_array() {
        let rho1 = r#"
        new x in {
          x!(
            [
              @"a"!(
                @"x"!("abc") |
                @"y"!(1)
              )
            ].toByteArray()
          )
        }
    "#;

        let rho2 = r#"
        new x in {
          x!(
            [
              @"a"!(
                @"y"!(1) |
                @"x"!("abc")
              )
            ].toByteArray()
          )
        }
    "#;
        assert_equal_normalized(&rho1, &rho2);
    }

    #[test]
    fn tuple_should_delegate() {
        let (inputs, env) = collection_proc_visit_inputs_and_env();

        let proc = AnnProc {
            proc: Box::leak(Box::new(Proc::Collection(Collection::Tuple(vec![
                AnnProc {
                    proc: Box::leak(Box::new(Proc::Eval {
                        name: AnnName {
                            name: Name::ProcVar(Var::Id(Id {
                                name: "y",
                                pos: SourcePos { line: 0, col: 0 },
                            })),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                    })),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                AnnProc {
                    proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                        name: "Q",
                        pos: SourcePos { line: 0, col: 0 },
                    })))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
            ])))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&proc, inputs.clone(), &env, &parser);
        let expected_result = prepend_expr(
            inputs.par.clone(),
            new_etuple_expr(
                vec![
                    new_freevar_par(0, Vec::new()),
                    new_freevar_par(1, Vec::new()),
                ],
                Vec::new(),
                true,
            ),
            0,
        );

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(
            result.clone().unwrap().free_map,
            inputs.free_map.put_all_pos(vec![
                ("y".to_string(), NameSort, SourcePos { line: 0, col: 0 }),
                ("Q".to_string(), ProcSort, SourcePos { line: 0, col: 0 })
            ])
        )
    }

    #[test]
    fn tuple_should_propagate_free_variables() {
        let (inputs, env) = collection_proc_visit_inputs_and_env();

        let proc = AnnProc {
            proc: Box::leak(Box::new(Proc::Collection(Collection::Tuple(vec![
                AnnProc {
                    proc: Box::leak(Box::new(Proc::LongLiteral(7))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                AnnProc {
                    proc: Box::leak(Box::new(Proc::Par {
                        left: AnnProc {
                            proc: Box::leak(Box::new(Proc::LongLiteral(7))),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                        right: AnnProc {
                            proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                                name: "Q",
                                pos: SourcePos { line: 0, col: 0 },
                            })))),
                            span: SourceSpan {
                                start: SourcePos { line: 0, col: 0 },
                                end: SourcePos { line: 0, col: 0 },
                            },
                        },
                    })),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                AnnProc {
                    proc: Box::leak(Box::new(Proc::ProcVar(Var::Id(Id {
                        name: "Q",
                        pos: SourcePos { line: 0, col: 0 },
                    })))),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
            ])))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&proc, inputs.clone(), &env, &parser);

        assert!(matches!(
            result,
            Err(InterpreterError::UnexpectedReuseOfProcContextFreeSpan { .. })
        ));
    }

    #[test]
    fn tuple_should_sort_the_insides_of_their_elements() {
        assert_equal_normalized("@0!(({1 | 2}))", "@0!(({2 | 1}))");
    }

    #[test]
    fn set_should_delegate() {
        let parser = rholang_parser::RholangParser::new();
        let (inputs, env) = collection_proc_visit_inputs_and_env();

        let proc = ParBuilderUtil::new_ast_set(
            vec![
                ParBuilderUtil::new_ast_add_with_par_of_var("P", "R", &parser),
                ParBuilderUtil::new_ast_int(7, &parser),
                ParBuilderUtil::new_ast_par_with_int_and_var(8, "Q", &parser),
            ],
            Some(ParBuilderUtil::new_ast_var("Z")),
            &parser,
        );

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&proc, inputs.clone(), &env, &parser);
        let expected_result = prepend_expr(
            inputs.par.clone(),
            new_eset_expr(
                vec![
                    new_eplus_par(
                        new_boundvar_par(1, create_bit_vector(&vec![1]), false),
                        new_freevar_par(1, Vec::new()),
                    ),
                    new_gint_par(7, Vec::new(), false),
                    prepend_expr(new_gint_par(8, Vec::new(), false), new_freevar_expr(2), 0),
                ],
                create_bit_vector(&vec![1]),
                true,
                Some(new_freevar_var(0)),
            ),
            0,
        );

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(
            result.unwrap().free_map,
            inputs.free_map.put_all_pos(vec![
                ("Z".to_string(), ProcSort, SourcePos { line: 0, col: 0 }),
                ("R".to_string(), ProcSort, SourcePos { line: 0, col: 0 }),
                ("Q".to_string(), ProcSort, SourcePos { line: 0, col: 0 }),
            ])
        );
    }

    #[test]
    fn set_should_sort_the_insides_of_their_elements() {
        assert_equal_normalized("@0!(Set({1 | 2}))", "@0!(Set({2 | 1}))")
    }

    #[test]
    fn map_should_delegate() {
        let parser = rholang_parser::RholangParser::new();
        let (inputs, env) = collection_proc_visit_inputs_and_env();

        let proc = ParBuilderUtil::new_ast_map(
            vec![
                ParBuilderUtil::new_ast_key_value_pair(
                    ParBuilderUtil::new_ast_int(7, &parser),
                    ParBuilderUtil::new_ast_string("Seven", &parser),
                ),
                ParBuilderUtil::new_ast_key_value_pair(
                    ParBuilderUtil::new_ast_proc_var("P", &parser),
                    ParBuilderUtil::new_ast_eval_name_var("Q", &parser),
                ),
            ],
            Some(ParBuilderUtil::new_ast_var("Z")),
            &parser,
        );

        let parser = rholang_parser::RholangParser::new();
        let result = normalize_ann_proc(&proc, inputs.clone(), &env, &parser);
        let expected_result = prepend_expr(
            inputs.par.clone(),
            new_emap_expr(
                vec![
                    KeyValuePair {
                        key: Some(new_gint_par(7, Vec::new(), false)),
                        value: Some(new_gstring_par("Seven".parse().unwrap(), Vec::new(), false)),
                    },
                    KeyValuePair {
                        key: Some(new_boundvar_par(1, create_bit_vector(&vec![1]), false)),
                        value: Some(new_freevar_par(1, Vec::new())),
                    },
                ],
                create_bit_vector(&vec![1]),
                true,
                Some(new_freevar_var(0)),
            ),
            0,
        );

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(
            result.unwrap().free_map,
            inputs.free_map.put_all_pos(vec![
                ("Z".to_string(), ProcSort, SourcePos { line: 0, col: 0 }),
                ("Q".to_string(), NameSort, SourcePos { line: 0, col: 0 }),
            ])
        );
    }

    #[test]
    fn map_should_sort_the_insides_of_their_keys() {
        assert_equal_normalized("@0!({{1 | 2} : 0})", "@0!({{2 | 1} : 0})")
    }

    #[test]
    fn map_should_sort_the_insides_of_their_values() {
        assert_equal_normalized("@0!({0 : {1 | 2}})", "@0!({0 : {2 | 1}})")
    }
}
