use crate::rust::interpreter::compiler::exports::{
    BoundMapChain, IdContextPos, NameVisitInputs, ProcVisitInputs, ProcVisitOutputs,
};
use crate::rust::interpreter::compiler::normalize::{normalize_ann_proc, VarSort};
use crate::rust::interpreter::compiler::normalizer::name_normalize_matcher::normalize_name;
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::util::filter_and_adjust_bitset;
use crate::rust::interpreter::util::prepend_new;
use models::rhoapi::{New, Par};
use std::collections::{BTreeMap, HashMap};

use rholang_parser::ast::{AnnProc, Name, NameDecl, Var};
use rholang_parser::SourcePos;

pub fn normalize_p_new<'ast>(
    decls: &[NameDecl<'ast>],
    proc: &'ast AnnProc<'ast>,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    // TODO: bindings within a single new shouldn't have overlapping names. - OLD
    // Tuple format: (uri, name, sort, line, col, space_type)
    let new_tagged_bindings: Vec<(Option<String>, String, VarSort, usize, usize, Option<Name<'ast>>)> = decls
        .iter()
        .map(|decl| match decl {
            NameDecl { id, uri: None, space_type, .. } => Ok((
                None,
                id.name.to_string(),
                VarSort::NameSort,
                id.pos.line,
                id.pos.col,
                space_type.clone(),
            )),
            NameDecl { id, uri: Some(urn), space_type, .. } => Ok((
                Some((**urn).to_string()), // Dereference Uri to get the inner &str
                id.name.to_string(),
                VarSort::NameSort,
                id.pos.line,
                id.pos.col,
                space_type.clone(),
            )),
        })
        .collect::<Result<Vec<_>, InterpreterError>>()?;

    // Sort bindings: None's first, then URI's lexicographically
    let mut sorted_bindings: Vec<(Option<String>, String, VarSort, usize, usize, Option<Name<'ast>>)> =
        new_tagged_bindings;
    sorted_bindings.sort_by(|a, b| a.0.cmp(&b.0));

    let new_bindings: Vec<IdContextPos<VarSort>> = sorted_bindings
        .iter()
        .map(|row| {
            (
                row.1.clone(),
                row.2.clone(),
                SourcePos {
                    line: row.3,
                    col: row.4,
                },
            )
        })
        .collect();

    let uris: Vec<String> = sorted_bindings
        .iter()
        .filter_map(|row| row.0.clone())
        .collect();

    // Normalize space_types to Pars
    // Space types are resolved in the outer scope (before adding new bindings)
    let mut space_types: Vec<Par> = Vec::with_capacity(sorted_bindings.len());
    let mut current_free_map = input.free_map.clone();

    for binding in &sorted_bindings {
        if let Some(space_name) = &binding.5 {
            // Skip wildcards - they mean "no space type specified"
            if matches!(space_name, Name::NameVar(Var::Wildcard)) {
                space_types.push(Par::default());
                continue;
            }

            // Normalize the space name in the outer scope
            let name_input = NameVisitInputs {
                bound_map_chain: input.bound_map_chain.clone(),
                free_map: current_free_map.clone(),
            };

            let name_result = normalize_name(space_name, name_input, env, parser)?;
            space_types.push(name_result.par);
            current_free_map = name_result.free_map;
        } else {
            // No space type - use empty Par as placeholder
            space_types.push(Par::default());
        }
    }

    let new_env: BoundMapChain<VarSort> = input.bound_map_chain.put_all_pos(new_bindings);
    let new_count: usize = new_env.get_count() - input.bound_map_chain.get_count();

    let body_result = normalize_ann_proc(
        proc,
        ProcVisitInputs {
            par: Par::default(),
            bound_map_chain: new_env.clone(),
            free_map: current_free_map,
        },
        env,
        parser,
    )?;

    // TODO: we should build btree_map with real values, not a copied references from env: ref &HashMap
    let btree_map: BTreeMap<String, Par> =
        env.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

    let result_new = New {
        bind_count: new_count as i32,
        p: Some(body_result.par.clone()),
        uri: uris,
        injections: btree_map,
        locally_free: filter_and_adjust_bitset(body_result.par.clone().locally_free, new_count),
        space_types,
    };

    Ok(ProcVisitOutputs {
        par: prepend_new(input.par.clone(), result_new),
        free_map: body_result.free_map.clone(),
    })
}

// See rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use models::{
        create_bit_vector,
        rhoapi::{New, Par},
        rust::utils::{new_boundvar_par, new_gint_par, new_send},
    };

    use crate::rust::interpreter::{
        test_utils::utils::proc_visit_inputs_and_env, util::prepend_new,
    };

    #[test]
    fn p_new_should_bind_new_variables() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;
        use rholang_parser::ast::{Id, NameDecl, SendType};
        use rholang_parser::SourcePos;

        let parser = rholang_parser::RholangParser::new();

        // Create declarations: x, y, z
        let decls = vec![
            NameDecl {
                id: Id {
                    name: "x",
                    pos: SourcePos { line: 0, col: 0 },
                },
                space_type: None,
                uri: None,
            },
            NameDecl {
                id: Id {
                    name: "y",
                    pos: SourcePos { line: 0, col: 0 },
                },
                space_type: None,
                uri: None,
            },
            NameDecl {
                id: Id {
                    name: "z",
                    pos: SourcePos { line: 0, col: 0 },
                },
                space_type: None,
                uri: None,
            },
        ];

        // Create sends: x!(7), y!(8), z!(9)
        let x_channel = ParBuilderUtil::create_ast_name_var("x");
        let send_x = ParBuilderUtil::create_ast_send(
            x_channel,
            SendType::Single,
            vec![ParBuilderUtil::create_ast_long_literal(7, &parser)],
            &parser,
        );

        let y_channel = ParBuilderUtil::create_ast_name_var("y");
        let send_y = ParBuilderUtil::create_ast_send(
            y_channel,
            SendType::Single,
            vec![ParBuilderUtil::create_ast_long_literal(8, &parser)],
            &parser,
        );

        let z_channel = ParBuilderUtil::create_ast_name_var("z");
        let send_z = ParBuilderUtil::create_ast_send(
            z_channel,
            SendType::Single,
            vec![ParBuilderUtil::create_ast_long_literal(9, &parser)],
            &parser,
        );

        // Create Par: (x!(7) | y!(8)) | z!(9)
        let par_xy = ParBuilderUtil::create_ast_par(send_x, send_y, &parser);
        let par_xyz = ParBuilderUtil::create_ast_par(par_xy, send_z, &parser);

        // Create new x, y, z in { ... }
        let p_new = ParBuilderUtil::create_ast_new_with_decls(decls, par_xyz, &parser);

        let result = normalize_ann_proc(
            &p_new,
            proc_visit_inputs_and_env().0,
            &proc_visit_inputs_and_env().1,
            &parser,
        );
        assert!(result.is_ok());

        let expected_result = prepend_new(
            Par::default(),
            New {
                bind_count: 3,
                p: Some(
                    Par::default()
                        .prepend_send(new_send(
                            new_boundvar_par(2, create_bit_vector(&vec![2]), false),
                            vec![new_gint_par(7, Vec::new(), false)],
                            false,
                            create_bit_vector(&vec![2]),
                            false,
                        ))
                        .prepend_send(new_send(
                            new_boundvar_par(1, create_bit_vector(&vec![1]), false),
                            vec![new_gint_par(8, Vec::new(), false)],
                            false,
                            create_bit_vector(&vec![1]),
                            false,
                        ))
                        .prepend_send(new_send(
                            new_boundvar_par(0, create_bit_vector(&vec![0]), false),
                            vec![new_gint_par(9, Vec::new(), false)],
                            false,
                            create_bit_vector(&vec![0]),
                            false,
                        )),
                ),
                uri: Vec::new(),
                injections: BTreeMap::new(),
                locally_free: Vec::new(),
                space_types: vec![Par::default(), Par::default(), Par::default()],
            },
        );

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(
            result.unwrap().free_map,
            proc_visit_inputs_and_env().0.free_map
        );
    }

    #[test]
    fn p_new_should_sort_uris_and_place_them_at_the_end() {
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use crate::rust::interpreter::test_utils::par_builder_util::ParBuilderUtil;
        use rholang_parser::ast::{Id, NameDecl, SendType, Uri};
        use rholang_parser::SourcePos;

        let parser = rholang_parser::RholangParser::new();

        // Create declarations: x, y, r (with URI), out (with URI), z
        let decls = vec![
            NameDecl {
                id: Id {
                    name: "x",
                    pos: SourcePos { line: 0, col: 0 },
                },
                space_type: None,
                uri: None,
            },
            NameDecl {
                id: Id {
                    name: "y",
                    pos: SourcePos { line: 0, col: 0 },
                },
                space_type: None,
                uri: None,
            },
            NameDecl {
                id: Id {
                    name: "r",
                    pos: SourcePos { line: 0, col: 0 },
                },
                space_type: None,
                uri: Some(Uri::from("rho:registry")),
            },
            NameDecl {
                id: Id {
                    name: "out",
                    pos: SourcePos { line: 0, col: 0 },
                },
                space_type: None,
                uri: Some(Uri::from("rho:stdout")),
            },
            NameDecl {
                id: Id {
                    name: "z",
                    pos: SourcePos { line: 0, col: 0 },
                },
                space_type: None,
                uri: None,
            },
        ];

        // Create sends: x!(7), y!(8), r!(9), out!(10), z!(11)
        let send_x = ParBuilderUtil::create_ast_send(
            ParBuilderUtil::create_ast_name_var("x"),
            SendType::Single,
            vec![ParBuilderUtil::create_ast_long_literal(7, &parser)],
            &parser,
        );

        let send_y = ParBuilderUtil::create_ast_send(
            ParBuilderUtil::create_ast_name_var("y"),
            SendType::Single,
            vec![ParBuilderUtil::create_ast_long_literal(8, &parser)],
            &parser,
        );

        let send_r = ParBuilderUtil::create_ast_send(
            ParBuilderUtil::create_ast_name_var("r"),
            SendType::Single,
            vec![ParBuilderUtil::create_ast_long_literal(9, &parser)],
            &parser,
        );

        let send_out = ParBuilderUtil::create_ast_send(
            ParBuilderUtil::create_ast_name_var("out"),
            SendType::Single,
            vec![ParBuilderUtil::create_ast_long_literal(10, &parser)],
            &parser,
        );

        let send_z = ParBuilderUtil::create_ast_send(
            ParBuilderUtil::create_ast_name_var("z"),
            SendType::Single,
            vec![ParBuilderUtil::create_ast_long_literal(11, &parser)],
            &parser,
        );

        // Create nested Par: ((((x!(7) | y!(8)) | r!(9)) | out!(10)) | z!(11))
        let par_xy = ParBuilderUtil::create_ast_par(send_x, send_y, &parser);
        let par_xyr = ParBuilderUtil::create_ast_par(par_xy, send_r, &parser);
        let par_xyro = ParBuilderUtil::create_ast_par(par_xyr, send_out, &parser);
        let par_xyroz = ParBuilderUtil::create_ast_par(par_xyro, send_z, &parser);

        // Create new x, y, r, out, z in { ... }
        let p_new = ParBuilderUtil::create_ast_new_with_decls(decls, par_xyroz, &parser);

        let result = normalize_ann_proc(
            &p_new,
            proc_visit_inputs_and_env().0,
            &proc_visit_inputs_and_env().1,
            &parser,
        );
        assert!(result.is_ok());

        let expected_result = prepend_new(
            Par::default(),
            New {
                bind_count: 5,
                p: Some(
                    Par::default()
                        .prepend_send(new_send(
                            new_boundvar_par(4, create_bit_vector(&vec![4]), false),
                            vec![new_gint_par(7, Vec::new(), false)],
                            false,
                            create_bit_vector(&vec![4]),
                            false,
                        ))
                        .prepend_send(new_send(
                            new_boundvar_par(3, create_bit_vector(&vec![3]), false),
                            vec![new_gint_par(8, Vec::new(), false)],
                            false,
                            create_bit_vector(&vec![3]),
                            false,
                        ))
                        .prepend_send(new_send(
                            new_boundvar_par(1, create_bit_vector(&vec![1]), false),
                            vec![new_gint_par(9, Vec::new(), false)],
                            false,
                            create_bit_vector(&vec![1]),
                            false,
                        ))
                        .prepend_send(new_send(
                            new_boundvar_par(0, create_bit_vector(&vec![0]), false),
                            vec![new_gint_par(10, Vec::new(), false)],
                            false,
                            create_bit_vector(&vec![0]),
                            false,
                        ))
                        .prepend_send(new_send(
                            new_boundvar_par(2, create_bit_vector(&vec![2]), false),
                            vec![new_gint_par(11, Vec::new(), false)],
                            false,
                            create_bit_vector(&vec![2]),
                            false,
                        )),
                ),
                uri: vec!["rho:registry".to_string(), "rho:stdout".to_string()],
                injections: BTreeMap::new(),
                locally_free: Vec::new(),
                space_types: vec![Par::default(), Par::default(), Par::default(), Par::default(), Par::default()],
            },
        );

        assert_eq!(result.clone().unwrap().par, expected_result);
        assert_eq!(
            result.clone().unwrap().par.news[0]
                .p
                .clone()
                .unwrap()
                .sends
                .into_iter()
                .map(|x| x.locally_free)
                .collect::<Vec<Vec<u8>>>(),
            vec![
                create_bit_vector(&vec![2]),
                create_bit_vector(&vec![0]),
                create_bit_vector(&vec![1]),
                create_bit_vector(&vec![3]),
                create_bit_vector(&vec![4])
            ]
        );
        assert_eq!(
            result.unwrap().par.news[0].p.clone().unwrap().locally_free,
            create_bit_vector(&vec![0, 1, 2, 3, 4])
        );
    }
}
