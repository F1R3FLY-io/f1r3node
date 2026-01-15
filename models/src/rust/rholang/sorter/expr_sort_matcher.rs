// See models/src/main/scala/coop/rchain/models/rholang/sorter/ExprSortMatcher.scala

use crate::{
    rhoapi::{
        expr::ExprInstance, EAnd, EDiv, EEq, EFree, EFunction, EGt, EGte, EList, ELt, ELte, EMatches, EMinus,
        EMinusMinus, EMod, EMult, ENeg, ENeq, ENot, EOr, EPathMap, EPercentPercent, EPlus, EPlusPlus, EVar,
        EZipper, Expr, Par, Var,
    },
    rust::{
        par_map::ParMap,
        par_map_type_mapper::ParMapTypeMapper,
        par_set::ParSet,
        par_set_type_mapper::ParSetTypeMapper,
        rholang::sorter::{
            par_sort_matcher::ParSortMatcher,
            score_tree::{Score, ScoreAtom, Tree},
            var_sort_matcher::VarSortMatcher,
        },
        sorted_par_hash_set::SortedParHashSet,
    },
};

use super::{score_tree::ScoredTerm, sortable::Sortable};

pub struct ExprSortMatcher;

impl Sortable<Expr> for ExprSortMatcher {
    fn sort_match(e: &Expr) -> ScoredTerm<Expr> {
        fn construct_expr(expr_instance: ExprInstance, score: Tree<ScoreAtom>) -> ScoredTerm<Expr> {
            ScoredTerm {
                term: Expr {
                    expr_instance: Some(expr_instance),
                },
                score,
            }
        }

        fn remainder_score(remainder: &Option<Var>) -> Tree<ScoreAtom> {
            match remainder {
                Some(_var) => VarSortMatcher::sort_match(_var).score,
                None => Tree::<ScoreAtom>::create_leaf_from_i64(-1),
            }
        }

        match &e.expr_instance {
            Some(expr) => match expr {
                ExprInstance::ENegBody(en) => {
                    let sorted_par = ParSortMatcher::sort_match(
                        &en.p.as_ref().expect("par field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::ENegBody(ENeg {
                            p: Some(sorted_par.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::ENEG,
                            vec![sorted_par.score],
                        ),
                    )
                }

                ExprInstance::EVarBody(ev) => {
                    let sorted_var = VarSortMatcher::sort_match(
                        &ev.v.as_ref().expect("var field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::EVarBody(EVar {
                            v: Some(sorted_var.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::EVAR,
                            vec![sorted_var.score],
                        ),
                    )
                }

                ExprInstance::ENotBody(en) => {
                    let sorted_par = ParSortMatcher::sort_match(
                        &en.p.as_ref().expect("par field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::ENotBody(ENot {
                            p: Some(sorted_par.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::ENOT,
                            vec![sorted_par.score],
                        ),
                    )
                }

                ExprInstance::EMultBody(em) => {
                    let sorted_par1 = ParSortMatcher::sort_match(
                        &em.p1.as_ref().expect("par field was None, should be Some"),
                    );
                    let sorted_par2 = ParSortMatcher::sort_match(
                        &em.p2.as_ref().expect("par field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::EMultBody(EMult {
                            p1: Some(sorted_par1.term),
                            p2: Some(sorted_par2.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::EMULT,
                            vec![sorted_par1.score, sorted_par2.score],
                        ),
                    )
                }

                ExprInstance::EDivBody(ed) => {
                    let sorted_par1 = ParSortMatcher::sort_match(
                        &ed.p1.as_ref().expect("par field was None, should be Some"),
                    );
                    let sorted_par2 = ParSortMatcher::sort_match(
                        &ed.p2.as_ref().expect("par field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::EDivBody(EDiv {
                            p1: Some(sorted_par1.term),
                            p2: Some(sorted_par2.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::EDIV,
                            vec![sorted_par1.score, sorted_par2.score],
                        ),
                    )
                }

                ExprInstance::EModBody(ed) => {
                    let sorted_par1 = ParSortMatcher::sort_match(
                        &ed.p1.as_ref().expect("par field was None, should be Some"),
                    );
                    let sorted_par2 = ParSortMatcher::sort_match(
                        &ed.p2.as_ref().expect("par field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::EModBody(EMod {
                            p1: Some(sorted_par1.term),
                            p2: Some(sorted_par2.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::EMOD,
                            vec![sorted_par1.score, sorted_par2.score],
                        ),
                    )
                }

                ExprInstance::EPlusBody(ep) => {
                    let sorted_par1 = ParSortMatcher::sort_match(
                        &ep.p1.as_ref().expect("par field was None, should be Some"),
                    );
                    let sorted_par2 = ParSortMatcher::sort_match(
                        &ep.p2.as_ref().expect("par field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::EPlusBody(EPlus {
                            p1: Some(sorted_par1.term),
                            p2: Some(sorted_par2.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::EPLUS,
                            vec![sorted_par1.score, sorted_par2.score],
                        ),
                    )
                }

                ExprInstance::EMinusBody(em) => {
                    let sorted_par1 = ParSortMatcher::sort_match(
                        &em.p1.as_ref().expect("par field was None, should be Some"),
                    );
                    let sorted_par2 = ParSortMatcher::sort_match(
                        &em.p2.as_ref().expect("par field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::EMinusBody(EMinus {
                            p1: Some(sorted_par1.term),
                            p2: Some(sorted_par2.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::EMINUS,
                            vec![sorted_par1.score, sorted_par2.score],
                        ),
                    )
                }

                ExprInstance::ELtBody(el) => {
                    let sorted_par1 = ParSortMatcher::sort_match(
                        &el.p1.as_ref().expect("par field was None, should be Some"),
                    );
                    let sorted_par2 = ParSortMatcher::sort_match(
                        &el.p2.as_ref().expect("par field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::ELtBody(ELt {
                            p1: Some(sorted_par1.term),
                            p2: Some(sorted_par2.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::ELT,
                            vec![sorted_par1.score, sorted_par2.score],
                        ),
                    )
                }

                ExprInstance::ELteBody(el) => {
                    let sorted_par1 = ParSortMatcher::sort_match(
                        &el.p1.as_ref().expect("par field was None, should be Some"),
                    );
                    let sorted_par2 = ParSortMatcher::sort_match(
                        &el.p2.as_ref().expect("par field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::ELteBody(ELte {
                            p1: Some(sorted_par1.term),
                            p2: Some(sorted_par2.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::ELTE,
                            vec![sorted_par1.score, sorted_par2.score],
                        ),
                    )
                }

                ExprInstance::EGtBody(eg) => {
                    let sorted_par1 = ParSortMatcher::sort_match(
                        &eg.p1.as_ref().expect("par field was None, should be Some"),
                    );
                    let sorted_par2 = ParSortMatcher::sort_match(
                        &eg.p2.as_ref().expect("par field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::EGtBody(EGt {
                            p1: Some(sorted_par1.term),
                            p2: Some(sorted_par2.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::EGT,
                            vec![sorted_par1.score, sorted_par2.score],
                        ),
                    )
                }

                ExprInstance::EGteBody(eg) => {
                    let sorted_par1 = ParSortMatcher::sort_match(
                        &eg.p1.as_ref().expect("par field was None, should be Some"),
                    );
                    let sorted_par2 = ParSortMatcher::sort_match(
                        &eg.p2.as_ref().expect("par field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::EGteBody(EGte {
                            p1: Some(sorted_par1.term),
                            p2: Some(sorted_par2.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::EGTE,
                            vec![sorted_par1.score, sorted_par2.score],
                        ),
                    )
                }

                ExprInstance::EEqBody(ee) => {
                    let sorted_par1 = ParSortMatcher::sort_match(
                        &ee.p1.as_ref().expect("par field was None, should be Some"),
                    );
                    let sorted_par2 = ParSortMatcher::sort_match(
                        &ee.p2.as_ref().expect("par field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::EEqBody(EEq {
                            p1: Some(sorted_par1.term),
                            p2: Some(sorted_par2.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::EEQ,
                            vec![sorted_par1.score, sorted_par2.score],
                        ),
                    )
                }

                ExprInstance::ENeqBody(en) => {
                    let sorted_par1 = ParSortMatcher::sort_match(
                        &en.p1.as_ref().expect("par field was None, should be Some"),
                    );
                    let sorted_par2 = ParSortMatcher::sort_match(
                        &en.p2.as_ref().expect("par field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::ENeqBody(ENeq {
                            p1: Some(sorted_par1.term),
                            p2: Some(sorted_par2.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::ENEQ,
                            vec![sorted_par1.score, sorted_par2.score],
                        ),
                    )
                }

                ExprInstance::EAndBody(ea) => {
                    let sorted_par1 = ParSortMatcher::sort_match(
                        &ea.p1.as_ref().expect("par field was None, should be Some"),
                    );
                    let sorted_par2 = ParSortMatcher::sort_match(
                        &ea.p2.as_ref().expect("par field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::EAndBody(EAnd {
                            p1: Some(sorted_par1.term),
                            p2: Some(sorted_par2.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::EAND,
                            vec![sorted_par1.score, sorted_par2.score],
                        ),
                    )
                }

                ExprInstance::EOrBody(eo) => {
                    let sorted_par1 = ParSortMatcher::sort_match(
                        &eo.p1.as_ref().expect("par field was None, should be Some"),
                    );
                    let sorted_par2 = ParSortMatcher::sort_match(
                        &eo.p2.as_ref().expect("par field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::EOrBody(EOr {
                            p1: Some(sorted_par1.term),
                            p2: Some(sorted_par2.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::EOR,
                            vec![sorted_par1.score, sorted_par2.score],
                        ),
                    )
                }

                ExprInstance::EMatchesBody(em) => {
                    let sorted_target = ParSortMatcher::sort_match(
                        &em.target
                            .as_ref()
                            .expect("target field was None, should be Some"),
                    );
                    let sorted_pattern = ParSortMatcher::sort_match(
                        &em.pattern
                            .as_ref()
                            .expect("pattern field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::EMatchesBody(EMatches {
                            target: Some(sorted_target.term),
                            pattern: Some(sorted_pattern.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::EMATCHES,
                            vec![sorted_target.score, sorted_pattern.score],
                        ),
                    )
                }

                ExprInstance::EPercentPercentBody(ep) => {
                    let sorted_par1 = ParSortMatcher::sort_match(
                        &ep.p1.as_ref().expect("par field was None, should be Some"),
                    );
                    let sorted_par2 = ParSortMatcher::sort_match(
                        &ep.p2.as_ref().expect("par field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::EPercentPercentBody(EPercentPercent {
                            p1: Some(sorted_par1.term),
                            p2: Some(sorted_par2.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::EPERCENT,
                            vec![sorted_par1.score, sorted_par2.score],
                        ),
                    )
                }

                ExprInstance::EPlusPlusBody(ep) => {
                    let sorted_par1 = ParSortMatcher::sort_match(
                        &ep.p1.as_ref().expect("par field was None, should be Some"),
                    );
                    let sorted_par2 = ParSortMatcher::sort_match(
                        &ep.p2.as_ref().expect("par field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::EPlusPlusBody(EPlusPlus {
                            p1: Some(sorted_par1.term),
                            p2: Some(sorted_par2.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::EPLUSPLUS,
                            vec![sorted_par1.score, sorted_par2.score],
                        ),
                    )
                }

                ExprInstance::EMinusMinusBody(em) => {
                    let sorted_par1 = ParSortMatcher::sort_match(
                        &em.p1.as_ref().expect("par field was None, should be Some"),
                    );
                    let sorted_par2 = ParSortMatcher::sort_match(
                        &em.p2.as_ref().expect("par field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::EMinusMinusBody(EMinusMinus {
                            p1: Some(sorted_par1.term),
                            p2: Some(sorted_par2.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::EMINUSMINUS,
                            vec![sorted_par1.score, sorted_par2.score],
                        ),
                    )
                }

                ExprInstance::EMapBody(emap) => {
                    let par_map = ParMapTypeMapper::emap_to_par_map(emap.clone());

                    fn sort_key_value_pair(key: &Par, value: &Par) -> ScoredTerm<(Par, Par)> {
                        let sorted_key = ParSortMatcher::sort_match(key);
                        let sorted_value = ParSortMatcher::sort_match(value);

                        ScoredTerm {
                            term: (sorted_key.term, sorted_value.term),
                            score: sorted_key.score,
                        }
                    }

                    let sorted_pars: Vec<ScoredTerm<(Par, Par)>> = par_map
                        .ps
                        .sorted_list
                        .iter()
                        .map(|kv| sort_key_value_pair(&kv.0, &kv.1))
                        .collect();

                    let remainder_score = remainder_score(&par_map.remainder);
                    let connective_used_score: i64 = if par_map.connective_used { 1 } else { 0 };

                    construct_expr(
                        ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(ParMap::new(
                            sorted_pars.clone().into_iter().map(|p| p.term).collect(),
                            par_map.connective_used,
                            par_map.locally_free,
                            par_map.remainder,
                        ))),
                        Tree::Node(
                            vec![
                                Tree::<ScoreAtom>::create_leaf_from_i64(Score::EMAP as i64),
                                remainder_score,
                            ]
                            .into_iter()
                            .chain(sorted_pars.into_iter().map(|p| p.score))
                            .chain(vec![Tree::<ScoreAtom>::create_leaf_from_i64(
                                connective_used_score,
                            )])
                            .collect(),
                        ),
                    )
                }

                ExprInstance::ESetBody(eset) => {
                    let par_set = ParSetTypeMapper::eset_to_par_set(eset.clone());
                    let sorted_pars: Vec<ScoredTerm<Par>> = par_set
                        .ps
                        .sorted_pars
                        .iter()
                        .map(|p| ParSortMatcher::sort_match(p))
                        .collect();

                    let remainder_score = remainder_score(&par_set.remainder);
                    let connective_used_score: i64 = if par_set.connective_used { 1 } else { 0 };

                    construct_expr(
                        ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(ParSet {
                            ps: SortedParHashSet::create_from_vec(
                                sorted_pars.clone().into_iter().map(|p| p.term).collect(),
                            ),
                            connective_used: par_set.connective_used,
                            locally_free: par_set.locally_free,
                            remainder: par_set.remainder,
                        })),
                        Tree::Node(
                            vec![
                                Tree::<ScoreAtom>::create_leaf_from_i64(Score::ESET as i64),
                                remainder_score,
                            ]
                            .into_iter()
                            .chain(sorted_pars.into_iter().map(|p| p.score))
                            .chain(vec![Tree::<ScoreAtom>::create_leaf_from_i64(
                                connective_used_score,
                            )])
                            .collect(),
                        ),
                    )
                }

                ExprInstance::EPathmapBody(pathmap) => {
                    // Similar to EListBody - sort all Par elements in the pathmap
                    let pars: Vec<ScoredTerm<Par>> = pathmap
                        .ps
                        .iter()
                        .map(|p| ParSortMatcher::sort_match(p))
                        .collect();

                    let remainder_score = remainder_score(&pathmap.remainder);
                    let connective_used_score: i64 = if pathmap.connective_used { 1 } else { 0 };

                    construct_expr(
                        ExprInstance::EPathmapBody(EPathMap {
                            ps: pars.clone().into_iter().map(|p| p.term).collect(),
                            locally_free: pathmap.locally_free.clone(),
                            connective_used: pathmap.connective_used,
                            remainder: pathmap.remainder.clone(),
                        }),
                        Tree::Node(
                            vec![
                                Tree::<ScoreAtom>::create_leaf_from_i64(Score::EPATHMAP as i64),
                                remainder_score,
                            ]
                            .into_iter()
                            .chain(pars.into_iter().map(|p| p.score))
                            .chain(vec![Tree::<ScoreAtom>::create_leaf_from_i64(
                                connective_used_score,
                            )])
                            .collect(),
                        ),
                    )
                }

                ExprInstance::EListBody(list) => {
                    let pars: Vec<ScoredTerm<Par>> = list
                        .ps
                        .iter()
                        .map(|p| ParSortMatcher::sort_match(p))
                        .collect();

                    let remainder_score = remainder_score(&list.remainder);
                    let connective_used_score: i64 = if list.connective_used { 1 } else { 0 };

                    construct_expr(
                        ExprInstance::EListBody(EList {
                            ps: pars.clone().into_iter().map(|p| p.term).collect(),
                            locally_free: list.locally_free.clone(),
                            connective_used: list.connective_used,
                            remainder: list.remainder.clone(),
                        }),
                        Tree::Node(
                            vec![
                                Tree::<ScoreAtom>::create_leaf_from_i64(Score::ELIST as i64),
                                remainder_score,
                            ]
                            .into_iter()
                            .chain(pars.into_iter().map(|p| p.score))
                            .chain(vec![Tree::<ScoreAtom>::create_leaf_from_i64(
                                connective_used_score,
                            )])
                            .collect(),
                        ),
                    )
                }

                ExprInstance::EZipperBody(zipper) => {
                    // Sort the zipper's PathMap and maintain zipper metadata
                    let pathmap = zipper.pathmap.as_ref().expect("zipper pathmap was None");
                    let pars: Vec<ScoredTerm<Par>> = pathmap
                        .ps
                        .iter()
                        .map(|p| ParSortMatcher::sort_match(p))
                        .collect();

                    let connective_used_score: i64 = if zipper.connective_used { 1 } else { 0 };

                    construct_expr(
                        ExprInstance::EZipperBody(EZipper {
                            pathmap: Some(EPathMap {
                                ps: pars.clone().into_iter().map(|p| p.term).collect(),
                                locally_free: pathmap.locally_free.clone(),
                                connective_used: pathmap.connective_used,
                                remainder: pathmap.remainder.clone(),
                            }),
                            current_path: zipper.current_path.clone(),
                            is_write_zipper: zipper.is_write_zipper,
                            locally_free: zipper.locally_free.clone(),
                            connective_used: zipper.connective_used,
                        }),
                        Tree::Node(
                            vec![
                                Tree::<ScoreAtom>::create_leaf_from_i64(Score::EPATHMAP as i64 + 1), // Use EPATHMAP + 1 for zipper
                            ]
                            .into_iter()
                            .chain(pars.into_iter().map(|p| p.score))
                            .chain(vec![Tree::<ScoreAtom>::create_leaf_from_i64(
                                connective_used_score,
                            )])
                            .collect(),
                        ),
                    )
                }

                ExprInstance::ETupleBody(tuple) => {
                    let sorted_pars: Vec<ScoredTerm<Par>> = tuple
                        .ps
                        .iter()
                        .map(|p| ParSortMatcher::sort_match(p))
                        .collect();

                    let connective_used_score: i64 = if tuple.connective_used { 1 } else { 0 };
                    let mut tuple_cloned = tuple.clone();
                    tuple_cloned.ps = sorted_pars.clone().into_iter().map(|p| p.term).collect();

                    construct_expr(
                        ExprInstance::ETupleBody(tuple_cloned),
                        Tree::Node(
                            vec![Tree::<ScoreAtom>::create_leaf_from_i64(
                                Score::ETUPLE as i64,
                            )]
                            .into_iter()
                            .chain(sorted_pars.into_iter().map(|p| p.score))
                            .chain(vec![Tree::<ScoreAtom>::create_leaf_from_i64(
                                connective_used_score,
                            )])
                            .collect(),
                        ),
                    )
                }

                ExprInstance::EMethodBody(em) => {
                    let args: Vec<ScoredTerm<Par>> = em
                        .arguments
                        .iter()
                        .map(|p| ParSortMatcher::sort_match(p))
                        .collect();

                    let sorted_target = ParSortMatcher::sort_match(
                        &em.target
                            .as_ref()
                            .expect("target field on EMethod was None, should be Some"),
                    );
                    let connective_used_score: i64 = if em.connective_used { 1 } else { 0 };

                    let mut em_cloned = em.clone();
                    em_cloned.arguments = args.clone().into_iter().map(|p| p.term).collect();
                    em_cloned.target = Some(sorted_target.term);

                    construct_expr(
                        ExprInstance::EMethodBody(em_cloned),
                        Tree::Node(
                            vec![
                                Tree::<ScoreAtom>::create_leaf_from_i64(Score::EMETHOD as i64),
                                Tree::<ScoreAtom>::create_leaf_from_string(em.method_name.clone()),
                                sorted_target.score,
                            ]
                            .into_iter()
                            .chain(args.into_iter().map(|p| p.score))
                            .chain(vec![Tree::<ScoreAtom>::create_leaf_from_i64(
                                connective_used_score,
                            )])
                            .collect(),
                        ),
                    )
                }

                ExprInstance::GBool(gb) => {
                    // See models/src/main/scala/coop/rchain/models/rholang/sorter/BoolSortMatcher.scala
                    let sorted = if *gb {
                        ScoredTerm {
                            term: gb,
                            score: Tree::<ScoreAtom>::create_node_from_i64s(vec![
                                Score::BOOL as i64,
                                0,
                            ]),
                        }
                    } else {
                        ScoredTerm {
                            term: gb,
                            score: Tree::<ScoreAtom>::create_node_from_i64s(vec![
                                Score::BOOL as i64,
                                1,
                            ]),
                        }
                    };

                    ScoredTerm {
                        term: e.clone(),
                        score: sorted.score,
                    }
                }

                ExprInstance::GInt(gi) => ScoredTerm {
                    term: e.clone(),
                    score: Tree::<ScoreAtom>::create_node_from_i64s(vec![Score::INT as i64, *gi]),
                },

                ExprInstance::GString(gs) => ScoredTerm {
                    term: e.clone(),
                    score: Tree::<ScoreAtom>::create_node_from_i32(
                        Score::STRING,
                        vec![Tree::<ScoreAtom>::create_leaf_from_string(gs.clone())],
                    ),
                },

                ExprInstance::GUri(gu) => ScoredTerm {
                    term: e.clone(),
                    score: Tree::<ScoreAtom>::create_node_from_i32(
                        Score::URI,
                        vec![Tree::<ScoreAtom>::create_leaf_from_string(gu.clone())],
                    ),
                },

                ExprInstance::GByteArray(ba) => ScoredTerm {
                    term: e.clone(),
                    score: Tree::<ScoreAtom>::create_node_from_i32(
                        Score::EBYTEARR,
                        vec![Tree::<ScoreAtom>::create_leaf_from_bytes(ba.clone())],
                    ),
                },

                ExprInstance::EFreeBody(ef) => {
                    let sorted_par = ParSortMatcher::sort_match(
                        &ef.body.as_ref().expect("body field was None, should be Some"),
                    );

                    construct_expr(
                        ExprInstance::EFreeBody(EFree {
                            body: Some(sorted_par.term),
                        }),
                        Tree::<ScoreAtom>::create_node_from_i32(
                            Score::EFREE,
                            vec![sorted_par.score],
                        ),
                    )
                }

                ExprInstance::EFunctionBody(ef) => {
                    // Sort arguments (similar to EMethodBody pattern)
                    let args: Vec<ScoredTerm<Par>> = ef
                        .arguments
                        .iter()
                        .map(|p| ParSortMatcher::sort_match(p))
                        .collect();

                    let connective_used_score: i64 = if ef.connective_used { 1 } else { 0 };

                    let mut ef_cloned = ef.clone();
                    ef_cloned.arguments = args.clone().into_iter().map(|p| p.term).collect();

                    construct_expr(
                        ExprInstance::EFunctionBody(ef_cloned),
                        Tree::Node(
                            vec![
                                Tree::<ScoreAtom>::create_leaf_from_i64(Score::EFUNCTION as i64),
                                Tree::<ScoreAtom>::create_leaf_from_string(ef.function_name.clone()),
                            ]
                            .into_iter()
                            .chain(args.into_iter().map(|p| p.score))
                            .chain(vec![Tree::<ScoreAtom>::create_leaf_from_i64(
                                connective_used_score,
                            )])
                            .collect(),
                        ),
                    )
                }
            },

            // TODO get rid of Empty nodes in Protobuf unless they represent sth indeed optional - OLD
            None => ScoredTerm {
                term: e.clone(),
                score: Tree::<ScoreAtom>::create_node_from_i32(Score::ABSENT, Vec::new()),
            },
        }
    }
}
