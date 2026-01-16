use models::rhoapi::connective::ConnectiveInstance;
use models::rhoapi::expr::ExprInstance;
use models::rhoapi::var::VarInstance;
use models::rhoapi::{
    Bundle, Connective, ConnectiveBody, EAnd, EDiv, EEq, EGt, EGte, EList, ELt, ELte, EMatches,
    EMethod, EMinus, EMinusMinus, EMod, EMult, ENeg, ENeq, ENot, EOr, EPercentPercent, EPlus,
    EPlusPlus, ETuple, EVar, Expr, Match, MatchCase, New, Par, Receive, ReceiveBind, Send,
    UseBlock, Var, VarRef,
};
use models::rust::bundle_ops::BundleOps;
use models::rust::par_map::ParMap;
use models::rust::par_map_type_mapper::ParMapTypeMapper;
use models::rust::par_set::ParSet;
use models::rust::par_set_type_mapper::ParSetTypeMapper;
use models::rust::rholang::implicits::{concatenate_pars, single_bundle};
use models::rust::rholang::sorter::match_sort_matcher::MatchSortMatcher;
use models::rust::rholang::sorter::new_sort_matcher::NewSortMatcher;
use models::rust::rholang::sorter::par_sort_matcher::ParSortMatcher;
use models::rust::rholang::sorter::receive_sort_matcher::ReceiveSortMatcher;
use models::rust::rholang::sorter::send_sort_matcher::SendSortMatcher;
use models::rust::rholang::sorter::sortable::Sortable;
use models::rust::sorted_par_hash_set::SortedParHashSet;
use models::rust::sorted_par_map::SortedParMap;
use rspace_plus_plus::rspace::history::Either;

use super::accounting::_cost;
use super::accounting::costs::Cost;
use super::env::Env;
use super::errors::InterpreterError;
use super::unwrap_option_safe;
use super::util::{prepend_connective, prepend_expr};

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/Substitute.scala
pub trait SubstituteTrait<A> {
    fn substitute(&self, term: A, depth: i32, env: &Env<Par>) -> Result<A, InterpreterError>;

    fn substitute_no_sort(
        &self,
        term: A,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<A, InterpreterError>;
}

#[derive(Clone)]
pub struct Substitute {
    pub cost: _cost,
}

impl Substitute {
    pub fn substitute_and_charge<A>(
        &self,
        term: &A,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<A, InterpreterError>
    where
        Self: SubstituteTrait<A>,
        A: Clone + prost::Message,
    {
        // scala 'charge' function built in here
        match self.substitute(term.clone(), depth, env) {
            Ok(subst_term) => {
                self.cost.charge(Cost::create_from_generic(
                    subst_term.clone(),
                    "substitution".to_string(),
                ))?;
                Ok(subst_term)
            }
            Err(th) => {
                self.cost
                    .charge(Cost::create_from_generic(term.clone(), "".to_string()))?;
                Err(th)
            }
        }
    }

    pub fn substitute_no_sort_and_charge<A>(
        &self,
        term: &A,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<A, InterpreterError>
    where
        Self: SubstituteTrait<A>,
        A: Clone + prost::Message,
    {
        // scala 'charge' function built in here
        match self.substitute_no_sort(term.clone(), depth, env) {
            Ok(subst_term) => {
                self.cost.charge(Cost::create_from_generic(
                    subst_term.clone(),
                    "substitution".to_string(),
                ))?;
                Ok(subst_term)
            }
            Err(th) => {
                self.cost
                    .charge(Cost::create_from_generic(term.clone(), "".to_string()))?;
                Err(th)
            }
        }
    }

    // pub here for testing purposes
    pub fn maybe_substitute_var(
        &self,
        term: Var,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<Either<Var, Par>, InterpreterError> {
        // println!("\nenv in maybe_substitute_var: {:?}", env);
        if depth != 0 {
            Ok(Either::Left(term))
        } else {
            match unwrap_option_safe(term.clone().var_instance)? {
                VarInstance::BoundVar(index) => {
                    // println!("\nindex in maybe_substitute_var: {:?}", index);
                    match env.get(&index) {
                        Some(p) => {
                            // println!("\np in maybe_substitute_var: {:?}", p);
                            Ok(Either::Right(p))
                        }
                        None => Ok(Either::Left(term)),
                    }
                }
                _ => Err(InterpreterError::SubstituteError(format!(
                    "Illegal Substitution [{:?}]",
                    term
                ))),
            }
        }
    }

    fn maybe_substitute_evar(
        &self,
        term: EVar,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<Either<EVar, Par>, InterpreterError> {
        match self.maybe_substitute_var(unwrap_option_safe(term.v)?, depth, env)? {
            Either::Left(v) => Ok(Either::Left(EVar { v: Some(v) })),
            Either::Right(p) => Ok(Either::Right(p)),
        }
    }

    fn maybe_substitute_var_ref(
        &self,
        term: VarRef,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<Either<VarRef, Par>, InterpreterError> {
        if term.depth != depth {
            Ok(Either::Left(term))
        } else {
            match env.get(&term.index) {
                Some(p) => Ok(Either::Right(p)),
                None => Ok(Either::Left(term)),
            }
        }
    }
}

impl SubstituteTrait<Bundle> for Substitute {
    fn substitute(
        &self,
        term: Bundle,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<Bundle, InterpreterError> {
        let sub_bundle = self.substitute(unwrap_option_safe(term.clone().body)?, depth, env)?;

        match single_bundle(&sub_bundle) {
            Some(b) => Ok(BundleOps::merge(&term, &b)),
            None => {
                let mut term_mut = term.clone();
                term_mut.body = Some(sub_bundle);
                Ok(term_mut)
            }
        }
    }

    fn substitute_no_sort(
        &self,
        term: Bundle,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<Bundle, InterpreterError> {
        let sub_bundle =
            self.substitute_no_sort(unwrap_option_safe(term.clone().body)?, depth, env)?;

        match single_bundle(&sub_bundle) {
            Some(b) => Ok(BundleOps::merge(&term, &b)),
            None => {
                let mut term_mut = term.clone();
                term_mut.body = Some(sub_bundle);
                Ok(term_mut)
            }
        }
    }
}

impl Substitute {
    fn sub_exp(
        &self,
        exprs: Vec<Expr>,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<Par, InterpreterError> {
        exprs.into_iter().try_fold(Par::default(), |par, expr| {
            match unwrap_option_safe(expr.clone().expr_instance)? {
                ExprInstance::EVarBody(e) => match self.maybe_substitute_evar(e, depth, env)? {
                    Either::Left(_e) => {
                        // println!("\npar in sub_expr: {:?}", par);
                        Ok(prepend_expr(
                            par,
                            Expr {
                                expr_instance: Some(ExprInstance::EVarBody(_e)),
                            },
                            depth,
                        ))
                    }
                    Either::Right(_par) => Ok(concatenate_pars(_par, par)),
                },
                _ => match self.substitute_no_sort(expr, depth, env) {
                    Ok(e) => Ok(prepend_expr(par, e, depth)),
                    Err(e) => Err(e),
                },
            }
        })
    }

    fn sub_conn(
        &self,
        conns: Vec<Connective>,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<Par, InterpreterError> {
        conns
            .into_iter()
            .try_fold(Par::default(), |par, conn| match conn.connective_instance {
                Some(ref conn_instance) => match conn_instance {
                    ConnectiveInstance::VarRefBody(v) => {
                        match self.maybe_substitute_var_ref(v.clone(), depth, env)? {
                            Either::Left(_) => Ok(prepend_connective(par, conn, depth)),
                            Either::Right(new_par) => Ok(concatenate_pars(new_par, par)),
                        }
                    }

                    ConnectiveInstance::ConnAndBody(ConnectiveBody { ps }) => {
                        let _ps: Vec<Par> = ps
                            .iter()
                            .map(|p| self.substitute_no_sort(p.clone(), depth, env))
                            .collect::<Result<Vec<Par>, InterpreterError>>()?;

                        Ok(prepend_connective(
                            par,
                            Connective {
                                connective_instance: Some(ConnectiveInstance::ConnAndBody(
                                    ConnectiveBody { ps: ps.to_vec() },
                                )),
                            },
                            depth,
                        ))
                    }

                    ConnectiveInstance::ConnOrBody(ConnectiveBody { ps }) => {
                        let _ps: Vec<Par> = ps
                            .iter()
                            .map(|p| self.substitute_no_sort(p.clone(), depth, env))
                            .collect::<Result<Vec<Par>, InterpreterError>>()?;

                        Ok(prepend_connective(
                            par,
                            Connective {
                                connective_instance: Some(ConnectiveInstance::ConnOrBody(
                                    ConnectiveBody { ps: ps.to_vec() },
                                )),
                            },
                            depth,
                        ))
                    }

                    ConnectiveInstance::ConnNotBody(p) => {
                        self.substitute_no_sort(p.clone(), depth, env).map(|p| {
                            prepend_connective(
                                par,
                                Connective {
                                    connective_instance: Some(ConnectiveInstance::ConnNotBody(p)),
                                },
                                depth,
                            )
                        })
                    }

                    ConnectiveInstance::ConnBool(c) => Ok(prepend_connective(
                        par,
                        Connective {
                            connective_instance: Some(ConnectiveInstance::ConnBool(*c)),
                        },
                        depth,
                    )),
                    ConnectiveInstance::ConnInt(c) => Ok(prepend_connective(
                        par,
                        Connective {
                            connective_instance: Some(ConnectiveInstance::ConnInt(*c)),
                        },
                        depth,
                    )),
                    ConnectiveInstance::ConnString(c) => Ok(prepend_connective(
                        par,
                        Connective {
                            connective_instance: Some(ConnectiveInstance::ConnString(*c)),
                        },
                        depth,
                    )),
                    ConnectiveInstance::ConnUri(c) => Ok(prepend_connective(
                        par,
                        Connective {
                            connective_instance: Some(ConnectiveInstance::ConnUri(*c)),
                        },
                        depth,
                    )),
                    ConnectiveInstance::ConnByteArray(c) => Ok(prepend_connective(
                        par,
                        Connective {
                            connective_instance: Some(ConnectiveInstance::ConnByteArray(*c)),
                        },
                        depth,
                    )),
                },
                None => Ok(par),
            })
    }
}

impl SubstituteTrait<Par> for Substitute {
    fn substitute_no_sort(
        &self,
        term: Par,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<Par, InterpreterError> {
        // println!("\nterm in substitute_no_sort for par: {:?}", term);
        let exprs = self.sub_exp(term.exprs, depth, env)?;
        // println!("\nexprs in substitute_no_sort for par: {:?}", exprs);
        let connectives = self.sub_conn(term.connectives, depth, env)?;

        let sends = term
            .sends
            .iter()
            .map(|s| self.substitute_no_sort(s.clone(), depth, env))
            .collect::<Result<Vec<Send>, InterpreterError>>()?;

        let bundles = term
            .bundles
            .iter()
            .map(|b| self.substitute_no_sort(b.clone(), depth, env))
            .collect::<Result<Vec<Bundle>, InterpreterError>>()?;

        let receives = term
            .receives
            .iter()
            .map(|r| self.substitute_no_sort(r.clone(), depth, env))
            .collect::<Result<Vec<Receive>, InterpreterError>>()?;

        let news = term
            .news
            .iter()
            .map(|n| self.substitute_no_sort(n.clone(), depth, env))
            .collect::<Result<Vec<New>, InterpreterError>>()?;

        let matches = term
            .matches
            .iter()
            .map(|m| self.substitute_no_sort(m.clone(), depth, env))
            .collect::<Result<Vec<Match>, InterpreterError>>()?;

        Ok(concatenate_pars(
            exprs,
            concatenate_pars(
                connectives,
                Par {
                    sends,
                    receives,
                    news,
                    exprs: Vec::new(),
                    matches,
                    unforgeables: term.unforgeables,
                    bundles,
                    connectives: Vec::new(),
                    locally_free: {
                        // println!("\nenv.shift in substitute_no_sort for par: {}", env.shift);
                        set_bits_until(term.locally_free, env.shift)
                    },
                    connective_used: term.connective_used,
                    // Reifying RSpaces: Substitute variables in use_blocks
                    use_blocks: term
                        .use_blocks
                        .iter()
                        .map(|ub| self.substitute_no_sort(ub.clone(), depth, env))
                        .collect::<Result<Vec<UseBlock>, InterpreterError>>()?,
                },
            ),
        ))
    }

    fn substitute(&self, term: Par, depth: i32, env: &Env<Par>) -> Result<Par, InterpreterError> {
        self.substitute_no_sort(term, depth, env)
            .map(|p| ParSortMatcher::sort_match(&p))
            .map(|st| st.term)
    }
}

impl SubstituteTrait<Send> for Substitute {
    fn substitute_no_sort(
        &self,
        term: Send,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<Send, InterpreterError> {
        let channels_sub =
            self.substitute_no_sort(unwrap_option_safe(term.clone().chan)?, depth, env)?;

        let pars_sub = term
            .data
            .iter()
            .map(|p| self.substitute_no_sort(p.clone(), depth, env))
            .collect::<Result<Vec<Par>, InterpreterError>>()?;

        // println!("\nterm in substitute_no_sort for Send {:?}", term);

        // Substitute hyperparam expressions
        use models::rhoapi::hyperparam::HyperparamInstance;
        let hyperparams_sub: Vec<models::rhoapi::Hyperparam> = term
            .hyperparams
            .iter()
            .map(|hp| {
                match &hp.hyperparam_instance {
                    Some(HyperparamInstance::Positional(p)) => {
                        let subst_p = self.substitute_no_sort(p.clone(), depth, env)?;
                        Ok(models::rhoapi::Hyperparam {
                            hyperparam_instance: Some(HyperparamInstance::Positional(subst_p)),
                        })
                    }
                    Some(HyperparamInstance::Named(named)) => {
                        let subst_value = match &named.value {
                            Some(v) => Some(self.substitute_no_sort(v.clone(), depth, env)?),
                            None => None,
                        };
                        Ok(models::rhoapi::Hyperparam {
                            hyperparam_instance: Some(HyperparamInstance::Named(
                                models::rhoapi::NamedHyperparam {
                                    key: named.key.clone(),
                                    value: subst_value,
                                },
                            )),
                        })
                    }
                    None => Ok(hp.clone()),
                }
            })
            .collect::<Result<Vec<_>, InterpreterError>>()?;

        Ok(Send {
            chan: Some(channels_sub),
            data: pars_sub,
            persistent: term.persistent,
            locally_free: set_bits_until(term.locally_free, env.shift),
            connective_used: term.connective_used,
            // Reifying RSpaces: Preserve hyperparams during substitution
            hyperparams: hyperparams_sub,
        })
    }

    fn substitute(&self, term: Send, depth: i32, env: &Env<Par>) -> Result<Send, InterpreterError> {
        self.substitute_no_sort(term, depth, env)
            .map(|s| SendSortMatcher::sort_match(&s))
            .map(|st| st.term)
    }
}

/// Reifying RSpaces: Substitute variables in UseBlock constructs.
///
/// UseBlocks contain a space expression and a body, both of which may contain
/// bound variables that need substitution during receive continuation evaluation.
///
/// # Formal Correspondence
/// - Registry/Invariants.v: inv_use_blocks_valid - Use blocks maintain validity
/// - GenericRSpace.v: UseBlock scope semantics
impl SubstituteTrait<UseBlock> for Substitute {
    fn substitute_no_sort(
        &self,
        term: UseBlock,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<UseBlock, InterpreterError> {
        let space_sub = match term.space {
            Some(s) => Some(self.substitute_no_sort(s, depth, env)?),
            None => None,
        };
        let body_sub = match term.body {
            Some(b) => Some(self.substitute_no_sort(b, depth, env)?),
            None => None,
        };

        Ok(UseBlock {
            space: space_sub,
            body: body_sub,
            locally_free: set_bits_until(term.locally_free, env.shift),
            connective_used: term.connective_used,
        })
    }

    fn substitute(
        &self,
        term: UseBlock,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<UseBlock, InterpreterError> {
        self.substitute_no_sort(term, depth, env)
    }
}

impl SubstituteTrait<Receive> for Substitute {
    fn substitute_no_sort(
        &self,
        term: Receive,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<Receive, InterpreterError> {
        let binds_sub = term
            .binds
            .into_iter()
            .map(
                |ReceiveBind {
                     patterns,
                     source,
                     remainder,
                     free_count,
                     pattern_modifiers,
                 }| {
                    let sub_channel =
                        self.substitute_no_sort(unwrap_option_safe(source)?, depth, env)?;
                    let sub_patterns = patterns
                        .iter()
                        .map(|p| self.substitute_no_sort(p.clone(), depth + 1, env))
                        .collect::<Result<Vec<Par>, InterpreterError>>()?;

                    // Substitute within pattern modifiers (EFunction: sim, rank, etc.)
                    let sub_modifiers = pattern_modifiers
                        .into_iter()
                        .map(|efunc| {
                            let sub_args = efunc
                                .arguments
                                .into_iter()
                                .map(|arg| self.substitute_no_sort(arg, depth, env))
                                .collect::<Result<Vec<_>, _>>()?;
                            Ok(models::rhoapi::EFunction {
                                function_name: efunc.function_name,
                                arguments: sub_args,
                                locally_free: efunc.locally_free,
                                connective_used: efunc.connective_used,
                            })
                        })
                        .collect::<Result<Vec<_>, InterpreterError>>()?;

                    Ok(ReceiveBind {
                        patterns: sub_patterns,
                        source: Some(sub_channel),
                        remainder,
                        free_count,
                        pattern_modifiers: sub_modifiers,
                    })
                },
            )
            .collect::<Result<Vec<ReceiveBind>, InterpreterError>>()?;

        let body_sub = self.substitute_no_sort(
            unwrap_option_safe(term.body)?,
            depth,
            &env.shift(term.bind_count),
        )?;

        Ok(Receive {
            binds: binds_sub,
            body: Some(body_sub),
            persistent: term.persistent,
            peek: term.peek,
            bind_count: term.bind_count,
            locally_free: set_bits_until(term.locally_free, env.shift),
            connective_used: term.connective_used,
        })
    }

    fn substitute(
        &self,
        term: Receive,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<Receive, InterpreterError> {
        self.substitute_no_sort(term, depth, env)
            .map(|r| ReceiveSortMatcher::sort_match(&r))
            .map(|st| st.term)
    }
}

impl SubstituteTrait<New> for Substitute {
    fn substitute_no_sort(
        &self,
        term: New,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<New, InterpreterError> {
        self.substitute_no_sort(
            unwrap_option_safe(term.p)?,
            depth,
            &env.shift(term.bind_count),
        )
        .map(|new_sub| New {
            bind_count: term.bind_count,
            p: Some(new_sub),
            uri: term.uri,
            injections: term.injections,
            locally_free: set_bits_until(term.locally_free, env.shift),
            space_types: term.space_types.clone(),
        })
    }

    fn substitute(&self, term: New, depth: i32, env: &Env<Par>) -> Result<New, InterpreterError> {
        self.substitute_no_sort(term, depth, env)
            .map(|n| NewSortMatcher::sort_match(&n))
            .map(|st| st.term)
    }
}

impl SubstituteTrait<Match> for Substitute {
    fn substitute_no_sort(
        &self,
        term: Match,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<Match, InterpreterError> {
        let target_sub = self.substitute_no_sort(unwrap_option_safe(term.target)?, depth, env)?;

        let cases_sub = term
            .cases
            .iter()
            .filter(|case| case.pattern.is_some() && case.source.is_some())
            .map(
                |MatchCase {
                     pattern,
                     source,
                     free_count,
                 }| {
                    let par = self.substitute_no_sort(
                        unwrap_option_safe(source.clone())?,
                        depth,
                        &env.shift(*free_count),
                    )?;

                    let sub_case = self.substitute_no_sort(
                        unwrap_option_safe(pattern.clone())?,
                        depth + 1,
                        env,
                    )?;

                    Ok(MatchCase {
                        pattern: Some(sub_case),
                        source: Some(par),
                        free_count: *free_count,
                    })
                },
            )
            .collect::<Result<Vec<MatchCase>, InterpreterError>>()?;

        Ok(Match {
            target: Some(target_sub),
            cases: cases_sub,
            locally_free: set_bits_until(term.locally_free, env.shift),
            connective_used: term.connective_used,
        })
    }

    fn substitute(
        &self,
        term: Match,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<Match, InterpreterError> {
        self.substitute_no_sort(term, depth, env)
            .map(|m| MatchSortMatcher::sort_match(&m))
            .map(|st| st.term)
    }
}

impl SubstituteTrait<Expr> for Substitute {
    fn substitute(&self, term: Expr, depth: i32, env: &Env<Par>) -> Result<Expr, InterpreterError> {
        match unwrap_option_safe(term.expr_instance.clone())? {
            ExprInstance::ENotBody(ENot { p }) => self
                .substitute(unwrap_option_safe(p)?, depth, env)
                .map(|p| {
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::ENotBody(ENot { p: Some(p) })),
                    })
                })?,

            ExprInstance::ENegBody(ENeg { p }) => self
                .substitute(unwrap_option_safe(p)?, depth, env)
                .map(|p| {
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::ENegBody(ENeg { p: Some(p) })),
                    })
                })?,

            ExprInstance::EMultBody(EMult { p1, p2 }) => {
                let _p1 = self.substitute(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EMultBody(EMult {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EDivBody(EDiv { p1, p2 }) => {
                let _p1 = self.substitute(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EDivBody(EDiv {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EModBody(EMod { p1, p2 }) => {
                let _p1 = self.substitute(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EModBody(EMod {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EPercentPercentBody(EPercentPercent { p1, p2 }) => {
                let _p1 = self.substitute(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EPercentPercentBody(EPercentPercent {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EPlusBody(EPlus { p1, p2 }) => {
                let _p1 = self.substitute(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EPlusBody(EPlus {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EMinusBody(EMinus { p1, p2 }) => {
                let _p1 = self.substitute(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EPlusBody(EPlus {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EPlusPlusBody(EPlusPlus { p1, p2 }) => {
                let _p1 = self.substitute(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EPlusPlusBody(EPlusPlus {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EMinusMinusBody(EMinusMinus { p1, p2 }) => {
                let _p1 = self.substitute(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EMinusMinusBody(EMinusMinus {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::ELtBody(ELt { p1, p2 }) => {
                let _p1 = self.substitute(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::ELtBody(ELt {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::ELteBody(ELte { p1, p2 }) => {
                let _p1 = self.substitute(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::ELteBody(ELte {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EGtBody(EGt { p1, p2 }) => {
                let _p1 = self.substitute(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EGtBody(EGt {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EGteBody(EGte { p1, p2 }) => {
                let _p1 = self.substitute(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EGteBody(EGte {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EEqBody(EEq { p1, p2 }) => {
                let _p1 = self.substitute(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EEqBody(EEq {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::ENeqBody(ENeq { p1, p2 }) => {
                let _p1 = self.substitute(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::ENeqBody(ENeq {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EAndBody(EAnd { p1, p2 }) => {
                let _p1 = self.substitute(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EAndBody(EAnd {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EOrBody(EOr { p1, p2 }) => {
                let _p1 = self.substitute(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EOrBody(EOr {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EMatchesBody(EMatches { target, pattern }) => {
                let _target = self.substitute(unwrap_option_safe(target)?, depth, env)?;
                let _pattern = self.substitute(unwrap_option_safe(pattern)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EMatchesBody(EMatches {
                        target: Some(_target),
                        pattern: Some(_pattern),
                    })),
                })
            }

            ExprInstance::EListBody(EList {
                ps,
                locally_free,
                connective_used,
                remainder,
            }) => {
                let _ps = ps
                    .iter()
                    .map(|p| self.substitute(p.clone(), depth, env))
                    .collect::<Result<Vec<Par>, InterpreterError>>()?;

                let new_locally_free = set_bits_until(locally_free, env.shift);

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EListBody(EList {
                        ps: _ps,
                        locally_free: new_locally_free,
                        connective_used,
                        remainder,
                    })),
                })
            }

            ExprInstance::ETupleBody(ETuple {
                ps,
                locally_free,
                connective_used,
            }) => {
                let _ps = ps
                    .iter()
                    .map(|p| self.substitute(p.clone(), depth, env))
                    .collect::<Result<Vec<Par>, InterpreterError>>()?;

                let new_locally_free = set_bits_until(locally_free, env.shift);

                Ok(Expr {
                    expr_instance: Some(ExprInstance::ETupleBody(ETuple {
                        ps: _ps,
                        locally_free: new_locally_free,
                        connective_used,
                    })),
                })
            }

            ExprInstance::ESetBody(eset) => {
                let par_set = ParSetTypeMapper::eset_to_par_set(eset);
                let _ps = par_set
                    .ps
                    .sorted_pars
                    .iter()
                    .map(|p| self.substitute(p.clone(), depth, env))
                    .collect::<Result<Vec<Par>, InterpreterError>>()?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
                        ParSet {
                            ps: SortedParHashSet::create_from_vec(_ps),
                            connective_used: par_set.connective_used,
                            locally_free: set_bits_until(par_set.locally_free, env.shift),
                            remainder: par_set.remainder,
                        },
                    ))),
                })
            }

            ExprInstance::EMapBody(emap) => {
                let par_map = ParMapTypeMapper::emap_to_par_map(emap);
                let _ps = par_map
                    .ps
                    .sorted_list
                    .iter()
                    .map(|p| {
                        let p1 = self.substitute(p.0.clone(), depth, env)?;
                        let p2 = self.substitute(p.1.clone(), depth, env)?;
                        Ok((p1, p2))
                    })
                    .collect::<Result<Vec<(Par, Par)>, InterpreterError>>()?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
                        ParMap {
                            ps: SortedParMap::create_from_vec(_ps),
                            connective_used: par_map.connective_used,
                            locally_free: set_bits_until(par_map.locally_free, env.shift),
                            remainder: par_map.remainder,
                        },
                    ))),
                })
            }

            ExprInstance::EMethodBody(EMethod {
                method_name,
                target,
                arguments,
                locally_free,
                connective_used,
            }) => {
                let sub_target = self.substitute(unwrap_option_safe(target)?, depth, env)?;
                let sub_arguments = arguments
                    .iter()
                    .map(|p| self.substitute(p.clone(), depth, env))
                    .collect::<Result<Vec<Par>, InterpreterError>>()?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                        method_name,
                        target: Some(sub_target),
                        arguments: sub_arguments,
                        locally_free: set_bits_until(locally_free, env.shift),
                        connective_used,
                    })),
                })
            }

            _ => Ok(term),
        }
    }

    fn substitute_no_sort(
        &self,
        term: Expr,
        depth: i32,
        env: &Env<Par>,
    ) -> Result<Expr, InterpreterError> {
        match unwrap_option_safe(term.expr_instance.clone())? {
            ExprInstance::ENotBody(ENot { p }) => self
                .substitute_no_sort(unwrap_option_safe(p)?, depth, env)
                .map(|p| {
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::ENotBody(ENot { p: Some(p) })),
                    })
                })?,

            ExprInstance::ENegBody(ENeg { p }) => self
                .substitute_no_sort(unwrap_option_safe(p)?, depth, env)
                .map(|p| {
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::ENegBody(ENeg { p: Some(p) })),
                    })
                })?,

            ExprInstance::EMultBody(EMult { p1, p2 }) => {
                let _p1 = self.substitute_no_sort(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute_no_sort(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EMultBody(EMult {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EDivBody(EDiv { p1, p2 }) => {
                let _p1 = self.substitute_no_sort(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute_no_sort(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EDivBody(EDiv {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EModBody(EMod { p1, p2 }) => {
                let _p1 = self.substitute_no_sort(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute_no_sort(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EModBody(EMod {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EPercentPercentBody(EPercentPercent { p1, p2 }) => {
                let _p1 = self.substitute_no_sort(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute_no_sort(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EPercentPercentBody(EPercentPercent {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EPlusBody(EPlus { p1, p2 }) => {
                let _p1 = self.substitute_no_sort(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute_no_sort(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EPlusBody(EPlus {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EMinusBody(EMinus { p1, p2 }) => {
                let _p1 = self.substitute_no_sort(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute_no_sort(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EMinusBody(EMinus {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EPlusPlusBody(EPlusPlus { p1, p2 }) => {
                let _p1 = self.substitute_no_sort(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute_no_sort(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EPlusPlusBody(EPlusPlus {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EMinusMinusBody(EMinusMinus { p1, p2 }) => {
                let _p1 = self.substitute_no_sort(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute_no_sort(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EMinusMinusBody(EMinusMinus {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::ELtBody(ELt { p1, p2 }) => {
                let _p1 = self.substitute_no_sort(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute_no_sort(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::ELtBody(ELt {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::ELteBody(ELte { p1, p2 }) => {
                let _p1 = self.substitute_no_sort(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute_no_sort(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::ELteBody(ELte {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EGtBody(EGt { p1, p2 }) => {
                let _p1 = self.substitute_no_sort(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute_no_sort(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EGtBody(EGt {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EGteBody(EGte { p1, p2 }) => {
                let _p1 = self.substitute_no_sort(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute_no_sort(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EGteBody(EGte {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EEqBody(EEq { p1, p2 }) => {
                let _p1 = self.substitute_no_sort(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute_no_sort(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EEqBody(EEq {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::ENeqBody(ENeq { p1, p2 }) => {
                let _p1 = self.substitute_no_sort(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute_no_sort(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::ENeqBody(ENeq {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EAndBody(EAnd { p1, p2 }) => {
                let _p1 = self.substitute_no_sort(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute_no_sort(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EAndBody(EAnd {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EOrBody(EOr { p1, p2 }) => {
                let _p1 = self.substitute_no_sort(unwrap_option_safe(p1)?, depth, env)?;
                let _p2 = self.substitute_no_sort(unwrap_option_safe(p2)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EOrBody(EOr {
                        p1: Some(_p1),
                        p2: Some(_p2),
                    })),
                })
            }

            ExprInstance::EMatchesBody(EMatches { target, pattern }) => {
                let _target = self.substitute_no_sort(unwrap_option_safe(target)?, depth, env)?;
                let _pattern = self.substitute_no_sort(unwrap_option_safe(pattern)?, depth, env)?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EMatchesBody(EMatches {
                        target: Some(_target),
                        pattern: Some(_pattern),
                    })),
                })
            }

            ExprInstance::EListBody(EList {
                ps,
                locally_free,
                connective_used,
                remainder,
            }) => {
                let _ps = ps
                    .iter()
                    .map(|p| self.substitute_no_sort(p.clone(), depth, env))
                    .collect::<Result<Vec<Par>, InterpreterError>>()?;

                let new_locally_free = set_bits_until(locally_free, env.shift);

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EListBody(EList {
                        ps: _ps,
                        locally_free: new_locally_free,
                        connective_used,
                        remainder,
                    })),
                })
            }

            ExprInstance::ETupleBody(ETuple {
                ps,
                locally_free,
                connective_used,
            }) => {
                let _ps = ps
                    .iter()
                    .map(|p| self.substitute_no_sort(p.clone(), depth, env))
                    .collect::<Result<Vec<Par>, InterpreterError>>()?;

                let new_locally_free = set_bits_until(locally_free, env.shift);

                Ok(Expr {
                    expr_instance: Some(ExprInstance::ETupleBody(ETuple {
                        ps: _ps,
                        locally_free: new_locally_free,
                        connective_used,
                    })),
                })
            }

            ExprInstance::ESetBody(eset) => {
                let par_set = ParSetTypeMapper::eset_to_par_set(eset);
                let _ps = par_set
                    .ps
                    .sorted_pars
                    .iter()
                    .map(|p| self.substitute_no_sort(p.clone(), depth, env))
                    .collect::<Result<Vec<Par>, InterpreterError>>()?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
                        ParSet {
                            ps: SortedParHashSet::create_from_vec(_ps),
                            connective_used: par_set.connective_used,
                            locally_free: set_bits_until(par_set.locally_free, env.shift),
                            remainder: par_set.remainder,
                        },
                    ))),
                })
            }

            ExprInstance::EMapBody(emap) => {
                let par_map = ParMapTypeMapper::emap_to_par_map(emap);
                let _ps = par_map
                    .ps
                    .sorted_list
                    .iter()
                    .map(|p| {
                        let p1 = self.substitute_no_sort(p.0.clone(), depth, env)?;
                        let p2 = self.substitute_no_sort(p.1.clone(), depth, env)?;
                        Ok((p1, p2))
                    })
                    .collect::<Result<Vec<(Par, Par)>, InterpreterError>>()?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
                        ParMap {
                            ps: SortedParMap::create_from_vec(_ps),
                            connective_used: par_map.connective_used,
                            locally_free: set_bits_until(par_map.locally_free, env.shift),
                            remainder: par_map.remainder,
                        },
                    ))),
                })
            }

            ExprInstance::EMethodBody(EMethod {
                method_name,
                target,
                arguments,
                locally_free,
                connective_used,
            }) => {
                let sub_target =
                    self.substitute_no_sort(unwrap_option_safe(target)?, depth, env)?;
                let sub_arguments = arguments
                    .iter()
                    .map(|p| self.substitute_no_sort(p.clone(), depth, env))
                    .collect::<Result<Vec<Par>, InterpreterError>>()?;

                Ok(Expr {
                    expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                        method_name,
                        target: Some(sub_target),
                        arguments: sub_arguments,
                        locally_free: set_bits_until(locally_free, env.shift),
                        connective_used,
                    })),
                })
            }

            _ => Ok(term),
        }
    }
}

fn set_bits_until(bits: Vec<u8>, until: i32) -> Vec<u8> {
    // println!("\nbits in set_bits_until: {:?}", bits);
    // println!("\nuntil in set_bits_until: {:?}", until);
    if until <= 0 {
        return Vec::new();
    }
    let until_usize = until as usize;

    let result = bits
        .iter()
        .enumerate()
        .filter(|&(i, &bit)| i < until_usize && bit == 1)
        .map(|(_, &bit)| bit)
        .collect();

    // println!("\nresult bits in set_bits_until: {:?}", result);
    result
}
