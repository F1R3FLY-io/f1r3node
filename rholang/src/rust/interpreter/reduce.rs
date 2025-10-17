// See See rholang/src/main/scala/coop/rchain/rholang/interpreter/Reduce.scala

use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::expr::ExprInstance;
use models::rhoapi::g_unforgeable::UnfInstance;
use models::rhoapi::tagged_continuation::TaggedCont;
use models::rhoapi::var::VarInstance;
use models::rhoapi::{
    BindPattern, Bundle, EAnd, EDiv, EEq, EGt, EGte, EList, ELt, ELte, EMatches, EMethod, EMinus,
    EMinusMinus, EMod, EMult, ENeq, EOr, EPathMap, EPercentPercent, EPlus, EPlusPlus, EVar, Expr, GPrivate,
    GUnforgeable, KeyValuePair, Match, MatchCase, New, ParWithRandom, Receive, ReceiveBind, Send,
    Var,
};
use models::rhoapi::{ETuple, ListParWithRandom, Par, TaggedContinuation};
use models::rust::par_map::ParMap;
use models::rust::par_map_type_mapper::ParMapTypeMapper;
use models::rust::par_set::ParSet;
use models::rust::par_set_type_mapper::ParSetTypeMapper;
use models::rust::rholang::implicits::{concatenate_pars, single_bundle, single_expr};
use models::rust::sorted_par_hash_set::SortedParHashSet;
use models::rust::sorted_par_map::SortedParMap;
use models::rust::string_ops::StringOps;
use models::rust::utils::{
    new_elist_par, new_emap_par, new_gint_expr, new_gint_par, new_gstring_par, union,
};
use prost::Message;
use rspace_plus_plus::rspace::util::unpack_option_with_peek;
use std::collections::{BTreeMap, BTreeSet};
use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use crate::rust::interpreter::accounting::costs::{
    add_cost, bytes_to_hex_cost, diff_cost, hex_to_bytes_cost, interpolate_cost, keys_method_cost,
    length_method_cost, lookup_cost, match_eval_cost, nth_method_call_cost, remove_cost,
    size_method_cost, slice_cost, take_cost, to_byte_array_cost, to_list_cost, union_cost,
};
use crate::rust::interpreter::matcher::spatial_matcher::SpatialMatcherContext;
use crate::rust::interpreter::rho_type::RhoTuple2;

use super::accounting::_cost;
use super::accounting::costs::{
    boolean_and_cost, boolean_or_cost, byte_array_append_cost, comparison_cost, division_cost,
    equality_check_cost, list_append_cost, method_call_cost, modulo_cost, multiplication_cost,
    new_bindings_cost, op_call_cost, receive_eval_cost, send_eval_cost, string_append_cost,
    subtraction_cost, sum_cost, var_eval_cost,
};
use super::dispatch::{DispatchType, RhoDispatch, RholangAndScalaDispatcher};
use super::env::Env;
use super::errors::InterpreterError;
use super::matcher::has_locally_free::HasLocallyFree;
use super::rho_runtime::RhoISpace;
use super::rho_type::{RhoExpression, RhoUnforgeable};
use super::substitute::Substitute;
use super::unwrap_option_safe;
use super::util::GeneratedMessage;
use models::rust::pathmap_crate_type_mapper::PathMapCrateTypeMapper;

/**
 * Reduce is the interface for evaluating Rholang expressions.
 */
// TODO: urn_map to use Arc to avoid deep clone
#[derive(Clone)]
pub struct DebruijnInterpreter {
    pub space: RhoISpace,
    pub dispatcher: RhoDispatch,
    pub urn_map: HashMap<String, Par>,
    pub merge_chs: Arc<RwLock<HashSet<Par>>>,
    pub mergeable_tag_name: Par,
    pub cost: _cost,
    pub substitute: Substitute,
}

type Application = Option<(
    TaggedContinuation,
    Vec<(Par, ListParWithRandom, ListParWithRandom, bool)>,
    bool,
)>;

trait Method {
    fn apply(&self, p: Par, args: Vec<Par>, env: &Env<Par>) -> Result<Par, InterpreterError>;
}

/**
 * Materialize a send in the store, optionally returning the matched continuation.
 *
 * @param chan  The channel on which data is being sent.
 * @param data  The par objects holding the processes being sent.
 * @param persistent  True if the write should remain in the tuplespace indefinitely.
 */
impl DebruijnInterpreter {
    pub async fn eval(
        &self,
        par: Par,
        env: &Env<Par>,
        rand: Blake2b512Random,
    ) -> Result<(), InterpreterError> {
        // println!("\neval");
        let terms: Vec<GeneratedMessage> = vec![
            par.sends
                .into_iter()
                .map(GeneratedMessage::Send)
                .collect::<Vec<_>>(),
            par.receives
                .into_iter()
                .map(GeneratedMessage::Receive)
                .collect(),
            par.news.into_iter().map(GeneratedMessage::New).collect(),
            par.matches
                .into_iter()
                .map(GeneratedMessage::Match)
                .collect(),
            par.bundles
                .into_iter()
                .map(GeneratedMessage::Bundle)
                .collect(),
            par.exprs
                .into_iter()
                .filter(|expr| match &expr.expr_instance {
                    Some(expr_instance) => match expr_instance {
                        ExprInstance::EVarBody(_) => true,
                        ExprInstance::EMethodBody(_) => true,
                        _ => false,
                    },
                    None => false,
                })
                .collect::<Vec<Expr>>()
                .into_iter()
                .map(GeneratedMessage::Expr)
                .collect(),
        ]
        .into_iter()
        .filter(|vec| !vec.is_empty())
        .flatten()
        .collect();

        fn split(
            id: i32,
            terms: &Vec<GeneratedMessage>,
            rand: Blake2b512Random,
        ) -> Blake2b512Random {
            if terms.len() == 1 {
                rand
            } else if terms.len() > 256 {
                rand.split_short(id.try_into().unwrap())
            } else {
                rand.split_byte(id.try_into().unwrap())
            }
        }

        let term_split_limit = i16::MAX;
        if terms.len() > term_split_limit.try_into().unwrap() {
            Err(InterpreterError::ReduceError(format!(
                "The number of terms in the Par is {}, which exceeds the limit of {}",
                terms.len(),
                term_split_limit
            )))
        } else {
            // Collect errors from all parallel execution paths (pars)
            // parTraverseSafe
            let futures: Vec<Pin<Box<dyn futures::Future<Output = Result<(), InterpreterError>>>>> =
                terms
                    .iter()
                    .enumerate()
                    .map(|(index, term)| {
                        Box::pin(self.generated_message_eval(
                            term,
                            env,
                            split(index.try_into().unwrap(), &terms, rand.clone()),
                        ))
                            as Pin<Box<dyn futures::Future<Output = Result<(), InterpreterError>>>>
                    })
                    .collect();

            let results: Vec<Result<(), InterpreterError>> =
                futures::future::join_all(futures).await;
            let flattened_results: Vec<InterpreterError> = results
                .into_iter()
                .filter_map(|result| result.err())
                .collect();

            match self.aggregate_evaluator_errors(flattened_results) {
                Ok(_) => Ok(()),
                Err(e) => Err(e),
            }
        }
    }

    pub async fn inj(&self, par: Par, rand: Blake2b512Random) -> Result<(), InterpreterError> {
        self.eval(par, &Env::new(), rand).await
    }

    /**
     * Materialize a send in the store, optionally returning the matched continuation.
     *
     * @param chan  The channel on which data is being sent.
     * @param data  The par objects holding the processes being sent.
     * @param persistent  True if the write should remain in the tuplespace indefinitely.
     */
    async fn produce(
        &self,
        chan: Par,
        data: ListParWithRandom,
        persistent: bool,
    ) -> Result<DispatchType, InterpreterError> {
        // println!("\nreduce produce");
        // println!("chan in reduce produce: {:?}", chan);
        // println!("data in reduce produce: {:?}", data);
        self.update_mergeable_channels(&chan);

        // println!("Attempting to lock space for produce");
        let mut space_locked = self.space.try_lock().unwrap();
        // println!("Locked space for produce");
        let produce_result = space_locked.produce(chan.clone(), data.clone(), persistent)?;
        let is_replay = space_locked.is_replay();
        drop(space_locked);

        match produce_result {
            Some((c, s, produce_event)) => {
                let dispatch_type = self
                    .continue_produce_process(
                        unpack_option_with_peek(Some((c, s))),
                        chan,
                        data,
                        persistent,
                        is_replay,
                        produce_event.clone().output_value,
                    )
                    .await?;

                match dispatch_type {
                    DispatchType::NonDeterministicCall(ref output) => {
                        let produce1 = produce_event.mark_as_non_deterministic(output.clone());
                        let mut space_locked = self.space.try_lock().unwrap();
                        space_locked.update_produce(produce1);
                        drop(space_locked);
                        Ok(dispatch_type)
                    }

                    _ => Ok(dispatch_type),
                }
            }
            None => Ok(DispatchType::Skip),
        }
    }

    async fn consume(
        &self,
        binds: Vec<(BindPattern, Par)>,
        body: ParWithRandom,
        persistent: bool,
        peek: bool,
    ) -> Result<DispatchType, InterpreterError> {
        // println!("\nreduce consume");
        // println!("binds in reduce consume: {:?}", binds);
        // println!("body in reduce consume: {:?}", body);
        let (patterns, sources): (Vec<BindPattern>, Vec<Par>) = binds.clone().into_iter().unzip();

        // Update mergeable channels
        for source in &sources {
            self.update_mergeable_channels(source);
        }

        // println!("\nsources in reduce consume: {:?}", sources);

        // println!("Attempting to lock space for produce");
        let mut space_locked = self.space.try_lock().unwrap();
        let consume_result = space_locked.consume(
            sources.clone(),
            patterns.clone(),
            TaggedContinuation {
                tagged_cont: Some(TaggedCont::ParBody(body.clone())),
            },
            persistent,
            if peek {
                BTreeSet::from_iter((0..sources.len() as i32).collect::<Vec<i32>>())
            } else {
                BTreeSet::new()
            },
        )?;
        let is_replay = space_locked.is_replay();
        drop(space_locked);

        // println!("space map in reduce consume: {:?}", self.space.lock().unwrap().to_map());
        // println!("\nconsume_result in reduce consume: {:?}", consume_result);

        self.continue_consume_process(
            unpack_option_with_peek(consume_result),
            binds,
            body,
            persistent,
            peek,
            is_replay,
            Vec::new(),
        )
        .await
    }

    async fn continue_produce_process(
        &self,
        res: Application,
        chan: Par,
        data: ListParWithRandom,
        persistent: bool,
        is_replay: bool,
        previous_output: Vec<Vec<u8>>,
    ) -> Result<DispatchType, InterpreterError> {
        // println!("\ncontinue_produce_process");
        let previous_output_as_par = previous_output
            .into_iter()
            .map(|bytes| {
                Par::decode(&bytes[..]).map_err(|e| InterpreterError::DecodeError(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        match res {
            Some((continuation, data_list, peek)) => {
                if persistent {
                    // dispatchAndRun
                    let mut futures: Vec<
                        Pin<
                            Box<
                                dyn futures::Future<
                                    Output = Result<DispatchType, InterpreterError>,
                                >,
                            >,
                        >,
                    > = vec![Box::pin(self.dispatch(
                        continuation,
                        data_list.clone(),
                        is_replay,
                        previous_output_as_par,
                    ))];
                    futures.push(Box::pin(self.produce(chan, data, persistent)));

                    // parTraverseSafe
                    let results: Vec<Result<DispatchType, InterpreterError>> =
                        futures::future::join_all(futures).await;
                    let flattened_results: Vec<InterpreterError> = results
                        .into_iter()
                        .filter_map(|result| result.err())
                        .collect();

                    self.aggregate_evaluator_errors(flattened_results)
                } else if peek {
                    // dispatchAndRun
                    let mut futures: Vec<
                        Pin<
                            Box<
                                dyn futures::Future<
                                    Output = Result<DispatchType, InterpreterError>,
                                >,
                            >,
                        >,
                    > = vec![Box::pin(self.dispatch(
                        continuation,
                        data_list.clone(),
                        is_replay,
                        previous_output_as_par,
                    ))];
                    futures.extend(self.produce_peeks(data_list).await);

                    // parTraverseSafe
                    let results: Vec<Result<DispatchType, InterpreterError>> =
                        futures::future::join_all(futures).await;
                    let flattened_results: Vec<InterpreterError> = results
                        .into_iter()
                        .filter_map(|result| result.err())
                        .collect();

                    self.aggregate_evaluator_errors(flattened_results)
                } else {
                    self.dispatch(continuation, data_list, is_replay, previous_output_as_par)
                        .await
                }
            }
            None => Ok(DispatchType::Skip),
        }
    }

    async fn continue_consume_process(
        &self,
        res: Application,
        binds: Vec<(BindPattern, Par)>,
        body: ParWithRandom,
        persistent: bool,
        peek: bool,
        is_replay: bool,
        previous_output: Vec<Vec<u8>>,
    ) -> Result<DispatchType, InterpreterError> {
        // println!("\ncontinue_consume_process");
        // println!("\napplication in continue_consume_process: {:?}", res);
        let previous_output_as_par = previous_output
            .into_iter()
            .map(|bytes| {
                Par::decode(&bytes[..]).map_err(|e| InterpreterError::DecodeError(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        match res {
            Some((continuation, data_list, _peek)) => {
                if persistent {
                    // dispatchAndRun
                    let mut futures: Vec<
                        Pin<
                            Box<
                                dyn futures::Future<
                                    Output = Result<DispatchType, InterpreterError>,
                                >,
                            >,
                        >,
                    > = vec![Box::pin(self.dispatch(
                        continuation,
                        data_list.clone(),
                        is_replay,
                        previous_output_as_par,
                    ))];
                    futures.push(Box::pin(self.consume(binds, body, persistent, peek)));

                    // parTraverseSafe
                    let results: Vec<Result<DispatchType, InterpreterError>> =
                        futures::future::join_all(futures).await;
                    let flattened_results: Vec<InterpreterError> = results
                        .into_iter()
                        .filter_map(|result| result.err())
                        .collect();

                    self.aggregate_evaluator_errors(flattened_results)
                } else if _peek {
                    // dispatchAndRun
                    let mut futures: Vec<
                        Pin<
                            Box<
                                dyn futures::Future<
                                    Output = Result<DispatchType, InterpreterError>,
                                >,
                            >,
                        >,
                    > = vec![Box::pin(self.dispatch(
                        continuation,
                        data_list.clone(),
                        is_replay,
                        previous_output_as_par,
                    ))];
                    futures.extend(self.produce_peeks(data_list).await);

                    // parTraverseSafe
                    let results: Vec<Result<DispatchType, InterpreterError>> =
                        futures::future::join_all(futures).await;
                    let flattened_results: Vec<InterpreterError> = results
                        .into_iter()
                        .filter_map(|result| result.err())
                        .collect();

                    self.aggregate_evaluator_errors(flattened_results)
                } else {
                    self.dispatch(continuation, data_list, is_replay, previous_output_as_par)
                        .await
                }
            }
            None => Ok(DispatchType::Skip),
        }
    }

    async fn dispatch(
        &self,
        continuation: TaggedContinuation,
        data_list: Vec<(Par, ListParWithRandom, ListParWithRandom, bool)>,
        is_replay: bool,
        previous_output: Vec<Par>,
    ) -> Result<DispatchType, InterpreterError> {
        // println!("\nreduce dispatch");
        let dispatcher_lock = &self.dispatcher.try_read().unwrap();
        // println!("Dispatcher lock acquired");
        dispatcher_lock
            .dispatch(
                continuation,
                data_list.into_iter().map(|tuple| tuple.1).collect(),
                is_replay,
                previous_output,
            )
            .await
    }

    async fn produce_peeks(
        &self,
        data_list: Vec<(Par, ListParWithRandom, ListParWithRandom, bool)>,
    ) -> Vec<Pin<Box<dyn futures::Future<Output = Result<DispatchType, InterpreterError>> + '_>>>
    {
        // println!("\nreduce produce_peeks");
        data_list
            .into_iter()
            .filter(|(_, _, _, persist)| !persist)
            .map(|(chan, _, removed_data, _)| {
                Box::pin(self.produce(chan, removed_data, false))
                    as Pin<
                        Box<dyn futures::Future<Output = Result<DispatchType, InterpreterError>>>,
                    >
            })
            .collect()
    }

    /* Collect mergeable channels */

    fn update_mergeable_channels(&self, chan: &Par) -> () {
        let is_mergeable = self.is_mergeable_channel(chan);
        // println!("\nis_mergeable: {:?}", is_mergeable);

        if is_mergeable {
            let mut merge_chs_write = self.merge_chs.try_write().unwrap();
            merge_chs_write.insert(chan.clone());
        }
    }

    fn is_mergeable_channel(&self, chan: &Par) -> bool {
        let tuple_elms: Vec<Par> = chan
            .exprs
            .iter()
            .flat_map(|y| match &y.expr_instance {
                Some(expr_instance) => match expr_instance {
                    ExprInstance::ETupleBody(etuple) => etuple.ps.clone(),
                    _ => ETuple::default().ps,
                },
                None => ETuple::default().ps,
            })
            .collect();

        tuple_elms
            .first()
            .map_or(false, |head| head == &self.mergeable_tag_name)
    }

    fn aggregate_evaluator_errors(
        &self,
        errors: Vec<InterpreterError>,
    ) -> Result<DispatchType, InterpreterError> {
        match errors.as_slice() {
            // No errors
            [] => Ok(DispatchType::Skip),

            // Out Of Phlogiston error is always single
            // - if one execution path is out of phlo, the whole evaluation is also
            err_list
                if err_list
                    .iter()
                    .any(|e| matches!(e, InterpreterError::OutOfPhlogistonsError)) =>
            {
                Err(InterpreterError::OutOfPhlogistonsError)
            }

            // Rethrow single error
            [ex] => Err(ex.clone()),

            // Collect errors from parallel execution
            err_list => Err(InterpreterError::AggregateError {
                interpreter_errors: err_list.to_vec(),
            }),
        }
    }

    async fn generated_message_eval(
        &self,
        term: &GeneratedMessage,
        env: &Env<Par>,
        rand: Blake2b512Random,
    ) -> Result<(), InterpreterError> {
        // println!("\ngenerated_message_eval, term: {:?}", term);
        match term {
            GeneratedMessage::Send(term) => self.eval_send(term, env, rand).await,
            GeneratedMessage::Receive(term) => self.eval_receive(term, env, rand).await,
            GeneratedMessage::New(term) => self.eval_new(term, env.clone(), rand).await,
            GeneratedMessage::Match(term) => self.eval_match(term, env, rand).await,
            GeneratedMessage::Bundle(term) => self.eval_bundle(term, env, rand).await,
            GeneratedMessage::Expr(term) => match &term.expr_instance {
                Some(expr_instance) => match expr_instance {
                    ExprInstance::EVarBody(e) => {
                        let res = self.eval_var(&e.clone().v.unwrap(), &env)?;
                        self.eval(res, env, rand).await
                    }
                    ExprInstance::EMethodBody(e) => {
                        let res = self.eval_expr_to_par(
                            &Expr {
                                expr_instance: Some(ExprInstance::EMethodBody(e.clone())),
                            },
                            env,
                        )?;
                        self.eval(res, env, rand).await
                    }
                    other => Err(InterpreterError::BugFoundError(format!(
                        "Undefined term: {:?}",
                        other
                    ))),
                },
                None => Err(InterpreterError::BugFoundError(format!(
                    "Undefined term, expr_instance was None"
                ))),
            },
        }
    }

    /** Algorithm as follows:
     *
     * 1. Fully evaluate the channel in given environment.
     * 2. Substitute any variable references in the channel so that it can be
     *    correctly used as a key in the tuple space.
     * 3. Evaluate any top level expressions in the data being sent.
     * 4. Call produce
     *
     * @param send An output process
     * @param env An execution context
     *
     */
    async fn eval_send(
        &self,
        send: &Send,
        env: &Env<Par>,
        rand: Blake2b512Random,
    ) -> Result<(), InterpreterError> {
        // println!("\nenv in eval_send: {:?}", env);
        self.cost.charge(send_eval_cost())?;
        let eval_chan = self.eval_expr(&unwrap_option_safe(send.chan.clone())?, env)?;
        let sub_chan = self.substitute.substitute_and_charge(&eval_chan, 0, env)?;
        let unbundled = match single_bundle(&sub_chan) {
            Some(value) => {
                if !value.write_flag {
                    return Err(InterpreterError::ReduceError(
                        "Trying to send on non-writeable channel.".to_string(),
                    ));
                } else {
                    unwrap_option_safe(value.body)?
                }
            }
            None => sub_chan,
        };

        let data = send
            .data
            .iter()
            .map(|expr| self.eval_expr(expr, env))
            .collect::<Result<Vec<_>, InterpreterError>>()?;

        let subst_data = data
            .clone()
            .into_iter()
            .map(|p| self.substitute.substitute_and_charge(&p, 0, env))
            .collect::<Result<Vec<_>, InterpreterError>>()?;

        // println!("\ndata in eval_send: {:?}", data);
        // println!("\nsubst_data in eval_send: {:?}", subst_data);

        // println!("\nrand in eval_send");
        // rand.debug_str();

        self.produce(
            unbundled,
            ListParWithRandom {
                pars: subst_data,
                random_state: rand.to_bytes(),
            },
            send.persistent,
        )
        .await?;
        Ok(())
    }

    async fn eval_receive(
        &self,
        receive: &Receive,
        env: &Env<Par>,
        rand: Blake2b512Random,
    ) -> Result<(), InterpreterError> {
        // println!("\nreceive in eval_receive: {:?}", receive);
        // println!("\nreceive binds length: {:?}", receive.binds.len());
        self.cost.charge(receive_eval_cost())?;
        let binds = receive
            .binds
            .clone()
            .into_iter()
            .map(|rb| {
                // println!("\nrb in eval_receive: {:?}", rb);
                let q = self.unbundle_receive(&rb, env)?;
                // println!("\nq in eval_receive: {:?}", q);
                let subst_patterns = rb
                    .patterns
                    .into_iter()
                    .map(|pattern| self.substitute.substitute_and_charge(&pattern, 1, env))
                    .collect::<Result<Vec<_>, InterpreterError>>()?;

                // println!("\nsubst_patterns in eval_receive: {:?}", subst_patterns);

                Ok((
                    BindPattern {
                        patterns: subst_patterns,
                        remainder: rb.remainder,
                        free_count: rb.free_count,
                    },
                    q,
                ))
            })
            .collect::<Result<Vec<_>, InterpreterError>>()?;

        // TODO: Allow for the environment to be stored with the body in the Tuplespace - OLD
        let subst_body = self.substitute.substitute_no_sort_and_charge(
            receive.body.as_ref().unwrap(),
            0,
            &env.shift(receive.bind_count),
        )?;

        // println!("\nbinds in eval_receive: {:?}", binds);
        // println!("\nsubst_body in eval_receive: {:?}", subst_body);

        // println!("\nrand in eval_receive");
        // rand.debug_str();

        self.consume(
            binds,
            ParWithRandom {
                body: Some(subst_body),
                random_state: rand.to_bytes(),
            },
            receive.persistent,
            receive.peek,
        )
        .await?;
        Ok(())
    }

    /**
     * Variable "evaluation" is an environment lookup, but
     * lookup of an unbound variable should be an error.
     *
     * @param valproc The variable to be evaluated
     * @param env  provides the environment (possibly) containing a binding for the given variable.
     * @return If the variable has a binding (par), lift the
     *                  binding into the monadic context, else signal
     *                  an exception.
     */
    fn eval_var(&self, valproc: &Var, env: &Env<Par>) -> Result<Par, InterpreterError> {
        self.cost.charge(var_eval_cost())?;
        // println!("\nenv in eval_var: {:?}", env);
        match valproc.var_instance {
            Some(VarInstance::BoundVar(level)) => match env.get(&level) {
                Some(p) => Ok(p),
                None => Err(InterpreterError::ReduceError(format!(
                    "Unbound variable: {} in {:?}",
                    level, env.env_map
                ))),
            },
            Some(VarInstance::Wildcard(_)) => Err(InterpreterError::ReduceError(
                "Unbound variable: attempting to evaluate a pattern".to_string(),
            )),
            Some(VarInstance::FreeVar(_)) => Err(InterpreterError::ReduceError(
                "Unbound variable: attempting to evaluate a pattern".to_string(),
            )),
            None => Err(InterpreterError::ReduceError(
                "Impossible var instance EMPTY".to_string(),
            )),
        }
    }

    // TODO: review 'loop' matches 'tailRecM'
    async fn eval_match(
        &self,
        mat: &Match,
        env: &Env<Par>,
        rand: Blake2b512Random,
    ) -> Result<(), InterpreterError> {
        fn add_to_env(env: &Env<Par>, free_map: BTreeMap<i32, Par>, free_count: i32) -> Env<Par> {
            (0..free_count).fold(env.clone(), |mut acc, e| {
                let value = free_map.get(&e).unwrap_or(&Par::default()).clone();
                acc.put(value)
            })
        }

        let first_match = Box::new(
            |target: Par, cases: Vec<MatchCase>, rand: Blake2b512Random| async {
                let mut state = (target, cases);

                loop {
                    let (_target, _cases) = state;

                    match _cases.as_slice() {
                        [] => return Ok(()),

                        [single_case, case_rem @ ..] => {
                            let pattern = self.substitute.substitute_and_charge(
                                &unwrap_option_safe(single_case.pattern.clone())?,
                                1,
                                env,
                            )?;

                            // println!("\ntarget in eval_matcher: {:?}", target);
                            // println!("\npattern in eval_matcher: {:?}", pattern);

                            let mut spatial_matcher = SpatialMatcherContext::new();
                            let match_result =
                                spatial_matcher.spatial_match_result(_target.clone(), pattern);

                            // println!("\nmatch_result in eval_matcher: {:?}", match_result);

                            match match_result {
                                None => {
                                    state = (_target, case_rem.to_vec());
                                }

                                Some(free_map) => {
                                    let eval_result = self
                                        .eval(
                                            single_case.source.clone().unwrap(),
                                            &add_to_env(
                                                env,
                                                free_map.clone(),
                                                single_case.free_count,
                                            ),
                                            rand,
                                        )
                                        .await?;

                                    return Ok(eval_result);
                                }
                            }
                        }
                    }
                }
            },
        );

        self.cost.charge(match_eval_cost())?;
        let evaled_target = self.eval_expr(&mat.target.as_ref().unwrap(), env)?;
        let subst_target = self
            .substitute
            .substitute_and_charge(&evaled_target, 0, env)?;

        // println!("\nsubst_target in eval_match: {:?}", subst_target);

        first_match(subst_target, mat.cases.clone(), rand).await
    }

    /**
     * Adds neu.bindCount new GPrivate from UUID's to the environment and then
     * proceeds to evaluate the body.
     */
    // TODO: Eliminate variable shadowing - OLD
    async fn eval_new(
        &self,
        new: &New,
        env: Env<Par>,
        mut rand: Blake2b512Random,
    ) -> Result<(), InterpreterError> {
        // println!("\nnew in eval_new: {:?}", new);
        // println!("\nrand in eval_new");
        // rand.debug_str();
        // println!("\nrand next: {:?}", rand.next());
        let mut alloc = |count: usize, urns: Vec<String>| {
            let simple_news =
                (0..(count - urns.len()))
                    .into_iter()
                    .fold(env.clone(), |mut _env: Env<Par>, _| {
                        let addr: Par = Par::default().with_unforgeables(vec![GUnforgeable {
                            unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
                                id: rand.next().iter().map(|&x| x as u8).collect::<Vec<u8>>(),
                            })),
                        }]);
                        // println!("\nrand in simple_news");
                        // rand.debug_str();
                        _env.put(addr)
                    });

            // println!("\nrand in eval_new after");
            // rand.debug_str();
            // println!("\nsimple_news in eval_new: {:?}", simple_news);

            let add_urn = |new_env: &mut Env<Par>, urn: String| {
                // println!("\nurn_map: {:?}", self.urn_map);
                if !self.urn_map.contains_key(&urn) {
                    // TODO: Injections (from normalizer) are not used currently, see [[NormalizerEnv]].
                    // If `urn` can't be found in `urnMap`, it must be referencing an injection - OLD
                    // println!("\nnew_injections: {:?}", new.injections);
                    match new.injections.get(&urn) {
                        Some(p) => {
                            if let Some(gunf) = RhoUnforgeable::unapply(p) {
                                if let Some(instance) = gunf.unf_instance {
                                    Ok(new_env.put(Par::default().with_unforgeables(vec![
                                        GUnforgeable {
                                            unf_instance: Some(instance),
                                        },
                                    ])))
                                } else {
                                    Err(InterpreterError::BugFoundError(
                                        "unf_instance field is None".to_string(),
                                    ))
                                }
                            } else if let Some(expr) = RhoExpression::unapply(p) {
                                if let Some(instance) = expr.expr_instance {
                                    Ok(new_env.put(Par::default().with_exprs(vec![Expr {
                                        expr_instance: Some(instance),
                                    }])))
                                } else {
                                    Err(InterpreterError::BugFoundError(
                                        "expr_instance field is None".to_string(),
                                    ))
                                }
                            } else {
                                Err(InterpreterError::BugFoundError(
                                    "invalid injection".to_string(),
                                ))
                            }
                        }
                        None => Err(InterpreterError::BugFoundError(format!(
                            "No value set for {}. This is a bug in the normalizer or on the path from it.",
                            urn
                        ))),
                    }
                } else {
                    match self.urn_map.get(&urn) {
                        Some(p) => Ok(new_env.put(p.clone())),
                        None => Err(InterpreterError::ReduceError(format!(
                            "Unknown urn for new: {}",
                            urn
                        ))),
                    }
                }
            };

            urns.iter().try_fold(simple_news, |mut acc, urn| {
                add_urn(&mut acc, urn.to_string())
            })
        };

        // println!("\nhit eval_new");
        self.cost.charge(new_bindings_cost(new.bind_count as i64))?;
        // println!("\nnew uri: {:?}", new.uri);
        match alloc(new.bind_count as usize, new.uri.clone()) {
            Ok(env) => {
                // println!("\nenv in eval_new: {:?}", env);
                self.eval(unwrap_option_safe(new.p.clone())?, &env, rand)
                    .await
            }
            Err(e) => Err(e),
        }
    }

    fn unbundle_receive(&self, rb: &ReceiveBind, env: &Env<Par>) -> Result<Par, InterpreterError> {
        let eval_src = self.eval_expr(&unwrap_option_safe(rb.source.clone())?, env)?;
        // println!("\neval_src in unbundle_receive: {:?}", eval_src);
        let subst = self.substitute.substitute_and_charge(&eval_src, 0, env)?;
        // println!("\nsubst in unbundle_receive: {:?}", eval_src);
        // Check if we try to read from bundled channel
        let unbndl = match single_bundle(&subst) {
            Some(value) => {
                if !value.read_flag {
                    return Err(InterpreterError::ReduceError(
                        "Trying to read from non-readable channel.".to_string(),
                    ));
                } else {
                    value.body.unwrap()
                }
            }
            None => subst,
        };

        // println!("\nunbndl in unbundle_receive: {:?}", unbndl);
        Ok(unbndl)
    }

    async fn eval_bundle(
        &self,
        bundle: &Bundle,
        env: &Env<Par>,
        rand: Blake2b512Random,
    ) -> Result<(), InterpreterError> {
        self.eval(unwrap_option_safe(bundle.body.clone())?, env, rand)
            .await
    }

    // Public here for testing purposes
    pub fn eval_expr_to_par(&self, expr: &Expr, env: &Env<Par>) -> Result<Par, InterpreterError> {
        match unwrap_option_safe(expr.expr_instance.clone())? {
            ExprInstance::EVarBody(evar) => {
                // println!("\nenv in eval_expr_to_par: {:?}", env);
                let p = self.eval_var(&unwrap_option_safe(evar.v)?, env)?;
                // println!("\np in eval_expr_to_par: {:?}", p);
                // println!("\nenv in eval_expr_to_par: {:?}", env);
                let evaled_p = self.eval_expr(&p, env)?;
                Ok(evaled_p)
            }
            ExprInstance::EMethodBody(emethod) => {
                self.cost.charge(method_call_cost())?;
                let evaled_target = self.eval_expr(&unwrap_option_safe(emethod.target)?, env)?;
                let evaled_args: Vec<Par> = emethod
                    .arguments
                    .iter()
                    .map(|arg| self.eval_expr(arg, env))
                    .collect::<Result<Vec<_>, InterpreterError>>()?;

                let result_par = match self.method_table().get(&emethod.method_name) {
                    Some(_method) => _method.apply(evaled_target, evaled_args, env)?,
                    None => {
                        return Err(InterpreterError::ReduceError(format!(
                            "Unimplemented method: {}",
                            emethod.method_name
                        )));
                    }
                };

                Ok(result_par)
            }
            _ => Ok(Par::default().with_exprs(vec![self.eval_expr_to_expr(expr, env)?])),
        }
    }

    fn eval_expr_to_expr(&self, expr: &Expr, env: &Env<Par>) -> Result<Expr, InterpreterError> {
        let relop = |p1: &Par,
                     p2: &Par,
                     relopb: fn(bool, bool) -> bool,
                     relopi: fn(i64, i64) -> bool,
                     relops: fn(String, String) -> bool| {
            let v1 = self.eval_single_expr(p1, env)?;
            let v2 = self.eval_single_expr(p2, env)?;

            match (
                v1.expr_instance.clone().unwrap(),
                v2.expr_instance.clone().unwrap(),
            ) {
                (ExprInstance::GBool(b1), ExprInstance::GBool(b2)) => Ok(Expr {
                    expr_instance: Some(ExprInstance::GBool(relopb(b1, b2))),
                }),

                (ExprInstance::GInt(i1), ExprInstance::GInt(i2)) => Ok(Expr {
                    expr_instance: Some(ExprInstance::GBool(relopi(i1, i2))),
                }),

                (ExprInstance::GString(s1), ExprInstance::GString(s2)) => Ok(Expr {
                    expr_instance: Some(ExprInstance::GBool(relops(s1, s2))),
                }),

                _ => Err(InterpreterError::ReduceError(format!(
                    "Unexpected compare: {:?} vs. {:?}",
                    v1, v2
                ))),
            }
        };

        match &expr.expr_instance {
            Some(expr_instance) => match expr_instance {
                ExprInstance::GBool(x) => Ok(Expr {
                    expr_instance: Some(ExprInstance::GBool(*x)),
                }),

                ExprInstance::GInt(x) => Ok(Expr {
                    expr_instance: Some(ExprInstance::GInt(*x)),
                }),

                ExprInstance::GString(x) => Ok(Expr {
                    expr_instance: Some(ExprInstance::GString(x.clone())),
                }),

                ExprInstance::GUri(x) => Ok(Expr {
                    expr_instance: Some(ExprInstance::GUri(x.clone())),
                }),

                ExprInstance::GByteArray(x) => Ok(Expr {
                    expr_instance: Some(ExprInstance::GByteArray(x.clone())),
                }),

                ExprInstance::ENotBody(enot) => {
                    let b = self.eval_to_bool(&enot.p.as_ref().unwrap(), env)?;
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GBool(!b)),
                    })
                }

                ExprInstance::ENegBody(eneg) => {
                    let v = self.eval_to_i64(&eneg.p.as_ref().unwrap(), env)?;
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GInt(-v)),
                    })
                }

                ExprInstance::EMultBody(EMult { p1, p2 }) => {
                    let v1 = self.eval_to_i64(&p1.clone().unwrap(), env)?;
                    let v2 = self.eval_to_i64(&p2.clone().unwrap(), env)?;
                    self.cost.charge(multiplication_cost())?;
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GInt(v1 * v2)),
                    })
                }

                ExprInstance::EDivBody(EDiv { p1, p2 }) => {
                    let v1 = self.eval_to_i64(&p1.clone().unwrap(), env)?;
                    let v2 = self.eval_to_i64(&p2.clone().unwrap(), env)?;
                    self.cost.charge(division_cost())?;
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GInt(v1 / v2)),
                    })
                }

                ExprInstance::EModBody(EMod { p1, p2 }) => {
                    let v1 = self.eval_to_i64(&p1.clone().unwrap(), env)?;
                    let v2 = self.eval_to_i64(&p2.clone().unwrap(), env)?;
                    self.cost.charge(modulo_cost())?;
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GInt(v1 % v2)),
                    })
                }

                ExprInstance::EPlusBody(EPlus { p1, p2 }) => {
                    let v1 = self.eval_single_expr(&p1.clone().unwrap(), env)?;
                    let v2 = self.eval_single_expr(&p2.clone().unwrap(), env)?;

                    match (v1.expr_instance.unwrap(), v2.expr_instance.unwrap()) {
                        (ExprInstance::GInt(lhs), ExprInstance::GInt(rhs)) => {
                            self.cost.charge(sum_cost())?;
                            Ok(Expr {
                                expr_instance: Some(ExprInstance::GInt(lhs.wrapping_add(rhs))),
                            })
                        }

                        (ExprInstance::ESetBody(lhs), rhs) => {
                            self.cost.charge(op_call_cost())?;
                            let result_par = self.add_method().apply(
                                Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(ExprInstance::ESetBody(lhs)),
                                }]),
                                vec![Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(rhs),
                                }])],
                                env,
                            )?;

                            let result_expr = self.eval_single_expr(&result_par, env)?;
                            Ok(result_expr)
                        }

                        (ExprInstance::GInt(_), other) => {
                            Err(InterpreterError::OperatorExpectedError {
                                op: "+".to_string(),
                                expected: "Int".to_string(),
                                other_type: get_type(other),
                            })
                        }

                        (other, _) => Err(InterpreterError::OperatorNotDefined {
                            op: "+".to_string(),
                            other_type: get_type(other),
                        }),
                    }
                }

                ExprInstance::EMinusBody(EMinus { p1, p2 }) => {
                    let v1 = self.eval_single_expr(&p1.clone().unwrap(), env)?;
                    let v2 = self.eval_single_expr(&p2.clone().unwrap(), env)?;

                    match (v1.expr_instance.unwrap(), v2.expr_instance.unwrap()) {
                        (ExprInstance::GInt(lhs), ExprInstance::GInt(rhs)) => {
                            self.cost.charge(subtraction_cost())?;
                            Ok(Expr {
                                expr_instance: Some(ExprInstance::GInt(lhs - rhs)),
                            })
                        }

                        (ExprInstance::EMapBody(lhs), rhs) => {
                            self.cost.charge(op_call_cost())?;
                            let result_par = self.delete_method().apply(
                                Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(ExprInstance::EMapBody(lhs)),
                                }]),
                                vec![Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(rhs),
                                }])],
                                env,
                            )?;

                            let result_expr = self.eval_single_expr(&result_par, env)?;
                            Ok(result_expr)
                        }

                        (ExprInstance::ESetBody(lhs), rhs) => {
                            self.cost.charge(op_call_cost())?;
                            let result_par = self.delete_method().apply(
                                Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(ExprInstance::ESetBody(lhs)),
                                }]),
                                vec![Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(rhs),
                                }])],
                                env,
                            )?;

                            let result_expr = self.eval_single_expr(&result_par, env)?;
                            Ok(result_expr)
                        }

                        (ExprInstance::GInt(_), other) => {
                            Err(InterpreterError::OperatorExpectedError {
                                op: "-".to_string(),
                                expected: "Int".to_string(),
                                other_type: get_type(other),
                            })
                        }

                        (other, _) => Err(InterpreterError::OperatorNotDefined {
                            op: "-".to_string(),
                            other_type: get_type(other),
                        }),
                    }
                }

                ExprInstance::ELtBody(ELt { p1, p2 }) => {
                    self.cost.charge(comparison_cost())?;
                    relop(
                        &p1.clone().unwrap(),
                        &p2.clone().unwrap(),
                        |b1: bool, b2: bool| b1 < b2,
                        |i1: i64, i2: i64| i1 < i2,
                        |s1: String, s2: String| s1 < s2,
                    )
                }

                ExprInstance::ELteBody(ELte { p1, p2 }) => {
                    self.cost.charge(comparison_cost())?;
                    relop(
                        &p1.clone().unwrap(),
                        &p2.clone().unwrap(),
                        |b1: bool, b2: bool| b1 <= b2,
                        |i1: i64, i2: i64| i1 <= i2,
                        |s1: String, s2: String| s1 <= s2,
                    )
                }

                ExprInstance::EGtBody(EGt { p1, p2 }) => {
                    self.cost.charge(comparison_cost())?;
                    relop(
                        &p1.clone().unwrap(),
                        &p2.clone().unwrap(),
                        |b1: bool, b2: bool| b1 > b2,
                        |i1: i64, i2: i64| i1 > i2,
                        |s1: String, s2: String| s1 > s2,
                    )
                }

                ExprInstance::EGteBody(EGte { p1, p2 }) => {
                    self.cost.charge(comparison_cost())?;
                    relop(
                        &p1.clone().unwrap(),
                        &p2.clone().unwrap(),
                        |b1: bool, b2: bool| b1 >= b2,
                        |i1: i64, i2: i64| i1 >= i2,
                        |s1: String, s2: String| s1 >= s2,
                    )
                }

                ExprInstance::EEqBody(EEq { p1, p2 }) => {
                    let v1 = self.eval_expr(&p1.clone().unwrap(), env)?;
                    let v2 = self.eval_expr(&p2.clone().unwrap(), env)?;
                    // TODO: build an equality operator that takes in an environment. - OLD
                    let sv1 = self.substitute.substitute_and_charge(&v1, 0, env)?;
                    let sv2 = self.substitute.substitute_and_charge(&v2, 0, env)?;
                    self.cost.charge(equality_check_cost(&sv1, &sv2))?;

                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GBool(sv1 == sv2)),
                    })
                }

                ExprInstance::ENeqBody(ENeq { p1, p2 }) => {
                    let v1 = self.eval_expr(&p1.clone().unwrap(), env)?;
                    let v2 = self.eval_expr(&p2.clone().unwrap(), env)?;
                    let sv1 = self.substitute.substitute_and_charge(&v1, 0, env)?;
                    let sv2 = self.substitute.substitute_and_charge(&v2, 0, env)?;
                    self.cost.charge(equality_check_cost(&sv1, &sv2))?;

                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GBool(sv1 != sv2)),
                    })
                }

                ExprInstance::EAndBody(EAnd { p1, p2 }) => {
                    let b1 = self.eval_to_bool(&p1.clone().unwrap(), env)?;
                    let b2 = self.eval_to_bool(&p2.clone().unwrap(), env)?;
                    self.cost.charge(boolean_and_cost())?;

                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GBool(b1 && b2)),
                    })
                }

                ExprInstance::EOrBody(EOr { p1, p2 }) => {
                    let b1 = self.eval_to_bool(&p1.clone().unwrap(), env)?;
                    let b2 = self.eval_to_bool(&p2.clone().unwrap(), env)?;
                    self.cost.charge(boolean_or_cost())?;

                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GBool(b1 || b2)),
                    })
                }

                ExprInstance::EMatchesBody(EMatches { target, pattern }) => {
                    let evaled_target = self.eval_expr(&target.clone().unwrap(), env)?;
                    let subst_target =
                        self.substitute
                            .substitute_and_charge(&evaled_target, 0, env)?;
                    let subst_pattern =
                        self.substitute
                            .substitute_and_charge(&pattern.clone().unwrap(), 1, env)?;

                    let mut spatial_matcher = SpatialMatcherContext::new();
                    let match_result =
                        spatial_matcher.spatial_match_result(subst_target, subst_pattern);

                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GBool(match_result.is_some())),
                    })
                }

                ExprInstance::EPercentPercentBody(EPercentPercent { p1, p2 }) => {
                    fn eval_to_string_pair(
                        key_expr: Expr,
                        value_expr: Expr,
                    ) -> Result<(String, String), InterpreterError> {
                        match (
                            key_expr.expr_instance.unwrap(),
                            value_expr.expr_instance.unwrap(),
                        ) {
                            (
                                ExprInstance::GString(key_string),
                                ExprInstance::GString(value_string),
                            ) => Ok((key_string, value_string)),

                            (ExprInstance::GString(key_string), ExprInstance::GInt(value_int)) => {
                                Ok((key_string, value_int.to_string()))
                            }

                            (
                                ExprInstance::GString(key_string),
                                ExprInstance::GBool(value_bool),
                            ) => Ok((key_string, value_bool.to_string())),

                            (ExprInstance::GString(key_string), ExprInstance::GUri(uri)) => {
                                Ok((key_string, uri))
                            }

                            // TODO: Add cases for other ground terms as well? Maybe it would be better
                            // to implement cats.Show for all ground terms. - OLD
                            (ExprInstance::GString(_), value) => {
                                Err(InterpreterError::ReduceError(format!(
                                    "Error: interpolation doesn't support {:?}",
                                    get_type(value),
                                )))
                            }

                            _ => Err(InterpreterError::ReduceError(format!(
                                "Error: interpolation Map should only contain String keys"
                            ))),
                        }
                    }

                    fn interpolate(string: &str, key_value_pairs: &[(String, String)]) -> String {
                        let mut result = String::new();
                        let mut current = string.to_string();

                        while !current.is_empty() {
                            let mut found = false;

                            for (k, v) in key_value_pairs {
                                if current.starts_with(&format!("${{{}}}", k)) {
                                    result.push_str(v);
                                    current = current.split_at(k.len() + 3).1.to_string();
                                    found = true;

                                    break;
                                }
                            }

                            if !found {
                                result.push(current.chars().next().unwrap());
                                current.remove(0);
                            }
                        }

                        result
                    }

                    self.cost.charge(op_call_cost())?;
                    let v1 = self.eval_single_expr(&p1.clone().unwrap(), env)?;
                    let v2 = self.eval_single_expr(&p2.clone().unwrap(), env)?;

                    match (v1.expr_instance.unwrap(), v2.expr_instance.unwrap()) {
                        (ExprInstance::GString(lhs), ExprInstance::EMapBody(emap)) => {
                            let rhs = ParMapTypeMapper::emap_to_par_map(emap).ps;
                            if !lhs.is_empty() || !rhs.is_empty() {
                                let key_value_pairs = rhs
                                    .clone()
                                    .into_iter()
                                    .map(|(k, v)| {
                                        let key_expr = self.eval_single_expr(&k, env)?;
                                        let value_expr = self.eval_single_expr(&v, env)?;
                                        let result = eval_to_string_pair(key_expr, value_expr)?;
                                        Ok(result)
                                    })
                                    .collect::<Result<Vec<_>, InterpreterError>>()?;

                                self.cost.charge(interpolate_cost(
                                    lhs.len() as i64,
                                    rhs.length() as i64,
                                ))?;

                                Ok(Expr {
                                    expr_instance: Some(ExprInstance::GString(interpolate(
                                        &lhs,
                                        &key_value_pairs,
                                    ))),
                                })
                            } else {
                                Ok(Expr {
                                    expr_instance: Some(ExprInstance::GString(lhs)),
                                })
                            }
                        }

                        (ExprInstance::GString(_), other) => {
                            Err(InterpreterError::OperatorExpectedError {
                                op: "%%".to_string(),
                                expected: String::from("Map"),
                                other_type: get_type(other),
                            })
                        }

                        (other, _) => Err(InterpreterError::OperatorNotDefined {
                            op: String::from("%%"),
                            other_type: get_type(other),
                        }),
                    }
                }

                ExprInstance::EPlusPlusBody(EPlusPlus { p1, p2 }) => {
                    self.cost.charge(op_call_cost())?;
                    let v1 = self.eval_single_expr(&p1.clone().unwrap(), env)?;
                    let v2 = self.eval_single_expr(&p2.clone().unwrap(), env)?;

                    match (v1.expr_instance.unwrap(), v2.expr_instance.unwrap()) {
                        (ExprInstance::GString(lhs), ExprInstance::GString(rhs)) => {
                            self.cost
                                .charge(string_append_cost(lhs.len() as i64, rhs.len() as i64))?;
                            Ok(Expr {
                                expr_instance: Some(ExprInstance::GString(lhs + &rhs)),
                            })
                        }

                        (ExprInstance::GByteArray(lhs), ExprInstance::GByteArray(rhs)) => {
                            self.cost.charge(byte_array_append_cost(lhs.clone()))?;
                            Ok(Expr {
                                expr_instance: Some(ExprInstance::GByteArray(
                                    lhs.into_iter().chain(rhs.into_iter()).collect(),
                                )),
                            })
                        }

                        (ExprInstance::EListBody(lhs), ExprInstance::EListBody(rhs)) => {
                            self.cost.charge(list_append_cost(lhs.clone().ps))?;
                            Ok(Expr {
                                expr_instance: Some(ExprInstance::EListBody(EList {
                                    ps: lhs.ps.into_iter().chain(rhs.ps.into_iter()).collect(),
                                    locally_free: union(lhs.locally_free, rhs.locally_free),
                                    connective_used: lhs.connective_used || rhs.connective_used,
                                    remainder: None,
                                })),
                            })
                        }

                        (ExprInstance::EMapBody(lhs), ExprInstance::EMapBody(rhs)) => {
                            let result_par = self.union_method().apply(
                                Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(ExprInstance::EMapBody(lhs)),
                                }]),
                                vec![Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(ExprInstance::EMapBody(rhs)),
                                }])],
                                env,
                            )?;
                            let result_expr = self.eval_single_expr(&result_par, env)?;
                            Ok(result_expr)
                        }

                        (ExprInstance::ESetBody(lhs), ExprInstance::ESetBody(rhs)) => {
                            let result_par = self.union_method().apply(
                                Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(ExprInstance::ESetBody(lhs)),
                                }]),
                                vec![Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(ExprInstance::ESetBody(rhs)),
                                }])],
                                env,
                            )?;
                            let result_expr = self.eval_single_expr(&result_par, env)?;
                            Ok(result_expr)
                        }

                        (ExprInstance::GString(_), other) => {
                            Err(InterpreterError::OperatorExpectedError {
                                op: "++".to_string(),
                                expected: String::from("String"),
                                other_type: get_type(other),
                            })
                        }

                        (ExprInstance::EListBody(_), other) => {
                            Err(InterpreterError::OperatorExpectedError {
                                op: "++".to_string(),
                                expected: String::from("List"),
                                other_type: get_type(other),
                            })
                        }

                        (ExprInstance::EMapBody(_), other) => {
                            Err(InterpreterError::OperatorExpectedError {
                                op: "++".to_string(),
                                expected: String::from("Map"),
                                other_type: get_type(other),
                            })
                        }

                        (ExprInstance::ESetBody(_), other) => {
                            Err(InterpreterError::OperatorExpectedError {
                                op: "++".to_string(),
                                expected: String::from("Set"),
                                other_type: get_type(other),
                            })
                        }

                        (other, _) => Err(InterpreterError::OperatorNotDefined {
                            op: String::from("++"),
                            other_type: get_type(other),
                        }),
                    }
                }

                ExprInstance::EMinusMinusBody(EMinusMinus { p1, p2 }) => {
                    self.cost.charge(op_call_cost())?;
                    let v1 = self.eval_single_expr(&p1.clone().unwrap(), env)?;
                    let v2 = self.eval_single_expr(&p2.clone().unwrap(), env)?;

                    match (v1.expr_instance.unwrap(), v2.expr_instance.unwrap()) {
                        (ExprInstance::ESetBody(lhs), ExprInstance::ESetBody(rhs)) => {
                            let result_par = self.diff_method().apply(
                                Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(ExprInstance::ESetBody(lhs)),
                                }]),
                                vec![Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(ExprInstance::ESetBody(rhs)),
                                }])],
                                env,
                            )?;
                            let result_expr = self.eval_single_expr(&result_par, env)?;
                            Ok(result_expr)
                        }

                        (ExprInstance::ESetBody(_), other) => {
                            Err(InterpreterError::OperatorExpectedError {
                                op: "--".to_string(),
                                expected: String::from("Set"),
                                other_type: get_type(other),
                            })
                        }

                        (other, _) => Err(InterpreterError::OperatorNotDefined {
                            op: String::from("--"),
                            other_type: get_type(other),
                        }),
                    }
                }

                ExprInstance::EVarBody(EVar { v }) => {
                    let p = self.eval_var(&v.clone().unwrap(), env)?;
                    let expr_val = self.eval_single_expr(&p, env)?;
                    Ok(expr_val)
                }

                ExprInstance::EListBody(e1) => {
                    let evaled_ps = e1
                        .ps
                        .iter()
                        .map(|p| self.eval_expr(p, env))
                        .collect::<Result<Vec<_>, InterpreterError>>()?;

                    let updated_ps: Vec<Par> = evaled_ps
                        .iter()
                        .map(|p| self.update_locally_free_par(p.clone()))
                        .collect();

                    Ok(Expr {
                        expr_instance: Some(ExprInstance::EListBody(
                            self.update_locally_free_elist(EList {
                                ps: updated_ps,
                                locally_free: e1.locally_free.clone(),
                                connective_used: e1.connective_used,
                                remainder: None,
                            }),
                        )),
                    })
                }

                ExprInstance::ETupleBody(e1) => {
                    let evaled_ps = e1
                        .ps
                        .iter()
                        .map(|p| self.eval_expr(p, env))
                        .collect::<Result<Vec<_>, InterpreterError>>()?;

                    let updated_ps: Vec<Par> = evaled_ps
                        .iter()
                        .map(|p| self.update_locally_free_par(p.clone()))
                        .collect();

                    Ok(Expr {
                        expr_instance: Some(ExprInstance::ETupleBody(
                            self.update_locally_free_etuple(ETuple {
                                ps: updated_ps,
                                locally_free: e1.locally_free.clone(),
                                connective_used: e1.connective_used,
                            }),
                        )),
                    })
                }

                ExprInstance::ESetBody(eset) => {
                    let set = ParSetTypeMapper::eset_to_par_set(eset.clone());
                    let evaled_ps = set
                        .ps
                        .sorted_pars
                        .iter()
                        .map(|p| self.eval_expr(p, env))
                        .collect::<Result<Vec<_>, InterpreterError>>()?;

                    let updated_ps: Vec<Par> = evaled_ps
                        .iter()
                        .map(|p| self.update_locally_free_par(p.clone()))
                        .collect();

                    let mut cloned_set = set.clone();
                    cloned_set.ps = SortedParHashSet::create_from_vec(updated_ps);
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::ESetBody(
                            ParSetTypeMapper::par_set_to_eset(cloned_set),
                        )),
                    })
                }

                ExprInstance::EMapBody(emap) => {
                    let map = ParMapTypeMapper::emap_to_par_map(emap.clone());
                    let evaled_ps = map
                        .ps
                        .clone()
                        .into_iter()
                        .map(|(k, v)| {
                            let e_key = self.eval_expr(&k, env)?;
                            let e_value = self.eval_expr(&v, env)?;
                            Ok((e_key, e_value))
                        })
                        .collect::<Result<Vec<_>, InterpreterError>>()?;

                    let mut cloned_map = map.clone();
                    cloned_map.ps = SortedParMap::create_from_vec(evaled_ps);
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::EMapBody(
                            ParMapTypeMapper::par_map_to_emap(cloned_map),
                        )),
                    })
                }

                ExprInstance::EPathmapBody(e1) => {
                    // Similar to EListBody - evaluate all elements
                    let evaled_ps = e1
                        .ps
                        .iter()
                        .map(|p| self.eval_expr(p, env))
                        .collect::<Result<Vec<_>, InterpreterError>>()?;

                    let updated_ps: Vec<Par> = evaled_ps
                        .iter()
                        .map(|p| self.update_locally_free_par(p.clone()))
                        .collect();

                    Ok(Expr {
                        expr_instance: Some(ExprInstance::EPathmapBody(EPathMap {
                            ps: updated_ps,
                            locally_free: e1.locally_free.clone(),
                            connective_used: e1.connective_used,
                            remainder: None,
                        })),
                    })
                }

                ExprInstance::EMethodBody(EMethod {
                    method_name,
                    target,
                    arguments,
                    ..
                }) => {
                    self.cost.charge(method_call_cost())?;
                    let evaled_target = self.eval_expr(&target.as_ref().unwrap(), env)?;
                    let evaled_args = arguments
                        .iter()
                        .map(|arg| self.eval_expr(arg, env))
                        .collect::<Result<Vec<_>, InterpreterError>>()?;

                    let result_par = match self.method_table().get(method_name) {
                        Some(method_function) => {
                            method_function.apply(evaled_target, evaled_args, env)?
                        }
                        None => {
                            return Err(InterpreterError::ReduceError(format!(
                                "Unimplemented method: {:?}",
                                method_name
                            )));
                        }
                    };

                    let result_expr = self.eval_single_expr(&result_par, env)?;
                    Ok(result_expr)
                }
            },
            None => Err(InterpreterError::ReduceError(format!(
                "Unimplemented expression: {:?}",
                expr
            ))),
        }
    }

    fn nth_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct NthMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> NthMethod<'a> {
            fn local_nth(&self, ps: &[Par], nth: usize) -> Result<Par, InterpreterError> {
                if ps.len() > nth {
                    Ok(ps[nth].clone())
                } else {
                    Err(InterpreterError::ReduceError(format!(
                        "Error: index out of bound: {}",
                        nth
                    )))
                }
            }
        }

        impl<'a> Method for NthMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: "nth".to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                self.outer.cost.charge(nth_method_call_cost())?;
                let nth = self.outer.eval_to_i64(&args[0], env)? as usize;
                let v = self.outer.eval_single_expr(&p, env)?;

                match v.expr_instance.unwrap() {
                    ExprInstance::EListBody(EList { ps, .. }) => self.local_nth(&ps, nth),
                    ExprInstance::ETupleBody(ETuple { ps, .. }) => self.local_nth(&ps, nth),
                    ExprInstance::GByteArray(bs) => {
                        if nth < bs.len() {
                            let b = bs[nth] & 0xff; // Convert to unsigned;
                            let p = new_gint_par(b as i64, Vec::new(), false);
                            Ok(p)
                        } else {
                            Err(InterpreterError::ReduceError(format!(
                                "Error: index out of bound: {}",
                                nth
                            )))
                        }
                    }
                    _ => Err(InterpreterError::ReduceError(String::from(
                        "Error: nth applied to something that wasn't a list or tuple.",
                    ))),
                }
            }
        }

        Box::new(NthMethod { outer: self })
    }

    fn to_byte_array_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct ToByteArrayMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> ToByteArrayMethod<'a> {
            fn serialize(&self, p: &Par) -> Result<Vec<u8>, InterpreterError> {
                Ok(p.encode_to_vec())
            }
        }

        impl<'a> Method for ToByteArrayMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: "toByteArray".to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }

                let expr_evaled = self.outer.eval_expr(&p, env)?;
                // println!("\nexpr_evaled in to_byte_array_method: {:?}", expr_evaled);
                let expr_subst =
                    self.outer
                        .substitute
                        .substitute_and_charge(&expr_evaled, 0, env)?;

                // println!("\nexpr_subst in to_byte_array_method: {:?}", expr_subst);

                self.outer.cost.charge(to_byte_array_cost(&expr_subst))?;
                let ba = self.serialize(&expr_subst)?;

                Ok(Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::GByteArray(ba)),
                }]))
            }
        }

        Box::new(ToByteArrayMethod { outer: self })
    }

    fn hex_to_bytes_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct HexToBytesMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> Method for HexToBytesMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                _env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("hexToBytes"),
                        expected: 0,
                        actual: args.len(),
                    });
                } else {
                    match single_expr(&p) {
                        Some(expr) => match unwrap_option_safe(expr.expr_instance)? {
                            ExprInstance::GString(encoded) => {
                                self.outer.cost.charge(hex_to_bytes_cost(&encoded))?;
                                Ok(Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(ExprInstance::GByteArray(
                                        StringOps::unsafe_decode_hex(encoded),
                                    )),
                                }]))
                            }

                            other => Err(InterpreterError::MethodNotDefined {
                                method: String::from("hexToBytes"),
                                other_type: get_type(other),
                            }),
                        },

                        None => Err(InterpreterError::ReduceError(String::from(
                            "Error: Method can only be called on singular expressions.",
                        ))),
                    }
                }
            }
        }

        Box::new(HexToBytesMethod { outer: self })
    }

    fn bytes_to_hex_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct BytesToHexMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> Method for BytesToHexMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                _env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("bytesToHex"),
                        expected: 0,
                        actual: args.len(),
                    });
                } else {
                    match single_expr(&p) {
                        Some(expr) => match expr.expr_instance.unwrap() {
                            ExprInstance::GByteArray(bytes) => {
                                self.outer.cost.charge(bytes_to_hex_cost(&bytes))?;

                                let str =
                                    bytes.iter().map(|byte| format!("{:02x}", byte)).collect();

                                Ok(new_gstring_par(str, Vec::new(), false))
                            }

                            other => Err(InterpreterError::MethodNotDefined {
                                method: String::from("BytesToHex"),
                                other_type: get_type(other),
                            }),
                        },

                        None => Err(InterpreterError::ReduceError(String::from(
                            "Error: Method can only be called on singular expressions.",
                        ))),
                    }
                }
            }
        }

        Box::new(BytesToHexMethod { outer: self })
    }

    fn to_utf8_bytes_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct ToUtf8BytesMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> Method for ToUtf8BytesMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                _env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("toUtf8Bytes"),
                        expected: 0,
                        actual: args.len(),
                    });
                } else {
                    match single_expr(&p) {
                        Some(expr) => match expr.expr_instance.unwrap() {
                            ExprInstance::GString(utf8_string) => {
                                self.outer.cost.charge(hex_to_bytes_cost(&utf8_string))?;

                                Ok(Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(ExprInstance::GByteArray(
                                        utf8_string.as_str().as_bytes().to_vec(),
                                    )),
                                }]))
                            }

                            other => Err(InterpreterError::MethodNotDefined {
                                method: String::from("toUtf8Bytes"),
                                other_type: get_type(other),
                            }),
                        },

                        None => Err(InterpreterError::ReduceError(String::from(
                            "Error: Method can only be called on singular expressions.",
                        ))),
                    }
                }
            }
        }

        Box::new(ToUtf8BytesMethod { outer: self })
    }

    fn union_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct UnionMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> UnionMethod<'a> {
            fn union(&self, base_expr: &Expr, other_expr: &Expr) -> Result<Expr, InterpreterError> {
                match (
                    base_expr.expr_instance.clone().unwrap(),
                    other_expr.expr_instance.clone().unwrap(),
                ) {
                    (ExprInstance::ESetBody(base_set), ExprInstance::ESetBody(other_set)) => {
                        let base_par_set = ParSetTypeMapper::eset_to_par_set(base_set);
                        let other_par_set = ParSetTypeMapper::eset_to_par_set(other_set);

                        let base_ps = base_par_set.ps;
                        let other_ps = other_par_set.ps;

                        self.outer
                            .cost
                            .charge(union_cost(other_ps.length() as i64))?;

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::ESetBody(
                                ParSetTypeMapper::par_set_to_eset(ParSet {
                                    ps: base_ps.union(other_ps.ps),
                                    connective_used: base_par_set.connective_used
                                        || other_par_set.connective_used,
                                    locally_free: union(
                                        base_par_set.locally_free,
                                        other_par_set.locally_free,
                                    ),
                                    remainder: None,
                                }),
                            )),
                        })
                    }

                    (ExprInstance::EMapBody(base_map), ExprInstance::EMapBody(other_map)) => {
                        let base_par_map = ParMapTypeMapper::emap_to_par_map(base_map);
                        let other_par_map = ParMapTypeMapper::emap_to_par_map(other_map.clone());

                        let mut base_sorted_par_map = base_par_map.ps;
                        let other_sorted_par_map = other_par_map.ps;

                        self.outer
                            .cost
                            .charge(union_cost(other_map.kvs.len() as i64))?;

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EMapBody(
                                ParMapTypeMapper::par_map_to_emap(ParMap::new(
                                    base_sorted_par_map
                                        .extend(other_sorted_par_map.into_iter().collect())
                                        .into_iter()
                                        .collect(),
                                    base_par_map.connective_used || other_par_map.connective_used,
                                    union(base_par_map.locally_free, other_par_map.locally_free),
                                    None,
                                )),
                            )),
                        })
                    }

                    (ExprInstance::EPathmapBody(base_pathmap), ExprInstance::EPathmapBody(other_pathmap)) => {
                        let base_rmap = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&base_pathmap);
                        let other_rmap = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&other_pathmap);

                        self.outer.cost.charge(union_cost(other_pathmap.ps.len() as i64))?;
                        let result_map = base_rmap.map.join(&other_rmap.map);

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(
                                PathMapCrateTypeMapper::rholang_pathmap_to_e_pathmap(
                                    &result_map,
                                    base_rmap.connective_used || other_rmap.connective_used,
                                    &union(base_rmap.locally_free, other_rmap.locally_free),
                                    None
                                ),
                            )),
                        })
                    }

                    (other, _) => Err(InterpreterError::MethodNotDefined {
                        method: String::from("union"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for UnionMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("union"),
                        expected: 1,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    let other_expr = self.outer.eval_single_expr(&args[0], env)?;
                    let result = self.union(&base_expr, &other_expr)?;
                    Ok(Par::default().with_exprs(vec![result]))
                }
            }
        }

        Box::new(UnionMethod { outer: self })
    }

    fn diff_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct DiffMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> DiffMethod<'a> {
            fn diff(&self, base_expr: &Expr, other_expr: &Expr) -> Result<Expr, InterpreterError> {
                match (
                    base_expr.expr_instance.clone().unwrap(),
                    other_expr.expr_instance.clone().unwrap(),
                ) {
                    (ExprInstance::ESetBody(base_set), ExprInstance::ESetBody(other_set)) => {
                        let base_par_set = ParSetTypeMapper::eset_to_par_set(base_set);
                        let other_par_set = ParSetTypeMapper::eset_to_par_set(other_set);

                        let base_ps = base_par_set.ps;
                        let other_ps = other_par_set.ps;

                        // diff is implemented in terms of foldLeft that at each step
                        // removes one element from the collection.
                        self.outer
                            .cost
                            .charge(diff_cost(other_ps.length() as i64))?;

                        let base_sorted_pars_set: HashSet<Par> =
                            base_ps.sorted_pars.into_iter().collect();
                        let other_sorted_pars_set: HashSet<Par> =
                            other_ps.sorted_pars.into_iter().collect();
                        let new_par_set = ParSet::create_from_vec(
                            base_sorted_pars_set
                                .difference(&other_sorted_pars_set)
                                .into_iter()
                                .cloned()
                                .collect(),
                        );

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::ESetBody(
                                ParSetTypeMapper::par_set_to_eset(new_par_set),
                            )),
                        })
                    }

                    (ExprInstance::EMapBody(base_emap), ExprInstance::EMapBody(other_emap)) => {
                        let base_par_map = ParMapTypeMapper::emap_to_par_map(base_emap);
                        let other_par_map = ParMapTypeMapper::emap_to_par_map(other_emap);

                        let mut base_ps = base_par_map.ps;
                        let other_ps = other_par_map.ps;

                        self.outer
                            .cost
                            .charge(diff_cost(other_ps.length() as i64))?;

                        let new_par_map = ParMap::create_from_sorted_par_map(
                            base_ps.remove_multiple(other_ps.keys()),
                        );

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EMapBody(
                                ParMapTypeMapper::par_map_to_emap(new_par_map),
                            )),
                        })
                    }

                    (ExprInstance::EPathmapBody(base_pathmap), ExprInstance::EPathmapBody(other_pathmap)) => {
                        let base_rmap = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&base_pathmap);
                        let other_rmap = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&other_pathmap);

                        self.outer.cost.charge(diff_cost(other_pathmap.ps.len() as i64))?;
                        let result_map = base_rmap.map.subtract(&other_rmap.map);

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(
                                PathMapCrateTypeMapper::rholang_pathmap_to_e_pathmap(
                                    &result_map,
                                    base_rmap.connective_used,
                                    &base_rmap.locally_free,
                                    None
                                ),
                            )),
                        })
                    }

                    (other, _) => Err(InterpreterError::MethodNotDefined {
                        method: String::from("diff"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for DiffMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("diff"),
                        expected: 1,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    let other_expr = self.outer.eval_single_expr(&args[0], env)?;
                    let result = self.diff(&base_expr, &other_expr)?;
                    Ok(Par::default().with_exprs(vec![result]))
                }
            }
        }

        Box::new(DiffMethod { outer: self })
    }

    fn intersection_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct IntersectionMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> IntersectionMethod<'a> {
            fn intersection(&self, base_expr: &Expr, other_expr: &Expr) -> Result<Expr, InterpreterError> {
                match (
                    base_expr.expr_instance.clone().unwrap(),
                    other_expr.expr_instance.clone().unwrap(),
                ) {
                    (ExprInstance::EPathmapBody(base_pathmap), ExprInstance::EPathmapBody(other_pathmap)) => {
                        let base_rmap = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&base_pathmap);
                        let other_rmap = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&other_pathmap);

                        self.outer.cost.charge(union_cost(other_pathmap.ps.len() as i64))?;
                        let result_map = base_rmap.map.meet(&other_rmap.map);

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(
                                PathMapCrateTypeMapper::rholang_pathmap_to_e_pathmap(
                                    &result_map,
                                    base_rmap.connective_used || other_rmap.connective_used,
                                    &union(base_rmap.locally_free, other_rmap.locally_free),
                                    None
                                ),
                            )),
                        })
                    }

                    (other, _) => Err(InterpreterError::MethodNotDefined {
                        method: String::from("intersection"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for IntersectionMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("intersection"),
                        expected: 1,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    let other_par = &args[0];
                    let other_expr = self.outer.eval_single_expr(other_par, env)?;
                    let result = self.intersection(&base_expr, &other_expr)?;
                    Ok(Par::default().with_exprs(vec![result]))
                }
            }
        }

        Box::new(IntersectionMethod { outer: self })
    }

    fn restriction_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct RestrictionMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> RestrictionMethod<'a> {
            fn restriction(&self, base_expr: &Expr, other_expr: &Expr) -> Result<Expr, InterpreterError> {
                match (
                    base_expr.expr_instance.clone().unwrap(),
                    other_expr.expr_instance.clone().unwrap(),
                ) {
                    (ExprInstance::EPathmapBody(base_pathmap), ExprInstance::EPathmapBody(other_pathmap)) => {
                        let base_rmap = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&base_pathmap);
                        let other_rmap = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&other_pathmap);

                        self.outer.cost.charge(union_cost(other_pathmap.ps.len() as i64))?;
                        let result_map = base_rmap.map.restrict(&other_rmap.map);

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(
                                PathMapCrateTypeMapper::rholang_pathmap_to_e_pathmap(
                                    &result_map,
                                    base_rmap.connective_used,
                                    &base_rmap.locally_free,
                                    None
                                ),
                            )),
                        })
                    }

                    (other, _) => Err(InterpreterError::MethodNotDefined {
                        method: String::from("restriction"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for RestrictionMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("restriction"),
                        expected: 1,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    let other_par = &args[0];
                    let other_expr = self.outer.eval_single_expr(other_par, env)?;
                    let result = self.restriction(&base_expr, &other_expr)?;
                    Ok(Par::default().with_exprs(vec![result]))
                }
            }
        }

        Box::new(RestrictionMethod { outer: self })
    }

    fn drop_head_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct DropHeadMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> DropHeadMethod<'a> {
            fn drop_head(&self, base_expr: &Expr, n: i64) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EPathmapBody(base_pathmap) => {
                        let base_rmap = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&base_pathmap);
                        if n < 0 {
                            return Err(InterpreterError::ReduceError(format!("dropHead argument must be non-negative, got: {}", n)));
                        }
                        self.outer.cost.charge(union_cost(n))?;

                        // For dropHead, we need to return a new EPathMap with modified path elements
                        // Instead of using PathMap, directly construct the result elements
                        let mut result_elements = Vec::new();
                        
                        for par in &base_pathmap.ps {
                            // Check if this Par is a list
                            if let Some(models::rhoapi::expr::ExprInstance::EListBody(list)) = par.exprs.first().and_then(|e| e.expr_instance.as_ref()) {
                                // It's a list - drop n elements from the beginning
                                if list.ps.len() > n as usize {
                                    let remaining = list.ps[(n as usize)..].to_vec();
                                    let new_list = models::rhoapi::EList {
                                        ps: remaining,
                                        locally_free: list.locally_free.clone(),
                                        connective_used: list.connective_used,
                                        remainder: list.remainder.clone(),
                                    };
                                    let new_par = Par {
                                        exprs: vec![models::rhoapi::Expr {
                                            expr_instance: Some(models::rhoapi::expr::ExprInstance::EListBody(new_list)),
                                        }],
                                        ..par.clone()
                                    };
                                    result_elements.push(new_par);
                                }
                                // If not enough elements, skip this entry
                            } else {
                                // Not a list - can't drop head, skip or keep as-is based on n
                                if n == 0 {
                                    result_elements.push(par.clone());
                                }
                                // If n > 0, we skip non-list entries
                            }
                        }
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(
                                models::rhoapi::EPathMap {
                                    ps: result_elements,
                                    locally_free: base_rmap.locally_free.clone(),
                                    connective_used: base_rmap.connective_used,
                                    remainder: None,
                                }
                            )),
                        })
                    }

                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("dropHead"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for DropHeadMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("dropHead"),
                        expected: 1,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    let n_par = &args[0];
                    let n = self.outer.eval_to_i64(n_par, env)?;
                    let result = self.drop_head(&base_expr, n)?;
                    Ok(Par::default().with_exprs(vec![result]))
                }
            }
        }

        Box::new(DropHeadMethod { outer: self })
    }

    fn run_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct RunMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> RunMethod<'a> {
            fn run(&self, base_expr: &Expr, _other_expr: &Expr) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EPathmapBody(base_pathmap) => {
                        // For run method, we ignore the other parameter and return self
                        self.outer.cost.charge(union_cost(1))?;
                        
                        // Simply return the base PathMap unchanged
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(base_pathmap)),
                        })
                    }

                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("run"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for RunMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("run"),
                        expected: 1,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    let other_expr = self.outer.eval_single_expr(&args[0], env)?;
                    let result = self.run(&base_expr, &other_expr)?;
                    Ok(Par::default().with_exprs(vec![result]))
                }
            }
        }

        Box::new(RunMethod { outer: self })
    }

    // ============ ZIPPER METHODS ============

    fn read_zipper_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct ReadZipperMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> ReadZipperMethod<'a> {
            fn create_read_zipper(&self, base_expr: &Expr) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EPathmapBody(pathmap) => {
                        // For now, we'll represent the zipper as a special PathMap with metadata
                        // In a full implementation, we'd need a custom Expr type for zippers
                        // For simplicity, return the pathmap with a marker that it's a "zipper"
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(pathmap)),
                        })
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("readZipper"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for ReadZipperMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("readZipper"),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                self.outer.cost.charge(union_cost(1))?;
                let result = self.create_read_zipper(&base_expr)?;
                Ok(Par::default().with_exprs(vec![result]))
            }
        }

        Box::new(ReadZipperMethod { outer: self })
    }

    fn read_zipper_at_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct ReadZipperAtMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> ReadZipperAtMethod<'a> {
            fn create_read_zipper_at(&self, base_expr: &Expr, _path: &Par) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EPathmapBody(pathmap) => {
                        // TODO: Filter PathMap to subtree at path
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(pathmap)),
                        })
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("readZipperAt"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for ReadZipperAtMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("readZipperAt"),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                let path = self.outer.eval_expr(&args[0], env)?;
                self.outer.cost.charge(union_cost(1))?;
                let result = self.create_read_zipper_at(&base_expr, &path)?;
                Ok(Par::default().with_exprs(vec![result]))
            }
        }

        Box::new(ReadZipperAtMethod { outer: self })
    }

    fn write_zipper_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct WriteZipperMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> WriteZipperMethod<'a> {
            fn create_write_zipper(&self, base_expr: &Expr) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EPathmapBody(pathmap) => {
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(pathmap)),
                        })
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("writeZipper"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for WriteZipperMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("writeZipper"),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                self.outer.cost.charge(union_cost(1))?;
                let result = self.create_write_zipper(&base_expr)?;
                Ok(Par::default().with_exprs(vec![result]))
            }
        }

        Box::new(WriteZipperMethod { outer: self })
    }

    fn write_zipper_at_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct WriteZipperAtMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> WriteZipperAtMethod<'a> {
            fn create_write_zipper_at(&self, base_expr: &Expr, _path: &Par) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EPathmapBody(pathmap) => {
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(pathmap)),
                        })
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("writeZipperAt"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for WriteZipperAtMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("writeZipperAt"),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                let path = self.outer.eval_expr(&args[0], env)?;
                self.outer.cost.charge(union_cost(1))?;
                let result = self.create_write_zipper_at(&base_expr, &path)?;
                Ok(Par::default().with_exprs(vec![result]))
            }
        }

        Box::new(WriteZipperAtMethod { outer: self })
    }

    fn descend_to_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct DescendToMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> DescendToMethod<'a> {
            fn descend_to(&self, base_expr: &Expr, _path: &Par) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EPathmapBody(pathmap) => {
                        // TODO: Navigate zipper to path
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(pathmap)),
                        })
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("descendTo"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for DescendToMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("descendTo"),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                let path = self.outer.eval_expr(&args[0], env)?;
                self.outer.cost.charge(union_cost(1))?;
                let result = self.descend_to(&base_expr, &path)?;
                Ok(Par::default().with_exprs(vec![result]))
            }
        }

        Box::new(DescendToMethod { outer: self })
    }

    fn get_val_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct GetValMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> GetValMethod<'a> {
            fn get_val(&self, base_expr: &Expr) -> Result<Par, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EPathmapBody(pathmap) => {
                        // For a zipper, get value at current position
                        // For now, just return first element as a placeholder
                        if let Some(first) = pathmap.ps.first() {
                            Ok(first.clone())
                        } else {
                            Ok(Par::default()) // Nil
                        }
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("getVal"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for GetValMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("getVal"),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                self.outer.cost.charge(lookup_cost())?;
                self.get_val(&base_expr)
            }
        }

        Box::new(GetValMethod { outer: self })
    }

    fn set_val_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct SetValMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> SetValMethod<'a> {
            fn set_val(&self, base_expr: &Expr, value: &Par) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EPathmapBody(mut pathmap) => {
                        // For a write zipper, set value at current position
                        // For now, add to the pathmap
                        pathmap.ps.push(value.clone());
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(pathmap)),
                        })
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("setVal"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for SetValMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("setVal"),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                let value = self.outer.eval_expr(&args[0], env)?;
                self.outer.cost.charge(add_cost())?;
                let result = self.set_val(&base_expr, &value)?;
                Ok(Par::default().with_exprs(vec![result]))
            }
        }

        Box::new(SetValMethod { outer: self })
    }

    fn remove_val_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct RemoveValMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> RemoveValMethod<'a> {
            fn remove_val(&self, base_expr: &Expr) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EPathmapBody(mut pathmap) => {
                        // Remove value at current position
                        pathmap.ps.pop();
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(pathmap)),
                        })
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("removeVal"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for RemoveValMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("removeVal"),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                self.outer.cost.charge(remove_cost())?;
                let result = self.remove_val(&base_expr)?;
                Ok(Par::default().with_exprs(vec![result]))
            }
        }

        Box::new(RemoveValMethod { outer: self })
    }

    fn remove_branches_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct RemoveBranchesMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> RemoveBranchesMethod<'a> {
            fn remove_branches(&self, base_expr: &Expr) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EPathmapBody(pathmap) => {
                        // Remove all branches below current position
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(models::rhoapi::EPathMap {
                                ps: vec![],
                                locally_free: pathmap.locally_free,
                                connective_used: pathmap.connective_used,
                                remainder: pathmap.remainder,
                            })),
                        })
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("removeBranches"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for RemoveBranchesMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("removeBranches"),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                self.outer.cost.charge(remove_cost())?;
                let result = self.remove_branches(&base_expr)?;
                Ok(Par::default().with_exprs(vec![result]))
            }
        }

        Box::new(RemoveBranchesMethod { outer: self })
    }

    fn graft_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct GraftMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> GraftMethod<'a> {
            fn graft(&self, base_expr: &Expr, source_expr: &Expr) -> Result<Expr, InterpreterError> {
                match (
                    base_expr.expr_instance.clone().unwrap(),
                    source_expr.expr_instance.clone().unwrap(),
                ) {
                    (ExprInstance::EPathmapBody(mut dest_pathmap), ExprInstance::EPathmapBody(source_pathmap)) => {
                        // Graft: copy subtrie from source to destination
                        dest_pathmap.ps.extend(source_pathmap.ps);
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(dest_pathmap)),
                        })
                    }
                    (other, _) => Err(InterpreterError::MethodNotDefined {
                        method: String::from("graft"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for GraftMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("graft"),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                let source_expr = self.outer.eval_single_expr(&args[0], env)?;
                self.outer.cost.charge(union_cost(1))?;
                let result = self.graft(&base_expr, &source_expr)?;
                Ok(Par::default().with_exprs(vec![result]))
            }
        }

        Box::new(GraftMethod { outer: self })
    }

    fn join_into_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct JoinIntoMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> JoinIntoMethod<'a> {
            fn join_into(&self, base_expr: &Expr, source_expr: &Expr) -> Result<Expr, InterpreterError> {
                match (
                    base_expr.expr_instance.clone().unwrap(),
                    source_expr.expr_instance.clone().unwrap(),
                ) {
                    (ExprInstance::EPathmapBody(base_pathmap), ExprInstance::EPathmapBody(source_pathmap)) => {
                        // JoinInto: union-merge subtries
                        let base_rmap = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&base_pathmap);
                        let source_rmap = PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&source_pathmap);

                        self.outer.cost.charge(union_cost(source_pathmap.ps.len() as i64))?;
                        let result_map = base_rmap.map.join(&source_rmap.map);

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(
                                PathMapCrateTypeMapper::rholang_pathmap_to_e_pathmap(
                                    &result_map,
                                    base_rmap.connective_used || source_rmap.connective_used,
                                    &union(base_rmap.locally_free, source_rmap.locally_free),
                                    None
                                ),
                            )),
                        })
                    }
                    (other, _) => Err(InterpreterError::MethodNotDefined {
                        method: String::from("joinInto"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for JoinIntoMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("joinInto"),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                let source_expr = self.outer.eval_single_expr(&args[0], env)?;
                self.outer.cost.charge(union_cost(1))?;
                let result = self.join_into(&base_expr, &source_expr)?;
                Ok(Par::default().with_exprs(vec![result]))
            }
        }

        Box::new(JoinIntoMethod { outer: self })
    }

    // ============ END ZIPPER METHODS ============

    fn add_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct AddMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> AddMethod<'a> {
            fn add(&self, base_expr: Expr, par: Par) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance {
                    Some(expr_instance) => match expr_instance {
                        ExprInstance::ESetBody(eset) => {
                            let base = ParSetTypeMapper::eset_to_par_set(eset);
                            let mut base_ps = base.ps;

                            Ok(Expr {
                                expr_instance: Some(ExprInstance::ESetBody(
                                    ParSetTypeMapper::par_set_to_eset(ParSet {
                                        ps: base_ps.insert(par.clone()),
                                        connective_used: base.connective_used
                                            || par.connective_used,
                                        locally_free: union(base.locally_free, par.locally_free),
                                        remainder: None,
                                    }),
                                )),
                            })
                        }

                        other => Err(InterpreterError::MethodNotDefined {
                            method: String::from("add"),
                            other_type: get_type(other),
                        }),
                    },

                    None => Err(InterpreterError::MethodNotDefined {
                        method: String::from("add"),
                        other_type: String::from("None"),
                    }),
                }
            }
        }

        impl<'a> Method for AddMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("add"),
                        expected: 1,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    let element = self.outer.eval_expr(&args[0], env)?;
                    self.outer.cost.charge(add_cost())?;
                    let result = self.add(base_expr, element)?;
                    Ok(Par::default().with_exprs(vec![result]))
                }
            }
        }

        Box::new(AddMethod { outer: self })
    }

    fn delete_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct DeleteMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> DeleteMethod<'a> {
            fn delete(&self, base_expr: Expr, par: Par) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance {
                    Some(expr_instance) => match expr_instance {
                        ExprInstance::ESetBody(eset) => {
                            let base = ParSetTypeMapper::eset_to_par_set(eset);
                            let mut base_ps = base.ps;

                            Ok(Expr {
                                expr_instance: Some(ExprInstance::ESetBody(
                                    ParSetTypeMapper::par_set_to_eset(ParSet {
                                        ps: base_ps.remove(par.clone()),
                                        connective_used: base.connective_used
                                            || par.connective_used,
                                        locally_free: union(base.locally_free, par.locally_free),
                                        remainder: None,
                                    }),
                                )),
                            })
                        }

                        ExprInstance::EMapBody(emap) => {
                            let base = ParMapTypeMapper::emap_to_par_map(emap);
                            let mut base_ps = base.ps;

                            Ok(Expr {
                                expr_instance: Some(ExprInstance::EMapBody(
                                    ParMapTypeMapper::par_map_to_emap(ParMap {
                                        ps: base_ps.remove(par.clone()),
                                        connective_used: base.connective_used
                                            || par.connective_used,
                                        locally_free: union(base.locally_free, par.locally_free),
                                        remainder: None,
                                    }),
                                )),
                            })
                        }

                        other => Err(InterpreterError::MethodNotDefined {
                            method: String::from("delete"),
                            other_type: get_type(other),
                        }),
                    },

                    None => Err(InterpreterError::MethodNotDefined {
                        method: String::from("delete"),
                        other_type: String::from("None"),
                    }),
                }
            }
        }

        impl<'a> Method for DeleteMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("delete"),
                        expected: 1,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    let element = self.outer.eval_expr(&args[0], env)?;
                    //TODO(mateusz.gorski): think whether deletion of an element from the collection should dependent on the collection type/size - OLD
                    self.outer.cost.charge(remove_cost())?;
                    let result = self.delete(base_expr, element)?;
                    Ok(Par::default().with_exprs(vec![result]))
                }
            }
        }

        Box::new(DeleteMethod { outer: self })
    }

    fn contains_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct ContainsMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> ContainsMethod<'a> {
            fn contains(&self, base_expr: Expr, par: Par) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance {
                    Some(expr_instance) => match expr_instance {
                        ExprInstance::ESetBody(eset) => {
                            let base_ps = ParSetTypeMapper::eset_to_par_set(eset).ps;

                            Ok(Expr {
                                expr_instance: Some(ExprInstance::GBool(base_ps.contains(par))),
                            })
                        }

                        ExprInstance::EMapBody(emap) => {
                            let base_ps = ParMapTypeMapper::emap_to_par_map(emap).ps;

                            Ok(Expr {
                                expr_instance: Some(ExprInstance::GBool(base_ps.contains(par))),
                            })
                        }

                        other => Err(InterpreterError::MethodNotDefined {
                            method: String::from("contains"),
                            other_type: get_type(other),
                        }),
                    },

                    None => Err(InterpreterError::MethodNotDefined {
                        method: String::from("contains"),
                        other_type: String::from("None"),
                    }),
                }
            }
        }

        impl<'a> Method for ContainsMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("contains"),
                        expected: 1,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    let element = self.outer.eval_expr(&args[0], env)?;
                    self.outer.cost.charge(lookup_cost())?;
                    let result = self.contains(base_expr, element)?;
                    Ok(Par::default().with_exprs(vec![result]))
                }
            }
        }

        Box::new(ContainsMethod { outer: self })
    }

    fn get_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct GetMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> GetMethod<'a> {
            fn get(&self, base_expr: Expr, key: Par) -> Result<Par, InterpreterError> {
                match base_expr.expr_instance {
                    Some(expr_instance) => match expr_instance {
                        ExprInstance::EMapBody(emap) => {
                            let base_ps = ParMapTypeMapper::emap_to_par_map(emap).ps;
                            Ok(base_ps.get_or_else(key, Par::default()))
                        }

                        other => Err(InterpreterError::MethodNotDefined {
                            method: String::from("get"),
                            other_type: get_type(other),
                        }),
                    },

                    None => Err(InterpreterError::MethodNotDefined {
                        method: String::from("get"),
                        other_type: String::from("None"),
                    }),
                }
            }
        }

        impl<'a> Method for GetMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("get"),
                        expected: 1,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    let key = self.outer.eval_expr(&args[0], env)?;
                    self.outer.cost.charge(lookup_cost())?;
                    let result = self.get(base_expr, key)?;
                    Ok(result)
                }
            }
        }

        Box::new(GetMethod { outer: self })
    }

    fn get_or_else_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct GetOrElseMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> GetOrElseMethod<'a> {
            fn get_or_else(
                &self,
                base_expr: Expr,
                key: Par,
                default: Par,
            ) -> Result<Par, InterpreterError> {
                match base_expr.expr_instance {
                    Some(expr_instance) => match expr_instance {
                        ExprInstance::EMapBody(emap) => {
                            let base_ps = ParMapTypeMapper::emap_to_par_map(emap).ps;
                            Ok(base_ps.get_or_else(key, default))
                        }

                        other => Err(InterpreterError::MethodNotDefined {
                            method: String::from("get_or_else"),
                            other_type: get_type(other),
                        }),
                    },

                    None => Err(InterpreterError::MethodNotDefined {
                        method: String::from("get_or_else"),
                        other_type: String::from("None"),
                    }),
                }
            }
        }

        impl<'a> Method for GetOrElseMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 2 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("get_or_else"),
                        expected: 2,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    let key = self.outer.eval_expr(&args[0], env)?;
                    let default = self.outer.eval_expr(&args[1], env)?;
                    self.outer.cost.charge(lookup_cost())?;
                    let result = self.get_or_else(base_expr, key, default)?;
                    Ok(result)
                }
            }
        }

        Box::new(GetOrElseMethod { outer: self })
    }

    fn set_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct SetMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> SetMethod<'a> {
            fn set(&self, base_expr: Expr, key: Par, value: Par) -> Result<Par, InterpreterError> {
                match base_expr.expr_instance {
                    Some(expr_instance) => match expr_instance {
                        ExprInstance::EMapBody(emap) => {
                            let mut base_ps = ParMapTypeMapper::emap_to_par_map(emap).ps;
                            // let sorted_par_map = base_ps.insert((key, value));
                            let par_map =
                                ParMap::create_from_sorted_par_map(base_ps.insert((key, value)));

                            // println!("\nsorted_par_map in set_method: {:?}", sorted_par_map);
                            // println!("\npar_map in set_method: {:?}", par_map);

                            Ok(Par::default().with_exprs(vec![Expr {
                                expr_instance: Some(ExprInstance::EMapBody(
                                    ParMapTypeMapper::par_map_to_emap(par_map),
                                )),
                            }]))
                        }

                        other => Err(InterpreterError::MethodNotDefined {
                            method: String::from("set"),
                            other_type: get_type(other),
                        }),
                    },

                    None => Err(InterpreterError::MethodNotDefined {
                        method: String::from("set"),
                        other_type: String::from("None"),
                    }),
                }
            }
        }

        impl<'a> Method for SetMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 2 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("set"),
                        expected: 2,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    let key = self.outer.eval_expr(&args[0], env)?;
                    let value = self.outer.eval_expr(&args[1], env)?;
                    self.outer.cost.charge(add_cost())?;
                    let result = self.set(base_expr, key, value)?;
                    Ok(result)
                }
            }
        }

        Box::new(SetMethod { outer: self })
    }

    fn keys_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct KeysMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> KeysMethod<'a> {
            fn keys(&self, base_expr: Expr) -> Result<Par, InterpreterError> {
                match base_expr.expr_instance {
                    Some(expr_instance) => match expr_instance {
                        ExprInstance::EMapBody(emap) => {
                            let base_ps = ParMapTypeMapper::emap_to_par_map(emap).ps;
                            let par_set = ParSet::create_from_vec(base_ps.keys());

                            Ok(Par::default().with_exprs(vec![Expr {
                                expr_instance: Some(ExprInstance::ESetBody(
                                    ParSetTypeMapper::par_set_to_eset(par_set),
                                )),
                            }]))
                        }

                        other => Err(InterpreterError::MethodNotDefined {
                            method: String::from("keys"),
                            other_type: get_type(other),
                        }),
                    },

                    None => Err(InterpreterError::MethodNotDefined {
                        method: String::from("keys"),
                        other_type: String::from("None"),
                    }),
                }
            }
        }

        impl<'a> Method for KeysMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("keys"),
                        expected: 0,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    self.outer.cost.charge(keys_method_cost())?;
                    let result = self.keys(base_expr)?;
                    Ok(result)
                }
            }
        }

        Box::new(KeysMethod { outer: self })
    }

    fn size_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct SizeMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> SizeMethod<'a> {
            fn size(&self, base_expr: Expr) -> Result<(i64, Par), InterpreterError> {
                match base_expr.expr_instance {
                    Some(expr_instance) => match expr_instance {
                        ExprInstance::EMapBody(emap) => {
                            let base_ps = ParMapTypeMapper::emap_to_par_map(emap).ps;
                            let size = base_ps.length() as i64;

                            Ok((size, new_gint_par(size, Vec::new(), false)))
                        }

                        ExprInstance::ESetBody(eset) => {
                            let base_ps = ParSetTypeMapper::eset_to_par_set(eset).ps;
                            let size = base_ps.length() as i64;

                            Ok((size, new_gint_par(size, Vec::new(), false)))
                        }

                        other => Err(InterpreterError::MethodNotDefined {
                            method: String::from("size"),
                            other_type: get_type(other),
                        }),
                    },

                    None => Err(InterpreterError::MethodNotDefined {
                        method: String::from("size"),
                        other_type: String::from("None"),
                    }),
                }
            }
        }

        impl<'a> Method for SizeMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("size"),
                        expected: 0,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    let result = self.size(base_expr)?;
                    self.outer.cost.charge(size_method_cost(result.0))?;
                    Ok(result.1)
                }
            }
        }

        Box::new(SizeMethod { outer: self })
    }

    fn length_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct LengthMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> LengthMethod<'a> {
            fn length(&self, base_expr: Expr) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance {
                    Some(expr_instance) => match expr_instance {
                        ExprInstance::GString(string) => Ok(new_gint_expr(string.len() as i64)),

                        ExprInstance::GByteArray(bytes) => Ok(new_gint_expr(bytes.len() as i64)),

                        ExprInstance::EListBody(elist) => Ok(new_gint_expr(elist.ps.len() as i64)),

                        other => Err(InterpreterError::MethodNotDefined {
                            method: String::from("length"),
                            other_type: get_type(other),
                        }),
                    },

                    None => Err(InterpreterError::MethodNotDefined {
                        method: String::from("length"),
                        other_type: String::from("None"),
                    }),
                }
            }
        }

        impl<'a> Method for LengthMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("length"),
                        expected: 0,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    self.outer.cost.charge(length_method_cost())?;
                    let result = self.length(base_expr)?;
                    Ok(Par::default().with_exprs(vec![result]))
                }
            }
        }

        Box::new(LengthMethod { outer: self })
    }

    fn slice_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct SliceMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> SliceMethod<'a> {
            fn slice(
                &self,
                base_expr: Expr,
                from: usize,
                until: usize,
            ) -> Result<Par, InterpreterError> {
                match base_expr.expr_instance {
                    Some(expr_instance) => match expr_instance {
                        ExprInstance::GString(string) => Ok(new_gstring_par(
                            if from <= until && until <= string.len() {
                                string[from..until].to_string()
                            } else {
                                "".to_string()
                            },
                            Vec::new(),
                            false,
                        )),

                        ExprInstance::EListBody(elist) => Ok(new_elist_par(
                            if from <= until && until <= elist.ps.len() {
                                elist.ps[from..until].to_vec()
                            } else {
                                vec![]
                            },
                            elist.locally_free,
                            elist.connective_used,
                            elist.remainder,
                            Vec::new(),
                            false,
                        )),

                        ExprInstance::GByteArray(bytes) => {
                            Ok(Par::default().with_exprs(vec![Expr {
                                expr_instance: Some(ExprInstance::GByteArray(
                                    if from <= until && until <= bytes.len() {
                                        bytes[from..until].to_vec()
                                    } else {
                                        vec![]
                                    },
                                )),
                            }]))
                        }

                        other => Err(InterpreterError::MethodNotDefined {
                            method: String::from("slice"),
                            other_type: get_type(other),
                        }),
                    },

                    None => Err(InterpreterError::MethodNotDefined {
                        method: String::from("slice"),
                        other_type: String::from("None"),
                    }),
                }
            }
        }

        impl<'a> Method for SliceMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 2 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("slice"),
                        expected: 2,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    let from_arg = self.outer.eval_to_i64(&args[0], env)?;
                    let to_arg = self.outer.eval_to_i64(&args[1], env)?;
                    self.outer.cost.charge(slice_cost(to_arg))?;
                    let result = self.slice(
                        base_expr,
                        if from_arg > 0 { from_arg as usize } else { 0 },
                        if to_arg > 0 { to_arg as usize } else { 0 },
                    )?;
                    Ok(result)
                }
            }
        }

        Box::new(SliceMethod { outer: self })
    }

    fn take_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct TakeMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> TakeMethod<'a> {
            fn take(&self, base_expr: Expr, n: usize) -> Result<Par, InterpreterError> {
                match base_expr.expr_instance {
                    Some(expr_instance) => match expr_instance {
                        ExprInstance::EListBody(elist) => Ok(new_elist_par(
                            elist.ps.into_iter().take(n).collect(),
                            elist.locally_free,
                            elist.connective_used,
                            elist.remainder,
                            Vec::new(),
                            false,
                        )),

                        other => Err(InterpreterError::MethodNotDefined {
                            method: String::from("take"),
                            other_type: get_type(other),
                        }),
                    },

                    None => Err(InterpreterError::MethodNotDefined {
                        method: String::from("take"),
                        other_type: String::from("None"),
                    }),
                }
            }
        }

        impl<'a> Method for TakeMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("take"),
                        expected: 1,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    let n_arg = self.outer.eval_to_i64(&args[0], env)?;
                    self.outer.cost.charge(take_cost(n_arg))?;
                    let result = self.take(base_expr, n_arg as usize)?;
                    Ok(result)
                }
            }
        }

        Box::new(TakeMethod { outer: self })
    }

    fn to_list_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct ToListMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> ToListMethod<'a> {
            fn to_list(&self, base_expr: Expr) -> Result<Par, InterpreterError> {
                match base_expr.expr_instance {
                    Some(expr_instance) => match expr_instance {
                        ExprInstance::EListBody(elist) => {
                            Ok(Par::default().with_exprs(vec![Expr {
                                expr_instance: Some(ExprInstance::EListBody(elist)),
                            }]))
                        }

                        ExprInstance::ESetBody(eset) => {
                            let ps = ParSetTypeMapper::eset_to_par_set(eset).ps;
                            self.outer.cost.charge(to_list_cost(ps.length() as i64))?;

                            Ok(Par::default().with_exprs(vec![Expr {
                                expr_instance: Some(ExprInstance::EListBody(EList {
                                    ps: ps.sorted_pars,
                                    locally_free: Vec::new(),
                                    connective_used: false,
                                    remainder: None,
                                })),
                            }]))
                        }

                        ExprInstance::EMapBody(emap) => {
                            let ps = ParMapTypeMapper::emap_to_par_map(emap).ps;
                            self.outer.cost.charge(to_list_cost(ps.length() as i64))?;

                            Ok(Par::default().with_exprs(vec![Expr {
                                expr_instance: Some(ExprInstance::EListBody(EList {
                                    ps: ps
                                        .sorted_list
                                        .into_iter()
                                        .map(|(k, v)| {
                                            Par::default().with_exprs(vec![Expr {
                                                expr_instance: Some(ExprInstance::ETupleBody(
                                                    ETuple {
                                                        ps: vec![k, v],
                                                        locally_free: Vec::new(),
                                                        connective_used: false,
                                                    },
                                                )),
                                            }])
                                        })
                                        .collect(),
                                    locally_free: Vec::new(),
                                    connective_used: false,
                                    remainder: None,
                                })),
                            }]))
                        }

                        ExprInstance::ETupleBody(etuple) => {
                            let ps = etuple.ps;
                            self.outer.cost.charge(to_list_cost(ps.len() as i64))?;

                            Ok(Par::default().with_exprs(vec![Expr {
                                expr_instance: Some(ExprInstance::EListBody(EList {
                                    ps: ps,
                                    locally_free: Vec::new(),
                                    connective_used: false,
                                    remainder: None,
                                })),
                            }]))
                        }

                        other => Err(InterpreterError::MethodNotDefined {
                            method: String::from("to_list"),
                            other_type: get_type(other),
                        }),
                    },

                    None => Err(InterpreterError::MethodNotDefined {
                        method: String::from("to_list"),
                        other_type: String::from("None"),
                    }),
                }
            }
        }

        impl<'a> Method for ToListMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("to_list"),
                        expected: 0,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    let result = self.to_list(base_expr)?;
                    Ok(result)
                }
            }
        }

        Box::new(ToListMethod { outer: self })
    }

    fn to_set_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct ToSetMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> ToSetMethod<'a> {
            fn to_set(&self, base_expr: Expr) -> Result<Par, InterpreterError> {
                match base_expr.expr_instance {
                    Some(expr_instance) => match expr_instance {
                        ExprInstance::ESetBody(eset) => Ok(Par::default().with_exprs(vec![Expr {
                            expr_instance: Some(ExprInstance::ESetBody(eset)),
                        }])),

                        ExprInstance::EMapBody(emap) => {
                            let map = ParMapTypeMapper::emap_to_par_map(emap);

                            Ok(Par::default().with_exprs(vec![Expr {
                                expr_instance: Some(ExprInstance::ESetBody(
                                    ParSetTypeMapper::par_set_to_eset(ParSet::new(
                                        map.ps
                                            .into_iter()
                                            .map(|t| {
                                                Par::default().with_exprs(vec![Expr {
                                                    expr_instance: Some(ExprInstance::ETupleBody(
                                                        ETuple {
                                                            ps: vec![t.0, t.1],
                                                            locally_free: Vec::new(),
                                                            connective_used: false,
                                                        },
                                                    )),
                                                }])
                                            })
                                            .collect(),
                                        map.connective_used,
                                        map.locally_free,
                                        map.remainder,
                                    )),
                                )),
                            }]))
                        }

                        ExprInstance::EListBody(elist) => {
                            Ok(Par::default().with_exprs(vec![Expr {
                                expr_instance: Some(ExprInstance::ESetBody(
                                    ParSetTypeMapper::par_set_to_eset(ParSet::new(
                                        elist.ps,
                                        elist.connective_used,
                                        elist.locally_free,
                                        elist.remainder,
                                    )),
                                )),
                            }]))
                        }

                        other => Err(InterpreterError::MethodNotDefined {
                            method: String::from("to_set"),
                            other_type: get_type(other),
                        }),
                    },

                    None => Err(InterpreterError::MethodNotDefined {
                        method: String::from("to_set"),
                        other_type: String::from("None"),
                    }),
                }
            }
        }

        impl<'a> Method for ToSetMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("to_set"),
                        expected: 0,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    let result = self.to_set(base_expr)?;
                    Ok(result)
                }
            }
        }

        Box::new(ToSetMethod { outer: self })
    }

    fn to_map_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct ToMapMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> ToMapMethod<'a> {
            fn make_map(
                &self,
                ps: Vec<Par>,
                connective_used: bool,
                locally_free: Vec<u8>,
                remainder: Option<Var>,
            ) -> Result<Par, InterpreterError> {
                let key_pairs: Vec<Option<(Par, Par)>> =
                    ps.into_iter().map(|p| RhoTuple2::unapply(p)).collect();

                if key_pairs.iter().any(|pair| !pair.is_some()) {
                    Err(InterpreterError::MethodNotDefined {
                        method: String::from("to_map"),
                        other_type: String::from("types except List[(K,V)]"),
                    })
                } else {
                    Ok(new_emap_par(
                        key_pairs
                            .into_iter()
                            .map(|pair| {
                                let (key, value) = pair.unwrap();
                                KeyValuePair {
                                    key: Some(key),
                                    value: Some(value),
                                }
                            })
                            .collect(),
                        locally_free,
                        connective_used,
                        remainder,
                        Vec::new(),
                        false,
                    ))
                }
            }

            fn to_map(&self, base_expr: Expr) -> Result<Par, InterpreterError> {
                match base_expr.expr_instance {
                    Some(expr_instance) => match expr_instance {
                        ExprInstance::EMapBody(emap) => Ok(Par::default().with_exprs(vec![Expr {
                            expr_instance: Some(ExprInstance::EMapBody(emap)),
                        }])),

                        ExprInstance::ESetBody(eset) => {
                            let base = ParSetTypeMapper::eset_to_par_set(eset);
                            self.make_map(
                                base.ps.sorted_pars,
                                base.connective_used,
                                base.locally_free,
                                base.remainder,
                            )
                        }

                        ExprInstance::EListBody(elist) => self.make_map(
                            elist.ps,
                            elist.connective_used,
                            elist.locally_free,
                            elist.remainder,
                        ),

                        other => Err(InterpreterError::MethodNotDefined {
                            method: String::from("to_map"),
                            other_type: get_type(other),
                        }),
                    },

                    None => Err(InterpreterError::MethodNotDefined {
                        method: String::from("to_map"),
                        other_type: String::from("None"),
                    }),
                }
            }
        }

        impl<'a> Method for ToMapMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("to_map"),
                        expected: 0,
                        actual: args.len(),
                    });
                } else {
                    let base_expr = self.outer.eval_single_expr(&p, env)?;
                    let result = self.to_map(base_expr)?;
                    Ok(result)
                }
            }
        }

        Box::new(ToMapMethod { outer: self })
    }

    fn to_string_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct ToStringMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> ToStringMethod<'a> {
            fn to_string(&self, un: &GUnforgeable) -> Result<Par, InterpreterError> {
                let unf_instance =
                    un.unf_instance
                        .as_ref()
                        .ok_or_else(|| InterpreterError::MethodNotDefined {
                            method: String::from("to_string"),
                            other_type: String::from("None"),
                        })?;

                match unf_instance {
                    UnfInstance::GDeployIdBody(deploy_id) => {
                        Ok(Par::default().with_exprs(vec![Expr {
                            expr_instance: Some(ExprInstance::GString(hex::encode(&deploy_id.sig))),
                        }]))
                    }

                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("to_string"),
                        other_type: get_unforgeable_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for ToStringMethod<'a> {
            fn apply(&self, p: Par, args: Vec<Par>, _: &Env<Par>) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("to_map"),
                        expected: 0,
                        actual: args.len(),
                    });
                } else {
                    let un = self.outer.eval_single_unforgeable(&p)?;
                    let result = self.to_string(un)?;
                    Ok(result)
                }
            }
        }

        Box::new(ToStringMethod { outer: self })
    }

    fn method_table<'a>(&'a self) -> HashMap<String, Box<dyn Method + 'a>> {
        let mut table = HashMap::new();
        table.insert("nth".to_string(), self.nth_method());
        table.insert("toByteArray".to_string(), self.to_byte_array_method());
        table.insert("hexToBytes".to_string(), self.hex_to_bytes_method());
        table.insert("bytesToHex".to_string(), self.bytes_to_hex_method());
        table.insert("toUtf8Bytes".to_string(), self.to_utf8_bytes_method());
        table.insert("union".to_string(), self.union_method());
        table.insert("diff".to_string(), self.diff_method());
        table.insert("intersection".to_string(), self.intersection_method());
        table.insert("restriction".to_string(), self.restriction_method());
        table.insert("dropHead".to_string(), self.drop_head_method());
        table.insert("run".to_string(), self.run_method());
        // Zipper methods
        table.insert("readZipper".to_string(), self.read_zipper_method());
        table.insert("readZipperAt".to_string(), self.read_zipper_at_method());
        table.insert("writeZipper".to_string(), self.write_zipper_method());
        table.insert("writeZipperAt".to_string(), self.write_zipper_at_method());
        table.insert("descendTo".to_string(), self.descend_to_method());
        table.insert("getVal".to_string(), self.get_val_method());
        table.insert("setVal".to_string(), self.set_val_method());
        table.insert("removeVal".to_string(), self.remove_val_method());
        table.insert("removeBranches".to_string(), self.remove_branches_method());
        table.insert("graft".to_string(), self.graft_method());
        table.insert("joinInto".to_string(), self.join_into_method());
        table.insert("add".to_string(), self.add_method());
        table.insert("delete".to_string(), self.delete_method());
        table.insert("contains".to_string(), self.contains_method());
        table.insert("get".to_string(), self.get_method());
        table.insert("getOrElse".to_string(), self.get_or_else_method());
        table.insert("set".to_string(), self.set_method());
        table.insert("keys".to_string(), self.keys_method());
        table.insert("size".to_string(), self.size_method());
        table.insert("length".to_string(), self.length_method());
        table.insert("slice".to_string(), self.slice_method());
        table.insert("take".to_string(), self.take_method());
        table.insert("toList".to_string(), self.to_list_method());
        table.insert("toSet".to_string(), self.to_set_method());
        table.insert("toMap".to_string(), self.to_map_method());
        table.insert("toString".to_string(), self.to_string_method());
        table
    }

    fn eval_single_expr(&self, p: &Par, env: &Env<Par>) -> Result<Expr, InterpreterError> {
        if !p.sends.is_empty()
            && !p.receives.is_empty()
            && !p.news.is_empty()
            && !p.matches.is_empty()
            && !p.unforgeables.is_empty()
            && !p.bundles.is_empty()
        {
            Err(InterpreterError::ReduceError(String::from(
                "Error: parallel or non expression found where expression expected.",
            )))
        } else {
            match p.exprs.as_slice() {
                [e] => Ok(self.eval_expr_to_expr(&e, env)?),

                _ => Err(InterpreterError::ReduceError(
                    "Error: Multiple expressions given.".to_string(),
                )),
            }
        }
    }

    fn eval_single_unforgeable<'a>(
        &self,
        p: &'a Par,
    ) -> Result<&'a GUnforgeable, InterpreterError> {
        if !p.sends.is_empty()
            && !p.receives.is_empty()
            && !p.news.is_empty()
            && !p.matches.is_empty()
            && !p.exprs.is_empty()
            && !p.bundles.is_empty()
        {
            Err(InterpreterError::ReduceError(String::from(
                "Error: non unforgeable found where unforgeable expected.",
            )))
        } else {
            match p.unforgeables.as_slice() {
                [e] => Ok(e),

                _ => Err(InterpreterError::ReduceError(
                    "Error: Multiple unforgeables given.".to_string(),
                )),
            }
        }
    }

    fn eval_to_i64(&self, p: &Par, env: &Env<Par>) -> Result<i64, InterpreterError> {
        if !p.sends.is_empty()
            && !p.receives.is_empty()
            && !p.news.is_empty()
            && !p.matches.is_empty()
            && !p.unforgeables.is_empty()
            && !p.bundles.is_empty()
        {
            Err(InterpreterError::ReduceError(String::from(
                "Error: parallel or non expression found where expression expected.",
            )))
        } else {
            // println!("\np: {:?}", p);
            // println!("\np.exprs: {:?}", p.exprs);
            match p.exprs.as_slice() {
                [Expr {
                    expr_instance: Some(ExprInstance::GInt(v)),
                }] => Ok(*v),

                [Expr {
                    expr_instance: Some(ExprInstance::EVarBody(EVar { v })),
                }] => {
                    let p = self.eval_var(&unwrap_option_safe(v.clone())?, env)?;
                    self.eval_to_i64(&p, env)
                }

                [e] => {
                    let evaled = self.eval_expr_to_expr(&e, env)?;

                    match evaled.expr_instance {
                        Some(expr_instance) => match expr_instance {
                            ExprInstance::GInt(v) => Ok(v),

                            _ => Err(InterpreterError::ReduceError(
                                "Error: expression didn't evaluate to integer.".to_string(),
                            )),
                        },
                        None => Err(InterpreterError::MethodNotDefined {
                            method: String::from("expr_instance"),
                            other_type: String::from("None"),
                        }),
                    }
                }

                _ => Err(InterpreterError::ReduceError(
                    "Error: Integer expected, or unimplemented expression.".to_string(),
                )),
            }
        }
    }

    fn eval_to_bool(&self, p: &Par, env: &Env<Par>) -> Result<bool, InterpreterError> {
        if !p.sends.is_empty()
            && !p.receives.is_empty()
            && !p.news.is_empty()
            && !p.matches.is_empty()
            && !p.unforgeables.is_empty()
            && !p.bundles.is_empty()
        {
            Err(InterpreterError::ReduceError(String::from(
                "Error: parallel or non expression found where expression expected.",
            )))
        } else {
            match p.exprs.as_slice() {
                [Expr {
                    expr_instance: Some(ExprInstance::GBool(b)),
                }] => Ok(*b),

                [Expr {
                    expr_instance: Some(ExprInstance::EVarBody(EVar { v })),
                }] => {
                    let p = self.eval_var(&unwrap_option_safe(v.clone())?, env)?;
                    self.eval_to_bool(&p, env)
                }

                [e] => {
                    let evaled = self.eval_expr_to_expr(&e, env)?;

                    match evaled.expr_instance {
                        Some(expr_instance) => match expr_instance {
                            ExprInstance::GBool(b) => Ok(b),

                            _ => Err(InterpreterError::ReduceError(
                                "Error: expression didn't evaluate to boolean.".to_string(),
                            )),
                        },
                        None => Err(InterpreterError::MethodNotDefined {
                            method: String::from("expr_instance"),
                            other_type: String::from("None"),
                        }),
                    }
                }

                _ => Err(InterpreterError::ReduceError(
                    "Error: Multiple expressions given.".to_string(),
                )),
            }
        }
    }

    fn update_locally_free_par(&self, mut par: Par) -> Par {
        let mut locally_free = Vec::new();

        locally_free = union(
            locally_free,
            par.sends
                .iter()
                .flat_map(|send| send.locally_free.clone())
                .collect(),
        );

        locally_free = union(
            locally_free,
            par.receives
                .iter()
                .flat_map(|receive| receive.locally_free.clone())
                .collect(),
        );

        locally_free = union(
            locally_free,
            par.news
                .iter()
                .flat_map(|new_proc| new_proc.locally_free.clone())
                .collect(),
        );

        locally_free = union(
            locally_free,
            par.exprs
                .iter()
                .flat_map(|expr| expr.locally_free(expr.clone(), 0))
                .collect(),
        );

        locally_free = union(
            locally_free,
            par.matches
                .iter()
                .flat_map(|match_proc| match_proc.locally_free.clone())
                .collect(),
        );

        locally_free = union(
            locally_free,
            par.bundles
                .iter()
                .flat_map(|bundle_proc| bundle_proc.body.clone().unwrap().locally_free.clone())
                .collect(),
        );

        par.locally_free = locally_free;
        par
    }

    fn update_locally_free_elist(&self, mut elist: EList) -> EList {
        elist.locally_free = elist
            .ps
            .iter()
            .map(|p| p.locally_free.clone())
            .fold(Vec::new(), |acc, locally_free| union(acc, locally_free));

        elist
    }

    fn update_locally_free_etuple(&self, mut etuple: ETuple) -> ETuple {
        etuple.locally_free = etuple
            .ps
            .iter()
            .map(|p| p.locally_free.clone())
            .fold(Vec::new(), |acc, locally_free| union(acc, locally_free));

        etuple
    }

    /**
     * Evaluate any top level expressions in @param Par .
     *
     * Public here to be used in tests / Scala code has it as private but still able to use in tests?
     */
    pub fn eval_expr(&self, par: &Par, env: &Env<Par>) -> Result<Par, InterpreterError> {
        let evaled_exprs = par
            .exprs
            .iter()
            .map(|expr| self.eval_expr_to_par(expr, env))
            .collect::<Result<Vec<_>, InterpreterError>>()?;
        // println!("\npar in eval_expr: {:?}", par);
        // println!("\nevaled_exprs in eval_expr: {:?}", evaled_exprs);

        // Note: the locallyFree cache in par could now be invalid, but given
        // that locallyFree is for use in the matcher, and the matcher uses
        // substitution, it will resolve in that case. AlwaysEqual makes sure
        // that this isn't an issue in the rest of cases.
        let result = evaled_exprs
            .into_iter()
            .fold(par.with_exprs(Vec::new()), |acc, expr| {
                // acc.exprs.iter().chain(expr.exprs.iter()).cloned().collect()
                concatenate_pars(acc, expr)
            });

        Ok(result)
    }

    pub fn new(
        space: RhoISpace,
        urn_map: HashMap<String, Par>,
        merge_chs: Arc<RwLock<HashSet<Par>>>,
        mergeable_tag_name: Par,
        cost: _cost,
    ) -> Self {
        let dispatcher = Arc::new(RwLock::new(RholangAndScalaDispatcher {
            _dispatch_table: Arc::new(RwLock::new(HashMap::new())),
            reducer: None,
        }));

        let reducer = DebruijnInterpreter {
            space,
            dispatcher: dispatcher.clone(),
            urn_map,
            merge_chs,
            mergeable_tag_name,
            cost: cost.clone(),
            substitute: Substitute { cost: cost.clone() },
        };

        dispatcher.try_write().unwrap().reducer = Some(reducer.clone());
        reducer
    }
}

fn get_type(expr_instance: ExprInstance) -> String {
    match expr_instance {
        ExprInstance::GBool(_) => String::from("bool"),
        ExprInstance::GInt(_) => String::from("int"),
        ExprInstance::GString(_) => String::from("string"),
        ExprInstance::GUri(_) => String::from("uri"),
        ExprInstance::GByteArray(_) => String::from("byte array"),
        ExprInstance::ENotBody(_) => String::from("enot"),
        ExprInstance::ENegBody(_) => String::from("eneg"),
        ExprInstance::EMultBody(_) => String::from("mult"),
        ExprInstance::EDivBody(_) => String::from("div"),
        ExprInstance::EPlusBody(_) => String::from("plus"),
        ExprInstance::EMinusBody(_) => String::from("minus"),
        ExprInstance::ELtBody(_) => String::from("elt"),
        ExprInstance::ELteBody(_) => String::from("elte"),
        ExprInstance::EGtBody(_) => String::from("egt"),
        ExprInstance::EGteBody(_) => String::from("egte"),
        ExprInstance::EEqBody(_) => String::from("eeq"),
        ExprInstance::ENeqBody(_) => String::from("eneq"),
        ExprInstance::EAndBody(_) => String::from("eand"),
        ExprInstance::EOrBody(_) => String::from("eor"),
        ExprInstance::EVarBody(_) => String::from("evar"),
        ExprInstance::EListBody(_) => String::from("list"),
        ExprInstance::ETupleBody(_) => String::from("tuple"),
        ExprInstance::ESetBody(_) => String::from("set"),
        ExprInstance::EMapBody(_) => String::from("map"),
        ExprInstance::EPathmapBody(_) => String::from("pathmap"),
        ExprInstance::EMethodBody(_) => String::from("emethod"),
        ExprInstance::EMatchesBody(_) => String::from("ematches"),
        ExprInstance::EPercentPercentBody(_) => String::from("epercent percent"),
        ExprInstance::EPlusPlusBody(_) => String::from("plus plus"),
        ExprInstance::EMinusMinusBody(_) => String::from("minus minus"),
        ExprInstance::EModBody(_) => String::from("mod"),
    }
}

fn get_unforgeable_type(inf_instance: &UnfInstance) -> String {
    match inf_instance {
        UnfInstance::GPrivateBody(_) => String::from("PrivateBody"),
        UnfInstance::GDeployIdBody(_) => String::from("DeployId"),
        UnfInstance::GDeployerIdBody(_) => String::from("DeployerId"),
        UnfInstance::GSysAuthTokenBody(_) => String::from("SysAuthToken"),
    }
}
