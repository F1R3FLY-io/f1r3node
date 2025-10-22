use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::{tagged_continuation::TaggedCont, ListParWithRandom, Par, TaggedContinuation};
use prost::Message;
use std::sync::{Arc, OnceLock};

use super::system_processes::{non_deterministic_ops, RhoDispatchMap};
use super::{env::Env, errors::InterpreterError, reduce::DebruijnInterpreter, unwrap_option_safe};

pub fn build_env(data_list: Vec<ListParWithRandom>) -> Env<Par> {
    let pars: Vec<Par> = data_list.into_iter().flat_map(|list| list.pars).collect();
    let mut env = Env::new();

    for par in pars {
        env = env.put(par);
    }

    env
}

#[derive(Clone)]
pub struct RholangAndScalaDispatcher {
    pub _dispatch_table: RhoDispatchMap,
    pub reducer: Arc<OnceLock<DebruijnInterpreter>>,
}

pub type RhoDispatch = Arc<RholangAndScalaDispatcher>;

pub enum DispatchType {
    NonDeterministicCall(Vec<Vec<u8>>),
    DeterministicCall,
    Skip,
}

impl RholangAndScalaDispatcher {
    pub async fn dispatch(
        &self,
        continuation: TaggedContinuation,
        data_list: Vec<ListParWithRandom>,
        is_replay: bool,
        previous_output: Vec<Par>,
    ) -> Result<DispatchType, InterpreterError> {
        // println!("\ndispatcher dispatch");
        // println!("continuation: {:?}", continuation);
        match continuation.tagged_cont {
            Some(cont) => match cont {
                TaggedCont::ParBody(par_with_rand) => {
                    let env = build_env(data_list.clone());
                    let mut randoms =
                        vec![Blake2b512Random::from_bytes(&par_with_rand.random_state)];
                    randoms.extend(
                        data_list
                            .iter()
                            .map(|p| Blake2b512Random::from_bytes(&p.random_state)),
                    );

                    let reducer = self.reducer.get().expect("Reducer not initialized");
                    let body = unwrap_option_safe(par_with_rand.body)?;
                    let merged_rand = Blake2b512Random::merge(randoms);
                    reducer.eval(body, &env, merged_rand).await?;

                    Ok(DispatchType::DeterministicCall)
                }
                TaggedCont::ScalaBodyRef(_ref) => {
                    let is_non_deterministic = non_deterministic_ops().contains(&_ref);
                    // println!("self {:p}", self);
                    let dispatch_table = self._dispatch_table.read().await;
                    // println!(
                    //     "dispatch_table at ScalaBodyRef: {:?}",
                    //     dispatch_table.keys()
                    // );
                    match dispatch_table.get(&_ref) {
                        Some(f) => {
                            let output = f((data_list, is_replay, previous_output)).await?;
                            RholangAndScalaDispatcher::dispatch_type(is_non_deterministic, output)
                        }
                        None => Err(InterpreterError::BugFoundError(format!(
                            "dispatch: no function for {}",
                            _ref,
                        ))),
                    }
                }
            },
            None => Ok(DispatchType::Skip),
        }
    }

    fn dispatch_type(
        is_non_deterministic: bool,
        output: Vec<Par>,
    ) -> Result<DispatchType, InterpreterError> {
        if is_non_deterministic {
            Ok(DispatchType::NonDeterministicCall(
                output.iter().map(|p| p.encode_to_vec()).collect(),
            ))
        } else {
            Ok(DispatchType::DeterministicCall)
        }
    }
}
