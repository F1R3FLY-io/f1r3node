use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::{tagged_continuation::TaggedCont, ListParWithRandom, Par, TaggedContinuation};
use prost::Message;
use std::sync::{Arc, OnceLock, Weak};

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
    pub reducer: Arc<OnceLock<Weak<DebruijnInterpreter>>>,
}

pub type RhoDispatch = Arc<RholangAndScalaDispatcher>;

pub enum DispatchType {
    NonDeterministicCall(Vec<Vec<u8>>),
    /// Indicates a non-deterministic process failed during execution.
    /// Contains the error wrapped for proper replay handling.
    FailedNonDeterministicCall(InterpreterError),
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

                    let reducer = self
                        .reducer
                        .get()
                        .and_then(|weak| weak.upgrade())
                        .ok_or_else(|| {
                            InterpreterError::BugFoundError("Reducer not initialized".to_string())
                        })?;
                    let body = unwrap_option_safe(par_with_rand.body)?;
                    let merged_rand = Blake2b512Random::merge(randoms);
                    let cont_rs_bytes: Vec<u8> = par_with_rand.random_state.iter().map(|&b| b as u8).collect();
                    let cont_rand_hash = {
                        use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
                        hex::encode(Blake2b256Hash::new(&cont_rs_bytes).bytes())
                    };
                    let data_rand_hashes: Vec<String> = data_list.iter()
                        .map(|d| {
                            use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
                            hex::encode(Blake2b256Hash::new(&d.random_state).bytes())
                        })
                        .collect();
                    let merged_rand_bytes = merged_rand.to_bytes();
                    let merged_rand_hash = {
                        use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
                        hex::encode(Blake2b256Hash::new(&merged_rand_bytes).bytes())
                    };
                    tracing::info!(
                        target: "f1r3fly.rspace.cost_trace",
                        cont_rand_hash = %&cont_rand_hash[..16],
                        merged_rand_hash = %&merged_rand_hash[..16],
                        merged_rand_pos = merged_rand.position,
                        merged_rand_path_pos = merged_rand.path_position,
                        data_count = data_list.len(),
                        data_rand_hashes = %data_rand_hashes.join(","),
                        "DISPATCH_RAND: cont_hash={} merged_hash={} merged_pos={} merged_path_pos={} data_count={} data_hashes=[{}]",
                        &cont_rand_hash[..16], &merged_rand_hash[..16],
                        merged_rand.position, merged_rand.path_position,
                        data_list.len(), data_rand_hashes.iter().map(|h| &h[..16]).collect::<Vec<_>>().join(",")
                    );
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
                            match f((data_list, is_replay, previous_output)).await {
                                Ok(output) => RholangAndScalaDispatcher::dispatch_type(
                                    is_non_deterministic,
                                    output,
                                ),
                                Err(e) if is_non_deterministic => {
                                    // Non-deterministic process failed - return FailedNonDeterministicCall
                                    // so the produce event can be marked as failed for replay safety
                                    Ok(DispatchType::FailedNonDeterministicCall(e))
                                }
                                Err(e) => Err(e),
                            }
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
