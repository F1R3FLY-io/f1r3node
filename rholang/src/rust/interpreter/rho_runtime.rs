// See rholang/src/main/scala/coop/rchain/rholang/interpreter/RhoRuntime.scala
use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::expr::ExprInstance::EMapBody;
use models::rhoapi::tagged_continuation::TaggedCont;
use models::rhoapi::Bundle;
use models::rhoapi::Var;
use models::rhoapi::{BindPattern, Expr, ListParWithRandom, Par, TaggedContinuation};
use models::rust::block_hash::BlockHash;
use models::rust::par_map::ParMap;
use models::rust::par_map_type_mapper::ParMapTypeMapper;
use models::rust::sorted_par_map::SortedParMap;
use models::rust::utils::new_freevar_par;
use models::rust::validator::Validator;
use rspace_plus_plus::rspace::checkpoint::{Checkpoint, SoftCheckpoint};
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use rspace_plus_plus::rspace::history::history_repository::HistoryRepository;
use rspace_plus_plus::rspace::internal::{Datum, Row, WaitingContinuation};
use rspace_plus_plus::rspace::r#match::Match;
use rspace_plus_plus::rspace::replay_rspace_interface::IReplayRSpace;
use rspace_plus_plus::rspace::rspace::RSpace;
use rspace_plus_plus::rspace::rspace::RSpaceStore;
use rspace_plus_plus::rspace::rspace_interface::ISpace;
use rspace_plus_plus::rspace::trace::Log;
use rspace_plus_plus::rspace::tuplespace_interface::Tuplespace;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::rust::interpreter::openai_service::OpenAIService;
use crate::rust::interpreter::system_processes::{BodyRefs, FixedChannels};

use super::accounting::_cost;
use super::accounting::cost_accounting::CostAccounting;
use super::accounting::costs::Cost;
use super::accounting::has_cost::HasCost;
use super::dispatch::RhoDispatch;
use super::dispatch::RholangAndScalaDispatcher;
use super::env::Env;
use super::errors::InterpreterError;
use super::interpreter::{EvaluateResult, Interpreter, InterpreterImpl};
use super::reduce::DebruijnInterpreter;
use super::registry::registry_bootstrap::ast;
use super::storage::charging_rspace::ChargingRSpace;
use super::substitute::Substitute;
use super::system_processes::{
    Arity, BlockData, BodyRef, Definition, DeployData, InvalidBlocks, Name, ProcessContext, Remainder,
    RhoDispatchMap,
};
use models::rhoapi::expr::ExprInstance::GByteArray;

/*
 * This trait has been combined with the 'ReplayRhoRuntime' trait
*/
#[allow(async_fn_in_trait)]
pub trait RhoRuntime: HasCost {
    /**
     * Parse the rholang term into [[coop.rchain.models.Par]] and execute it with provided initial phlo.
     *
     * This function would change the state in the runtime.
     * @param term The rholang contract which would run on the runtime
     * @param initialPhlo initial cost for the this evaluation. If the phlo is not enough,
     *                    [[coop.rchain.rholang.interpreter.errors.OutOfPhlogistonsError]] would return.
     * @param normalizerEnv additional env for Par when parsing term into Par
     * @param rand random seed for rholang execution
     * @return
     */
    async fn evaluate(
        &self,
        term: &str,
        initial_phlo: Cost,
        normalizer_env: HashMap<String, Par>,
        rand: Blake2b512Random,
    ) -> Result<EvaluateResult, InterpreterError>;

    // See rholang/src/main/scala/coop/rchain/rholang/interpreter/RhoRuntimeSyntax.scala
    async fn evaluate_with_env(
        &mut self,
        term: &str,
        normalizer_env: HashMap<String, Par>,
    ) -> Result<EvaluateResult, InterpreterError> {
        self.evaluate_with_env_and_phlo(term, Cost::unsafe_max(), normalizer_env)
            .await
    }

    async fn evaluate_with_term(&mut self, term: &str) -> Result<EvaluateResult, InterpreterError> {
        self.evaluate_with_env_and_phlo(term, Cost::unsafe_max(), HashMap::new())
            .await
    }

    async fn evaluate_with_phlo(
        &mut self,
        term: &str,
        initial_phlo: Cost,
    ) -> Result<EvaluateResult, InterpreterError> {
        self.evaluate_with_env_and_phlo(term, initial_phlo, HashMap::new())
            .await
    }

    async fn evaluate_with_env_and_phlo(
        &mut self,
        term: &str,
        initial_phlo: Cost,
        normalizer_env: HashMap<String, Par>,
    ) -> Result<EvaluateResult, InterpreterError> {
        let rand = Blake2b512Random::create_from_length(128);
        let checkpoint = self.create_soft_checkpoint();
        match self
            .evaluate(term, initial_phlo, normalizer_env, rand)
            .await
        {
            Ok(eval_result) => {
                if !eval_result.errors.is_empty() {
                    self.revert_to_soft_checkpoint(checkpoint);
                    Ok(eval_result)
                } else {
                    Ok(eval_result)
                }
            }
            Err(err) => {
                self.revert_to_soft_checkpoint(checkpoint);
                Err(err)
            }
        }
    }

    /**
     * The function would execute the par regardless setting cost which would possibly cause
     * [[coop.rchain.rholang.interpreter.errors.OutOfPhlogistonsError]]. Because of that, use this
     * function in some situation which is not cost sensitive.
     *
     * This function would change the state in the runtime.
     *
     * Ideally, this function should be removed or hack the runtime without cost accounting in the future .
     * @param par [[coop.rchain.models.Par]] for the execution
     * @param env additional env for execution
     * @param rand random seed for rholang execution
     * @return
     */
    async fn inj(
        &self,
        par: Par,
        env: Env<Par>,
        rand: Blake2b512Random,
    ) -> Result<(), InterpreterError>;

    /**
     * After some executions([[evaluate]]) on the runtime, you can create a soft checkpoint which is the changes
     * for the current state of the runtime. You can revert the changes by [[revertToSoftCheckpoint]]
     * @return
     */
    fn create_soft_checkpoint(
        &mut self,
    ) -> SoftCheckpoint<Par, BindPattern, ListParWithRandom, TaggedContinuation>;

    fn revert_to_soft_checkpoint(
        &mut self,
        soft_checkpoint: SoftCheckpoint<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
    ) -> ();

    /**
     * Create a checkpoint for the runtime. All the changes which happened in the runtime would persistent in the disk
     * and result in a new stateHash for the new state.
     * @return
     */
    fn create_checkpoint(&mut self) -> Checkpoint;

    /**
     * Reset the runtime to the specific state. Then you can operate some execution on the state.
     * @param root the target state hash to reset
     * @return
     */
    fn reset(&mut self, root: &Blake2b256Hash) -> ();

    /**
     * Consume the result in the rspace.
     *
     * This function would change the state in the runtime.
     * @param channel target channel for the consume
     * @param pattern pattern for the consume
     * @return
     */
    fn consume_result(
        &mut self,
        channel: Vec<Par>,
        pattern: Vec<BindPattern>,
    ) -> Result<Option<(TaggedContinuation, Vec<ListParWithRandom>)>, InterpreterError>;

    /**
     * get data directly from history repository
     *
     * This function would not change the state in the runtime
     */
    fn get_data(&self, channel: &Par) -> Vec<Datum<ListParWithRandom>>;

    fn get_joins(&self, channel: Par) -> Vec<Vec<Par>>;

    /**
     * get continuation directly from history repository
     *
     * This function would not change the state in the runtime
     */
    fn get_continuations(
        &self,
        channels: Vec<Par>,
    ) -> Vec<WaitingContinuation<BindPattern, TaggedContinuation>>;

    /**
     * Set the runtime block data environment.
     */
    async fn set_block_data(&self, block_data: BlockData) -> ();

    /**
     * Set the runtime invalid blocks environment.
     */
    async fn set_invalid_blocks(&self, invalid_blocks: HashMap<BlockHash, Validator>) -> ();

    /**
     * Set the runtime deploy data environment.
     */
    async fn set_deploy_data(&self, deploy_data: DeployData) -> ();

    /**
     * Get the hot changes after some executions for the runtime.
     * Currently this is only for debug info mostly.
     */
    fn get_hot_changes(
        &self,
    ) -> HashMap<Vec<Par>, Row<BindPattern, ListParWithRandom, TaggedContinuation>>;

    /* Replay functions */

    fn rig(&self, log: Log) -> Result<(), InterpreterError>;

    fn check_replay_data(&self) -> Result<(), InterpreterError>;
}

/*
 * We use this struct for both normal and replay RhoRuntime instances
*/
#[derive(Clone)]
pub struct RhoRuntimeImpl {
    pub reducer: DebruijnInterpreter,
    pub cost: _cost,
    pub block_data_ref: Arc<tokio::sync::RwLock<BlockData>>,
    pub invalid_blocks_param: InvalidBlocks,
    pub deploy_data_ref: Arc<tokio::sync::RwLock<DeployData>>,
    pub merge_chs: Arc<std::sync::RwLock<HashSet<Par>>>,
}

impl RhoRuntimeImpl {
    fn new(
        reducer: DebruijnInterpreter,
        cost: _cost,
        block_data_ref: Arc<tokio::sync::RwLock<BlockData>>,
        invalid_blocks_param: InvalidBlocks,
        deploy_data_ref: Arc<tokio::sync::RwLock<DeployData>>,
        merge_chs: Arc<std::sync::RwLock<HashSet<Par>>>,
    ) -> RhoRuntimeImpl {
        RhoRuntimeImpl {
            reducer,
            cost,
            block_data_ref,
            invalid_blocks_param,
            deploy_data_ref,
            merge_chs,
        }
    }

    pub fn get_cost_log(&self) -> Vec<Cost> {
        self.cost.get_log()
    }

    pub fn clear_cost_log(&self) {
        self.cost.clear_log()
    }
}

impl RhoRuntime for RhoRuntimeImpl {
    async fn evaluate(
        &self,
        term: &str,
        initial_phlo: Cost,
        normalizer_env: HashMap<String, Par>,
        rand: Blake2b512Random,
    ) -> Result<EvaluateResult, InterpreterError> {
        // println!(
        //     "\nspace hot store size before in evaluate: {:?}",
        //     self.get_hot_changes().len()
        // );
        // rand.debug_str();
        let i = InterpreterImpl::new(self.cost.clone(), self.merge_chs.clone());
        let reducer = &self.reducer;
        let res = i
            .inj_attempt(reducer, term, initial_phlo, normalizer_env, rand)
            .await;
        // println!(
        //     "space hot store size after in evaluate: {:?}",
        //     self.get_hot_changes().len()
        // );
        // println!("\nevaluate result: {:?}", res);
        res
    }

    async fn inj(
        &self,
        par: Par,
        _env: Env<Par>,
        rand: Blake2b512Random,
    ) -> Result<(), InterpreterError> {
        // println!(
        //     "\nspace hot store size before in inj: {:?}",
        //     self.get_hot_changes().len()
        // );
        // println!("\nenv in inj: {:?}", _env);
        // println!("\npar in inj: {:?}", par);
        let res = self.reducer.inj(par, rand).await;
        // println!(
        //     "space hot store size after in inj: {:?}",
        //     self.get_hot_changes().len()
        // );
        res
    }

    fn create_soft_checkpoint(
        &mut self,
    ) -> SoftCheckpoint<Par, BindPattern, ListParWithRandom, TaggedContinuation> {
        self.reducer
            .space
            .try_lock()
            .unwrap()
            .create_soft_checkpoint()
    }

    fn revert_to_soft_checkpoint(
        &mut self,
        soft_checkpoint: SoftCheckpoint<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
    ) -> () {
        self.reducer
            .space
            .try_lock()
            .unwrap()
            .revert_to_soft_checkpoint(soft_checkpoint)
            .unwrap()
    }

    fn create_checkpoint(&mut self) -> Checkpoint {
        // let _ = self.reducer.space.try_lock().unwrap().create_checkpoint().unwrap();
        let checkpoint = self
            .reducer
            .space
            .try_lock()
            .unwrap()
            .create_checkpoint()
            .unwrap();
        // println!(
        //     "\nruntime space after create_checkpoint, {:?}",
        //     self.get_hot_changes().len()
        // );
        // println!(
        //     "\nreducer space after create_checkpoint, {:?}",
        //     self.get_reducer_hot_changes().len()
        // );
        checkpoint
    }

    fn reset(&mut self, root: &Blake2b256Hash) -> () {
        // retaining graceful behavior; detailed error handling now lives in FFI reset returning codes
        let mut space_lock = match self.reducer.space.try_lock() {
            Ok(lock) => lock,
            Err(e) => {
                println!("ERROR: failed to lock reducer.space in reset: {:?}", e);
                return ();
            }
        };

        match space_lock.reset(root) {
            Ok(_) => (),
            Err(e) => {
                println!("ERROR: reset failed with error: {:?}", e);
                println!("Error details: {}", e);
                println!("Failed root: {:?}", root);
                return ();
            }
        }
    }

    fn consume_result(
        &mut self,
        channel: Vec<Par>,
        pattern: Vec<BindPattern>,
    ) -> Result<Option<(TaggedContinuation, Vec<ListParWithRandom>)>, InterpreterError> {
        Ok(self
            .reducer
            .space
            .try_lock()
            .unwrap()
            .consume_result(channel, pattern)?)
    }

    fn get_data(&self, channel: &Par) -> Vec<Datum<ListParWithRandom>> {
        self.reducer.space.try_lock().unwrap().get_data(channel)
    }

    fn get_joins(&self, channel: Par) -> Vec<Vec<Par>> {
        self.reducer.space.try_lock().unwrap().get_joins(channel)
    }

    fn get_continuations(
        &self,
        channels: Vec<Par>,
    ) -> Vec<WaitingContinuation<BindPattern, TaggedContinuation>> {
        self.reducer
            .space
            .try_lock()
            .unwrap()
            .get_waiting_continuations(channels)
    }

    async fn set_block_data(&self, block_data: BlockData) -> () {
        let mut lock = self.block_data_ref.write().await;
        *lock = block_data;
    }

    async fn set_deploy_data(&self, deploy_data: DeployData) -> () {
        let mut lock = self.deploy_data_ref.write().await;
        *lock = deploy_data;
    }

    async fn set_invalid_blocks(&self, invalid_blocks: HashMap<BlockHash, Validator>) -> () {
        let invalid_blocks: Par = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(EMapBody(ParMapTypeMapper::par_map_to_emap(
                ParMap::create_from_sorted_par_map(SortedParMap::create_from_map(
                    invalid_blocks
                        .into_iter()
                        .map(|(validator, block_hash)| {
                            (
                                Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(GByteArray(validator.into())),
                                }]),
                                Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(GByteArray(block_hash.into())),
                                }]),
                            )
                        })
                        .collect(),
                )),
            ))),
        }]);

        self.invalid_blocks_param.set_params(invalid_blocks).await
    }

    fn get_hot_changes(
        &self,
    ) -> HashMap<Vec<Par>, Row<BindPattern, ListParWithRandom, TaggedContinuation>> {
        self.reducer.space.try_lock().unwrap().to_map()
    }

    fn rig(&self, log: Log) -> Result<(), InterpreterError> {
        self.reducer.space.try_lock().unwrap().rig(log)?;
        Ok(())
    }

    fn check_replay_data(&self) -> Result<(), InterpreterError> {
        self.reducer.space.try_lock().unwrap().check_replay_data()?;
        Ok(())
    }
}

impl HasCost for RhoRuntimeImpl {
    fn cost(&self) -> &_cost {
        &self.cost
    }
}

pub type RhoTuplespace = Arc<
    tokio::sync::Mutex<
        Box<dyn Tuplespace<Par, BindPattern, ListParWithRandom, TaggedContinuation> + Send + Sync>,
    >,
>;

pub type RhoISpace = Arc<
    tokio::sync::Mutex<
        Box<dyn ISpace<Par, BindPattern, ListParWithRandom, TaggedContinuation> + Send + Sync>,
    >,
>;

pub type RhoReplayISpace = Arc<
    tokio::sync::Mutex<
        Box<
            dyn IReplayRSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>
                + Send
                + Sync,
        >,
    >,
>;

pub type RhoHistoryRepository = Arc<
    Box<
        dyn HistoryRepository<Par, BindPattern, ListParWithRandom, TaggedContinuation>
            + Send
            + Sync
            + 'static,
    >,
>;

pub type ISpaceAndReplay = (RhoISpace, RhoReplayISpace);

fn introduce_system_process<T>(
    mut spaces: Vec<&mut T>,
    processes: Vec<(Name, Arity, Remainder, BodyRef)>,
) -> Vec<Option<(TaggedContinuation, Vec<ListParWithRandom>)>>
where
    T: ISpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
{
    let mut results: Vec<Option<(TaggedContinuation, Vec<ListParWithRandom>)>> = Vec::new();

    for (name, arity, remainder, body_ref) in processes {
        let channels = vec![name];
        let patterns = vec![BindPattern {
            patterns: (0..arity).map(|i| new_freevar_par(i, Vec::new())).collect(),
            remainder,
            free_count: arity,
        }];

        let continuation = TaggedContinuation {
            tagged_cont: Some(TaggedCont::ScalaBodyRef(body_ref)),
        };

        for space in &mut spaces {
            let result = space.install(channels.clone(), patterns.clone(), continuation.clone());
            results.push(result.map_err(|err| panic!("{}", err)).unwrap());
        }
    }

    results
}

fn std_system_processes() -> Vec<Definition> {
    vec![
        Definition {
            urn: "rho:io:stdout".to_string(),
            fixed_channel: FixedChannels::stdout(),
            arity: 1,
            body_ref: BodyRefs::STDOUT,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { ctx.system_processes.clone().std_out(args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:io:stdoutAck".to_string(),
            fixed_channel: FixedChannels::stdout_ack(),
            arity: 2,
            body_ref: BodyRefs::STDOUT_ACK,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { ctx.system_processes.clone().std_out_ack(args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:io:stderr".to_string(),
            fixed_channel: FixedChannels::stderr(),
            arity: 1,
            body_ref: BodyRefs::STDERR,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { ctx.system_processes.clone().std_err(args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:io:stderrAck".to_string(),
            fixed_channel: FixedChannels::stderr_ack(),
            arity: 2,
            body_ref: BodyRefs::STDERR_ACK,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { ctx.system_processes.clone().std_err_ack(args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:block:data".to_string(),
            fixed_channel: FixedChannels::get_block_data(),
            arity: 1,
            body_ref: BodyRefs::GET_BLOCK_DATA,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move {
                        ctx.system_processes
                            .clone()
                            .get_block_data(args, ctx.block_data.clone())
                            .await
                    })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:casper:invalidBlocks".to_string(),
            fixed_channel: FixedChannels::get_invalid_blocks(),
            arity: 1,
            body_ref: BodyRefs::GET_INVALID_BLOCKS,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move {
                        ctx.system_processes
                            .clone()
                            .invalid_blocks(args, &ctx.invalid_blocks)
                            .await
                    })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:rev:address".to_string(),
            fixed_channel: FixedChannels::rev_address(),
            arity: 3,
            body_ref: BodyRefs::REV_ADDRESS,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { ctx.system_processes.clone().rev_address(args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:rchain:deployerId:ops".to_string(),
            fixed_channel: FixedChannels::deployer_id_ops(),
            arity: 3,
            body_ref: BodyRefs::DEPLOYER_ID_OPS,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(
                        async move { ctx.system_processes.clone().deployer_id_ops(args).await },
                    )
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:registry:ops".to_string(),
            fixed_channel: FixedChannels::reg_ops(),
            arity: 3,
            body_ref: BodyRefs::REG_OPS,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { ctx.system_processes.clone().registry_ops(args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "sys:authToken:ops".to_string(),
            fixed_channel: FixedChannels::sys_authtoken_ops(),
            arity: 3,
            body_ref: BodyRefs::SYS_AUTHTOKEN_OPS,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(
                        async move { ctx.system_processes.clone().sys_auth_token_ops(args).await },
                    )
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:io:grpcTell".to_string(),
            fixed_channel: FixedChannels::grpc_tell(),
            arity: 3,
            body_ref: BodyRefs::GRPC_TELL,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { ctx.system_processes.clone().grpc_tell(args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:io:devNull".to_string(),
            fixed_channel: FixedChannels::dev_null(),
            arity: 1,
            body_ref: BodyRefs::DEV_NULL,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { ctx.system_processes.clone().dev_null(args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:deploy:data".to_string(),
            fixed_channel: FixedChannels::deploy_data(),
            arity: 1,
            body_ref: BodyRefs::DEPLOY_DATA,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move {
                        ctx.system_processes
                            .clone()
                            .get_deploy_data(args, ctx.deploy_data.clone())
                            .await
                    })
                })
            }),
            remainder: None,
        },
    ]
}

fn std_rho_crypto_processes() -> Vec<Definition> {
    vec![
        Definition {
            urn: "rho:crypto:secp256k1Verify".to_string(),
            fixed_channel: FixedChannels::secp256k1_verify(),
            arity: 4,
            body_ref: BodyRefs::SECP256K1_VERIFY,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(
                        async move { ctx.system_processes.clone().secp256k1_verify(args).await },
                    )
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:crypto:blake2b256Hash".to_string(),
            fixed_channel: FixedChannels::blake2b256_hash(),
            arity: 2,
            body_ref: BodyRefs::BLAKE2B256_HASH,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(
                        async move { ctx.system_processes.clone().blake2b256_hash(args).await },
                    )
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:crypto:keccak256Hash".to_string(),
            fixed_channel: FixedChannels::keccak256_hash(),
            arity: 2,
            body_ref: BodyRefs::KECCAK256_HASH,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { ctx.system_processes.clone().keccak256_hash(args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:crypto:sha256Hash".to_string(),
            fixed_channel: FixedChannels::sha256_hash(),
            arity: 2,
            body_ref: BodyRefs::SHA256_HASH,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { ctx.system_processes.clone().sha256_hash(args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:crypto:ed25519Verify".to_string(),
            fixed_channel: FixedChannels::ed25519_verify(),
            arity: 4,
            body_ref: BodyRefs::ED25519_VERIFY,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { ctx.system_processes.clone().ed25519_verify(args).await })
                })
            }),
            remainder: None,
        },
    ]
}

fn std_rho_ai_processes() -> Vec<Definition> {
    vec![
        Definition {
            urn: "rho:ai:gpt4".to_string(),
            fixed_channel: FixedChannels::gpt4(),
            arity: 2,
            body_ref: BodyRefs::GPT4,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { ctx.system_processes.clone().gpt4(args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:ai:dalle3".to_string(),
            fixed_channel: FixedChannels::dalle3(),
            arity: 2,
            body_ref: BodyRefs::DALLE3,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { ctx.system_processes.clone().dalle3(args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:ai:textToAudio".to_string(),
            fixed_channel: FixedChannels::text_to_audio(),
            arity: 2,
            body_ref: BodyRefs::TEXT_TO_AUDIO,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { ctx.system_processes.clone().text_to_audio(args).await })
                })
            }),
            remainder: None,
        },
    ]
}

fn dispatch_table_creator(
    space: RhoISpace,
    dispatcher: RhoDispatch,
    block_data: Arc<tokio::sync::RwLock<BlockData>>,
    invalid_blocks: InvalidBlocks,
    deploy_data: Arc<tokio::sync::RwLock<DeployData>>,
    extra_system_processes: &mut Vec<Definition>,
    openai_service: Arc<tokio::sync::Mutex<OpenAIService>>,
) -> RhoDispatchMap {
    let mut dispatch_table = HashMap::new();

    for def in std_system_processes().iter_mut().chain(
        std_rho_crypto_processes()
            .iter_mut()
            .chain(std_rho_ai_processes().iter_mut())
            .chain(extra_system_processes.iter_mut()),
    ) {
        // TODO: Remove cloning every time
        let tuple = def.to_dispatch_table(ProcessContext::create(
            space.clone(),
            dispatcher.clone(),
            block_data.clone(),
            invalid_blocks.clone(),
            deploy_data.clone(),
            openai_service.clone(),
        ));

        dispatch_table.insert(tuple.0, tuple.1);
    }

    Arc::new(tokio::sync::RwLock::new(dispatch_table))
}

fn basic_processes() -> HashMap<String, Par> {
    let mut map = HashMap::new();

    map.insert(
        "rho:registry:lookup".to_string(),
        Par::default().with_bundles(vec![Bundle {
            body: Some(FixedChannels::reg_lookup()),
            write_flag: true,
            read_flag: false,
        }]),
    );

    map.insert(
        "rho:registry:insertArbitrary".to_string(),
        Par::default().with_bundles(vec![Bundle {
            body: Some(FixedChannels::reg_insert_random()),
            write_flag: true,
            read_flag: false,
        }]),
    );

    map.insert(
        "rho:registry:insertSigned:secp256k1".to_string(),
        Par::default().with_bundles(vec![Bundle {
            body: Some(FixedChannels::reg_insert_signed()),
            write_flag: true,
            read_flag: false,
        }]),
    );

    map
}

async fn setup_reducer(
    charging_rspace: RhoISpace,
    block_data_ref: Arc<tokio::sync::RwLock<BlockData>>,
    invalid_blocks: InvalidBlocks,
    deploy_data_ref: Arc<tokio::sync::RwLock<DeployData>>,
    extra_system_processes: &mut Vec<Definition>,
    urn_map: HashMap<String, Par>,
    merge_chs: Arc<std::sync::RwLock<HashSet<Par>>>,
    mergeable_tag_name: Par,
    openai_service: Arc<tokio::sync::Mutex<OpenAIService>>,
    cost: _cost,
) -> DebruijnInterpreter {
    // println!("\nsetup_reducer");

    let reducer_cell = Arc::new(std::sync::OnceLock::new());

    let temp_dispatcher = Arc::new(RholangAndScalaDispatcher {
        _dispatch_table: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        reducer: reducer_cell.clone(),
    });

    let replay_dispatch_table = dispatch_table_creator(
        charging_rspace.clone(),
        temp_dispatcher.clone(),
        block_data_ref,
        invalid_blocks,
        deploy_data_ref,
        extra_system_processes,
        openai_service,
    );

    let dispatcher = Arc::new(RholangAndScalaDispatcher {
        _dispatch_table: replay_dispatch_table,
        reducer: reducer_cell.clone(),
    });

    let reducer = DebruijnInterpreter {
        space: charging_rspace.clone(),
        dispatcher: dispatcher.clone(),
        urn_map: Arc::new(urn_map),
        merge_chs,
        mergeable_tag_name,
        cost: cost.clone(),
        substitute: Substitute { cost: cost.clone() },
    };

    reducer_cell.set(reducer.clone()).ok().unwrap();
    reducer
}

fn setup_maps_and_refs(
    extra_system_processes: &Vec<Definition>,
) -> (
    Arc<tokio::sync::RwLock<BlockData>>,
    InvalidBlocks,
    Arc<tokio::sync::RwLock<DeployData>>,
    HashMap<String, Name>,
    Vec<(Name, Arity, Remainder, BodyRef)>,
) {
    let block_data_ref = Arc::new(tokio::sync::RwLock::new(BlockData::empty()));
    let invalid_blocks = InvalidBlocks::new();
    let deploy_data_ref = Arc::new(tokio::sync::RwLock::new(DeployData::empty()));

    let system_binding = std_system_processes();
    let rho_crypto_binding = std_rho_crypto_processes();
    let rho_ai_binding = std_rho_ai_processes();
    let combined_processes = system_binding
        .iter()
        .chain(rho_crypto_binding.iter())
        .chain(rho_ai_binding.iter())
        .chain(extra_system_processes.iter())
        .collect::<Vec<&Definition>>();

    let mut urn_map: HashMap<_, _> = basic_processes();
    combined_processes
        .iter()
        .map(|process| process.to_urn_map())
        .for_each(|(key, value)| {
            urn_map.insert(key, value);
        });

    // println!("\nurn_map length: {:?}", urn_map.len());

    let proc_defs: Vec<(Par, i32, Option<Var>, i64)> = combined_processes
        .iter()
        .map(|process| process.to_proc_defs())
        .collect();

    // println!("\nproc_defs length: {:?}", proc_defs.len());

    (block_data_ref, invalid_blocks, deploy_data_ref, urn_map, proc_defs)
}

async fn create_rho_env<T>(
    mut rspace: T,
    merge_chs: Arc<std::sync::RwLock<HashSet<Par>>>,
    mergeable_tag_name: Par,
    extra_system_processes: &mut Vec<Definition>,
    cost: _cost,
) -> (
    DebruijnInterpreter,
    Arc<tokio::sync::RwLock<BlockData>>,
    InvalidBlocks,
    Arc<tokio::sync::RwLock<DeployData>>,
)
where
    T: ISpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>
        + Clone
        + Send
        + Sync
        + 'static,
{
    let maps_and_refs = setup_maps_and_refs(&extra_system_processes);
    let (block_data_ref, invalid_blocks, deploy_data_ref, urn_map, proc_defs) = maps_and_refs;
    let res = introduce_system_process(vec![&mut rspace], proc_defs);
    assert!(res.iter().all(|s| s.is_none()));

    let charging_rspace: RhoISpace = Arc::new(tokio::sync::Mutex::new(Box::new(
        ChargingRSpace::charging_rspace(rspace, cost.clone()),
    )));

    let openai_service = Arc::new(tokio::sync::Mutex::new(OpenAIService::new()));
    let reducer = setup_reducer(
        charging_rspace,
        block_data_ref.clone(),
        invalid_blocks.clone(),
        deploy_data_ref.clone(),
        extra_system_processes,
        urn_map,
        merge_chs,
        mergeable_tag_name,
        openai_service,
        cost,
    )
    .await;

    (reducer, block_data_ref, invalid_blocks, deploy_data_ref)
}

// This is from Nassim Taleb's "Skin in the Game"
fn bootstrap_rand() -> Blake2b512Random {
    // println!("\nhit bootstrap_rand");
    Blake2b512Random::create_from_bytes("Decentralization is based on the simple notion that it is easier to macrobull***t than microbull***t. \
         Decentralization reduces large structural asymmetries."
         .as_bytes())
}

pub async fn bootstrap_registry(runtime: &RhoRuntimeImpl) -> () {
    // println!("\ncalling bootstrap_registry");
    let rand = bootstrap_rand();
    // rand.debug_str();
    let cost = runtime.cost().get();
    let _ = runtime
        .cost()
        .set(Cost::create(i64::MAX, "bootstrap registry".to_string()));
    // println!("\nast: {:?}", ast());
    // println!(
    //     "\nruntime space before inject, {:?}",
    //     runtime_lock.get_hot_changes().len()
    // );
    runtime.inj(ast(), Env::new(), rand).await.unwrap();
    // println!(
    //     "\nruntime space after inject, {:?}",
    //     runtime_lock.get_hot_changes().len()
    // );
    let _ = runtime.cost().set(Cost::create_from_cost(cost));
}

async fn create_runtime<T>(
    rspace: T,
    extra_system_processes: &mut Vec<Definition>,
    init_registry: bool,
    mergeable_tag_name: Par,
) -> RhoRuntimeImpl
where
    T: ISpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>
        + Clone
        + Send
        + Sync
        + 'static,
{
    // println!("\nrust create_runtime");
    let cost = CostAccounting::empty_cost();
    let merge_chs = Arc::new(std::sync::RwLock::new({
        let mut set = HashSet::new();
        set.insert(Par::default());
        set
    }));

    let rho_env = create_rho_env(
        rspace,
        merge_chs.clone(),
        mergeable_tag_name,
        extra_system_processes,
        cost.clone(),
    )
    .await;

    let (reducer, block_ref, invalid_blocks, deploy_ref) = rho_env;
    let mut runtime = RhoRuntimeImpl::new(reducer, cost, block_ref, invalid_blocks, deploy_ref, merge_chs);

    if init_registry {
        // println!("\ninit_registry");
        bootstrap_registry(&runtime).await;
        runtime.create_checkpoint();
    }

    runtime
}

/// Creates a runtime for executing Rholang code.
///
/// # Parameters
///
/// - `rspace`: The rspace which the runtime would operate on
/// - `extra_system_processes`: Extra system rholang processes exposed to the runtime
///   which you can execute functions on
/// - `init_registry`: For a newly created rspace, you might need to bootstrap registry
///   in the runtime to use rholang registry normally. This is not the only thing you need
///   for rholang registry - after the bootstrap registry, you still need to insert registry
///   contract on the rspace. For an existing rspace which bootstrapped registry before, you
///   can skip this. For some test cases, you don't need the registry, then you can skip this
///   init process which can be faster.
/// - `mergeable_tag_name`: Tag name for mergeable channels
///
/// # Returns
///
/// A configured `RhoRuntimeImpl` instance ready for executing Rholang code.
#[tracing::instrument(name = "create-play-runtime", target = "f1r3fly.rholang.runtime", skip_all)]
pub async fn create_rho_runtime<T>(
    rspace: T,
    mergeable_tag_name: Par,
    init_registry: bool,
    extra_system_processes: &mut Vec<Definition>,
) -> RhoRuntimeImpl
where
    T: ISpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>
        + Clone
        + Send
        + Sync
        + 'static,
{
    create_runtime(
        rspace,
        extra_system_processes,
        init_registry,
        mergeable_tag_name,
    )
    .await
}

/// Creates a replay runtime for executing Rholang code with replay capabilities.
///
/// # Parameters
///
/// - `rspace`: The replay rspace which the runtime operates on
/// - `extra_system_processes`: Same as `create_rho_runtime`
/// - `init_registry`: Same as `create_rho_runtime`
/// - `mergeable_tag_name`: Tag name for mergeable channels
///
/// # Returns
///
/// A configured `RhoRuntimeImpl` instance with replay capabilities.
#[tracing::instrument(name = "create-replay-runtime", target = "f1r3fly.rholang.runtime", skip_all)]
pub async fn create_replay_rho_runtime<T>(
    rspace: T,
    mergeable_tag_name: Par,
    init_registry: bool,
    extra_system_processes: &mut Vec<Definition>,
) -> RhoRuntimeImpl
where
    T: ISpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>
        + Clone
        + Send
        + Sync
        + 'static,
{
    create_runtime(
        rspace,
        extra_system_processes,
        init_registry,
        mergeable_tag_name,
    )
    .await
}

pub(crate) async fn _create_runtimes<T, R>(
    space: T,
    replay_space: R,
    init_registry: bool,
    additional_system_processes: &mut Vec<Definition>,
    mergeable_tag_name: Par,
) -> (RhoRuntimeImpl, RhoRuntimeImpl)
where
    T: ISpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>
        + Clone
        + Send
        + Sync
        + 'static,
    R: IReplayRSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>
        + Clone
        + Send
        + Sync
        + 'static,
{
    let rho_runtime = create_rho_runtime(
        space,
        mergeable_tag_name.clone(),
        init_registry,
        additional_system_processes,
    )
    .await;

    let replay_rho_runtime = create_replay_rho_runtime(
        replay_space,
        mergeable_tag_name,
        init_registry,
        additional_system_processes,
    )
    .await;

    (rho_runtime, replay_rho_runtime)
}

#[tracing::instrument(name = "create-play-runtime", target = "f1r3fly.rholang.runtime.create-play", skip_all)]
pub async fn create_runtime_from_kv_store(
    stores: RSpaceStore,
    mergeable_tag_name: Par,
    init_registry: bool,
    additional_system_processes: &mut Vec<Definition>,
    matcher: Arc<Box<dyn Match<BindPattern, ListParWithRandom>>>,
) -> RhoRuntimeImpl {
    
    let space: RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation> =
        RSpace::create(stores, matcher).unwrap();

    let runtime = create_rho_runtime(
        space,
        mergeable_tag_name,
        init_registry,
        additional_system_processes,
    )
    .await;

    runtime
}
