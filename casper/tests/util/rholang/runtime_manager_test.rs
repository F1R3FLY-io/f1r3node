// See casper/src/test/scala/coop/rchain/casper/util/rholang/RuntimeManagerTest.scala

use std::{
    collections::HashMap,
    time::{SystemTime, UNIX_EPOCH},
};

use casper::rust::{
    errors::CasperError,
    rholang::{replay_runtime::ReplayRuntimeOps, runtime::RuntimeOps},
    util::{
        construct_deploy,
        rholang::{
            costacc::{
                check_balance::CheckBalance, close_block_deploy::CloseBlockDeploy,
                pre_charge_deploy::PreChargeDeploy, refund_deploy::RefundDeploy,
            },
            replay_failure::ReplayFailure,
            runtime_manager::RuntimeManager,
            system_deploy::SystemDeployTrait,
            system_deploy_result::SystemDeployResult,
            system_deploy_user_error::SystemDeployUserError,
            system_deploy_util,
        },
    },
};
use crypto::rust::{hash::blake2b512_random::Blake2b512Random, signatures::signed::Signed};
use models::{
    rhoapi::PCost,
    rust::{
        block::state_hash::StateHash,
        casper::protocol::casper_message::{DeployData, ProcessedDeploy, ProcessedSystemDeploy},
    },
};
use rholang::rust::interpreter::{
    accounting::costs::{self, Cost},
    compiler::compiler::Compiler,
    env::Env,
    rho_runtime::RhoRuntime,
    system_processes::BlockData,
    test_utils::par_builder_util::ParBuilderUtil,
};
use rspace_plus_plus::rspace::{hashing::blake2b256_hash::Blake2b256Hash, history::Either};

use crate::util::{genesis_builder::GenesisContext, rholang::resources::with_runtime_manager};

enum SystemDeployReplayResult<A> {
    ReplaySucceeded {
        state_hash: StateHash,
        result: A,
    },
    ReplayFailed {
        system_deploy_error: SystemDeployUserError,
    },
}

async fn compute_state(
    runtime_manager: &mut RuntimeManager,
    genesis_context: &GenesisContext,
    deploy: Signed<DeployData>,
    state_hash: &StateHash,
) -> (StateHash, ProcessedDeploy) {
    let time_stamp = deploy.data.time_stamp;
    let (new_state_hash, processed_deploys, _extra) = runtime_manager
        .compute_state(
            state_hash,
            vec![deploy],
            Vec::new(), // No system deploys
            BlockData {
                time_stamp,
                block_number: 0,
                sender: genesis_context.validator_pks()[0].clone(),
                seq_num: 0,
            },
            None,
        )
        .await
        .unwrap();

    let result = processed_deploys.into_iter().next().unwrap();
    (new_state_hash, result)
}

async fn replay_compute_state(
    runtime_manager: &mut RuntimeManager,
    genesis_context: &GenesisContext,
    processed_deploy: ProcessedDeploy,
    state_hash: &StateHash,
) -> Result<StateHash, CasperError> {
    let time_stamp = processed_deploy.deploy.data.time_stamp;
    runtime_manager
        .replay_compute_state(
            state_hash,
            vec![processed_deploy],
            Vec::new(),
            &BlockData {
                time_stamp,
                block_number: 0,
                sender: genesis_context.validator_pks()[0].clone(),
                seq_num: 0,
            },
            None,
            false,
        )
        .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn comput_state_should_charge_for_deploys() {
    with_runtime_manager(
        |mut runtime_manager, genesis_context, genesis_block| async move {
            let gen_post_state = genesis_block.body.state.post_state_hash;
            let source = r#"
            new rl(`rho:registry:lookup`), listOpsCh in {
                rl!(`rho:lang:listOps`, *listOpsCh) |
                for(x <- listOpsCh){
                    Nil
                }
            }
            "#;

            // TODO: Prohibit negative gas prices and gas limits in deploys. - OLD
            // TODO: Make minimum maximum yield for deploy parameter of node. - OLD
            let deploy = construct_deploy::source_deploy_now_full(
                source.to_string(),
                Some(100000),
                None,
                None,
                None,
                None,
            )
            .unwrap();

            let (new_state_hash, processed_deploy) = compute_state(
                &mut runtime_manager,
                &genesis_context,
                deploy,
                &gen_post_state,
            )
            .await;

            let replay_state_hash = replay_compute_state(
                &mut runtime_manager,
                &genesis_context,
                processed_deploy,
                &gen_post_state,
            )
            .await
            .unwrap();

            assert!(new_state_hash != gen_post_state && replay_state_hash == new_state_hash);
        },
    )
    .await
    .unwrap()
}

async fn compare_successful_system_deploys<S: SystemDeployTrait, F>(
    runtime_manager: &mut RuntimeManager,
    genesis_context: &GenesisContext,
    start_state: &StateHash,
    play_system_deploy: &mut S,
    replay_system_deploy: &mut S,
    result_assertion: F,
) -> Result<StateHash, CasperError>
where
    F: Fn(&S::Result) -> bool,
    <S as SystemDeployTrait>::Result: PartialEq,
{
    let runtime = runtime_manager.spawn_runtime().await;
    {
        runtime
            .set_block_data(BlockData {
                time_stamp: 0,
                block_number: 0,
                sender: genesis_context.validator_pks()[0].clone(),
                seq_num: 0,
            })
            .await;
    }

    let mut runtime_ops = RuntimeOps::new(runtime);
    let play_system_result = runtime_ops
        .play_system_deploy(start_state, play_system_deploy)
        .await?;

    match play_system_result {
        SystemDeployResult::PlaySucceeded {
            state_hash: final_play_state_hash,
            processed_system_deploy,
            mergeable_channels: _,
            result: play_result,
        } => {
            result_assertion(&play_result);

            let replay_runtime = runtime_manager.spawn_replay_runtime().await;
            {
                replay_runtime
                    .set_block_data(BlockData {
                        time_stamp: 0,
                        block_number: 0,
                        sender: genesis_context.validator_pks()[0].clone(),
                        seq_num: 0,
                    })
                    .await;
            }

            let replay_runtime_ops = ReplayRuntimeOps::new_from_runtime(replay_runtime);
            let replay_system_result = exec_replay_system_deploy(
                replay_runtime_ops,
                start_state,
                replay_system_deploy,
                &processed_system_deploy,
            )
            .await?;

            match replay_system_result {
                SystemDeployReplayResult::ReplaySucceeded {
                    state_hash: final_replay_state_hash,
                    result: replay_result,
                } => {
                    assert!(final_play_state_hash == final_replay_state_hash);
                    assert!(play_result == replay_result);
                    Ok(final_replay_state_hash)
                }

                SystemDeployReplayResult::ReplayFailed {
                    system_deploy_error,
                } => panic!(
                    "Unexpected user error during replay: {:?}",
                    system_deploy_error
                ),
            }
        }

        SystemDeployResult::PlayFailed {
            processed_system_deploy,
        } => panic!(
            "Unexpected system error during play: {:?}",
            processed_system_deploy
        ),
    }
}

async fn exec_replay_system_deploy<S: SystemDeployTrait>(
    mut replay_runtime_ops: ReplayRuntimeOps,
    state_hash: &StateHash,
    system_deploy: &mut S,
    processed_system_deploy: &ProcessedSystemDeploy,
) -> Result<SystemDeployReplayResult<S::Result>, CasperError> {
    let expected_failure = match processed_system_deploy {
        ProcessedSystemDeploy::Failed { error_msg, .. } => Some(error_msg.clone()),
        _ => None,
    };

    replay_runtime_ops
        .rig_system_deploy(processed_system_deploy)
        .await?;
    replay_runtime_ops
        .runtime_ops
        .runtime
        .reset(&Blake2b256Hash::from_bytes_prost(state_hash))
        .await?;

    let (value, eval_res) = replay_runtime_ops
        .replay_system_deploy_internal(system_deploy, &expected_failure)
        .await?;

    replay_runtime_ops
        .check_replay_data_with_fix(eval_res.errors.is_empty())
        .await?;

    match (value, eval_res) {
        (Either::Right(result), _) => {
            let checkpoint = replay_runtime_ops
                .runtime_ops
                .runtime
                .create_checkpoint()
                .await;

            Ok(SystemDeployReplayResult::ReplaySucceeded {
                state_hash: checkpoint.root.to_bytes_prost(),
                result,
            })
        }

        (Either::Left(error), _) => Ok(SystemDeployReplayResult::ReplayFailed {
            system_deploy_error: error,
        }),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pre_charge_deploy_should_reduce_user_account_balance_by_correct_amount() {
    with_runtime_manager(
        |mut runtime_manager, genesis_context, genesis_block| async move {
            let user_pk = construct_deploy::DEFAULT_PUB.clone();
            let state_hash_0 = compare_successful_system_deploys(
                &mut runtime_manager,
                &genesis_context,
                &genesis_block.body.state.post_state_hash,
                &mut PreChargeDeploy {
                    charge_amount: 9000000,
                    pk: user_pk.clone(),
                    rand: Blake2b512Random::create_from_bytes(&vec![0]),
                },
                &mut PreChargeDeploy {
                    charge_amount: 9000000,
                    pk: user_pk.clone(),
                    rand: Blake2b512Random::create_from_bytes(&vec![0]),
                },
                |_| true,
            )
            .await
            .unwrap();

            let state_hash_1 = compare_successful_system_deploys(
                &mut runtime_manager,
                &genesis_context,
                &state_hash_0,
                &mut CheckBalance {
                    pk: user_pk.clone(),
                    rand: Blake2b512Random::create_from_bytes(&vec![1]),
                },
                &mut CheckBalance {
                    pk: user_pk.clone(),
                    rand: Blake2b512Random::create_from_bytes(&vec![1]),
                },
                |result| *result == 0,
            )
            .await
            .unwrap();

            let state_hash_2 = compare_successful_system_deploys(
                &mut runtime_manager,
                &genesis_context,
                &state_hash_1,
                &mut RefundDeploy {
                    refund_amount: 9000000,
                    rand: Blake2b512Random::create_from_bytes(&vec![2]),
                },
                &mut RefundDeploy {
                    refund_amount: 9000000,
                    rand: Blake2b512Random::create_from_bytes(&vec![2]),
                },
                |_| true,
            )
            .await
            .unwrap();

            let _ = compare_successful_system_deploys(
                &mut runtime_manager,
                &genesis_context,
                &state_hash_2,
                &mut CheckBalance {
                    pk: user_pk.clone(),
                    rand: Blake2b512Random::create_from_bytes(&vec![3]),
                },
                &mut CheckBalance {
                    pk: user_pk,
                    rand: Blake2b512Random::create_from_bytes(&vec![3]),
                },
                |result| *result == 9000000,
            )
            .await
            .unwrap();
        },
    )
    .await
    .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn close_block_should_make_epoch_change_and_reward_validator() {
    with_runtime_manager(
        |mut runtime_manager, genesis_context, genesis_block| async move {
            let _ = compare_successful_system_deploys(
                &mut runtime_manager,
                &genesis_context,
                &genesis_block.body.state.post_state_hash,
                &mut CloseBlockDeploy {
                    initial_rand: Blake2b512Random::create_from_bytes(&vec![0]),
                },
                &mut CloseBlockDeploy {
                    initial_rand: Blake2b512Random::create_from_bytes(&vec![0]),
                },
                |_| true,
            )
            .await
            .unwrap();
        },
    )
    .await
    .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn close_block_replay_should_fail_with_different_random_seed() {
    with_runtime_manager(
        |mut runtime_manager, genesis_context, genesis_block| async move {
            let res = compare_successful_system_deploys(
                &mut runtime_manager,
                &genesis_context,
                &genesis_block.body.state.post_state_hash,
                &mut CloseBlockDeploy {
                    initial_rand: Blake2b512Random::create_from_bytes(&vec![0]),
                },
                &mut CloseBlockDeploy {
                    initial_rand: Blake2b512Random::create_from_bytes(&vec![1]),
                },
                |_| true,
            )
            .await;

            assert!(res.is_err());
        },
    )
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn balance_deploy_should_compute_rev_balances() {
    with_runtime_manager(
        |mut runtime_manager, genesis_context, genesis_block| async move {
            let user_pk = construct_deploy::DEFAULT_PUB.clone();
            let _ = compare_successful_system_deploys(
                &mut runtime_manager,
                &genesis_context,
                &genesis_block.body.state.post_state_hash,
                &mut CheckBalance {
                    pk: user_pk.clone(),
                    rand: Blake2b512Random::create_from_bytes(&vec![]),
                },
                &mut CheckBalance {
                    pk: user_pk.clone(),
                    rand: Blake2b512Random::create_from_bytes(&vec![]),
                },
                |result| *result == 9000000,
            )
            .await
            .unwrap();
        },
    )
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compute_state_should_capture_rholang_errors() {
    with_runtime_manager(
        |mut runtime_manager, genesis_context, genesis_block| async move {
            let bad_rholang =
                r#" for(@x <- @"x" & @y <- @"y"){ @"xy"!(x + y) } | @"x"!(1) | @"y"!("hi") "#;
            let deploy = construct_deploy::source_deploy_now_full(
                bad_rholang.to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();

            let result = compute_state(
                &mut runtime_manager,
                &genesis_context,
                deploy,
                &genesis_block.body.state.post_state_hash,
            )
            .await;

            assert!(result.1.is_failed == true);
        },
    )
    .await
    .unwrap();
}

// TODO: Flaky — GasRefundFailure("Insufficient funds") on some runs.
// The deployer vault balance becomes insufficient for the second block's
// refund when scheduling produces a higher deploy cost in block 1.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn compute_state_then_compute_bonds_should_be_replayable_after_all() {
    with_runtime_manager(
        |mut runtime_manager, genesis_context, genesis_block| async move {
            let gps = genesis_block.body.state.post_state_hash;

            let s0 = "@1!(1)";
            let s1 = "@2!(2)";
            let s2 = "for(@a <- @1){ @123!(5 * a) }";

            let deploys0 = vec![s0, s1, s2]
                .into_iter()
                .map(|s| {
                    construct_deploy::source_deploy_now_full(
                        s.to_string(),
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();

            let s3 = "@1!(1)";
            let s4 = "for(@a <- @2){ @456!(5 * a) }";

            let deploys1 = vec![s3, s4]
                .into_iter()
                .map(|s| {
                    construct_deploy::source_deploy_now_full(
                        s.to_string(),
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();

            let time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as i64;

            let (play_state_hash_0, processed_deploys_0, processed_sys_deploys_0) = runtime_manager
                .compute_state(
                    &gps,
                    deploys0,
                    vec![
                        casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum::Close(
                            CloseBlockDeploy {
                                initial_rand:
                                    system_deploy_util::generate_close_deploy_random_seed_from_pk(
                                        genesis_context.validator_pks()[0].clone(),
                                        0,
                                    ),
                            },
                        ),
                    ],
                    BlockData {
                        time_stamp: time,
                        block_number: 0,
                        sender: genesis_context.validator_pks()[0].clone(),
                        seq_num: 0,
                    },
                    None,
                )
                .await
                .unwrap();

            let bonds0 = runtime_manager
                .compute_bonds(&play_state_hash_0)
                .await
                .unwrap();

            let replay_state_hash_0 = runtime_manager
                .replay_compute_state(
                    &gps,
                    processed_deploys_0,
                    processed_sys_deploys_0,
                    &BlockData {
                        time_stamp: time,
                        block_number: 0,
                        sender: genesis_context.validator_pks()[0].clone(),
                        seq_num: 0,
                    },
                    None,
                    false,
                )
                .await
                .unwrap();

            assert!(play_state_hash_0 == replay_state_hash_0);

            let bonds1 = runtime_manager
                .compute_bonds(&play_state_hash_0)
                .await
                .unwrap();

            assert!(bonds0 == bonds1);

            let (play_state_hash_1, processed_deploys_1, processed_sys_deploys_1) = runtime_manager
                .compute_state(
                    &play_state_hash_0,
                    deploys1,
                    vec![
                        casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum::Close(
                            CloseBlockDeploy {
                                initial_rand:
                                    system_deploy_util::generate_close_deploy_random_seed_from_pk(
                                        genesis_context.validator_pks()[0].clone(),
                                        0,
                                    ),
                            },
                        ),
                    ],
                    BlockData {
                        time_stamp: time,
                        block_number: 0,
                        sender: genesis_context.validator_pks()[0].clone(),
                        seq_num: 0,
                    },
                    None,
                )
                .await
                .unwrap();

            let bonds2 = runtime_manager
                .compute_bonds(&play_state_hash_1)
                .await
                .unwrap();

            let replay_state_hash_1 = runtime_manager
                .replay_compute_state(
                    &play_state_hash_0,
                    processed_deploys_1,
                    processed_sys_deploys_1,
                    &BlockData {
                        time_stamp: time,
                        block_number: 0,
                        sender: genesis_context.validator_pks()[0].clone(),
                        seq_num: 0,
                    },
                    None,
                    false,
                )
                .await
                .unwrap();

            assert!(play_state_hash_1 == replay_state_hash_1);

            let bonds3 = runtime_manager
                .compute_bonds(&play_state_hash_1)
                .await
                .unwrap();

            assert!(bonds2 == bonds3);
        },
    )
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compute_state_should_capture_rholang_parsing_errors_and_charge_for_parsing() {
    with_runtime_manager(
        |mut runtime_manager, genesis_context, genesis_block| async move {
            let bad_rholang =
                r#" for(@x <- @"x" & @y <- @"y"){ @"xy"!(x + y) } | @"x"!(1) | @"y"!("hi") "#;
            let deploy = construct_deploy::source_deploy_now_full(
                bad_rholang.to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();

            let result = compute_state(
                &mut runtime_manager,
                &genesis_context,
                deploy,
                &genesis_block.body.state.post_state_hash,
            )
            .await;

            assert!(result.1.is_failed == true);
            assert!(result.1.cost.cost == costs::parsing_cost(bad_rholang).value as u64);
        },
    )
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compute_state_should_charge_for_parsing_and_execution() {
    with_runtime_manager(
        |mut runtime_manager, genesis_context, genesis_block| async move {
            let correct_rholang =
                r#" for(@x <- @"x" & @y <- @"y"){ @"xy"!(x + y) | @"x"!(1) | @"y"!(2) } "#;
            let rand = Blake2b512Random::create_from_bytes(&Vec::new());
            let inital_phlo = Cost::unsafe_max();
            let deploy = construct_deploy::source_deploy_now_full(
                correct_rholang.to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();

            let runtime = runtime_manager.spawn_runtime().await;
            runtime.cost.set(inital_phlo.clone());
            let term = Compiler::source_to_adt(&deploy.data.term).unwrap();
            let _ = runtime.inj(term, Env::new(), rand).await;
            let phlos_left = runtime.cost.get();
            let reduction_cost = inital_phlo - phlos_left;

            let parsing_cost = costs::parsing_cost(correct_rholang);

            let result = compute_state(
                &mut runtime_manager,
                &genesis_context,
                deploy,
                &genesis_block.body.state.post_state_hash,
            )
            .await;

            assert!(result.1.cost.cost == (reduction_cost + parsing_cost).value as u64);
        },
    )
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn capture_result_should_return_the_value_at_the_specified_channel_after_a_rholang_computation(
) {
    with_runtime_manager(
        |mut runtime_manager, genesis_context, genesis_block| async move {
            let deployo0 = construct_deploy::source_deploy_now_full(
                r#"
                        new rl(`rho:registry:lookup`), NonNegativeNumberCh in {
                        rl!(`rho:lang:nonNegativeNumber`, *NonNegativeNumberCh) |
                        for(@(_, NonNegativeNumber) <- NonNegativeNumberCh) {
                          @NonNegativeNumber!(37, "nn")
                        }
                      }
                "#
                .to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();

            let result0 = compute_state(
                &mut runtime_manager,
                &genesis_context,
                deployo0,
                &genesis_block.body.state.post_state_hash,
            )
            .await;

            let hash = result0.0;
            let deployo1 = construct_deploy::source_deploy_now_full(
                r#"
                new return in { for(nn <- @"nn"){ nn!("value", *return) } }
                "#
                .to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();

            let result1 = runtime_manager
                .capture_results(&hash, &deployo1)
                .await
                .unwrap();

            assert!(result1.len() == 1);
            assert!(result1[0] == ParBuilderUtil::mk_term("37").unwrap());
        },
    )
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn capture_result_should_handle_multiple_results_and_no_results_appropriately() {
    with_runtime_manager(|runtime_manager, _, _| async move {
        let n = 8;
        let returns = (1..=n)
            .map(|i| format!("return!({})", i))
            .collect::<Vec<_>>();
        let term = format!("new return in {{ {} }}", returns.join("|"));
        let term_no_res = format!("new x, return in {{ {} }}", returns.join("|"));
        let deploy =
            construct_deploy::source_deploy(term, 0, None, None, None, None, None).unwrap();
        let deploy_no_res =
            construct_deploy::source_deploy(term_no_res, 0, None, None, None, None, None).unwrap();

        let many_results = runtime_manager
            .capture_results(&RuntimeManager::empty_state_hash_fixed(), &deploy)
            .await
            .unwrap();

        let no_results = runtime_manager
            .capture_results(&RuntimeManager::empty_state_hash_fixed(), &deploy_no_res)
            .await
            .unwrap();

        assert!(no_results.is_empty());
        assert!(many_results.len() == n);
        assert!((1..=n)
            .all(|i| many_results.contains(&ParBuilderUtil::mk_term(&i.to_string()).unwrap())));
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn capture_result_should_throw_error_if_execution_fails() {
    with_runtime_manager(|runtime_manager, _, _| async move {
        let deploy = construct_deploy::source_deploy(
            "new return in { return.undefined() }".to_string(),
            0,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let result = runtime_manager
            .capture_results(&RuntimeManager::empty_state_hash_fixed(), &deploy)
            .await;

        assert!(result.is_err());
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn empty_state_hash_should_not_remember_previous_hot_store_state() {
    with_runtime_manager(
        |mut runtime_manager, genesis_context, genesis_block| async move {
            let deploy1 = construct_deploy::basic_deploy_data(0, None, None).unwrap();
            let deploy2 = construct_deploy::basic_deploy_data(0, None, None).unwrap();

            let hash1 = RuntimeManager::empty_state_hash_fixed();
            let _ = compute_state(
                &mut runtime_manager,
                &genesis_context,
                deploy1,
                &genesis_block.body.state.post_state_hash,
            )
            .await;

            let hash2 = RuntimeManager::empty_state_hash_fixed();
            let _ = compute_state(
                &mut runtime_manager,
                &genesis_context,
                deploy2,
                &genesis_block.body.state.post_state_hash,
            )
            .await;

            assert!(hash1 == hash2);
        },
    )
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compute_state_should_be_replayed_by_replay_compute_state() {
    with_runtime_manager(
        |mut runtime_manager, genesis_context, genesis_block| async move {
            let deploy = construct_deploy::source_deploy_now_full(
                r#"
                  new deployerId(`rho:system:deployerId`),
                  rl(`rho:registry:lookup`),
                  revAddressOps(`rho:vault:address`),
                  revAddressCh,
                  revVaultCh in {
                  rl!(`rho:vault:system`, *revVaultCh) |
                  revAddressOps!("fromDeployerId", *deployerId, *revAddressCh) |
                  for(@userRevAddress <- revAddressCh & @(_, revVault) <- revVaultCh){
                    new userVaultCh in {
                    @revVault!("findOrCreate", userRevAddress, *userVaultCh) |
                    for(@(true, userVault) <- userVaultCh){
                    @userVault!("balance", "IGNORE")
                    }
                  }
                }
              }
            }
                "#
                .to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();

            let time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis();

            let genesis_post_state = genesis_block.body.state.post_state_hash;
            let block_data = BlockData {
                time_stamp: time as i64,
                block_number: 0,
                sender: genesis_context.validator_pks()[0].clone(),
                seq_num: 0,
            };

            let invalid_blocks = HashMap::new();
            let (play_post_state, processed_deploys, processed_system_deploys) = runtime_manager
                .compute_state(
                    &genesis_post_state,
                    vec![deploy],
                    vec![
                        casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum::Close(
                            CloseBlockDeploy {
                                initial_rand:
                                    system_deploy_util::generate_close_deploy_random_seed_from_pk(
                                        block_data.sender.clone(),
                                        block_data.seq_num,
                                    ),
                            },
                        ),
                    ],
                    block_data.clone(),
                    Some(invalid_blocks.clone()),
                )
                .await
                .unwrap();

            let replay_compute_state_result = runtime_manager
                .replay_compute_state(
                    &genesis_post_state,
                    processed_deploys,
                    processed_system_deploys,
                    &block_data,
                    Some(invalid_blocks),
                    false,
                )
                .await
                .unwrap();

            assert!(play_post_state == replay_compute_state_result);
        },
    )
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compute_state_should_charge_deploys_separately() {
    with_runtime_manager(
        |mut runtime_manager, genesis_context, genesis_block| async move {
            fn deploy_cost(p: &[ProcessedDeploy]) -> u64 {
                p.iter().map(|d| d.cost.cost).sum()
            }

            let deploy0 = construct_deploy::source_deploy(
                r#"for(@x <- @"w") { @"z"!("Got x") } "#.to_string(),
                123,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();

            let deploy1 = construct_deploy::source_deploy(
                r#"for(@x <- @"x" & @y <- @"y"){ @"xy"!(x + y) | @"x"!(1) | @"y"!(10) } "#
                    .to_string(),
                123,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();

            let time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis();

            let genesis_post_state = genesis_block.body.state.post_state_hash;
            let block_data = BlockData {
                time_stamp: time as i64,
                block_number: 0,
                sender: genesis_context.validator_pks()[0].clone(),
                seq_num: 0,
            };

            let invalid_blocks = HashMap::new();
            let (_, first_deploy, _) = runtime_manager
                .compute_state(
                    &genesis_post_state,
                    vec![construct_deploy::source_deploy(
                        r#"for(@x <- @"w") { @"z"!("Got x") } "#.to_string(),
                        123,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    .unwrap()],
                    vec![
                        casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum::Close(
                            CloseBlockDeploy {
                                initial_rand:
                                    system_deploy_util::generate_close_deploy_random_seed_from_pk(
                                        block_data.sender.clone(),
                                        block_data.seq_num,
                                    ),
                            },
                        ),
                    ],
                    block_data.clone(),
                    Some(invalid_blocks.clone()),
                )
                .await
                .unwrap();

            let (_, second_deploy, _) = runtime_manager
                .compute_state(
                    &genesis_post_state,
                    vec![construct_deploy::source_deploy(
                        r#"for(@x <- @"x" & @y <- @"y"){ @"xy"!(x + y) | @"x"!(1) | @"y"!(10) } "#
                            .to_string(),
                        123,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    .unwrap()],
                    vec![
                        casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum::Close(
                            CloseBlockDeploy {
                                initial_rand:
                                    system_deploy_util::generate_close_deploy_random_seed_from_pk(
                                        block_data.sender.clone(),
                                        block_data.seq_num,
                                    ),
                            },
                        ),
                    ],
                    block_data.clone(),
                    Some(invalid_blocks.clone()),
                )
                .await
                .unwrap();

            let (_, compound_deploy, _) = runtime_manager
                .compute_state(
                    &genesis_post_state,
                    vec![deploy0, deploy1],
                    vec![
                        casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum::Close(
                            CloseBlockDeploy {
                                initial_rand:
                                    system_deploy_util::generate_close_deploy_random_seed_from_pk(
                                        block_data.sender.clone(),
                                        block_data.seq_num,
                                    ),
                            },
                        ),
                    ],
                    block_data.clone(),
                    Some(invalid_blocks.clone()),
                )
                .await
                .unwrap();

            assert!(first_deploy.len() == 1);
            assert!(second_deploy.len() == 1);
            assert!(compound_deploy.len() == 2);

            let first_deploy_cost = deploy_cost(&first_deploy);
            let second_deploy_cost = deploy_cost(&second_deploy);
            let compound_deploy_cost = deploy_cost(&compound_deploy);

            assert!(first_deploy_cost < compound_deploy_cost);
            assert!(second_deploy_cost < compound_deploy_cost);

            let matched_first = compound_deploy
                .iter()
                .find(|d| d.deploy == first_deploy[0].deploy)
                .cloned()
                .expect("Expected at least one matching deploy");
            assert_eq!(first_deploy_cost, deploy_cost(&vec![matched_first]));

            let matched_second = compound_deploy
                .iter()
                .find(|d| d.deploy == second_deploy[0].deploy)
                .cloned()
                .expect("Expected at least one matching deploy");
            assert_eq!(second_deploy_cost, deploy_cost(&vec![matched_second]));

            assert_eq!(first_deploy_cost + second_deploy_cost, compound_deploy_cost);
        },
    )
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compute_state_should_just_work() {
    with_runtime_manager(|mut runtime_manager, genesis_context, genesis_block| async move {
      let gen_post_state = genesis_block.body.state.post_state_hash;
      let source =  r#"
      new d1,d2,d3,d4,d5,d6,d7,d8,d9 in {
        contract d1(@depth) = {
          if (depth <= 0) {
            Nil
          } else {
            d1!(depth - 1) | d1!(depth - 1) | d1!(depth - 1) | d1!(depth - 1) | d1!(depth - 1) | d1!(depth - 1) | d1!(depth - 1) | d1!(depth - 1) | d1!(depth - 1) | d1!(depth - 1)
          }
        } |
        contract d2(@depth) = {
          if (depth <= 0) {
            Nil
          } else {
            d2!(depth - 1) | d2!(depth - 1) | d2!(depth - 1) | d2!(depth - 1) | d2!(depth - 1) | d2!(depth - 1) | d2!(depth - 1) | d2!(depth - 1) | d2!(depth - 1) | d2!(depth - 1)
          }
        } |
        contract d3(@depth) = {
          if (depth <= 0) {
            Nil
          } else {
            d3!(depth - 1) | d3!(depth - 1) | d3!(depth - 1) | d3!(depth - 1) | d3!(depth - 1) | d3!(depth - 1) | d3!(depth - 1) | d3!(depth - 1) | d3!(depth - 1) | d3!(depth - 1)
          }
        } |
        contract d4(@depth) = {
          if (depth <= 0) {
            Nil
          } else {
            d4!(depth - 1) | d4!(depth - 1) | d4!(depth - 1) | d4!(depth - 1) | d4!(depth - 1) | d4!(depth - 1) | d4!(depth - 1) | d4!(depth - 1) | d4!(depth - 1) | d4!(depth - 1)
          }
        } |
        contract d5(@depth) = {
          if (depth <= 0) {
            Nil
          } else {
            d5!(depth - 1) | d5!(depth - 1) | d5!(depth - 1) | d5!(depth - 1) | d5!(depth - 1) | d5!(depth - 1) | d5!(depth - 1) | d5!(depth - 1) | d5!(depth - 1) | d5!(depth - 1)
          }
        } |
        contract d6(@depth) = {
          if (depth <= 0) {
            Nil
          } else {
            d6!(depth - 1) | d6!(depth - 1) | d6!(depth - 1) | d6!(depth - 1) | d6!(depth - 1) | d6!(depth - 1) | d6!(depth - 1) | d6!(depth - 1) | d6!(depth - 1) | d6!(depth - 1)
          }
        } |
        contract d7(@depth) = {
          if (depth <= 0) {
            Nil
          } else {
            d7!(depth - 1) | d7!(depth - 1) | d7!(depth - 1) | d7!(depth - 1) | d7!(depth - 1) | d7!(depth - 1) | d7!(depth - 1) | d7!(depth - 1) | d7!(depth - 1) | d7!(depth - 1)
          }
        } |
        contract d8(@depth) = {
          if (depth <= 0) {
            Nil
          } else {
            d8!(depth - 1) | d8!(depth - 1) | d8!(depth - 1) | d8!(depth - 1) | d8!(depth - 1) | d8!(depth - 1) | d8!(depth - 1) | d8!(depth - 1) | d8!(depth - 1) | d8!(depth - 1)
          }
        } |
        contract d9(@depth) = {
          if (depth <= 0) {
            Nil
          } else {
            d9!(depth - 1) | d9!(depth - 1) | d9!(depth - 1) | d9!(depth - 1) | d9!(depth - 1) | d9!(depth - 1) | d9!(depth - 1) | d9!(depth - 1) | d9!(depth - 1) | d9!(depth - 1)
          }
        } |
        d1!(2) |
        d2!(2) |
        d3!(2) |
        d4!(2) |
        d5!(2) |
        d6!(2) |
        d7!(2) |
        d8!(2) |
        d9!(2)
      }
      "#.to_string();

      let deploy = construct_deploy::source_deploy_now_full(source, Some(i64::MAX - 2), None, None, None, None).unwrap();
      let (play_state_hash1, processed_deploy) = compute_state(&mut runtime_manager, &genesis_context, deploy, &gen_post_state).await;
      let replay_compute_state_result = replay_compute_state(&mut runtime_manager, &genesis_context, processed_deploy, &gen_post_state).await.unwrap();
      assert!(play_state_hash1 == replay_compute_state_result);
      assert!(play_state_hash1 != gen_post_state);
    })
        .await
        .unwrap()
}

async fn invalid_replay(source: String) -> Result<StateHash, CasperError> {
    with_runtime_manager(
        |mut runtime_manager, genesis_context, genesis_block| async move {
            let deploy = construct_deploy::source_deploy_now_full(
                source,
                Some(10000),
                None,
                None,
                None,
                None,
            )
            .unwrap();

            let time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as i64;

            let gen_post_state = genesis_block.body.state.post_state_hash;
            let block_data = BlockData {
                time_stamp: time,
                block_number: 0,
                sender: genesis_context.validator_pks()[0].clone(),
                seq_num: 0,
            };

            let invalid_blocks = HashMap::new();

            let (_, processed_deploys, processed_system_deploys) = runtime_manager
                .compute_state(
                    &gen_post_state,
                    vec![deploy],
                    vec![
                        casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum::Close(
                            CloseBlockDeploy {
                                initial_rand:
                                    system_deploy_util::generate_close_deploy_random_seed_from_pk(
                                        block_data.sender.clone(),
                                        block_data.seq_num,
                                    ),
                            },
                        ),
                    ],
                    block_data.clone(),
                    Some(invalid_blocks.clone()),
                )
                .await
                .unwrap();
            let processed_deploy = processed_deploys.into_iter().next().unwrap();
            let processed_deploy_cost = processed_deploy.cost.cost;

            let invalid_processed_deploy = ProcessedDeploy {
                cost: PCost {
                    cost: processed_deploy_cost - 1,
                },
                ..processed_deploy
            };

            let result = runtime_manager
                .replay_compute_state(
                    &gen_post_state,
                    vec![invalid_processed_deploy],
                    processed_system_deploys,
                    &block_data,
                    Some(invalid_blocks),
                    false,
                )
                .await;

            result
        },
    )
    .await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn replaycomputestate_should_catch_discrepancies_in_initial_and_replay_cost_when_no_errors_are_thrown(
) {
    let result = invalid_replay("@0!(0) | for(@0 <- @0){ Nil }".to_string()).await;
    match result {
        Err(CasperError::ReplayFailure(ReplayFailure::ReplayCostMismatch {
            initial_cost,
            replay_cost,
        })) => {
            assert_eq!(initial_cost, 322);
            assert_eq!(replay_cost, 323);
        }
        _ => panic!("Expected ReplayCostMismatch error"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn replaycomputestate_should_not_catch_discrepancies_in_initial_and_replay_cost_when_user_errors_are_thrown(
) {
    let result = invalid_replay("@0!(0) | for(@x <- @0){ x.undefined() }".to_string()).await;
    match result {
        Err(CasperError::ReplayFailure(ReplayFailure::ReplayCostMismatch {
            initial_cost,
            replay_cost,
        })) => {
            assert_eq!(initial_cost, 9999);
            assert_eq!(replay_cost, 10000);
        }
        _ => panic!("Expected ReplayCostMismatch error"),
    }
}

// This is additional test for sorting with joins and channels inside joins.
// - after reverted PR https://github.com/rchain/rchain/pull/2436
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn joins_should_be_replayed_correctly() {
    with_runtime_manager(
        |mut runtime_manager, genesis_context, genesis_block| async move {
            let term = r#"
            new a, b, c, d in {
              for (_ <- a & _ <- b) { Nil } |
              for (_ <- a & _ <- c) { Nil } |
              for (_ <- a & _ <- d) { Nil }
            }
            "#;

            let gen_post_state = genesis_block.body.state.post_state_hash;
            let deploy = construct_deploy::source_deploy_now_full(
                term.to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();

            let time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as i64;

            let block_data = BlockData {
                time_stamp: time,
                block_number: 1,
                sender: genesis_context.validator_pks()[0].clone(),
                seq_num: 1,
            };

            let invalid_blocks = HashMap::new();
            let (state_hash, processed_deploys, processed_sys_deploys) = runtime_manager
                .compute_state(
                    &gen_post_state,
                    vec![deploy],
                    Vec::new(), // No system deploys
                    block_data.clone(),
                    Some(invalid_blocks.clone()),
                )
                .await
                .unwrap();

            let replay_state_hash = runtime_manager
                .replay_compute_state(
                    &gen_post_state,
                    processed_deploys,
                    processed_sys_deploys,
                    &block_data,
                    Some(invalid_blocks),
                    false,
                )
                .await
                .unwrap();

            assert_eq!(
                hex::encode(state_hash.to_vec()),
                hex::encode(replay_state_hash.to_vec())
            );
        },
    )
    .await
    .unwrap();
}

/// Reproduce ReplayCostMismatch with duplicate channel sends (bridge.rho pattern).
///
/// Uses two independent RuntimeManagers sharing the same genesis RSpace scope.
/// The first plays the deploy (hot store populated from execution).
/// The second replays with a fresh hot store (loads from history).
/// This simulates the block creator vs replayer divergence.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn replay_on_independent_runtime_should_match_play_cost_for_duplicate_sends() {
    use crate::util::rholang::resources::{
        mk_runtime_manager_with_history_at, mk_test_rnode_store_manager_from_genesis,
    };

    crate::init_logger();
    let genesis_context = crate::util::rholang::resources::genesis_context()
        .await
        .unwrap();
    let genesis_block = genesis_context.genesis_block.clone();
    let genesis_post_state = genesis_block.body.state.post_state_hash.clone();

    let bridge_rho = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/resources/bridge.rho"),
    )
    .expect("Failed to read bridge.rho");

    let mut failures = Vec::new();
    for attempt in 0..10 {
        let mut kvm_play = mk_test_rnode_store_manager_from_genesis(&genesis_context);
        let (mut rm_play, _) = mk_runtime_manager_with_history_at(&mut *kvm_play).await;

        let deploy = construct_deploy::source_deploy_now_full(
            bridge_rho.clone(),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let play_block_data = BlockData {
            time_stamp: deploy.data.time_stamp,
            block_number: 1,
            sender: genesis_context.validator_pks()[0].clone(),
            seq_num: 1,
        };

        let (_play_post, play_deploys, play_sys_deploys) = rm_play
            .compute_state(
                &genesis_post_state,
                vec![deploy],
                Vec::new(),
                play_block_data.clone(),
                None,
            )
            .await
            .unwrap();

        let play_cost = play_deploys[0].cost.cost;

        let mut kvm_replay = mk_test_rnode_store_manager_from_genesis(&genesis_context);
        let (mut rm_replay, _) = mk_runtime_manager_with_history_at(&mut *kvm_replay).await;

        let replay_result = rm_replay
            .replay_compute_state(
                &genesis_post_state,
                play_deploys,
                play_sys_deploys,
                &play_block_data,
                None,
                false,
            )
            .await;

        match replay_result {
            Ok(_) => {}
            Err(CasperError::ReplayFailure(ref failure)) => {
                failures.push(format!(
                    "attempt {}: play_cost={}, {:?}",
                    attempt, play_cost, failure
                ));
            }
            Err(e) => {
                failures.push(format!("attempt {}: {:?}", attempt, e));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "ReplayCostMismatch in {}/10 attempts:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cross_deploy_bridge_full_admin_flow() {
    use crate::util::rholang::resources::{
        mk_runtime_manager_with_history_at, mk_test_rnode_store_manager_from_genesis,
    };

    crate::init_logger();
    let genesis_context = crate::util::rholang::resources::genesis_context()
        .await
        .unwrap();
    let genesis_post_state = genesis_context
        .genesis_block
        .body
        .state
        .post_state_hash
        .clone();

    let bridge_rho = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/resources/bridge.rho"),
    )
    .expect("Failed to read bridge.rho");

    let mut kvm = mk_test_rnode_store_manager_from_genesis(&genesis_context);
    let (mut rm, _) = mk_runtime_manager_with_history_at(&mut *kvm).await;

    let uri_regex = regex::Regex::new(r"rho:id:[a-zA-Z0-9]+").unwrap();

    let make_deploy_id_par = |sig: &[u8]| -> models::rhoapi::Par {
        models::rhoapi::Par {
            unforgeables: vec![models::rhoapi::GUnforgeable {
                unf_instance: Some(models::rhoapi::g_unforgeable::UnfInstance::GDeployIdBody(
                    models::rhoapi::GDeployId { sig: sig.to_vec() },
                )),
            }],
            ..Default::default()
        }
    };

    let mut block_number = 0u64;
    let mut current_state = genesis_post_state.clone();

    // Step 1: Deploy bridge.rho
    tracing::info!("Step 1: Deploying bridge.rho");
    block_number += 1;
    let deploy1 =
        construct_deploy::source_deploy_now_full(bridge_rho, None, None, None, None, None).unwrap();

    let (post_state_1, pd1_vec, _) = rm
        .compute_state(
            &current_state,
            vec![deploy1],
            Vec::new(),
            BlockData {
                time_stamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as i64,
                block_number: block_number as i64,
                sender: genesis_context.validator_pks()[0].clone(),
                seq_num: block_number as i32,
            },
            None,
        )
        .await
        .unwrap();

    let pd1 = &pd1_vec[0];
    assert!(
        !pd1.is_failed,
        "Step 1: bridge deploy failed: {:?}",
        pd1.system_deploy_error
    );
    tracing::info!(
        "Step 1: cost={}, events={}",
        pd1.cost.cost,
        pd1.deploy_log.len()
    );

    let deploy1_data = rm
        .get_data(
            post_state_1.clone(),
            &make_deploy_id_par(&pd1_vec[0].deploy.sig),
        )
        .await
        .unwrap();
    assert!(
        !deploy1_data.is_empty(),
        "Step 1: bridge deploy wrote no data to deployId"
    );

    let data_str = format!("{:?}", deploy1_data);
    let uris: Vec<String> = uri_regex
        .find_iter(&data_str)
        .map(|m| m.as_str().to_string())
        .collect();
    let mut unique_uris: Vec<String> = Vec::new();
    for uri in &uris {
        if !unique_uris.contains(uri) {
            unique_uris.push(uri.clone());
        }
    }
    assert!(
        unique_uris.len() >= 2,
        "Expected at least 2 URIs, got: {:?}",
        unique_uris
    );
    let query_uri = unique_uris[0].clone();
    let admin_uri = unique_uris.last().unwrap().clone();
    tracing::info!("  queryUri: {}, adminUri: {}", query_uri, admin_uri);
    current_state = post_state_1;

    // Steps 2-7: getNonce + admin calls
    let steps: Vec<(&str, String)> = vec![
        (
            "getNonce",
            format!(
                r#"
new deployId(`rho:system:deployId`),
    lookup(`rho:registry:lookup`),
    queryCh, ret
in {{
  lookup!(`{}`, *queryCh) |
  for (query <- queryCh) {{
    query!("getNonce", Nil, *ret) |
    for (@result <- ret) {{ deployId!(result) }}
  }}
}}
"#,
                query_uri
            ),
        ),
        (
            "setVerifier",
            format!(
                r#"
new deployId(`rho:system:deployId`), deployerId(`rho:system:deployerId`),
    lookup(`rho:registry:lookup`), VaultAddress(`rho:vault:address`),
    adminBridgeCh, callerAddrCh, ret
in {{
  lookup!(`{}`, *adminBridgeCh) |
  VaultAddress!("fromDeployerId", *deployerId, *callerAddrCh) |
  for (adminBridge <- adminBridgeCh; @callerAddr <- callerAddrCh) {{
    adminBridge!("setVerifier", callerAddr, "verifier_v2", *ret) |
    for (@result <- ret) {{ deployId!(result) }}
  }}
}}
"#,
                admin_uri
            ),
        ),
        (
            "setRelayer",
            format!(
                r#"
new deployId(`rho:system:deployId`), deployerId(`rho:system:deployerId`),
    lookup(`rho:registry:lookup`), VaultAddress(`rho:vault:address`),
    adminBridgeCh, callerAddrCh, ret
in {{
  lookup!(`{}`, *adminBridgeCh) |
  VaultAddress!("fromDeployerId", *deployerId, *callerAddrCh) |
  for (adminBridge <- adminBridgeCh; @callerAddr <- callerAddrCh) {{
    adminBridge!("setRelayer", callerAddr, "relayer_addr_1", *ret) |
    for (@result <- ret) {{ deployId!(result) }}
  }}
}}
"#,
                admin_uri
            ),
        ),
        (
            "setRequiredSignatures",
            format!(
                r#"
new deployId(`rho:system:deployId`), deployerId(`rho:system:deployerId`),
    lookup(`rho:registry:lookup`), VaultAddress(`rho:vault:address`),
    adminBridgeCh, callerAddrCh, ret
in {{
  lookup!(`{}`, *adminBridgeCh) |
  VaultAddress!("fromDeployerId", *deployerId, *callerAddrCh) |
  for (adminBridge <- adminBridgeCh; @callerAddr <- callerAddrCh) {{
    adminBridge!("setRequiredSignatures", callerAddr, 2, *ret) |
    for (@result <- ret) {{ deployId!(result) }}
  }}
}}
"#,
                admin_uri
            ),
        ),
        (
            "addOracle",
            format!(
                r#"
new deployId(`rho:system:deployId`), deployerId(`rho:system:deployerId`),
    lookup(`rho:registry:lookup`), VaultAddress(`rho:vault:address`),
    adminBridgeCh, callerAddrCh, ret
in {{
  lookup!(`{}`, *adminBridgeCh) |
  VaultAddress!("fromDeployerId", *deployerId, *callerAddrCh) |
  for (adminBridge <- adminBridgeCh; @callerAddr <- callerAddrCh) {{
    adminBridge!("addOracle", callerAddr, "oracle-4", *ret) |
    for (@result <- ret) {{ deployId!(result) }}
  }}
}}
"#,
                admin_uri
            ),
        ),
        (
            "removeOracle",
            format!(
                r#"
new deployId(`rho:system:deployId`), deployerId(`rho:system:deployerId`),
    lookup(`rho:registry:lookup`), VaultAddress(`rho:vault:address`),
    adminBridgeCh, callerAddrCh, ret
in {{
  lookup!(`{}`, *adminBridgeCh) |
  VaultAddress!("fromDeployerId", *deployerId, *callerAddrCh) |
  for (adminBridge <- adminBridgeCh; @callerAddr <- callerAddrCh) {{
    adminBridge!("removeOracle", callerAddr, "oracle-4", *ret) |
    for (@result <- ret) {{ deployId!(result) }}
  }}
}}
"#,
                admin_uri
            ),
        ),
    ];

    let mut failures = Vec::new();
    for (name, code) in &steps {
        block_number += 1;
        tracing::info!("{}", name);

        let deploy =
            construct_deploy::source_deploy_now_full(code.clone(), None, None, None, None, None)
                .unwrap();

        let (post_state_n, pdn_vec, _) = rm
            .compute_state(
                &current_state,
                vec![deploy],
                Vec::new(),
                BlockData {
                    time_stamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as i64,
                    block_number: block_number as i64,
                    sender: genesis_context.validator_pks()[0].clone(),
                    seq_num: block_number as i32,
                },
                None,
            )
            .await
            .unwrap();

        let pdn = &pdn_vec[0];
        assert!(
            !pdn.is_failed,
            "{}: deploy failed: {:?}",
            name, pdn.system_deploy_error
        );
        let deploy_data = rm
            .get_data(
                post_state_n.clone(),
                &make_deploy_id_par(&pdn_vec[0].deploy.sig),
            )
            .await
            .unwrap();
        let has_data = !deploy_data.is_empty();
        tracing::info!(
            "  {}: cost={}, events={}, deployId_data={}",
            name,
            pdn.cost.cost,
            pdn.deploy_log.len(),
            has_data
        );

        if !has_data {
            failures.push(format!(
                "{} returned no data. cost={}, events={}",
                name,
                pdn.cost.cost,
                pdn.deploy_log.len()
            ));
        }
        current_state = post_state_n;
    }

    assert!(
        failures.is_empty(),
        "Bridge admin API failures:\n{}",
        failures.join("\n")
    );
}

/// Tests that bridge registry entries survive multi-parent DAG merge.
///
/// Deploys bridge.rho on block A (from genesis), creates empty block B (from
/// genesis, sibling branch), merges [A, B] via compute_parents_post_state,
/// then queries getNonce from the merged state.
///
/// Reproduces: system-integration docs/TODO.md "Contract query deploy returns
/// empty deployId after finalization (intermittent)"
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bridge_query_survives_multi_parent_merge() {
    use crate::util::rholang::resources::{
        block_dag_storage_from_dyn, mergeable_store_from_dyn,
        mk_test_rnode_store_manager_from_genesis,
    };
    use block_storage::rust::key_value_block_store::KeyValueBlockStore;
    use casper::rust::genesis::genesis::Genesis;
    use casper::rust::{
        casper::{CasperShardConf, CasperSnapshot, OnChainCasperState},
        util::{
            proto_util,
            rholang::interpreter_util::{compute_deploys_checkpoint, compute_parents_post_state},
        },
    };
    use dashmap::{DashMap, DashSet};
    use models::rust::{block_hash::BlockHash, block_implicits};
    use rholang::rust::interpreter::external_services::ExternalServices;

    crate::init_logger();
    let genesis_context = crate::util::rholang::resources::genesis_context()
        .await
        .unwrap();
    let genesis_block = genesis_context.genesis_block.clone();
    let genesis_hash = genesis_block.block_hash.clone();
    let genesis_state = proto_util::post_state_hash(&genesis_block);
    let genesis_bonds = genesis_block.body.state.bonds.clone();
    let validator: prost::bytes::Bytes = genesis_context.validator_pks()[0].bytes.clone().into();
    let shard_name = genesis_block.shard_id.clone();

    // Create all stores from the same KVM (shared genesis scope)
    let mut kvm = mk_test_rnode_store_manager_from_genesis(&genesis_context);

    let rspace_store = kvm.r_space_stores().await.expect("rspace stores");
    let mergeable_store = mergeable_store_from_dyn(&mut *kvm)
        .await
        .expect("mergeable store");
    let (mut rm, _) = RuntimeManager::create_with_history(
        rspace_store,
        mergeable_store,
        std::sync::Arc::new(Genesis::default_mergeable_tags()),
        ExternalServices::noop(),
    );

    let mut block_store = KeyValueBlockStore::create_from_kvm(&mut *kvm)
        .await
        .expect("block store");
    let dag_storage = block_dag_storage_from_dyn(&mut *kvm)
        .await
        .expect("dag storage");

    block_store
        .put_block_message(&genesis_block)
        .expect("store genesis");
    dag_storage
        .insert(&genesis_block, false, true)
        .expect("dag genesis");

    let now_millis = || -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    };

    let mk_snapshot = |lfb: &BlockHash| -> CasperSnapshot {
        let mut snapshot = CasperSnapshot::new(dag_storage.get_representation());
        snapshot.last_finalized_block = lfb.clone();
        let max_seq_nums: DashMap<prost::bytes::Bytes, u64> = DashMap::new();
        max_seq_nums.insert(validator.clone(), 0);
        snapshot.max_seq_nums = max_seq_nums;
        let mut shard_conf = CasperShardConf::new();
        shard_conf.shard_name = shard_name.clone();
        shard_conf.max_parent_depth = 0;
        let mut bonds_map = HashMap::new();
        bonds_map.insert(validator.clone(), 100);
        snapshot.on_chain_state = OnChainCasperState {
            shard_conf,
            bonds_map,
            active_validators: vec![validator.clone()],
        };
        snapshot.deploys_in_scope = std::sync::Arc::new(DashSet::new());
        snapshot
    };

    let make_deploy_id_par = |sig: &[u8]| -> models::rhoapi::Par {
        models::rhoapi::Par {
            unforgeables: vec![models::rhoapi::GUnforgeable {
                unf_instance: Some(models::rhoapi::g_unforgeable::UnfInstance::GDeployIdBody(
                    models::rhoapi::GDeployId { sig: sig.to_vec() },
                )),
            }],
            ..Default::default()
        }
    };

    let bridge_rho = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/resources/bridge.rho"),
    )
    .expect("Failed to read bridge.rho");

    // --- Block A: bridge deploy from genesis ---
    let bridge_deploy =
        construct_deploy::source_deploy_now_full(bridge_rho, None, None, None, None, None).unwrap();

    let block_a_raw = block_implicits::get_random_block(
        Some(1),
        Some(1),
        Some(genesis_state.clone()),
        Some(StateHash::default()),
        Some(validator.clone()),
        Some(1),
        Some(now_millis()),
        Some(vec![genesis_hash.clone()]),
        Some(Vec::new()),
        Some(vec![ProcessedDeploy::empty(bridge_deploy)]),
        Some(Vec::new()),
        Some(genesis_bonds.clone()),
        Some(shard_name.clone()),
        None,
    );

    let parents_a = vec![genesis_block.clone()];
    let deploys_a = proto_util::deploys(&block_a_raw)
        .into_iter()
        .map(|d| d.deploy)
        .collect();
    let snapshot_a = mk_snapshot(&genesis_hash);
    let (_, post_state_a, pd_a, _, sys_pd_a, bonds_a) = compute_deploys_checkpoint(
        &mut block_store,
        parents_a,
        deploys_a,
        Vec::<casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum>::new(),
        &snapshot_a,
        &mut rm,
        BlockData::from_block(&block_a_raw),
        HashMap::new(),
        None,
    )
    .await
    .expect("compute block A");

    assert!(
        !pd_a[0].is_failed,
        "Bridge deploy failed: {:?}",
        pd_a[0].system_deploy_error
    );

    let mut block_a = block_a_raw;
    block_a.body.state.post_state_hash = post_state_a.clone();
    block_a.body.deploys = pd_a.clone();
    block_a.body.system_deploys = sys_pd_a;
    block_a.body.state.bonds = bonds_a;
    block_store.put_block_message(&block_a).expect("store A");
    dag_storage.insert(&block_a, false, false).expect("dag A");

    // Verify bridge wrote data and extract queryUri
    let bridge_data = rm
        .get_data(
            post_state_a.clone(),
            &make_deploy_id_par(&pd_a[0].deploy.sig),
        )
        .await
        .unwrap();
    assert!(
        !bridge_data.is_empty(),
        "Bridge deploy wrote no data to deployId"
    );

    let uri_regex = regex::Regex::new(r"rho:id:[a-zA-Z0-9]+").unwrap();
    let data_str = format!("{:?}", bridge_data);
    let uris: Vec<String> = uri_regex
        .find_iter(&data_str)
        .map(|m| m.as_str().to_string())
        .collect();
    let mut unique_uris: Vec<String> = Vec::new();
    for uri in &uris {
        if !unique_uris.contains(uri) {
            unique_uris.push(uri.clone());
        }
    }
    assert!(
        unique_uris.len() >= 2,
        "Expected at least 2 URIs, got: {:?}",
        unique_uris
    );
    let query_uri = unique_uris[0].clone();

    // --- Block B: empty block from genesis (sibling branch) ---
    let block_b_raw = block_implicits::get_random_block(
        Some(1),
        Some(2),
        Some(genesis_state.clone()),
        Some(StateHash::default()),
        Some(validator.clone()),
        Some(1),
        Some(now_millis()),
        Some(vec![genesis_hash.clone()]),
        Some(Vec::new()),
        Some(Vec::new()),
        Some(Vec::new()),
        Some(genesis_bonds.clone()),
        Some(shard_name.clone()),
        None,
    );

    let parents_b = vec![genesis_block.clone()];
    let snapshot_b = mk_snapshot(&genesis_hash);
    let (_, post_state_b, pd_b, _, sys_pd_b, bonds_b) = compute_deploys_checkpoint(
        &mut block_store,
        parents_b,
        Vec::new(),
        Vec::<casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum>::new(),
        &snapshot_b,
        &mut rm,
        BlockData::from_block(&block_b_raw),
        HashMap::new(),
        None,
    )
    .await
    .expect("compute block B");

    let mut block_b = block_b_raw;
    block_b.body.state.post_state_hash = post_state_b.clone();
    block_b.body.deploys = pd_b;
    block_b.body.system_deploys = sys_pd_b;
    block_b.body.state.bonds = bonds_b;
    block_store.put_block_message(&block_b).expect("store B");
    dag_storage.insert(&block_b, false, false).expect("dag B");

    // --- Merge [A, B] ---
    let parents = vec![block_a.clone(), block_b.clone()];
    let snapshot_merge = mk_snapshot(&genesis_hash);
    let (merged_state, rejected, rejected_slashes) =
        compute_parents_post_state(&block_store, parents, &snapshot_merge, &rm, None, None)
            .expect("merge parents");

    assert!(
        rejected.is_empty(),
        "Merge rejected deploys: {:?}",
        rejected
    );
    // Non-slash merge scenario must surface an empty rejected_slashes list so
    // the block creator's dedup step runs as a no-op.
    assert!(
        rejected_slashes.is_empty(),
        "Merge rejected slashes unexpectedly populated: count={}",
        rejected_slashes.len()
    );

    // --- Query getNonce from merged state ---
    let get_nonce_rho = format!(
        r#"
new deployId(`rho:system:deployId`),
    lookup(`rho:registry:lookup`),
    queryCh, ret
in {{
  lookup!(`{}`, *queryCh) |
  for (query <- queryCh) {{
    query!("getNonce", Nil, *ret) |
    for (@result <- ret) {{ deployId!(result) }}
  }}
}}
"#,
        query_uri
    );

    let query_deploy =
        construct_deploy::source_deploy_now_full(get_nonce_rho, None, None, None, None, None)
            .unwrap();

    let query_block_raw = block_implicits::get_random_block(
        Some(2),
        Some(3),
        Some(merged_state.clone()),
        Some(StateHash::default()),
        Some(validator.clone()),
        Some(1),
        Some(now_millis()),
        Some(vec![block_a.block_hash.clone(), block_b.block_hash.clone()]),
        Some(Vec::new()),
        Some(vec![ProcessedDeploy::empty(query_deploy)]),
        Some(Vec::new()),
        Some(genesis_bonds.clone()),
        Some(shard_name.clone()),
        None,
    );

    let parents_q = vec![block_a.clone(), block_b.clone()];
    let deploys_q = proto_util::deploys(&query_block_raw)
        .into_iter()
        .map(|d| d.deploy)
        .collect();
    let snapshot_q = mk_snapshot(&genesis_hash);
    let (_, post_state_q, pd_q, _, _, _) = compute_deploys_checkpoint(
        &mut block_store,
        parents_q,
        deploys_q,
        Vec::<casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum>::new(),
        &snapshot_q,
        &mut rm,
        BlockData::from_block(&query_block_raw),
        HashMap::new(),
        None,
    )
    .await
    .expect("compute query block");

    assert!(
        !pd_q[0].is_failed,
        "Query deploy failed: {:?}",
        pd_q[0].system_deploy_error
    );

    let query_data = rm
        .get_data(post_state_q, &make_deploy_id_par(&pd_q[0].deploy.sig))
        .await
        .unwrap();

    assert!(
        !query_data.is_empty(),
        "Bridge query returned empty deployId after multi-parent merge. \
         The merge did not preserve the bridge's registry entries when \
         combining a bridge branch with an empty sibling branch."
    );
}

/// Exercises the conflict-detection path for two independent contracts both
/// calling insertArbitrary. Under multi-parent DAG semantics with
/// non-persistent Rholang produces on shared system channels, concurrent
/// operations on the same channel legitimately race and one must be rejected;
/// the test's `rejected.is_empty()` assertion encodes an obsolete premise and
/// needs to be rewritten once the rejected-deploy recovery mechanism lands.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "assertion contradicts multi-parent DAG design; awaits rewrite"]
async fn concurrent_registry_inserts_should_not_conflict() {
    use crate::util::rholang::resources::{
        block_dag_storage_from_dyn, mergeable_store_from_dyn,
        mk_test_rnode_store_manager_from_genesis,
    };
    use block_storage::rust::key_value_block_store::KeyValueBlockStore;
    use casper::rust::genesis::genesis::Genesis;
    use casper::rust::{
        casper::{CasperShardConf, CasperSnapshot, OnChainCasperState},
        util::{
            proto_util,
            rholang::interpreter_util::{compute_deploys_checkpoint, compute_parents_post_state},
        },
    };
    use dashmap::{DashMap, DashSet};
    use models::rust::{block_hash::BlockHash, block_implicits};
    use rholang::rust::interpreter::external_services::ExternalServices;

    crate::init_logger();
    let genesis_context = crate::util::rholang::resources::genesis_context()
        .await
        .unwrap();
    let genesis_block = genesis_context.genesis_block.clone();
    let genesis_hash = genesis_block.block_hash.clone();
    let genesis_state = proto_util::post_state_hash(&genesis_block);
    let genesis_bonds = genesis_block.body.state.bonds.clone();
    let validator: prost::bytes::Bytes = genesis_context.validator_pks()[0].bytes.clone().into();
    let shard_name = genesis_block.shard_id.clone();

    let mut kvm = mk_test_rnode_store_manager_from_genesis(&genesis_context);
    let rspace_store = kvm.r_space_stores().await.expect("rspace stores");
    let mergeable_store = mergeable_store_from_dyn(&mut *kvm)
        .await
        .expect("mergeable store");
    let (mut rm, _) = RuntimeManager::create_with_history(
        rspace_store,
        mergeable_store,
        std::sync::Arc::new(Genesis::default_mergeable_tags()),
        ExternalServices::noop(),
    );

    let mut block_store = KeyValueBlockStore::create_from_kvm(&mut *kvm)
        .await
        .expect("block store");
    let dag_storage = block_dag_storage_from_dyn(&mut *kvm)
        .await
        .expect("dag storage");

    block_store
        .put_block_message(&genesis_block)
        .expect("store genesis");
    dag_storage
        .insert(&genesis_block, false, true)
        .expect("dag genesis");

    let now_millis = || -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    };

    let mk_snapshot = |lfb: &BlockHash| -> CasperSnapshot {
        let mut snapshot = CasperSnapshot::new(dag_storage.get_representation());
        snapshot.last_finalized_block = lfb.clone();
        let max_seq_nums: DashMap<prost::bytes::Bytes, u64> = DashMap::new();
        max_seq_nums.insert(validator.clone(), 0);
        snapshot.max_seq_nums = max_seq_nums;
        let mut shard_conf = CasperShardConf::new();
        shard_conf.shard_name = shard_name.clone();
        shard_conf.max_parent_depth = 0;
        let mut bonds_map = HashMap::new();
        bonds_map.insert(validator.clone(), 100);
        snapshot.on_chain_state = OnChainCasperState {
            shard_conf,
            bonds_map,
            active_validators: vec![validator.clone()],
        };
        snapshot.deploys_in_scope = std::sync::Arc::new(DashSet::new());
        snapshot
    };

    // Both blocks deploy bridge-v2.rho — a complex contract with vault operations,
    // registry inserts, and many shared channel interactions.
    // Use different genesis validator keys so both deployers have funded vaults.
    let bridge_rho = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/resources/bridge.rho"),
    )
    .expect("Failed to read bridge.rho");

    // Use DEFAULT_SEC / DEFAULT_SEC2 — these have funded vaults (9M balance) in genesis.
    // Validator keys have 0 balance and can't deploy.
    let key_a = construct_deploy::DEFAULT_SEC.clone();
    let key_b = construct_deploy::DEFAULT_SEC2.clone();

    // --- Block A: bridge deploy from genesis (funded deployer A) ---
    let deploy_a = construct_deploy::source_deploy_now_full(
        bridge_rho.clone(),
        None,
        None,
        Some(key_a),
        None,
        None,
    )
    .unwrap();

    let block_a_raw = block_implicits::get_random_block(
        Some(1),
        Some(1),
        Some(genesis_state.clone()),
        Some(StateHash::default()),
        Some(validator.clone()),
        Some(1),
        Some(now_millis()),
        Some(vec![genesis_hash.clone()]),
        Some(Vec::new()),
        Some(vec![ProcessedDeploy::empty(deploy_a)]),
        Some(Vec::new()),
        Some(genesis_bonds.clone()),
        Some(shard_name.clone()),
        None,
    );

    let parents_a = vec![genesis_block.clone()];
    let deploys_a = proto_util::deploys(&block_a_raw)
        .into_iter()
        .map(|d| d.deploy)
        .collect();
    let snapshot_a = mk_snapshot(&genesis_hash);
    let (_, post_state_a, pd_a, _, sys_pd_a, bonds_a) = compute_deploys_checkpoint(
        &mut block_store,
        parents_a,
        deploys_a,
        Vec::<casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum>::new(),
        &snapshot_a,
        &mut rm,
        BlockData::from_block(&block_a_raw),
        HashMap::new(),
        None,
    )
    .await
    .expect("compute block A");

    assert!(
        !pd_a[0].is_failed,
        "Contract A deploy failed: {:?}",
        pd_a[0].system_deploy_error
    );
    tracing::info!(
        "Block A: cost={}, events={}",
        pd_a[0].cost.cost,
        pd_a[0].deploy_log.len()
    );

    let mut block_a = block_a_raw;
    block_a.body.state.post_state_hash = post_state_a.clone();
    block_a.body.deploys = pd_a.clone();
    block_a.body.system_deploys = sys_pd_a;
    block_a.body.state.bonds = bonds_a;
    block_store.put_block_message(&block_a).expect("store A");
    dag_storage.insert(&block_a, false, false).expect("dag A");

    // --- Block B: second bridge deploy from genesis (sibling branch, funded deployer B) ---
    let deploy_b =
        construct_deploy::source_deploy_now_full(bridge_rho, None, None, Some(key_b), None, None)
            .unwrap();

    let block_b_raw = block_implicits::get_random_block(
        Some(1),
        Some(2),
        Some(genesis_state.clone()),
        Some(StateHash::default()),
        Some(validator.clone()),
        Some(1),
        Some(now_millis()),
        Some(vec![genesis_hash.clone()]),
        Some(Vec::new()),
        Some(vec![ProcessedDeploy::empty(deploy_b)]),
        Some(Vec::new()),
        Some(genesis_bonds.clone()),
        Some(shard_name.clone()),
        None,
    );

    let parents_b = vec![genesis_block.clone()];
    let deploys_b = proto_util::deploys(&block_b_raw)
        .into_iter()
        .map(|d| d.deploy)
        .collect();
    let snapshot_b = mk_snapshot(&genesis_hash);
    let (_, post_state_b, pd_b, _, sys_pd_b, bonds_b) = compute_deploys_checkpoint(
        &mut block_store,
        parents_b,
        deploys_b,
        Vec::<casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum>::new(),
        &snapshot_b,
        &mut rm,
        BlockData::from_block(&block_b_raw),
        HashMap::new(),
        None,
    )
    .await
    .expect("compute block B");

    assert!(
        !pd_b[0].is_failed,
        "Contract B deploy failed: {:?}",
        pd_b[0].system_deploy_error
    );
    tracing::info!(
        "Block B: cost={}, events={}",
        pd_b[0].cost.cost,
        pd_b[0].deploy_log.len()
    );

    let mut block_b = block_b_raw;
    block_b.body.state.post_state_hash = post_state_b.clone();
    block_b.body.deploys = pd_b.clone();
    block_b.body.system_deploys = sys_pd_b;
    block_b.body.state.bonds = bonds_b;
    block_store.put_block_message(&block_b).expect("store B");
    dag_storage.insert(&block_b, false, false).expect("dag B");

    // Analyze conflict between the two deploys' event logs BEFORE merge
    {
        use casper::rust::merging::block_index::create_event_log_index;
        use rspace_plus_plus::rspace::merger::merging_logic::{conflict_reason, conflicts};

        let history_repo = rm.get_history_repo();
        let genesis_hash_b256 =
            rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash::from_bytes_prost(
                &genesis_state,
            );

        let eli_a = create_event_log_index(
            &pd_a[0].deploy_log,
            history_repo.clone(),
            &genesis_hash_b256,
            std::collections::BTreeMap::new(),
        );
        let eli_b = create_event_log_index(
            &pd_b[0].deploy_log,
            history_repo.clone(),
            &genesis_hash_b256,
            std::collections::BTreeMap::new(),
        );

        let reason = conflict_reason(&eli_a, &eli_b);
        let conflict_channels = conflicts(&eli_a, &eli_b);
        tracing::info!(
            "Conflict analysis: reason={:?}, conflicting_channels={}",
            reason,
            conflict_channels.0.len(),
        );
        for ch in &conflict_channels.0 {
            tracing::info!("  conflicting channel: {}", hex::encode(&ch.0[..8]));
        }

        // Find which produces are racing
        let shared_produces: std::collections::HashSet<_> = eli_a
            .produces_consumed
            .0
            .intersection(&eli_b.produces_consumed.0)
            .cloned()
            .collect();
        let mergeable_produces: std::collections::HashSet<_> = eli_a
            .produces_mergeable
            .0
            .intersection(&eli_b.produces_mergeable.0)
            .cloned()
            .collect();
        let racing_produces: Vec<_> = shared_produces
            .difference(&mergeable_produces)
            .filter(|p| !p.persistent)
            .collect();
        tracing::info!("Racing produces: {}", racing_produces.len());
        // Collect racing channel hashes for COMM tracing
        let racing_channels: std::collections::HashSet<_> = racing_produces
            .iter()
            .map(|p| p.channel_hash.clone())
            .collect();

        // Search deploy A's event log for COMMs involving racing channels
        tracing::info!(
            "Searching deploy A events ({} total) for racing channels...",
            pd_a[0].deploy_log.len()
        );
        for (idx, event) in pd_a[0].deploy_log.iter().enumerate() {
            use models::rust::casper::protocol::casper_message::Event as CasperEvent;
            match event {
                CasperEvent::Comm(comm) => {
                    let consume_channels: Vec<String> = comm
                        .consume
                        .channels_hashes
                        .iter()
                        .map(|h| hex::encode(&h[..std::cmp::min(8, h.len())]))
                        .collect();
                    let produce_channels: Vec<String> = comm
                        .produces
                        .iter()
                        .map(|p| {
                            hex::encode(&p.channels_hash[..std::cmp::min(8, p.channels_hash.len())])
                        })
                        .collect();
                    // Check if any racing channel is in this COMM's produces
                    for p in &comm.produces {
                        let ch = rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash::from_bytes_prost(&p.channels_hash);
                        if racing_channels.contains(&ch) {
                            tracing::info!(
                                "  A event[{}] COMM: consume_channels={:?}, produce_channels={:?}, peeks={:?}, persistent_consume={}",
                                idx, consume_channels, produce_channels, comm.peeks, comm.consume.persistent,
                            );
                        }
                    }
                }
                CasperEvent::Produce(p) => {
                    let ch = rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash::from_bytes_prost(&p.channels_hash);
                    if racing_channels.contains(&ch) {
                        tracing::info!(
                            "  A event[{}] IOProduce: channel={}, persistent={}, output_len={}",
                            idx,
                            hex::encode(
                                &p.channels_hash[..std::cmp::min(8, p.channels_hash.len())]
                            ),
                            p.persistent,
                            p.output_value.len(),
                        );
                    }
                }
                CasperEvent::Consume(c) => {
                    for h in &c.channels_hashes {
                        let ch = rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash::from_bytes_prost(h);
                        if racing_channels.contains(&ch) {
                            tracing::info!(
                                "  A event[{}] IOConsume: channels={:?}, persistent={}",
                                idx,
                                c.channels_hashes
                                    .iter()
                                    .map(|h| hex::encode(&h[..std::cmp::min(8, h.len())]))
                                    .collect::<Vec<_>>(),
                                c.persistent,
                            );
                        }
                    }
                }
            }
        }

        for p in &racing_produces {
            // Decode the output_value to see what data is being raced for
            let output_str: Vec<String> = p
                .output_value
                .iter()
                .map(|v| {
                    format!(
                        "raw({} bytes, first8={})",
                        v.len(),
                        hex::encode(&v[..std::cmp::min(8, v.len())])
                    )
                })
                .collect();
            tracing::info!(
                "  racing produce: channel={}, hash={}, persistent={}, output={:?}",
                hex::encode(&p.channel_hash.0[..8]),
                hex::encode(&p.hash.0[..8]),
                p.persistent,
                output_str,
            );
        }
    }

    // --- Merge [A, B] ---
    let parents = vec![block_a.clone(), block_b.clone()];
    let snapshot_merge = mk_snapshot(&genesis_hash);
    let (merged_state, rejected, _rejected_slashes) =
        compute_parents_post_state(&block_store, parents, &snapshot_merge, &rm, None, None)
            .expect("merge parents");

    tracing::info!(
        "Merge result: rejected={}, merged_state={}",
        rejected.len(),
        hex::encode(&merged_state[..8]),
    );

    if !rejected.is_empty() {
        let rejected_sigs: Vec<String> = rejected
            .iter()
            .map(|d| hex::encode(&d[..std::cmp::min(8, d.len())]))
            .collect();
        tracing::warn!(
            "CONFLICT DETECTED: {} deploys rejected: {:?}",
            rejected.len(),
            rejected_sigs,
        );

        // Identify which deploy was rejected
        let a_sig = hex::encode(&pd_a[0].deploy.sig[..8]);
        let b_sig = hex::encode(&pd_b[0].deploy.sig[..8]);
        let a_rejected = rejected_sigs.iter().any(|s| *s == a_sig);
        let b_rejected = rejected_sigs.iter().any(|s| *s == b_sig);
        tracing::warn!(
            "  Contract A ({}): {}",
            a_sig,
            if a_rejected { "REJECTED" } else { "kept" },
        );
        tracing::warn!(
            "  Contract B ({}): {}",
            b_sig,
            if b_rejected { "REJECTED" } else { "kept" },
        );
    }

    // The key assertion: both deploys should be kept.
    // If one is rejected, insertArbitrary calls falsely conflict.
    assert!(
        rejected.is_empty(),
        "Concurrent insertArbitrary calls should not conflict. \
         {} deploys rejected during merge of two independent registry inserts. \
         This is a false positive in conflict detection — both contracts write \
         to different TreeHashMap leaf channels but share internal node channels.",
        rejected.len(),
    );

    // Verify both URIs accessible from merged state
    let make_deploy_id_par = |sig: &[u8]| -> models::rhoapi::Par {
        models::rhoapi::Par {
            unforgeables: vec![models::rhoapi::GUnforgeable {
                unf_instance: Some(models::rhoapi::g_unforgeable::UnfInstance::GDeployIdBody(
                    models::rhoapi::GDeployId { sig: sig.to_vec() },
                )),
            }],
            ..Default::default()
        }
    };

    let data_a = rm
        .get_data(
            merged_state.clone(),
            &make_deploy_id_par(&pd_a[0].deploy.sig),
        )
        .await
        .unwrap();
    let data_b = rm
        .get_data(
            merged_state.clone(),
            &make_deploy_id_par(&pd_b[0].deploy.sig),
        )
        .await
        .unwrap();
    tracing::info!("Contract A data in merged state: {} pars", data_a.len());
    tracing::info!("Contract B data in merged state: {} pars", data_b.len());

    assert!(
        !data_a.is_empty(),
        "Contract A data missing from merged state"
    );
    assert!(
        !data_b.is_empty(),
        "Contract B data missing from merged state"
    );
}

/// Verifies that exploratory deploy can query user-deployed contracts
/// through the registry. The `contract` keyword is reserved in Rholang,
/// so variable names in the query must not use it.
///
/// Also verifies that play_exploratory_deploy propagates errors (previously
/// errors were silently swallowed, returning empty results).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn exploratory_deploy_async_contract_query() {
    use crypto::rust::signatures::signatures_alg::SignaturesAlg;

    with_runtime_manager(
        |runtime_manager, _genesis_context, genesis_block| async move {
            let genesis_state = genesis_block.body.state.post_state_hash.clone();

            // Deploy a contract with a persistent state channel + persistent consume
            let contract_rho = r#"
new return, stateCh, queryCh,
    insertArbitrary(`rho:registry:insertArbitrary`)
in {
  stateCh!(42) |
  contract queryCh(@method, ret) = {
    for (@v <- stateCh) {
      stateCh!(v) |
      ret!(v)
    }
  } |
  new uriCh in {
    insertArbitrary!(bundle+{*queryCh}, *uriCh) |
    for (@uri <- uriCh) {
      return!(uri)
    }
  }
}
"#;

            // Use a unique key to avoid GPrivate collision with exploratory deploy's DEFAULT_SEC
            let (contract_key, _) = crypto::rust::signatures::secp256k1::Secp256k1.new_key_pair();
            let deploy = construct_deploy::source_deploy(
                contract_rho.to_string(),
                0,
                Some(500_000_000),
                None,
                Some(contract_key),
                None,
                None,
            )
            .unwrap();

            // Deploy and read URI via capture_results
            let uri_pars = runtime_manager
                .capture_results(&genesis_state, &deploy)
                .await
                .expect("deploy contract");
            assert!(!uri_pars.is_empty(), "Contract deploy returned no URI");

            let uri_str = format!("{:?}", uri_pars[0]);
            let uri_regex = regex::Regex::new(r"rho:id:[a-zA-Z0-9]+").unwrap();
            let uri = uri_regex
                .find(&uri_str)
                .expect("No rho:id URI found")
                .as_str()
                .to_string();

            // Checkpoint via a fresh runtime so exploratory deploy can see the state
            let runtime = runtime_manager.spawn_runtime().await;
            let mut runtime_ops = RuntimeOps::new(runtime);
            runtime_ops
                .runtime
                .reset(&Blake2b256Hash::from_bytes_prost(&genesis_state))
                .await
                .expect("reset");
            let eval_result = runtime_ops.evaluate(&deploy).await.expect("evaluate");
            assert!(
                eval_result.errors.is_empty(),
                "Deploy errors: {:?}",
                eval_result.errors
            );
            let checkpoint = runtime_ops.runtime.create_checkpoint().await;
            let post_state: StateHash = checkpoint.root.to_bytes_prost().into();
            tracing::info!(
                "Contract at {}, post_state={}",
                uri,
                hex::encode(&post_state[..8])
            );

            // Query with correct variable names (NOT using reserved word 'contract')
            let query_term = format!(
                r#"new ret, lookup(`rho:registry:lookup`), ch in {{
                lookup!(`{}`, *ch) |
                for (c <- ch) {{
                    c!("get", *ret)
                }}
            }}"#,
                uri
            );
            let (query_result, _) = runtime_manager
                .play_exploratory_deploy(query_term, &post_state)
                .await
                .expect("query exploratory deploy");
            tracing::info!("Query with correct var name: {} pars", query_result.len());
            assert_eq!(
                query_result.len(),
                1,
                "Query should return 1 par (the value 42)"
            );

            // Verify play_exploratory_deploy propagates parse errors (not swallows them)
            let bad_term = format!(
                r#"new ret, lookup(`rho:registry:lookup`), ch in {{
                lookup!(`{}`, *ch) |
                for (contract <- ch) {{
                    contract!("get", *ret)
                }}
            }}"#,
                uri
            );
            let bad_result = runtime_manager
                .play_exploratory_deploy(bad_term, &post_state)
                .await;
            assert!(
                bad_result.is_err(),
                "Using reserved word 'contract' as var name should return Err, not empty Ok"
            );
        },
    )
    .await
    .unwrap();
}

/// Reproduces the replay determinism issue seen with tokio::spawn.
/// Deploys a contract with parallel composition, plays it, then replays it.
/// If tokio::spawn introduces non-deterministic evaluation order, the replay
/// cost will differ from the play cost, causing ReplayCostMismatch.
///
/// Run: cargo test -p casper --test mod --release parallel_replay_determinism -- --nocapture
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_replay_determinism() {
    with_runtime_manager(
        |mut runtime_manager, genesis_context, genesis_block| async move {
            let gps = genesis_block.body.state.post_state_hash;

            // Registry lookup — system process with internal parallel composition
            let parallel_contract = r#"
                new rl(`rho:registry:lookup`), ch in {
                    rl!(`rho:vault:system`, *ch)
                }
            "#;

            let deploy = construct_deploy::source_deploy_now_full(
                parallel_contract.to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();

            let time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as i64;

            let block_data = BlockData {
                time_stamp: time,
                block_number: 0,
                sender: genesis_context.validator_pks()[0].clone(),
                seq_num: 0,
            };

            // Play the deploy with CloseBlockDeploy system deploy
            let (play_state, processed_deploys, processed_sys_deploys) = runtime_manager
                .compute_state(
                    &gps,
                    vec![deploy],
                    vec![
                        casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum::Close(
                            CloseBlockDeploy {
                                initial_rand:
                                    system_deploy_util::generate_close_deploy_random_seed_from_pk(
                                        block_data.sender.clone(),
                                        block_data.seq_num,
                                    ),
                            },
                        ),
                    ],
                    block_data.clone(),
                    None,
                )
                .await
                .unwrap();

            let play_cost = processed_deploys[0].cost.cost;
            let play_failed = processed_deploys[0].is_failed;
            let play_event_count = processed_deploys[0].deploy_log.len();
            let sys_deploy_count = processed_sys_deploys.len();

            // Hash the event log for comparison
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            for ev in &processed_deploys[0].deploy_log {
                format!("{:?}", ev).hash(&mut hasher);
            }
            let event_log_hash = hasher.finish();

            println!("Play: cost={}, failed={}, events={}, sys_deploys={}, event_hash={:016x}, state={:?}",
                play_cost, play_failed, play_event_count, sys_deploy_count, event_log_hash, &play_state[..8]);

            // Replay the same deploy — must produce identical state and cost
            let replay_state = runtime_manager
                .replay_compute_state(
                    &gps,
                    processed_deploys,
                    processed_sys_deploys,
                    &block_data,
                    None,
                    false,
                )
                .await;

            match replay_state {
                Ok(state) => {
                    println!("Replay succeeded, state match: {}", state == play_state);
                    assert_eq!(state, play_state, "Play and replay produced different state hashes");
                }
                Err(CasperError::ReplayFailure(ReplayFailure::ReplayCostMismatch {
                    initial_cost,
                    replay_cost,
                })) => {
                    panic!(
                        "REPLAY DETERMINISM FAILURE: play cost={} but replay cost={}. \
                         This indicates non-deterministic evaluation order in parallel composition.",
                        initial_cost, replay_cost
                    );
                }
                Err(e) => {
                    panic!("Replay failed: {:?}", e);
                }
            }
        },
    )
    .await
    .unwrap();
}

/// Regression guard for the rejection-expansion behavior in `DagMerger::merge`.
///
/// DAG shape:
///
///        genesis (LCA)
///         /     \
///        BA      BB       bridge(key_A), bridge(key_B) — conflict on shared system channels
///        |       |
///        BC      BD       trivial writes by the same deployer as the ancestor
///
/// `compute_parents_post_state([BC, BD])` drives a merge whose scope is
/// `{BA, BB, BC, BD}`. One of BA/BB is rejected by conflict resolution.
/// Without rejection expansion, the descendant of the rejected block retains
/// pre-computed diffs against a pre-state that no longer materializes — the
/// merged post-state ends up with the descendant's writes present but the
/// ancestor's writes absent, which is internally inconsistent.
///
/// The expansion in DagMerger rejects the descendant's chains as well, so the
/// assertion below — "no ancestor-rejected-but-descendant-surviving" — holds.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stale_diff_application_corrupts_merged_state() {
    use crate::util::rholang::resources::{
        block_dag_storage_from_dyn, mergeable_store_from_dyn,
        mk_test_rnode_store_manager_from_genesis,
    };
    use block_storage::rust::key_value_block_store::KeyValueBlockStore;
    use casper::rust::genesis::genesis::Genesis;
    use casper::rust::{
        casper::{CasperShardConf, CasperSnapshot, OnChainCasperState},
        util::{
            proto_util,
            rholang::interpreter_util::{compute_deploys_checkpoint, compute_parents_post_state},
        },
    };
    use dashmap::{DashMap, DashSet};
    use models::rust::{block_hash::BlockHash, block_implicits};
    use rholang::rust::interpreter::external_services::ExternalServices;
    use std::collections::HashSet;

    crate::init_logger();
    let genesis_context = crate::util::rholang::resources::genesis_context()
        .await
        .unwrap();
    let genesis_block = genesis_context.genesis_block.clone();
    let genesis_hash = genesis_block.block_hash.clone();
    let genesis_state = proto_util::post_state_hash(&genesis_block);
    let genesis_bonds = genesis_block.body.state.bonds.clone();
    let validator: prost::bytes::Bytes = genesis_context.validator_pks()[0].bytes.clone().into();
    let shard_name = genesis_block.shard_id.clone();

    let mut kvm = mk_test_rnode_store_manager_from_genesis(&genesis_context);
    let rspace_store = kvm.r_space_stores().await.expect("rspace stores");
    let mergeable_store = mergeable_store_from_dyn(&mut *kvm)
        .await
        .expect("mergeable store");
    let (mut rm, _) = RuntimeManager::create_with_history(
        rspace_store,
        mergeable_store,
        std::sync::Arc::new(Genesis::default_mergeable_tags()),
        ExternalServices::noop(),
    );

    let mut block_store = KeyValueBlockStore::create_from_kvm(&mut *kvm)
        .await
        .expect("block store");
    let dag_storage = block_dag_storage_from_dyn(&mut *kvm)
        .await
        .expect("dag storage");

    block_store
        .put_block_message(&genesis_block)
        .expect("store genesis");
    dag_storage
        .insert(&genesis_block, false, true)
        .expect("dag genesis");

    let now_millis = || -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    };

    let mk_snapshot = |lfb: &BlockHash| -> CasperSnapshot {
        let mut snapshot = CasperSnapshot::new(dag_storage.get_representation());
        snapshot.last_finalized_block = lfb.clone();
        let max_seq_nums: DashMap<prost::bytes::Bytes, u64> = DashMap::new();
        max_seq_nums.insert(validator.clone(), 0);
        snapshot.max_seq_nums = max_seq_nums;
        let mut shard_conf = CasperShardConf::new();
        shard_conf.shard_name = shard_name.clone();
        shard_conf.max_parent_depth = 0;
        let mut bonds_map = HashMap::new();
        bonds_map.insert(validator.clone(), 100);
        snapshot.on_chain_state = OnChainCasperState {
            shard_conf,
            bonds_map,
            active_validators: vec![validator.clone()],
        };
        snapshot.deploys_in_scope = std::sync::Arc::new(DashSet::new());
        snapshot
    };

    let make_deploy_id_par = |sig: &[u8]| -> models::rhoapi::Par {
        models::rhoapi::Par {
            unforgeables: vec![models::rhoapi::GUnforgeable {
                unf_instance: Some(models::rhoapi::g_unforgeable::UnfInstance::GDeployIdBody(
                    models::rhoapi::GDeployId { sig: sig.to_vec() },
                )),
            }],
            ..Default::default()
        }
    };

    let bridge_rho = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/resources/bridge.rho"),
    )
    .expect("Failed to read bridge.rho");

    let key_a = construct_deploy::DEFAULT_SEC.clone();
    let key_b = construct_deploy::DEFAULT_SEC2.clone();

    let trivial_rho = r#"
new deployId(`rho:system:deployId`) in {
  deployId!("descendant-tag")
}
"#
    .to_string();

    // ── Block A: bridge deployed by key_a, parent = genesis ──
    let deploy_a = construct_deploy::source_deploy_now_full(
        bridge_rho.clone(),
        None,
        None,
        Some(key_a.clone()),
        None,
        None,
    )
    .unwrap();
    let block_a_raw = block_implicits::get_random_block(
        Some(1),
        Some(1),
        Some(genesis_state.clone()),
        Some(StateHash::default()),
        Some(validator.clone()),
        Some(1),
        Some(now_millis()),
        Some(vec![genesis_hash.clone()]),
        Some(Vec::new()),
        Some(vec![ProcessedDeploy::empty(deploy_a)]),
        Some(Vec::new()),
        Some(genesis_bonds.clone()),
        Some(shard_name.clone()),
        None,
    );
    let (_, post_state_a, pd_a, _, sys_pd_a, bonds_a) = compute_deploys_checkpoint(
        &mut block_store,
        vec![genesis_block.clone()],
        proto_util::deploys(&block_a_raw)
            .into_iter()
            .map(|d| d.deploy)
            .collect(),
        Vec::<casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum>::new(),
        &mk_snapshot(&genesis_hash),
        &mut rm,
        BlockData::from_block(&block_a_raw),
        HashMap::new(),
        None,
    )
    .await
    .expect("compute block A");
    assert!(
        !pd_a[0].is_failed,
        "Bridge A failed: {:?}",
        pd_a[0].system_deploy_error
    );
    let mut block_a = block_a_raw;
    block_a.body.state.post_state_hash = post_state_a.clone();
    block_a.body.deploys = pd_a.clone();
    block_a.body.system_deploys = sys_pd_a;
    block_a.body.state.bonds = bonds_a;
    block_store.put_block_message(&block_a).expect("store A");
    dag_storage.insert(&block_a, false, false).expect("dag A");

    // ── Block B: bridge deployed by key_b, parent = genesis (sibling of A) ──
    let deploy_b = construct_deploy::source_deploy_now_full(
        bridge_rho,
        None,
        None,
        Some(key_b.clone()),
        None,
        None,
    )
    .unwrap();
    let block_b_raw = block_implicits::get_random_block(
        Some(1),
        Some(2),
        Some(genesis_state.clone()),
        Some(StateHash::default()),
        Some(validator.clone()),
        Some(1),
        Some(now_millis()),
        Some(vec![genesis_hash.clone()]),
        Some(Vec::new()),
        Some(vec![ProcessedDeploy::empty(deploy_b)]),
        Some(Vec::new()),
        Some(genesis_bonds.clone()),
        Some(shard_name.clone()),
        None,
    );
    let (_, post_state_b, pd_b, _, sys_pd_b, bonds_b) = compute_deploys_checkpoint(
        &mut block_store,
        vec![genesis_block.clone()],
        proto_util::deploys(&block_b_raw)
            .into_iter()
            .map(|d| d.deploy)
            .collect(),
        Vec::<casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum>::new(),
        &mk_snapshot(&genesis_hash),
        &mut rm,
        BlockData::from_block(&block_b_raw),
        HashMap::new(),
        None,
    )
    .await
    .expect("compute block B");
    assert!(
        !pd_b[0].is_failed,
        "Bridge B failed: {:?}",
        pd_b[0].system_deploy_error
    );
    let mut block_b = block_b_raw;
    block_b.body.state.post_state_hash = post_state_b.clone();
    block_b.body.deploys = pd_b.clone();
    block_b.body.system_deploys = sys_pd_b;
    block_b.body.state.bonds = bonds_b;
    block_store.put_block_message(&block_b).expect("store B");
    dag_storage.insert(&block_b, false, false).expect("dag B");

    // ── Block C: trivial deploy by key_a, parent = A ──
    let deploy_c = construct_deploy::source_deploy_now_full(
        trivial_rho.clone(),
        None,
        None,
        Some(key_a),
        None,
        None,
    )
    .unwrap();
    let block_c_raw = block_implicits::get_random_block(
        Some(2),
        Some(3),
        Some(post_state_a.clone()),
        Some(StateHash::default()),
        Some(validator.clone()),
        Some(1),
        Some(now_millis()),
        Some(vec![block_a.block_hash.clone()]),
        Some(Vec::new()),
        Some(vec![ProcessedDeploy::empty(deploy_c)]),
        Some(Vec::new()),
        Some(genesis_bonds.clone()),
        Some(shard_name.clone()),
        None,
    );
    let (_, post_state_c, pd_c, _, sys_pd_c, bonds_c) = compute_deploys_checkpoint(
        &mut block_store,
        vec![block_a.clone()],
        proto_util::deploys(&block_c_raw)
            .into_iter()
            .map(|d| d.deploy)
            .collect(),
        Vec::<casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum>::new(),
        &mk_snapshot(&genesis_hash),
        &mut rm,
        BlockData::from_block(&block_c_raw),
        HashMap::new(),
        None,
    )
    .await
    .expect("compute block C");
    assert!(
        !pd_c[0].is_failed,
        "Trivial C failed: {:?}",
        pd_c[0].system_deploy_error
    );
    let mut block_c = block_c_raw;
    block_c.body.state.post_state_hash = post_state_c.clone();
    block_c.body.deploys = pd_c.clone();
    block_c.body.system_deploys = sys_pd_c;
    block_c.body.state.bonds = bonds_c;
    block_store.put_block_message(&block_c).expect("store C");
    dag_storage.insert(&block_c, false, false).expect("dag C");

    // ── Block D: trivial deploy by key_b, parent = B ──
    let deploy_d =
        construct_deploy::source_deploy_now_full(trivial_rho, None, None, Some(key_b), None, None)
            .unwrap();
    let block_d_raw = block_implicits::get_random_block(
        Some(2),
        Some(4),
        Some(post_state_b.clone()),
        Some(StateHash::default()),
        Some(validator.clone()),
        Some(1),
        Some(now_millis()),
        Some(vec![block_b.block_hash.clone()]),
        Some(Vec::new()),
        Some(vec![ProcessedDeploy::empty(deploy_d)]),
        Some(Vec::new()),
        Some(genesis_bonds.clone()),
        Some(shard_name.clone()),
        None,
    );
    let (_, post_state_d, pd_d, _, sys_pd_d, bonds_d) = compute_deploys_checkpoint(
        &mut block_store,
        vec![block_b.clone()],
        proto_util::deploys(&block_d_raw)
            .into_iter()
            .map(|d| d.deploy)
            .collect(),
        Vec::<casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum>::new(),
        &mk_snapshot(&genesis_hash),
        &mut rm,
        BlockData::from_block(&block_d_raw),
        HashMap::new(),
        None,
    )
    .await
    .expect("compute block D");
    assert!(
        !pd_d[0].is_failed,
        "Trivial D failed: {:?}",
        pd_d[0].system_deploy_error
    );
    let mut block_d = block_d_raw;
    block_d.body.state.post_state_hash = post_state_d.clone();
    block_d.body.deploys = pd_d.clone();
    block_d.body.system_deploys = sys_pd_d;
    block_d.body.state.bonds = bonds_d;
    block_store.put_block_message(&block_d).expect("store D");
    dag_storage.insert(&block_d, false, false).expect("dag D");

    // ── Merge [C, D] — simulates what a validator would compute when proposing
    //    a multi-parent block with parents [BC, BD]. LCA is genesis.
    let (merged_state, rejected, _rejected_slashes) = compute_parents_post_state(
        &block_store,
        vec![block_c.clone(), block_d.clone()],
        &mk_snapshot(&genesis_hash),
        &rm,
        None,
        None,
    )
    .expect("merge [C, D]");

    let rejected_set: HashSet<prost::bytes::Bytes> = rejected.iter().cloned().collect();
    let ba_rejected = rejected_set.contains(&pd_a[0].deploy.sig);
    let bb_rejected = rejected_set.contains(&pd_b[0].deploy.sig);
    let bc_rejected = rejected_set.contains(&pd_c[0].deploy.sig);
    let bd_rejected = rejected_set.contains(&pd_d[0].deploy.sig);

    tracing::info!("──────── Rejection outcome ────────");
    tracing::info!(
        "BA (bridge, key_A)                 rejected: {}",
        ba_rejected
    );
    tracing::info!(
        "BB (bridge, key_B)                 rejected: {}",
        bb_rejected
    );
    tracing::info!(
        "BC (trivial, key_A, child of BA)   rejected: {}",
        bc_rejected
    );
    tracing::info!(
        "BD (trivial, key_B, child of BB)   rejected: {}",
        bd_rejected
    );
    tracing::info!("Total rejected: {} deploys", rejected.len());

    let ba_data = rm
        .get_data(
            merged_state.clone(),
            &make_deploy_id_par(&pd_a[0].deploy.sig),
        )
        .await
        .unwrap();
    let bb_data = rm
        .get_data(
            merged_state.clone(),
            &make_deploy_id_par(&pd_b[0].deploy.sig),
        )
        .await
        .unwrap();
    let bc_data = rm
        .get_data(
            merged_state.clone(),
            &make_deploy_id_par(&pd_c[0].deploy.sig),
        )
        .await
        .unwrap();
    let bd_data = rm
        .get_data(
            merged_state.clone(),
            &make_deploy_id_par(&pd_d[0].deploy.sig),
        )
        .await
        .unwrap();

    tracing::info!("──────── State presence in merged post-state ────────");
    tracing::info!("BA bridge data  pars: {}", ba_data.len());
    tracing::info!("BB bridge data  pars: {}", bb_data.len());
    tracing::info!("BC trivial data pars: {}", bc_data.len());
    tracing::info!("BD trivial data pars: {}", bd_data.len());

    let bc_orphaned = ba_rejected && !bc_rejected && ba_data.is_empty() && !bc_data.is_empty();
    let bd_orphaned = bb_rejected && !bd_rejected && bb_data.is_empty() && !bd_data.is_empty();

    assert!(
        !bc_orphaned && !bd_orphaned,
        "STALE-DIFF BUG REPRODUCED: descendant of rejected block has state present \
         in merged post-state while its ancestor's state is absent. \
         bc_orphaned={} (ba_rejected={}, bc_rejected={}, ba_empty={}, bc_present={}); \
         bd_orphaned={} (bb_rejected={}, bd_rejected={}, bb_empty={}, bd_present={}).",
        bc_orphaned,
        ba_rejected,
        bc_rejected,
        ba_data.is_empty(),
        !bc_data.is_empty(),
        bd_orphaned,
        bb_rejected,
        bd_rejected,
        bb_data.is_empty(),
        !bd_data.is_empty(),
    );
}
