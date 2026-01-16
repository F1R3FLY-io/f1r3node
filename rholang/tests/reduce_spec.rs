// See rholang/src/test/scala/coop/rchain/rholang/interpreter/ReduceSpec.scala

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    i64,
    sync::Arc,
};

use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::{
    rhoapi::{
        connective::ConnectiveInstance::VarRefBody, expr::ExprInstance, g_unforgeable::UnfInstance,
        Bundle, Connective, EEq, EList, EMatches, EMethod, EMinus, EMinusMinus, EPercentPercent,
        EPlus, EPlusPlus, ETuple, GPrivate, GUnforgeable, Match, MatchCase, New, Receive,
        ReceiveBind, VarRef,
    },
    rust::{
        par_map::ParMap,
        par_map_type_mapper::ParMapTypeMapper,
        par_set::ParSet,
        par_set_type_mapper::ParSetTypeMapper,
        rholang::implicits::GPrivateBuilder,
        string_ops::StringOps,
        utils::{
            new_boundvar_par, new_bundle_par, new_elist_expr, new_elist_par, new_etuple_par,
            new_freevar_par, new_freevar_var, new_gbool_expr, new_gbool_par, new_gstring_expr,
            new_gstring_par, new_wildcard_par,
        },
    },
};
use models::{
    rhoapi::{
        tagged_continuation::TaggedCont, BindPattern, Expr, GDeployId, ListParWithRandom, Par,
        ParWithRandom, Send, TaggedContinuation,
    },
    rust::utils::{new_eplus_par_gint, new_gint_expr, new_gint_par},
};
use prost::Message;
use rholang::rust::interpreter::{
    accounting::{cost_accounting::CostAccounting, costs::Cost},
    env::Env,
    errors::InterpreterError,
    matcher::r#match::Matcher,
    reduce::DebruijnInterpreter,
    rho_runtime::RhoISpace,
    spaces::SpaceQualifier,
    test_utils::persistent_store_tester::create_test_space,
};
use rspace_plus_plus::rspace::{
    internal::{Datum, Row, WaitingContinuation},
    rspace::RSpace,
    rspace_interface::ISpace,
    shared::{
        in_mem_store_manager::InMemoryStoreManager, key_value_store_manager::KeyValueStoreManager,
    },
};

fn rand() -> Blake2b512Random {
    Blake2b512Random::create_from_bytes(&Vec::new())
}

struct DataMapEntry {
    data: Vec<Par>,
    rand: Blake2b512Random,
}

fn map_data(
    elements: HashMap<Par, (Vec<Par>, Blake2b512Random)>,
) -> HashMap<Vec<Par>, Row<BindPattern, ListParWithRandom, TaggedContinuation>> {
    let mapped_entries = elements.into_iter().map(|(channel, (data, rand))| {
        let entry = DataMapEntry { data, rand };

        let row = Row {
            data: vec![Datum::create(
                &channel,
                ListParWithRandom {
                    pars: entry.data.clone(),
                    random_state: entry.rand.to_bytes(),
                },
                false,
            )],
            wks: vec![],
        };

        (vec![channel], row)
    });

    mapped_entries.collect()
}

fn check_continuation(
    result: HashMap<Vec<Par>, Row<BindPattern, ListParWithRandom, TaggedContinuation>>,
    channels: Vec<Par>,
    bind_patterns: Vec<BindPattern>,
    body: ParWithRandom,
) -> bool {
    let mut expected_result = HashMap::new();
    expected_result.insert(
        channels.clone(),
        Row {
            data: Vec::new(),
            wks: vec![WaitingContinuation::create(
                &channels,
                &bind_patterns,
                &TaggedContinuation {
                    tagged_cont: Some(TaggedCont::ParBody(body)),
                },
                false,
                BTreeSet::new(),
            )],
        },
    );

    if result.len() != expected_result.len() {
        return false;
    }

    for (key, value) in &result {
        if let Some(expected_value) = expected_result.get(key) {
            if value != expected_value {
                return false;
            }
        } else {
            return false;
        }
    }

    true
}

#[tokio::test]
async fn eval_expr_should_handle_simple_addition() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;
    let add_expr = new_eplus_par_gint(7, 8, Vec::new(), false);
    let env: Env<Par> = Env::new();
    let result = reducer.eval_expr(&add_expr, &env);
    let expected = vec![new_gint_expr(15)];

    assert!(result.is_ok());
    assert_eq!(result.unwrap().exprs, expected);
}

#[tokio::test]
async fn eval_expr_should_handle_long_addition() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;
    let add_expr = new_eplus_par_gint(i64::MAX, i64::MAX, Vec::new(), false);
    let env: Env<Par> = Env::new();
    let result = reducer.eval_expr(&add_expr, &env);
    let expected = vec![new_gint_expr(i64::MAX.wrapping_mul(2))];

    assert!(result.is_ok());
    assert_eq!(result.unwrap().exprs, expected);
}

#[tokio::test]
async fn eval_expr_should_leave_ground_values_alone() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;
    let ground_expr = new_gint_par(7, Vec::new(), false);
    let env: Env<Par> = Env::new();
    let result = reducer.eval_expr(&ground_expr, &env);
    let expected = vec![new_gint_expr(7)];

    assert!(result.is_ok());
    assert_eq!(result.unwrap().exprs, expected);
}

#[tokio::test]
async fn eval_expr_should_handle_equality_between_arbitary_processes() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;
    let eq_expr = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EEqBody(EEq {
            p1: Some(GPrivateBuilder::new_par_from_string(String::from(
                "private_name",
            ))),
            p2: Some(GPrivateBuilder::new_par_from_string(String::from(
                "private_name",
            ))),
        })),
    }]);
    let env: Env<Par> = Env::new();
    let result = reducer.eval_expr(&eq_expr, &env);
    let expected = vec![new_gbool_expr(true)];

    assert!(result.is_ok());
    assert_eq!(result.unwrap().exprs, expected);
}

#[tokio::test]
async fn eval_expr_should_substitute_before_comparison() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let eq_expr = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EEqBody(EEq {
            p1: Some(new_boundvar_par(0, Vec::new(), false)),
            p2: Some(new_boundvar_par(1, Vec::new(), false)),
        })),
    }]);

    let mut env: Env<Par> = Env::new();
    env = env.put(Par::default());
    env = env.put(Par::default());

    let result = reducer.eval_expr(&eq_expr, &env);
    let expected = vec![new_gbool_expr(true)];

    assert!(result.is_ok());
    assert_eq!(result.unwrap().exprs, expected);
}

#[tokio::test]
async fn eval_of_bundle_should_evaluate_contents_of_bundle() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let split_rand = rand().split_byte(0);
    let channel = new_gstring_par(String::from("channel"), Vec::new(), false);
    let bundle_send = Par::default().with_bundles(vec![Bundle {
        body: Some(Par::default().with_sends(vec![Send {
            chan: Some(channel.clone()),
            data: vec![
                new_gint_par(7, Vec::new(), false),
                new_gint_par(8, Vec::new(), false),
                new_gint_par(9, Vec::new(), false),
            ],
            persistent: false,
            locally_free: Vec::new(),
            connective_used: false,
            hyperparams: vec![],
        }])),
        write_flag: false,
        read_flag: false,
    }]);

    let env: Env<Par> = Env::new();
    let result = reducer.eval(bundle_send, &env, split_rand.clone()).await;
    assert!(result.is_ok());
    let mut expected_elements = HashMap::new();
    expected_elements.insert(
        channel,
        (
            vec![
                new_gint_par(7, Vec::new(), false),
                new_gint_par(8, Vec::new(), false),
                new_gint_par(9, Vec::new(), false),
            ],
            split_rand,
        ),
    );
    let expected_result = map_data(expected_elements);
    assert_eq!(space.to_map(), expected_result);
}

#[tokio::test]
async fn eval_of_bundle_should_throw_an_error_if_names_are_used_against_their_polarity_1() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    /* for (n <- @bundle+ { y } ) { }  -> for (n <- y) { }
     */
    let y = new_gstring_par(String::from("y"), Vec::new(), false);
    let receive = Par::default().with_receives(vec![Receive {
        binds: vec![ReceiveBind {
            patterns: vec![Par::default()],
            source: Some(new_bundle_par(y, true, false)),
            remainder: None,
            free_count: 0,
            pattern_modifiers: vec![],
        }],
        body: Some(Par::default()),
        persistent: false,
        peek: false,
        bind_count: 0,
        locally_free: Vec::new(),
        connective_used: false,
    }]);

    let env: Env<Par> = Env::new();
    let result = reducer
        .eval(receive, &env, rand())
        .await
        .map(|_| space.to_map());
    assert!(result.is_err());
    if let Err(e) = result {
        assert_eq!(
            e,
            InterpreterError::ReduceError(String::from(
                "Trying to read from non-readable channel."
            ))
        );
    }
}

#[tokio::test]
async fn eval_of_bundle_should_throw_an_error_if_names_are_used_against_their_polarity_2() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    /* @bundle- { x } !(7) -> x!(7)
     */
    let x = new_gstring_par(String::from("channel"), Vec::new(), false);
    let send = Par::default().with_sends(vec![Send {
        chan: Some(new_bundle_par(x, false, true)),
        data: vec![new_gint_par(7, Vec::new(), false)],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);

    let env: Env<Par> = Env::new();
    let result = reducer
        .eval(send, &env, rand())
        .await
        .map(|_| space.to_map());
    assert!(result.is_err());
    if let Err(e) = result {
        assert_eq!(
            e,
            InterpreterError::ReduceError(String::from("Trying to send on non-writeable channel."))
        );
    }
}

#[tokio::test]
async fn eval_of_send_should_place_something_in_the_tuplespace() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let channel = new_gstring_par(String::from("channel"), Vec::new(), false);
    let split_rand = rand().split_byte(0);
    let send = Par::default().with_sends(vec![Send {
        chan: Some(channel.clone()),
        data: vec![
            new_gint_par(7, Vec::new(), false),
            new_gint_par(8, Vec::new(), false),
            new_gint_par(9, Vec::new(), false),
        ],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);

    let env: Env<Par> = Env::new();
    let result = reducer
        .eval(send, &env, split_rand.clone())
        .await
        .map(|_| space.to_map());

    assert!(result.is_ok());
    let mut expected_elements = HashMap::new();
    expected_elements.insert(
        channel,
        (
            vec![
                new_gint_par(7, Vec::new(), false),
                new_gint_par(8, Vec::new(), false),
                new_gint_par(9, Vec::new(), false),
            ],
            split_rand,
        ),
    );
    let expected_result = map_data(expected_elements);
    assert_eq!(result.unwrap(), expected_result);
}

#[tokio::test]
async fn eval_of_send_should_verify_that_bundle_is_writeable_before_sending_on_bundle() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let split_rand = rand().split_byte(0);

    /* @bundle+ { x } !(7) -> x!(7)
     */
    let channel = new_gstring_par(String::from("channel"), Vec::new(), false);
    let send = Par::default().with_sends(vec![Send {
        chan: Some(new_bundle_par(channel.clone(), true, false)),
        data: vec![new_gint_par(7, Vec::new(), false)],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);

    let env: Env<Par> = Env::new();
    let result = reducer
        .eval(send, &env, split_rand.clone())
        .await
        .map(|_| space.to_map());

    assert!(result.is_ok());
    let mut expected_elements = HashMap::new();
    expected_elements.insert(
        channel,
        (vec![new_gint_par(7, Vec::new(), false)], split_rand),
    );
    let expected_result = map_data(expected_elements);
    assert_eq!(result.unwrap(), expected_result);
}

#[tokio::test]
async fn eval_of_single_channel_receive_should_place_something_in_the_tuplespace() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let split_rand = rand().split_byte(0);
    let channel = new_gstring_par(String::from("channel"), Vec::new(), false);
    let receive = Par::default().with_receives(vec![Receive {
        binds: vec![ReceiveBind {
            patterns: vec![
                new_freevar_par(0, Vec::new()),
                new_freevar_par(1, Vec::new()),
                new_freevar_par(2, Vec::new()),
            ],
            source: Some(channel.clone()),
            remainder: None,
            free_count: 0,
            pattern_modifiers: vec![],
        }],
        body: Some(Par::default()),
        persistent: false,
        peek: false,
        bind_count: 3,
        locally_free: Vec::new(),
        connective_used: false,
    }]);

    let env: Env<Par> = Env::new();
    let result = reducer
        .eval(receive, &env, split_rand.clone())
        .await
        .map(|_| space.to_map());

    assert!(result.is_ok());
    let bind_pattern = BindPattern {
        patterns: vec![
            new_freevar_par(0, Vec::new()),
            new_freevar_par(1, Vec::new()),
            new_freevar_par(2, Vec::new()),
        ],
        remainder: None,
        free_count: 0,
    };

    assert!(check_continuation(
        result.unwrap(),
        vec![channel],
        vec![bind_pattern],
        ParWithRandom {
            body: Some(Par::default()),
            random_state: split_rand.to_bytes()
        }
    ))
}

#[tokio::test]
async fn eval_of_single_channel_receive_should_verify_that_bundle_is_readable_if_receiving_on_bundle(
) {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let split_rand = rand().split_byte(1);

    /* for (@Nil <- @bundle- { y } ) { }  -> for (n <- y) { }
     */
    let y = new_gstring_par(String::from("y"), Vec::new(), false);
    let receive = Par::default().with_receives(vec![Receive {
        binds: vec![ReceiveBind {
            patterns: vec![Par::default()],
            source: Some(new_bundle_par(y.clone(), false, true)),
            remainder: None,
            free_count: 0,
            pattern_modifiers: vec![],
        }],
        body: Some(Par::default()),
        persistent: false,
        peek: false,
        bind_count: 0,
        locally_free: Vec::new(),
        connective_used: false,
    }]);

    let env: Env<Par> = Env::new();
    let result = reducer
        .eval(receive, &env, split_rand.clone())
        .await
        .map(|_| space.to_map());

    assert!(result.is_ok());
    assert!(check_continuation(
        result.unwrap(),
        vec![y],
        vec![BindPattern {
            patterns: vec![Par::default()],
            remainder: None,
            free_count: 0
        }],
        ParWithRandom {
            body: Some(Par::default()),
            random_state: split_rand.to_bytes()
        }
    ))
}

#[tokio::test]
async fn eval_of_send_pipe_receive_should_meet_in_the_tuple_space_and_proceed() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let split_rand0 = rand().split_byte(0);
    let split_rand1 = rand().split_byte(1);
    let merge_rand = Blake2b512Random::merge(vec![split_rand1.clone(), split_rand0.clone()]);
    let send = Par::default().with_sends(vec![Send {
        chan: Some(new_gstring_par(String::from("channel"), Vec::new(), false)),
        data: vec![
            new_gint_par(7, Vec::new(), false),
            new_gint_par(8, Vec::new(), false),
            new_gint_par(9, Vec::new(), false),
        ],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);

    let receive = Par::default().with_receives(vec![Receive {
        binds: vec![ReceiveBind {
            patterns: vec![
                new_freevar_par(0, Vec::new()),
                new_freevar_par(1, Vec::new()),
                new_freevar_par(2, Vec::new()),
            ],
            source: Some(new_gstring_par("channel".to_string(), Vec::new(), false)),
            remainder: None,
            free_count: 3,
            pattern_modifiers: vec![],
        }],
        body: Some(Par::default().with_sends(vec![Send {
            chan: Some(new_gstring_par("result".to_string(), Vec::new(), false)),
            data: vec![new_gstring_par("Success".to_string(), Vec::new(), false)],
            persistent: false,
            locally_free: Vec::new(),
            connective_used: false,
            hyperparams: vec![],
        }])),
        persistent: false,
        peek: false,
        bind_count: 3,
        locally_free: Vec::new(),
        connective_used: false,
    }]);

    let env: Env<Par> = Env::new();
    assert!(reducer
        .eval(send.clone(), &env, split_rand0.clone())
        .await
        .is_ok());
    assert!(reducer
        .eval(receive.clone(), &env, split_rand1.clone())
        .await
        .is_ok());

    let send_result = space.to_map();
    let mut expected_elements = HashMap::new();
    expected_elements.insert(
        new_gstring_par(String::from("result"), Vec::new(), false),
        (
            vec![new_gstring_par("Success".to_string(), Vec::new(), false)],
            merge_rand,
        ),
    );
    assert_eq!(send_result, map_data(expected_elements.clone()));

    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;
    assert!(reducer
        .eval(receive, &env, split_rand1.clone())
        .await
        .is_ok());
    assert!(reducer.eval(send, &env, split_rand0.clone()).await.is_ok());

    let receive_result = space.to_map();
    assert_eq!(receive_result, map_data(expected_elements));
}

#[tokio::test]
async fn eval_of_send_pipe_receive_with_peek_should_meet_in_the_tuple_space_and_proceed() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let channel = new_gstring_par("channel".to_string(), Vec::new(), false);
    let result_channel = new_gstring_par("result".to_string(), Vec::new(), false);

    let split_rand0 = rand().split_byte(0);
    let split_rand1 = rand().split_byte(1);
    let merge_rand = Blake2b512Random::merge(vec![split_rand1.clone(), split_rand0.clone()]);

    let send = Par::default().with_sends(vec![Send {
        chan: Some(channel.clone()),
        data: vec![
            new_gint_par(7, Vec::new(), false),
            new_gint_par(8, Vec::new(), false),
            new_gint_par(9, Vec::new(), false),
        ],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);

    let receive = Par::default().with_receives(vec![Receive {
        binds: vec![ReceiveBind {
            patterns: vec![
                new_freevar_par(0, Vec::new()),
                new_freevar_par(1, Vec::new()),
                new_freevar_par(2, Vec::new()),
            ],
            source: Some(channel.clone()),
            remainder: None,
            free_count: 3,
            pattern_modifiers: vec![],
        }],
        body: Some(Par::default().with_sends(vec![Send {
            chan: Some(result_channel.clone()),
            data: vec![new_gstring_par("Success".to_string(), Vec::new(), false)],
            persistent: false,
            locally_free: Vec::new(),
            connective_used: false,
            hyperparams: vec![],
        }])),
        persistent: false,
        peek: true,
        bind_count: 3,
        locally_free: Vec::new(),
        connective_used: false,
    }]);

    let mut expected_elements = HashMap::new();
    expected_elements.insert(
        channel,
        (
            vec![
                new_gint_par(7, Vec::new(), false),
                new_gint_par(8, Vec::new(), false),
                new_gint_par(9, Vec::new(), false),
            ],
            split_rand0.clone(),
        ),
    );
    expected_elements.insert(
        result_channel,
        (
            vec![new_gstring_par("Success".to_string(), Vec::new(), false)],
            merge_rand,
        ),
    );

    let env: Env<Par> = Env::new();
    assert!(reducer
        .eval(send.clone(), &env, split_rand0.clone())
        .await
        .is_ok());
    assert!(reducer
        .eval(receive.clone(), &env, split_rand1.clone())
        .await
        .is_ok());

    let send_result = space.to_map();
    assert_eq!(send_result, map_data(expected_elements.clone()));

    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;
    assert!(reducer
        .eval(receive, &env, split_rand1.clone())
        .await
        .is_ok());
    assert!(reducer.eval(send, &env, split_rand0.clone()).await.is_ok());

    let receive_result = space.to_map();
    assert_eq!(receive_result, map_data(expected_elements));
}

#[tokio::test]
async fn eval_of_send_pipe_receive_when_whole_list_is_bound_to_list_remainder_should_meet_in_the_tuple_space_and_proceed(
) {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    // for(@[...a] <- @"channel") { … } | @"channel"!([7,8,9])
    let channel = new_gstring_par("channel".to_string(), Vec::new(), false);
    let result_channel = new_gstring_par("result".to_string(), Vec::new(), false);
    let split_rand0 = rand().split_byte(0);
    let split_rand1 = rand().split_byte(1);
    let merge_rand = Blake2b512Random::merge(vec![split_rand1.clone(), split_rand0.clone()]);

    let send = Par::default().with_sends(vec![Send {
        chan: Some(channel.clone()),
        data: vec![new_elist_par(
            vec![
                new_gint_par(7, Vec::new(), false),
                new_gint_par(8, Vec::new(), false),
                new_gint_par(9, Vec::new(), false),
            ],
            Vec::new(),
            false,
            None,
            Vec::new(),
            false,
        )],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);

    let receive = Par::default().with_receives(vec![Receive {
        binds: vec![ReceiveBind {
            patterns: vec![new_elist_par(
                vec![new_freevar_par(0, Vec::new())],
                Vec::new(),
                true,
                Some(new_freevar_var(0)),
                Vec::new(),
                false,
            )],
            source: Some(channel.clone()),
            remainder: None,
            free_count: 1,
            pattern_modifiers: vec![],
        }],
        body: Some(Par::default().with_sends(vec![Send {
            chan: Some(result_channel.clone()),
            data: vec![new_gstring_par("Success".to_string(), Vec::new(), false)],
            persistent: false,
            locally_free: Vec::new(),
            connective_used: false,
            hyperparams: vec![],
        }])),
        persistent: false,
        peek: false,
        bind_count: 1,
        locally_free: Vec::new(),
        connective_used: false,
    }]);

    let env: Env<Par> = Env::new();
    assert!(reducer
        .eval(send.clone(), &env, split_rand0.clone())
        .await
        .is_ok());
    assert!(reducer
        .eval(receive.clone(), &env, split_rand1.clone())
        .await
        .is_ok());

    let send_result = space.to_map();

    let mut expected_elements = HashMap::new();
    expected_elements.insert(
        result_channel,
        (
            vec![new_gstring_par("Success".to_string(), Vec::new(), false)],
            merge_rand,
        ),
    );
    assert_eq!(send_result, map_data(expected_elements.clone()));

    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;
    assert!(reducer
        .eval(receive, &env, split_rand1.clone())
        .await
        .is_ok());
    assert!(reducer.eval(send, &env, split_rand0.clone()).await.is_ok());

    let receive_result = space.to_map();
    assert_eq!(receive_result, map_data(expected_elements));
}

#[tokio::test]
async fn eval_of_send_on_seven_plus_eight_pipe_receive_on_fifteen_should_meet_in_the_tuple_space_and_proceed(
) {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let split_rand0 = rand().split_byte(0);
    let split_rand1 = rand().split_byte(1);
    let merge_rand = Blake2b512Random::merge(vec![split_rand1.clone(), split_rand0.clone()]);

    let send = Par::default().with_sends(vec![Send {
        chan: Some(new_eplus_par_gint(7, 8, Vec::new(), false)),
        data: vec![
            new_gint_par(7, Vec::new(), false),
            new_gint_par(8, Vec::new(), false),
            new_gint_par(9, Vec::new(), false),
        ],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);

    let receive = Par::default().with_receives(vec![Receive {
        binds: vec![ReceiveBind {
            patterns: vec![
                new_freevar_par(0, Vec::new()),
                new_freevar_par(1, Vec::new()),
                new_freevar_par(2, Vec::new()),
            ],
            source: Some(new_gint_par(15, Vec::new(), false)),
            remainder: None,
            free_count: 3,
            pattern_modifiers: vec![],
        }],
        body: Some(Par::default().with_sends(vec![Send {
            chan: Some(new_gstring_par("result".to_string(), Vec::new(), false)),
            data: vec![new_gstring_par("Success".to_string(), Vec::new(), false)],
            persistent: false,
            locally_free: Vec::new(),
            connective_used: false,
            hyperparams: vec![],
        }])),
        persistent: false,
        peek: false,
        bind_count: 3,
        locally_free: Vec::new(),
        connective_used: false,
    }]);

    let env: Env<Par> = Env::new();
    assert!(reducer
        .eval(send.clone(), &env, split_rand0.clone())
        .await
        .is_ok());
    assert!(reducer
        .eval(receive.clone(), &env, split_rand1.clone())
        .await
        .is_ok());

    let send_result = space.to_map();

    let mut expected_elements = HashMap::new();
    expected_elements.insert(
        new_gstring_par("result".to_string(), Vec::new(), false),
        (
            vec![new_gstring_par("Success".to_string(), Vec::new(), false)],
            merge_rand,
        ),
    );
    assert_eq!(send_result, map_data(expected_elements.clone()));

    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;
    assert!(reducer
        .eval(receive, &env, split_rand1.clone())
        .await
        .is_ok());
    assert!(reducer.eval(send, &env, split_rand0.clone()).await.is_ok());

    let receive_result = space.to_map();
    assert_eq!(receive_result, map_data(expected_elements));
}

#[tokio::test]
async fn eval_of_send_of_receive_pipe_receive_should_meet_in_the_tuple_space_and_proceed() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let base_rand = rand().split_byte(2);
    let split_rand0 = base_rand.split_byte(0);
    let split_rand1 = base_rand.split_byte(1);
    let merge_rand = Blake2b512Random::merge(vec![split_rand1.clone(), split_rand0.clone()]);

    let simple_receive = Par::default().with_receives(vec![Receive {
        binds: vec![ReceiveBind {
            patterns: vec![new_gint_par(2, Vec::new(), false)],
            source: Some(new_gint_par(2, Vec::new(), false)),
            remainder: None,
            free_count: 0,
            pattern_modifiers: vec![],
        }],
        body: Some(Par::default()),
        persistent: false,
        peek: false,
        bind_count: 0,
        locally_free: Vec::new(),
        connective_used: false,
    }]);

    let send = Par::default().with_sends(vec![Send {
        chan: Some(new_gint_par(1, Vec::new(), false)),
        data: vec![simple_receive.clone()],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);

    let receive = Par::default().with_receives(vec![Receive {
        binds: vec![ReceiveBind {
            patterns: vec![new_freevar_par(0, Vec::new())],
            source: Some(new_gint_par(1, Vec::new(), false)),
            remainder: None,
            free_count: 1,
            pattern_modifiers: vec![],
        }],
        body: Some(new_boundvar_par(0, Vec::new(), false)),
        persistent: false,
        peek: false,
        bind_count: 1,
        locally_free: Vec::new(),
        connective_used: false,
    }]);

    let env: Env<Par> = Env::new();
    assert!(reducer
        .eval(send.clone(), &env, split_rand0.clone())
        .await
        .is_ok());
    assert!(reducer
        .eval(receive.clone(), &env, split_rand1.clone())
        .await
        .is_ok());

    let send_result = space.to_map();
    let channels = vec![new_gint_par(2, Vec::new(), false)];
    // Because they are evaluated separately, nothing is split.
    assert!(check_continuation(
        send_result,
        channels.clone(),
        vec![BindPattern {
            patterns: vec![new_gint_par(2, Vec::new(), false)],
            remainder: None,
            free_count: 0,
        }],
        ParWithRandom {
            body: Some(Par::default()),
            random_state: merge_rand.to_bytes(),
        },
    ));

    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;
    assert!(reducer
        .eval(receive, &env, split_rand1.clone())
        .await
        .is_ok());
    assert!(reducer.eval(send, &env, split_rand0.clone()).await.is_ok());

    let receive_result = space.to_map();
    assert!(check_continuation(
        receive_result,
        channels.clone(),
        vec![BindPattern {
            patterns: vec![new_gint_par(2, Vec::new(), false)],
            remainder: None,
            free_count: 0,
        }],
        ParWithRandom {
            body: Some(Par::default()),
            random_state: merge_rand.to_bytes(),
        },
    ));

    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;
    let mut par_param = Par::default().with_receives(vec![Receive {
        binds: vec![ReceiveBind {
            patterns: vec![new_freevar_par(0, Vec::new())],
            source: Some(new_gint_par(1, Vec::new(), false)),
            remainder: None,
            free_count: 1,
            pattern_modifiers: vec![],
        }],
        body: Some(new_boundvar_par(0, Vec::new(), false)),
        persistent: false,
        peek: false,
        bind_count: 1,
        locally_free: Vec::new(),
        connective_used: false,
    }]);
    par_param = par_param.with_sends(vec![Send {
        chan: Some(new_gint_par(1, Vec::new(), false)),
        data: vec![simple_receive],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);
    assert!(reducer.eval(par_param, &env, base_rand).await.is_ok());

    let both_result = space.to_map();
    assert!(check_continuation(
        both_result,
        channels,
        vec![BindPattern {
            patterns: vec![new_gint_par(2, Vec::new(), false)],
            remainder: None,
            free_count: 0,
        }],
        ParWithRandom {
            body: Some(Par::default()),
            random_state: merge_rand.to_bytes(),
        },
    ));
}

#[tokio::test]
async fn simple_match_should_capture_and_add_to_the_environment() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let split_rand = rand().split_byte(0);
    let mut pattern = Par::default().with_sends(vec![Send {
        chan: Some(new_freevar_par(0, Vec::new())),
        data: vec![
            new_gint_par(7, Vec::new(), false),
            new_freevar_par(1, Vec::new()),
        ],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: true,
        hyperparams: vec![],
    }]);
    pattern.connective_used = true;

    let send_target = Par::default().with_sends(vec![Send {
        chan: Some(new_boundvar_par(1, Vec::new(), false)),
        data: vec![
            new_gint_par(7, Vec::new(), false),
            new_boundvar_par(0, Vec::new(), false),
        ],
        persistent: false,
        locally_free: vec![0b00000011],
        connective_used: false,
        hyperparams: vec![],
    }]);

    let match_term = Par::default().with_matches(vec![Match {
        target: Some(send_target),
        cases: vec![MatchCase {
            pattern: Some(pattern),
            source: Some(Par::default().with_sends(vec![Send {
                chan: Some(new_gstring_par("result".to_string(), Vec::new(), false)),
                data: vec![
                    new_boundvar_par(1, Vec::new(), false),
                    new_boundvar_par(0, Vec::new(), false),
                ],
                persistent: false,
                locally_free: vec![0b00000011],
                connective_used: false,
                hyperparams: vec![],
            }])),
            free_count: 2,
        }],
        locally_free: Vec::new(),
        connective_used: false,
    }]);

    let mut env: Env<Par> = Env::new();
    env = env.put(GPrivateBuilder::new_par_from_string("one".to_string()));
    env = env.put(GPrivateBuilder::new_par_from_string("zero".to_string()));
    assert!(reducer
        .eval(match_term.clone(), &env, split_rand.clone())
        .await
        .is_ok());

    let match_result = space.to_map();
    let mut expected_elements = HashMap::new();
    expected_elements.insert(
        new_gstring_par("result".to_string(), Vec::new(), false),
        (
            vec![
                GPrivateBuilder::new_par_from_string("one".to_string()),
                GPrivateBuilder::new_par_from_string("zero".to_string()),
            ],
            split_rand,
        ),
    );
    assert_eq!(match_result, map_data(expected_elements));
}

#[tokio::test]
async fn eval_of_send_pipe_send_pipe_receive_join_should_meet_in_tuplespace_and_proceed() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let split_rand0 = rand().split_byte(0);
    let split_rand1 = rand().split_byte(1);
    let split_rand2 = rand().split_byte(2);
    let merge_rand = Blake2b512Random::merge(vec![
        split_rand2.clone(),
        split_rand0.clone(),
        split_rand1.clone(),
    ]);

    let send1 = Par::default().with_sends(vec![Send {
        chan: Some(new_gstring_par("channel1".to_string(), Vec::new(), false)),
        data: vec![
            new_gint_par(7, Vec::new(), false),
            new_gint_par(8, Vec::new(), false),
            new_gint_par(9, Vec::new(), false),
        ],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);

    let send2 = Par::default().with_sends(vec![Send {
        chan: Some(new_gstring_par("channel2".to_string(), Vec::new(), false)),
        data: vec![
            new_gint_par(7, Vec::new(), false),
            new_gint_par(8, Vec::new(), false),
            new_gint_par(9, Vec::new(), false),
        ],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);

    let receive = Par::default().with_receives(vec![Receive {
        binds: vec![
            ReceiveBind {
                patterns: vec![
                    new_freevar_par(0, Vec::new()),
                    new_freevar_par(1, Vec::new()),
                    new_freevar_par(2, Vec::new()),
                ],
                source: Some(new_gstring_par("channel1".to_string(), Vec::new(), false)),
                remainder: None,
                free_count: 3,
            pattern_modifiers: vec![],
            },
            ReceiveBind {
                patterns: vec![
                    new_freevar_par(0, Vec::new()),
                    new_freevar_par(1, Vec::new()),
                    new_freevar_par(2, Vec::new()),
                ],
                source: Some(new_gstring_par("channel2".to_string(), Vec::new(), false)),
                remainder: None,
                free_count: 3,
            pattern_modifiers: vec![],
            },
        ],
        body: Some(Par::default().with_sends(vec![Send {
            chan: Some(new_gstring_par("result".to_string(), Vec::new(), false)),
            data: vec![new_gstring_par("Success".to_string(), Vec::new(), false)],
            persistent: false,
            locally_free: Vec::new(),
            connective_used: false,
            hyperparams: vec![],
        }])),
        persistent: false,
        peek: false,
        bind_count: 3,
        locally_free: Vec::new(),
        connective_used: false,
    }]);

    assert!(reducer
        .inj(send1.clone(), split_rand0.clone())
        .await
        .is_ok());
    assert!(reducer
        .inj(send2.clone(), split_rand1.clone())
        .await
        .is_ok());
    assert!(reducer
        .inj(receive.clone(), split_rand2.clone())
        .await
        .is_ok());

    let send_result = space.to_map();
    let mut expected_elements = HashMap::new();
    expected_elements.insert(
        new_gstring_par("result".to_string(), Vec::new(), false),
        (
            vec![new_gstring_par("Success".to_string(), Vec::new(), false)],
            merge_rand,
        ),
    );
    assert_eq!(send_result, map_data(expected_elements.clone()));

    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;
    assert!(reducer
        .inj(receive.clone(), split_rand2.clone())
        .await
        .is_ok());
    assert!(reducer
        .inj(send1.clone(), split_rand0.clone())
        .await
        .is_ok());
    assert!(reducer
        .inj(send2.clone(), split_rand1.clone())
        .await
        .is_ok());

    let receive_result = space.to_map();
    assert_eq!(receive_result, map_data(expected_elements.clone()));

    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;
    assert!(reducer
        .inj(send1.clone(), split_rand0.clone())
        .await
        .is_ok());
    assert!(reducer
        .inj(receive.clone(), split_rand2.clone())
        .await
        .is_ok());
    assert!(reducer
        .inj(send2.clone(), split_rand1.clone())
        .await
        .is_ok());

    let inter_leaved_result = space.to_map();
    assert_eq!(inter_leaved_result, map_data(expected_elements));
}

#[tokio::test]
async fn eval_of_send_with_remainder_receive_should_capture_the_remainder() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let split_rand0 = rand().split_byte(0);
    let split_rand1 = rand().split_byte(1);
    let merge_rand = Blake2b512Random::merge(vec![split_rand1.clone(), split_rand0.clone()]);

    let send = Par::default().with_sends(vec![Send {
        chan: Some(new_gstring_par("channel".to_string(), Vec::new(), false)),
        data: vec![
            new_gint_par(7, Vec::new(), false),
            new_gint_par(8, Vec::new(), false),
            new_gint_par(9, Vec::new(), false),
        ],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);

    let receive = Par::default().with_receives(vec![Receive {
        binds: vec![ReceiveBind {
            patterns: vec![],
            source: Some(new_gstring_par("channel".to_string(), Vec::new(), false)),
            remainder: Some(new_freevar_var(0)),
            free_count: 1,
            pattern_modifiers: vec![],
        }],
        body: Some(Par::default().with_sends(vec![Send {
            chan: Some(new_gstring_par("result".to_string(), Vec::new(), false)),
            data: vec![new_boundvar_par(0, Vec::new(), false)],
            persistent: false,
            locally_free: Vec::new(),
            connective_used: false,
            hyperparams: vec![],
        }])),
        persistent: false,
        peek: false,
        bind_count: 0,
        locally_free: Vec::new(),
        connective_used: false,
    }]);

    let env = Env::new();
    assert!(reducer
        .eval(receive.clone(), &env, split_rand1.clone())
        .await
        .is_ok());
    assert!(reducer
        .eval(send.clone(), &env, split_rand0.clone())
        .await
        .is_ok());

    let result = space.to_map();
    let mut expected_elements = HashMap::new();
    expected_elements.insert(
        new_gstring_par("result".to_string(), Vec::new(), false),
        (
            vec![new_elist_par(
                vec![
                    new_gint_par(7, Vec::new(), false),
                    new_gint_par(8, Vec::new(), false),
                    new_gint_par(9, Vec::new(), false),
                ],
                Vec::new(),
                false,
                None,
                Vec::new(),
                false,
            )],
            merge_rand,
        ),
    );
    assert_eq!(result, map_data(expected_elements));
}

#[tokio::test]
async fn eval_of_nth_method_should_pick_out_the_nth_item_from_a_list() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let split_rand = rand().split_byte(0);
    let nth_call = Expr {
        expr_instance: Some(ExprInstance::EMethodBody(EMethod {
            method_name: "nth".to_string(),
            target: Some(new_elist_par(
                vec![
                    new_gint_par(7, Vec::new(), false),
                    new_gint_par(8, Vec::new(), false),
                    new_gint_par(9, Vec::new(), false),
                ],
                Vec::new(),
                false,
                None,
                Vec::new(),
                false,
            )),
            arguments: vec![new_gint_par(2, Vec::new(), false)],
            locally_free: Vec::new(),
            connective_used: false,
        })),
    };

    let env = Env::new();
    let direct_result = reducer.eval_expr_to_par(&nth_call, &env).unwrap();
    assert_eq!(new_gint_par(9, Vec::new(), false), direct_result);

    let nth_call_eval_to_send = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EMethodBody(EMethod {
            method_name: "nth".to_string(),
            target: Some(new_elist_par(
                vec![
                    new_gint_par(7, Vec::new(), false),
                    Par::default().with_sends(vec![Send {
                        chan: Some(new_gstring_par("result".to_string(), Vec::new(), false)),
                        data: vec![new_gstring_par("Success".to_string(), Vec::new(), false)],
                        persistent: false,
                        locally_free: Vec::new(),
                        connective_used: false,
                        hyperparams: vec![],
                    }]),
                    new_gint_par(9, Vec::new(), false),
                    new_gint_par(10, Vec::new(), false),
                ],
                Vec::new(),
                false,
                None,
                Vec::new(),
                false,
            )),
            arguments: vec![new_gint_par(1, Vec::new(), false)],
            locally_free: Vec::new(),
            connective_used: false,
        })),
    }]);

    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;
    assert!(reducer
        .eval(nth_call_eval_to_send, &env, split_rand.clone())
        .await
        .is_ok());

    let indirect_result = space.to_map();
    let mut expected_elements = HashMap::new();
    expected_elements.insert(
        new_gstring_par("result".to_string(), Vec::new(), false),
        (
            vec![new_gstring_par("Success".to_string(), Vec::new(), false)],
            split_rand,
        ),
    );
    assert_eq!(indirect_result, map_data(expected_elements));
}

#[tokio::test]
async fn eval_of_nth_method_should_pick_out_the_nth_item_from_a_byte_array() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let nth_call = Expr {
        expr_instance: Some(ExprInstance::EMethodBody(EMethod {
            method_name: "nth".to_string(),
            target: Some(Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GByteArray(vec![1, 2, 255])),
            }])),
            arguments: vec![new_gint_par(2, Vec::new(), false)],
            locally_free: Vec::new(),
            connective_used: false,
        })),
    };

    let env = Env::new();
    let direct_result = reducer.eval_expr_to_par(&nth_call, &env).unwrap();
    assert_eq!(new_gint_par(255, Vec::new(), false), direct_result);
}

#[tokio::test]
async fn eval_of_length_method_should_get_length_of_byte_array() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let nth_call = Expr {
        expr_instance: Some(ExprInstance::EMethodBody(EMethod {
            method_name: "length".to_string(),
            target: Some(Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GByteArray(vec![1, 2, 255])),
            }])),
            arguments: vec![],
            locally_free: Vec::new(),
            connective_used: false,
        })),
    };

    let env = Env::new();
    let direct_result = reducer.eval_expr_to_par(&nth_call, &env).unwrap();
    assert_eq!(new_gint_par(3, Vec::new(), false), direct_result);
}

#[tokio::test]
async fn eval_of_new_should_use_deterministic_names_and_provide_urn_based_resources() {
    let split_rand = rand().split_byte(42);
    let mut result_rand = rand().split_byte(42);
    let chosen_name = result_rand.next();
    let result0_rand = result_rand.split_byte(0);
    let result1_rand = result_rand.split_byte(1);

    let new = Par::default().with_news(vec![New {
        bind_count: 2,
        p: Some(Par::default().with_sends(vec![
            Send {
                chan: Some(new_gstring_par("result0".to_string(), Vec::new(), false)),
                data: vec![new_boundvar_par(0, Vec::new(), false)],
                persistent: false,
                locally_free: vec![0],
                connective_used: false,
                hyperparams: vec![],
            },
            Send {
                chan: Some(new_gstring_par("result1".to_string(), Vec::new(), false)),
                data: vec![new_boundvar_par(1, Vec::new(), false)],
                persistent: false,
                locally_free: vec![1],
                connective_used: false,
                hyperparams: vec![],
            },
        ])),
        uri: vec!["rho:test:foo".to_string()],
        injections: BTreeMap::new(),
        locally_free: vec![0b00000011],
        space_types: vec![],
    }]);

    let cost = CostAccounting::empty_cost();
    let mut urn_map = HashMap::new();
    urn_map.insert(
        "rho:test:foo".to_string(),
        Par::default().with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GPrivateBody(GPrivate { id: vec![42] })),
        }]),
    );

    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.unwrap();
    let space = RSpace::create(store, Arc::new(Box::new(Matcher))).unwrap();
    let rspace: RhoISpace = Arc::new(tokio::sync::Mutex::new(Box::new(space.clone())));
    let reducer = DebruijnInterpreter::new(
        rspace,
        Arc::new(urn_map),
        Arc::new(std::sync::RwLock::new(HashSet::new())),
        Par::default(),
        cost.clone(),
    );
    cost.set(Cost::unsafe_max());
    let env = Env::new();
    assert!(reducer.eval(new, &env, split_rand).await.is_ok());
    let result = space.to_map();

    let channel0 = new_gstring_par("result0".to_string(), Vec::new(), false);
    let channel1 = new_gstring_par("result1".to_string(), Vec::new(), false);

    let mut expected_result = HashMap::new();
    expected_result.insert(
        vec![channel0.clone()],
        Row {
            data: vec![Datum::create(
                &channel0,
                ListParWithRandom {
                    pars: vec![Par::default().with_unforgeables(vec![GUnforgeable {
                        unf_instance: Some(UnfInstance::GPrivateBody(GPrivate { id: vec![42] })),
                    }])],
                    random_state: result0_rand.to_bytes(),
                },
                false,
            )],
            wks: vec![],
        },
    );

    expected_result.insert(
        vec![channel1.clone()],
        Row {
            data: vec![Datum::create(
                &channel1,
                ListParWithRandom {
                    pars: vec![Par::default().with_unforgeables(vec![GUnforgeable {
                        unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
                            id: chosen_name.iter().map(|&x| x as u8).collect::<Vec<u8>>(),
                        })),
                    }])],
                    random_state: result1_rand.to_bytes(),
                },
                false,
            )],
            wks: vec![],
        },
    );

    assert_eq!(result, expected_result);
}

#[tokio::test]
async fn eval_of_nth_method_in_send_position_should_change_what_is_sent() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let split_rand = rand().split_byte(0);
    let nth_call_eval_to_send = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EMethodBody(EMethod {
            method_name: "nth".to_string(),
            target: Some(new_elist_par(
                vec![
                    new_gint_par(7, Vec::new(), false),
                    Par::default().with_sends(vec![Send {
                        chan: Some(new_gstring_par("result".to_string(), Vec::new(), false)),
                        data: vec![new_gstring_par("Success".to_string(), Vec::new(), false)],
                        persistent: false,
                        locally_free: Vec::new(),
                        connective_used: false,
                        hyperparams: vec![],
                    }]),
                    new_gint_par(9, Vec::new(), false),
                    new_gint_par(10, Vec::new(), false),
                ],
                Vec::new(),
                false,
                None,
                Vec::new(),
                false,
            )),
            arguments: vec![new_gint_par(1, Vec::new(), false)],
            locally_free: Vec::new(),
            connective_used: false,
        })),
    }]);

    let send = Par::default().with_sends(vec![Send {
        chan: Some(new_gstring_par("result".to_string(), Vec::new(), false)),
        data: vec![nth_call_eval_to_send],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);

    let env = Env::new();
    assert!(reducer.eval(send, &env, split_rand.clone()).await.is_ok());
    let result = space.to_map();

    let channel = new_gstring_par("result".to_string(), Vec::new(), false);
    let mut expected_result = HashMap::new();
    expected_result.insert(
        channel.clone(),
        (
            vec![Par::default().with_sends(vec![Send {
                chan: Some(channel),
                data: vec![new_gstring_par("Success".to_string(), Vec::new(), false)],
                persistent: false,
                locally_free: Vec::new(),
                connective_used: false,
                hyperparams: vec![],
            }])],
            split_rand,
        ),
    );
    assert_eq!(result, map_data(expected_result));
}

#[tokio::test]
async fn eval_of_a_method_should_substitute_target_before_evaluating() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let hex_to_bytes_call = Expr {
        expr_instance: Some(ExprInstance::EMethodBody(EMethod {
            method_name: "hexToBytes".to_string(),
            target: Some(new_boundvar_par(0, Vec::new(), false)),
            arguments: vec![],
            locally_free: Vec::new(),
            connective_used: false,
        })),
    };

    let mut env = Env::new();
    env = env.put(new_gstring_par("deadbeef".to_string(), Vec::new(), false));
    let direct_result = reducer.eval_expr_to_par(&hex_to_bytes_call, &env).unwrap();
    assert_eq!(
        direct_result,
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::GByteArray(StringOps::unsafe_decode_hex(
                "deadbeef".to_string()
            )))
        }])
    );
}

#[tokio::test]
async fn eval_of_to_byte_array_method_on_any_process_should_return_that_process_serialized() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let split_rand = rand().split_byte(0);
    let proc = Par::default().with_receives(vec![Receive {
        binds: vec![ReceiveBind {
            patterns: vec![new_freevar_par(0, Vec::new())],
            source: Some(new_gstring_par("channel".to_string(), Vec::new(), false)),
            remainder: None,
            free_count: 0,
            pattern_modifiers: vec![],
        }],
        body: Some(Par::default()),
        persistent: false,
        peek: false,
        bind_count: 1,
        locally_free: Vec::new(),
        connective_used: false,
    }]);

    let serialized_process = proc.encode_to_vec();
    let to_byte_array_call = Par::default().with_sends(vec![Send {
        chan: Some(new_gstring_par("result".to_string(), Vec::new(), false)),
        data: vec![Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toByteArray".to_string(),
                target: Some(proc),
                arguments: Vec::new(),
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }])],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);

    let env: Env<Par> = Env::new();
    assert!(reducer
        .eval(to_byte_array_call, &env, split_rand.clone())
        .await
        .is_ok());
    let result = space.to_map();
    let mut expected_result = HashMap::new();
    expected_result.insert(
        new_gstring_par("result".to_string(), Vec::new(), false),
        (
            vec![Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GByteArray(serialized_process)),
            }])],
            split_rand,
        ),
    );
    assert_eq!(result, map_data(expected_result));
}

#[tokio::test]
async fn eval_of_to_byte_array_method_on_any_process_should_substitute_before_serialization() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let split_rand = rand().split_byte(0);
    let unsub_proc = Par::default().with_news(vec![New {
        bind_count: 1,
        p: Some(new_boundvar_par(1, Vec::new(), false)),
        uri: Vec::new(),
        injections: BTreeMap::new(),
        locally_free: vec![0],
        space_types: vec![],
    }]);
    let sub_proc = Par::default().with_news(vec![New {
        bind_count: 1,
        p: Some(GPrivateBuilder::new_par_from_string("zero".to_string())),
        uri: Vec::new(),
        injections: BTreeMap::new(),
        locally_free: vec![],
        space_types: vec![],
    }]);

    let serialized_process = sub_proc.encode_to_vec();
    let to_byte_array_call = Par::default().with_sends(vec![Send {
        chan: Some(new_gstring_par("result".to_string(), Vec::new(), false)),
        data: vec![Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toByteArray".to_string(),
                target: Some(unsub_proc),
                arguments: Vec::new(),
                locally_free: vec![0],
                connective_used: false,
            })),
        }])],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);

    let mut env: Env<Par> = Env::new();
    env = env.put(GPrivateBuilder::new_par_from_string("one".to_string()));
    env = env.put(GPrivateBuilder::new_par_from_string("zero".to_string()));

    assert!(reducer
        .eval(to_byte_array_call, &env, split_rand.clone())
        .await
        .is_ok());
    let result = space.to_map();
    let mut expected_result = HashMap::new();
    expected_result.insert(
        new_gstring_par("result".to_string(), Vec::new(), false),
        (
            vec![Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GByteArray(serialized_process)),
            }])],
            split_rand,
        ),
    );
    assert_eq!(result, map_data(expected_result));
}

#[tokio::test]
async fn eval_of_to_string_method_on_deploy_id_return_that_id_serialized() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;
    let split_rand = rand().split_byte(0);

    let deploy_id = Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(UnfInstance::GDeployIdBody(GDeployId {
            sig: "deployId".as_bytes().to_vec(),
        })),
    }]);

    let to_string_call = Par::default().with_sends(vec![Send {
        chan: Some(new_gstring_par("result".to_string(), Vec::new(), false)),
        data: vec![Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toString".to_string(),
                target: Some(deploy_id),
                arguments: vec![],
                locally_free: vec![],
                connective_used: false,
            })),
        }])],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);

    let env: Env<Par> = Env::new();

    assert!(reducer
        .eval(to_string_call, &env, split_rand.clone())
        .await
        .is_ok());

    let result = space.to_map();
    let mut expected_result = HashMap::new();
    expected_result.insert(
        new_gstring_par("result".to_string(), Vec::new(), false),
        (
            vec![Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GString("6465706c6f794964".into())),
            }])],
            split_rand,
        ),
    );
    assert_eq!(result, map_data(expected_result));
}

#[tokio::test]
async fn eval_of_to_byte_array_method_on_any_process_should_return_an_error_when_to_byte_array_is_called_with_arguments(
) {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let to_byte_array_call = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EMethodBody(EMethod {
            method_name: "toByteArray".to_string(),
            target: Some(new_gint_par(1, Vec::new(), false)),
            arguments: vec![new_gint_par(1, Vec::new(), false)],
            locally_free: vec![],
            connective_used: false,
        })),
    }]);

    let env: Env<Par> = Env::new();
    let result = reducer.eval(to_byte_array_call, &env, rand()).await;
    assert!(result.is_err());
    assert_eq!(
        result,
        Err(InterpreterError::MethodArgumentNumberMismatch {
            method: String::from("toByteArray"),
            expected: 0,
            actual: 1
        })
    );
}

#[tokio::test]
async fn eval_of_hex_to_bytes_should_transform_encoded_string_to_byte_array_not_the_rholang_term() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let split_rand = rand().split_byte(0);
    let test_string = String::from("testing testing");
    let base16_repr = hex::encode(test_string.clone());
    let proc = new_gstring_par(base16_repr, Vec::new(), false);

    let to_byte_array_call = Par::default().with_sends(vec![Send {
        chan: Some(new_gstring_par("result".to_string(), Vec::new(), false)),
        data: vec![Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "hexToBytes".to_string(),
                target: Some(proc),
                arguments: Vec::new(),
                locally_free: vec![],
                connective_used: false,
            })),
        }])],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);

    let env: Env<Par> = Env::new();
    assert!(reducer
        .eval(to_byte_array_call, &env, split_rand.clone())
        .await
        .is_ok());
    let result = space.to_map();
    let mut expected_result = HashMap::new();
    expected_result.insert(
        new_gstring_par("result".to_string(), Vec::new(), false),
        (
            vec![Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GByteArray(test_string.as_bytes().to_vec())),
            }])],
            split_rand,
        ),
    );
    assert_eq!(result, map_data(expected_result));
}

#[tokio::test]
async fn eval_of_bytes_to_hex_should_transform_byte_array_to_hex_string_not_the_rholang_term() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let split_rand = rand().split_byte(0);
    let base16_repr = String::from("0123456789abcdef");
    let test_bytes = StringOps::unsafe_decode_hex(base16_repr.clone());
    let proc = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::GByteArray(test_bytes)),
    }]);

    let to_string_call = Par::default().with_sends(vec![Send {
        chan: Some(new_gstring_par("result".to_string(), Vec::new(), false)),
        data: vec![Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "bytesToHex".to_string(),
                target: Some(proc),
                arguments: Vec::new(),
                locally_free: vec![],
                connective_used: false,
            })),
        }])],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);

    let env: Env<Par> = Env::new();
    assert!(reducer
        .eval(to_string_call, &env, split_rand.clone())
        .await
        .is_ok());
    let result = space.to_map();
    let mut expected_result = HashMap::new();
    expected_result.insert(
        new_gstring_par("result".to_string(), Vec::new(), false),
        (
            vec![new_gstring_par(base16_repr, Vec::new(), false)],
            split_rand,
        ),
    );
    assert_eq!(result, map_data(expected_result));
}

#[tokio::test]
async fn eval_of_to_utf8_bytes_should_transform_string_to_utf8_byte_array_not_the_rholang_term() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let split_rand = rand().split_byte(0);
    let test_string = String::from("testing testing");
    let proc = new_gstring_par(test_string.clone(), Vec::new(), false);

    let to_utf8_bytes_call = Par::default().with_sends(vec![Send {
        chan: Some(new_gstring_par("result".to_string(), Vec::new(), false)),
        data: vec![Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toUtf8Bytes".to_string(),
                target: Some(proc),
                arguments: Vec::new(),
                locally_free: vec![],
                connective_used: false,
            })),
        }])],
        persistent: false,
        locally_free: Vec::new(),
        connective_used: false,
        hyperparams: vec![],
    }]);

    let env: Env<Par> = Env::new();
    assert!(reducer
        .eval(to_utf8_bytes_call, &env, split_rand.clone())
        .await
        .is_ok());
    let result = space.to_map();
    let mut expected_result = HashMap::new();
    expected_result.insert(
        new_gstring_par("result".to_string(), Vec::new(), false),
        (
            vec![Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GByteArray(test_string.as_bytes().to_vec())),
            }])],
            split_rand,
        ),
    );
    assert_eq!(result, map_data(expected_result));
}

#[tokio::test]
async fn eval_of_to_utf8_bytes_should_return_an_error_when_to_utf8_bytes_is_called_with_arguments()
{
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let to_utf8_bytes_call = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EMethodBody(EMethod {
            method_name: "toUtf8Bytes".to_string(),
            target: Some(new_gint_par(1, Vec::new(), false)),
            arguments: vec![new_gint_par(1, Vec::new(), false)],
            locally_free: vec![],
            connective_used: false,
        })),
    }]);

    let env: Env<Par> = Env::new();
    let result = reducer.eval(to_utf8_bytes_call, &env, rand()).await;
    assert!(result.is_err());
    assert_eq!(
        result,
        Err(InterpreterError::MethodArgumentNumberMismatch {
            method: String::from("toUtf8Bytes"),
            expected: 0,
            actual: 1
        })
    );
}

#[tokio::test]
async fn eval_of_to_utf8_bytes_should_return_an_error_when_to_utf8_bytes_is_evaluated_on_a_non_string(
) {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let to_utf8_bytes_call = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EMethodBody(EMethod {
            method_name: "toUtf8Bytes".to_string(),
            target: Some(new_gint_par(44, Vec::new(), false)),
            arguments: vec![],
            locally_free: vec![],
            connective_used: false,
        })),
    }]);

    let env: Env<Par> = Env::new();
    let result = reducer.eval(to_utf8_bytes_call, &env, rand()).await;
    assert!(result.is_err());
    assert_eq!(
        result,
        Err(InterpreterError::MethodNotDefined {
            method: "toUtf8Bytes".to_string(),
            other_type: String::from("int")
        })
    );
}

#[tokio::test]
async fn variable_references_should_be_substituted_before_being_used() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let mut split_rand_result = rand().split_byte(3);
    let split_rand_src = rand().split_byte(3);
    split_rand_result.next();
    let merge_rand = Blake2b512Random::merge(vec![
        split_rand_result.split_byte(1),
        split_rand_result.split_byte(0),
    ]);

    let proc = Par::default().with_news(vec![New {
        bind_count: 1,
        p: Some(
            Par::default()
                .with_sends(vec![Send {
                    chan: Some(new_boundvar_par(0, Vec::new(), false)),
                    data: vec![new_boundvar_par(0, Vec::new(), false)],
                    persistent: false,
                    locally_free: Vec::new(),
                    connective_used: false,
                    hyperparams: vec![],
                }])
                .with_receives(vec![Receive {
                    binds: vec![ReceiveBind {
                        patterns: vec![Par::default().with_connectives(vec![Connective {
                            connective_instance: Some(VarRefBody(VarRef { index: 0, depth: 1 })),
                        }])],
                        source: Some(new_boundvar_par(0, Vec::new(), false)),
                        remainder: None,
                        free_count: 0,
                        pattern_modifiers: vec![],
                    }],
                    body: Some(Par::default().with_sends(vec![Send {
                        chan: Some(new_gstring_par("result".to_string(), Vec::new(), false)),
                        data: vec![new_gstring_par("true".to_string(), Vec::new(), false)],
                        persistent: false,
                        locally_free: Vec::new(),
                        connective_used: false,
                        hyperparams: vec![],
                    }])),
                    persistent: false,
                    peek: false,
                    bind_count: 0,
                    locally_free: Vec::new(),
                    connective_used: false,
                }]),
        ),
        uri: Vec::new(),
        injections: BTreeMap::new(),
        locally_free: vec![],
        space_types: vec![],
    }]);

    let env = Env::new();
    let res = reducer
        .eval(proc.clone(), &env, split_rand_src.clone())
        .await;
    assert!(res.is_ok());

    let result = space.to_map();
    let mut expected_elements = HashMap::new();
    expected_elements.insert(
        new_gstring_par("result".to_string(), Vec::new(), false),
        (
            vec![new_gstring_par("true".to_string(), Vec::new(), false)],
            merge_rand.clone(),
        ),
    );
    println!("\nmerge rand");
    merge_rand.debug_str();
    assert_eq!(result, map_data(expected_elements));
}

#[tokio::test]
async fn variable_references_should_be_substituted_before_being_used_in_a_match() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let mut split_rand_result = rand().split_byte(4);
    let split_rand_src = rand().split_byte(4);
    let _ = split_rand_result.next();

    let proc = Par::default().with_news(vec![New {
        bind_count: 1,
        p: Some(Par::default().with_matches(vec![Match {
            target: Some(new_boundvar_par(0, Vec::new(), false)),
            cases: vec![MatchCase {
                pattern: Some(Par::default().with_connectives(vec![Connective {
                    connective_instance: Some(VarRefBody(VarRef { index: 0, depth: 1 })),
                }])),
                source: Some(Par::default().with_sends(vec![Send {
                    chan: Some(new_gstring_par("result".to_string(), Vec::new(), false)),
                    data: vec![new_gstring_par("true".to_string(), Vec::new(), false)],
                    persistent: false,
                    locally_free: Vec::new(),
                    connective_used: false,
                    hyperparams: vec![],
                }])),
                free_count: 0,
            }],
            locally_free: Vec::new(),
            connective_used: false,
        }])),
        uri: Vec::new(),
        injections: BTreeMap::new(),
        locally_free: vec![],
        space_types: vec![],
    }]);

    let env = Env::new();
    let res = reducer
        .eval(proc.clone(), &env, split_rand_src.clone())
        .await;
    assert!(res.is_ok());

    let result = space.to_map();
    let mut expected_elements = HashMap::new();
    expected_elements.insert(
        new_gstring_par("result".to_string(), Vec::new(), false),
        (
            vec![new_gstring_par("true".to_string(), Vec::new(), false)],
            split_rand_result,
        ),
    );
    assert_eq!(result, map_data(expected_elements));
}

#[tokio::test]
async fn variable_references_should_reference_a_variable_that_comes_from_a_match_in_tuplespace() {
    let (space, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let base_rand = rand().split_byte(7);
    let split_rand0 = base_rand.split_byte(0);
    let split_rand1 = base_rand.split_byte(1);
    let merge_rand = Blake2b512Random::merge(vec![split_rand1, split_rand0]);

    let proc = Par::default()
        .with_sends(vec![Send {
            chan: Some(new_gint_par(7, Vec::new(), false)),
            data: vec![new_gint_par(10, Vec::new(), false)],
            persistent: false,
            locally_free: Vec::new(),
            connective_used: false,
            hyperparams: vec![],
        }])
        .with_receives(vec![Receive {
            binds: vec![ReceiveBind {
                patterns: vec![new_freevar_par(0, Vec::new())],
                source: Some(new_gint_par(7, Vec::new(), false)),
                remainder: None,
                free_count: 1,
            pattern_modifiers: vec![],
            }],
            body: Some(Par::default().with_matches(vec![Match {
                target: Some(new_gint_par(10, Vec::new(), false)),
                cases: vec![MatchCase {
                    pattern: Some(Par::default().with_connectives(vec![Connective {
                        connective_instance: Some(VarRefBody(VarRef { index: 0, depth: 1 })),
                    }])),
                    source: Some(Par::default().with_sends(vec![Send {
                        chan: Some(new_gstring_par("result".to_string(), Vec::new(), false)),
                        data: vec![new_gstring_par("true".to_string(), Vec::new(), false)],
                        persistent: false,
                        locally_free: Vec::new(),
                        connective_used: false,
                        hyperparams: vec![],
                    }])),
                    free_count: 0,
                }],
                locally_free: Vec::new(),
                connective_used: false,
            }])),
            persistent: false,
            peek: false,
            bind_count: 0,
            locally_free: Vec::new(),
            connective_used: false,
        }]);

    let env = Env::new();
    let res = reducer.eval(proc.clone(), &env, base_rand.clone()).await;
    assert!(res.is_ok());

    let result = space.to_map();
    let mut expected_elements = HashMap::new();
    expected_elements.insert(
        new_gstring_par("result".to_string(), Vec::new(), false),
        (
            vec![new_gstring_par("true".to_string(), Vec::new(), false)],
            merge_rand,
        ),
    );
    assert_eq!(result, map_data(expected_elements));
}

#[tokio::test]
async fn one_matches_one_should_return_true() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMatchesBody(EMatches {
                target: Some(new_gint_par(1, Vec::new(), false)),
                pattern: Some(new_gint_par(1, Vec::new(), false)),
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(res.unwrap().exprs, vec![new_gbool_expr(true)])
}

#[tokio::test]
async fn one_matches_zero_should_return_false() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMatchesBody(EMatches {
                target: Some(new_gint_par(1, Vec::new(), false)),
                pattern: Some(new_gint_par(0, Vec::new(), false)),
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(res.unwrap().exprs, vec![new_gbool_expr(false)])
}

// "1 matches _"
#[tokio::test]
async fn one_matches_wildcard_should_return_true() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMatchesBody(EMatches {
                target: Some(new_gint_par(1, Vec::new(), false)),
                pattern: Some(new_wildcard_par(Vec::new(), false)),
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(res.unwrap().exprs, vec![new_gbool_expr(true)])
}

#[tokio::test]
async fn x_matches_one_should_return_true_when_x_is_bound_to_one() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let mut env = Env::new();
    env = env.put(new_gint_par(1, Vec::new(), false));
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMatchesBody(EMatches {
                target: Some(new_boundvar_par(0, Vec::new(), false)),
                pattern: Some(new_gint_par(1, Vec::new(), false)),
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(res.unwrap().exprs, vec![new_gbool_expr(true)])
}

// "1 matches =x"
#[tokio::test]
async fn one_matches_equal_sign_x_should_return_true_when_x_is_bound_to_one() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let mut env = Env::new();
    env = env.put(new_gint_par(1, Vec::new(), false));
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMatchesBody(EMatches {
                target: Some(new_gint_par(1, Vec::new(), false)),
                pattern: Some(Par::default().with_connectives(vec![Connective {
                    connective_instance: Some(VarRefBody(VarRef { index: 0, depth: 1 })),
                }])),
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(res.unwrap().exprs, vec![new_gbool_expr(true)])
}

#[tokio::test]
async fn length_should_return_the_length_of_the_string() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "length".to_string(),
                target: Some(new_gstring_par("abc".to_string(), Vec::new(), false)),
                arguments: Vec::new(),
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(res.unwrap().exprs, vec![new_gint_expr(3)])
}

// "'abcabac'.slice(3, 6)" should "return 'aba'"
#[tokio::test]
async fn slice_should_work_correctly_1() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "slice".to_string(),
                target: Some(new_gstring_par("abcabac".to_string(), Vec::new(), false)),
                arguments: vec![
                    new_gint_par(3, Vec::new(), false),
                    new_gint_par(6, Vec::new(), false),
                ],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![new_gstring_expr("aba".to_string())]
    )
}

// "'abcabcac'.slice(2,1)" should "return empty string"
#[tokio::test]
async fn slice_should_work_correctly_2() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "slice".to_string(),
                target: Some(new_gstring_par("abcabac".to_string(), Vec::new(), false)),
                arguments: vec![
                    new_gint_par(2, Vec::new(), false),
                    new_gint_par(1, Vec::new(), false),
                ],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(res.unwrap().exprs, vec![new_gstring_expr("".to_string())])
}

// "'abcabcac'.slice(8,9)" should "return empty string"
#[tokio::test]
async fn slice_should_work_correctly_3() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "slice".to_string(),
                target: Some(new_gstring_par("abcabac".to_string(), Vec::new(), false)),
                arguments: vec![
                    new_gint_par(8, Vec::new(), false),
                    new_gint_par(9, Vec::new(), false),
                ],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(res.unwrap().exprs, vec![new_gstring_expr("".to_string())])
}

// "'abcabcac'.slice(-2,2)" should "return 'ab'"
#[tokio::test]
async fn slice_should_work_correctly_4() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "slice".to_string(),
                target: Some(new_gstring_par("abcabac".to_string(), Vec::new(), false)),
                arguments: vec![
                    new_gint_par(-2, Vec::new(), false),
                    new_gint_par(2, Vec::new(), false),
                ],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(res.unwrap().exprs, vec![new_gstring_expr("ab".to_string())])
}

// "'Hello, ${name}!' % {'name': 'Alice'}" should "return 'Hello, Alice!"
#[tokio::test]
async fn percent_percent_should_work_correctly() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPercentPercentBody(EPercentPercent {
                p1: Some(new_gstring_par(
                    "Hello, ${name}!".to_string(),
                    Vec::new(),
                    false,
                )),
                p2: Some(Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
                        ParMap::create_from_vec(vec![(
                            new_gstring_par("name".to_string(), Vec::new(), false),
                            new_gstring_par("Alice".to_string(), Vec::new(), false),
                        )]),
                    ))),
                }])),
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![new_gstring_expr("Hello, Alice!".to_string())]
    )
}

// "'abc' ++ 'def'" should "return 'abcdef"
#[tokio::test]
async fn plus_plus_should_work_correctly_with_string() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPlusPlusBody(EPlusPlus {
                p1: Some(new_gstring_par("abc".to_string(), Vec::new(), false)),
                p2: Some(new_gstring_par("def".to_string(), Vec::new(), false)),
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![new_gstring_expr("abcdef".to_string())]
    )
}

// "ByteArray('dead') ++ ByteArray('beef)'" should "return ByteArray('deadbeef')"
#[tokio::test]
async fn plus_plus_should_work_correctly_with_byte_array() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPlusPlusBody(EPlusPlus {
                p1: Some(Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::GByteArray(StringOps::unsafe_decode_hex(
                        "dead".to_string(),
                    ))),
                }])),
                p2: Some(Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::GByteArray(StringOps::unsafe_decode_hex(
                        "beef".to_string(),
                    ))),
                }])),
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::GByteArray(StringOps::unsafe_decode_hex(
                "deadbeef".to_string(),
            ))),
        }]
    )
}

fn interpolate(base: String, substitutes: Vec<(Par, Par)>) -> Par {
    Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EPercentPercentBody(EPercentPercent {
            p1: Some(new_gstring_par(base, Vec::new(), false)),
            p2: Some(Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
                    ParMap::create_from_vec(substitutes),
                ))),
            }])),
        })),
    }])
}
// "'${a} ${b}' % {'a': '1 ${b}', 'b': '2 ${a}'" should "return '1 ${b} 2 ${a}"
#[tokio::test]
async fn interpolate_should_work_correctly() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &interpolate(
            "${a} ${b}".to_string(),
            vec![
                (
                    new_gstring_par("a".to_string(), Vec::new(), false),
                    new_gstring_par("1 ${b}".to_string(), Vec::new(), false),
                ),
                (
                    new_gstring_par("b".to_string(), Vec::new(), false),
                    new_gstring_par("2 ${a}".to_string(), Vec::new(), false),
                ),
            ],
        ),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![new_gstring_expr("1 ${b} 2 ${a}".to_string())]
    )
}

#[tokio::test]
async fn interpolate_should_interpolate_boolean_values() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &interpolate(
            "${a} ${b}".to_string(),
            vec![
                (
                    new_gstring_par("a".to_string(), Vec::new(), false),
                    new_gbool_par(false, Vec::new(), false),
                ),
                (
                    new_gstring_par("b".to_string(), Vec::new(), false),
                    new_gbool_par(true, Vec::new(), false),
                ),
            ],
        ),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![new_gstring_expr("false true".to_string())]
    )
}

#[tokio::test]
async fn interpolate_should_interpolate_uris() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &interpolate(
            "${a} ${b}".to_string(),
            vec![
                (
                    new_gstring_par("a".to_string(), Vec::new(), false),
                    Par::default().with_exprs(vec![Expr {
                        expr_instance: Some(ExprInstance::GUri("testUriA".to_string())),
                    }]),
                ),
                (
                    new_gstring_par("b".to_string(), Vec::new(), false),
                    Par::default().with_exprs(vec![Expr {
                        expr_instance: Some(ExprInstance::GUri("testUriB".to_string())),
                    }]),
                ),
            ],
        ),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![new_gstring_expr("testUriA testUriB".to_string())]
    )
}

#[tokio::test]
async fn length_should_return_the_length_of_the_list() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let list = new_elist_par(
        vec![
            new_gint_par(0, Vec::new(), false),
            new_gint_par(1, Vec::new(), false),
            new_gint_par(2, Vec::new(), false),
            new_gint_par(3, Vec::new(), false),
        ],
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "length".to_string(),
                target: Some(list),
                arguments: vec![],
                locally_free: vec![],
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(res.unwrap().exprs, vec![new_gint_expr(4)])
}

// "[3, 7, 2, 9, 4, 3, 7].slice(3, 5)" should "return [9, 4]"
#[tokio::test]
async fn slice_should_work_correctly_with_list_1() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let list = new_elist_par(
        vec![
            new_gint_par(3, Vec::new(), false),
            new_gint_par(7, Vec::new(), false),
            new_gint_par(2, Vec::new(), false),
            new_gint_par(9, Vec::new(), false),
            new_gint_par(4, Vec::new(), false),
            new_gint_par(3, Vec::new(), false),
            new_gint_par(7, Vec::new(), false),
        ],
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "slice".to_string(),
                target: Some(list),
                arguments: vec![
                    new_gint_par(3, Vec::new(), false),
                    new_gint_par(5, Vec::new(), false),
                ],
                locally_free: vec![],
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::EListBody(EList {
                ps: vec![
                    new_gint_par(9, Vec::new(), false),
                    new_gint_par(4, Vec::new(), false),
                ],
                locally_free: Vec::new(),
                connective_used: false,
                remainder: None
            }))
        }]
    )
}

// "[3, 7, 2, 9, 4, 3, 7].slice(5, 4)" should "return []"
#[tokio::test]
async fn slice_should_work_correctly_with_list_2() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let list = new_elist_par(
        vec![
            new_gint_par(3, Vec::new(), false),
            new_gint_par(7, Vec::new(), false),
            new_gint_par(2, Vec::new(), false),
            new_gint_par(9, Vec::new(), false),
            new_gint_par(4, Vec::new(), false),
            new_gint_par(3, Vec::new(), false),
            new_gint_par(7, Vec::new(), false),
        ],
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "slice".to_string(),
                target: Some(list),
                arguments: vec![
                    new_gint_par(5, Vec::new(), false),
                    new_gint_par(4, Vec::new(), false),
                ],
                locally_free: vec![],
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::EListBody(EList {
                ps: vec![],
                locally_free: Vec::new(),
                connective_used: false,
                remainder: None
            }))
        }]
    )
}

// "[3, 7, 2, 9, 4, 3, 7].slice(7, 8)" should "return []"
#[tokio::test]
async fn slice_should_work_correctly_with_list_3() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let list = new_elist_par(
        vec![
            new_gint_par(3, Vec::new(), false),
            new_gint_par(7, Vec::new(), false),
            new_gint_par(2, Vec::new(), false),
            new_gint_par(9, Vec::new(), false),
            new_gint_par(4, Vec::new(), false),
            new_gint_par(3, Vec::new(), false),
            new_gint_par(7, Vec::new(), false),
        ],
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "slice".to_string(),
                target: Some(list),
                arguments: vec![
                    new_gint_par(7, Vec::new(), false),
                    new_gint_par(8, Vec::new(), false),
                ],
                locally_free: vec![],
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::EListBody(EList {
                ps: vec![],
                locally_free: Vec::new(),
                connective_used: false,
                remainder: None
            }))
        }]
    )
}

// "[3, 7, 2, 9, 4, 3, 7].slice(-2, 2)" should "return [3, 7]"
#[tokio::test]
async fn slice_should_work_correctly_with_list_4() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let list = new_elist_par(
        vec![
            new_gint_par(3, Vec::new(), false),
            new_gint_par(7, Vec::new(), false),
            new_gint_par(2, Vec::new(), false),
            new_gint_par(9, Vec::new(), false),
            new_gint_par(4, Vec::new(), false),
            new_gint_par(3, Vec::new(), false),
            new_gint_par(7, Vec::new(), false),
        ],
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "slice".to_string(),
                target: Some(list),
                arguments: vec![
                    new_gint_par(-2, Vec::new(), false),
                    new_gint_par(2, Vec::new(), false),
                ],
                locally_free: vec![],
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::EListBody(EList {
                ps: vec![
                    new_gint_par(3, Vec::new(), false),
                    new_gint_par(7, Vec::new(), false),
                ],
                locally_free: Vec::new(),
                connective_used: false,
                remainder: None
            }))
        }]
    )
}

// "[3, 2, 9] ++ [6, 1, 7]" should "return [3, 2, 9, 6, 1, 7]"
#[tokio::test]
async fn plus_plus_should_work_correctly_with_list() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let lhs_list = new_elist_par(
        vec![
            new_gint_par(3, Vec::new(), false),
            new_gint_par(2, Vec::new(), false),
            new_gint_par(9, Vec::new(), false),
        ],
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );
    let rhs_list = new_elist_par(
        vec![
            new_gint_par(6, Vec::new(), false),
            new_gint_par(1, Vec::new(), false),
            new_gint_par(7, Vec::new(), false),
        ],
        Vec::new(),
        false,
        None,
        Vec::new(),
        false,
    );

    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPlusPlusBody(EPlusPlus {
                p1: Some(lhs_list),
                p2: Some(rhs_list),
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![new_elist_expr(
            vec![
                new_gint_par(3, Vec::new(), false),
                new_gint_par(2, Vec::new(), false),
                new_gint_par(9, Vec::new(), false),
                new_gint_par(6, Vec::new(), false),
                new_gint_par(1, Vec::new(), false),
                new_gint_par(7, Vec::new(), false),
            ],
            Vec::new(),
            false,
            None
        )]
    )
}

// "{1: 'a', 2: 'b'}.getOrElse(1, 'c')" should "return 'a'"
#[tokio::test]
async fn get_or_else_method_should_work_correctly_1() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let map = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
            ParMap::create_from_vec(vec![
                (
                    new_gint_par(1, Vec::new(), false),
                    new_gstring_par("a".to_string(), Vec::new(), false),
                ),
                (
                    new_gint_par(2, Vec::new(), false),
                    new_gstring_par("b".to_string(), Vec::new(), false),
                ),
            ]),
        ))),
    }]);

    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "getOrElse".to_string(),
                target: Some(map),
                arguments: vec![
                    new_gint_par(1, Vec::new(), false),
                    new_gstring_par("c".to_string(), Vec::new(), false),
                ],
                locally_free: vec![],
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(res.unwrap().exprs, vec![new_gstring_expr("a".to_string())])
}

// "{1: 'a', 2: 'b'}.getOrElse(3, 'c')" should "return 'c'"
#[tokio::test]
async fn get_or_else_method_should_work_correctly_2() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let map = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
            ParMap::create_from_vec(vec![
                (
                    new_gint_par(1, Vec::new(), false),
                    new_gstring_par("a".to_string(), Vec::new(), false),
                ),
                (
                    new_gint_par(2, Vec::new(), false),
                    new_gstring_par("b".to_string(), Vec::new(), false),
                ),
            ]),
        ))),
    }]);

    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "getOrElse".to_string(),
                target: Some(map),
                arguments: vec![
                    new_gint_par(3, Vec::new(), false),
                    new_gstring_par("c".to_string(), Vec::new(), false),
                ],
                locally_free: vec![],
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(res.unwrap().exprs, vec![new_gstring_expr("c".to_string())])
}

// "{1: 'a', 2: 'b'}.set(3, 'c')" should "return {1: 'a', 2: 'b', 3: 'c'}"
#[tokio::test]
async fn set_method_should_work_correctly_1() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let map = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
            ParMap::create_from_vec(vec![
                (
                    new_gint_par(1, Vec::new(), false),
                    new_gstring_par("a".to_string(), Vec::new(), false),
                ),
                (
                    new_gint_par(2, Vec::new(), false),
                    new_gstring_par("b".to_string(), Vec::new(), false),
                ),
            ]),
        ))),
    }]);

    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "set".to_string(),
                target: Some(map),
                arguments: vec![
                    new_gint_par(3, Vec::new(), false),
                    new_gstring_par("c".to_string(), Vec::new(), false),
                ],
                locally_free: vec![],
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
                ParMap::create_from_vec(vec![
                    (
                        new_gint_par(1, Vec::new(), false),
                        new_gstring_par("a".to_string(), Vec::new(), false),
                    ),
                    (
                        new_gint_par(2, Vec::new(), false),
                        new_gstring_par("b".to_string(), Vec::new(), false),
                    ),
                    (
                        new_gint_par(3, Vec::new(), false),
                        new_gstring_par("c".to_string(), Vec::new(), false),
                    ),
                ]),
            ))),
        }]
    )
}

// "{1: 'a', 2: 'b'}.set(2, 'c')" should "return {1: 'a', 2: 'c'}"
#[tokio::test]
async fn set_method_should_work_correctly_2() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let map = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
            ParMap::create_from_vec(vec![
                (
                    new_gint_par(1, Vec::new(), false),
                    new_gstring_par("a".to_string(), Vec::new(), false),
                ),
                (
                    new_gint_par(2, Vec::new(), false),
                    new_gstring_par("b".to_string(), Vec::new(), false),
                ),
            ]),
        ))),
    }]);

    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "set".to_string(),
                target: Some(map),
                arguments: vec![
                    new_gint_par(2, Vec::new(), false),
                    new_gstring_par("c".to_string(), Vec::new(), false),
                ],
                locally_free: vec![],
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
                ParMap::create_from_vec(vec![
                    (
                        new_gint_par(1, Vec::new(), false),
                        new_gstring_par("a".to_string(), Vec::new(), false),
                    ),
                    (
                        new_gint_par(2, Vec::new(), false),
                        new_gstring_par("c".to_string(), Vec::new(), false),
                    ),
                ]),
            ))),
        }]
    )
}

// "{1: 'a', 2: 'b', 3: 'c'}.keys()" should "return Set(1, 2, 3)"
#[tokio::test]
async fn keys_method_should_work_correctly() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let map = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
            ParMap::create_from_vec(vec![
                (
                    new_gint_par(1, Vec::new(), false),
                    new_gstring_par("a".to_string(), Vec::new(), false),
                ),
                (
                    new_gint_par(2, Vec::new(), false),
                    new_gstring_par("b".to_string(), Vec::new(), false),
                ),
                (
                    new_gint_par(3, Vec::new(), false),
                    new_gstring_par("c".to_string(), Vec::new(), false),
                ),
            ]),
        ))),
    }]);

    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "keys".to_string(),
                target: Some(map),
                arguments: vec![],
                locally_free: vec![],
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
                ParSet::create_from_vec(vec![
                    new_gint_par(1, Vec::new(), false),
                    new_gint_par(2, Vec::new(), false),
                    new_gint_par(3, Vec::new(), false),
                ])
            ))),
        }]
    )
}

#[tokio::test]
async fn size_method_should_work_correctly_emap() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let map = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
            ParMap::create_from_vec(vec![
                (
                    new_gint_par(1, Vec::new(), false),
                    new_gstring_par("a".to_string(), Vec::new(), false),
                ),
                (
                    new_gint_par(2, Vec::new(), false),
                    new_gstring_par("b".to_string(), Vec::new(), false),
                ),
                (
                    new_gint_par(3, Vec::new(), false),
                    new_gstring_par("c".to_string(), Vec::new(), false),
                ),
            ]),
        ))),
    }]);

    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "size".to_string(),
                target: Some(map),
                arguments: vec![],
                locally_free: vec![],
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(res.unwrap().exprs, vec![new_gint_expr(3)])
}

#[tokio::test]
async fn size_method_should_work_correctly_with_eset() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let set = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
            ParSet::create_from_vec(vec![
                new_gint_par(1, Vec::new(), false),
                new_gint_par(2, Vec::new(), false),
                new_gint_par(3, Vec::new(), false),
            ]),
        ))),
    }]);

    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "size".to_string(),
                target: Some(set),
                arguments: vec![],
                locally_free: vec![],
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(res.unwrap().exprs, vec![new_gint_expr(3)])
}

#[tokio::test]
async fn plus_method_should_work_correctly_with_eset() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let set = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
            ParSet::create_from_vec(vec![
                new_gint_par(1, Vec::new(), false),
                new_gint_par(2, Vec::new(), false),
            ]),
        ))),
    }]);

    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPlusBody(EPlus {
                p1: Some(set),
                p2: Some(new_gint_par(3, Vec::new(), false)),
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
                ParSet::create_from_vec(vec![
                    new_gint_par(1, Vec::new(), false),
                    new_gint_par(2, Vec::new(), false),
                    new_gint_par(3, Vec::new(), false),
                ]),
            ))),
        }]
    )
}

// "{1: 'a', 2: 'b', 3: 'c'} - 3" should "return {1: 'a', 2: 'b'}"
#[tokio::test]
async fn minus_method_should_work_correctly_with_emap() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let map = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
            ParMap::create_from_vec(vec![
                (
                    new_gint_par(1, Vec::new(), false),
                    new_gstring_par("a".to_string(), Vec::new(), false),
                ),
                (
                    new_gint_par(2, Vec::new(), false),
                    new_gstring_par("b".to_string(), Vec::new(), false),
                ),
                (
                    new_gint_par(3, Vec::new(), false),
                    new_gstring_par("c".to_string(), Vec::new(), false),
                ),
            ]),
        ))),
    }]);

    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMinusBody(EMinus {
                p1: Some(map),
                p2: Some(new_gint_par(3, Vec::new(), false)),
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
                ParMap::create_from_vec(vec![
                    (
                        new_gint_par(1, Vec::new(), false),
                        new_gstring_par("a".to_string(), Vec::new(), false),
                    ),
                    (
                        new_gint_par(2, Vec::new(), false),
                        new_gstring_par("b".to_string(), Vec::new(), false),
                    ),
                ]),
            ))),
        }]
    )
}

#[tokio::test]
async fn minus_method_should_work_correctly_with_eset() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let set = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
            ParSet::create_from_vec(vec![
                new_gint_par(1, Vec::new(), false),
                new_gint_par(2, Vec::new(), false),
                new_gint_par(3, Vec::new(), false),
            ]),
        ))),
    }]);

    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMinusBody(EMinus {
                p1: Some(set),
                p2: Some(new_gint_par(3, Vec::new(), false)),
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
                ParSet::create_from_vec(vec![
                    new_gint_par(1, Vec::new(), false),
                    new_gint_par(2, Vec::new(), false),
                ]),
            ))),
        }]
    )
}

#[tokio::test]
async fn plus_plus_method_should_work_correctly_with_eset() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let lhs_set = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
            ParSet::create_from_vec(vec![
                new_gint_par(1, Vec::new(), false),
                new_gint_par(2, Vec::new(), false),
            ]),
        ))),
    }]);
    let rhs_set = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
            ParSet::create_from_vec(vec![
                new_gint_par(3, Vec::new(), false),
                new_gint_par(4, Vec::new(), false),
            ]),
        ))),
    }]);

    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPlusPlusBody(EPlusPlus {
                p1: Some(lhs_set),
                p2: Some(rhs_set),
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
                ParSet::create_from_vec(vec![
                    new_gint_par(1, Vec::new(), false),
                    new_gint_par(2, Vec::new(), false),
                    new_gint_par(3, Vec::new(), false),
                    new_gint_par(4, Vec::new(), false),
                ]),
            ))),
        }]
    )
}

#[tokio::test]
async fn plus_plus_method_with_map_should_return_union() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let lhs_map = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
            ParMap::create_from_vec(vec![
                (
                    new_gint_par(1, Vec::new(), false),
                    new_gstring_par("a".to_string(), Vec::new(), false),
                ),
                (
                    new_gint_par(2, Vec::new(), false),
                    new_gstring_par("b".to_string(), Vec::new(), false),
                ),
            ]),
        ))),
    }]);
    let rhs_map = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
            ParMap::create_from_vec(vec![
                (
                    new_gint_par(3, Vec::new(), false),
                    new_gstring_par("c".to_string(), Vec::new(), false),
                ),
                (
                    new_gint_par(4, Vec::new(), false),
                    new_gstring_par("d".to_string(), Vec::new(), false),
                ),
            ]),
        ))),
    }]);

    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPlusPlusBody(EPlusPlus {
                p1: Some(lhs_map),
                p2: Some(rhs_map),
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
                ParMap::create_from_vec(vec![
                    (
                        new_gint_par(1, Vec::new(), false),
                        new_gstring_par("a".to_string(), Vec::new(), false),
                    ),
                    (
                        new_gint_par(2, Vec::new(), false),
                        new_gstring_par("b".to_string(), Vec::new(), false),
                    ),
                    (
                        new_gint_par(3, Vec::new(), false),
                        new_gstring_par("c".to_string(), Vec::new(), false),
                    ),
                    (
                        new_gint_par(4, Vec::new(), false),
                        new_gstring_par("d".to_string(), Vec::new(), false),
                    ),
                ]),
            ))),
        }]
    )
}

#[tokio::test]
async fn minus_minus_method_should_work_correctly_with_eset() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let lhs_set = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
            ParSet::create_from_vec(vec![
                new_gint_par(1, Vec::new(), false),
                new_gint_par(2, Vec::new(), false),
                new_gint_par(3, Vec::new(), false),
                new_gint_par(4, Vec::new(), false),
            ]),
        ))),
    }]);
    let rhs_set = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
            ParSet::create_from_vec(vec![
                new_gint_par(1, Vec::new(), false),
                new_gint_par(2, Vec::new(), false),
            ]),
        ))),
    }]);

    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMinusMinusBody(EMinusMinus {
                p1: Some(lhs_set),
                p2: Some(rhs_set),
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
                ParSet::create_from_vec(vec![
                    new_gint_par(3, Vec::new(), false),
                    new_gint_par(4, Vec::new(), false),
                ]),
            ))),
        }]
    )
}

#[tokio::test]
async fn get_method_on_set_should_not_be_defined() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let set = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
            ParSet::create_from_vec(vec![
                new_gint_par(1, Vec::new(), false),
                new_gint_par(2, Vec::new(), false),
                new_gint_par(3, Vec::new(), false),
            ]),
        ))),
    }]);

    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "get".to_string(),
                target: Some(set),
                arguments: vec![new_gint_par(1, Vec::new(), false)],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_err());
    assert_eq!(
        res,
        Err(InterpreterError::MethodNotDefined {
            method: "get".to_string(),
            other_type: "set".to_string()
        })
    )
}

#[tokio::test]
async fn add_method_on_map_should_not_be_defined() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let map = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
            ParMap::create_from_vec(vec![
                (
                    new_gint_par(1, Vec::new(), false),
                    new_gstring_par("a".to_string(), Vec::new(), false),
                ),
                (
                    new_gint_par(2, Vec::new(), false),
                    new_gstring_par("b".to_string(), Vec::new(), false),
                ),
            ]),
        ))),
    }]);

    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "add".to_string(),
                target: Some(map),
                arguments: vec![new_gint_par(1, Vec::new(), false)],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_err());
    assert_eq!(
        res,
        Err(InterpreterError::MethodNotDefined {
            method: "add".to_string(),
            other_type: "map".to_string()
        })
    )
}

#[tokio::test]
async fn to_list_method_should_error_when_called_with_arguments() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toList".to_string(),
                target: Some(new_elist_par(
                    vec![],
                    Vec::new(),
                    false,
                    None,
                    Vec::new(),
                    false,
                )),
                arguments: vec![new_gint_par(1, Vec::new(), false)],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_err());
    assert_eq!(
        res,
        Err(InterpreterError::MethodArgumentNumberMismatch {
            method: "to_list".to_string(),
            expected: 0,
            actual: 1
        })
    )
}

#[tokio::test]
async fn to_list_method_should_transform_set_into_list() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toList".to_string(),
                target: Some(Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
                        ParSet::create_from_vec(vec![
                            new_gint_par(1, Vec::new(), false),
                            new_gint_par(2, Vec::new(), false),
                            new_gint_par(3, Vec::new(), false),
                        ]),
                    ))),
                }])),
                arguments: vec![],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![new_elist_expr(
            vec![
                new_gint_par(1, Vec::new(), false),
                new_gint_par(2, Vec::new(), false),
                new_gint_par(3, Vec::new(), false),
            ],
            Vec::new(),
            false,
            None
        )]
    )
}

#[tokio::test]
async fn to_list_method_should_transform_map_into_list_of_tuples() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toList".to_string(),
                target: Some(Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
                        ParMap::create_from_vec(vec![
                            (
                                new_gstring_par("a".to_string(), Vec::new(), false),
                                new_gint_par(1, Vec::new(), false),
                            ),
                            (
                                new_gstring_par("b".to_string(), Vec::new(), false),
                                new_gint_par(2, Vec::new(), false),
                            ),
                            (
                                new_gstring_par("c".to_string(), Vec::new(), false),
                                new_gint_par(3, Vec::new(), false),
                            ),
                        ]),
                    ))),
                }])),
                arguments: vec![],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![new_elist_expr(
            vec![
                new_etuple_par(vec![
                    new_gstring_par("a".to_string(), Vec::new(), false),
                    new_gint_par(1, Vec::new(), false),
                ]),
                new_etuple_par(vec![
                    new_gstring_par("b".to_string(), Vec::new(), false),
                    new_gint_par(2, Vec::new(), false),
                ]),
                new_etuple_par(vec![
                    new_gstring_par("c".to_string(), Vec::new(), false),
                    new_gint_par(3, Vec::new(), false),
                ])
            ],
            Vec::new(),
            false,
            None
        )]
    )
}

#[tokio::test]
async fn to_list_method_should_transform_tuple_into_list() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toList".to_string(),
                target: Some(Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::ETupleBody(ETuple {
                        ps: vec![
                            new_gint_par(1, Vec::new(), false),
                            new_gint_par(2, Vec::new(), false),
                            new_gint_par(3, Vec::new(), false),
                        ],
                        locally_free: Vec::new(),
                        connective_used: false,
                    })),
                }])),
                arguments: vec![],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![new_elist_expr(
            vec![
                new_gint_par(1, Vec::new(), false),
                new_gint_par(2, Vec::new(), false),
                new_gint_par(3, Vec::new(), false),
            ],
            Vec::new(),
            false,
            None
        )]
    )
}

#[tokio::test]
async fn to_set_method_should_turn_list_into_set() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toSet".to_string(),
                target: Some(new_elist_par(
                    vec![
                        new_gint_par(1, Vec::new(), false),
                        new_gint_par(2, Vec::new(), false),
                        new_gint_par(3, Vec::new(), false),
                    ],
                    Vec::new(),
                    false,
                    None,
                    Vec::new(),
                    false,
                )),
                arguments: vec![],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
                ParSet::create_from_vec(vec![
                    new_gint_par(1, Vec::new(), false),
                    new_gint_par(2, Vec::new(), false),
                    new_gint_par(3, Vec::new(), false),
                ]),
            ))),
        }]
    )
}

#[tokio::test]
async fn to_set_method_should_turn_list_with_duplicate_into_set_without_duplicate() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toSet".to_string(),
                target: Some(new_elist_par(
                    vec![
                        new_gint_par(1, Vec::new(), false),
                        new_gint_par(1, Vec::new(), false),
                    ],
                    Vec::new(),
                    false,
                    None,
                    Vec::new(),
                    false,
                )),
                arguments: vec![],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
                ParSet::create_from_vec(vec![new_gint_par(1, Vec::new(), false)]),
            ))),
        }]
    )
}

#[tokio::test]
async fn to_set_method_should_turn_empty_list_into_empty_set() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toSet".to_string(),
                target: Some(new_elist_par(
                    vec![],
                    Vec::new(),
                    false,
                    None,
                    Vec::new(),
                    false,
                )),
                arguments: vec![],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
                ParSet::create_from_vec(vec![]),
            ))),
        }]
    )
}

#[tokio::test]
async fn to_map_method_should_transform_list_of_tuples_into_map() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toMap".to_string(),
                target: Some(Par::default().with_exprs(vec![new_elist_expr(
                    vec![
                        new_etuple_par(vec![
                            new_gstring_par("a".to_string(), Vec::new(), false),
                            new_gint_par(1, Vec::new(), false),
                        ]),
                        new_etuple_par(vec![
                            new_gstring_par("b".to_string(), Vec::new(), false),
                            new_gint_par(2, Vec::new(), false),
                        ]),
                        new_etuple_par(vec![
                            new_gstring_par("c".to_string(), Vec::new(), false),
                            new_gint_par(3, Vec::new(), false),
                        ]),
                    ],
                    Vec::new(),
                    false,
                    None,
                )])),
                arguments: vec![],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );
    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
                ParMap::create_from_vec(vec![
                    (
                        new_gstring_par("a".to_string(), Vec::new(), false),
                        new_gint_par(1, Vec::new(), false),
                    ),
                    (
                        new_gstring_par("b".to_string(), Vec::new(), false),
                        new_gint_par(2, Vec::new(), false),
                    ),
                    (
                        new_gstring_par("c".to_string(), Vec::new(), false),
                        new_gint_par(3, Vec::new(), false),
                    ),
                ]),
            ))),
        }]
    )
}

#[tokio::test]
async fn to_map_method_should_transform_set_of_tuples_into_map() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toMap".to_string(),
                target: Some(Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
                        ParSet::create_from_vec(vec![
                            new_etuple_par(vec![
                                new_gstring_par("a".to_string(), Vec::new(), false),
                                new_gint_par(1, Vec::new(), false),
                            ]),
                            new_etuple_par(vec![
                                new_gstring_par("b".to_string(), Vec::new(), false),
                                new_gint_par(2, Vec::new(), false),
                            ]),
                            new_etuple_par(vec![
                                new_gstring_par("c".to_string(), Vec::new(), false),
                                new_gint_par(3, Vec::new(), false),
                            ]),
                        ]),
                    ))),
                }])),
                arguments: vec![],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );

    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
                ParMap::create_from_vec(vec![
                    (
                        new_gstring_par("a".to_string(), Vec::new(), false),
                        new_gint_par(1, Vec::new(), false),
                    ),
                    (
                        new_gstring_par("b".to_string(), Vec::new(), false),
                        new_gint_par(2, Vec::new(), false),
                    ),
                    (
                        new_gstring_par("c".to_string(), Vec::new(), false),
                        new_gint_par(3, Vec::new(), false),
                    ),
                ]),
            ))),
        }]
    )
}

#[tokio::test]
async fn to_set_method_should_turn_map_into_set() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toSet".to_string(),
                target: Some(Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
                        ParMap::create_from_vec(vec![
                            (
                                new_gstring_par("a".to_string(), Vec::new(), false),
                                new_gint_par(1, Vec::new(), false),
                            ),
                            (
                                new_gstring_par("b".to_string(), Vec::new(), false),
                                new_gint_par(2, Vec::new(), false),
                            ),
                            (
                                new_gstring_par("c".to_string(), Vec::new(), false),
                                new_gint_par(3, Vec::new(), false),
                            ),
                        ]),
                    ))),
                }])),
                arguments: vec![],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );

    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
                ParSet::create_from_vec(vec![
                    new_etuple_par(vec![
                        new_gstring_par("a".to_string(), Vec::new(), false),
                        new_gint_par(1, Vec::new(), false),
                    ]),
                    new_etuple_par(vec![
                        new_gstring_par("b".to_string(), Vec::new(), false),
                        new_gint_par(2, Vec::new(), false),
                    ]),
                    new_etuple_par(vec![
                        new_gstring_par("c".to_string(), Vec::new(), false),
                        new_gint_par(3, Vec::new(), false),
                    ]),
                ]),
            ))),
        }]
    )
}

#[tokio::test]
async fn to_map_method_should_correctly_do_put_operations() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toMap".to_string(),
                target: Some(Par::default().with_exprs(vec![new_elist_expr(
                    vec![
                        new_etuple_par(vec![
                            new_gstring_par("a".to_string(), Vec::new(), false),
                            new_gint_par(1, Vec::new(), false),
                        ]),
                        new_etuple_par(vec![
                            new_gstring_par("a".to_string(), Vec::new(), false),
                            new_gint_par(2, Vec::new(), false),
                        ]),
                    ],
                    Vec::new(),
                    false,
                    None,
                )])),
                arguments: vec![],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );

    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
                ParMap::create_from_vec(vec![(
                    new_gstring_par("a".to_string(), Vec::new(), false),
                    new_gint_par(2, Vec::new(), false),
                ),]),
            ))),
        }]
    )
}

#[tokio::test]
async fn to_map_method_should_turn_empty_list_into_empty_map() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toMap".to_string(),
                target: Some(Par::default().with_exprs(vec![new_elist_expr(
                    vec![],
                    Vec::new(),
                    false,
                    None,
                )])),
                arguments: vec![],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );

    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
                ParMap::create_from_vec(vec![]),
            ))),
        }]
    )
}

#[tokio::test]
async fn to_set_method_should_not_change_the_object_it_is_applied_on() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
                ParSet::create_from_vec(vec![
                    new_gint_par(1, Vec::new(), false),
                    new_gint_par(2, Vec::new(), false),
                    new_gint_par(3, Vec::new(), false),
                ]),
            ))),
        }]),
        &env,
    );

    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
                ParSet::create_from_vec(vec![
                    new_gint_par(1, Vec::new(), false),
                    new_gint_par(2, Vec::new(), false),
                    new_gint_par(3, Vec::new(), false),
                ]),
            ))),
        }]
    )
}

#[tokio::test]
async fn to_map_method_should_not_change_the_object_it_is_applied_on() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toMap".to_string(),
                target: Some(Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
                        ParMap::create_from_vec(vec![
                            (
                                new_gstring_par("a".to_string(), Vec::new(), false),
                                new_gint_par(1, Vec::new(), false),
                            ),
                            (
                                new_gstring_par("b".to_string(), Vec::new(), false),
                                new_gint_par(2, Vec::new(), false),
                            ),
                            (
                                new_gstring_par("c".to_string(), Vec::new(), false),
                                new_gint_par(3, Vec::new(), false),
                            ),
                        ]),
                    ))),
                }])),
                arguments: vec![],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );

    assert!(res.is_ok());
    assert_eq!(
        res.unwrap().exprs,
        vec![Expr {
            expr_instance: Some(ExprInstance::EMapBody(ParMapTypeMapper::par_map_to_emap(
                ParMap::create_from_vec(vec![
                    (
                        new_gstring_par("a".to_string(), Vec::new(), false),
                        new_gint_par(1, Vec::new(), false),
                    ),
                    (
                        new_gstring_par("b".to_string(), Vec::new(), false),
                        new_gint_par(2, Vec::new(), false),
                    ),
                    (
                        new_gstring_par("c".to_string(), Vec::new(), false),
                        new_gint_par(3, Vec::new(), false),
                    ),
                ]),
            ))),
        }]
    )
}

#[tokio::test]
async fn to_map_method_should_throw_error_if_not_called_with_correct_types() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toMap".to_string(),
                target: Some(new_elist_par(
                    vec![
                        new_gstring_par("a".to_string(), Vec::new(), false),
                        new_etuple_par(vec![
                            new_gstring_par("b".to_string(), Vec::new(), false),
                            new_gint_par(2, Vec::new(), false),
                        ]),
                    ],
                    Vec::new(),
                    false,
                    None,
                    Vec::new(),
                    false,
                )),
                arguments: vec![],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );

    assert!(res.is_err());
    assert_eq!(
        res,
        Err(InterpreterError::MethodNotDefined {
            method: "to_map".to_string(),
            other_type: "types except List[(K,V)]".to_string()
        })
    )
}

#[tokio::test]
async fn to_map_method_should_throw_error_when_called_with_arguments() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toMap".to_string(),
                target: Some(new_elist_par(
                    vec![new_etuple_par(vec![
                        new_gstring_par("b".to_string(), Vec::new(), false),
                        new_gint_par(2, Vec::new(), false),
                    ])],
                    Vec::new(),
                    false,
                    None,
                    Vec::new(),
                    false,
                )),
                arguments: vec![new_gint_par(2, Vec::new(), false)],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );

    assert!(res.is_err());
    assert_eq!(
        res,
        Err(InterpreterError::MethodArgumentNumberMismatch {
            method: "to_map".to_string(),
            expected: 0,
            actual: 1
        })
    )
}

#[tokio::test]
async fn to_map_method_should_throw_error_when_called_on_an_int() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toMap".to_string(),
                target: Some(new_gint_par(2, Vec::new(), false)),
                arguments: vec![],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );

    assert!(res.is_err());
    assert_eq!(
        res,
        Err(InterpreterError::MethodNotDefined {
            method: "to_map".to_string(),
            other_type: "int".to_string()
        })
    )
}

#[tokio::test]
async fn to_set_method_should_throw_error_when_called_with_arguments() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toSet".to_string(),
                target: Some(Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::ESetBody(ParSetTypeMapper::par_set_to_eset(
                        ParSet::create_from_vec(vec![]),
                    ))),
                }])),
                arguments: vec![new_gint_par(2, Vec::new(), false)],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );

    assert!(res.is_err());
    assert_eq!(
        res,
        Err(InterpreterError::MethodArgumentNumberMismatch {
            method: "to_set".to_string(),
            expected: 0,
            actual: 1
        })
    )
}

#[tokio::test]
async fn to_set_method_should_throw_error_when_called_on_an_int() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let res = reducer.eval_expr(
        &Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMethodBody(EMethod {
                method_name: "toSet".to_string(),
                target: Some(new_gint_par(2, Vec::new(), false)),
                arguments: vec![],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }]),
        &env,
    );

    assert!(res.is_err());
    assert_eq!(
        res,
        Err(InterpreterError::MethodNotDefined {
            method: "to_set".to_string(),
            other_type: "int".to_string()
        })
    )
}

#[tokio::test]
async fn term_split_size_max_should_be_evaluated_for_max_size() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let p = New {
        bind_count: 1,
        p: Some(Par::default()),
        uri: vec![],
        injections: BTreeMap::new(),
        locally_free: vec![],
        space_types: vec![],
    };
    let news = vec![p; std::i16::MAX as usize];
    let proc = Par::default().with_news(news);

    let res = reducer.eval(proc, &env, rand()).await;
    assert!(res.is_ok());
}

#[tokio::test]
async fn term_split_size_max_should_limited_to_max_value() {
    let (_, reducer) =
        create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
            .await;

    let env = Env::new();
    let p = New {
        bind_count: 1,
        p: Some(Par::default()),
        uri: vec![],
        injections: BTreeMap::new(),
        locally_free: vec![],
        space_types: vec![],
    };
    let news = vec![p; std::i16::MAX as usize + 1];
    let proc = Par::default().with_news(news);

    let res = reducer.eval(proc, &env, rand()).await;
    assert!(res.is_err());
    assert_eq!(
        res,
        Err(InterpreterError::ReduceError(
            "The number of terms in the Par is 32768, which exceeds the limit of 32767".to_string()
        ))
    )
}

// ==============================================================================
// Seq Channel Concurrent Access Rejection Tests
// ==============================================================================
//
// These tests verify that the Seq qualifier correctly prevents concurrent access
// to channels. The implementation uses a guard-based mechanism where:
// 1. Space IDs are registered with SpaceQualifier::Seq in space_qualifier_map
// 2. Channels are mapped to spaces in channel_space_map
// 3. Active operations insert channel IDs into seq_channel_guards
// 4. Concurrent access attempts are rejected with SeqChannelConcurrencyError
//
// Formal Correspondence:
// - Safety/Properties.v:161-167 (seq_implies_not_concurrent)
// - GenericRSpace.v:1330-1335 (single_accessor_invariant)
// ==============================================================================

#[tokio::test]
async fn seq_channel_concurrent_access_rejected() {
    // Create an interpreter with a Seq-qualified space
    let cost = CostAccounting::empty_cost();
    cost.set(Cost::unsafe_max());

    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.expect("should create store");
    let space = RSpace::create(store, Arc::new(Box::new(Matcher))).expect("should create space");
    let rspace: RhoISpace = Arc::new(tokio::sync::Mutex::new(Box::new(space)));

    let reducer = DebruijnInterpreter::new(
        rspace,
        Arc::new(HashMap::new()),
        Arc::new(std::sync::RwLock::new(HashSet::new())),
        Par::default(),
        cost.clone(),
    );

    // Set up a Seq-qualified space (using DashMap for lock-free access)
    let seq_space_id: Vec<u8> = vec![1, 2, 3, 4];
    reducer.space_qualifier_map.insert(seq_space_id.clone(), SpaceQualifier::Seq);

    // Create a channel and map it to the Seq space
    let channel_id: Vec<u8> = vec![10, 20, 30];
    let channel = Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
            id: channel_id.clone(),
        })),
    }]);
    reducer.channel_space_map.insert(channel_id.clone(), seq_space_id.clone());

    // Pre-populate guards to simulate an active operation on this channel (using DashSet)
    reducer.seq_channel_guards.insert(channel_id.clone());

    // Attempt to consume from this channel - should fail with concurrency error
    let receive = Par::default().with_receives(vec![Receive {
        binds: vec![ReceiveBind {
            patterns: vec![Par::default()], // Simple wildcard pattern
            source: Some(channel.clone()),
            remainder: None,
            free_count: 0,
            pattern_modifiers: vec![],
        }],
        body: Some(Par::default()),
        persistent: false,
        peek: false,
        bind_count: 0,
        locally_free: vec![],
        connective_used: false,
    }]);

    let env = Env::new();
    let result = reducer.eval(receive, &env, rand()).await;

    assert!(result.is_err(), "consume should fail on guarded Seq channel");
    match result {
        Err(InterpreterError::SeqChannelConcurrencyError { channel_description }) => {
            assert!(
                channel_description.contains("GPrivate"),
                "error should describe the channel: got {:?}",
                channel_description
            );
        }
        Err(other) => panic!("expected SeqChannelConcurrencyError, got {:?}", other),
        Ok(_) => panic!("expected error but got Ok"),
    }
}

#[tokio::test]
async fn seq_channel_sequential_access_succeeds() {
    // Create an interpreter with a Seq-qualified space
    let cost = CostAccounting::empty_cost();
    cost.set(Cost::unsafe_max());

    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.expect("should create store");
    let space = RSpace::create(store, Arc::new(Box::new(Matcher))).expect("should create space");
    let rspace: RhoISpace = Arc::new(tokio::sync::Mutex::new(Box::new(space)));

    let reducer = DebruijnInterpreter::new(
        rspace,
        Arc::new(HashMap::new()),
        Arc::new(std::sync::RwLock::new(HashSet::new())),
        Par::default(),
        cost.clone(),
    );

    // Set up a Seq-qualified space (using DashMap for lock-free access)
    let seq_space_id: Vec<u8> = vec![1, 2, 3, 4];
    reducer.space_qualifier_map.insert(seq_space_id.clone(), SpaceQualifier::Seq);

    // Create a channel and map it to the Seq space
    let channel_id: Vec<u8> = vec![10, 20, 30];
    let channel = Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
            id: channel_id.clone(),
        })),
    }]);
    reducer.channel_space_map.insert(channel_id.clone(), seq_space_id.clone());

    // NOTE: Do NOT pre-populate guards - the channel is NOT currently accessed

    // Attempt to consume from this channel - should succeed (no concurrent access)
    let receive = Par::default().with_receives(vec![Receive {
        binds: vec![ReceiveBind {
            patterns: vec![Par::default()],
            source: Some(channel.clone()),
            remainder: None,
            free_count: 0,
            pattern_modifiers: vec![],
        }],
        body: Some(Par::default()),
        persistent: false,
        peek: false,
        bind_count: 0,
        locally_free: vec![],
        connective_used: false,
    }]);

    let env = Env::new();
    let result = reducer.eval(receive, &env, rand()).await;

    // The consume should succeed (no data, so it waits as a continuation)
    assert!(result.is_ok(), "consume on unguarded Seq channel should succeed: {:?}", result);

    // After the operation completes, the guard should be released (using DashSet)
    assert!(
        !reducer.seq_channel_guards.contains(&channel_id),
        "guard should be released after operation completes"
    );
}

#[tokio::test]
async fn seq_channel_guard_released_after_consume() {
    // This test verifies that guards are properly acquired and released
    let cost = CostAccounting::empty_cost();
    cost.set(Cost::unsafe_max());

    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.expect("should create store");
    let space = RSpace::create(store, Arc::new(Box::new(Matcher))).expect("should create space");
    let rspace: RhoISpace = Arc::new(tokio::sync::Mutex::new(Box::new(space)));

    let reducer = DebruijnInterpreter::new(
        rspace,
        Arc::new(HashMap::new()),
        Arc::new(std::sync::RwLock::new(HashSet::new())),
        Par::default(),
        cost.clone(),
    );

    // Set up a Seq-qualified space (using DashMap for lock-free access)
    let seq_space_id: Vec<u8> = vec![1, 2, 3, 4];
    reducer.space_qualifier_map.insert(seq_space_id.clone(), SpaceQualifier::Seq);

    // Create a channel and map it to the Seq space
    let channel_id: Vec<u8> = vec![10, 20, 30];
    let channel = Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
            id: channel_id.clone(),
        })),
    }]);
    reducer.channel_space_map.insert(channel_id.clone(), seq_space_id.clone());

    // Verify guards are empty initially (using DashSet)
    assert!(reducer.seq_channel_guards.is_empty(), "guards should be empty initially");

    // Execute first consume
    let receive = Par::default().with_receives(vec![Receive {
        binds: vec![ReceiveBind {
            patterns: vec![Par::default()],
            source: Some(channel.clone()),
            remainder: None,
            free_count: 0,
            pattern_modifiers: vec![],
        }],
        body: Some(Par::default()),
        persistent: false,
        peek: false,
        bind_count: 0,
        locally_free: vec![],
        connective_used: false,
    }]);

    let env = Env::new();
    let result1 = reducer.eval(receive.clone(), &env, rand()).await;
    assert!(result1.is_ok(), "first consume should succeed");

    // Guards should be released after the operation (using DashSet)
    assert!(
        !reducer.seq_channel_guards.contains(&channel_id),
        "guard should be released after first consume"
    );

    // Execute second consume - should also succeed since guard was released
    let result2 = reducer.eval(receive, &env, rand()).await;
    assert!(result2.is_ok(), "second sequential consume should also succeed");

    // Guards should still be empty after second operation (using DashSet)
    assert!(
        !reducer.seq_channel_guards.contains(&channel_id),
        "guard should be released after second consume"
    );
}

#[tokio::test]
async fn default_space_allows_concurrent_access() {
    // Verify that Default-qualified spaces (the default) allow concurrent access
    let cost = CostAccounting::empty_cost();
    cost.set(Cost::unsafe_max());

    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.expect("should create store");
    let space = RSpace::create(store, Arc::new(Box::new(Matcher))).expect("should create space");
    let rspace: RhoISpace = Arc::new(tokio::sync::Mutex::new(Box::new(space)));

    let reducer = DebruijnInterpreter::new(
        rspace,
        Arc::new(HashMap::new()),
        Arc::new(std::sync::RwLock::new(HashSet::new())),
        Par::default(),
        cost.clone(),
    );

    // Use the default space (empty space_id) which has Default qualifier
    // No need to set up space_qualifier_map - empty ID defaults to SpaceQualifier::Default

    // Create a channel (will use default space since not in channel_space_map)
    let channel_id: Vec<u8> = vec![10, 20, 30];
    let channel = Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
            id: channel_id.clone(),
        })),
    }]);

    // Pre-populate guards (simulating concurrent access) using DashSet
    // For Default spaces, this should NOT block access
    reducer.seq_channel_guards.insert(channel_id.clone());

    // Attempt to consume - should succeed even with "guard" because Default space is concurrent
    let receive = Par::default().with_receives(vec![Receive {
        binds: vec![ReceiveBind {
            patterns: vec![Par::default()],
            source: Some(channel.clone()),
            remainder: None,
            free_count: 0,
            pattern_modifiers: vec![],
        }],
        body: Some(Par::default()),
        persistent: false,
        peek: false,
        bind_count: 0,
        locally_free: vec![],
        connective_used: false,
    }]);

    let env = Env::new();
    let result = reducer.eval(receive, &env, rand()).await;

    // Default space allows concurrent access, so this should succeed
    assert!(
        result.is_ok(),
        "Default space should allow concurrent access: {:?}",
        result
    );
}

#[tokio::test]
async fn temp_space_allows_concurrent_access() {
    // Verify that Temp-qualified spaces allow concurrent access (like Default)
    let cost = CostAccounting::empty_cost();
    cost.set(Cost::unsafe_max());

    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.expect("should create store");
    let space = RSpace::create(store, Arc::new(Box::new(Matcher))).expect("should create space");
    let rspace: RhoISpace = Arc::new(tokio::sync::Mutex::new(Box::new(space)));

    let reducer = DebruijnInterpreter::new(
        rspace,
        Arc::new(HashMap::new()),
        Arc::new(std::sync::RwLock::new(HashSet::new())),
        Par::default(),
        cost.clone(),
    );

    // Set up a Temp-qualified space (concurrent, non-persistent, mobile)
    let temp_space_id: Vec<u8> = vec![5, 6, 7, 8];
    reducer.space_qualifier_map.insert(temp_space_id.clone(), SpaceQualifier::Temp);

    // Create a channel and map it to the Temp space
    let channel_id: Vec<u8> = vec![10, 20, 30];
    let channel = Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
            id: channel_id.clone(),
        })),
    }]);
    reducer.channel_space_map.insert(channel_id.clone(), temp_space_id.clone());

    // Pre-populate guards (simulating concurrent access)
    // For Temp spaces, this should NOT block access
    reducer.seq_channel_guards.insert(channel_id.clone());

    // Attempt to consume - should succeed because Temp space is concurrent
    let receive = Par::default().with_receives(vec![Receive {
        binds: vec![ReceiveBind {
            patterns: vec![Par::default()],
            source: Some(channel.clone()),
            remainder: None,
            free_count: 0,
            pattern_modifiers: vec![],
        }],
        body: Some(Par::default()),
        persistent: false,
        peek: false,
        bind_count: 0,
        locally_free: vec![],
        connective_used: false,
    }]);

    let env = Env::new();
    let result = reducer.eval(receive, &env, rand()).await;

    // Temp space allows concurrent access, so this should succeed
    assert!(
        result.is_ok(),
        "Temp space should allow concurrent access: {:?}",
        result
    );
}

// ==============================================================================
// Seq Channel Mobility Tests
// ==============================================================================
//
// These tests verify that the Seq qualifier correctly prevents channels from
// being sent to other processes (non-mobility constraint). The implementation
// extracts all GPrivate channel IDs from data being sent and verifies none
// belong to a Seq-qualified space.
//
// Formal Correspondence:
// - Safety/Properties.v:161-167 (seq_cannot_be_sent theorem)
// - GenericRSpace.v:1203-1212 (seq_implies_not_mobile theorem)
// ==============================================================================

#[tokio::test]
async fn seq_channel_direct_send_rejected() {
    // Create an interpreter with a Seq-qualified space
    let cost = CostAccounting::empty_cost();
    cost.set(Cost::unsafe_max());

    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.expect("should create store");
    let space = RSpace::create(store, Arc::new(Box::new(Matcher))).expect("should create space");
    let rspace: RhoISpace = Arc::new(tokio::sync::Mutex::new(Box::new(space)));

    let reducer = DebruijnInterpreter::new(
        rspace,
        Arc::new(HashMap::new()),
        Arc::new(std::sync::RwLock::new(HashSet::new())),
        Par::default(),
        cost.clone(),
    );

    // Set up a Seq-qualified space
    let seq_space_id: Vec<u8> = vec![1, 2, 3, 4];
    reducer.space_qualifier_map.insert(seq_space_id.clone(), SpaceQualifier::Seq);

    // Create a seq channel and map it to the Seq space
    let seq_channel_id: Vec<u8> = vec![10, 20, 30];
    reducer.channel_space_map.insert(seq_channel_id.clone(), seq_space_id.clone());

    // Create a target channel (in default space, so it's fine to send to)
    let target_channel_id: Vec<u8> = vec![40, 50, 60];
    let target_channel = Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
            id: target_channel_id.clone(),
        })),
    }]);

    // Create the seq channel as data to be sent
    let seq_channel_data = Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
            id: seq_channel_id.clone(),
        })),
    }]);

    // Create a send that tries to send the seq channel
    let send = Par::default().with_sends(vec![Send {
        chan: Some(target_channel.clone()),
        data: vec![seq_channel_data], // Trying to send seq channel as data
        persistent: false,
        locally_free: vec![],
        connective_used: false,
        hyperparams: vec![],
    }]);

    let env = Env::new();
    let result = reducer.eval(send, &env, rand()).await;

    assert!(result.is_err(), "send should fail when trying to send a seq channel");
    match result {
        Err(InterpreterError::SeqChannelMobilityError { channel_description }) => {
            assert!(
                channel_description.contains("non-mobile"),
                "error should mention non-mobility: got {:?}",
                channel_description
            );
        }
        Err(other) => panic!("expected SeqChannelMobilityError, got {:?}", other),
        Ok(_) => panic!("expected error but got Ok"),
    }
}

#[tokio::test]
async fn seq_channel_in_tuple_rejected() {
    // Create an interpreter with a Seq-qualified space
    let cost = CostAccounting::empty_cost();
    cost.set(Cost::unsafe_max());

    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.expect("should create store");
    let space = RSpace::create(store, Arc::new(Box::new(Matcher))).expect("should create space");
    let rspace: RhoISpace = Arc::new(tokio::sync::Mutex::new(Box::new(space)));

    let reducer = DebruijnInterpreter::new(
        rspace,
        Arc::new(HashMap::new()),
        Arc::new(std::sync::RwLock::new(HashSet::new())),
        Par::default(),
        cost.clone(),
    );

    // Set up a Seq-qualified space
    let seq_space_id: Vec<u8> = vec![1, 2, 3, 4];
    reducer.space_qualifier_map.insert(seq_space_id.clone(), SpaceQualifier::Seq);

    // Create a seq channel and map it to the Seq space
    let seq_channel_id: Vec<u8> = vec![10, 20, 30];
    reducer.channel_space_map.insert(seq_channel_id.clone(), seq_space_id.clone());

    // Create a target channel
    let target_channel_id: Vec<u8> = vec![40, 50, 60];
    let target_channel = Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
            id: target_channel_id.clone(),
        })),
    }]);

    // Create the seq channel as part of a tuple
    let seq_channel_par = Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
            id: seq_channel_id.clone(),
        })),
    }]);

    // Create a tuple containing the seq channel
    let tuple_data = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(models::rhoapi::expr::ExprInstance::ETupleBody(ETuple {
            ps: vec![
                seq_channel_par, // Seq channel hidden in tuple
                Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(models::rhoapi::expr::ExprInstance::GInt(42)),
                }]),
            ],
            locally_free: vec![],
            connective_used: false,
        })),
    }]);

    // Create a send that tries to send the tuple containing seq channel
    let send = Par::default().with_sends(vec![Send {
        chan: Some(target_channel.clone()),
        data: vec![tuple_data], // Tuple containing seq channel
        persistent: false,
        locally_free: vec![],
        connective_used: false,
        hyperparams: vec![],
    }]);

    let env = Env::new();
    let result = reducer.eval(send, &env, rand()).await;

    assert!(result.is_err(), "send should fail when trying to send a tuple containing seq channel");
    match result {
        Err(InterpreterError::SeqChannelMobilityError { .. }) => {
            // Expected error
        }
        Err(other) => panic!("expected SeqChannelMobilityError, got {:?}", other),
        Ok(_) => panic!("expected error but got Ok"),
    }
}

#[tokio::test]
async fn default_channel_can_be_sent() {
    // Verify that non-seq channels can still be sent normally
    let cost = CostAccounting::empty_cost();
    cost.set(Cost::unsafe_max());

    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.expect("should create store");
    let space = RSpace::create(store, Arc::new(Box::new(Matcher))).expect("should create space");
    let rspace: RhoISpace = Arc::new(tokio::sync::Mutex::new(Box::new(space)));

    let reducer = DebruijnInterpreter::new(
        rspace,
        Arc::new(HashMap::new()),
        Arc::new(std::sync::RwLock::new(HashSet::new())),
        Par::default(),
        cost.clone(),
    );

    // Set up a Default-qualified space
    let default_space_id: Vec<u8> = vec![1, 2, 3, 4];
    reducer.space_qualifier_map.insert(default_space_id.clone(), SpaceQualifier::Default);

    // Create a channel in the Default space (mobile)
    let mobile_channel_id: Vec<u8> = vec![10, 20, 30];
    reducer.channel_space_map.insert(mobile_channel_id.clone(), default_space_id.clone());

    // Create a target channel
    let target_channel_id: Vec<u8> = vec![40, 50, 60];
    let target_channel = Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
            id: target_channel_id.clone(),
        })),
    }]);

    // Create the mobile channel as data to be sent
    let mobile_channel_data = Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
            id: mobile_channel_id.clone(),
        })),
    }]);

    // Create a send that sends the mobile channel (should succeed)
    let send = Par::default().with_sends(vec![Send {
        chan: Some(target_channel.clone()),
        data: vec![mobile_channel_data],
        persistent: false,
        locally_free: vec![],
        connective_used: false,
        hyperparams: vec![],
    }]);

    let env = Env::new();
    let result = reducer.eval(send, &env, rand()).await;

    assert!(
        result.is_ok(),
        "Default-qualified channels should be sendable: {:?}",
        result
    );
}
