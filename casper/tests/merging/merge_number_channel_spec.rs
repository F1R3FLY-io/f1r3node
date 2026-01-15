// See casper/src/test/scala/coop/rchain/casper/merging/MergeNumberChannelSpec.scala

use futures::future::join_all;
use std::collections::HashMap;

use casper::rust::{
    merging::{
        block_index, conflict_set_merger, dag_merger, deploy_chain_index::DeployChainIndex,
        deploy_index::DeployIndex,
    },
    rholang::runtime::RuntimeOps,
    util::{event_converter, rholang::runtime_manager::RuntimeManager},
};
use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::{
    g_unforgeable::UnfInstance, BindPattern, GPrivate, GUnforgeable, ListParWithRandom, Par,
    TaggedContinuation,
};
use rholang::rust::interpreter::{
    accounting::costs::Cost,
    merging::rholang_merging_logic::RholangMergingLogic,
    rho_runtime::{RhoRuntime, RhoRuntimeImpl},
    rho_type::RhoNumber,
};
use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash,
    hot_store_trie_action::HotStoreTrieAction,
    merger::{
        channel_change::ChannelChange,
        event_log_index::EventLogIndex,
        merging_logic::{self, NumberChannelsDiff},
        state_change::StateChange,
        state_change_merger,
    },
};
use shared::rust::hashable_set::HashableSet;

use casper::rust::test_utils::util::rholang::resources::mk_runtime_manager;

#[derive(Debug)]
struct DeployTestInfo {
    term: String,
    cost: u64,
    sig: String,
}

static RHO_ST: &str = r#"
new MergeableTag, stCh  in {
  @(*MergeableTag, *stCh)!(0) |

  contract @"SET"(ret, @v) = {
    for(@s <- @(*MergeableTag, *stCh)) {
      @(*MergeableTag, *stCh)!(s + v) | ret!(s, s + v)
    }
  } |

  contract @"READ"(ret) = {
    for(@s <<- @(*MergeableTag, *stCh)) {
      ret!(s)
    }
  }
}
"#;

fn rho_change(num: i64) -> String {
    format!(
        r#"
new retCh, out(`rho:io:stdout`) in {{
  out!(("Begin change", {})) |
  @"SET"!(*retCh, {}) |
  for(@old, @new_ <- retCh) {{
    out!(("Changed", old, "=>", new_))
  }}
}}
"#,
        num, num
    )
}

static RHO_EXPLORE_READ: &str = r#"
new return in {
  @"READ"!(*return)
}
"#;

fn par_rho(ori: &str, append_rho: &str) -> String {
    format!("{}|{}", ori, append_rho)
}

fn make_sig(hex: &str) -> Vec<u8> {
    let hex_str = if hex.starts_with("0x") {
        &hex[2..]
    } else {
        hex
    };
    hex::decode(hex_str).unwrap()
}

fn make_sig_pb(hex: &str) -> prost::bytes::Bytes {
    prost::bytes::Bytes::from(make_sig(hex))
}

fn base_rho_seed() -> Blake2b512Random {
    let bytes: [u8; 128] = [1; 128];
    Blake2b512Random::create_from_bytes(&bytes)
}

fn unforgeable_name_seed() -> Par {
    Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
            id: base_rho_seed()
                .next()
                .into_iter()
                .map(|b| b as u8)
                .collect(),
        })),
    }])
}

async fn test_case(
    base_terms: Vec<String>,
    left_terms: Vec<DeployTestInfo>,
    right_terms: Vec<DeployTestInfo>,
    expected_rejected: HashableSet<prost::bytes::Bytes>,
    expected_final_result: i64,
) {
    let rm = mk_runtime_manager("merging-test", Some(unforgeable_name_seed())).await;
    let mut runtime = rm.spawn_runtime().await;

    async fn run_rholang(
        runtime: &mut RhoRuntimeImpl,
        rm: &RuntimeManager,
        terms: Vec<DeployTestInfo>,
        pre_state: Blake2b256Hash,
    ) -> (HashableSet<DeployIndex>, Blake2b256Hash) {
        runtime.reset(&pre_state);

        let futures = terms
            .iter()
            .map(|deploy| {
                let term = deploy.term.clone();
                let mut runtime = runtime.clone();
                async move {
                    let runtime_ops = RuntimeOps::new(runtime.clone());
                    let eval_result = runtime.evaluate_with_term(&term).await.unwrap();
                    assert!(
                        eval_result.errors.is_empty(),
                        "{:?}\n{}",
                        eval_result.errors,
                        term
                    );

                    let num_chan_final = runtime_ops
                        .get_number_channels_data(&eval_result.mergeable)
                        .unwrap();

                    let soft_point = runtime.create_soft_checkpoint();
                    (soft_point, num_chan_final)
                }
            })
            .collect::<Vec<_>>();

        let eval_results = join_all(futures).await;
        let end_checkpoint = runtime.create_checkpoint();
        let (log_vec, num_chan_abs) = eval_results.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
        let num_chan_diffs = rm.convert_number_channels_to_diff(num_chan_abs, &pre_state);

        let event_log_indices: Vec<DeployIndex> = log_vec
            .iter()
            .zip(num_chan_diffs)
            .zip(terms)
            .map(|((cp, number_chan_diff), deploy)| {
                let event_log_index = block_index::create_event_log_index(
                    cp.log
                        .iter()
                        .map(|event: &rspace_plus_plus::rspace::trace::event::Event| {
                            event_converter::to_casper_event(event.clone())
                        })
                        .collect(),
                    rm.get_history_repo(),
                    &pre_state,
                    number_chan_diff,
                );

                let sig_bs = make_sig_pb(deploy.sig.as_str());
                DeployIndex {
                    deploy_id: sig_bs,
                    cost: deploy.cost,
                    event_log_index,
                }
            })
            .collect();

        (
            event_log_indices.into_iter().collect::<HashableSet<_>>(),
            end_checkpoint.root,
        )
    }

    let history_repo = rm.get_history_repo();

    let futures = base_terms
        .iter()
        .enumerate()
        .map(|(i, term)| {
            let term = term.clone();
            let runtime_clone = runtime.clone();

            async move {
                let base_res = runtime_clone
                    .evaluate(&term, Cost::unsafe_max(), HashMap::new(), base_rho_seed())
                    .await
                    .unwrap();

                assert!(
                    base_res.errors.is_empty(),
                    "BASE {} {:?}",
                    i,
                    base_res.errors
                );
            }
        })
        .collect::<Vec<_>>();

    join_all(futures).await;
    let base_cp = runtime.create_checkpoint();

    let (left_ev_indices, left_post_state) =
        run_rholang(&mut runtime, &rm, left_terms, base_cp.root.clone()).await;
    let left_deploy_indices = merging_logic::compute_related_sets(
        &left_ev_indices,
        |x: &DeployIndex, y: &DeployIndex| {
            merging_logic::depends(&x.event_log_index, &y.event_log_index)
        },
    );

    let (right_ev_indices, right_post_state) =
        run_rholang(&mut runtime, &rm, right_terms, base_cp.root.clone()).await;
    let right_deploy_indices = merging_logic::compute_related_sets(
        &right_ev_indices,
        |x: &DeployIndex, y: &DeployIndex| {
            merging_logic::depends(&x.event_log_index, &y.event_log_index)
        },
    );

    let left_deploy_chains = left_deploy_indices
        .0
        .iter()
        .map(|deploy_index| {
            DeployChainIndex::new(
                deploy_index,
                &base_cp.root,
                &left_post_state,
                history_repo.clone(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    let right_deploy_chains = right_deploy_indices
        .0
        .iter()
        .map(|deploy_index| {
            DeployChainIndex::new(
                deploy_index,
                &base_cp.root,
                &right_post_state,
                history_repo.clone(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    println!("LEFT DEPLOY CHAINS: {:?}", left_deploy_chains.len());
    println!("RIGHT DEPLOY CHAINS: {:?}", right_deploy_chains.len());

    let branches_are_conflicting =
        |a: &HashableSet<DeployChainIndex>, b: &HashableSet<DeployChainIndex>| {
            merging_logic::are_conflicting(
                &a.0.iter()
                    .map(|x| &x.event_log_index)
                    .fold(EventLogIndex::empty(), |acc, x| {
                        EventLogIndex::combine(&acc, x)
                    }),
                &b.0.iter()
                    .map(|x| &x.event_log_index)
                    .fold(EventLogIndex::empty(), |acc, x| {
                        EventLogIndex::combine(&acc, x)
                    }),
            )
        };

    let base_reader = history_repo.get_history_reader(&base_cp.root).unwrap();

    let override_trie_action =
        |hash: &Blake2b256Hash,
         changes: &ChannelChange<Vec<u8>>,
         number_channels: &NumberChannelsDiff| {
            match number_channels.get(&hash) {
                Some(number_channel_diff) => {
                    Ok(Some(RholangMergingLogic::calculate_number_channel_merge(
                        hash,
                        *number_channel_diff,
                        changes,
                        |_hash| base_reader.get_data(_hash),
                    )))
                }
                None => Ok(None),
            }
        };

    let compute_trie_actions = |changes: StateChange, mergeable_chs: NumberChannelsDiff| {
        state_change_merger::compute_trie_actions(
            &changes,
            &base_reader,
            &mergeable_chs,
            |_hash, _changes, _number_channels| {
                override_trie_action(_hash, _changes, _number_channels)
            },
        )
    };

    let apply_trie_actions = |actions: Vec<
        HotStoreTrieAction<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
    >| {
        rm.get_history_repo().reset(&base_cp.root).map(|r1| {
            let r2 = r1.do_checkpoint(actions);
            r2.root()
        })
    };

    let mut actual_set: HashableSet<DeployChainIndex> = left_deploy_chains.into_iter().collect();
    actual_set
        .0
        .extend(right_deploy_chains.into_iter().collect::<HashableSet<_>>());

    let (final_hash, rejected) = conflict_set_merger::merge(
        actual_set,
        HashableSet::new(),
        |target, source| merging_logic::depends(&target.event_log_index, &source.event_log_index),
        |arg0: &HashableSet<DeployChainIndex>, arg1: &HashableSet<DeployChainIndex>| {
            branches_are_conflicting(arg0, arg1)
        },
        dag_merger::cost_optimal_rejection_alg(),
        |r| Ok(r.state_changes.clone()),
        |r| r.event_log_index.number_channels_data.clone(),
        compute_trie_actions,
        apply_trie_actions,
        |x| base_reader.get_data(&x),
    )
    .unwrap();

    let rejected_sigs: HashableSet<prost::bytes::Bytes> = rejected
        .0
        .iter()
        .flat_map(|r| r.deploys_with_cost.0.iter().map(|d| d.deploy_id.clone()))
        .collect();

    assert_eq!(rejected_sigs, expected_rejected);

    let mut runtime_ops = RuntimeOps::new(runtime);
    let res = runtime_ops
        .play_exploratory_deploy(RHO_EXPLORE_READ.to_owned(), &final_hash.to_bytes_prost())
        .await
        .unwrap();

    assert_eq!(RhoNumber::unapply(&res[0]).unwrap(), expected_final_result);
}

#[tokio::test]
async fn multiple_branches_should_reject_deploy_when_mergeable_number_channels_got_negative_number()
{
    test_case(
        vec![RHO_ST.to_owned(), rho_change(10)],
        vec![DeployTestInfo {
            term: rho_change(-5),
            cost: 10,
            sig: "0x11".to_string(),
        }],
        vec![DeployTestInfo {
            term: rho_change(-6),
            cost: 10,
            sig: "0x22".to_string(),
        }],
        HashableSet::from_iter(vec![make_sig_pb("0x22")]),
        5,
    )
    .await;
}

#[tokio::test]
async fn multiple_branches_should_reject_deploy_when_mergeable_number_channels_got_overflow() {
    test_case(
        vec![RHO_ST.to_owned(), rho_change(10)],
        vec![DeployTestInfo {
            term: rho_change(-5),
            cost: 10,
            sig: "0x11".to_string(),
        }],
        vec![DeployTestInfo {
            term: rho_change(9223372036854775806),
            cost: 10,
            sig: "0x22".to_string(),
        }],
        HashableSet::from_iter(vec![make_sig_pb("0x22")]),
        5,
    )
    .await;
}

#[tokio::test]
async fn multiple_branches_with_normal_rejection_should_choose_from_normal_reject_options() {
    test_case(
        vec![RHO_ST.to_owned(), rho_change(100)],
        vec![
            DeployTestInfo {
                term: par_rho(rho_change(-20).as_str(), "@\"X\"!(1)"),
                cost: 10,
                sig: "0x11".to_string(),
            },
            DeployTestInfo {
                term: rho_change(-10),
                cost: 10,
                sig: "0x12".to_string(),
            },
        ],
        vec![
            DeployTestInfo {
                term: rho_change(-60),
                cost: 10,
                sig: "0x22".to_string(),
            },
            DeployTestInfo {
                term: par_rho(rho_change(-20).as_str(), "for(_ <- @\"X\") {Nil}"),
                cost: 11,
                sig: "0x21".to_string(),
            },
        ],
        HashableSet::from_iter(vec![make_sig_pb("0x11")]),
        10,
    )
    .await;
}

#[tokio::test]
async fn multiple_branches_should_merge_number_channels() {
    test_case(
        vec![RHO_ST.to_owned()],
        vec![
            DeployTestInfo {
                term: rho_change(10),
                cost: 10,
                sig: "0x10".to_string(),
            },
            DeployTestInfo {
                term: rho_change(-5),
                cost: 10,
                sig: "0x11".to_string(),
            },
        ],
        vec![
            DeployTestInfo {
                term: rho_change(15),
                cost: 10,
                sig: "0x20".to_string(),
            },
            DeployTestInfo {
                term: rho_change(10),
                cost: 10,
                sig: "0x21".to_string(),
            },
            DeployTestInfo {
                term: rho_change(-20),
                cost: 10,
                sig: "0x22".to_string(),
            },
        ],
        HashableSet::new(),
        10,
    )
    .await;
}
