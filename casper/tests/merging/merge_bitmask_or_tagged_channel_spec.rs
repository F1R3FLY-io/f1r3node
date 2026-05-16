// Direct-merge code-level reproductions of the integration-suite
// `test_shard_degradation` wedge: two siblings each write to the SAME
// BitmaskOr-tagged channel whose end-of-deploy value is NON-numeric (a
// Map rather than an Int). The numeric-channel path can't represent the
// diff, so the channel falls out of `number_channels_data` and the merger
// drops to `make_trie_action`'s multiset-union path. Without PR #520
// check #4 firing, the merged state ends up with TWO datums on a tagged
// channel that is required to hold one.
//
// The merge wiring mirrors `merge_number_channel_spec::test_case` exactly
// so any difference in outcome is attributable to (a) the merge_type
// (BitmaskOr vs IntegerAdd) and (b) the final-value shape (Map vs Int) and
// (c) the sibling contract's read pattern (linear-consume vs peek-then-...).
//
// `run_direct_merge` reports per-deploy `EventLogIndex` shape, the merge's
// rejection set, and the final state's datum count on the wedge channel.
// Tests assert on the OUTCOME (rejected vs multi-Datum) and on each
// upstream layer so failures bisect the actual gap.

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
};
use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash,
    hot_store_trie_action::HotStoreTrieAction,
    merger::{
        channel_change::ChannelChange,
        event_log_index::EventLogIndex,
        merging_logic::{self, MergeType, NumberChannelsDiff},
        state_change::StateChange,
        state_change_merger,
    },
};
use shared::rust::hashable_set::HashableSet;

use crate::util::rholang::resources::mk_runtime_manager;

fn base_rho_seed() -> Blake2b512Random {
    let bytes: [u8; 128] = [2; 128];
    Blake2b512Random::create_from_bytes(&bytes)
}

/// Derives the unforgeable Par that the BASE term's `new MergeableTag`
/// will create. Registration into the runtime's `mergeable_tags` map keys
/// off this Par.
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

fn make_sig_pb(hex: &str) -> prost::bytes::Bytes {
    let h = if hex.starts_with("0x") { &hex[2..] } else { hex };
    prost::bytes::Bytes::from(hex::decode(h).unwrap())
}

#[derive(Debug, Clone)]
struct SiblingInfo {
    term: String,
    cost: u64,
    sig: String,
}

#[derive(Debug)]
struct MergeOutcome {
    rejected_sigs: Vec<String>,
    post_merge_datum_count: usize,
    /// Per-sibling EventLogIndex shape, printed for diagnostic bisection.
    per_sibling: Vec<EventLogShape>,
    wedge_channel: Option<Blake2b256Hash>,
}

#[derive(Debug)]
struct EventLogShape {
    sig: String,
    identity_tagged: usize,
    number_channels_data: usize,
    produces_linear: usize,
    produces_persistent: usize,
    produces_consumed: usize,
    produces_copied_by_peek: usize,
    produces_mergeable: usize,
    /// Surviving produces (linear ∪ persistent) minus consumed minus
    /// copied_by_peek, restricted to the wedge channel.
    surviving_on_wedge: usize,
    /// Of those surviving produces on the wedge channel, how many are in
    /// `produces_mergeable`. If zero, check #4 should fire.
    surviving_on_wedge_in_mergeable: usize,
}

async fn run_direct_merge(
    label: &str,
    base_rho: &str,
    siblings: Vec<SiblingInfo>,
) -> MergeOutcome {
    eprintln!("\n========== {} ==========", label);

    let mergeable_tags = {
        let mut m = HashMap::new();
        m.insert(unforgeable_name_seed(), MergeType::BitmaskOr);
        std::sync::Arc::new(m)
    };
    let rm = mk_runtime_manager("merge-bitmask-or-test", Some(mergeable_tags)).await;
    let mut runtime = rm.spawn_runtime().await;

    // -------- Evaluate BASE --------
    let base_res = runtime
        .evaluate(base_rho, Cost::unsafe_max(), HashMap::new(), base_rho_seed())
        .await
        .unwrap();
    assert!(
        base_res.errors.is_empty(),
        "BASE eval errors: {:?}",
        base_res.errors
    );
    let base_cp = runtime.create_checkpoint().await;

    async fn run_sibling(
        runtime: &mut RhoRuntimeImpl,
        rm: &RuntimeManager,
        sib: &SiblingInfo,
        pre_state: &Blake2b256Hash,
    ) -> (DeployIndex, Blake2b256Hash) {
        runtime.reset(pre_state).await.expect("reset to pre-state");

        let runtime_ops = RuntimeOps::new(runtime.clone());
        let eval_result = runtime.evaluate_with_term(&sib.term).await.unwrap();
        assert!(
            eval_result.errors.is_empty(),
            "sibling {} eval errors: {:?}",
            sib.sig,
            eval_result.errors
        );

        let num_chan_final = runtime_ops
            .get_number_channels_data(&eval_result.mergeable)
            .await
            .unwrap();

        let soft_cp = runtime.create_soft_checkpoint().await;
        let end_cp = runtime.create_checkpoint().await;

        let num_chan_diff = rm
            .convert_number_channels_to_diff(vec![num_chan_final], pre_state)
            .expect("convert_number_channels_to_diff");
        let num_chan_diff = num_chan_diff.into_iter().next().unwrap();

        let casper_events = soft_cp
            .log
            .iter()
            .map(|event| event_converter::to_casper_event(event.clone()))
            .collect::<Vec<_>>();
        let event_log_index = block_index::create_event_log_index(
            &casper_events,
            rm.get_history_repo(),
            pre_state,
            num_chan_diff,
        );

        let deploy_index = DeployIndex {
            deploy_id: make_sig_pb(&sib.sig),
            cost: sib.cost,
            event_log_index,
        };
        (deploy_index, end_cp.root)
    }

    let mut deploy_indices: Vec<(DeployIndex, Blake2b256Hash)> = Vec::new();
    for sib in &siblings {
        deploy_indices.push(run_sibling(&mut runtime, &rm, sib, &base_cp.root).await);
    }

    // Discover the wedge channel = identity-tagged channel touched by ALL
    // siblings (intersection of their identity_tagged sets).
    let wedge_channel: Option<Blake2b256Hash> = {
        let mut iter = deploy_indices
            .iter()
            .map(|(idx, _)| &idx.event_log_index.identity_tagged_channels.0);
        let first = iter.next().cloned();
        first.and_then(|mut acc| {
            for s in iter {
                acc = acc.intersection(s).cloned().collect();
            }
            acc.into_iter().next()
        })
    };
    if let Some(h) = &wedge_channel {
        eprintln!("WEDGE CHANNEL: {}", hex::encode(h.bytes()));
    } else {
        eprintln!("WEDGE CHANNEL: none — no shared identity-tagged channel");
    }

    // Per-sibling shape diagnostics
    let mut per_sibling: Vec<EventLogShape> = Vec::new();
    for ((idx, _), sib) in deploy_indices.iter().zip(siblings.iter()) {
        let eli = &idx.event_log_index;
        let surviving: Vec<_> = if let Some(wc) = &wedge_channel {
            eli.produces_linear
                .0
                .iter()
                .chain(eli.produces_persistent.0.iter())
                .filter(|p| p.channel_hash == *wc)
                .filter(|p| !eli.produces_consumed.0.contains(p))
                .filter(|p| !eli.produces_copied_by_peek.0.contains(p))
                .collect()
        } else {
            Vec::new()
        };
        let surviving_in_mergeable = surviving
            .iter()
            .filter(|p| eli.produces_mergeable.0.contains(*p))
            .count();
        let shape = EventLogShape {
            sig: sib.sig.clone(),
            identity_tagged: eli.identity_tagged_channels.0.len(),
            number_channels_data: eli.number_channels_data.len(),
            produces_linear: eli.produces_linear.0.len(),
            produces_persistent: eli.produces_persistent.0.len(),
            produces_consumed: eli.produces_consumed.0.len(),
            produces_copied_by_peek: eli.produces_copied_by_peek.0.len(),
            produces_mergeable: eli.produces_mergeable.0.len(),
            surviving_on_wedge: surviving.len(),
            surviving_on_wedge_in_mergeable: surviving_in_mergeable,
        };
        eprintln!(
            "  {} EventLogIndex: identity_tagged={} number_channels_data={} \
             produces_linear={} produces_persistent={} produces_consumed={} \
             produces_copied_by_peek={} produces_mergeable={} \
             surviving_on_wedge={} surviving_on_wedge_in_mergeable={}",
            shape.sig,
            shape.identity_tagged,
            shape.number_channels_data,
            shape.produces_linear,
            shape.produces_persistent,
            shape.produces_consumed,
            shape.produces_copied_by_peek,
            shape.produces_mergeable,
            shape.surviving_on_wedge,
            shape.surviving_on_wedge_in_mergeable,
        );
        per_sibling.push(shape);
    }

    // -------- Build branch deploy chains and drive conflict_set_merger --
    let history_repo = rm.get_history_repo();
    let mut deploy_chains: Vec<DeployChainIndex> = Vec::new();
    for (i, (idx, post)) in deploy_indices.iter().enumerate() {
        let block_hash_marker = prost::bytes::Bytes::from(vec![0xA0u8 + i as u8; 32]);
        let chain = DeployChainIndex::new(
            &HashableSet::from_iter(vec![idx.clone()]),
            &base_cp.root,
            post,
            history_repo.clone(),
            block_hash_marker,
            (i + 1) as i64,
        )
        .unwrap();
        deploy_chains.push(chain);
    }

    let base_reader = history_repo.get_history_reader(&base_cp.root).unwrap();
    let override_trie_action =
        |hash: &Blake2b256Hash,
         changes: &ChannelChange<Vec<u8>>,
         number_channels: &NumberChannelsDiff| {
            match number_channels.get(&hash) {
                Some(number_channel_diff) => {
                    let (diff, merge_type) = *number_channel_diff;
                    Ok(Some(RholangMergingLogic::calculate_number_channel_merge(
                        hash,
                        diff,
                        merge_type,
                        changes,
                        |_hash| base_reader.get_data(_hash),
                    )?))
                }
                None => Ok(None),
            }
        };
    let compute_trie_actions =
        |changes: StateChange, mergeable_chs: NumberChannelsDiff| {
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

    let mut actual_seq: Vec<DeployChainIndex> = deploy_chains;
    actual_seq.sort();

    let (final_hash, rejected) = conflict_set_merger::merge(
        actual_seq,
        Vec::new(),
        |target, source| {
            merging_logic::depends(&target.event_log_index, &source.event_log_index)
        },
        dag_merger::cost_optimal_rejection_alg(),
        |r| Ok(r.state_changes.clone()),
        |r| r.event_log_index.number_channels_data.clone(),
        compute_trie_actions,
        apply_trie_actions,
        |x| base_reader.get_data(&x),
        |merge_set: &HashableSet<DeployChainIndex>| {
            let chains_vec: Vec<DeployChainIndex> = merge_set.0.iter().cloned().collect();
            let event_logs: Vec<&EventLogIndex> =
                chains_vec.iter().map(|c| &c.event_log_index).collect();
            let depends_map =
                merging_logic::compute_depends_map_event_indexed(&chains_vec, &event_logs);
            merging_logic::gather_related_sets(&depends_map)
        },
        |branches_set: &HashableSet<HashableSet<DeployChainIndex>>| {
            let branches_refs: Vec<&HashableSet<DeployChainIndex>> =
                branches_set.0.iter().collect();
            let branches_owned: Vec<HashableSet<DeployChainIndex>> =
                branches_refs.iter().map(|b| (*b).clone()).collect();

            let combined_logs: Vec<EventLogIndex> = branches_refs
                .iter()
                .map(|b| {
                    let logs: Vec<&EventLogIndex> =
                        b.0.iter().map(|chain| &chain.event_log_index).collect();
                    let mut acc = EventLogIndex::empty();
                    for l in logs {
                        acc = EventLogIndex::combine(&acc, l)?;
                    }
                    Ok::<_, rspace_plus_plus::rspace::errors::HistoryError>(acc)
                })
                .collect::<Result<_, _>>()?;
            let event_log_refs: Vec<&EventLogIndex> = combined_logs.iter().collect();

            let result = merging_logic::compute_conflict_map_event_indexed(
                &branches_owned,
                &event_log_refs,
            );
            Ok(result)
        },
    )
    .unwrap();

    let rejected_sigs: Vec<String> = rejected
        .0
        .iter()
        .flat_map(|r| r.deploys_with_cost.0.iter())
        .map(|d| hex::encode(&d.deploy_id))
        .collect();
    let post_merge_datum_count = if let Some(wc) = &wedge_channel {
        let post_merge_reader = history_repo.get_history_reader(&final_hash).unwrap();
        let datums = post_merge_reader.get_data(wc).unwrap();
        eprintln!(
            "MERGE OUTCOME: rejected={} post_merge_datums_on_wedge={}",
            rejected_sigs.len(),
            datums.len()
        );
        for (i, d) in datums.iter().enumerate() {
            let src_bytes = d.source.hash.bytes();
            let src_short = hex::encode(&src_bytes[..std::cmp::min(8, src_bytes.len())]);
            eprintln!(
                "    datum[{}] source_hash={} persistent={}",
                i, src_short, d.persist
            );
        }
        datums.len()
    } else {
        eprintln!("MERGE OUTCOME: rejected={} (no wedge channel)", rejected_sigs.len());
        0
    };

    MergeOutcome {
        rejected_sigs,
        post_merge_datum_count,
        per_sibling,
        wedge_channel,
    }
}

// ============================================================
// Test variants — each isolates a structural element of the CI
// wedge to see which one makes check #4 stop firing.
// ============================================================

/// VARIANT A — the canonical "two siblings linear-consume + produce" case.
/// Already passes (verified in prior session): PR #520 check #4 correctly
/// rejects one sibling, post-merge state has a single datum.
static RHO_BASE_LINEAR_CONSUME: &str = r#"
new MergeableTag, ch in {
  @(*MergeableTag, *ch)!({"v": 0}) |

  contract @"UPDATE"(@bit, ret) = {
    for(@m <- @(*MergeableTag, *ch)) {
      @(*MergeableTag, *ch)!({"v": bit, "prev": m}) |
      ret!(Nil)
    }
  }
}
"#;

/// VARIANT B — peek then linear-consume. Mirrors TreeHashMap.set's
/// double-binding pattern: `<<-` peek (non-destructive observation),
/// then `<-` linear consume. The peek causes the consumed produce to
/// ALSO be in `produces_copied_by_peek`, which is filtered OUT of the
/// surviving-produce predicate at merging_logic.rs:664-678. If the
/// CHANNEL'S new produce ends up filtered out by `produces_copied_by_peek`,
/// check #4 would skip it.
static RHO_BASE_PEEK_THEN_CONSUME: &str = r#"
new MergeableTag, ch in {
  @(*MergeableTag, *ch)!({"v": 0}) |

  contract @"UPDATE"(@bit, ret) = {
    for(@observed <<- @(*MergeableTag, *ch)) {
      for(@m <- @(*MergeableTag, *ch)) {
        @(*MergeableTag, *ch)!({"v": bit, "prev": m, "observed": observed}) |
        ret!(Nil)
      }
    }
  }
}
"#;

/// VARIANT C — peek + persistent-produce style. Persistent produces stay
/// on the channel forever (`!!`); their interaction with tagged channels
/// is different. Useful to bisect: does check #4 fire for persistent
/// produces on a tagged channel?
static RHO_BASE_PERSISTENT_PRODUCE: &str = r#"
new MergeableTag, ch in {
  @(*MergeableTag, *ch)!!({"v": 0}) |

  contract @"UPDATE_PERSISTENT"(@bit, ret) = {
    @(*MergeableTag, *ch)!!({"v": bit}) |
    ret!(Nil)
  }
}
"#;

/// VARIANT D — linear produce, NO consume of base. Each sibling just adds
/// a new linear datum to the tagged channel without consuming what's there.
/// This is the "orphan was already there, deploy didn't touch it" scenario:
///   - Base produces a non-numeric value (stays on channel)
///   - Sibling A produces a different value (also stays, channel now 2 datums
///     under sibling A's POV)
///   - Sibling B produces a different value (channel now 3 datums under B's POV)
///   - Merge rejects one — but neither A nor B's produces_consumed includes
///     the base produce → base produce survives → final state has at least
///     base + winner = 2 datums.
///
/// This mirrors the CI hypothesis: orphan was placed pre-deploy (genesis or
/// system code) and the deploys never consumed it.
static RHO_BASE_PRODUCE_NO_CONSUME: &str = r#"
new MergeableTag, ch in {
  @(*MergeableTag, *ch)!({"v": 0}) |

  contract @"ADD"(@bit, ret) = {
    @(*MergeableTag, *ch)!({"v": bit}) |
    ret!(Nil)
  }
}
"#;

fn rho_sibling_update(bit: i64) -> String {
    format!(
        r#"
new ret in {{
  @"UPDATE"!({bit}, *ret)
}}
"#,
        bit = bit
    )
}

fn rho_sibling_update_persistent(bit: i64) -> String {
    format!(
        r#"
new ret in {{
  @"UPDATE_PERSISTENT"!({bit}, *ret)
}}
"#,
        bit = bit
    )
}

fn rho_sibling_add(bit: i64) -> String {
    format!(
        r#"
new ret in {{
  @"ADD"!({bit}, *ret)
}}
"#,
        bit = bit
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn variant_a_linear_consume_should_be_caught_by_check_4() {
    let outcome = run_direct_merge(
        "VARIANT A: linear-consume + produce",
        RHO_BASE_LINEAR_CONSUME,
        vec![
            SiblingInfo {
                term: rho_sibling_update(0x01),
                cost: 10,
                sig: "0x11".to_string(),
            },
            SiblingInfo {
                term: rho_sibling_update(0x02),
                cost: 10,
                sig: "0x22".to_string(),
            },
        ],
    )
    .await;

    assert!(
        outcome.wedge_channel.is_some(),
        "no shared identity-tagged channel"
    );
    for s in &outcome.per_sibling {
        assert!(
            s.identity_tagged >= 1,
            "{}: identity_tagged should be populated",
            s.sig
        );
        assert_eq!(
            s.number_channels_data, 0,
            "{}: channel value is Map, should be excluded from number_channels_data",
            s.sig
        );
        assert_eq!(
            s.surviving_on_wedge_in_mergeable, 0,
            "{}: surviving produce should NOT be in produces_mergeable",
            s.sig
        );
    }

    assert!(
        !outcome.rejected_sigs.is_empty(),
        "VARIANT A: expected check #4 to reject one sibling, got rejected={}, datums={}",
        outcome.rejected_sigs.len(),
        outcome.post_merge_datum_count,
    );
    assert!(
        outcome.post_merge_datum_count <= 1,
        "VARIANT A: post-merge state should hold at most 1 datum, got {}",
        outcome.post_merge_datum_count
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn variant_b_peek_then_consume_reveals_check_4_behavior() {
    let outcome = run_direct_merge(
        "VARIANT B: peek then linear-consume + produce",
        RHO_BASE_PEEK_THEN_CONSUME,
        vec![
            SiblingInfo {
                term: rho_sibling_update(0x01),
                cost: 10,
                sig: "0x11".to_string(),
            },
            SiblingInfo {
                term: rho_sibling_update(0x02),
                cost: 10,
                sig: "0x22".to_string(),
            },
        ],
    )
    .await;

    // Diagnostic — do NOT assert on the outcome. Report what happens.
    if outcome.rejected_sigs.is_empty() && outcome.post_merge_datum_count >= 2 {
        panic!(
            "VARIANT B BUG REPRODUCED: peek-then-consume + non-numeric \
             value yielded {} datums on the wedge channel with ZERO \
             rejections. PR #520 check #4 did NOT fire here even though \
             the channel is identity-tagged and surviving_on_wedge_in_mergeable=0. \
             Per-sibling shape: {:?}",
            outcome.post_merge_datum_count, outcome.per_sibling,
        );
    }
    eprintln!(
        "VARIANT B summary: rejected={} datums={} (clean reject path == check #4 fires)",
        outcome.rejected_sigs.len(),
        outcome.post_merge_datum_count
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn variant_d_produce_no_consume_reveals_check_4_behavior() {
    let outcome = run_direct_merge(
        "VARIANT D: linear produce, NO consume of base",
        RHO_BASE_PRODUCE_NO_CONSUME,
        vec![
            SiblingInfo {
                term: rho_sibling_add(0x01),
                cost: 10,
                sig: "0x11".to_string(),
            },
            SiblingInfo {
                term: rho_sibling_add(0x02),
                cost: 10,
                sig: "0x22".to_string(),
            },
        ],
    )
    .await;

    if outcome.rejected_sigs.is_empty() && outcome.post_merge_datum_count >= 2 {
        panic!(
            "VARIANT D BUG REPRODUCED: zero rejection + {} datums = check #4 \
             did not fire. Per-sibling shape: {:?}",
            outcome.post_merge_datum_count, outcome.per_sibling,
        );
    }
    eprintln!(
        "VARIANT D summary: rejected={} datums={}",
        outcome.rejected_sigs.len(),
        outcome.post_merge_datum_count
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn variant_c_persistent_produce_reveals_check_4_behavior() {
    let outcome = run_direct_merge(
        "VARIANT C: persistent-produce on tagged channel",
        RHO_BASE_PERSISTENT_PRODUCE,
        vec![
            SiblingInfo {
                term: rho_sibling_update_persistent(0x01),
                cost: 10,
                sig: "0x11".to_string(),
            },
            SiblingInfo {
                term: rho_sibling_update_persistent(0x02),
                cost: 10,
                sig: "0x22".to_string(),
            },
        ],
    )
    .await;

    if outcome.rejected_sigs.is_empty() && outcome.post_merge_datum_count >= 2 {
        panic!(
            "VARIANT C BUG REPRODUCED: persistent-produce + non-numeric \
             value yielded {} datums with ZERO rejections. \
             Per-sibling shape: {:?}",
            outcome.post_merge_datum_count, outcome.per_sibling,
        );
    }
    eprintln!(
        "VARIANT C summary: rejected={} datums={}",
        outcome.rejected_sigs.len(),
        outcome.post_merge_datum_count
    );
}
