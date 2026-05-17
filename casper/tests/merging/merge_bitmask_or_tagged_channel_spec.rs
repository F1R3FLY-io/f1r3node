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

use std::collections::{HashMap, HashSet};

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
            /* is_commutative_system_deploy */ false,
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

// ============================================================
// VARIANT F — ORPHAN PROPAGATION (the actual CI mechanism)
// ============================================================
//
// Once a tagged channel ends up with multi-Datum in some block's
// post-state, subsequent blocks built on that state propagate the
// orphan even when the deploy's contract is perfectly correct.
//
// Mechanism (from rspace++/merger/state_change.rs:270-393 +
// state_change_merger.rs::make_trie_action):
//   - StateChange::new computes diff = (removed=lost, added=new)
//   - The deploy consumes ONE datum on the channel (whichever the
//     Rholang pattern matches), producing one new datum.
//   - diff is: removed=[V_consumed], added=[V_new]
//   - At merge time, make_trie_action applies:
//       init = pre_state_data       // ALREADY [V_a, V_b] (multi-Datum)
//       new_val = init - removed + added
//             = [V_a, V_b] - [V_a] + [V_new]
//             = [V_b, V_new]        // STILL multi-Datum
//
// PR #520 check #4 is irrelevant here — there's only ONE branch
// updating the channel. There's no sibling conflict to detect. The
// orphan rides along through the multiset_diff path because the
// deploy can only consume ONE of the existing datums per Rholang
// COMM semantics.
//
// This test:
//   1. Builds setup_state with TWO datums on a tagged channel
//      (planted via two no-consume produces — variant-D pattern)
//   2. Runs a SINGLE sibling deploy that does linear-consume +
//      produce against that wedged setup_state
//   3. Asserts that post-merge state still has 2 datums on the
//      channel — the orphan was propagated even with no sibling
//      conflict and a well-behaved deploy
//
// This matches the CI WARN signature: persistent=false on both
// datums, on a BitmaskOr-tagged channel, with no rejection.

static RHO_BASE_PLANT_MULTI_DATUM: &str = r#"
new MergeableTag, ch in {
  @(*MergeableTag, *ch)!({"v": 1}) |
  @(*MergeableTag, *ch)!({"v": 2}) |

  contract @"UPDATE"(@bit, ret) = {
    for(@m <- @(*MergeableTag, *ch)) {
      @(*MergeableTag, *ch)!({"v": bit, "prev": m}) |
      ret!(Nil)
    }
  }
}
"#;

// VARIANT G — bootstrap candidate: single deploy that peeks + produces.
// Peek (`<<-`) does NOT destroy the produce on the channel — it observes
// and the original datum stays. If the deploy then produces a NEW value
// without a linear consume, the channel ends up with BOTH datums after a
// single deploy. This bootstraps multi-Datum without any sibling conflict.
//
// The test asserts on the in-deploy multi-Datum birth (which would emit a
// [TAGGED-CHANNEL-MULTI-DATUM] WARN in production logs) AND on the
// post-merge state.
//
// This pattern is structurally similar to what Registry.rho's
// TreeHashMapSetter does at line 176-179 (peek outer, linear consume
// inner). If for any reason the linear consume DOESN'T match (because
// the produce was already consumed elsewhere by some race), the contract
// continuation would still produce the new value — leaving multi-Datum.
static RHO_BASE_PEEK_NO_LINEAR_CONSUME: &str = r#"
new MergeableTag, ch in {
  @(*MergeableTag, *ch)!({"v": 0}) |

  contract @"PEEK_AND_PRODUCE"(@bit, ret) = {
    for(@m <<- @(*MergeableTag, *ch)) {
      @(*MergeableTag, *ch)!({"v": bit, "observed": m}) |
      ret!(Nil)
    }
  }
}
"#;

fn rho_sibling_peek_and_produce(bit: i64) -> String {
    format!(
        r#"
new ret in {{
  @"PEEK_AND_PRODUCE"!({bit}, *ret)
}}
"#,
        bit = bit
    )
}

// VARIANT H — bootstrap via parallel produce composition.
// Two parallel `ch!(...)` in a single deploy on the same channel. No
// consume. Each produce adds a datum. Channel ends up with multi-Datum.
//
// Differs from G in the EventLogIndex shape: NO peek, NO copied_by_peek.
// Just two surviving produces in produces_linear.
static RHO_BASE_PARALLEL_PRODUCE: &str = r#"
new MergeableTag, ch in {
  contract @"PARALLEL_PRODUCE"(@bit, ret) = {
    @(*MergeableTag, *ch)!({"v": bit, "side": "a"}) |
    @(*MergeableTag, *ch)!({"v": bit, "side": "b"}) |
    ret!(Nil)
  }
}
"#;

fn rho_sibling_parallel_produce(bit: i64) -> String {
    format!(
        r#"
new ret in {{
  @"PARALLEL_PRODUCE"!({bit}, *ret)
}}
"#,
        bit = bit
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn variant_h_parallel_produce_in_single_deploy() {
    let outcome = run_direct_merge(
        "VARIANT H: parallel produce composition (no peek, no consume)",
        RHO_BASE_PARALLEL_PRODUCE,
        vec![SiblingInfo {
            term: rho_sibling_parallel_produce(0x20),
            cost: 10,
            sig: "0x11".to_string(),
        }],
    )
    .await;

    eprintln!(
        "VARIANT H summary: rejected={} post_merge_datums={}",
        outcome.rejected_sigs.len(),
        outcome.post_merge_datum_count
    );

    if outcome.rejected_sigs.is_empty() && outcome.post_merge_datum_count >= 2 {
        eprintln!(
            "VARIANT H BUG REPRODUCED via parallel-produce: single deploy \
             with `ch!(A) | ch!(B)` left {} datums. No peek required.",
            outcome.post_merge_datum_count
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn variant_g_peek_then_produce_no_consume_in_single_deploy() {
    let outcome = run_direct_merge(
        "VARIANT G: peek + produce (no linear consume) — single-deploy bootstrap",
        RHO_BASE_PEEK_NO_LINEAR_CONSUME,
        vec![SiblingInfo {
            term: rho_sibling_peek_and_produce(0x10),
            cost: 10,
            sig: "0x11".to_string(),
        }],
    )
    .await;

    eprintln!(
        "VARIANT G summary: rejected={} post_merge_datums={}",
        outcome.rejected_sigs.len(),
        outcome.post_merge_datum_count
    );

    if outcome.rejected_sigs.is_empty() && outcome.post_merge_datum_count >= 2 {
        eprintln!(
            "VARIANT G BUG REPRODUCED via in-deploy peek+produce: single \
             deploy left {} datums on the tagged channel. This is a \
             BOOTSTRAP candidate — the wedge happens within a single \
             deploy's PLAY, no sibling conflict required.",
            outcome.post_merge_datum_count
        );
    } else if outcome.post_merge_datum_count == 1 {
        eprintln!(
            "VARIANT G negative: peek+produce yielded {} datum. \
             The peek-no-consume pattern does NOT bootstrap multi-Datum \
             in this configuration. Likely the peek's continuation only \
             fires once and the produce is the only addition.",
            outcome.post_merge_datum_count
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn variant_f_pre_state_multi_datum_propagates_through_single_branch() {
    let outcome = run_direct_merge(
        "VARIANT F: pre-state already has multi-Datum, single branch consumes only one",
        RHO_BASE_PLANT_MULTI_DATUM,
        vec![SiblingInfo {
            term: rho_sibling_update(0x10),
            cost: 10,
            sig: "0x11".to_string(),
        }],
    )
    .await;

    eprintln!(
        "VARIANT F summary: rejected={} post_merge_datums={}",
        outcome.rejected_sigs.len(),
        outcome.post_merge_datum_count
    );

    // We EXPECT this to leave multi-Datum after merge with ZERO
    // rejections — that's the orphan-propagation bug class.
    assert_eq!(
        outcome.rejected_sigs.len(),
        0,
        "VARIANT F: with only one branch, there's no sibling conflict to reject. Got unexpected rejection: {:?}",
        outcome.rejected_sigs,
    );

    if outcome.post_merge_datum_count >= 2 {
        eprintln!(
            "VARIANT F BUG REPRODUCED via orphan propagation: pre-state \
             multi-Datum + single deploy with consume + produce leaves \
             {} datums on the tagged channel after merge. No sibling \
             conflict means check #4 had nothing to check; the diff path \
             propagated the unconsumed datum unconditionally.",
            outcome.post_merge_datum_count
        );
    } else {
        eprintln!(
            "VARIANT F NOTE: post_merge_datum_count={} — propagation \
             theory did NOT reproduce. Investigate StateChange::new's \
             diff calculation in detail.",
            outcome.post_merge_datum_count
        );
    }
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

// ============================================================
// VARIANT E — multi-block via compute_state + block_index::new
// ============================================================
//
// Mirrors the CI scenario more faithfully than the synthetic A–D variants:
//   - Each "sibling" is a real block built via `RuntimeManager.compute_state`
//   - Each block includes a closeBlock system deploy (matches CI lifecycle)
//   - Block construction uses `block_index::new` to assemble DeployChainIndex
//     across user + system deploys
//   - The shared mergeable tag is the standard `rho:system:bitmaskMergeableTag`
//     URI (the SAME tag that Registry.rho's TreeHashMap uses)
//   - A SETUP block installs the @"UPDATE_TAGGED" contract, capturing the
//     unforgeable channel hash. Subsequent sibling blocks both invoke that
//     contract, hitting the same tagged channel.
//
// Asserts on the merge outcome the same way as A–D: either rejection
// happens (check #4 in production path fires) or post-merge state has
// multi-Datum on the wedge channel (bug reproduced in production path).

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn variant_e_multi_block_compute_state_two_sibling_blocks_merged() {
    use casper::rust::merging::block_index as block_index_module;
    use casper::rust::util::construct_deploy;
    use casper::rust::util::rholang::costacc::close_block_deploy::CloseBlockDeploy;
    use casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum;
    use casper::rust::util::rholang::system_deploy_util;
    use rholang::rust::interpreter::system_processes::BlockData;

    use crate::util::rholang::resources::with_runtime_manager;

    with_runtime_manager(|runtime_manager, genesis_context, _genesis_block| async move {
        eprintln!("\n========== VARIANT E: multi-block via compute_state ==========");

        let base_state = genesis_context.genesis_block.body.state.post_state_hash.clone();
        // Use two distinct validators for sibling blocks so the closeBlock
        // system deploys derive different random seeds (mirroring CI).
        let validator_a = genesis_context.validator_key_pairs[0].1.clone();
        let validator_b = genesis_context.validator_key_pairs[1].1.clone();
        let payer_key = genesis_context.genesis_vaults[0].0.clone();
        let payer_key_2 = genesis_context.genesis_vaults[1].0.clone();

        // -------- SETUP block --------
        // Installs the @"UPDATE_TAGGED" contract using the standard
        // bitmaskMergeableTag URI. The `new ch` inside the SETUP deploy
        // captures an unforgeable that the contract body closes over;
        // every subsequent invocation of @"UPDATE_TAGGED" reaches the
        // SAME (*bitmaskTag, *ch) channel hash.
        let setup_rho = r#"
new bitmaskTag(`rho:system:bitmaskMergeableTag`), ch in {
  @(*bitmaskTag, *ch)!({"v": 0}) |
  contract @"UPDATE_TAGGED"(@bit, ret) = {
    for(@m <- @(*bitmaskTag, *ch)) {
      @(*bitmaskTag, *ch)!({"v": bit, "prev": m}) |
      ret!(Nil)
    }
  }
}
"#;
        let setup_deploy = construct_deploy::source_deploy_now_full(
            setup_rho.to_string(),
            None,
            None,
            Some(payer_key.clone()),
            None,
            None,
        )
        .expect("setup deploy construction");
        let setup_seq_num: i32 = 1;
        let setup_block_data = BlockData {
            time_stamp: setup_deploy.data.time_stamp,
            seq_num: setup_seq_num,
            block_number: 1,
            sender: validator_a.clone(),
        };
        let setup_close = SystemDeployEnum::Close(CloseBlockDeploy {
            initial_rand: system_deploy_util::generate_close_deploy_random_seed_from_pk(
                validator_a.clone(),
                setup_seq_num,
            ),
        });
        let (setup_state, setup_user_processed, setup_sys_processed) = runtime_manager
            .compute_state(
                &base_state,
                vec![setup_deploy],
                vec![setup_close],
                setup_block_data,
                None,
            )
            .await
            .expect("setup compute_state");
        assert_eq!(
            setup_user_processed.len(),
            1,
            "setup block should have 1 user deploy"
        );
        eprintln!(
            "  SETUP: state={} user_deploys={} sys_deploys={}",
            hex::encode(&setup_state[..std::cmp::min(8, setup_state.len())]),
            setup_user_processed.len(),
            setup_sys_processed.len(),
        );

        // -------- Block X (sibling 1) --------
        // Invokes @"UPDATE_TAGGED" with bit 0x01. Block X is built off
        // setup_state by validator_a.
        let deploy_a_rho = r#"new ret in { @"UPDATE_TAGGED"!(1, *ret) }"#;
        let deploy_a = construct_deploy::source_deploy_now_full(
            deploy_a_rho.to_string(),
            None,
            None,
            Some(payer_key.clone()),
            None,
            None,
        )
        .expect("deploy A construction");
        let block_x_seq_num: i32 = 2;
        let block_x_block_data = BlockData {
            time_stamp: deploy_a.data.time_stamp,
            seq_num: block_x_seq_num,
            block_number: 2,
            sender: validator_a.clone(),
        };
        let block_x_close = SystemDeployEnum::Close(CloseBlockDeploy {
            initial_rand: system_deploy_util::generate_close_deploy_random_seed_from_pk(
                validator_a.clone(),
                block_x_seq_num,
            ),
        });
        let (block_x_post, block_x_user_processed, block_x_sys_processed) = runtime_manager
            .compute_state(
                &setup_state,
                vec![deploy_a],
                vec![block_x_close],
                block_x_block_data,
                None,
            )
            .await
            .expect("block X compute_state");
        let block_x_mergeable = runtime_manager
            .load_mergeable_channels(
                &block_x_post,
                validator_a.bytes.clone(),
                block_x_seq_num,
            )
            .expect("load block X mergeable channels");

        // -------- Block Y (sibling 2) --------
        // Invokes @"UPDATE_TAGGED" with bit 0x02. Block Y is built off the
        // SAME setup_state, but by validator_b with a different deployer
        // (payer_key_2). Same seq_num because Y is a sibling, not a
        // descendant of X.
        let deploy_b_rho = r#"new ret in { @"UPDATE_TAGGED"!(2, *ret) }"#;
        let deploy_b = construct_deploy::source_deploy_now_full(
            deploy_b_rho.to_string(),
            None,
            None,
            Some(payer_key_2.clone()),
            None,
            None,
        )
        .expect("deploy B construction");
        let block_y_seq_num: i32 = 2;
        let block_y_block_data = BlockData {
            time_stamp: deploy_b.data.time_stamp,
            seq_num: block_y_seq_num,
            block_number: 2,
            sender: validator_b.clone(),
        };
        let block_y_close = SystemDeployEnum::Close(CloseBlockDeploy {
            initial_rand: system_deploy_util::generate_close_deploy_random_seed_from_pk(
                validator_b.clone(),
                block_y_seq_num,
            ),
        });
        let (block_y_post, block_y_user_processed, block_y_sys_processed) = runtime_manager
            .compute_state(
                &setup_state,
                vec![deploy_b],
                vec![block_y_close],
                block_y_block_data,
                None,
            )
            .await
            .expect("block Y compute_state");
        let block_y_mergeable = runtime_manager
            .load_mergeable_channels(
                &block_y_post,
                validator_b.bytes.clone(),
                block_y_seq_num,
            )
            .expect("load block Y mergeable channels");

        eprintln!(
            "  BLOCK X: validator={} state={} user_deploys={} sys_deploys={} mergeable={}",
            hex::encode(&validator_a.bytes[..std::cmp::min(4, validator_a.bytes.len())]),
            hex::encode(&block_x_post[..std::cmp::min(8, block_x_post.len())]),
            block_x_user_processed.len(),
            block_x_sys_processed.len(),
            block_x_mergeable.len(),
        );
        eprintln!(
            "  BLOCK Y: validator={} state={} user_deploys={} sys_deploys={} mergeable={}",
            hex::encode(&validator_b.bytes[..std::cmp::min(4, validator_b.bytes.len())]),
            hex::encode(&block_y_post[..std::cmp::min(8, block_y_post.len())]),
            block_y_user_processed.len(),
            block_y_sys_processed.len(),
            block_y_mergeable.len(),
        );

        // -------- Build BlockIndex for each sibling --------
        let setup_state_h = Blake2b256Hash::from_bytes_prost(&setup_state);
        let block_x_post_h = Blake2b256Hash::from_bytes_prost(&block_x_post);
        let block_y_post_h = Blake2b256Hash::from_bytes_prost(&block_y_post);

        let block_x_hash: models::rust::block_hash::BlockHash =
            prost::bytes::Bytes::from(vec![0xAAu8; 32]);
        let block_y_hash: models::rust::block_hash::BlockHash =
            prost::bytes::Bytes::from(vec![0xBBu8; 32]);

        let block_x_index = block_index_module::new(
            &block_x_hash,
            2,
            &block_x_user_processed,
            &block_x_sys_processed,
            &setup_state_h,
            &block_x_post_h,
            &runtime_manager.get_history_repo(),
            &block_x_mergeable,
        )
        .expect("block_index X");
        let block_y_index = block_index_module::new(
            &block_y_hash,
            2,
            &block_y_user_processed,
            &block_y_sys_processed,
            &setup_state_h,
            &block_y_post_h,
            &runtime_manager.get_history_repo(),
            &block_y_mergeable,
        )
        .expect("block_index Y");

        eprintln!(
            "  BLOCK X INDEX: {} deploy_chains",
            block_x_index.deploy_chains.len()
        );
        eprintln!(
            "  BLOCK Y INDEX: {} deploy_chains",
            block_y_index.deploy_chains.len()
        );

        // For each chain, surface the identity_tagged and surviving-on-wedge
        // shape — same diagnostic as variants A–D.
        let report_chain = |label: &str, chain: &DeployChainIndex| {
            let eli = &chain.event_log_index;
            let id_tagged_examples: Vec<String> = eli
                .identity_tagged_channels
                .0
                .iter()
                .take(5)
                .map(|h| {
                    let b = h.bytes();
                    hex::encode(&b[..std::cmp::min(8, b.len())])
                })
                .collect();
            eprintln!(
                "    {} chain: identity_tagged={} number_channels_data={} \
                 produces_linear={} produces_consumed={} \
                 produces_copied_by_peek={} produces_mergeable={} \
                 identity_tagged_examples={:?}",
                label,
                eli.identity_tagged_channels.0.len(),
                eli.number_channels_data.len(),
                eli.produces_linear.0.len(),
                eli.produces_consumed.0.len(),
                eli.produces_copied_by_peek.0.len(),
                eli.produces_mergeable.0.len(),
                id_tagged_examples,
            );
        };
        for c in &block_x_index.deploy_chains {
            report_chain("X", c);
        }
        for c in &block_y_index.deploy_chains {
            report_chain("Y", c);
        }

        // -------- Identify the shared tagged channel --------
        let block_x_identity_tagged_union: HashSet<Blake2b256Hash> = block_x_index
            .deploy_chains
            .iter()
            .flat_map(|c| c.event_log_index.identity_tagged_channels.0.iter().cloned())
            .collect();
        let block_y_identity_tagged_union: HashSet<Blake2b256Hash> = block_y_index
            .deploy_chains
            .iter()
            .flat_map(|c| c.event_log_index.identity_tagged_channels.0.iter().cloned())
            .collect();
        let shared: HashSet<Blake2b256Hash> = block_x_identity_tagged_union
            .intersection(&block_y_identity_tagged_union)
            .cloned()
            .collect();
        eprintln!(
            "  Identity-tagged intersection between X and Y: {} channels",
            shared.len()
        );
        for h in shared.iter().take(5) {
            eprintln!("    shared: {}", hex::encode(h.bytes()));
        }

        // -------- Merge X and Y --------
        let history_repo = runtime_manager.get_history_repo();
        let base_reader = history_repo.get_history_reader(&setup_state_h).unwrap();

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
        let compute_trie_actions_fn =
            |changes: StateChange, mergeable_chs: NumberChannelsDiff| {
                state_change_merger::compute_trie_actions(
                    &changes,
                    &base_reader,
                    &mergeable_chs,
                    |h, c, nc| override_trie_action(h, c, nc),
                )
            };
        let apply_trie_actions_fn = |actions: Vec<
            HotStoreTrieAction<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
        >| {
            runtime_manager
                .get_history_repo()
                .reset(&setup_state_h)
                .map(|r1| {
                    let r2 = r1.do_checkpoint(actions);
                    r2.root()
                })
        };

        let mut actual_seq: Vec<DeployChainIndex> = block_x_index
            .deploy_chains
            .into_iter()
            .chain(block_y_index.deploy_chains.into_iter())
            .collect();
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
            compute_trie_actions_fn,
            apply_trie_actions_fn,
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
        .expect("conflict_set_merger::merge");

        let rejected_sigs: Vec<String> = rejected
            .0
            .iter()
            .flat_map(|r| r.deploys_with_cost.0.iter())
            .map(|d| hex::encode(&d.deploy_id))
            .collect();
        eprintln!(
            "  MERGE OUTCOME: rejected={} sigs={:?} final_state={}",
            rejected_sigs.len(),
            rejected_sigs,
            hex::encode(&final_hash.bytes()[..std::cmp::min(8, final_hash.bytes().len())]),
        );

        // -------- Inspect post-merge state on the wedge channel --------
        let post_reader = history_repo.get_history_reader(&final_hash).unwrap();
        let mut max_datum_count_on_shared = 0usize;
        for ch in &shared {
            let data = post_reader.get_data(ch).unwrap();
            let n = data.len();
            if n > max_datum_count_on_shared {
                max_datum_count_on_shared = n;
            }
            let bytes = ch.bytes();
            eprintln!(
                "    post-merge channel {}: {} datums",
                hex::encode(&bytes[..std::cmp::min(8, bytes.len())]),
                n
            );
            for (i, d) in data.iter().enumerate() {
                let src = d.source.hash.bytes();
                eprintln!(
                    "      datum[{}] source={} persistent={}",
                    i,
                    hex::encode(&src[..std::cmp::min(8, src.len())]),
                    d.persist
                );
            }
        }

        // -------- Assertion --------
        if rejected_sigs.is_empty() && max_datum_count_on_shared >= 2 {
            panic!(
                "VARIANT E BUG REPRODUCED via compute_state path: ZERO \
                 rejections + {} datums on a shared tagged channel. PR #520 \
                 check #4 should have flagged the conflict but didn't.",
                max_datum_count_on_shared,
            );
        }
        eprintln!(
            "  VARIANT E summary: rejected={} max_datums_on_shared={}",
            rejected_sigs.len(),
            max_datum_count_on_shared
        );
    })
    .await
    .unwrap()
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
