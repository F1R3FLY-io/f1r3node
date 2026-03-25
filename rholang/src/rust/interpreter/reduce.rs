// See See rholang/src/main/scala/coop/rchain/rholang/interpreter/Reduce.scala

use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use crypto::rust::hash::blake2b256::Blake2b256;
use models::rhoapi::expr::ExprInstance;
use models::rhoapi::g_unforgeable::UnfInstance;
use models::rhoapi::tagged_continuation::TaggedCont;
use models::rhoapi::var::VarInstance;
use models::rhoapi::{
    BindPattern, Bundle, EAnd, EDiv, EEq, EGt, EGte, EList, ELt, ELte, EMatches, EMethod, EMinus,
    EMinusMinus, EMod, EMult, ENeq, EOr, EPathMap, EPercentPercent, EPlus, EPlusPlus, EVar,
    EZipper, Expr, GPrivate, GUnforgeable, KeyValuePair, Match, MatchCase, New, ParWithRandom,
    Receive, ReceiveBind, Send, Var,
};
use models::rhoapi::{ETuple, ListParWithRandom, Par, TaggedContinuation};
use models::rust::par_map::ParMap;
use models::rust::par_map_type_mapper::ParMapTypeMapper;
use models::rust::par_set::ParSet;
use models::rust::par_set_type_mapper::ParSetTypeMapper;
use models::rust::pathmap_zipper::RholangReadZipper;
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
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, RwLock};
use std::task::{Context, Poll};
use tracing::debug;

use crate::rust::interpreter::accounting::costs::{
    add_cost, bytes_to_hex_cost, diff_cost, hex_to_bytes_cost, interpolate_cost, keys_method_cost,
    length_method_cost, lookup_cost, match_eval_cost, nth_method_call_cost, remove_cost,
    size_method_cost, slice_cost, take_cost, to_byte_array_cost, to_list_cost, union_cost,
};
use crate::rust::interpreter::matcher::spatial_matcher::SpatialMatcherContext;
use crate::rust::interpreter::rho_type::RhoTuple2;

use super::accounting::_cost;
use super::accounting::costs::{
    bigint_comparison_cost, bigint_division_cost, bigint_modulo_cost,
    bigint_multiplication_cost, bigint_negation_cost, bigint_subtraction_cost, bigint_sum_cost,
    bigrat_comparison_cost, bigrat_division_cost, bigrat_multiplication_cost, bigrat_negation_cost,
    bigrat_subtraction_cost, bigrat_sum_cost, boolean_and_cost, boolean_or_cost,
    byte_array_append_cost, comparison_cost, division_cost, equality_check_cost, list_append_cost,
    method_call_cost, modulo_cost, multiplication_cost, new_bindings_cost, op_call_cost,
    receive_eval_cost, send_eval_cost, string_append_cost, subtraction_cost, sum_cost,
    var_eval_cost,
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
use super::storage::continuation_store::{
    ContinuationHandle, ContinuationStatus, ContinuationStoreError, CreateContinuation,
    FundingPolicy, InMemoryContinuationStore, PersistedContinuation,
};
use models::rust::pathmap_crate_type_mapper::PathMapCrateTypeMapper;

/// Minimum remaining stack space (in bytes) before growing.
/// When the current stack has less than this amount remaining, a new stack segment is allocated.
// 128 KB is too small: a single recursion frame in the Rholang interpreter
// (eval → produce/consume → dispatch → eval) consumes more than 128 KB between
// stacker checks, so the overflow happens before stacker can grow the stack.
const STACK_RED_ZONE: usize = 1024 * 1024; // 1 MB

/// Size of each new stack segment allocated when the red zone is reached.
const STACK_GROW_SIZE: usize = 2 * 1024 * 1024; // 2 MB

fn parse_env_flag(value: &str) -> bool {
    value == "1" || value.eq_ignore_ascii_case("true")
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| parse_env_flag(value.trim()))
        .unwrap_or(false)
}

fn read_vm_rss_kb() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find(|line| line.starts_with("VmRSS:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<usize>().ok())
}

static REDUCE_INNER_PROFILE_ENABLED: LazyLock<bool> =
    LazyLock::new(|| env_flag("F1R3_REDUCE_INNER_PROFILE"));
static REDUCE_OP_PROFILE_ENABLED: LazyLock<bool> =
    LazyLock::new(|| env_flag("F1R3_REDUCE_OP_PROFILE"));

/// A Future wrapper that dynamically grows the thread stack during polling.
///
/// The Rholang interpreter uses deep async recursion: eval → produce/consume → dispatch → eval.
/// Each poll of this recursive future chain adds stack frames. In debug builds, unoptimized
/// async state machines consume ~1-2KB per recursion level, causing stack overflow with the
/// default 2MB thread stack.
///
/// `StackGrowingFuture` wraps each recursive entry point (eval, produce, consume, dispatch).
/// On each poll, `stacker::maybe_grow` checks remaining stack space. If below STACK_RED_ZONE,
/// it allocates a new STACK_GROW_SIZE segment and runs the poll there. This allows arbitrarily
/// deep Rholang recursion (e.g., longslow.rho with 32768 iterations) without stack overflow.
///
/// See: https://github.com/F1R3FLY-io/f1r3node/issues/305
/// See: https://github.com/F1R3FLY-io/f1r3node/issues/306
struct StackGrowingFuture<F> {
    inner: F,
}

impl<F: Future> Future for StackGrowingFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: Structural pin projection on a single-field struct with no Drop impl.
        // `inner` is only accessed through this pinned projection, and StackGrowingFuture
        // does not implement Unpin when F doesn't, preserving pin guarantees.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.inner) };
        stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW_SIZE, || inner.poll(cx))
    }
}

/**
 * Reduce is the interface for evaluating Rholang expressions.
 */
#[derive(Clone)]
pub struct DebruijnInterpreter {
    pub space: RhoISpace,
    pub dispatcher: RhoDispatch,
    pub urn_map: Arc<HashMap<String, Par>>,
    pub merge_chs: Arc<RwLock<HashSet<Par>>>,
    pub mergeable_tag_name: Par,
    pub cost: _cost,
    pub substitute: Substitute,
    pub(crate) split_post_match_config: Arc<RwLock<SplitPostMatchConfig>>,
    pub(crate) continuation_store: Arc<RwLock<InMemoryContinuationStore>>,
    pub(crate) continuation_nonce: Arc<AtomicU64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SplitPostMatchConfig {
    pub enabled: bool,
    pub initial_step_gas_limit: i64,
}

impl Default for SplitPostMatchConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            initial_step_gas_limit: i64::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContinuationExecutionStatus {
    Suspended,
    Completed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContinuationExecutionResult {
    pub handle: ContinuationHandle,
    pub status: ContinuationExecutionStatus,
    pub version: u64,
    pub state_root: Vec<u8>,
    pub consumed_gas: i64,
    pub emitted_effects: bool,
}

type MatchedData = (Par, ListParWithRandom, ListParWithRandom, bool);
type Application = Option<(TaggedContinuation, Vec<MatchedData>, bool)>;
#[cfg(test)]
type DispatchResultFuture<'a> = Pin<
    Box<
        dyn futures::Future<Output = Result<DispatchType, InterpreterError>> + std::marker::Send + 'a,
    >,
>;

#[derive(Clone)]
struct DispatchExecState {
    continuation: TaggedContinuation,
    data_list: Vec<MatchedData>,
    is_replay: bool,
    previous_output: Vec<Par>,
    // Phase 1 metadata capture for later step-based gas accounting.
    _remaining_phlo: i64,
}

#[derive(Clone)]
struct ReceiveRegistrationState {
    binds: Vec<(BindPattern, Par)>,
    body: ParWithRandom,
    persistent: bool,
    peek: bool,
}

#[derive(Clone)]
struct ProduceRegistrationState {
    chan: Par,
    data: ListParWithRandom,
    persistent: bool,
}

#[derive(Clone)]
enum PostMatchControlState {
    DispatchOnly,
    DispatchAndReconsume(ReceiveRegistrationState),
    DispatchAndReproduce(ProduceRegistrationState),
    DispatchAndRestorePeeks,
    Reconsume(ReceiveRegistrationState),
    Reproduce(ProduceRegistrationState),
    RestorePeeks,
}

#[derive(Clone)]
struct PostMatchExecState {
    dispatch: DispatchExecState,
    control: PostMatchControlState,
}

#[derive(Clone, Copy)]
enum StepTerminalStatus {
    Completed,
    Suspended,
}

struct StepResult {
    dispatch_result: DispatchType,
    emitted_effects: bool,
    next_state: Option<PostMatchExecState>,
    consumed_gas: i64,
    terminal_status: StepTerminalStatus,
}

fn should_suspend_for_step(gas_limit: i64) -> bool {
    gas_limit <= 0
}

fn compute_consumed_gas(starting_phlo: i64, remaining_phlo: i64) -> i64 {
    if remaining_phlo >= starting_phlo {
        0
    } else {
        starting_phlo - remaining_phlo
    }
}

fn dispatch_emits_effects(dispatch_result: &DispatchType) -> bool {
    !matches!(dispatch_result, DispatchType::Skip)
}

#[cfg(test)]
fn next_control_after_dispatch(control: PostMatchControlState) -> Option<PostMatchControlState> {
    match control {
        PostMatchControlState::DispatchOnly => None,
        PostMatchControlState::DispatchAndReconsume(state) => {
            Some(PostMatchControlState::Reconsume(state))
        }
        PostMatchControlState::DispatchAndReproduce(state) => {
            Some(PostMatchControlState::Reproduce(state))
        }
        PostMatchControlState::DispatchAndRestorePeeks => Some(PostMatchControlState::RestorePeeks),
        PostMatchControlState::Reconsume(_)
        | PostMatchControlState::Reproduce(_)
        | PostMatchControlState::RestorePeeks => None,
    }
}

#[cfg(test)]
mod step_helpers_tests {
    use super::*;
    use crate::rust::interpreter::test_utils::persistent_store_tester::create_test_space;
    use rspace_plus_plus::rspace::rspace::RSpace;
    use rspace_plus_plus::rspace::rspace_interface::ISpace;

    #[test]
    fn step_should_suspend_when_gas_limit_is_non_positive() {
        assert!(should_suspend_for_step(0));
        assert!(should_suspend_for_step(-1));
        assert!(!should_suspend_for_step(1));
    }

    #[test]
    fn step_consumed_gas_is_monotonic_and_clamped() {
        assert_eq!(compute_consumed_gas(100, 70), 30);
        assert_eq!(compute_consumed_gas(100, 100), 0);
        assert_eq!(compute_consumed_gas(70, 100), 0);
    }

    #[test]
    fn step_effect_marker_matches_dispatch_type() {
        assert!(!dispatch_emits_effects(&DispatchType::Skip));
        assert!(dispatch_emits_effects(&DispatchType::DeterministicCall));
        assert!(dispatch_emits_effects(&DispatchType::NonDeterministicCall(vec![])));
    }

    #[test]
    fn next_control_after_dispatch_maps_expected_variants() {
        assert!(next_control_after_dispatch(PostMatchControlState::DispatchOnly).is_none());

        let receive_state = ReceiveRegistrationState {
            binds: Vec::new(),
            body: ParWithRandom::default(),
            persistent: true,
            peek: false,
        };
        assert!(matches!(
            next_control_after_dispatch(PostMatchControlState::DispatchAndReconsume(
                receive_state
            )),
            Some(PostMatchControlState::Reconsume(_))
        ));

        let produce_state = ProduceRegistrationState {
            chan: Par::default(),
            data: ListParWithRandom::default(),
            persistent: true,
        };
        assert!(matches!(
            next_control_after_dispatch(PostMatchControlState::DispatchAndReproduce(
                produce_state
            )),
            Some(PostMatchControlState::Reproduce(_))
        ));

        assert!(matches!(
            next_control_after_dispatch(PostMatchControlState::DispatchAndRestorePeeks),
            Some(PostMatchControlState::RestorePeeks)
        ));
    }

    #[tokio::test]
    async fn reduce_step_suspends_after_dispatch_for_reconsume_path() {
        let (_, reducer) =
            create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
                .await;

        let state = PostMatchExecState {
            dispatch: DispatchExecState {
                continuation: TaggedContinuation::default(),
                data_list: Vec::new(),
                is_replay: false,
                previous_output: Vec::new(),
                _remaining_phlo: reducer.cost.get().value,
            },
            control: PostMatchControlState::DispatchAndReconsume(ReceiveRegistrationState {
                binds: Vec::new(),
                body: ParWithRandom::default(),
                persistent: true,
                peek: false,
            }),
        };

        let step_result = reducer.reduce_step(state, i64::MAX).await.unwrap();
        assert!(matches!(step_result.terminal_status, StepTerminalStatus::Suspended));
        assert!(matches!(
            step_result.next_state.unwrap().control,
            PostMatchControlState::Reconsume(_)
        ));
    }

    #[tokio::test]
    async fn reduce_step_suspends_after_dispatch_for_restore_peeks_path() {
        let (_, reducer) =
            create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
                .await;

        let state = PostMatchExecState {
            dispatch: DispatchExecState {
                continuation: TaggedContinuation::default(),
                data_list: Vec::new(),
                is_replay: false,
                previous_output: Vec::new(),
                _remaining_phlo: reducer.cost.get().value,
            },
            control: PostMatchControlState::DispatchAndRestorePeeks,
        };

        let step_result = reducer.reduce_step(state, i64::MAX).await.unwrap();
        assert!(matches!(step_result.terminal_status, StepTerminalStatus::Suspended));
        assert!(matches!(
            step_result.next_state.unwrap().control,
            PostMatchControlState::RestorePeeks
        ));
    }

    #[tokio::test]
    async fn reduce_step_completes_reconsume_follow_up_path() {
        let (_, reducer) =
            create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
                .await;

        let state = PostMatchExecState {
            dispatch: DispatchExecState {
                continuation: TaggedContinuation::default(),
                data_list: Vec::new(),
                is_replay: false,
                previous_output: Vec::new(),
                _remaining_phlo: reducer.cost.get().value,
            },
            control: PostMatchControlState::Reconsume(ReceiveRegistrationState {
                binds: vec![(
                    BindPattern {
                        patterns: vec![Par::default()],
                        remainder: None,
                        free_count: 0,
                    },
                    Par::default(),
                )],
                body: ParWithRandom::default(),
                persistent: true,
                peek: false,
            }),
        };

        let step_result = reducer.reduce_step(state, i64::MAX).await.unwrap();
        assert!(matches!(step_result.terminal_status, StepTerminalStatus::Completed));
        assert!(step_result.next_state.is_none());
    }

    #[tokio::test]
    async fn reduce_step_completes_restore_peeks_follow_up_path() {
        let (_, reducer) =
            create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
                .await;

        let state = PostMatchExecState {
            dispatch: DispatchExecState {
                continuation: TaggedContinuation::default(),
                data_list: Vec::new(),
                is_replay: false,
                previous_output: Vec::new(),
                _remaining_phlo: reducer.cost.get().value,
            },
            control: PostMatchControlState::RestorePeeks,
        };

        let step_result = reducer.reduce_step(state, i64::MAX).await.unwrap();
        assert!(matches!(step_result.terminal_status, StepTerminalStatus::Completed));
        assert!(step_result.next_state.is_none());
    }

    #[tokio::test]
    async fn execute_post_match_to_completion_advances_suspended_state() {
        let (_, reducer) =
            create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
                .await;

        let state = PostMatchExecState {
            dispatch: DispatchExecState {
                continuation: TaggedContinuation::default(),
                data_list: Vec::new(),
                is_replay: false,
                previous_output: Vec::new(),
                _remaining_phlo: reducer.cost.get().value,
            },
            control: PostMatchControlState::DispatchAndRestorePeeks,
        };

        let result = reducer
            .execute_post_match_to_completion(state, i64::MAX)
            .await
            .unwrap();
        assert!(matches!(result, DispatchType::Skip));
    }

    fn sample_dispatch_and_restore_state(remaining_phlo: i64) -> PostMatchExecState {
        let result_channel = new_gstring_par("step-result".to_string(), Vec::new(), false);
        let restored_channel = new_gstring_par("step-peek".to_string(), Vec::new(), false);
        let restored_payload = ListParWithRandom {
            pars: vec![new_gint_par(7, Vec::new(), false)],
            random_state: Vec::new(),
        };
        let continuation_body = Par::default().with_sends(vec![Send {
            chan: Some(result_channel),
            data: vec![new_gstring_par("ok".to_string(), Vec::new(), false)],
            persistent: false,
            locally_free: Vec::new(),
            connective_used: false,
        }]);

        PostMatchExecState {
            dispatch: DispatchExecState {
                continuation: TaggedContinuation {
                    tagged_cont: Some(TaggedCont::ParBody(ParWithRandom {
                        body: Some(continuation_body),
                        random_state: Vec::new(),
                    })),
                },
                data_list: vec![(
                    restored_channel,
                    restored_payload.clone(),
                    restored_payload,
                    false,
                )],
                is_replay: false,
                previous_output: Vec::new(),
                _remaining_phlo: remaining_phlo,
            },
            control: PostMatchControlState::DispatchAndRestorePeeks,
        }
    }

    #[tokio::test]
    async fn full_execution_and_step_execution_are_equivalent_for_dispatch_restore_peeks() {
        let (full_space, full_reducer) =
            create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
                .await;
        let full_state = sample_dispatch_and_restore_state(full_reducer.cost.get().value);
        full_reducer
            .execute_post_match_exec_state_full(full_state)
            .await
            .unwrap();
        let full_map = full_space.to_map();

        let (step_space, step_reducer) =
            create_test_space::<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>()
                .await;
        let step_state = sample_dispatch_and_restore_state(step_reducer.cost.get().value);
        let step_result_1 = step_reducer.reduce_step(step_state, i64::MAX).await.unwrap();
        assert!(matches!(
            step_result_1.terminal_status,
            StepTerminalStatus::Suspended
        ));
        let step_result_2 = step_reducer
            .reduce_step(step_result_1.next_state.unwrap(), i64::MAX)
            .await
            .unwrap();
        assert!(matches!(
            step_result_2.terminal_status,
            StepTerminalStatus::Completed
        ));
        let step_map = step_space.to_map();

        assert_eq!(full_map, step_map);
    }

    #[test]
    fn serialize_post_match_state_supports_dispatch_and_follow_up_controls() {
        let dispatch = DispatchExecState {
            continuation: TaggedContinuation::default(),
            data_list: Vec::new(),
            is_replay: false,
            previous_output: Vec::new(),
            _remaining_phlo: 42,
        };

        let receive_state = ReceiveRegistrationState {
            binds: vec![(
                BindPattern::default(),
                new_gstring_par("channel".to_string(), Vec::new(), false),
            )],
            body: ParWithRandom::default(),
            persistent: true,
            peek: false,
        };

        let produce_state = ProduceRegistrationState {
            chan: new_gstring_par("channel".to_string(), Vec::new(), false),
            data: ListParWithRandom {
                pars: vec![new_gint_par(7, Vec::new(), false)],
                random_state: Vec::new(),
            },
            persistent: true,
        };

        let states = vec![
            PostMatchExecState {
                dispatch: dispatch.clone(),
                control: PostMatchControlState::DispatchOnly,
            },
            PostMatchExecState {
                dispatch: dispatch.clone(),
                control: PostMatchControlState::DispatchAndReconsume(receive_state.clone()),
            },
            PostMatchExecState {
                dispatch: dispatch.clone(),
                control: PostMatchControlState::DispatchAndReproduce(produce_state.clone()),
            },
            PostMatchExecState {
                dispatch: dispatch.clone(),
                control: PostMatchControlState::DispatchAndRestorePeeks,
            },
            PostMatchExecState {
                dispatch: dispatch.clone(),
                control: PostMatchControlState::Reconsume(receive_state),
            },
            PostMatchExecState {
                dispatch: dispatch.clone(),
                control: PostMatchControlState::Reproduce(produce_state),
            },
            PostMatchExecState {
                dispatch,
                control: PostMatchControlState::RestorePeeks,
            },
        ];

        for state in states.iter() {
            let encoded = serialize_post_match_state(state).unwrap();
            assert!(!encoded.is_empty());
            let decoded = deserialize_post_match_state(&encoded).unwrap();
            let reencoded = serialize_post_match_state(&decoded).unwrap();
            assert_eq!(encoded, reencoded);
        }
    }
}

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
    pub fn eval<'a>(
        &'a self,
        par: Par,
        env: &'a Env<Par>,
        rand: Blake2b512Random,
    ) -> Pin<
        Box<
            dyn std::future::Future<Output = Result<(), InterpreterError>> + std::marker::Send + 'a,
        >,
    > {
        Box::pin(StackGrowingFuture {
            inner: self.eval_inner(par, env, rand),
        })
    }

    async fn eval_inner(
        &self,
        par: Par,
        env: &Env<Par>,
        rand: Blake2b512Random,
    ) -> Result<(), InterpreterError> {
        let mem_profile_enabled = *REDUCE_INNER_PROFILE_ENABLED;
        let env_level = env.level;
        let mut rss_baseline = if mem_profile_enabled {
            read_vm_rss_kb()
        } else {
            None
        };
        let mut rss_prev = rss_baseline;
        let mut log_mem_step = |step: &str, terms_len: Option<usize>, errors_len: Option<usize>| {
            if !mem_profile_enabled {
                return;
            }
            if env_level != 0 {
                return;
            }
            if let Some(curr) = read_vm_rss_kb() {
                let prev = rss_prev.unwrap_or(curr);
                let baseline = rss_baseline.unwrap_or(curr);
                let delta_prev = curr as i64 - prev as i64;
                if delta_prev == 0 {
                    rss_prev = Some(curr);
                    if rss_baseline.is_none() {
                        rss_baseline = Some(curr);
                    }
                    return;
                }
                eprintln!(
                    "reduce_eval_inner.mem step={} env_level={} terms_len={} errors_len={} rss_kb={} delta_prev_kb={} delta_total_kb={}",
                    step,
                    env_level,
                    terms_len.map(|v| v as i64).unwrap_or(-1),
                    errors_len.map(|v| v as i64).unwrap_or(-1),
                    curr,
                    delta_prev,
                    curr as i64 - baseline as i64
                );
                rss_prev = Some(curr);
                if rss_baseline.is_none() {
                    rss_baseline = Some(curr);
                }
            }
        };
        log_mem_step("start", None, None);

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
        log_mem_step("after_collect_terms", Some(terms.len()), None);

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
            log_mem_step("term_split_limit_exceeded", Some(terms.len()), None);
            Err(InterpreterError::ReduceError(format!(
                "The number of terms in the Par is {}, which exceeds the limit of {}",
                terms.len(),
                term_split_limit
            )))
        } else {
            // Collect errors from all parallel execution paths (pars)
            // parTraverseSafe
            log_mem_step("before_build_futures", Some(terms.len()), None);
            let futures: Vec<
                Pin<
                    Box<
                        dyn futures::Future<Output = Result<(), InterpreterError>>
                            + std::marker::Send,
                    >,
                >,
            > = terms
                .iter()
                .enumerate()
                .map(|(index, term)| {
                    let self_clone = self.clone();
                    let term_clone = term.clone();
                    let rand_split = split(index.try_into().unwrap(), &terms, rand.clone());
                    Box::pin(async move {
                        self_clone
                            .generated_message_eval(&term_clone, env, rand_split)
                            .await
                    })
                        as Pin<
                            Box<
                                dyn futures::Future<Output = Result<(), InterpreterError>>
                                    + std::marker::Send,
                            >,
                        >
                })
                .collect();
            log_mem_step("after_build_futures", Some(futures.len()), None);
            log_mem_step("before_join_all", Some(terms.len()), None);

            let results: Vec<Result<(), InterpreterError>> =
                futures::future::join_all(futures).await;
            log_mem_step("after_join_all", Some(terms.len()), None);
            let (ok_count, err_count) =
                results.iter().fold((0usize, 0usize), |(ok, err), result| {
                    if result.is_ok() {
                        (ok + 1, err)
                    } else {
                        (ok, err + 1)
                    }
                });
            if mem_profile_enabled && env_level == 0 {
                eprintln!(
                    "reduce_eval_inner.meta step=after_join_all results_len={} results_cap={} ok_count={} err_count={}",
                    results.len(),
                    results.capacity(),
                    ok_count,
                    err_count
                );
            }
            log_mem_step("after_scan_results", Some(terms.len()), Some(err_count));
            log_mem_step("before_collect_errors", Some(terms.len()), Some(err_count));
            let mut flattened_results: Vec<InterpreterError> = Vec::with_capacity(err_count);
            for result in results {
                if let Err(err) = result {
                    flattened_results.push(err);
                }
            }
            log_mem_step(
                "after_collect_errors",
                Some(terms.len()),
                Some(flattened_results.len()),
            );
            log_mem_step(
                "after_flatten_errors",
                Some(terms.len()),
                Some(flattened_results.len()),
            );
            log_mem_step(
                "before_aggregate",
                Some(terms.len()),
                Some(flattened_results.len()),
            );

            match self.aggregate_evaluator_errors(flattened_results) {
                Ok(_) => {
                    log_mem_step("after_aggregate_ok", Some(terms.len()), Some(0));
                    Ok(())
                }
                Err(e) => {
                    log_mem_step("after_aggregate_err", Some(terms.len()), None);
                    Err(e)
                }
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
    fn produce<'a>(
        &'a self,
        chan: Par,
        data: ListParWithRandom,
        persistent: bool,
    ) -> Pin<
        Box<
            dyn std::future::Future<Output = Result<DispatchType, InterpreterError>>
                + std::marker::Send
                + 'a,
        >,
    > {
        Box::pin(StackGrowingFuture {
            inner: self.produce_inner(chan, data, persistent),
        })
    }

    async fn produce_inner(
        &self,
        chan: Par,
        data: ListParWithRandom,
        persistent: bool,
    ) -> Result<DispatchType, InterpreterError> {
        let op_mem_profile_enabled = *REDUCE_OP_PROFILE_ENABLED;
        let data_len = data.pars.len();
        let mut op_rss_prev = if op_mem_profile_enabled {
            read_vm_rss_kb()
        } else {
            None
        };
        let mut log_op_step = |step: &str| {
            if !op_mem_profile_enabled {
                return;
            }
            if let Some(curr) = read_vm_rss_kb() {
                let prev = op_rss_prev.unwrap_or(curr);
                let delta = curr as i64 - prev as i64;
                if delta != 0 {
                    eprintln!(
                        "reduce_op.mem fn=produce_inner step={} persistent={} data_len={} rss_kb={} delta_prev_kb={}",
                        step,
                        persistent,
                        data_len,
                        curr,
                        delta
                    );
                }
                op_rss_prev = Some(curr);
            }
        };
        log_op_step("start");
        // println!("\nreduce produce");
        // println!("chan in reduce produce: {:?}", chan);
        // println!("data in reduce produce: {:?}", data);
        self.update_mergeable_channels(&chan).await;
        log_op_step("after_update_mergeable_channels");

        // println!("Attempting to lock space for produce");
        let mut space_locked = self.space.try_lock().unwrap();
        // println!("Locked space for produce");
        let produce_result = space_locked.produce(chan.clone(), data.clone(), persistent)?;
        let is_replay = space_locked.is_replay();
        drop(space_locked);
        log_op_step("after_space_produce");

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
                        produce_event.failed,
                    )
                    .await?;
                log_op_step("after_continue_produce_process");

                match dispatch_type {
                    DispatchType::NonDeterministicCall(ref output) => {
                        let produce1 = produce_event.mark_as_non_deterministic(output.clone());
                        let mut space_locked = self.space.try_lock().unwrap();
                        space_locked.update_produce(produce1);
                        drop(space_locked);
                        log_op_step("after_update_produce_nondeterministic");
                        Ok(dispatch_type)
                    }

                    DispatchType::FailedNonDeterministicCall(error) => {
                        // Mark the produce as failed for replay safety
                        let failed_produce = produce_event.with_error();
                        let mut space_locked = self.space.try_lock().unwrap();
                        space_locked.update_produce(failed_produce);
                        drop(space_locked);
                        log_op_step("after_update_produce_failed_nondeterministic");
                        // Wrap the original error in NonDeterministicProcessFailure
                        Err(InterpreterError::NonDeterministicProcessFailure {
                            cause: Box::new(error),
                            output_not_produced: vec![],
                        })
                    }

                    _ => Ok(dispatch_type),
                }
            }
            None => Ok(DispatchType::Skip),
        }
    }

    fn consume<'a>(
        &'a self,
        binds: Vec<(BindPattern, Par)>,
        body: ParWithRandom,
        persistent: bool,
        peek: bool,
    ) -> Pin<
        Box<
            dyn std::future::Future<Output = Result<DispatchType, InterpreterError>>
                + std::marker::Send
                + 'a,
        >,
    > {
        Box::pin(StackGrowingFuture {
            inner: self.consume_inner(binds, body, persistent, peek),
        })
    }

    async fn consume_inner(
        &self,
        binds: Vec<(BindPattern, Par)>,
        body: ParWithRandom,
        persistent: bool,
        peek: bool,
    ) -> Result<DispatchType, InterpreterError> {
        let op_mem_profile_enabled = *REDUCE_OP_PROFILE_ENABLED;
        let binds_len = binds.len();
        let mut op_rss_prev = if op_mem_profile_enabled {
            read_vm_rss_kb()
        } else {
            None
        };
        let mut log_op_step = |step: &str, sources_len: usize| {
            if !op_mem_profile_enabled {
                return;
            }
            if let Some(curr) = read_vm_rss_kb() {
                let prev = op_rss_prev.unwrap_or(curr);
                let delta = curr as i64 - prev as i64;
                if delta != 0 {
                    eprintln!(
                        "reduce_op.mem fn=consume_inner step={} persistent={} peek={} binds_len={} sources_len={} rss_kb={} delta_prev_kb={}",
                        step,
                        persistent,
                        peek,
                        binds_len,
                        sources_len,
                        curr,
                        delta
                    );
                }
                op_rss_prev = Some(curr);
            }
        };
        // println!("\nreduce consume");
        // println!("binds in reduce consume: {:?}", binds);
        // println!("body in reduce consume: {:?}", body);
        let (patterns, sources): (Vec<BindPattern>, Vec<Par>) = binds.clone().into_iter().unzip();
        log_op_step("after_split_binds", sources.len());

        // Update mergeable channels
        for source in &sources {
            self.update_mergeable_channels(source).await;
        }
        log_op_step("after_update_mergeable_channels", sources.len());

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
        log_op_step("after_space_consume", sources.len());

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
        .map(|dispatch| {
            log_op_step("after_continue_consume_process", sources.len());
            dispatch
        })
    }

    async fn continue_produce_process(
        &self,
        res: Application,
        chan: Par,
        data: ListParWithRandom,
        persistent: bool,
        is_replay: bool,
        previous_output: Vec<Vec<u8>>,
        trace_failed: bool,
    ) -> Result<DispatchType, InterpreterError> {
        // println!("\ncontinue_produce_process");
        // During replay, if the trace shows a failed non-deterministic process,
        // we cannot replay it - the external service call failed during original execution
        if is_replay && trace_failed {
            return Err(InterpreterError::CanNotReplayFailedNonDeterministicProcess);
        }

        let previous_output_as_par = self.decode_previous_output(previous_output)?;
        let Some(state) = self.build_post_match_produce_exec_state(
            res,
            chan,
            data,
            persistent,
            is_replay,
            previous_output_as_par,
        ) else {
            return Ok(DispatchType::Skip);
        };

        self.execute_post_match_with_runtime_policy(state).await
    }

    fn decode_previous_output(
        &self,
        previous_output: Vec<Vec<u8>>,
    ) -> Result<Vec<Par>, InterpreterError> {
        previous_output
            .into_iter()
            .map(|bytes| {
                Par::decode(&bytes[..]).map_err(|e| InterpreterError::DecodeError(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()
    }

    fn build_post_match_exec_state(
        &self,
        res: Application,
        binds: Vec<(BindPattern, Par)>,
        body: ParWithRandom,
        persistent: bool,
        peek: bool,
        is_replay: bool,
        previous_output: Vec<Par>,
    ) -> Option<PostMatchExecState> {
        res.map(|(continuation, data_list, matched_peek)| {
            let control = if persistent {
                PostMatchControlState::DispatchAndReconsume(ReceiveRegistrationState {
                    binds,
                    body,
                    persistent,
                    peek,
                })
            } else if matched_peek {
                PostMatchControlState::DispatchAndRestorePeeks
            } else {
                PostMatchControlState::DispatchOnly
            };

            PostMatchExecState {
                dispatch: DispatchExecState {
                    continuation,
                    data_list,
                    is_replay,
                    previous_output,
                    _remaining_phlo: self.cost.get().value,
                },
                control,
            }
        })
    }

    fn build_post_match_produce_exec_state(
        &self,
        res: Application,
        chan: Par,
        data: ListParWithRandom,
        persistent: bool,
        is_replay: bool,
        previous_output: Vec<Par>,
    ) -> Option<PostMatchExecState> {
        res.map(|(continuation, data_list, matched_peek)| {
            let control = if persistent {
                PostMatchControlState::DispatchAndReproduce(ProduceRegistrationState {
                    chan,
                    data,
                    persistent,
                })
            } else if matched_peek {
                PostMatchControlState::DispatchAndRestorePeeks
            } else {
                PostMatchControlState::DispatchOnly
            };

            PostMatchExecState {
                dispatch: DispatchExecState {
                    continuation,
                    data_list,
                    is_replay,
                    previous_output,
                    _remaining_phlo: self.cost.get().value,
                },
                control,
            }
        })
    }

    async fn reduce_step(
        &self,
        exec_state: PostMatchExecState,
        gas_limit: i64,
    ) -> Result<StepResult, InterpreterError> {
        if should_suspend_for_step(gas_limit) {
            return Ok(StepResult {
                dispatch_result: DispatchType::Skip,
                emitted_effects: false,
                next_state: Some(exec_state),
                consumed_gas: 0,
                terminal_status: StepTerminalStatus::Suspended,
            });
        }

        let PostMatchExecState { dispatch, control } = exec_state;
        let remaining_before = dispatch._remaining_phlo;

        match control {
            PostMatchControlState::DispatchOnly => {
                let dispatch_result = self
                    .dispatch(
                        dispatch.continuation,
                        dispatch.data_list,
                        dispatch.is_replay,
                        dispatch.previous_output,
                    )
                    .await?;
                Ok(self.completed_step_result(dispatch_result, remaining_before))
            }
            PostMatchControlState::DispatchAndReconsume(receive_state) => {
                let dispatch_state = dispatch.clone();
                let dispatch_result = self
                    .dispatch(
                        dispatch.continuation,
                        dispatch.data_list,
                        dispatch.is_replay,
                        dispatch.previous_output,
                    )
                    .await?;
                Ok(self.suspended_step_result(
                    dispatch_state,
                    dispatch_result,
                    PostMatchControlState::Reconsume(receive_state),
                    remaining_before,
                ))
            }
            PostMatchControlState::DispatchAndReproduce(produce_state) => {
                let dispatch_state = dispatch.clone();
                let dispatch_result = self
                    .dispatch(
                        dispatch.continuation,
                        dispatch.data_list,
                        dispatch.is_replay,
                        dispatch.previous_output,
                    )
                    .await?;
                Ok(self.suspended_step_result(
                    dispatch_state,
                    dispatch_result,
                    PostMatchControlState::Reproduce(produce_state),
                    remaining_before,
                ))
            }
            PostMatchControlState::DispatchAndRestorePeeks => {
                let dispatch_state = dispatch.clone();
                let dispatch_result = self
                    .dispatch(
                        dispatch.continuation,
                        dispatch.data_list,
                        dispatch.is_replay,
                        dispatch.previous_output,
                    )
                    .await?;
                Ok(self.suspended_step_result(
                    dispatch_state,
                    dispatch_result,
                    PostMatchControlState::RestorePeeks,
                    remaining_before,
                ))
            }
            PostMatchControlState::Reconsume(receive_state) => {
                let reduce_result = self
                    .consume(
                        receive_state.binds,
                        receive_state.body,
                        receive_state.persistent,
                        receive_state.peek,
                    )
                    .await?;
                Ok(self.completed_step_result(reduce_result, remaining_before))
            }
            PostMatchControlState::Reproduce(produce_state) => {
                let reduce_result = self
                    .produce(
                        produce_state.chan,
                        produce_state.data,
                        produce_state.persistent,
                    )
                    .await?;
                Ok(self.completed_step_result(reduce_result, remaining_before))
            }
            PostMatchControlState::RestorePeeks => {
                let restore_result = self.restore_peeks(dispatch.data_list).await?;
                Ok(self.completed_step_result(restore_result, remaining_before))
            }
        }
    }

    fn completed_step_result(&self, dispatch_result: DispatchType, remaining_before: i64) -> StepResult {
        StepResult {
            emitted_effects: dispatch_emits_effects(&dispatch_result),
            dispatch_result,
            next_state: None,
            consumed_gas: compute_consumed_gas(remaining_before, self.cost.get().value),
            terminal_status: StepTerminalStatus::Completed,
        }
    }

    fn suspended_step_result(
        &self,
        dispatch_state: DispatchExecState,
        dispatch_result: DispatchType,
        next_control: PostMatchControlState,
        remaining_before: i64,
    ) -> StepResult {
        let next_state = PostMatchExecState {
            dispatch: DispatchExecState {
                _remaining_phlo: self.cost.get().value,
                ..dispatch_state
            },
            control: next_control,
        };

        StepResult {
            emitted_effects: dispatch_emits_effects(&dispatch_result),
            dispatch_result,
            next_state: Some(next_state),
            consumed_gas: compute_consumed_gas(remaining_before, self.cost.get().value),
            terminal_status: StepTerminalStatus::Suspended,
        }
    }

    async fn execute_post_match_to_completion(
        &self,
        mut state: PostMatchExecState,
        gas_limit_per_step: i64,
    ) -> Result<DispatchType, InterpreterError> {
        loop {
            let step_result = self.reduce_step(state, gas_limit_per_step).await?;
            let StepResult {
                dispatch_result,
                emitted_effects: _emitted_effects,
                next_state,
                consumed_gas: _consumed_gas,
                terminal_status,
            } = step_result;

            match terminal_status {
                StepTerminalStatus::Completed => return Ok(dispatch_result),
                StepTerminalStatus::Suspended => {
                    state = next_state.ok_or_else(|| {
                        InterpreterError::BugFoundError(
                            "reduce_step suspended without next state".to_string(),
                        )
                    })?;
                }
            }
        }
    }

    async fn execute_post_match_with_runtime_policy(
        &self,
        state: PostMatchExecState,
    ) -> Result<DispatchType, InterpreterError> {
        let split_config = *self.split_post_match_config.read().unwrap();
        if !split_config.enabled {
            return self.execute_post_match_to_completion(state, i64::MAX).await;
        }

        let step_result = self
            .reduce_step(state, split_config.initial_step_gas_limit)
            .await?;
        self.handle_split_step_result(step_result, split_config.initial_step_gas_limit)
    }

    fn handle_split_step_result(
        &self,
        step_result: StepResult,
        gas_limit_per_step: i64,
    ) -> Result<DispatchType, InterpreterError> {
        let StepResult {
            dispatch_result,
            emitted_effects,
            next_state,
            consumed_gas,
            terminal_status,
        } = step_result;

        debug!(
            target: "f1r3fly.rholang",
            split_post_match_enabled = true,
            emitted_effects,
            consumed_gas,
            gas_limit_per_step,
            terminal_status = match terminal_status {
                StepTerminalStatus::Completed => "completed",
                StepTerminalStatus::Suspended => "suspended",
            },
            "post-match split step completed",
        );

        match terminal_status {
            StepTerminalStatus::Completed => {
                metrics::counter!("rholang.split_post_match.completed").increment(1);
                Ok(dispatch_result)
            }
            StepTerminalStatus::Suspended => {
                metrics::counter!("rholang.split_post_match.suspended").increment(1);
                let suspended_state = next_state.ok_or_else(|| {
                    InterpreterError::BugFoundError(
                        "split step suspended without next state".to_string(),
                    )
                })?;
                let _ = self.persist_suspended_post_match_state(
                    suspended_state,
                    gas_limit_per_step,
                )?;
                Ok(dispatch_result)
            }
        }
    }

    fn persist_suspended_post_match_state(
        &self,
        state: PostMatchExecState,
        gas_limit_per_step: i64,
    ) -> Result<ContinuationHandle, InterpreterError> {
        let serialized_state = serialize_post_match_state(&state)?;
        let origin_reference = state.dispatch.continuation.encode_to_vec();
        let origin = Blake2b256::hash(origin_reference.clone()).to_vec();
        let nonce = self.continuation_nonce.fetch_add(1, Ordering::SeqCst);
        let handle = ContinuationHandle::new(origin, nonce);

        let create = CreateContinuation {
            handle: handle.clone(),
            origin_reference,
            serialized_state,
            gas_limit_per_step,
            funding_policy: FundingPolicy::ProducerOnly,
        };

        let mut store = self.continuation_store.write().unwrap();
        store
            .create(create)
            .map_err(|e| {
                InterpreterError::BugFoundError(format!(
                    "failed to persist suspended continuation: {e:?}"
                ))
            })?;

        metrics::counter!("rholang.split_post_match.continuation_persisted").increment(1);
        debug!(
            target: "f1r3fly.rholang",
            continuation_nonce = nonce,
            "persisted suspended post-match continuation",
        );

        Ok(handle)
    }

    /// Compatibility path: run the post-match machine in a single full call.
    /// Kept for equivalence testing against `reduce_step` progression.
    #[cfg(test)]
    async fn execute_post_match_exec_state_full(
        &self,
        state: PostMatchExecState,
    ) -> Result<DispatchType, InterpreterError> {
        let PostMatchExecState { dispatch, control } = state;

        match control {
            PostMatchControlState::DispatchOnly => {
                self.dispatch(
                    dispatch.continuation,
                    dispatch.data_list,
                    dispatch.is_replay,
                    dispatch.previous_output,
                )
                .await
            }
            PostMatchControlState::DispatchAndReconsume(receive_state) => {
                let self_clone1 = self.clone();
                let self_clone2 = self.clone();

                let dispatch_fut = self_clone1.dispatch(
                    dispatch.continuation,
                    dispatch.data_list,
                    dispatch.is_replay,
                    dispatch.previous_output,
                );
                let consume_fut = self_clone2.consume(
                    receive_state.binds,
                    receive_state.body,
                    receive_state.persistent,
                    receive_state.peek,
                );

                let futures: Vec<DispatchResultFuture<'_>> = vec![
                    Box::pin(dispatch_fut) as DispatchResultFuture<'_>,
                    Box::pin(consume_fut) as DispatchResultFuture<'_>,
                ];

                let results: Vec<Result<DispatchType, InterpreterError>> =
                    futures::future::join_all(futures).await;
                let flattened_results: Vec<InterpreterError> = results
                    .into_iter()
                    .filter_map(|result| result.err())
                    .collect();

                self.aggregate_evaluator_errors(flattened_results)
            }
            PostMatchControlState::DispatchAndReproduce(produce_state) => {
                let self_clone1 = self.clone();
                let self_clone2 = self.clone();

                let dispatch_fut = self_clone1.dispatch(
                    dispatch.continuation,
                    dispatch.data_list,
                    dispatch.is_replay,
                    dispatch.previous_output,
                );
                let produce_fut =
                    self_clone2.produce(produce_state.chan, produce_state.data, produce_state.persistent);

                let futures: Vec<DispatchResultFuture<'_>> = vec![
                    Box::pin(dispatch_fut) as DispatchResultFuture<'_>,
                    Box::pin(produce_fut) as DispatchResultFuture<'_>,
                ];

                let results: Vec<Result<DispatchType, InterpreterError>> =
                    futures::future::join_all(futures).await;
                let flattened_results: Vec<InterpreterError> = results
                    .into_iter()
                    .filter_map(|result| result.err())
                    .collect();

                self.aggregate_evaluator_errors(flattened_results)
            }
            PostMatchControlState::DispatchAndRestorePeeks => {
                let self_clone = self.clone();
                let dispatch_state = dispatch.clone();

                let mut futures: Vec<DispatchResultFuture<'_>> = vec![Box::pin(async move {
                    self_clone
                        .dispatch(
                            dispatch_state.continuation,
                            dispatch_state.data_list,
                            dispatch_state.is_replay,
                            dispatch_state.previous_output,
                        )
                        .await
                }) as DispatchResultFuture<'_>];
                futures.extend(self.produce_peeks(dispatch.data_list).await);

                let results: Vec<Result<DispatchType, InterpreterError>> =
                    futures::future::join_all(futures).await;
                let flattened_results: Vec<InterpreterError> = results
                    .into_iter()
                    .filter_map(|result| result.err())
                    .collect();

                self.aggregate_evaluator_errors(flattened_results)
            }
            PostMatchControlState::Reconsume(receive_state) => {
                self.consume(
                    receive_state.binds,
                    receive_state.body,
                    receive_state.persistent,
                    receive_state.peek,
                )
                .await
            }
            PostMatchControlState::Reproduce(produce_state) => {
                self.produce(produce_state.chan, produce_state.data, produce_state.persistent)
                    .await
            }
            PostMatchControlState::RestorePeeks => self.restore_peeks(dispatch.data_list).await,
        }
    }

    async fn restore_peeks(&self, data_list: Vec<MatchedData>) -> Result<DispatchType, InterpreterError> {
        let futures = self.produce_peeks(data_list).await;
        let results: Vec<Result<DispatchType, InterpreterError>> = futures::future::join_all(futures).await;
        let flattened_results: Vec<InterpreterError> = results
            .into_iter()
            .filter_map(|result| result.err())
            .collect();

        self.aggregate_evaluator_errors(flattened_results)
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
        let previous_output_as_par = self.decode_previous_output(previous_output)?;
        let Some(state) = self.build_post_match_exec_state(
            res,
            binds,
            body,
            persistent,
            peek,
            is_replay,
            previous_output_as_par,
        ) else {
            return Ok(DispatchType::Skip);
        };

        self.execute_post_match_with_runtime_policy(state).await
    }

    fn dispatch<'a>(
        &'a self,
        continuation: TaggedContinuation,
        data_list: Vec<MatchedData>,
        is_replay: bool,
        previous_output: Vec<Par>,
    ) -> Pin<
        Box<
            dyn std::future::Future<Output = Result<DispatchType, InterpreterError>>
                + std::marker::Send
                + 'a,
        >,
    > {
        Box::pin(StackGrowingFuture {
            inner: self.dispatch_inner(continuation, data_list, is_replay, previous_output),
        })
    }

    async fn dispatch_inner(
        &self,
        continuation: TaggedContinuation,
        data_list: Vec<MatchedData>,
        is_replay: bool,
        previous_output: Vec<Par>,
    ) -> Result<DispatchType, InterpreterError> {
        let op_mem_profile_enabled = *REDUCE_OP_PROFILE_ENABLED;
        let data_list_len = data_list.len();
        let previous_output_len = previous_output.len();
        let mut op_rss_prev = if op_mem_profile_enabled {
            read_vm_rss_kb()
        } else {
            None
        };
        let mut log_op_step = |step: &str| {
            if !op_mem_profile_enabled {
                return;
            }
            if let Some(curr) = read_vm_rss_kb() {
                let prev = op_rss_prev.unwrap_or(curr);
                let delta = curr as i64 - prev as i64;
                if delta != 0 {
                    eprintln!(
                        "reduce_op.mem fn=dispatch_inner step={} is_replay={} data_list_len={} prev_output_len={} rss_kb={} delta_prev_kb={}",
                        step,
                        is_replay,
                        data_list_len,
                        previous_output_len,
                        curr,
                        delta
                    );
                }
                op_rss_prev = Some(curr);
            }
        };
        log_op_step("start");
        // println!("\nreduce dispatch");
        let result = self
            .dispatcher
            .dispatch(
                continuation,
                data_list.into_iter().map(|tuple| tuple.1).collect(),
                is_replay,
                previous_output,
            )
            .await;
        log_op_step("after_dispatch");
        result
    }

    async fn produce_peeks(
        &self,
        data_list: Vec<MatchedData>,
    ) -> Vec<
        Pin<
            Box<
                dyn futures::Future<Output = Result<DispatchType, InterpreterError>>
                    + std::marker::Send,
            >,
        >,
    > {
        // println!("\nreduce produce_peeks");
        data_list
            .into_iter()
            .filter(|(_, _, _, persist)| !persist)
            .map(|(chan, _, removed_data, _)| {
                let self_clone = self.clone();
                Box::pin(async move { self_clone.produce(chan, removed_data, false).await })
                    as Pin<
                        Box<
                            dyn futures::Future<Output = Result<DispatchType, InterpreterError>>
                                + std::marker::Send,
                        >,
                    >
            })
            .collect()
    }

    /* Collect mergeable channels */

    async fn update_mergeable_channels(&self, chan: &Par) -> () {
        let is_mergeable = self.is_mergeable_channel(chan);
        // println!("\nis_mergeable: {:?}", is_mergeable);

        if is_mergeable {
            {
                let mut merge_chs_write = self.merge_chs.write().unwrap();
                merge_chs_write.insert(chan.clone());
            }
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

            // Out Of Phlogiston or User Abort error is always single
            // - if one execution path hits these, the whole evaluation stops as well
            // UserAbortError takes precedence over OutOfPhlogistonsError
            // Use single-pass find() to avoid double iteration
            err_list
                if err_list
                    .iter()
                    .find(|e| matches!(e, InterpreterError::UserAbortError))
                    .is_some() =>
            {
                Err(InterpreterError::UserAbortError)
            }

            err_list
                if err_list
                    .iter()
                    .find(|e| matches!(e, InterpreterError::OutOfPhlogistonsError))
                    .is_some() =>
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
        let op_mem_profile_enabled = *REDUCE_OP_PROFILE_ENABLED;
        let mut op_rss_prev = if op_mem_profile_enabled {
            read_vm_rss_kb()
        } else {
            None
        };
        let mut log_op_step = |step: &str| {
            if !op_mem_profile_enabled {
                return;
            }
            if let Some(curr) = read_vm_rss_kb() {
                let prev = op_rss_prev.unwrap_or(curr);
                let delta = curr as i64 - prev as i64;
                if delta != 0 {
                    eprintln!(
                        "reduce_op.mem fn=eval_send step={} env_level={} send_data_len={} persistent={} rss_kb={} delta_prev_kb={}",
                        step,
                        env.level,
                        send.data.len(),
                        send.persistent,
                        curr,
                        delta
                    );
                }
                op_rss_prev = Some(curr);
            }
        };
        log_op_step("start");
        // println!("\nenv in eval_send: {:?}", env);
        self.cost.charge(send_eval_cost())?;
        let eval_chan = self.eval_expr(&unwrap_option_safe(send.chan.clone())?, env)?;
        log_op_step("after_eval_chan");
        let sub_chan = self.substitute.substitute_and_charge(&eval_chan, 0, env)?;
        log_op_step("after_substitute_chan");
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

        let subst_data = send
            .data
            .iter()
            .map(|expr| {
                let evaluated = self.eval_expr(expr, env)?;
                self.substitute.substitute_and_charge(&evaluated, 0, env)
            })
            .collect::<Result<Vec<_>, InterpreterError>>()?;
        log_op_step("after_substitute_data");

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
        log_op_step("after_produce");
        Ok(())
    }

    async fn eval_receive(
        &self,
        receive: &Receive,
        env: &Env<Par>,
        rand: Blake2b512Random,
    ) -> Result<(), InterpreterError> {
        let op_mem_profile_enabled = *REDUCE_OP_PROFILE_ENABLED;
        let body_locally_free = receive
            .body
            .as_ref()
            .map(|b| b.locally_free.clone())
            .unwrap_or_default();
        let body_subst_shift = env.shift + receive.bind_count;
        let body_needs_subst = if body_subst_shift <= 0 {
            body_locally_free.iter().any(|bit| *bit == 1)
        } else {
            let s = body_subst_shift as usize;
            body_locally_free
                .iter()
                .enumerate()
                .any(|(idx, bit)| *bit == 1 && idx >= s)
        };
        let mut op_rss_prev = if op_mem_profile_enabled {
            read_vm_rss_kb()
        } else {
            None
        };
        let mut log_op_step = |step: &str| {
            if !op_mem_profile_enabled {
                return;
            }
            if let Some(curr) = read_vm_rss_kb() {
                let prev = op_rss_prev.unwrap_or(curr);
                let delta = curr as i64 - prev as i64;
                if delta != 0 {
                    eprintln!(
                        "reduce_op.mem fn=eval_receive step={} env_level={} binds_len={} bind_count={} body_subst_shift={} body_needs_subst={} persistent={} peek={} rss_kb={} delta_prev_kb={}",
                        step,
                        env.level,
                        receive.binds.len(),
                        receive.bind_count,
                        body_subst_shift,
                        body_needs_subst,
                        receive.persistent,
                        receive.peek,
                        curr,
                        delta
                    );
                }
                op_rss_prev = Some(curr);
            }
        };
        log_op_step("start");
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
        log_op_step("after_build_binds");

        // TODO: Allow for the environment to be stored with the body in the Tuplespace - OLD
        let subst_body = self.substitute.substitute_no_sort_and_charge(
            receive.body.as_ref().unwrap(),
            0,
            &env.shift(receive.bind_count),
        )?;
        log_op_step("after_substitute_body");

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
        log_op_step("after_consume");
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
        let op_mem_profile_enabled = *REDUCE_OP_PROFILE_ENABLED;
        let mut op_rss_prev = if op_mem_profile_enabled {
            read_vm_rss_kb()
        } else {
            None
        };
        let mut log_op_step = |step: &str| {
            if !op_mem_profile_enabled {
                return;
            }
            if let Some(curr) = read_vm_rss_kb() {
                let prev = op_rss_prev.unwrap_or(curr);
                let delta = curr as i64 - prev as i64;
                if delta != 0 {
                    eprintln!(
                        "reduce_op.mem fn=eval_new step={} env_level={} bind_count={} uri_len={} rss_kb={} delta_prev_kb={}",
                        step,
                        env.level,
                        new.bind_count,
                        new.uri.len(),
                        curr,
                        delta
                    );
                }
                op_rss_prev = Some(curr);
            }
        };
        log_op_step("start");
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
        log_op_step("after_charge_new_bindings");
        // println!("\nnew uri: {:?}", new.uri);
        match alloc(new.bind_count as usize, new.uri.clone()) {
            Ok(env) => {
                log_op_step("after_alloc");
                // println!("\nenv in eval_new: {:?}", env);
                let result = self
                    .eval(unwrap_option_safe(new.p.clone())?, &env, rand)
                    .await;
                log_op_step("after_eval_new_body");
                result
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
                (ExprInstance::GBool(b1), ExprInstance::GBool(b2)) => {
                    self.cost.charge(comparison_cost())?;
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GBool(relopb(b1, b2))),
                    })
                }

                (ExprInstance::GInt(i1), ExprInstance::GInt(i2)) => {
                    self.cost.charge(comparison_cost())?;
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GBool(relopi(i1, i2))),
                    })
                }

                (ExprInstance::GString(s1), ExprInstance::GString(s2)) => {
                    self.cost.charge(comparison_cost())?;
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GBool(relops(s1, s2))),
                    })
                }

                (ExprInstance::GDouble(d1), ExprInstance::GDouble(d2)) => {
                    self.cost.charge(comparison_cost())?;
                    let f1 = f64::from_bits(d1);
                    let f2 = f64::from_bits(d2);
                    if f1.is_nan() || f2.is_nan() {
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::GBool(false)),
                        })
                    } else {
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::GBool(relopi(
                                f1.partial_cmp(&f2).map_or(0, |o| o as i64),
                                0,
                            ))),
                        })
                    }
                }

                (ExprInstance::GBigInt(b1), ExprInstance::GBigInt(b2)) => {
                    self.cost.charge(bigint_comparison_cost(b1.len(), b2.len()))?;
                    let cmp = compare_twos_complement_bytes(&b1, &b2);
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GBool(relopi(cmp as i64, 0))),
                    })
                }

                (ExprInstance::GBigRat(r1), ExprInstance::GBigRat(r2)) => {
                    self.cost.charge(bigrat_comparison_cost(
                        r1.numerator.len(), r1.denominator.len(),
                        r2.numerator.len(), r2.denominator.len(),
                    ))?;
                    let cmp = compare_big_rationals(&r1, &r2);
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GBool(relopi(cmp as i64, 0))),
                    })
                }

                (ExprInstance::GFixedPoint(fp1), ExprInstance::GFixedPoint(fp2)) => {
                    self.cost.charge(bigint_comparison_cost(fp1.unscaled.len(), fp2.unscaled.len()))?;
                    let cmp = compare_fixed_points(&fp1, &fp2)?;
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GBool(relopi(cmp as i64, 0))),
                    })
                }

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

                ExprInstance::GDouble(x) => Ok(Expr {
                    expr_instance: Some(ExprInstance::GDouble(*x)),
                }),

                ExprInstance::GBigInt(x) => Ok(Expr {
                    expr_instance: Some(ExprInstance::GBigInt(x.clone())),
                }),

                ExprInstance::GBigRat(x) => Ok(Expr {
                    expr_instance: Some(ExprInstance::GBigRat(x.clone())),
                }),

                ExprInstance::GFixedPoint(x) => Ok(Expr {
                    expr_instance: Some(ExprInstance::GFixedPoint(x.clone())),
                }),

                ExprInstance::ENotBody(enot) => {
                    let b = self.eval_to_bool(&enot.p.as_ref().unwrap(), env)?;
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GBool(!b)),
                    })
                }

                ExprInstance::ENegBody(eneg) => {
                    let v = self.eval_single_expr(&eneg.p.as_ref().unwrap(), env)?;
                    match v.expr_instance.unwrap() {
                        ExprInstance::GInt(i) => {
                            let result = i.checked_neg().ok_or_else(|| {
                                InterpreterError::ReduceError(
                                    "Arithmetic overflow in negation".to_string(),
                                )
                            })?;
                            Ok(Expr {
                                expr_instance: Some(ExprInstance::GInt(result)),
                            })
                        }
                        ExprInstance::GDouble(bits) => {
                            let f = f64::from_bits(bits);
                            Ok(Expr {
                                expr_instance: Some(ExprInstance::GDouble((-f).to_bits())),
                            })
                        }
                        ExprInstance::GBigInt(bytes) => {
                            self.cost.charge(bigint_negation_cost(bytes.len()))?;
                            make_bigint_expr(negate_twos_complement(&bytes), "negation")
                        }
                        ExprInstance::GBigRat(rat) => {
                            self.cost.charge(bigrat_negation_cost(rat.numerator.len()))?;
                            make_bigrat_expr(
                                models::rhoapi::GBigRational {
                                    numerator: negate_twos_complement(&rat.numerator),
                                    denominator: rat.denominator,
                                },
                                "negation",
                            )
                        }
                        ExprInstance::GFixedPoint(fp) => {
                            self.cost.charge(bigint_negation_cost(fp.unscaled.len()))?;
                            make_fixedpoint_expr(
                                models::rhoapi::GFixedPoint {
                                    unscaled: negate_twos_complement(&fp.unscaled),
                                    scale: fp.scale,
                                },
                                "negation",
                            )
                        }
                        other => Err(InterpreterError::OperatorNotDefined {
                            op: "neg".to_string(),
                            other_type: get_type(other),
                        }),
                    }
                }

                ExprInstance::EMultBody(EMult { p1, p2 }) => {
                    let v1 = self.eval_single_expr(&p1.clone().unwrap(), env)?;
                    let v2 = self.eval_single_expr(&p2.clone().unwrap(), env)?;

                    match (v1.expr_instance.unwrap(), v2.expr_instance.unwrap()) {
                        (ExprInstance::GInt(lhs), ExprInstance::GInt(rhs)) => {
                            self.cost.charge(multiplication_cost())?;
                            let result = lhs.checked_mul(rhs).ok_or_else(|| {
                                InterpreterError::ReduceError(
                                    "Arithmetic overflow in multiplication".to_string(),
                                )
                            })?;
                            Ok(Expr {
                                expr_instance: Some(ExprInstance::GInt(result)),
                            })
                        }
                        (ExprInstance::GDouble(d1), ExprInstance::GDouble(d2)) => {
                            self.cost.charge(multiplication_cost())?;
                            let result = f64::from_bits(d1) * f64::from_bits(d2);
                            Ok(Expr {
                                expr_instance: Some(ExprInstance::GDouble(result.to_bits())),
                            })
                        }
                        (ExprInstance::GBigInt(b1), ExprInstance::GBigInt(b2)) => {
                            self.cost.charge(bigint_multiplication_cost(b1.len(), b2.len()))?;
                            make_bigint_expr(multiply_twos_complement(&b1, &b2), "multiplication")
                        }
                        (ExprInstance::GBigRat(r1), ExprInstance::GBigRat(r2)) => {
                            self.cost.charge(bigrat_multiplication_cost(
                                r1.numerator.len(), r1.denominator.len(),
                                r2.numerator.len(), r2.denominator.len(),
                            ))?;
                            make_bigrat_expr(multiply_big_rationals(&r1, &r2), "multiplication")
                        }
                        (ExprInstance::GFixedPoint(fp1), ExprInstance::GFixedPoint(fp2)) => {
                            if fp1.scale != fp2.scale {
                                return Err(InterpreterError::OperatorExpectedError {
                                    op: "*".to_string(),
                                    expected: format!("FixedPoint(p{})", fp1.scale),
                                    other_type: format!("FixedPoint(p{})", fp2.scale),
                                });
                            }
                            self.cost.charge(bigint_multiplication_cost(fp1.unscaled.len(), fp2.unscaled.len()))?;
                            make_fixedpoint_expr(multiply_fixed_points(&fp1, &fp2), "multiplication")
                        }
                        (lhs, rhs) => {
                            let lhs_type = get_type(lhs);
                            let rhs_type = get_type(rhs);
                            if lhs_type == rhs_type {
                                Err(InterpreterError::OperatorNotDefined {
                                    op: "*".to_string(),
                                    other_type: lhs_type,
                                })
                            } else {
                                Err(InterpreterError::OperatorExpectedError {
                                    op: "*".to_string(),
                                    expected: lhs_type,
                                    other_type: rhs_type,
                                })
                            }
                        }
                    }
                }

                ExprInstance::EDivBody(EDiv { p1, p2 }) => {
                    let v1 = self.eval_single_expr(&p1.clone().unwrap(), env)?;
                    let v2 = self.eval_single_expr(&p2.clone().unwrap(), env)?;

                    match (v1.expr_instance.unwrap(), v2.expr_instance.unwrap()) {
                        (ExprInstance::GInt(lhs), ExprInstance::GInt(rhs)) => {
                            self.cost.charge(division_cost())?;
                            if rhs == 0 {
                                return Err(InterpreterError::ReduceError(
                                    "Division by zero".to_string(),
                                ));
                            }
                            if lhs == i64::MIN && rhs == -1 {
                                return Err(InterpreterError::ReduceError(
                                    "Arithmetic overflow in division".to_string(),
                                ));
                            }
                            Ok(Expr {
                                expr_instance: Some(ExprInstance::GInt(lhs / rhs)),
                            })
                        }
                        (ExprInstance::GDouble(d1), ExprInstance::GDouble(d2)) => {
                            self.cost.charge(division_cost())?;
                            let result = f64::from_bits(d1) / f64::from_bits(d2);
                            Ok(Expr {
                                expr_instance: Some(ExprInstance::GDouble(result.to_bits())),
                            })
                        }
                        (ExprInstance::GBigInt(b1), ExprInstance::GBigInt(b2)) => {
                            self.cost.charge(bigint_division_cost(b1.len(), b2.len()))?;
                            if is_zero_twos_complement(&b2) {
                                return Err(InterpreterError::ReduceError(
                                    "Division by zero".to_string(),
                                ));
                            }
                            make_bigint_expr(divide_twos_complement(&b1, &b2), "division")
                        }
                        (ExprInstance::GBigRat(r1), ExprInstance::GBigRat(r2)) => {
                            self.cost.charge(bigrat_division_cost(
                                r1.numerator.len(), r1.denominator.len(),
                                r2.numerator.len(), r2.denominator.len(),
                            ))?;
                            if is_zero_twos_complement(&r2.numerator) {
                                return Err(InterpreterError::ReduceError(
                                    "Division by zero".to_string(),
                                ));
                            }
                            make_bigrat_expr(divide_big_rationals(&r1, &r2), "division")
                        }
                        (ExprInstance::GFixedPoint(fp1), ExprInstance::GFixedPoint(fp2)) => {
                            if fp1.scale != fp2.scale {
                                return Err(InterpreterError::OperatorExpectedError {
                                    op: "/".to_string(),
                                    expected: format!("FixedPoint(p{})", fp1.scale),
                                    other_type: format!("FixedPoint(p{})", fp2.scale),
                                });
                            }
                            self.cost.charge(bigint_division_cost(fp1.unscaled.len(), fp2.unscaled.len()))?;
                            if is_zero_twos_complement(&fp2.unscaled) {
                                return Err(InterpreterError::ReduceError(
                                    "Division by zero".to_string(),
                                ));
                            }
                            make_fixedpoint_expr(divide_fixed_points(&fp1, &fp2), "division")
                        }
                        (lhs, rhs) => {
                            let lhs_type = get_type(lhs);
                            let rhs_type = get_type(rhs);
                            if lhs_type == rhs_type {
                                Err(InterpreterError::OperatorNotDefined {
                                    op: "/".to_string(),
                                    other_type: lhs_type,
                                })
                            } else {
                                Err(InterpreterError::OperatorExpectedError {
                                    op: "/".to_string(),
                                    expected: lhs_type,
                                    other_type: rhs_type,
                                })
                            }
                        }
                    }
                }

                ExprInstance::EModBody(EMod { p1, p2 }) => {
                    let v1 = self.eval_single_expr(&p1.clone().unwrap(), env)?;
                    let v2 = self.eval_single_expr(&p2.clone().unwrap(), env)?;

                    match (v1.expr_instance.unwrap(), v2.expr_instance.unwrap()) {
                        (ExprInstance::GInt(lhs), ExprInstance::GInt(rhs)) => {
                            self.cost.charge(modulo_cost())?;
                            if rhs == 0 {
                                return Err(InterpreterError::ReduceError(
                                    "Modulo by zero".to_string(),
                                ));
                            }
                            if lhs == i64::MIN && rhs == -1 {
                                return Err(InterpreterError::ReduceError(
                                    "Arithmetic overflow in modulo".to_string(),
                                ));
                            }
                            Ok(Expr {
                                expr_instance: Some(ExprInstance::GInt(lhs % rhs)),
                            })
                        }
                        (ExprInstance::GDouble(_), ExprInstance::GDouble(_)) => {
                            Err(InterpreterError::ReduceError(
                                "Modulus not defined on floating point".to_string(),
                            ))
                        }
                        (ExprInstance::GBigInt(b1), ExprInstance::GBigInt(b2)) => {
                            self.cost.charge(bigint_modulo_cost(b1.len(), b2.len()))?;
                            if is_zero_twos_complement(&b2) {
                                return Err(InterpreterError::ReduceError(
                                    "Modulo by zero".to_string(),
                                ));
                            }
                            make_bigint_expr(modulo_twos_complement(&b1, &b2), "%")
                        }
                        (ExprInstance::GBigRat(_), ExprInstance::GBigRat(r2)) => {
                            if is_zero_twos_complement(&r2.numerator) {
                                return Err(InterpreterError::ReduceError(
                                    "Modulo by zero".to_string(),
                                ));
                            }
                            Ok(Expr {
                                expr_instance: Some(ExprInstance::GBigRat(
                                    models::rhoapi::GBigRational {
                                        numerator: vec![0],
                                        denominator: vec![1],
                                    },
                                )),
                            })
                        }
                        (ExprInstance::GFixedPoint(fp1), ExprInstance::GFixedPoint(fp2)) => {
                            if fp1.scale != fp2.scale {
                                return Err(InterpreterError::OperatorExpectedError {
                                    op: "%".to_string(),
                                    expected: format!("FixedPoint(p{})", fp1.scale),
                                    other_type: format!("FixedPoint(p{})", fp2.scale),
                                });
                            }
                            self.cost.charge(bigint_modulo_cost(
                                fp1.unscaled.len(),
                                fp2.unscaled.len(),
                            ))?;
                            if is_zero_twos_complement(&fp2.unscaled) {
                                return Err(InterpreterError::ReduceError(
                                    "Modulo by zero".to_string(),
                                ));
                            }
                            let ua = bytes_to_bigint(&fp1.unscaled);
                            let ub = bytes_to_bigint(&fp2.unscaled);
                            let remainder = &ua % &ub;
                            make_fixedpoint_expr(
                                models::rhoapi::GFixedPoint {
                                    unscaled: bigint_to_bytes(&remainder),
                                    scale: fp1.scale,
                                },
                                "%",
                            )
                        }
                        (lhs, rhs) => {
                            let lhs_type = get_type(lhs);
                            let rhs_type = get_type(rhs);
                            if lhs_type == rhs_type {
                                Err(InterpreterError::OperatorNotDefined {
                                    op: "%".to_string(),
                                    other_type: lhs_type,
                                })
                            } else {
                                Err(InterpreterError::OperatorExpectedError {
                                    op: "%".to_string(),
                                    expected: lhs_type,
                                    other_type: rhs_type,
                                })
                            }
                        }
                    }
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

                        (ExprInstance::GDouble(d1), ExprInstance::GDouble(d2)) => {
                            self.cost.charge(sum_cost())?;
                            let result = f64::from_bits(d1) + f64::from_bits(d2);
                            Ok(Expr {
                                expr_instance: Some(ExprInstance::GDouble(result.to_bits())),
                            })
                        }

                        (ExprInstance::GBigInt(b1), ExprInstance::GBigInt(b2)) => {
                            self.cost.charge(bigint_sum_cost(b1.len(), b2.len()))?;
                            make_bigint_expr(add_twos_complement(&b1, &b2), "+")
                        }

                        (ExprInstance::GBigRat(r1), ExprInstance::GBigRat(r2)) => {
                            self.cost.charge(bigrat_sum_cost(
                                r1.numerator.len(), r1.denominator.len(),
                                r2.numerator.len(), r2.denominator.len(),
                            ))?;
                            make_bigrat_expr(add_big_rationals(&r1, &r2), "+")
                        }

                        (ExprInstance::GFixedPoint(fp1), ExprInstance::GFixedPoint(fp2)) => {
                            if fp1.scale != fp2.scale {
                                return Err(InterpreterError::OperatorExpectedError {
                                    op: "+".to_string(),
                                    expected: format!("FixedPoint(p{})", fp1.scale),
                                    other_type: format!("FixedPoint(p{})", fp2.scale),
                                });
                            }
                            self.cost.charge(bigint_sum_cost(fp1.unscaled.len(), fp2.unscaled.len()))?;
                            make_fixedpoint_expr(
                                models::rhoapi::GFixedPoint {
                                    unscaled: add_twos_complement(&fp1.unscaled, &fp2.unscaled),
                                    scale: fp1.scale,
                                },
                                "+",
                            )
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

                        (ExprInstance::GInt(_), other)
                        | (ExprInstance::GDouble(_), other)
                        | (ExprInstance::GBigInt(_), other)
                        | (ExprInstance::GBigRat(_), other)
                        | (ExprInstance::GFixedPoint(_), other) => {
                            Err(InterpreterError::OperatorExpectedError {
                                op: "+".to_string(),
                                expected: "matching numeric types".to_string(),
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
                                expr_instance: Some(ExprInstance::GInt(lhs.wrapping_sub(rhs))),
                            })
                        }

                        (ExprInstance::GDouble(d1), ExprInstance::GDouble(d2)) => {
                            self.cost.charge(subtraction_cost())?;
                            let result = f64::from_bits(d1) - f64::from_bits(d2);
                            Ok(Expr {
                                expr_instance: Some(ExprInstance::GDouble(result.to_bits())),
                            })
                        }

                        (ExprInstance::GBigInt(b1), ExprInstance::GBigInt(b2)) => {
                            self.cost.charge(bigint_subtraction_cost(b1.len(), b2.len()))?;
                            make_bigint_expr(subtract_twos_complement(&b1, &b2), "-")
                        }

                        (ExprInstance::GBigRat(r1), ExprInstance::GBigRat(r2)) => {
                            self.cost.charge(bigrat_subtraction_cost(
                                r1.numerator.len(), r1.denominator.len(),
                                r2.numerator.len(), r2.denominator.len(),
                            ))?;
                            make_bigrat_expr(subtract_big_rationals(&r1, &r2), "-")
                        }

                        (ExprInstance::GFixedPoint(fp1), ExprInstance::GFixedPoint(fp2)) => {
                            if fp1.scale != fp2.scale {
                                return Err(InterpreterError::OperatorExpectedError {
                                    op: "-".to_string(),
                                    expected: format!("FixedPoint(p{})", fp1.scale),
                                    other_type: format!("FixedPoint(p{})", fp2.scale),
                                });
                            }
                            self.cost.charge(bigint_subtraction_cost(fp1.unscaled.len(), fp2.unscaled.len()))?;
                            make_fixedpoint_expr(
                                models::rhoapi::GFixedPoint {
                                    unscaled: subtract_twos_complement(
                                        &fp1.unscaled,
                                        &fp2.unscaled,
                                    ),
                                    scale: fp1.scale,
                                },
                                "-",
                            )
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

                        (ExprInstance::GInt(_), other)
                        | (ExprInstance::GDouble(_), other)
                        | (ExprInstance::GBigInt(_), other)
                        | (ExprInstance::GBigRat(_), other)
                        | (ExprInstance::GFixedPoint(_), other) => {
                            Err(InterpreterError::OperatorExpectedError {
                                op: "-".to_string(),
                                expected: "matching numeric types".to_string(),
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
                    relop(
                        &p1.clone().unwrap(),
                        &p2.clone().unwrap(),
                        |b1: bool, b2: bool| b1 < b2,
                        |i1: i64, i2: i64| i1 < i2,
                        |s1: String, s2: String| s1 < s2,
                    )
                }

                ExprInstance::ELteBody(ELte { p1, p2 }) => {
                    relop(
                        &p1.clone().unwrap(),
                        &p2.clone().unwrap(),
                        |b1: bool, b2: bool| b1 <= b2,
                        |i1: i64, i2: i64| i1 <= i2,
                        |s1: String, s2: String| s1 <= s2,
                    )
                }

                ExprInstance::EGtBody(EGt { p1, p2 }) => {
                    relop(
                        &p1.clone().unwrap(),
                        &p2.clone().unwrap(),
                        |b1: bool, b2: bool| b1 > b2,
                        |i1: i64, i2: i64| i1 > i2,
                        |s1: String, s2: String| s1 > s2,
                    )
                }

                ExprInstance::EGteBody(EGte { p1, p2 }) => {
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

                    let result = if par_contains_nan_double(&sv1)
                        || par_contains_nan_double(&sv2)
                    {
                        false
                    } else {
                        sv1 == sv2
                    };
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GBool(result)),
                    })
                }

                ExprInstance::ENeqBody(ENeq { p1, p2 }) => {
                    let v1 = self.eval_expr(&p1.clone().unwrap(), env)?;
                    let v2 = self.eval_expr(&p2.clone().unwrap(), env)?;
                    let sv1 = self.substitute.substitute_and_charge(&v1, 0, env)?;
                    let sv2 = self.substitute.substitute_and_charge(&v2, 0, env)?;
                    self.cost.charge(equality_check_cost(&sv1, &sv2))?;

                    let result = if par_contains_nan_double(&sv1)
                        || par_contains_nan_double(&sv2)
                    {
                        true
                    } else {
                        sv1 != sv2
                    };
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::GBool(result)),
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

                ExprInstance::EZipperBody(zipper) => {
                    // For zippers, just return them as-is (they're already evaluated)
                    Ok(Expr {
                        expr_instance: Some(ExprInstance::EZipperBody(zipper.clone())),
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

                    (
                        ExprInstance::EPathmapBody(base_pathmap),
                        ExprInstance::EPathmapBody(other_pathmap),
                    ) => {
                        let base_rmap =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&base_pathmap);
                        let other_rmap =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&other_pathmap);

                        self.outer
                            .cost
                            .charge(union_cost(other_pathmap.ps.len() as i64))?;
                        let result_map = base_rmap.map.join(&other_rmap.map);

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(
                                PathMapCrateTypeMapper::rholang_pathmap_to_e_pathmap(
                                    &result_map,
                                    base_rmap.connective_used || other_rmap.connective_used,
                                    &union(base_rmap.locally_free, other_rmap.locally_free),
                                    None,
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

                    (
                        ExprInstance::EPathmapBody(base_pathmap),
                        ExprInstance::EPathmapBody(other_pathmap),
                    ) => {
                        let base_rmap =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&base_pathmap);
                        let other_rmap =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&other_pathmap);

                        self.outer
                            .cost
                            .charge(diff_cost(other_pathmap.ps.len() as i64))?;
                        let result_map = base_rmap.map.subtract(&other_rmap.map);

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(
                                PathMapCrateTypeMapper::rholang_pathmap_to_e_pathmap(
                                    &result_map,
                                    base_rmap.connective_used,
                                    &base_rmap.locally_free,
                                    None,
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
            fn intersection(
                &self,
                base_expr: &Expr,
                other_expr: &Expr,
            ) -> Result<Expr, InterpreterError> {
                match (
                    base_expr.expr_instance.clone().unwrap(),
                    other_expr.expr_instance.clone().unwrap(),
                ) {
                    (
                        ExprInstance::EPathmapBody(base_pathmap),
                        ExprInstance::EPathmapBody(other_pathmap),
                    ) => {
                        let base_rmap =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&base_pathmap);
                        let other_rmap =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&other_pathmap);

                        self.outer
                            .cost
                            .charge(union_cost(other_pathmap.ps.len() as i64))?;
                        let result_map = base_rmap.map.meet(&other_rmap.map);

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(
                                PathMapCrateTypeMapper::rholang_pathmap_to_e_pathmap(
                                    &result_map,
                                    base_rmap.connective_used || other_rmap.connective_used,
                                    &union(base_rmap.locally_free, other_rmap.locally_free),
                                    None,
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
            fn restriction(
                &self,
                base_expr: &Expr,
                other_expr: &Expr,
            ) -> Result<Expr, InterpreterError> {
                match (
                    base_expr.expr_instance.clone().unwrap(),
                    other_expr.expr_instance.clone().unwrap(),
                ) {
                    (
                        ExprInstance::EPathmapBody(base_pathmap),
                        ExprInstance::EPathmapBody(other_pathmap),
                    ) => {
                        let base_rmap =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&base_pathmap);
                        let other_rmap =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&other_pathmap);

                        self.outer
                            .cost
                            .charge(union_cost(other_pathmap.ps.len() as i64))?;
                        let result_map = base_rmap.map.restrict(&other_rmap.map);

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(
                                PathMapCrateTypeMapper::rholang_pathmap_to_e_pathmap(
                                    &result_map,
                                    base_rmap.connective_used,
                                    &base_rmap.locally_free,
                                    None,
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
                        let base_rmap =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&base_pathmap);
                        if n < 0 {
                            return Err(InterpreterError::ReduceError(format!(
                                "dropHead argument must be non-negative, got: {}",
                                n
                            )));
                        }
                        self.outer.cost.charge(union_cost(n))?;

                        // For dropHead, we need to return a new EPathMap with modified path elements
                        // Instead of using PathMap, directly construct the result elements
                        let mut result_elements = Vec::new();

                        for par in &base_pathmap.ps {
                            // Check if this Par is a list
                            if let Some(models::rhoapi::expr::ExprInstance::EListBody(list)) =
                                par.exprs.first().and_then(|e| e.expr_instance.as_ref())
                            {
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
                                            expr_instance: Some(
                                                models::rhoapi::expr::ExprInstance::EListBody(
                                                    new_list,
                                                ),
                                            ),
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
                                },
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
                        // Create an EZipper from the PathMap
                        let ezipper = EZipper {
                            pathmap: Some(pathmap),
                            current_path: vec![], // Start at root
                            is_write_zipper: false,
                            locally_free: vec![],
                            connective_used: false,
                        };
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EZipperBody(ezipper)),
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
            fn create_read_zipper_at(
                &self,
                base_expr: &Expr,
                path_par: &Par,
            ) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EPathmapBody(pathmap) => {
                        use models::rust::pathmap_integration::par_to_path;

                        // Convert the path argument to byte segments
                        let path_segments = par_to_path(path_par);

                        // Store the COMPLETE ORIGINAL PathMap for correct operations
                        // Display will show absolute paths, but operations will work correctly
                        // TODO: To show relative paths in display, we'd need to modify serialization/display code
                        let complete_pathmap = pathmap.clone();

                        // Create an EZipper with the complete PathMap
                        // current_path indicates the position within the complete tree
                        let ezipper = EZipper {
                            pathmap: Some(complete_pathmap),
                            current_path: path_segments.clone(),
                            is_write_zipper: false,
                            locally_free: pathmap.locally_free.clone(),
                            connective_used: pathmap.connective_used,
                        };

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EZipperBody(ezipper)),
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
                        // Create an EZipper for writing
                        let ezipper = EZipper {
                            pathmap: Some(pathmap),
                            current_path: vec![], // Start at root
                            is_write_zipper: true,
                            locally_free: vec![],
                            connective_used: false,
                        };
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EZipperBody(ezipper)),
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
            fn create_write_zipper_at(
                &self,
                base_expr: &Expr,
                path_par: &Par,
            ) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EPathmapBody(pathmap) => {
                        use models::rust::pathmap_integration::par_to_path;

                        // Convert the path argument to byte segments
                        let path_segments = par_to_path(path_par);

                        // Store the COMPLETE ORIGINAL PathMap for correct operations
                        let complete_pathmap = pathmap.clone();

                        // Create an EZipper with the complete PathMap (write mode)
                        let ezipper = EZipper {
                            pathmap: Some(complete_pathmap),
                            current_path: path_segments.clone(),
                            is_write_zipper: true,
                            locally_free: pathmap.locally_free.clone(),
                            connective_used: pathmap.connective_used,
                        };

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EZipperBody(ezipper)),
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
            fn descend_to(
                &self,
                base_expr: &Expr,
                path_par: &Par,
            ) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EZipperBody(mut zipper) => {
                        use models::rust::pathmap_integration::par_to_path;

                        // Convert the path argument to byte segments
                        let path_segments = par_to_path(path_par);

                        // Update the zipper's current_path to navigate to the new location
                        // Append the new path segments to the current path
                        zipper.current_path.extend(path_segments);

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EZipperBody(zipper)),
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

    fn get_leaf_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct GetLeafMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> GetLeafMethod<'a> {
            fn get_leaf(&self, base_expr: &Expr) -> Result<Par, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EZipperBody(zipper) => {
                        // Get the pathmap from the zipper
                        let pathmap = zipper.pathmap.as_ref().expect("zipper pathmap was None");
                        let pathmap_result =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(pathmap);
                        let rholang_pathmap = pathmap_result.map;

                        // Use the zipper's current_path to look up the value
                        // Build the key from current_path segments (same encoding as create_pathmap_from_elements)
                        let key: Vec<u8> = zipper
                            .current_path
                            .iter()
                            .flat_map(|seg| {
                                let mut s = seg.clone();
                                s.push(0xFF); // separator
                                s
                            })
                            .collect();

                        // Look up value at this path
                        if let Some(value) = rholang_pathmap.get(&key) {
                            Ok(value.clone())
                        } else {
                            Ok(Par::default()) // Nil - no value at this path
                        }
                    }
                    ExprInstance::EPathmapBody(pathmap) => {
                        // Convert EPathMap to RholangPathMap
                        let pathmap_result =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&pathmap);
                        let rholang_pathmap = pathmap_result.map;

                        // Create a read zipper and get the value at current position
                        let read_zipper = RholangReadZipper::new(
                            &rholang_pathmap,
                            pathmap_result.connective_used,
                            pathmap_result.locally_free,
                        );

                        // Get value at current position (root)
                        if let Some(value) = read_zipper.get_val() {
                            Ok(value.clone())
                        } else {
                            Ok(Par::default()) // Nil
                        }
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("getLeaf"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for GetLeafMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("getLeaf"),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                self.outer.cost.charge(lookup_cost())?;
                self.get_leaf(&base_expr)
            }
        }

        Box::new(GetLeafMethod { outer: self })
    }

    fn get_subtrie_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct GetSubtrieMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> GetSubtrieMethod<'a> {
            fn get_subtrie(&self, base_expr: &Expr) -> Result<Par, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EZipperBody(zipper) => {
                        // Get the pathmap from the zipper
                        let pathmap = zipper.pathmap.as_ref().expect("zipper pathmap was None");
                        let pathmap_result =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(pathmap);
                        let rholang_pathmap = pathmap_result.map;

                        // Build prefix key from current_path
                        let prefix_key: Vec<u8> = zipper
                            .current_path
                            .iter()
                            .flat_map(|seg| {
                                let mut s = seg.clone();
                                s.push(0xFF); // separator
                                s
                            })
                            .collect();

                        // Collect all entries with this prefix
                        let mut subtrie_elements = Vec::new();
                        for (key, value) in rholang_pathmap.iter() {
                            if key.starts_with(&prefix_key) {
                                subtrie_elements.push(value.clone());
                            }
                        }

                        // Return as PathMap
                        Ok(Par::default().with_exprs(vec![Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(EPathMap {
                                ps: subtrie_elements,
                                locally_free: pathmap_result.locally_free,
                                connective_used: pathmap_result.connective_used,
                                remainder: None,
                            })),
                        }]))
                    }
                    ExprInstance::EPathmapBody(pathmap) => {
                        // For PathMap without zipper, return entire PathMap (all is subtrie at root)
                        Ok(Par::default().with_exprs(vec![Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(pathmap)),
                        }]))
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("getSubtrie"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for GetSubtrieMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("getSubtrie"),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                self.outer.cost.charge(lookup_cost())?;
                self.get_subtrie(&base_expr)
            }
        }

        Box::new(GetSubtrieMethod { outer: self })
    }

    fn set_leaf_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct SetLeafMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> SetLeafMethod<'a> {
            fn set_leaf(&self, base_expr: &Expr, value: &Par) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EZipperBody(zipper) => {
                        // For a write zipper, set value at current position
                        let mut pathmap = zipper.pathmap.expect("zipper pathmap was None");
                        pathmap.ps.push(value.clone());
                        // Return the modified PathMap (not zipper)
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(pathmap)),
                        })
                    }
                    ExprInstance::EPathmapBody(mut pathmap) => {
                        // For a write zipper, set value at current position
                        // For now, add to the pathmap
                        pathmap.ps.push(value.clone());
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(pathmap)),
                        })
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("setLeaf"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for SetLeafMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("setLeaf"),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                let value = self.outer.eval_expr(&args[0], env)?;
                self.outer.cost.charge(add_cost())?;
                let result = self.set_leaf(&base_expr, &value)?;
                Ok(Par::default().with_exprs(vec![result]))
            }
        }

        Box::new(SetLeafMethod { outer: self })
    }

    fn set_subtrie_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct SetSubtrieMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> SetSubtrieMethod<'a> {
            fn set_subtrie(
                &self,
                base_expr: &Expr,
                source_par: &Par,
            ) -> Result<Expr, InterpreterError> {
                match (
                    base_expr.expr_instance.clone().unwrap(),
                    source_par
                        .exprs
                        .first()
                        .and_then(|e| e.expr_instance.clone()),
                ) {
                    // Only works on write zippers
                    (
                        ExprInstance::EZipperBody(zipper),
                        Some(ExprInstance::EPathmapBody(source)),
                    ) if zipper.is_write_zipper => {
                        // Step 1: Extract base PathMap and build prefix
                        let pathmap = zipper.pathmap.expect("zipper pathmap was None");
                        let pathmap_result =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&pathmap);
                        let mut rholang_pathmap = pathmap_result.map;

                        let prefix_key: Vec<u8> = zipper
                            .current_path
                            .iter()
                            .flat_map(|seg| {
                                let mut s = seg.clone();
                                s.push(0xFF);
                                s
                            })
                            .collect();

                        // Step 2: Remove all entries with this prefix
                        let keys_to_remove: Vec<Vec<u8>> = rholang_pathmap
                            .iter()
                            .filter_map(|(key, _)| {
                                if key.starts_with(&prefix_key) {
                                    Some(key.clone())
                                } else {
                                    None
                                }
                            })
                            .collect();

                        for key in keys_to_remove {
                            rholang_pathmap.remove(&key);
                        }

                        // Step 3: Add source entries with prepended prefix
                        for source_entry in source.ps.iter() {
                            use models::rust::pathmap_integration::par_to_path;
                            let source_segments = par_to_path(source_entry);

                            // Prepend current_path to make absolute
                            let mut absolute_segments = zipper.current_path.clone();
                            absolute_segments.extend(source_segments.clone());

                            // Encode as key
                            let key: Vec<u8> = absolute_segments
                                .iter()
                                .flat_map(|seg| {
                                    let mut s = seg.clone();
                                    s.push(0xFF);
                                    s
                                })
                                .collect();

                            // Build the Par that represents the absolute path
                            // Extract elements from an existing entry to understand their structure
                            let mut absolute_elements = Vec::new();

                            // Find an existing entry that starts with current_path
                            let found_existing = if let Some(existing_entry) =
                                pathmap.ps.iter().find(|entry| {
                                    if let Some(ExprInstance::EListBody(existing_list)) =
                                        &entry.exprs.first().and_then(|e| e.expr_instance.as_ref())
                                    {
                                        if existing_list.ps.len() < zipper.current_path.len() {
                                            return false;
                                        }
                                        // Check if the entry actually starts with current_path
                                        use models::rust::pathmap_integration::par_to_path;
                                        let entry_segments = par_to_path(entry);
                                        entry_segments.starts_with(&zipper.current_path)
                                    } else {
                                        false
                                    }
                                }) {
                                if let Some(ExprInstance::EListBody(existing_list)) =
                                    &existing_entry
                                        .exprs
                                        .first()
                                        .and_then(|e| e.expr_instance.as_ref())
                                {
                                    // Take first N elements where N = current_path length
                                    absolute_elements.extend(
                                        existing_list.ps[..zipper.current_path.len()].to_vec(),
                                    );
                                }
                                true
                            } else {
                                false
                            };

                            // If no existing entry found, reconstruct Par elements from current_path bytes
                            if !found_existing {
                                use models::rust::path_map_encoder::SExpr;
                                for segment_bytes in &zipper.current_path {
                                    // Decode the S-expr bytes to extract the string
                                    if let Ok(sexpr) = SExpr::decode(segment_bytes) {
                                        if let SExpr::Symbol(mut s) = sexpr {
                                            // Strip quotes if present (S-expr includes them for string literals)
                                            if s.starts_with('"')
                                                && s.ends_with('"')
                                                && s.len() >= 2
                                            {
                                                s = s[1..s.len() - 1].to_string();
                                            }
                                            absolute_elements.push(new_gstring_par(
                                                s,
                                                vec![],
                                                false,
                                            ));
                                        }
                                    }
                                }
                            }

                            // Add source_entry's elements
                            if let Some(ExprInstance::EListBody(source_list)) = &source_entry
                                .exprs
                                .first()
                                .and_then(|e| e.expr_instance.as_ref())
                            {
                                absolute_elements.extend(source_list.ps.clone());
                            }

                            // Create the absolute path Par
                            let absolute_path_par = Par::default().with_exprs(vec![Expr {
                                expr_instance: Some(ExprInstance::EListBody(
                                    models::rhoapi::EList {
                                        ps: absolute_elements,
                                        locally_free: vec![],
                                        connective_used: false,
                                        remainder: None,
                                    },
                                )),
                            }]);

                            rholang_pathmap.insert(key, absolute_path_par);
                        }

                        // Step 3b: If source is empty, add current_path as entry
                        if source.ps.is_empty() && !zipper.current_path.is_empty() {
                            // Encode current_path as key
                            let key: Vec<u8> = zipper
                                .current_path
                                .iter()
                                .flat_map(|seg| {
                                    let mut s = seg.clone();
                                    s.push(0xFF);
                                    s
                                })
                                .collect();

                            // Build the Par for current_path
                            let mut absolute_elements = Vec::new();

                            // Find an existing entry that starts with current_path
                            let found_existing = if let Some(existing_entry) =
                                pathmap.ps.iter().find(|entry| {
                                    if let Some(ExprInstance::EListBody(existing_list)) =
                                        &entry.exprs.first().and_then(|e| e.expr_instance.as_ref())
                                    {
                                        if existing_list.ps.len() < zipper.current_path.len() {
                                            return false;
                                        }
                                        // Check if the entry actually starts with current_path
                                        use models::rust::pathmap_integration::par_to_path;
                                        let entry_segments = par_to_path(entry);
                                        entry_segments.starts_with(&zipper.current_path)
                                    } else {
                                        false
                                    }
                                }) {
                                if let Some(ExprInstance::EListBody(existing_list)) =
                                    &existing_entry
                                        .exprs
                                        .first()
                                        .and_then(|e| e.expr_instance.as_ref())
                                {
                                    // Take first N elements where N = current_path length
                                    absolute_elements.extend(
                                        existing_list.ps[..zipper.current_path.len()].to_vec(),
                                    );
                                }
                                true
                            } else {
                                false
                            };

                            // If no existing entry found, reconstruct Par elements from current_path bytes
                            if !found_existing {
                                use models::rust::path_map_encoder::SExpr;
                                for segment_bytes in &zipper.current_path {
                                    // Decode the S-expr bytes to extract the string
                                    if let Ok(sexpr) = SExpr::decode(segment_bytes) {
                                        if let SExpr::Symbol(mut s) = sexpr {
                                            // Strip quotes if present (S-expr includes them for string literals)
                                            if s.starts_with('"')
                                                && s.ends_with('"')
                                                && s.len() >= 2
                                            {
                                                s = s[1..s.len() - 1].to_string();
                                            }
                                            absolute_elements.push(new_gstring_par(
                                                s,
                                                vec![],
                                                false,
                                            ));
                                        }
                                    }
                                }
                            }

                            // Create the Par for current_path
                            let current_path_par = Par::default().with_exprs(vec![Expr {
                                expr_instance: Some(ExprInstance::EListBody(
                                    models::rhoapi::EList {
                                        ps: absolute_elements,
                                        locally_free: vec![],
                                        connective_used: false,
                                        remainder: None,
                                    },
                                )),
                            }]);

                            rholang_pathmap.insert(key, current_path_par);
                        }

                        // Step 4: Convert back to EPathMap
                        let result_pathmap = PathMapCrateTypeMapper::rholang_pathmap_to_e_pathmap(
                            &rholang_pathmap,
                            pathmap_result.connective_used,
                            &pathmap_result.locally_free,
                            None,
                        );

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(result_pathmap)),
                        })
                    }

                    // Error cases
                    (ExprInstance::EZipperBody(zipper), _) if !zipper.is_write_zipper => {
                        Err(InterpreterError::MethodNotDefined {
                            method: String::from("setSubtrie (requires write zipper)"),
                            other_type: "read zipper".to_string(),
                        })
                    }
                    (other, _) => Err(InterpreterError::MethodNotDefined {
                        method: String::from("setSubtrie"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for SetSubtrieMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("setSubtrie"),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                let source_par = self.outer.eval_expr(&args[0], env)?;
                self.outer.cost.charge(union_cost(1))?;
                let result = self.set_subtrie(&base_expr, &source_par)?;
                Ok(Par::default().with_exprs(vec![result]))
            }
        }

        Box::new(SetSubtrieMethod { outer: self })
    }

    fn remove_leaf_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct RemoveLeafMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> RemoveLeafMethod<'a> {
            fn remove_leaf(&self, base_expr: &Expr) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EZipperBody(zipper) => {
                        // Extract pathmap from zipper
                        let pathmap = zipper.pathmap.expect("zipper pathmap was None");
                        let pathmap_result =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&pathmap);
                        let mut rholang_pathmap = pathmap_result.map;

                        // Build key from current_path
                        let key: Vec<u8> = zipper
                            .current_path
                            .iter()
                            .flat_map(|seg| {
                                let mut s = seg.clone();
                                s.push(0xFF); // separator
                                s
                            })
                            .collect();

                        // Remove value at this path
                        rholang_pathmap.remove(&key);

                        // Convert back to EPathMap
                        let result_pathmap = PathMapCrateTypeMapper::rholang_pathmap_to_e_pathmap(
                            &rholang_pathmap,
                            pathmap_result.connective_used,
                            &pathmap_result.locally_free,
                            None,
                        );

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(result_pathmap)),
                        })
                    }
                    ExprInstance::EPathmapBody(mut pathmap) => {
                        // Remove value at current position (root)
                        pathmap.ps.pop();
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(pathmap)),
                        })
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("removeLeaf"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for RemoveLeafMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("removeLeaf"),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                self.outer.cost.charge(remove_cost())?;
                let result = self.remove_leaf(&base_expr)?;
                Ok(Par::default().with_exprs(vec![result]))
            }
        }

        Box::new(RemoveLeafMethod { outer: self })
    }

    fn remove_branches_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct RemoveBranchesMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> RemoveBranchesMethod<'a> {
            fn remove_branches(&self, base_expr: &Expr) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EZipperBody(zipper) => {
                        // Extract pathmap from zipper
                        let pathmap = zipper.pathmap.expect("zipper pathmap was None");
                        let pathmap_result =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&pathmap);
                        let mut rholang_pathmap = pathmap_result.map;

                        // Build prefix key from current_path
                        let prefix_key: Vec<u8> = zipper
                            .current_path
                            .iter()
                            .flat_map(|seg| {
                                let mut s = seg.clone();
                                s.push(0xFF); // separator
                                s
                            })
                            .collect();

                        // Remove all branches with this prefix
                        // Collect keys to remove (can't modify while iterating)
                        let keys_to_remove: Vec<Vec<u8>> = rholang_pathmap
                            .iter()
                            .filter_map(|(key, _)| {
                                if key.starts_with(&prefix_key) && key.len() > prefix_key.len() {
                                    Some(key.clone())
                                } else {
                                    None
                                }
                            })
                            .collect();

                        // Remove the collected keys
                        for key in keys_to_remove {
                            rholang_pathmap.remove(&key);
                        }

                        // Convert back to EPathMap
                        let result_pathmap = PathMapCrateTypeMapper::rholang_pathmap_to_e_pathmap(
                            &rholang_pathmap,
                            pathmap_result.connective_used,
                            &pathmap_result.locally_free,
                            None,
                        );

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(result_pathmap)),
                        })
                    }
                    ExprInstance::EPathmapBody(pathmap) => {
                        // Remove all branches below current position (root = remove everything)
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(
                                models::rhoapi::EPathMap {
                                    ps: vec![],
                                    locally_free: pathmap.locally_free,
                                    connective_used: pathmap.connective_used,
                                    remainder: pathmap.remainder,
                                },
                            )),
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
            fn graft(
                &self,
                base_expr: &Expr,
                source_expr: &Expr,
            ) -> Result<Expr, InterpreterError> {
                match (
                    base_expr.expr_instance.clone().unwrap(),
                    source_expr.expr_instance.clone().unwrap(),
                ) {
                    // Both are zippers
                    (
                        ExprInstance::EZipperBody(dest_zipper),
                        ExprInstance::EZipperBody(source_zipper),
                    ) => {
                        let mut dest_pathmap =
                            dest_zipper.pathmap.expect("dest zipper pathmap was None");
                        let source_pathmap = source_zipper
                            .pathmap
                            .expect("source zipper pathmap was None");

                        // Graft: copy subtrie from source to destination
                        dest_pathmap.ps.extend(source_pathmap.ps);

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(dest_pathmap)),
                        })
                    }
                    // Destination is zipper, source is PathMap
                    (
                        ExprInstance::EZipperBody(dest_zipper),
                        ExprInstance::EPathmapBody(source_pathmap),
                    ) => {
                        let mut dest_pathmap =
                            dest_zipper.pathmap.expect("dest zipper pathmap was None");

                        // Graft: copy subtrie from source to destination
                        dest_pathmap.ps.extend(source_pathmap.ps);

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(dest_pathmap)),
                        })
                    }
                    // Destination is PathMap, source is zipper
                    (
                        ExprInstance::EPathmapBody(mut dest_pathmap),
                        ExprInstance::EZipperBody(source_zipper),
                    ) => {
                        let source_pathmap = source_zipper
                            .pathmap
                            .expect("source zipper pathmap was None");

                        // Graft: copy subtrie from source to destination
                        dest_pathmap.ps.extend(source_pathmap.ps);

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(dest_pathmap)),
                        })
                    }
                    // Both are PathMaps (existing case)
                    (
                        ExprInstance::EPathmapBody(mut dest_pathmap),
                        ExprInstance::EPathmapBody(source_pathmap),
                    ) => {
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
            fn join_into(
                &self,
                base_expr: &Expr,
                source_expr: &Expr,
            ) -> Result<Expr, InterpreterError> {
                match (
                    base_expr.expr_instance.clone().unwrap(),
                    source_expr.expr_instance.clone().unwrap(),
                ) {
                    // Both are zippers
                    (
                        ExprInstance::EZipperBody(base_zipper),
                        ExprInstance::EZipperBody(source_zipper),
                    ) => {
                        let base_pathmap =
                            base_zipper.pathmap.expect("base zipper pathmap was None");
                        let source_pathmap = source_zipper
                            .pathmap
                            .expect("source zipper pathmap was None");

                        let base_rmap =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&base_pathmap);
                        let source_rmap =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&source_pathmap);

                        self.outer
                            .cost
                            .charge(union_cost(source_pathmap.ps.len() as i64))?;
                        let result_map = base_rmap.map.join(&source_rmap.map);

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(
                                PathMapCrateTypeMapper::rholang_pathmap_to_e_pathmap(
                                    &result_map,
                                    base_rmap.connective_used || source_rmap.connective_used,
                                    &union(base_rmap.locally_free, source_rmap.locally_free),
                                    None,
                                ),
                            )),
                        })
                    }
                    // Base is zipper, source is PathMap
                    (
                        ExprInstance::EZipperBody(base_zipper),
                        ExprInstance::EPathmapBody(source_pathmap),
                    ) => {
                        let base_pathmap =
                            base_zipper.pathmap.expect("base zipper pathmap was None");

                        let base_rmap =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&base_pathmap);
                        let source_rmap =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&source_pathmap);

                        self.outer
                            .cost
                            .charge(union_cost(source_pathmap.ps.len() as i64))?;
                        let result_map = base_rmap.map.join(&source_rmap.map);

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(
                                PathMapCrateTypeMapper::rholang_pathmap_to_e_pathmap(
                                    &result_map,
                                    base_rmap.connective_used || source_rmap.connective_used,
                                    &union(base_rmap.locally_free, source_rmap.locally_free),
                                    None,
                                ),
                            )),
                        })
                    }
                    // Base is PathMap, source is zipper
                    (
                        ExprInstance::EPathmapBody(base_pathmap),
                        ExprInstance::EZipperBody(source_zipper),
                    ) => {
                        let source_pathmap = source_zipper
                            .pathmap
                            .expect("source zipper pathmap was None");

                        let base_rmap =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&base_pathmap);
                        let source_rmap =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&source_pathmap);

                        self.outer
                            .cost
                            .charge(union_cost(source_pathmap.ps.len() as i64))?;
                        let result_map = base_rmap.map.join(&source_rmap.map);

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(
                                PathMapCrateTypeMapper::rholang_pathmap_to_e_pathmap(
                                    &result_map,
                                    base_rmap.connective_used || source_rmap.connective_used,
                                    &union(base_rmap.locally_free, source_rmap.locally_free),
                                    None,
                                ),
                            )),
                        })
                    }
                    // Both are PathMaps (existing case)
                    (
                        ExprInstance::EPathmapBody(base_pathmap),
                        ExprInstance::EPathmapBody(source_pathmap),
                    ) => {
                        // JoinInto: union-merge subtries
                        let base_rmap =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&base_pathmap);
                        let source_rmap =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&source_pathmap);

                        self.outer
                            .cost
                            .charge(union_cost(source_pathmap.ps.len() as i64))?;
                        let result_map = base_rmap.map.join(&source_rmap.map);

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(
                                PathMapCrateTypeMapper::rholang_pathmap_to_e_pathmap(
                                    &result_map,
                                    base_rmap.connective_used || source_rmap.connective_used,
                                    &union(base_rmap.locally_free, source_rmap.locally_free),
                                    None,
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

    fn at_path_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct AtPathMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> AtPathMethod<'a> {
            fn at_path(&self, base_expr: &Expr, path_par: &Par) -> Result<Par, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EZipperBody(zipper) => {
                        use models::rust::pathmap_integration::par_to_path;

                        // Get PathMap from zipper
                        let pathmap = zipper.pathmap.expect("zipper pathmap was None");
                        let pathmap_result =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&pathmap);
                        let rholang_pathmap = pathmap_result.map;

                        // Combine current_path with requested path
                        let path_segments = par_to_path(path_par);
                        let mut full_path = zipper.current_path.clone();
                        full_path.extend(path_segments);

                        // Build key from full path
                        let key: Vec<u8> = full_path
                            .iter()
                            .flat_map(|seg| {
                                let mut s = seg.clone();
                                s.push(0xFF); // separator
                                s
                            })
                            .collect();

                        // Get value at this path
                        match rholang_pathmap.get(&key) {
                            Some(val) => Ok(val.clone()),
                            None => Ok(Par::default()), // Return Nil if not found
                        }
                    }
                    ExprInstance::EPathmapBody(pathmap) => {
                        use models::rust::pathmap_integration::par_to_path;

                        // Get value at path from PathMap root
                        let pathmap_result =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&pathmap);
                        let rholang_pathmap = pathmap_result.map;

                        let path_segments = par_to_path(path_par);
                        let key: Vec<u8> = path_segments
                            .iter()
                            .flat_map(|seg| {
                                let mut s = seg.clone();
                                s.push(0xFF); // separator
                                s
                            })
                            .collect();

                        match rholang_pathmap.get(&key) {
                            Some(val) => Ok(val.clone()),
                            None => Ok(Par::default()), // Return Nil if not found
                        }
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("atPath"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for AtPathMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("atPath"),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                let path_par = self.outer.eval_expr(&args[0], env)?;
                self.outer.cost.charge(union_cost(1))?;
                self.at_path(&base_expr, &path_par)
            }
        }

        Box::new(AtPathMethod { outer: self })
    }

    fn path_exists_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct PathExistsMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> PathExistsMethod<'a> {
            fn path_exists(&self, base_expr: &Expr) -> Result<bool, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EZipperBody(zipper) => {
                        // Get PathMap from zipper
                        let pathmap = zipper.pathmap.as_ref().expect("zipper pathmap was None");
                        let pathmap_result =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(pathmap);
                        let rholang_pathmap = pathmap_result.map;

                        // Build key from current_path
                        let key: Vec<u8> = zipper
                            .current_path
                            .iter()
                            .flat_map(|seg| {
                                let mut s = seg.clone();
                                s.push(0xFF); // separator
                                s
                            })
                            .collect();

                        // Check if path exists (either has value or has children)
                        if key.is_empty() {
                            // Root always exists if PathMap is not empty
                            Ok(!pathmap.ps.is_empty())
                        } else {
                            // Check if exact path or any path with this prefix exists
                            Ok(rholang_pathmap.iter().any(|(k, _)| k.starts_with(&key)))
                        }
                    }
                    ExprInstance::EPathmapBody(pathmap) => {
                        // For PathMap at root, it exists if not empty
                        Ok(!pathmap.ps.is_empty())
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("pathExists"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for PathExistsMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("pathExists"),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                self.outer.cost.charge(union_cost(1))?;
                let result = self.path_exists(&base_expr)?;

                // Return as GBool
                Ok(Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::GBool(result)),
                }]))
            }
        }

        Box::new(PathExistsMethod { outer: self })
    }

    fn create_path_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct CreatePathMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> CreatePathMethod<'a> {
            fn create_path(
                &self,
                base_expr: &Expr,
                path_par: &Par,
            ) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EZipperBody(zipper) if zipper.is_write_zipper => {
                        use models::rust::pathmap_integration::par_to_path;

                        // Get PathMap from zipper
                        let pathmap = zipper.pathmap.expect("zipper pathmap was None");

                        // Parse requested path to validate format
                        let _path_segments = par_to_path(path_par);

                        // Combine with current path
                        let _ = zipper.current_path.clone(); // Use for future implementation

                        // Create path structure by ensuring intermediate nodes exist
                        // We don't set values, just ensure the path structure exists
                        // In a trie, paths are implicitly created when you add values
                        // Since we want to create structure without values, we'll just
                        // return the PathMap as-is (the structure will be created when needed)
                        // Alternatively, we could insert empty markers but that changes semantics

                        // For now, just return the PathMap unchanged
                        // This is a no-op but validates the path format
                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(pathmap)),
                        })
                    }
                    ExprInstance::EZipperBody(_) => Err(InterpreterError::MethodNotDefined {
                        method: String::from("createPath (requires write zipper)"),
                        other_type: "read zipper".to_string(),
                    }),
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("createPath"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for CreatePathMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("createPath"),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                let path_par = self.outer.eval_expr(&args[0], env)?;
                self.outer.cost.charge(union_cost(1))?;
                let result = self.create_path(&base_expr, &path_par)?;
                Ok(Par::default().with_exprs(vec![result]))
            }
        }

        Box::new(CreatePathMethod { outer: self })
    }

    fn prune_path_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct PrunePathMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> PrunePathMethod<'a> {
            fn prune_path(&self, base_expr: &Expr) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EZipperBody(zipper) if zipper.is_write_zipper => {
                        // Get PathMap from zipper
                        let pathmap = zipper.pathmap.expect("zipper pathmap was None");
                        let pathmap_result =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&pathmap);
                        let mut rholang_pathmap = pathmap_result.map;

                        // Build key from current_path
                        let prefix_key: Vec<u8> = zipper
                            .current_path
                            .iter()
                            .flat_map(|seg| {
                                let mut s = seg.clone();
                                s.push(0xFF); // separator
                                s
                            })
                            .collect();

                        // Remove all entries at and below this path
                        let keys_to_remove: Vec<Vec<u8>> = rholang_pathmap
                            .iter()
                            .filter_map(|(key, _)| {
                                if key.starts_with(&prefix_key) {
                                    Some(key.clone())
                                } else {
                                    None
                                }
                            })
                            .collect();

                        for key in keys_to_remove {
                            rholang_pathmap.remove(&key);
                        }

                        // Convert back to EPathMap
                        let result_pathmap = PathMapCrateTypeMapper::rholang_pathmap_to_e_pathmap(
                            &rholang_pathmap,
                            pathmap_result.connective_used,
                            &pathmap_result.locally_free,
                            None,
                        );

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EPathmapBody(result_pathmap)),
                        })
                    }
                    ExprInstance::EZipperBody(_) => Err(InterpreterError::MethodNotDefined {
                        method: String::from("prunePath (requires write zipper)"),
                        other_type: "read zipper".to_string(),
                    }),
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("prunePath"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for PrunePathMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("prunePath"),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                self.outer.cost.charge(remove_cost())?;
                let result = self.prune_path(&base_expr)?;
                Ok(Par::default().with_exprs(vec![result]))
            }
        }

        Box::new(PrunePathMethod { outer: self })
    }

    fn reset_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct ResetMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> ResetMethod<'a> {
            fn reset(&self, base_expr: &Expr) -> Result<Expr, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EZipperBody(mut zipper) => {
                        // Reset to root by clearing current_path
                        zipper.current_path = vec![];

                        Ok(Expr {
                            expr_instance: Some(ExprInstance::EZipperBody(zipper)),
                        })
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("reset"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for ResetMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("reset"),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                self.outer.cost.charge(union_cost(1))?;
                let result = self.reset(&base_expr)?;
                Ok(Par::default().with_exprs(vec![result]))
            }
        }

        Box::new(ResetMethod { outer: self })
    }

    fn ascend_one_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct AscendOneMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> AscendOneMethod<'a> {
            fn ascend_one(&self, base_expr: &Expr) -> Result<Par, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EZipperBody(mut zipper) => {
                        // Check if at root
                        if zipper.current_path.is_empty() {
                            // At root, cannot ascend - return Nil
                            return Ok(Par::default());
                        }

                        // Remove last segment from current_path (ascend one level)
                        zipper.current_path.pop();

                        Ok(Par::default().with_exprs(vec![Expr {
                            expr_instance: Some(ExprInstance::EZipperBody(zipper)),
                        }]))
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("ascendOne"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for AscendOneMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("ascendOne"),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                self.outer.cost.charge(union_cost(1))?;
                self.ascend_one(&base_expr)
            }
        }

        Box::new(AscendOneMethod { outer: self })
    }

    fn ascend_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct AscendMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> AscendMethod<'a> {
            fn ascend(&self, base_expr: &Expr, steps_par: &Par) -> Result<Par, InterpreterError> {
                // Extract integer from Par
                let steps = match steps_par
                    .exprs
                    .first()
                    .and_then(|e| e.expr_instance.as_ref())
                {
                    Some(ExprInstance::GInt(n)) => *n,
                    _ => {
                        return Err(InterpreterError::MethodNotDefined {
                            method: String::from("ascend (requires integer argument)"),
                            other_type: "non-integer".to_string(),
                        })
                    }
                };

                if steps < 0 {
                    return Err(InterpreterError::MethodNotDefined {
                        method: String::from("ascend (steps must be non-negative)"),
                        other_type: format!("negative: {}", steps),
                    });
                }

                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EZipperBody(mut zipper) => {
                        // Remove up to 'steps' segments, cap at root
                        let depth = zipper.current_path.len();
                        let actual_steps = std::cmp::min(steps as usize, depth);

                        // Remove segments from end
                        for _ in 0..actual_steps {
                            zipper.current_path.pop();
                        }

                        Ok(Par::default().with_exprs(vec![Expr {
                            expr_instance: Some(ExprInstance::EZipperBody(zipper)),
                        }]))
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("ascend"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for AscendMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("ascend"),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                let steps_par = self.outer.eval_expr(&args[0], env)?;
                self.outer.cost.charge(union_cost(1))?;
                self.ascend(&base_expr, &steps_par)
            }
        }

        Box::new(AscendMethod { outer: self })
    }

    fn child_count_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct ChildCountMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> ChildCountMethod<'a> {
            fn child_count(&self, base_expr: &Expr) -> Result<i64, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EZipperBody(zipper) => {
                        let pathmap = zipper.pathmap.as_ref().expect("zipper pathmap was None");
                        let pathmap_result =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(pathmap);
                        let rholang_pathmap = pathmap_result.map;

                        // Build prefix from current_path
                        let prefix_key: Vec<u8> = zipper
                            .current_path
                            .iter()
                            .flat_map(|seg| {
                                let mut s = seg.clone();
                                s.push(0xFF);
                                s
                            })
                            .collect();

                        // Find all unique immediate children
                        let mut children: Vec<Vec<u8>> = Vec::new();

                        for (key, _) in rholang_pathmap.iter() {
                            if key.starts_with(&prefix_key) && key.len() > prefix_key.len() {
                                // Extract first segment after prefix
                                let remaining = &key[prefix_key.len()..];
                                if let Some(pos) = remaining.iter().position(|&b| b == 0xFF) {
                                    let segment = remaining[..pos].to_vec();
                                    children.push(segment);
                                }
                            }
                        }

                        // Deduplicate
                        children.sort();
                        children.dedup();

                        Ok(children.len() as i64)
                    }
                    ExprInstance::EPathmapBody(pathmap) => {
                        // For PathMap at root, count top-level paths
                        let pathmap_result =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(&pathmap);
                        let rholang_pathmap = pathmap_result.map;

                        let mut children: Vec<Vec<u8>> = Vec::new();

                        for (key, _) in rholang_pathmap.iter() {
                            // Extract first segment
                            if let Some(pos) = key.iter().position(|&b| b == 0xFF) {
                                let segment = key[..pos].to_vec();
                                children.push(segment);
                            }
                        }

                        children.sort();
                        children.dedup();

                        Ok(children.len() as i64)
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("childCount"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for ChildCountMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("childCount"),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                self.outer.cost.charge(union_cost(1))?;
                let count = self.child_count(&base_expr)?;

                Ok(Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::GInt(count)),
                }]))
            }
        }

        Box::new(ChildCountMethod { outer: self })
    }

    fn descend_first_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct DescendFirstMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> DescendFirstMethod<'a> {
            fn descend_first(&self, base_expr: &Expr) -> Result<Par, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EZipperBody(mut zipper) => {
                        let pathmap = zipper.pathmap.as_ref().expect("zipper pathmap was None");
                        let pathmap_result =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(pathmap);
                        let rholang_pathmap = pathmap_result.map;

                        // Build prefix from current_path
                        let prefix_key: Vec<u8> = zipper
                            .current_path
                            .iter()
                            .flat_map(|seg| {
                                let mut s = seg.clone();
                                s.push(0xFF);
                                s
                            })
                            .collect();

                        // Find all unique immediate children
                        let mut children: Vec<Vec<u8>> = Vec::new();

                        for (key, _) in rholang_pathmap.iter() {
                            if key.starts_with(&prefix_key) && key.len() > prefix_key.len() {
                                // Extract first segment after prefix
                                let remaining = &key[prefix_key.len()..];
                                if let Some(pos) = remaining.iter().position(|&b| b == 0xFF) {
                                    let segment = remaining[..pos].to_vec();
                                    children.push(segment);
                                }
                            }
                        }

                        // Sort and deduplicate for deterministic ordering
                        children.sort();
                        children.dedup();

                        // Get first child
                        if let Some(first_child) = children.first() {
                            zipper.current_path.push(first_child.clone());
                            Ok(Par::default().with_exprs(vec![Expr {
                                expr_instance: Some(ExprInstance::EZipperBody(zipper)),
                            }]))
                        } else {
                            // No children, return Nil
                            Ok(Par::default())
                        }
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("descendFirst"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for DescendFirstMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("descendFirst"),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                self.outer.cost.charge(union_cost(1))?;
                self.descend_first(&base_expr)
            }
        }

        Box::new(DescendFirstMethod { outer: self })
    }

    fn descend_indexed_branch_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct DescendIndexedBranchMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> DescendIndexedBranchMethod<'a> {
            fn descend_indexed(
                &self,
                base_expr: &Expr,
                idx_par: &Par,
            ) -> Result<Par, InterpreterError> {
                // Extract integer index
                let idx = match idx_par.exprs.first().and_then(|e| e.expr_instance.as_ref()) {
                    Some(ExprInstance::GInt(n)) => *n,
                    _ => {
                        return Err(InterpreterError::MethodNotDefined {
                            method: String::from(
                                "descendIndexedBranch (requires integer argument)",
                            ),
                            other_type: "non-integer".to_string(),
                        })
                    }
                };

                if idx < 0 {
                    // Negative index, return Nil
                    return Ok(Par::default());
                }

                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EZipperBody(mut zipper) => {
                        let pathmap = zipper.pathmap.as_ref().expect("zipper pathmap was None");
                        let pathmap_result =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(pathmap);
                        let rholang_pathmap = pathmap_result.map;

                        // Build prefix from current_path
                        let prefix_key: Vec<u8> = zipper
                            .current_path
                            .iter()
                            .flat_map(|seg| {
                                let mut s = seg.clone();
                                s.push(0xFF);
                                s
                            })
                            .collect();

                        // Find all unique immediate children
                        let mut children: Vec<Vec<u8>> = Vec::new();

                        for (key, _) in rholang_pathmap.iter() {
                            if key.starts_with(&prefix_key) && key.len() > prefix_key.len() {
                                // Extract first segment after prefix
                                let remaining = &key[prefix_key.len()..];
                                if let Some(pos) = remaining.iter().position(|&b| b == 0xFF) {
                                    let segment = remaining[..pos].to_vec();
                                    children.push(segment);
                                }
                            }
                        }

                        // Sort and deduplicate for deterministic ordering
                        children.sort();
                        children.dedup();

                        // Get child at index
                        if let Some(child) = children.get(idx as usize) {
                            zipper.current_path.push(child.clone());
                            Ok(Par::default().with_exprs(vec![Expr {
                                expr_instance: Some(ExprInstance::EZipperBody(zipper)),
                            }]))
                        } else {
                            // Index out of bounds, return Nil
                            Ok(Par::default())
                        }
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("descendIndexedBranch"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for DescendIndexedBranchMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if args.len() != 1 {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("descendIndexedBranch"),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                let idx_par = self.outer.eval_expr(&args[0], env)?;
                self.outer.cost.charge(union_cost(1))?;
                self.descend_indexed(&base_expr, &idx_par)
            }
        }

        Box::new(DescendIndexedBranchMethod { outer: self })
    }

    fn to_next_sibling_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct ToNextSiblingMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> ToNextSiblingMethod<'a> {
            fn to_next_sibling(&self, base_expr: &Expr) -> Result<Par, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EZipperBody(mut zipper) => {
                        // Check if at root (no siblings at root)
                        if zipper.current_path.is_empty() {
                            return Ok(Par::default());
                        }

                        let pathmap = zipper.pathmap.as_ref().expect("zipper pathmap was None");
                        let pathmap_result =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(pathmap);
                        let rholang_pathmap = pathmap_result.map;

                        // Get parent path and current segment
                        let current_segment = zipper.current_path.last().unwrap().clone();
                        let parent_path = &zipper.current_path[..zipper.current_path.len() - 1];
                        let parent_key: Vec<u8> = parent_path
                            .iter()
                            .flat_map(|seg| {
                                let mut s = seg.clone();
                                s.push(0xFF);
                                s
                            })
                            .collect();

                        // Find all siblings (children of parent)
                        let mut siblings: Vec<Vec<u8>> = Vec::new();

                        for (key, _) in rholang_pathmap.iter() {
                            if key.starts_with(&parent_key) && key.len() > parent_key.len() {
                                let remaining = &key[parent_key.len()..];
                                if let Some(pos) = remaining.iter().position(|&b| b == 0xFF) {
                                    let segment = remaining[..pos].to_vec();
                                    siblings.push(segment);
                                }
                            }
                        }

                        // Sort and deduplicate for deterministic ordering
                        siblings.sort();
                        siblings.dedup();

                        // Find current position and get next
                        if let Some(current_idx) =
                            siblings.iter().position(|s| s == &current_segment)
                        {
                            if current_idx + 1 < siblings.len() {
                                // Replace current segment with next sibling
                                zipper.current_path.pop();
                                zipper.current_path.push(siblings[current_idx + 1].clone());
                                Ok(Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(ExprInstance::EZipperBody(zipper)),
                                }]))
                            } else {
                                // No next sibling, return Nil
                                Ok(Par::default())
                            }
                        } else {
                            // Current not found (shouldn't happen), return Nil
                            Ok(Par::default())
                        }
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("toNextSibling"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for ToNextSiblingMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("toNextSibling"),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                self.outer.cost.charge(union_cost(1))?;
                self.to_next_sibling(&base_expr)
            }
        }

        Box::new(ToNextSiblingMethod { outer: self })
    }

    fn to_prev_sibling_method<'a>(&'a self) -> Box<dyn Method + 'a> {
        struct ToPrevSiblingMethod<'a> {
            outer: &'a DebruijnInterpreter,
        }

        impl<'a> ToPrevSiblingMethod<'a> {
            fn to_prev_sibling(&self, base_expr: &Expr) -> Result<Par, InterpreterError> {
                match base_expr.expr_instance.clone().unwrap() {
                    ExprInstance::EZipperBody(mut zipper) => {
                        // Check if at root (no siblings at root)
                        if zipper.current_path.is_empty() {
                            return Ok(Par::default());
                        }

                        let pathmap = zipper.pathmap.as_ref().expect("zipper pathmap was None");
                        let pathmap_result =
                            PathMapCrateTypeMapper::e_pathmap_to_rholang_pathmap(pathmap);
                        let rholang_pathmap = pathmap_result.map;

                        // Get parent path and current segment
                        let current_segment = zipper.current_path.last().unwrap().clone();
                        let parent_path = &zipper.current_path[..zipper.current_path.len() - 1];
                        let parent_key: Vec<u8> = parent_path
                            .iter()
                            .flat_map(|seg| {
                                let mut s = seg.clone();
                                s.push(0xFF);
                                s
                            })
                            .collect();

                        // Find all siblings (children of parent)
                        let mut siblings: Vec<Vec<u8>> = Vec::new();

                        for (key, _) in rholang_pathmap.iter() {
                            if key.starts_with(&parent_key) && key.len() > parent_key.len() {
                                let remaining = &key[parent_key.len()..];
                                if let Some(pos) = remaining.iter().position(|&b| b == 0xFF) {
                                    let segment = remaining[..pos].to_vec();
                                    siblings.push(segment);
                                }
                            }
                        }

                        // Sort and deduplicate for deterministic ordering
                        siblings.sort();
                        siblings.dedup();

                        // Find current position and get previous
                        if let Some(current_idx) =
                            siblings.iter().position(|s| s == &current_segment)
                        {
                            if current_idx > 0 {
                                // Replace current segment with previous sibling
                                zipper.current_path.pop();
                                zipper.current_path.push(siblings[current_idx - 1].clone());
                                Ok(Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(ExprInstance::EZipperBody(zipper)),
                                }]))
                            } else {
                                // No previous sibling, return Nil
                                Ok(Par::default())
                            }
                        } else {
                            // Current not found (shouldn't happen), return Nil
                            Ok(Par::default())
                        }
                    }
                    other => Err(InterpreterError::MethodNotDefined {
                        method: String::from("toPrevSibling"),
                        other_type: get_type(other),
                    }),
                }
            }
        }

        impl<'a> Method for ToPrevSiblingMethod<'a> {
            fn apply(
                &self,
                p: Par,
                args: Vec<Par>,
                env: &Env<Par>,
            ) -> Result<Par, InterpreterError> {
                if !args.is_empty() {
                    return Err(InterpreterError::MethodArgumentNumberMismatch {
                        method: String::from("toPrevSibling"),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let base_expr = self.outer.eval_single_expr(&p, env)?;
                self.outer.cost.charge(union_cost(1))?;
                self.to_prev_sibling(&base_expr)
            }
        }

        Box::new(ToPrevSiblingMethod { outer: self })
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
        table.insert("getLeaf".to_string(), self.get_leaf_method());
        table.insert("getSubtrie".to_string(), self.get_subtrie_method());
        table.insert("setLeaf".to_string(), self.set_leaf_method());
        table.insert("setSubtrie".to_string(), self.set_subtrie_method());
        table.insert("removeLeaf".to_string(), self.remove_leaf_method());
        table.insert("removeBranches".to_string(), self.remove_branches_method());
        table.insert("graft".to_string(), self.graft_method());
        table.insert("joinInto".to_string(), self.join_into_method());
        table.insert("atPath".to_string(), self.at_path_method());
        table.insert("pathExists".to_string(), self.path_exists_method());
        table.insert("createPath".to_string(), self.create_path_method());
        table.insert("prunePath".to_string(), self.prune_path_method());
        table.insert("reset".to_string(), self.reset_method());
        // Advanced navigation methods
        table.insert("ascendOne".to_string(), self.ascend_one_method());
        table.insert("ascend".to_string(), self.ascend_method());
        table.insert("toNextSibling".to_string(), self.to_next_sibling_method());
        table.insert("toPrevSibling".to_string(), self.to_prev_sibling_method());
        table.insert("descendFirst".to_string(), self.descend_first_method());
        table.insert(
            "descendIndexedBranch".to_string(),
            self.descend_indexed_branch_method(),
        );
        table.insert("childCount".to_string(), self.child_count_method());
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
            || !p.receives.is_empty()
            || !p.news.is_empty()
            || !p.matches.is_empty()
            || !p.unforgeables.is_empty()
            || !p.bundles.is_empty()
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
            || !p.receives.is_empty()
            || !p.news.is_empty()
            || !p.matches.is_empty()
            || !p.exprs.is_empty()
            || !p.bundles.is_empty()
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
        urn_map: Arc<HashMap<String, Par>>,
        merge_chs: Arc<RwLock<HashSet<Par>>>,
        mergeable_tag_name: Par,
        cost: _cost,
    ) -> Arc<Self> {
        let reducer_cell = Arc::new(std::sync::OnceLock::new());
        let dispatcher = Arc::new(RholangAndScalaDispatcher {
            _dispatch_table: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            reducer: reducer_cell.clone(),
        });

        let reducer = Arc::new(DebruijnInterpreter {
            space,
            dispatcher: dispatcher.clone(),
            urn_map,
            merge_chs,
            mergeable_tag_name,
            cost: cost.clone(),
            substitute: Substitute { cost: cost.clone() },
            split_post_match_config: Arc::new(RwLock::new(SplitPostMatchConfig::default())),
            continuation_store: Arc::new(RwLock::new(InMemoryContinuationStore::default())),
            continuation_nonce: Arc::new(AtomicU64::new(0)),
        });

        reducer_cell.set(Arc::downgrade(&reducer)).ok().unwrap();
        reducer
    }

    pub fn set_split_post_match_config(&self, config: SplitPostMatchConfig) {
        let mut config_guard = self.split_post_match_config.write().unwrap();
        *config_guard = config;
    }

    pub fn continuation_store_len(&self) -> usize {
        self.continuation_store.read().unwrap().len()
    }

    pub fn list_continuation_handles(&self) -> Vec<ContinuationHandle> {
        self.continuation_store.read().unwrap().handles()
    }

    pub fn load_continuation(
        &self,
        handle: &ContinuationHandle,
    ) -> Result<PersistedContinuation, InterpreterError> {
        self.continuation_store
            .read()
            .unwrap()
            .load(handle)
            .map_err(map_continuation_store_error)
    }

    pub async fn execute_continuation(
        &self,
        handle: &ContinuationHandle,
        expected_version: u64,
        gas_limit: i64,
    ) -> Result<ContinuationExecutionResult, InterpreterError> {
        let persisted = self.load_continuation(handle)?;
        if persisted.status != ContinuationStatus::Active {
            return Err(InterpreterError::IllegalArgumentError(format!(
                "continuation {}:{} is not active (status {:?})",
                hex::encode(&persisted.handle.origin),
                persisted.handle.nonce,
                persisted.status
            )));
        }
        if persisted.version != expected_version {
            return Err(map_continuation_store_error(
                ContinuationStoreError::VersionMismatch {
                    handle: persisted.handle.clone(),
                    expected: expected_version,
                    actual: persisted.version,
                },
            ));
        }

        let effective_gas_limit = gas_limit.min(persisted.gas_limit_per_step);
        let state = deserialize_post_match_state(&persisted.serialized_state)?;
        let step_result = match self.reduce_step(state, effective_gas_limit).await {
            Ok(step_result) => step_result,
            Err(error) => {
                if let Err(mark_error) = self
                    .continuation_store
                    .write()
                    .unwrap()
                    .fail_with_version(handle, expected_version)
                    .map_err(map_continuation_store_error)
                {
                    return Err(InterpreterError::BugFoundError(format!(
                        "continuation step failed and could not mark continuation as failed: {mark_error}"
                    )));
                }

                metrics::counter!("rholang.continuation.resume.failed").increment(1);
                return Err(error);
            }
        };

        let StepResult {
            dispatch_result: _dispatch_result,
            emitted_effects,
            next_state,
            consumed_gas,
            terminal_status,
        } = step_result;

        match terminal_status {
            StepTerminalStatus::Completed => {
                let updated = self
                    .continuation_store
                    .write()
                    .unwrap()
                    .complete_with_version(handle, expected_version)
                    .map_err(map_continuation_store_error)?;
                metrics::counter!("rholang.continuation.resume.completed").increment(1);
                Ok(ContinuationExecutionResult {
                    handle: updated.handle,
                    status: ContinuationExecutionStatus::Completed,
                    version: updated.version,
                    state_root: updated.state_root,
                    consumed_gas,
                    emitted_effects,
                })
            }
            StepTerminalStatus::Suspended => {
                let suspended_state = next_state.ok_or_else(|| {
                    InterpreterError::BugFoundError(
                        "continuation step suspended without next state".to_string(),
                    )
                })?;
                let serialized_state = serialize_post_match_state(&suspended_state)?;
                let updated = self
                    .continuation_store
                    .write()
                    .unwrap()
                    .update_state_with_version(
                        handle,
                        expected_version,
                        serialized_state,
                        persisted.gas_limit_per_step,
                    )
                    .map_err(map_continuation_store_error)?;
                metrics::counter!("rholang.continuation.resume.suspended").increment(1);
                Ok(ContinuationExecutionResult {
                    handle: updated.handle,
                    status: ContinuationExecutionStatus::Suspended,
                    version: updated.version,
                    state_root: updated.state_root,
                    consumed_gas,
                    emitted_effects,
                })
            }
        }
    }
}

fn map_continuation_store_error(error: ContinuationStoreError) -> InterpreterError {
    match error {
        ContinuationStoreError::AlreadyExists(handle) => InterpreterError::BugFoundError(format!(
            "continuation already exists: {}:{}",
            hex::encode(&handle.origin),
            handle.nonce
        )),
        ContinuationStoreError::NotFound(handle) => InterpreterError::IllegalArgumentError(
            format!(
                "continuation not found: {}:{}",
                hex::encode(&handle.origin),
                handle.nonce
            ),
        ),
        ContinuationStoreError::VersionMismatch {
            handle,
            expected,
            actual,
        } => InterpreterError::IllegalArgumentError(format!(
            "stale continuation version for {}:{} (expected {}, actual {})",
            hex::encode(&handle.origin),
            handle.nonce,
            expected,
            actual
        )),
        ContinuationStoreError::InvalidTransition { handle, from, to } => {
            InterpreterError::IllegalArgumentError(format!(
                "invalid continuation transition for {}:{} from {:?} to {:?}",
                hex::encode(&handle.origin),
                handle.nonce,
                from,
                to
            ))
        }
        ContinuationStoreError::UnsupportedEncodingVersion(version) => {
            InterpreterError::DecodeError(format!(
                "unsupported continuation encoding version: {}",
                version
            ))
        }
        ContinuationStoreError::InvalidFormat(message) => {
            InterpreterError::DecodeError(format!("invalid continuation encoding: {}", message))
        }
    }
}

const POST_MATCH_STATE_ENCODING_VERSION: u8 = 1;

fn serialize_post_match_state(state: &PostMatchExecState) -> Result<Vec<u8>, InterpreterError> {
    let mut out = Vec::new();
    out.push(POST_MATCH_STATE_ENCODING_VERSION);

    let control_tag = match &state.control {
        PostMatchControlState::DispatchOnly => 0u8,
        PostMatchControlState::DispatchAndReconsume(_) => 1u8,
        PostMatchControlState::DispatchAndReproduce(_) => 2u8,
        PostMatchControlState::DispatchAndRestorePeeks => 3u8,
        PostMatchControlState::Reconsume(_) => 4u8,
        PostMatchControlState::Reproduce(_) => 5u8,
        PostMatchControlState::RestorePeeks => 6u8,
    };
    out.push(control_tag);

    serialize_dispatch_exec_state(&state.dispatch, &mut out);

    match &state.control {
        PostMatchControlState::DispatchAndReconsume(receive_state)
        | PostMatchControlState::Reconsume(receive_state) => {
            serialize_receive_registration_state(receive_state, &mut out);
        }
        PostMatchControlState::DispatchAndReproduce(produce_state)
        | PostMatchControlState::Reproduce(produce_state) => {
            serialize_produce_registration_state(produce_state, &mut out);
        }
        PostMatchControlState::DispatchOnly
        | PostMatchControlState::DispatchAndRestorePeeks
        | PostMatchControlState::RestorePeeks => {}
    }

    Ok(out)
}

fn serialize_receive_registration_state(receive_state: &ReceiveRegistrationState, out: &mut Vec<u8>) {
    put_u32(out, receive_state.binds.len() as u32);
    for (bind_pattern, source) in receive_state.binds.iter() {
        put_bytes(out, &bind_pattern.encode_to_vec());
        put_bytes(out, &source.encode_to_vec());
    }
    put_bytes(out, &receive_state.body.encode_to_vec());
    put_bool(out, receive_state.persistent);
    put_bool(out, receive_state.peek);
}

fn serialize_produce_registration_state(produce_state: &ProduceRegistrationState, out: &mut Vec<u8>) {
    put_bytes(out, &produce_state.chan.encode_to_vec());
    put_bytes(out, &produce_state.data.encode_to_vec());
    put_bool(out, produce_state.persistent);
}

fn serialize_dispatch_exec_state(dispatch: &DispatchExecState, out: &mut Vec<u8>) {
    put_bytes(out, &dispatch.continuation.encode_to_vec());
    put_u32(out, dispatch.data_list.len() as u32);
    for (chan, matched_data, removed_data, persistent) in dispatch.data_list.iter() {
        put_bytes(out, &chan.encode_to_vec());
        put_bytes(out, &matched_data.encode_to_vec());
        put_bytes(out, &removed_data.encode_to_vec());
        put_bool(out, *persistent);
    }
    put_bool(out, dispatch.is_replay);
    put_u32(out, dispatch.previous_output.len() as u32);
    for par in dispatch.previous_output.iter() {
        put_bytes(out, &par.encode_to_vec());
    }
    put_i64(out, dispatch._remaining_phlo);
}

fn put_bool(out: &mut Vec<u8>, value: bool) {
    out.push(if value { 1 } else { 0 });
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_i64(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_u32(out, bytes.len() as u32);
    out.extend_from_slice(bytes);
}

fn deserialize_post_match_state(input: &[u8]) -> Result<PostMatchExecState, InterpreterError> {
    let mut idx = 0usize;
    let encoding_version = read_u8(input, &mut idx)?;
    if encoding_version != POST_MATCH_STATE_ENCODING_VERSION {
        return Err(InterpreterError::DecodeError(format!(
            "unsupported post-match state encoding version: {}",
            encoding_version
        )));
    }

    let control_tag = read_u8(input, &mut idx)?;
    let dispatch = deserialize_dispatch_exec_state(input, &mut idx)?;
    let control = match control_tag {
        0 => PostMatchControlState::DispatchOnly,
        1 => PostMatchControlState::DispatchAndReconsume(deserialize_receive_registration_state(
            input, &mut idx,
        )?),
        2 => PostMatchControlState::DispatchAndReproduce(deserialize_produce_registration_state(
            input, &mut idx,
        )?),
        3 => PostMatchControlState::DispatchAndRestorePeeks,
        4 => PostMatchControlState::Reconsume(deserialize_receive_registration_state(
            input, &mut idx,
        )?),
        5 => PostMatchControlState::Reproduce(deserialize_produce_registration_state(
            input, &mut idx,
        )?),
        6 => PostMatchControlState::RestorePeeks,
        _ => {
            return Err(InterpreterError::DecodeError(format!(
                "unknown post-match control tag: {}",
                control_tag
            )))
        }
    };

    if idx != input.len() {
        return Err(InterpreterError::DecodeError(
            "trailing bytes in post-match state encoding".to_string(),
        ));
    }

    Ok(PostMatchExecState { dispatch, control })
}

fn deserialize_receive_registration_state(
    input: &[u8],
    idx: &mut usize,
) -> Result<ReceiveRegistrationState, InterpreterError> {
    let bind_count = read_u32(input, idx)? as usize;
    let mut binds = Vec::with_capacity(bind_count);
    for _ in 0..bind_count {
        let bind_pattern = decode_message::<BindPattern>(&read_bytes(input, idx)?, "BindPattern")?;
        let source = decode_message::<Par>(&read_bytes(input, idx)?, "Par")?;
        binds.push((bind_pattern, source));
    }

    let body = decode_message::<ParWithRandom>(&read_bytes(input, idx)?, "ParWithRandom")?;
    let persistent = read_bool(input, idx)?;
    let peek = read_bool(input, idx)?;

    Ok(ReceiveRegistrationState {
        binds,
        body,
        persistent,
        peek,
    })
}

fn deserialize_produce_registration_state(
    input: &[u8],
    idx: &mut usize,
) -> Result<ProduceRegistrationState, InterpreterError> {
    let chan = decode_message::<Par>(&read_bytes(input, idx)?, "Par")?;
    let data = decode_message::<ListParWithRandom>(&read_bytes(input, idx)?, "ListParWithRandom")?;
    let persistent = read_bool(input, idx)?;
    Ok(ProduceRegistrationState {
        chan,
        data,
        persistent,
    })
}

fn deserialize_dispatch_exec_state(
    input: &[u8],
    idx: &mut usize,
) -> Result<DispatchExecState, InterpreterError> {
    let continuation = decode_message::<TaggedContinuation>(
        &read_bytes(input, idx)?,
        "TaggedContinuation",
    )?;

    let data_list_count = read_u32(input, idx)? as usize;
    let mut data_list = Vec::with_capacity(data_list_count);
    for _ in 0..data_list_count {
        let chan = decode_message::<Par>(&read_bytes(input, idx)?, "Par")?;
        let matched_data =
            decode_message::<ListParWithRandom>(&read_bytes(input, idx)?, "ListParWithRandom")?;
        let removed_data =
            decode_message::<ListParWithRandom>(&read_bytes(input, idx)?, "ListParWithRandom")?;
        let persistent = read_bool(input, idx)?;
        data_list.push((chan, matched_data, removed_data, persistent));
    }

    let is_replay = read_bool(input, idx)?;
    let previous_output_count = read_u32(input, idx)? as usize;
    let mut previous_output = Vec::with_capacity(previous_output_count);
    for _ in 0..previous_output_count {
        previous_output.push(decode_message::<Par>(&read_bytes(input, idx)?, "Par")?);
    }
    let remaining_phlo = read_i64(input, idx)?;

    Ok(DispatchExecState {
        continuation,
        data_list,
        is_replay,
        previous_output,
        _remaining_phlo: remaining_phlo,
    })
}

fn decode_message<T>(bytes: &[u8], type_name: &str) -> Result<T, InterpreterError>
where
    T: Message + Default,
{
    T::decode(bytes).map_err(|e| {
        InterpreterError::DecodeError(format!("failed to decode {} from continuation state: {}", type_name, e))
    })
}

fn read_bool(input: &[u8], idx: &mut usize) -> Result<bool, InterpreterError> {
    match read_u8(input, idx)? {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(InterpreterError::DecodeError(format!(
            "invalid bool encoding: {}",
            value
        ))),
    }
}

fn read_u8(input: &[u8], idx: &mut usize) -> Result<u8, InterpreterError> {
    if *idx + 1 > input.len() {
        return Err(InterpreterError::DecodeError(
            "unexpected end of bytes while decoding u8".to_string(),
        ));
    }
    let value = input[*idx];
    *idx += 1;
    Ok(value)
}

fn read_u32(input: &[u8], idx: &mut usize) -> Result<u32, InterpreterError> {
    if *idx + 4 > input.len() {
        return Err(InterpreterError::DecodeError(
            "unexpected end of bytes while decoding u32".to_string(),
        ));
    }
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&input[*idx..*idx + 4]);
    *idx += 4;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i64(input: &[u8], idx: &mut usize) -> Result<i64, InterpreterError> {
    if *idx + 8 > input.len() {
        return Err(InterpreterError::DecodeError(
            "unexpected end of bytes while decoding i64".to_string(),
        ));
    }
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&input[*idx..*idx + 8]);
    *idx += 8;
    Ok(i64::from_le_bytes(bytes))
}

fn read_bytes(input: &[u8], idx: &mut usize) -> Result<Vec<u8>, InterpreterError> {
    let len = read_u32(input, idx)? as usize;
    let end = idx.checked_add(len).ok_or_else(|| {
        InterpreterError::DecodeError(
            "byte length overflow while decoding continuation state".to_string(),
        )
    })?;
    if end > input.len() {
        return Err(InterpreterError::DecodeError(
            "unexpected end of bytes while decoding length-delimited bytes".to_string(),
        ));
    }

    let bytes = input[*idx..end].to_vec();
    *idx = end;
    Ok(bytes)
}

fn get_type(expr_instance: ExprInstance) -> String {
    match expr_instance {
        ExprInstance::GBool(_) => String::from("bool"),
        ExprInstance::GInt(_) => String::from("int"),
        ExprInstance::GDouble(_) => String::from("float"),
        ExprInstance::GBigInt(_) => String::from("bigint"),
        ExprInstance::GBigRat(_) => String::from("bigrat"),
        ExprInstance::GFixedPoint(_) => String::from("fixedpoint"),
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
        ExprInstance::EZipperBody(_) => String::from("zipper"),
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

fn par_contains_nan_double(par: &Par) -> bool {
    par.exprs.iter().any(|e| match &e.expr_instance {
        Some(ExprInstance::GDouble(bits)) => f64::from_bits(*bits).is_nan(),
        Some(ExprInstance::EListBody(list)) => list.ps.iter().any(par_contains_nan_double),
        Some(ExprInstance::ETupleBody(tuple)) => tuple.ps.iter().any(par_contains_nan_double),
        Some(ExprInstance::ESetBody(set)) => set.ps.iter().any(par_contains_nan_double),
        Some(ExprInstance::EMapBody(map)) => map.kvs.iter().any(|kv| {
            kv.key.as_ref().map_or(false, par_contains_nan_double)
                || kv.value.as_ref().map_or(false, par_contains_nan_double)
        }),
        _ => false,
    })
}

fn bytes_to_bigint(bytes: &[u8]) -> num_bigint::BigInt {
    if bytes.is_empty() {
        num_bigint::BigInt::from(0)
    } else {
        num_bigint::BigInt::from_signed_bytes_be(bytes)
    }
}

fn bigint_to_bytes(n: &num_bigint::BigInt) -> Vec<u8> {
    use num_traits::Zero;
    if n.is_zero() {
        vec![0]
    } else {
        n.to_signed_bytes_be()
    }
}

fn make_bigint_expr(bytes: Vec<u8>, _op: &str) -> Result<Expr, InterpreterError> {
    Ok(Expr {
        expr_instance: Some(ExprInstance::GBigInt(bytes)),
    })
}

fn make_bigrat_expr(
    rat: models::rhoapi::GBigRational,
    _op: &str,
) -> Result<Expr, InterpreterError> {
    Ok(Expr {
        expr_instance: Some(ExprInstance::GBigRat(rat)),
    })
}

fn make_fixedpoint_expr(
    fp: models::rhoapi::GFixedPoint,
    _op: &str,
) -> Result<Expr, InterpreterError> {
    Ok(Expr {
        expr_instance: Some(ExprInstance::GFixedPoint(fp)),
    })
}

fn is_zero_twos_complement(bytes: &[u8]) -> bool {
    bytes.is_empty() || bytes.iter().all(|&b| b == 0)
}

fn negate_twos_complement(bytes: &[u8]) -> Vec<u8> {
    let n = bytes_to_bigint(bytes);
    bigint_to_bytes(&(-n))
}

fn add_twos_complement(a: &[u8], b: &[u8]) -> Vec<u8> {
    let result = bytes_to_bigint(a) + bytes_to_bigint(b);
    bigint_to_bytes(&result)
}

fn subtract_twos_complement(a: &[u8], b: &[u8]) -> Vec<u8> {
    let result = bytes_to_bigint(a) - bytes_to_bigint(b);
    bigint_to_bytes(&result)
}

fn multiply_twos_complement(a: &[u8], b: &[u8]) -> Vec<u8> {
    let result = bytes_to_bigint(a) * bytes_to_bigint(b);
    bigint_to_bytes(&result)
}

fn divide_twos_complement(a: &[u8], b: &[u8]) -> Vec<u8> {
    let result = bytes_to_bigint(a) / bytes_to_bigint(b);
    bigint_to_bytes(&result)
}

fn modulo_twos_complement(a: &[u8], b: &[u8]) -> Vec<u8> {
    let result = bytes_to_bigint(a) % bytes_to_bigint(b);
    bigint_to_bytes(&result)
}

fn compare_twos_complement_bytes(a: &[u8], b: &[u8]) -> i32 {
    match bytes_to_bigint(a).cmp(&bytes_to_bigint(b)) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

fn bytes_to_bigrat(
    rat: &models::rhoapi::GBigRational,
) -> num_rational::BigRational {
    num_rational::BigRational::new(
        bytes_to_bigint(&rat.numerator),
        bytes_to_bigint(&rat.denominator),
    )
}

fn bigrat_to_proto(r: &num_rational::BigRational) -> models::rhoapi::GBigRational {
    models::rhoapi::GBigRational {
        numerator: bigint_to_bytes(r.numer()),
        denominator: bigint_to_bytes(r.denom()),
    }
}

fn compare_big_rationals(
    a: &models::rhoapi::GBigRational,
    b: &models::rhoapi::GBigRational,
) -> i32 {
    match bytes_to_bigrat(a).cmp(&bytes_to_bigrat(b)) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

fn add_big_rationals(
    a: &models::rhoapi::GBigRational,
    b: &models::rhoapi::GBigRational,
) -> models::rhoapi::GBigRational {
    bigrat_to_proto(&(bytes_to_bigrat(a) + bytes_to_bigrat(b)))
}

fn subtract_big_rationals(
    a: &models::rhoapi::GBigRational,
    b: &models::rhoapi::GBigRational,
) -> models::rhoapi::GBigRational {
    bigrat_to_proto(&(bytes_to_bigrat(a) - bytes_to_bigrat(b)))
}

fn multiply_big_rationals(
    a: &models::rhoapi::GBigRational,
    b: &models::rhoapi::GBigRational,
) -> models::rhoapi::GBigRational {
    bigrat_to_proto(&(bytes_to_bigrat(a) * bytes_to_bigrat(b)))
}

fn divide_big_rationals(
    a: &models::rhoapi::GBigRational,
    b: &models::rhoapi::GBigRational,
) -> models::rhoapi::GBigRational {
    bigrat_to_proto(&(bytes_to_bigrat(a) / bytes_to_bigrat(b)))
}

fn compare_fixed_points(
    a: &models::rhoapi::GFixedPoint,
    b: &models::rhoapi::GFixedPoint,
) -> Result<i32, InterpreterError> {
    if a.scale != b.scale {
        return Err(InterpreterError::OperatorExpectedError {
            op: "cmp".to_string(),
            expected: format!("FixedPoint(p{})", a.scale),
            other_type: format!("FixedPoint(p{})", b.scale),
        });
    }
    Ok(compare_twos_complement_bytes(&a.unscaled, &b.unscaled))
}

fn multiply_fixed_points(
    a: &models::rhoapi::GFixedPoint,
    b: &models::rhoapi::GFixedPoint,
) -> models::rhoapi::GFixedPoint {
    debug_assert_eq!(a.scale, b.scale, "multiply_fixed_points called with mismatched scales");
    // Scale-preserving: (ua * ub) / 10^scale, using floor division
    let ua = bytes_to_bigint(&a.unscaled);
    let ub = bytes_to_bigint(&b.unscaled);
    let raw = &ua * &ub;
    let ten = num_bigint::BigInt::from(10);
    let scale_factor = num_traits::pow::pow(ten, a.scale as usize);
    let one = num_bigint::BigInt::from(1);
    let unscaled = if raw < num_bigint::BigInt::from(0) {
        // Floor division for negative values
        let abs_raw = -&raw;
        -((&abs_raw - &one) / &scale_factor + &one)
    } else {
        &raw / &scale_factor
    };
    models::rhoapi::GFixedPoint {
        unscaled: bigint_to_bytes(&unscaled),
        scale: a.scale,
    }
}

fn divide_fixed_points(
    a: &models::rhoapi::GFixedPoint,
    b: &models::rhoapi::GFixedPoint,
) -> models::rhoapi::GFixedPoint {
    debug_assert_eq!(a.scale, b.scale, "divide_fixed_points called with mismatched scales");
    let ten = num_bigint::BigInt::from(10);
    let factor = num_traits::pow::pow(ten, b.scale as usize);
    let scaled = bytes_to_bigint(&a.unscaled) * factor;
    let result = scaled / bytes_to_bigint(&b.unscaled);
    models::rhoapi::GFixedPoint {
        unscaled: bigint_to_bytes(&result),
        scale: a.scale,
    }
}
