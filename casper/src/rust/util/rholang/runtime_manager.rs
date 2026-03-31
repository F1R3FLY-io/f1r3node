// See casper/src/main/scala/coop/rchain/casper/util/rholang/RuntimeManager.scala
// See casper/src/main/scala/coop/rchain/casper/util/rholang/RuntimeManagerSyntax.scala

use dashmap::DashMap;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::Hash;
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};

use crypto::rust::hash::blake2b256::Blake2b256;
use crypto::rust::signatures::signed::Signed;
use hex::ToHex;
use models::rhoapi::{BindPattern, ListParWithRandom, Par, TaggedContinuation};
use models::rust::block::state_hash::{StateHash, StateHashSerde};
use models::rust::block_hash::BlockHash;
use models::rust::casper::protocol::casper_message::{
    Bond, DeployData, Event, ProcessedDeploy, ProcessedSystemDeploy, SystemDeployData,
};
use models::rust::validator::Validator;
use rholang::rust::interpreter::external_services::ExternalServices;
use rholang::rust::interpreter::matcher::r#match::Matcher;
use rholang::rust::interpreter::merging::rholang_merging_logic::{
    DeployMergeableData, NumberChannel, RholangMergingLogic,
};
use rholang::rust::interpreter::rho_runtime::{
    self, RhoHistoryRepository, RhoRuntime, RhoRuntimeImpl,
};
use rholang::rust::interpreter::system_processes::BlockData;
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use rspace_plus_plus::rspace::merger::merging_logic::{NumberChannelsDiff, NumberChannelsEndVal};
use rspace_plus_plus::rspace::replay_rspace::ReplayRSpace;
use rspace_plus_plus::rspace::rspace::{RSpace, RSpaceStore};
use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;
use shared::rust::store::key_value_store::KvStoreError;
use shared::rust::store::key_value_typed_store::KeyValueTypedStore;
use shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl;
use shared::rust::ByteVector;

use crate::rust::errors::CasperError;
use crate::rust::merging::block_index::BlockIndex;
use crate::rust::metrics_constants::{
    BLOCK_INDEX_CACHE_SIZE_METRIC, CASPER_METRICS_SOURCE, PARENTS_POST_STATE_CACHE_SIZE_METRIC,
};
use crate::rust::rholang::replay_runtime::ReplayRuntimeOps;
use crate::rust::rholang::runtime::RuntimeOps;
use crate::rust::util::rholang::replay_cache::{
    InMemoryReplayCache, ReplayCache, ReplayCacheEntry, ReplayCacheKey,
};
use crate::rust::util::rholang::state_hash_cache::StateHashCache;

type MergeableStore = KeyValueTypedStoreImpl<ByteVector, Vec<DeployMergeableData>>;

#[derive(serde::Serialize, serde::Deserialize)]
struct MergeableKey {
    state_hash: StateHashSerde,
    #[serde(with = "shared::rust::serde_bytes")]
    creator: prost::bytes::Bytes,
    seq_num: i32,
}

#[derive(Clone)]
pub struct RuntimeManager {
    pub space: RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
    pub replay_space: ReplayRSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
    pub history_repo: RhoHistoryRepository,
    pub mergeable_store: MergeableStore,
    pub mergeable_tag_name: Par,
    // TODO: make proper storage for block indices - OLD
    pub block_index_cache: Arc<DashMap<BlockHash, BlockIndex>>,
    pub block_index_cache_order: Arc<Mutex<VecDeque<BlockHash>>>,
    pub active_validators_cache: Arc<DashMap<StateHash, Vec<Validator>>>,
    pub active_validators_cache_order: Arc<Mutex<VecDeque<StateHash>>>,
    /// Cache for merged parent post-state computation keyed by parent-set snapshot context.
    pub parents_post_state_cache: Arc<DashMap<ParentsPostStateCacheKey, ParentsPostStateCacheVal>>,
    pub parents_post_state_cache_order: Arc<Mutex<VecDeque<ParentsPostStateCacheKey>>>,
    /// Optional replay cache for delta replay optimization
    pub replay_cache: Option<Arc<InMemoryReplayCache>>,
    /// Optional state hash cache for skipping known replays
    pub state_hash_cache: Option<Arc<StateHashCache>>,
    pub external_services: ExternalServices,
    /// The state hash after bootstrap_registry — the genesis block's pre-state.
    /// Set during compute_genesis(). Used by compute_parents_post_state for the
    /// 0-parents (genesis) case instead of a hardcoded constant.
    pub empty_state_hash: Option<StateHash>,
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct ParentsPostStateCacheKey {
    pub sorted_parent_hashes: Vec<BlockHash>,
    // Snapshot LFB participates in visible-ancestor filtering, so cache key must include it.
    pub snapshot_lfb_hash: BlockHash,
    pub disable_late_block_filtering: bool,
}

pub type ParentsPostStateCacheVal = (StateHash, Vec<prost::bytes::Bytes>);

impl RuntimeManager {
    const MAX_BLOCK_INDEX_CACHE_ENTRIES: usize = 64;
    const MAX_BLOCK_INDEX_CACHE_ENTRIES_ENV: &str = "F1R3_BLOCK_INDEX_CACHE_MAX_ENTRIES";
    const MAX_PARENTS_POST_STATE_CACHE_ENTRIES: usize = 128;
    const MAX_PARENTS_POST_STATE_CACHE_ENTRIES_ENV: &str =
        "F1R3_PARENTS_POST_STATE_CACHE_MAX_ENTRIES";
    const MAX_ACTIVE_VALIDATORS_CACHE_ENTRIES: usize = 256;
    const MAX_ACTIVE_VALIDATORS_CACHE_ENTRIES_ENV: &str =
        "F1R3_ACTIVE_VALIDATORS_CACHE_MAX_ENTRIES";
    const MAX_REPLAY_CACHE_ENTRIES: usize = 256;
    const MAX_REPLAY_CACHE_ENTRIES_ENV: &str = "F1R3_REPLAY_CACHE_MAX_ENTRIES";
    const MAX_REPLAY_CACHE_EVENT_LOG_ENTRIES: usize = 2048;
    const MAX_REPLAY_CACHE_EVENT_LOG_ENTRIES_ENV: &str = "F1R3_REPLAY_CACHE_MAX_EVENT_LOG_ENTRIES";
    const MAX_STATE_HASH_CACHE_ENTRIES: usize = 0;
    const MAX_STATE_HASH_CACHE_ENTRIES_ENV: &str = "F1R3_STATE_HASH_CACHE_MAX_ENTRIES";

    fn collect_replay_logs(
        usr_processed: &[ProcessedDeploy],
        sys_processed: &[ProcessedSystemDeploy],
    ) -> Vec<Event> {
        let user_log_len: usize = usr_processed.iter().map(|pd| pd.deploy_log.len()).sum();
        let sys_log_len: usize = sys_processed
            .iter()
            .map(|psd| match psd {
                ProcessedSystemDeploy::Succeeded { event_list, .. } => event_list.len(),
                ProcessedSystemDeploy::Failed { event_list, .. } => event_list.len(),
            })
            .sum();

        let mut all_logs = Vec::with_capacity(user_log_len + sys_log_len);

        for pd in usr_processed {
            all_logs.extend(pd.deploy_log.iter().cloned());
        }

        for psd in sys_processed {
            match psd {
                ProcessedSystemDeploy::Succeeded { event_list, .. } => {
                    all_logs.extend(event_list.iter().cloned());
                }
                ProcessedSystemDeploy::Failed { event_list, .. } => {
                    all_logs.extend(event_list.iter().cloned());
                }
            }
        }

        all_logs
    }

    fn replay_payload_hash(
        usr_processed: &[ProcessedDeploy],
        sys_processed: &[ProcessedSystemDeploy],
        is_genesis: bool,
    ) -> Vec<u8> {
        #[inline]
        fn push_len_prefixed(bytes: &mut Vec<u8>, data: &[u8]) {
            bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
            bytes.extend_from_slice(data);
        }

        // Fingerprint replay-relevant payload so cache keys stay safe under adversarial input.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(usr_processed.len() as u64).to_le_bytes());
        for pd in usr_processed {
            push_len_prefixed(&mut bytes, &pd.deploy.sig);
            bytes.extend_from_slice(&pd.cost.cost.to_le_bytes());
            bytes.push(u8::from(pd.is_failed));
            match &pd.system_deploy_error {
                Some(err) => {
                    bytes.push(1);
                    push_len_prefixed(&mut bytes, err.as_bytes());
                }
                None => bytes.push(0),
            }
        }
        bytes.extend_from_slice(&(sys_processed.len() as u64).to_le_bytes());
        for psd in sys_processed {
            match psd {
                ProcessedSystemDeploy::Succeeded { system_deploy, .. } => {
                    bytes.push(0);
                    match system_deploy {
                        SystemDeployData::Slash {
                            invalid_block_hash,
                            issuer_public_key,
                        } => {
                            bytes.push(0);
                            push_len_prefixed(&mut bytes, invalid_block_hash);
                            push_len_prefixed(&mut bytes, &issuer_public_key.bytes);
                        }
                        SystemDeployData::CloseBlockSystemDeployData => {
                            bytes.push(1);
                        }
                        SystemDeployData::Empty => {
                            bytes.push(2);
                        }
                    }
                }
                ProcessedSystemDeploy::Failed { error_msg, .. } => {
                    bytes.push(1);
                    push_len_prefixed(&mut bytes, error_msg.as_bytes());
                }
            }
        }
        bytes.push(u8::from(is_genesis));
        Blake2b256::hash(bytes)
    }

    fn max_block_index_cache_entries() -> usize {
        static VALUE: OnceLock<usize> = OnceLock::new();
        *VALUE.get_or_init(|| {
            std::env::var(Self::MAX_BLOCK_INDEX_CACHE_ENTRIES_ENV)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(Self::MAX_BLOCK_INDEX_CACHE_ENTRIES)
        })
    }

    fn max_parents_post_state_cache_entries() -> usize {
        static VALUE: OnceLock<usize> = OnceLock::new();
        *VALUE.get_or_init(|| {
            std::env::var(Self::MAX_PARENTS_POST_STATE_CACHE_ENTRIES_ENV)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(Self::MAX_PARENTS_POST_STATE_CACHE_ENTRIES)
        })
    }

    fn max_active_validators_cache_entries() -> usize {
        static VALUE: OnceLock<usize> = OnceLock::new();
        *VALUE.get_or_init(|| {
            std::env::var(Self::MAX_ACTIVE_VALIDATORS_CACHE_ENTRIES_ENV)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(Self::MAX_ACTIVE_VALIDATORS_CACHE_ENTRIES)
        })
    }

    fn max_replay_cache_entries() -> usize {
        static VALUE: OnceLock<usize> = OnceLock::new();
        *VALUE.get_or_init(|| {
            std::env::var(Self::MAX_REPLAY_CACHE_ENTRIES_ENV)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(Self::MAX_REPLAY_CACHE_ENTRIES)
        })
    }

    fn max_replay_cache_event_log_entries() -> usize {
        static VALUE: OnceLock<usize> = OnceLock::new();
        *VALUE.get_or_init(|| {
            std::env::var(Self::MAX_REPLAY_CACHE_EVENT_LOG_ENTRIES_ENV)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(Self::MAX_REPLAY_CACHE_EVENT_LOG_ENTRIES)
        })
    }

    fn max_state_hash_cache_entries() -> usize {
        static VALUE: OnceLock<usize> = OnceLock::new();
        *VALUE.get_or_init(|| {
            std::env::var(Self::MAX_STATE_HASH_CACHE_ENTRIES_ENV)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(Self::MAX_STATE_HASH_CACHE_ENTRIES)
        })
    }

    fn maybe_trim_allocator() {
        let enabled = std::env::var("F1R3_RUNTIME_MALLOC_TRIM")
            .ok()
            .map(|v| {
                let normalized = v.trim().to_ascii_lowercase();
                normalized == "1" || normalized == "true" || normalized == "yes"
            })
            .unwrap_or(true);
        if !enabled {
            return;
        }

        #[cfg(target_os = "linux")]
        unsafe {
            unsafe extern "C" {
                fn malloc_trim(pad: usize) -> i32;
            }
            let _ = malloc_trim(0);
        }
    }

    pub fn trim_allocator() {
        Self::maybe_trim_allocator();
    }

    fn touch_cache_key<K>(order: &Mutex<VecDeque<K>>, key: &K)
    where
        K: Eq + Clone,
    {
        // LRU touch is O(n) due VecDeque::position/remove. This is intentional for now:
        // these caches are tightly bounded (64-256 entries by default), so linear touch
        // remains cheaper than introducing additional synchronized index maps.
        if let Ok(mut guard) = order.lock() {
            if let Some(pos) = guard.iter().position(|existing| existing == key) {
                guard.remove(pos);
            }
            guard.push_back(key.clone());
        }
    }

    fn evict_fifo_entry<K, V>(map: &DashMap<K, V>, order: &Mutex<VecDeque<K>>)
    where
        K: Eq + Hash + Clone,
    {
        if let Ok(mut guard) = order.lock() {
            while let Some(evict_key) = guard.pop_front() {
                if map.remove(&evict_key).is_some() {
                    break;
                }
            }
        }
    }

    pub async fn spawn_runtime(&self) -> RhoRuntimeImpl {
        let new_space = self.space.spawn().expect("Failed to spawn RSpace");
        let runtime = rho_runtime::create_rho_runtime(
            new_space,
            self.mergeable_tag_name.clone(),
            false, // Registry already in state from genesis — do not re-initialize per block
            &mut Vec::new(),
            self.external_services.clone(),
        )
        .await;

        runtime
    }

    /// Spawns a runtime whose RSpace is positioned at the given state root.
    ///
    /// This avoids the bug where `spawn()` creates a child from the parent's
    /// (potentially empty/stale) root, followed by a `reset()` that updates
    /// the history repository but leaves the hot store's history reader stale.
    /// By creating the child directly at the target state, the history reader
    /// and hot store are consistent from the start.
    pub async fn spawn_runtime_at(&self, hash: &StateHash) -> RhoRuntimeImpl {
        let root = Blake2b256Hash::from_bytes_prost(hash);
        tracing::info!(
            target: "f1r3fly.rholang.diag",
            state_hash = %hex::encode(hash),
            "spawn_runtime_at: spawning child RSpace at target state hash"
        );
        let new_space = self
            .space
            .spawn_at(&root)
            .expect("Failed to spawn RSpace at state hash");
        tracing::info!(
            target: "f1r3fly.rholang.diag",
            state_hash = %hex::encode(hash),
            "spawn_runtime_at: spawn_at succeeded, creating runtime (init_registry=false)"
        );
        let runtime = rho_runtime::create_rho_runtime(
            new_space,
            self.mergeable_tag_name.clone(),
            false, // State already has registry — do not re-initialize
            &mut Vec::new(),
            self.external_services.clone(),
        )
        .await;
        tracing::info!(
            target: "f1r3fly.rholang.diag",
            state_hash = %hex::encode(hash),
            "spawn_runtime_at: create_rho_runtime completed (init_registry=false)"
        );
        runtime
    }

    pub async fn spawn_replay_runtime(&self) -> RhoRuntimeImpl {
        let new_replay_space = self
            .replay_space
            .spawn()
            .expect("Failed to spawn ReplayRSpace");

        let runtime = rho_runtime::create_replay_rho_runtime(
            new_replay_space,
            self.mergeable_tag_name.clone(),
            false, // State already has registry — do not re-initialize during replay
            &mut Vec::new(),
            self.external_services.clone(),
        )
        .await;

        runtime
    }

    pub async fn compute_state(
        &mut self,
        start_hash: &StateHash,
        terms: Vec<Signed<DeployData>>,
        system_deploys: Vec<super::system_deploy_enum::SystemDeployEnum>,
        block_data: BlockData,
        invalid_blocks: Option<HashMap<BlockHash, Validator>>,
    ) -> Result<(StateHash, Vec<ProcessedDeploy>, Vec<ProcessedSystemDeploy>), CasperError> {
        let invalid_blocks = invalid_blocks.unwrap_or_default();
        let runtime = self.spawn_runtime().await;
        let mut runtime_ops = RuntimeOps::new(runtime);

        // Block data used for mergeable key
        let sender = block_data.sender.clone();
        let seq_num = block_data.seq_num;

        let (state_hash, usr_deploy_res, sys_deploy_res) = runtime_ops
            .compute_state(
                start_hash,
                terms,
                system_deploys,
                block_data,
                invalid_blocks,
            )
            .await?;

        let (usr_processed, usr_mergeable): (Vec<ProcessedDeploy>, Vec<NumberChannelsEndVal>) =
            usr_deploy_res.into_iter().unzip();
        let (sys_processed, sys_mergeable): (
            Vec<ProcessedSystemDeploy>,
            Vec<NumberChannelsEndVal>,
        ) = sys_deploy_res.into_iter().unzip();
        let replay_cache_event_log_cap = Self::max_replay_cache_event_log_entries();

        // Concat user and system deploys mergeable channel maps
        let mergeable_chs = usr_mergeable
            .into_iter()
            .chain(sys_mergeable.into_iter())
            .collect();

        // Convert from final to diff values and persist mergeable (number) channels for post-state hash
        let pre_state_hash = Blake2b256Hash::from_bytes_prost(&start_hash);
        let post_state_hash = Blake2b256Hash::from_bytes_prost(&state_hash);

        // Save mergeable channels to store
        self.save_mergeable_channels(
            post_state_hash,
            sender.bytes.clone(),
            seq_num,
            mergeable_chs,
            &pre_state_hash,
        )?;

        // Cache replay result for potential replay shortcut (including event logs)
        if let Some(ref cache) = self.replay_cache {
            let all_logs = Self::collect_replay_logs(&usr_processed, &sys_processed);
            let replay_payload_hash =
                Self::replay_payload_hash(&usr_processed, &sys_processed, false);

            if !all_logs.is_empty() && all_logs.len() <= replay_cache_event_log_cap {
                let key = ReplayCacheKey::new(
                    start_hash.clone(),
                    sender.bytes.to_vec(),
                    seq_num as i64,
                    replay_payload_hash,
                );
                let entry = ReplayCacheEntry::new(all_logs, state_hash.clone());
                cache.put(key, entry);
                tracing::debug!(
                    "[CACHE] Stored replay cache entry for sender seq={}",
                    seq_num
                );
            } else if !all_logs.is_empty() {
                tracing::debug!(
                    "[CACHE] Skipped replay cache store for sender seq={} (event_log={})",
                    seq_num,
                    all_logs.len()
                );
            }
        }

        // Cache state hash mapping for skip-replay optimization
        if let Some(ref cache) = self.state_hash_cache {
            cache.put(start_hash.clone(), state_hash.clone());
            tracing::debug!("[CACHE] Stored state hash mapping");
        }

        Ok((state_hash, usr_processed, sys_processed))
    }

    pub async fn compute_state_with_bonds(
        &mut self,
        start_hash: &StateHash,
        terms: Vec<Signed<DeployData>>,
        system_deploys: Vec<super::system_deploy_enum::SystemDeployEnum>,
        block_data: BlockData,
        invalid_blocks: Option<HashMap<BlockHash, Validator>>,
    ) -> Result<
        (
            StateHash,
            Vec<ProcessedDeploy>,
            Vec<ProcessedSystemDeploy>,
            Vec<Bond>,
        ),
        CasperError,
    > {
        let mem_profile_enabled = crate::rust::util::rholang::mem_profiler::mem_profile_enabled();
        let read_vm_rss_kb = || -> Option<usize> {
            crate::rust::util::rholang::mem_profiler::read_vm_rss_kb()
        };
        let mut rss_baseline = if mem_profile_enabled {
            read_vm_rss_kb()
        } else {
            None
        };
        let mut rss_prev = rss_baseline;
        let mut log_mem_step = |step: &str| {
            if !mem_profile_enabled {
                return;
            }
            if let Some(curr) = read_vm_rss_kb() {
                let prev = rss_prev.unwrap_or(curr);
                let baseline = rss_baseline.unwrap_or(curr);
                eprintln!(
                    "compute_state_with_bonds.mem step={} rss_kb={} delta_prev_kb={} delta_total_kb={}",
                    step,
                    curr,
                    curr as i64 - prev as i64,
                    curr as i64 - baseline as i64
                );
                rss_prev = Some(curr);
                if rss_baseline.is_none() {
                    rss_baseline = Some(curr);
                }
            }
        };
        log_mem_step("start");

        let invalid_blocks = invalid_blocks.unwrap_or_default();
        let runtime = self.spawn_runtime().await;
        log_mem_step("after_spawn_runtime");
        let mut runtime_ops = RuntimeOps::new(runtime);

        // Block data used for mergeable key
        let sender = block_data.sender.clone();
        let seq_num = block_data.seq_num;

        let (state_hash, usr_deploy_res, sys_deploy_res) = runtime_ops
            .compute_state(
                start_hash,
                terms,
                system_deploys,
                block_data,
                invalid_blocks,
            )
            .await?;
        log_mem_step("after_compute_state");

        let (usr_processed, usr_mergeable): (Vec<ProcessedDeploy>, Vec<NumberChannelsEndVal>) =
            usr_deploy_res.into_iter().unzip();
        let (sys_processed, sys_mergeable): (
            Vec<ProcessedSystemDeploy>,
            Vec<NumberChannelsEndVal>,
        ) = sys_deploy_res.into_iter().unzip();
        let replay_cache_event_log_cap = Self::max_replay_cache_event_log_entries();

        // Concat user and system deploys mergeable channel maps
        let mergeable_chs = usr_mergeable
            .into_iter()
            .chain(sys_mergeable.into_iter())
            .collect();

        // Convert from final to diff values and persist mergeable (number) channels for post-state hash
        let pre_state_hash = Blake2b256Hash::from_bytes_prost(start_hash);
        let post_state_hash = Blake2b256Hash::from_bytes_prost(&state_hash);

        // Save mergeable channels to store
        self.save_mergeable_channels(
            post_state_hash,
            sender.bytes.clone(),
            seq_num,
            mergeable_chs,
            &pre_state_hash,
        )?;
        log_mem_step("after_save_mergeable_channels");

        // Cache replay result for potential replay shortcut (including event logs)
        if let Some(ref cache) = self.replay_cache {
            let all_logs = Self::collect_replay_logs(&usr_processed, &sys_processed);
            let replay_payload_hash =
                Self::replay_payload_hash(&usr_processed, &sys_processed, false);

            if !all_logs.is_empty() && all_logs.len() <= replay_cache_event_log_cap {
                let key = ReplayCacheKey::new(
                    start_hash.clone(),
                    sender.bytes.to_vec(),
                    seq_num as i64,
                    replay_payload_hash,
                );
                let entry = ReplayCacheEntry::new(all_logs, state_hash.clone());
                cache.put(key, entry);
                tracing::debug!(
                    "[CACHE] Stored replay cache entry for sender seq={}",
                    seq_num
                );
            } else if !all_logs.is_empty() {
                tracing::debug!(
                    "[CACHE] Skipped replay cache store for sender seq={} (event_log={})",
                    seq_num,
                    all_logs.len()
                );
            }
        }

        // Cache state hash mapping for skip-replay optimization
        if let Some(ref cache) = self.state_hash_cache {
            cache.put(start_hash.clone(), state_hash.clone());
            tracing::debug!("[CACHE] Stored state hash mapping");
        }
        log_mem_step("after_cache_updates");

        // Reuse the same spawned runtime for bonds query to avoid a second runtime init.
        let bonds = runtime_ops.compute_bonds(&state_hash).await?;
        log_mem_step("after_compute_bonds");
        drop(runtime_ops);
        log_mem_step("after_drop_runtime_ops");

        Ok((state_hash, usr_processed, sys_processed, bonds))
    }

    pub async fn compute_genesis(
        &mut self,
        terms: Vec<Signed<DeployData>>,
        block_time: i64,
        block_number: i64,
    ) -> Result<(StateHash, StateHash, Vec<ProcessedDeploy>), CasperError> {
        let runtime = self.spawn_runtime().await;
        let mut runtime_ops = RuntimeOps::new(runtime);

        let (pre_state, state_hash, processed) = runtime_ops
            .compute_genesis(terms, block_time, block_number)
            .await?;
        let (processed_deploys, mergeable_chs) = processed.into_iter().unzip();

        // Convert from final to diff values and persist mergeable (number) channels for post-state hash
        let pre_state_hash = Blake2b256Hash::from_bytes_prost(&pre_state);
        let post_state_hash = Blake2b256Hash::from_bytes_prost(&state_hash);

        // Save mergeable channels to store
        self.save_mergeable_channels(
            post_state_hash,
            prost::bytes::Bytes::new(),
            0,
            mergeable_chs,
            &pre_state_hash,
        )?;

        // Store the genesis pre-state hash for compute_parents_post_state
        self.empty_state_hash = Some(pre_state.clone());

        Ok((pre_state, state_hash, processed_deploys))
    }

    pub async fn replay_compute_state(
        &mut self,
        start_hash: &StateHash,
        terms: Vec<ProcessedDeploy>,
        system_deploys: Vec<ProcessedSystemDeploy>,
        block_data: &BlockData,
        invalid_blocks: Option<HashMap<BlockHash, Validator>>,
        is_genesis: bool, // FIXME have a better way of knowing this. Pass the replayDeploy function maybe? - OLD
    ) -> Result<StateHash, CasperError> {
        let sender = block_data.sender.clone();
        let seq_num = block_data.seq_num;
        let replay_payload_hash = Self::replay_payload_hash(&terms, &system_deploys, is_genesis);

        // Step 1: Check state-hash cache.
        //
        // IMPORTANT:
        // `StateHashCache` is keyed only by pre-state, while mergeable channels are keyed by
        // (post-state, creator, seq-num). Returning early here can skip writing mergeable data
        // for a distinct block that shares the same pre-state, which later breaks
        // parent-post-state/index reconstruction with "Missing mergeable entry ...".
        //
        // We only fast-return on cache hit if mergeable entry already exists for this block key.
        // For empty blocks we can safely synthesize and persist an empty mergeable entry.
        // Otherwise, fall through to full replay to materialize mergeable data.
        if let Some(ref cache) = self.state_hash_cache {
            if let Some(cached_post) = cache.get(start_hash) {
                let mergeable_key = MergeableKey {
                    state_hash: StateHashSerde(cached_post.clone()),
                    creator: sender.bytes.clone(),
                    seq_num,
                };
                let mergeable_key_encoded = bincode::serialize(&mergeable_key).map_err(|e| {
                    CasperError::KvStoreError(KvStoreError::SerializationError(e.to_string()))
                })?;

                if self
                    .mergeable_store
                    .contains_key(mergeable_key_encoded.clone())?
                {
                    tracing::info!(
                        "[CACHE] StateHashCache hit: mergeable entry present, skipping full replay"
                    );
                    return Ok(cached_post);
                }

                let no_user_deploys = terms.is_empty();
                let no_system_deploys = system_deploys.is_empty();
                if no_user_deploys && no_system_deploys {
                    if cached_post != *start_hash {
                        tracing::warn!(
                            "[CACHE] StateHashCache hit mismatch for empty block (seq={}): pre_state != cached_post, forcing full replay",
                            seq_num
                        );
                        // Continue to full replay path for validation.
                    } else {
                    let pre_state_hash = Blake2b256Hash::from_bytes_prost(start_hash);
                    let post_state_hash = Blake2b256Hash::from_bytes_prost(&cached_post);
                    self.save_mergeable_channels(
                        post_state_hash,
                        sender.bytes.clone(),
                        seq_num,
                        Vec::new(),
                        &pre_state_hash,
                    )?;
                    tracing::warn!(
                        "[CACHE] StateHashCache hit without mergeable entry for empty block (seq={}); synthesized empty mergeable metadata",
                        seq_num
                    );
                    return Ok(cached_post);
                    }
                }

                tracing::warn!(
                    "[CACHE] StateHashCache hit without mergeable entry for seq={}; falling back to full replay",
                    seq_num
                );
            }
        }

        // Step 2: Check replay cache (deterministic replay delta)
        let replay_cache_key = ReplayCacheKey::new(
            start_hash.clone(),
            sender.bytes.to_vec(),
            seq_num as i64,
            replay_payload_hash,
        );
        if let Some(ref cache) = self.replay_cache {
            if let Some(entry) = cache.get(&replay_cache_key) {
                tracing::info!("[CACHE] ReplayCache hit for sender seq={}", seq_num);

                // Rig the replay runtime with cached event log
                let replay_runtime = self.spawn_replay_runtime().await;
                let rspace_events: Vec<_> = entry
                    .event_log
                    .iter()
                    .map(crate::rust::util::event_converter::to_rspace_event)
                    .collect();
                replay_runtime.rig(rspace_events)?;

                return Ok(entry.post_state);
            }
        }

        // Step 3: Full replay (cache miss)
        let invalid_blocks = invalid_blocks.unwrap_or_default();
        let replay_runtime = self.spawn_replay_runtime().await;
        let runtime_ops = RuntimeOps::new(replay_runtime);
        let mut replay_runtime_ops = ReplayRuntimeOps::new(runtime_ops);

        let (state_hash, mergeable_chs) = replay_runtime_ops
            .replay_compute_state(
                start_hash,
                terms,
                system_deploys,
                block_data,
                Some(invalid_blocks),
                is_genesis,
            )
            .await?;

        // Convert from final to diff values and persist mergeable (number) channels for post-state hash
        let pre_state_hash = Blake2b256Hash::from_bytes_prost(&start_hash);
        let post_state = state_hash.to_bytes_prost();

        self.save_mergeable_channels(
            state_hash.clone(),
            sender.bytes,
            seq_num,
            mergeable_chs,
            &pre_state_hash,
        )
        .unwrap_or_else(|e| panic!("Failed to save mergeable channels: {:?}", e));

        // Cache the result for future replays
        if let Some(ref cache) = self.state_hash_cache {
            cache.put(start_hash.clone(), post_state.clone());
        }

        Ok(post_state)
    }

    pub async fn capture_results(
        &self,
        start: &StateHash,
        deploy: &Signed<DeployData>,
    ) -> Result<Vec<Par>, CasperError> {
        let runtime = self.spawn_runtime_at(start).await;
        let mut runtime_ops = RuntimeOps::new(runtime);
        let computed = runtime_ops.capture_results(start, deploy).await?;
        Ok(computed)
    }

    pub async fn get_active_validators(
        &self,
        start_hash: &StateHash,
    ) -> Result<Vec<Validator>, CasperError> {
        if let Some(cached) = self.active_validators_cache.get(start_hash) {
            Self::touch_cache_key(&self.active_validators_cache_order, start_hash);
            return Ok(cached.clone());
        }

        let runtime = self.spawn_runtime_at(start_hash).await;
        let mut runtime_ops = RuntimeOps::new(runtime);
        let computed = runtime_ops.get_active_validators(start_hash).await?;

        let max_entries = Self::max_active_validators_cache_entries();
        if self.active_validators_cache.len() >= max_entries {
            Self::evict_fifo_entry(&self.active_validators_cache, &self.active_validators_cache_order);
        }
        self.active_validators_cache
            .insert(start_hash.clone(), computed.clone());
        Self::touch_cache_key(&self.active_validators_cache_order, start_hash);

        Ok(computed)
    }

    pub async fn compute_bonds(&self, hash: &StateHash) -> Result<Vec<Bond>, CasperError> {
        let runtime = self.spawn_runtime_at(hash).await;
        let mut runtime_ops = RuntimeOps::new(runtime);
        let computed = runtime_ops.compute_bonds(hash).await?;
        Ok(computed)
    }

    // Executes deploy as user deploy with immediate rollback
    pub async fn play_exploratory_deploy(
        &self,
        term: String,
        hash: &StateHash,
    ) -> Result<Vec<Par>, CasperError> {
        tracing::info!(
            target: "f1r3fly.rholang.diag",
            state_hash = %hex::encode(hash),
            term_len = term.len(),
            "play_exploratory_deploy: starting — spawning runtime at state hash"
        );
        let runtime = self.spawn_runtime_at(hash).await;
        tracing::info!(
            target: "f1r3fly.rholang.diag",
            state_hash = %hex::encode(hash),
            "play_exploratory_deploy: child runtime created, invoking play_exploratory_deploy on RuntimeOps"
        );
        let mut runtime_ops = RuntimeOps::new(runtime);
        let computed = runtime_ops.play_exploratory_deploy(term, hash).await?;
        tracing::info!(
            target: "f1r3fly.rholang.diag",
            state_hash = %hex::encode(hash),
            result_count = computed.len(),
            "play_exploratory_deploy: completed — returned {} result pars",
            computed.len()
        );
        Ok(computed)
    }

    pub async fn get_data(&self, hash: StateHash, channel: &Par) -> Result<Vec<Par>, CasperError> {
        let runtime = self.spawn_runtime_at(&hash).await;
        let runtime_ops = RuntimeOps::new(runtime);
        let computed = runtime_ops.get_data_par(channel);
        Ok(computed)
    }

    pub async fn get_continuation(
        &self,
        hash: StateHash,
        channels: Vec<Par>,
    ) -> Result<Vec<(Vec<BindPattern>, Par)>, CasperError> {
        let runtime = self.spawn_runtime_at(&hash).await;
        let runtime_ops = RuntimeOps::new(runtime);
        let computed = runtime_ops.get_continuation_par(channels);
        Ok(computed)
    }

    pub fn get_history_repo(&self) -> RhoHistoryRepository {
        self.history_repo.clone()
    }

    /// Get or compute BlockIndex with caching
    pub fn get_or_compute_block_index(
        &self,
        block_hash: &BlockHash,
        usr_processed_deploys: &Vec<ProcessedDeploy>,
        sys_processed_deploys: &Vec<ProcessedSystemDeploy>,
        pre_state_hash: &Blake2b256Hash,
        post_state_hash: &Blake2b256Hash,
        mergeable_chs: &Vec<NumberChannelsDiff>,
    ) -> Result<BlockIndex, CasperError> {
        if let Some(cached) = self.block_index_cache.get(block_hash) {
            Self::touch_cache_key(&self.block_index_cache_order, block_hash);
            metrics::gauge!(BLOCK_INDEX_CACHE_SIZE_METRIC, "source" => CASPER_METRICS_SOURCE)
                .set(self.block_index_cache.len() as f64);
            return Ok(cached.clone());
        }

        // Cache miss - compute the BlockIndex.
        let block_index = crate::rust::merging::block_index::new(
            block_hash,
            usr_processed_deploys,
            sys_processed_deploys,
            pre_state_hash,
            post_state_hash,
            &self.history_repo,
            mergeable_chs,
        )?;

        // Keep index cache bounded for long-running validators.
        // Avoid DashMap re-entrant calls while holding an entry guard.
        let max_entries = Self::max_block_index_cache_entries();
        if self.block_index_cache.len() >= max_entries {
            Self::evict_fifo_entry(&self.block_index_cache, &self.block_index_cache_order);
        }

        self.block_index_cache
            .insert(block_hash.clone(), block_index.clone());
        Self::touch_cache_key(&self.block_index_cache_order, block_hash);
        metrics::gauge!(BLOCK_INDEX_CACHE_SIZE_METRIC, "source" => CASPER_METRICS_SOURCE)
            .set(self.block_index_cache.len() as f64);
        Ok(block_index)
    }

    /// Remove BlockIndex from cache (used during finalization)
    pub fn remove_block_index_cache(&self, block_hash: &BlockHash) {
        self.block_index_cache.remove(block_hash);
        metrics::gauge!(BLOCK_INDEX_CACHE_SIZE_METRIC, "source" => CASPER_METRICS_SOURCE)
            .set(self.block_index_cache.len() as f64);
    }

    pub fn get_cached_parents_post_state(
        &self,
        key: &ParentsPostStateCacheKey,
    ) -> Option<ParentsPostStateCacheVal> {
        let result = self.parents_post_state_cache.get(key).map(|entry| {
            Self::touch_cache_key(&self.parents_post_state_cache_order, key);
            entry.value().clone()
        });
        metrics::gauge!(PARENTS_POST_STATE_CACHE_SIZE_METRIC, "source" => CASPER_METRICS_SOURCE)
            .set(self.parents_post_state_cache.len() as f64);
        result
    }

    pub fn put_cached_parents_post_state(
        &self,
        key: ParentsPostStateCacheKey,
        value: ParentsPostStateCacheVal,
    ) {
        // Keep cache bounded with simple eviction strategy.
        let max_entries = Self::max_parents_post_state_cache_entries();
        if self.parents_post_state_cache.len() >= max_entries {
            Self::evict_fifo_entry(
                &self.parents_post_state_cache,
                &self.parents_post_state_cache_order,
            );
        }
        self.parents_post_state_cache.insert(key.clone(), value);
        Self::touch_cache_key(&self.parents_post_state_cache_order, &key);
        metrics::gauge!(PARENTS_POST_STATE_CACHE_SIZE_METRIC, "source" => CASPER_METRICS_SOURCE)
            .set(self.parents_post_state_cache.len() as f64);
    }

    /**
     * Load mergeable channels from store
     */
    pub fn load_mergeable_channels(
        &self,
        state_hash_bs: &StateHash,
        creator: prost::bytes::Bytes,
        seq_num: i32,
    ) -> Result<Vec<NumberChannelsDiff>, CasperError> {
        let state_hash = Blake2b256Hash::from_bytes_prost(state_hash_bs);
        let mergeable_key = MergeableKey {
            state_hash: StateHashSerde(state_hash.to_bytes_prost()),
            creator: creator.clone(),
            seq_num,
        };

        let get_key =
            bincode::serialize(&mergeable_key).expect("Failed to serialize mergeable key");

        let res = self.mergeable_store.get_one(&get_key)?;

        match res {
            Some(res) => {
                let res_map = res
                    .into_iter()
                    .map(|x| {
                        x.channels
                            .into_iter()
                            .map(|y| (y.hash, y.diff))
                            .collect::<BTreeMap<_, _>>()
                    })
                    .collect::<Vec<_>>();
                Ok(res_map)
            }
            None => {
                let msg = format!(
                    "Missing mergeable entry for state {} (creator={}, seq={})",
                    state_hash.bytes().encode_hex::<String>(),
                    creator.encode_hex::<String>(),
                    seq_num
                );
                tracing::error!("{}", msg);
                Err(CasperError::KvStoreError(KvStoreError::KeyNotFound(msg)))
            }
        }
    }

    /// Delete mergeable channels entry keyed by (post-state-hash, creator, seq-num).
    /// Returns `true` if the entry existed prior to deletion.
    pub fn delete_mergeable_channels(
        &self,
        state_hash_bs: &StateHash,
        creator: prost::bytes::Bytes,
        seq_num: i32,
    ) -> Result<bool, CasperError> {
        let mergeable_key = MergeableKey {
            state_hash: StateHashSerde(state_hash_bs.clone()),
            creator,
            seq_num,
        };
        let encoded_key =
            bincode::serialize(&mergeable_key).expect("Failed to serialize mergeable key");
        let existed = self.mergeable_store.contains_key(encoded_key.clone())?;
        if existed {
            self.mergeable_store.delete(vec![encoded_key])?;
        }
        Ok(existed)
    }

    /**
     * Converts final mergeable (number) channel values and save to mergeable store.
     *
     * Tuple (postStateHash, creator, seqNum) is used as a key, preStateHash is used to
     * read initial value to get the difference.
     */
    fn save_mergeable_channels(
        &mut self,
        post_state_hash: Blake2b256Hash,
        creator: prost::bytes::Bytes,
        seq_num: i32,
        channels_data: Vec<NumberChannelsEndVal>,
        // Used to calculate value difference from final values
        pre_state_hash: &Blake2b256Hash,
    ) -> Result<(), CasperError> {
        // Calculate difference values from final values on number channels
        let diffs = self.convert_number_channels_to_diff(channels_data, pre_state_hash)?;

        // Convert to storage types
        let deploy_channels = diffs
            .into_iter()
            .map(|data| {
                let channels: Vec<NumberChannel> = data
                    .into_iter()
                    .map(|(hash, diff)| NumberChannel { hash, diff })
                    .collect::<Vec<_>>();

                DeployMergeableData { channels }
            })
            .collect();

        // Key is composed from post-state hash and block creator with seq number
        let mergeable_key = MergeableKey {
            state_hash: StateHashSerde(post_state_hash.to_bytes_prost()),
            creator,
            seq_num,
        };

        let key_encoded = bincode::serialize(&mergeable_key).map_err(|e| {
            CasperError::KvStoreError(KvStoreError::SerializationError(e.to_string()))
        })?;

        // Save to mergeable channels store
        self.mergeable_store.put_one(key_encoded, deploy_channels)?;

        Ok(())
    }

    /**
     * Converts number channels final values to difference values. Excludes channels without an initial value.
     *
     * @param channelsData Final values
     * @param preStateHash Inital state
     * @return Map with values as difference on number channel
     */
    pub fn convert_number_channels_to_diff(
        &self,
        channels_data: Vec<NumberChannelsEndVal>,
        // Used to calculate value difference from final values
        pre_state_hash: &Blake2b256Hash,
    ) -> Result<Vec<NumberChannelsDiff>, CasperError> {
        let history_repo = self.history_repo.clone();
        let reader = history_repo.get_history_reader(pre_state_hash).map_err(|e| {
            CasperError::RuntimeError(format!(
                "Failed to get history reader for pre-state hash: {:?}",
                e
            ))
        })?;

        // Build a one-shot base-value map to avoid repeatedly creating history readers per key.
        let unique_channels = channels_data
            .iter()
            .flat_map(|m| m.keys().cloned())
            .collect::<std::collections::BTreeSet<_>>();
        let mut initial_values: BTreeMap<Blake2b256Hash, i64> = BTreeMap::new();
        for ch in unique_channels {
            let data = reader.get_data(&ch).map_err(|e| {
                CasperError::RuntimeError(format!("Error getting data for channel {:?}: {:?}", ch, e))
            })?;
            if data.len() > 1 {
                return Err(CasperError::RuntimeError(format!(
                    "Expected at most one value for number channel {:?}, found {}",
                    ch,
                    data.len()
                )));
            }
            let value = data
                .first()
                .map(|datum| RholangMergingLogic::get_number_with_rnd(&datum.a).0)
                .unwrap_or(0);
            initial_values.insert(ch, value);
        }

        // Calculate difference values from final values on number channels
        Ok(RholangMergingLogic::calculate_num_channel_diff(
            channels_data,
            move |ch| initial_values.get(ch).copied(),
        ))
    }

    /// Returns the genesis pre-state hash (state after bootstrap_registry).
    /// Set during compute_genesis(). Panics if called before genesis is computed.
    pub fn empty_state_hash(&self) -> StateHash {
        self.empty_state_hash
            .clone()
            .expect("empty_state_hash not initialized; compute_genesis must be called first")
    }

    pub fn create_with_space(
        rspace: RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
        replay_rspace: ReplayRSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
        history_repo: RhoHistoryRepository,
        mergeable_store: MergeableStore,
        mergeable_tag_name: Par,
        external_services: ExternalServices,
    ) -> RuntimeManager {
        let replay_cache_size = Self::max_replay_cache_entries();
        let state_hash_cache_size = Self::max_state_hash_cache_entries();

        RuntimeManager {
            space: rspace,
            replay_space: replay_rspace,
            history_repo,
            mergeable_store,
            mergeable_tag_name,
            block_index_cache: Arc::new(DashMap::new()),
            block_index_cache_order: Arc::new(Mutex::new(VecDeque::new())),
            active_validators_cache: Arc::new(DashMap::new()),
            active_validators_cache_order: Arc::new(Mutex::new(VecDeque::new())),
            parents_post_state_cache: Arc::new(DashMap::new()),
            parents_post_state_cache_order: Arc::new(Mutex::new(VecDeque::new())),
            replay_cache: (replay_cache_size > 0)
                .then(|| Arc::new(InMemoryReplayCache::new(replay_cache_size))),
            state_hash_cache: (state_hash_cache_size > 0)
                .then(|| Arc::new(StateHashCache::new(state_hash_cache_size))),
            external_services,
            empty_state_hash: None,
        }
    }

    pub fn create_with_store(
        store: RSpaceStore,
        mergeable_store: MergeableStore,
        mergeable_tag_name: Par,
        external_services: ExternalServices,
    ) -> RuntimeManager {
        let (rt_manager, _) = Self::create_with_history(
            store,
            mergeable_store,
            mergeable_tag_name,
            external_services,
        );
        rt_manager
    }

    pub fn create_with_history(
        store: RSpaceStore,
        mergeable_store: MergeableStore,
        mergeable_tag_name: Par,
        external_services: ExternalServices,
    ) -> (RuntimeManager, RhoHistoryRepository) {
        let (rspace, replay_rspace) =
            RSpace::create_with_replay(store, Arc::new(Box::new(Matcher)))
                .expect("Failed to create RSpaceWithReplay");

        let history_repo = rspace.history_repository.clone();

        let runtime_manager = RuntimeManager::create_with_space(
            rspace,
            replay_rspace,
            history_repo.clone(),
            mergeable_store,
            mergeable_tag_name,
            external_services,
        );

        (runtime_manager, history_repo)
    }

    /**
     * Creates connection to [[MergeableStore]] database.
     *
     * Mergeable (number) channels store is used in [[RuntimeManager]] implementation.
     * This function provides default instantiation.
     */
    pub async fn mergeable_store(
        kvm: &mut dyn KeyValueStoreManager,
    ) -> Result<MergeableStore, KvStoreError> {
        let store = kvm.store("mergeable-channel-cache".to_string()).await?;

        Ok(KeyValueTypedStoreImpl::new(store))
    }
}
