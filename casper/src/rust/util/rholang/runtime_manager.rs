// See casper/src/main/scala/coop/rchain/casper/util/rholang/RuntimeManager.scala
// See casper/src/main/scala/coop/rchain/casper/util/rholang/RuntimeManagerSyntax.scala

use dashmap::DashMap;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crypto::rust::signatures::signed::Signed;
use hex::ToHex;
use models::rhoapi::{BindPattern, ListParWithRandom, Par, TaggedContinuation};
use models::rust::block::state_hash::{StateHash, StateHashSerde};
use models::rust::block_hash::BlockHash;
use models::rust::casper::protocol::casper_message::{
    Bond, DeployData, ProcessedDeploy, ProcessedSystemDeploy,
};
use models::rust::validator::Validator;
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
use shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl;
use shared::rust::ByteVector;

use crate::rust::errors::CasperError;
use crate::rust::merging::block_index::BlockIndex;
use crate::rust::rholang::replay_runtime::ReplayRuntimeOps;
use crate::rust::rholang::runtime::RuntimeOps;
use crate::rust::util::rholang::replay_cache::{InMemoryReplayCache, ReplayCache, ReplayCacheEntry, ReplayCacheKey};
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
    /// Optional replay cache for delta replay optimization
    pub replay_cache: Option<Arc<InMemoryReplayCache>>,
    /// Optional state hash cache for skipping known replays
    pub state_hash_cache: Option<Arc<StateHashCache>>,
}

impl RuntimeManager {
    pub async fn spawn_runtime(&self) -> RhoRuntimeImpl {
        let new_space = self.space.spawn().expect("Failed to spawn RSpace");
        let runtime = rho_runtime::create_rho_runtime(
            new_space,
            self.mergeable_tag_name.clone(),
            true,
            &mut Vec::new(),
        )
        .await;

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
            true,
            &mut Vec::new(),
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

        // Cache replay result for potential replay shortcut
        if let Some(ref cache) = self.replay_cache {
            let key = ReplayCacheKey::new(
                start_hash.clone(),
                sender.bytes.to_vec(),
                seq_num as i64,
            );
            let entry = ReplayCacheEntry::new(state_hash.clone());
            cache.put(key, entry);
            tracing::debug!(
                "[CACHE] Stored replay cache entry for sender seq={}",
                seq_num
            );
        }

        // Cache state hash mapping for skip-replay optimization
        if let Some(ref cache) = self.state_hash_cache {
            cache.put(start_hash.clone(), state_hash.clone());
            tracing::debug!("[CACHE] Stored state hash mapping");
        }

        Ok((state_hash, usr_processed, sys_processed))
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

        // Step 1: Check state-hash cache (skip full replay if known)
        if let Some(ref cache) = self.state_hash_cache {
            if let Some(cached_post) = cache.get(start_hash) {
                tracing::info!(
                    "[CACHE] StateHashCache hit: skipping full replay for start_hash"
                );
                return Ok(cached_post);
            }
        }

        // Step 2: Check replay cache (deterministic replay delta)
        let replay_cache_key = ReplayCacheKey::new(
            start_hash.clone(),
            sender.bytes.to_vec(),
            seq_num as i64,
        );
        if let Some(ref cache) = self.replay_cache {
            if let Some(entry) = cache.get(&replay_cache_key) {
                tracing::info!(
                    "[CACHE] ReplayCache hit for sender seq={}",
                    seq_num
                );
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
        let runtime = self.spawn_runtime().await;
        let mut runtime_ops = RuntimeOps::new(runtime);
        let computed = runtime_ops.capture_results(start, deploy).await?;
        Ok(computed)
    }

    pub async fn get_active_validators(
        &self,
        start_hash: &StateHash,
    ) -> Result<Vec<Validator>, CasperError> {
        let runtime = self.spawn_runtime().await;
        let mut runtime_ops = RuntimeOps::new(runtime);
        let computed = runtime_ops.get_active_validators(start_hash).await?;
        Ok(computed)
    }

    pub async fn compute_bonds(&self, hash: &StateHash) -> Result<Vec<Bond>, CasperError> {
        let runtime = self.spawn_runtime().await;
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
        let runtime = self.spawn_runtime().await;
        let mut runtime_ops = RuntimeOps::new(runtime);
        let computed = runtime_ops.play_exploratory_deploy(term, hash).await?;
        Ok(computed)
    }

    pub async fn get_data(&self, hash: StateHash, channel: &Par) -> Result<Vec<Par>, CasperError> {
        let mut runtime = self.spawn_runtime().await;

        runtime.reset(&Blake2b256Hash::from_bytes_prost(&hash));

        let runtime_ops = RuntimeOps::new(runtime);
        let computed = runtime_ops.get_data_par(channel);
        Ok(computed)
    }

    pub async fn get_continuation(
        &self,
        hash: StateHash,
        channels: Vec<Par>,
    ) -> Result<Vec<(Vec<BindPattern>, Par)>, CasperError> {
        let mut runtime = self.spawn_runtime().await;

        runtime.reset(&Blake2b256Hash::from_bytes_prost(&hash));

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
        // Use entry API to atomically handle get-or-compute pattern
        match self.block_index_cache.entry(block_hash.clone()) {
            dashmap::mapref::entry::Entry::Occupied(entry) => {
                // Cache hit - return cached value
                Ok(entry.get().clone())
            }
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                // Cache miss - compute the BlockIndex
                let block_index = crate::rust::merging::block_index::new(
                    block_hash,
                    usr_processed_deploys,
                    sys_processed_deploys,
                    pre_state_hash,
                    post_state_hash,
                    &self.history_repo,
                    mergeable_chs,
                )?;

                // Insert and return the computed value
                entry.insert(block_index.clone());
                Ok(block_index)
            }
        }
    }

    /// Remove BlockIndex from cache (used during finalization)
    pub fn remove_block_index_cache(&self, block_hash: &BlockHash) {
        self.block_index_cache.remove(block_hash);
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
            creator,
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

            None => Err(CasperError::RuntimeError(format!(
                "Mergeable store invalid state hash {}",
                state_hash.bytes().encode_hex::<String>()
            ))),
        }
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
        let diffs = self.convert_number_channels_to_diff(channels_data, pre_state_hash);

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
    ) -> Vec<NumberChannelsDiff> {
        let history_repo = self.history_repo.clone();
        let get_data_func = move |ch: &Blake2b256Hash| {
            history_repo
                .get_history_reader(pre_state_hash)
                .and_then(|reader| reader.get_data(ch))
        };

        let get_num_func = RholangMergingLogic::convert_to_read_number(get_data_func);

        // Calculate difference values from final values on number channels
        RholangMergingLogic::calculate_num_channel_diff(channels_data, get_num_func)
    }

    /**
     * This is a hard-coded value for `emptyStateHash` which is calculated by
     * [[coop.rchain.casper.rholang.RuntimeOps.emptyStateHash]].
     * Because of the value is actually the same all
     * the time. For some situations, we can just use the value directly for better performance.
     */
    pub fn empty_state_hash_fixed() -> StateHash {
        hex::decode("cb75e7f94e8eac21f95c524a07590f2583fbdaba6fb59291cf52fa16a14c784d")
            .unwrap()
            .into()
    }

    pub fn create_with_space(
        rspace: RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
        replay_rspace: ReplayRSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
        history_repo: RhoHistoryRepository,
        mergeable_store: MergeableStore,
        mergeable_tag_name: Par,
    ) -> RuntimeManager {
        RuntimeManager {
            space: rspace,
            replay_space: replay_rspace,
            history_repo,
            mergeable_store,
            mergeable_tag_name,
            block_index_cache: Arc::new(DashMap::new()),
            // Initialize caches for replay optimization
            replay_cache: Some(Arc::new(InMemoryReplayCache::default_capacity())),
            state_hash_cache: Some(Arc::new(StateHashCache::default_capacity())),
        }
    }

    pub fn create_with_store(
        store: RSpaceStore,
        mergeable_store: MergeableStore,
        mergeable_tag_name: Par,
    ) -> RuntimeManager {
        let (rt_manager, _) = Self::create_with_history(store, mergeable_store, mergeable_tag_name);
        rt_manager
    }

    pub fn create_with_history(
        store: RSpaceStore,
        mergeable_store: MergeableStore,
        mergeable_tag_name: Par,
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
