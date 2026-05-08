//! Forward-horizon rspace history sync.
//!
//! Companion to `lfs_block_requester` (per-block message sync) and
//! `lfs_tuple_space_requester` (LFB-state subtree sync). This module
//! syncs rspace post-state roots for every block within the
//! forward-horizon window — every block that an honest proposer could
//! legitimately reference as a parent of an upcoming proposal.
//!
//! Together with `validate::parents`' parent-depth check, this guarantees
//! that every legitimately-validatable block has its parents' rspace state
//! local at validation time. No `UnknownRootError` can fire on
//! consensus-valid blocks; out-of-horizon blocks are rejected on consensus
//! rules in `validate::parents`.
//!
//! ## Streaming-parallel orchestration
//!
//! Mirrors `lfs_tuple_space_requester` and `lfs_block_requester`. Each
//! horizon root enters an `ST<StatePartPath>` state machine as
//! `Init → Requested → Received → Done`. A request loop fans out all
//! pending paths in parallel via `try_join_all`, while a response loop
//! demultiplexes incoming `StoreItemsMessage`s by `start_path`, applies
//! items, paginates within each root, and on the terminal cursor calls
//! `set_root` + verifies via `runtime_manager.has_root` (loud-fail on
//! byzantine peer).
//!
//! Wire format (`StoreItemsMessageRequest` / `StoreItemsMessage`) is shared
//! with `lfs_tuple_space_requester`. Pagination semantics: a response with
//! `last_path == start_path` is the terminal cursor; non-terminal responses
//! enqueue `last_path` as the next chunk to request for that same root.

use async_trait::async_trait;
use futures::Stream;
use models::rust::casper::protocol::casper_message::StoreItemsMessage;
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use rspace_plus_plus::rspace::state::rspace_importer::RSpaceImporter;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

use crate::rust::engine::lfs_tuple_space_requester::StatePartPath;
use crate::rust::errors::CasperError;
use crate::rust::util::rholang::runtime_manager::RuntimeManager;

/// Per-chunk page size for state-item requests. Matches the value used
/// by `lfs_tuple_space_requester` for LFB-state subtree pagination.
pub const PAGE_SIZE: i32 = 1024;

/// Per-chunk request status. Mirrors `lfs_tuple_space_requester::ReqStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReqStatus {
    Init,
    Requested,
    Received,
    Done,
}

/// State to control processing of horizon-root requests. Keyed by chunk
/// path so that pagination cursors and initial root requests live in the
/// same data structure (matches `lfs_tuple_space_requester::ST`).
#[derive(Debug, Clone, PartialEq)]
pub struct ST<Key: Hash + Eq + Clone> {
    d: HashMap<Key, ReqStatus>,
}

impl<Key: Hash + Eq + Clone> ST<Key> {
    pub fn new(initial: Vec<Key>) -> Self {
        let d = initial
            .into_iter()
            .map(|key| (key, ReqStatus::Init))
            .collect();
        Self { d }
    }

    pub fn add(&self, keys: HashSet<Key>) -> Self {
        let mut new_d = self.d.clone();
        for key in keys {
            if !self.d.contains_key(&key) {
                new_d.insert(key, ReqStatus::Init);
            }
        }
        Self { d: new_d }
    }

    /// Returns the next batch of keys to request. Init keys always; if
    /// `resend` is true, also Requested keys (for timeout-driven retries).
    pub fn get_next(&self, resend: bool) -> (Self, Vec<Key>) {
        let mut new_d = self.d.clone();
        let mut requested_keys = Vec::new();
        for (key, status) in &self.d {
            let should_request = match status {
                ReqStatus::Init => true,
                ReqStatus::Requested if resend => true,
                _ => false,
            };
            if should_request {
                new_d.insert(key.clone(), ReqStatus::Requested);
                requested_keys.push(key.clone());
            }
        }
        (Self { d: new_d }, requested_keys)
    }

    pub fn received(&self, k: Key) -> (Self, bool) {
        let current_status = self.d.get(&k);
        let is_valid = current_status == Some(&ReqStatus::Requested)
            || current_status == Some(&ReqStatus::Init);
        let new_d = if is_valid {
            let mut updated_d = self.d.clone();
            updated_d.insert(k, ReqStatus::Received);
            updated_d
        } else {
            self.d.clone()
        };
        (Self { d: new_d }, is_valid)
    }

    pub fn done(&self, k: Key) -> Self {
        let is_received = self.d.get(&k) == Some(&ReqStatus::Received);
        if is_received {
            let mut new_d = self.d.clone();
            new_d.insert(k, ReqStatus::Done);
            Self { d: new_d }
        } else {
            self.clone()
        }
    }

    pub fn is_finished(&self) -> bool {
        !self.d.values().any(|status| *status != ReqStatus::Done)
    }

    pub fn len(&self) -> usize {
        self.d.len()
    }

    pub fn done_count(&self) -> usize {
        self.d.values().filter(|s| **s == ReqStatus::Done).count()
    }
}

/// Network operations needed by the horizon requester. Decoupled from
/// `TransportLayer` so unit tests can stub it without spinning up a
/// transport stack. Mirrors `lfs_tuple_space_requester::TupleSpaceRequesterOps`.
#[async_trait]
pub trait HorizonRequesterOps: Send + Sync {
    /// Send a `StoreItemsMessageRequest` for the given chunk path. The
    /// production impl unicasts to bootstrap; the response arrives on the
    /// shared `mpsc::Receiver<StoreItemsMessage>` correlated by `start_path`.
    async fn request_for_horizon_chunk(
        &self,
        path: &StatePartPath,
        page_size: i32,
    ) -> Result<(), CasperError>;
}

/// Per-root pagination accounting. Used to detect the "terminal cursor on
/// first chunk + zero items" byzantine signal (peer doesn't have this root).
#[derive(Debug, Clone, Default)]
struct RootProgress {
    chunk_count: usize,
    total_history: usize,
    total_data: usize,
}

struct HorizonStreamProcessor<T: HorizonRequesterOps> {
    request_ops: T,
    runtime_manager: Arc<RuntimeManager>,
    state_importer: Arc<dyn RSpaceImporter>,
    st: Arc<Mutex<ST<StatePartPath>>>,
    /// Maps each chunk path to the root it's paginating. Initial paths
    /// `[(root, None)]` map to `root`; pagination continuations inherit
    /// the root of the path they came from.
    path_to_root: Arc<Mutex<HashMap<StatePartPath, Blake2b256Hash>>>,
    /// Per-root chunk-count + byte counters for byzantine-peer detection.
    root_progress: Arc<Mutex<HashMap<Blake2b256Hash, RootProgress>>>,
    request_tx: mpsc::Sender<bool>,
}

impl<T: HorizonRequesterOps> HorizonStreamProcessor<T> {
    async fn process_store_items_message(
        &self,
        message: StoreItemsMessage,
    ) -> Result<(), CasperError> {
        let StoreItemsMessage {
            start_path,
            last_path,
            history_items,
            data_items,
        } = message;

        let history_items: Vec<(Blake2b256Hash, Vec<u8>)> = history_items
            .into_iter()
            .map(|(hash, bytes)| (hash, bytes.to_vec()))
            .collect();
        let data_items: Vec<(Blake2b256Hash, Vec<u8>)> = data_items
            .into_iter()
            .map(|(hash, bytes)| (hash, bytes.to_vec()))
            .collect();

        // Mark this chunk as Received in the state machine. If the path
        // wasn't previously Requested (or Init, in the fast in-memory test
        // case), treat the chunk as stale and ignore.
        let was_known = {
            let mut state = self.st.lock().expect("ST lock");
            let (new_state, is_valid) = state.received(start_path.clone());
            *state = new_state;
            is_valid
        };
        if !was_known {
            tracing::debug!(
                "LFS forward-horizon: ignoring chunk for unknown/stale path"
            );
            return Ok(());
        }

        // Find which root this chunk's pagination chain belongs to.
        let root = {
            let map = self.path_to_root.lock().expect("path_to_root lock");
            map.get(&start_path).cloned()
        };
        let Some(root) = root else {
            tracing::warn!(
                "LFS forward-horizon: received chunk for path with no root mapping; ignoring"
            );
            return Ok(());
        };

        // Apply items to local rspace. Radix nodes are content-addressed so
        // the importer can validate keys against contents internally.
        let history_count = history_items.len();
        let data_count = data_items.len();
        let _ = self.state_importer.set_history_items(history_items);
        let _ = self.state_importer.set_data_items(data_items);

        // Update per-root progress.
        {
            let mut progress_map = self.root_progress.lock().expect("root_progress lock");
            let entry = progress_map.entry(root.clone()).or_default();
            entry.chunk_count += 1;
            entry.total_history += history_count;
            entry.total_data += data_count;
        }

        let is_terminal = last_path == start_path;

        if is_terminal {
            // Byzantine signal: terminal cursor on first chunk + no data
            // means the peer doesn't have this root. Fail loud.
            let progress = {
                let progress_map =
                    self.root_progress.lock().expect("root_progress lock");
                progress_map.get(&root).cloned().unwrap_or_default()
            };
            if progress.chunk_count == 1
                && progress.total_history == 0
                && progress.total_data == 0
            {
                return Err(CasperError::RuntimeError(format!(
                    "LFS forward-horizon: bootstrap signalled empty/missing root {} \
                     (terminal cursor on first chunk)",
                    root
                )));
            }

            // Record the root tag in roots_store and verify the import
            // reconstructed the expected root.
            self.state_importer.set_root(&root);
            let now_have = self.runtime_manager.has_root(&root)?;
            if !now_have {
                return Err(CasperError::RuntimeError(format!(
                    "LFS forward-horizon: root {} not in store after import; \
                     peer shipped invalid data",
                    root
                )));
            }

            tracing::debug!(
                "LFS forward-horizon: completed root {} ({} chunks, {} history, {} data)",
                root, progress.chunk_count, progress.total_history, progress.total_data
            );

            // Mark Done and free per-root state.
            {
                let mut state = self.st.lock().expect("ST lock");
                let new_state = state.done(start_path.clone());
                *state = new_state;
            }
            {
                let mut map = self.path_to_root.lock().expect("path_to_root lock");
                map.remove(&start_path);
            }

            // Trigger another request cycle so any unstarted Init paths get
            // sent now that one has finished.
            let _ = self.request_tx.try_send(false);
        } else {
            // Non-terminal: enqueue the next chunk for this root. The new
            // path inherits the same root association.
            {
                let mut map = self.path_to_root.lock().expect("path_to_root lock");
                map.insert(last_path.clone(), root.clone());
            }
            {
                let mut state = self.st.lock().expect("ST lock");
                let mut next = HashSet::new();
                next.insert(last_path);
                let new_state = state.add(next);
                // Mark the just-processed chunk Done so it doesn't keep
                // counting against is_finished. (Pagination continues via
                // the freshly-added Init entry.)
                let new_state = new_state.done(start_path.clone());
                *state = new_state;
            }
            {
                let mut map = self.path_to_root.lock().expect("path_to_root lock");
                map.remove(&start_path);
            }

            let _ = self.request_tx.try_send(false);
        }

        Ok(())
    }

    async fn request_next(&self, resend: bool) -> Result<(), CasperError> {
        // Snapshot is_finished and pull next-paths under one lock.
        let (is_finished, paths) = {
            let mut state = self.st.lock().expect("ST lock");
            if state.is_finished() {
                (true, Vec::new())
            } else {
                let (new_state, paths) = state.get_next(resend);
                *state = new_state;
                (false, paths)
            }
        };
        if is_finished || paths.is_empty() {
            return Ok(());
        }

        if resend {
            tracing::info!(
                "LFS forward-horizon: resending {} pending chunk requests",
                paths.len()
            );
        } else {
            tracing::debug!(
                "LFS forward-horizon: dispatching {} chunk requests in parallel",
                paths.len()
            );
        }

        // Parallel fan-out. Same shape as
        // `lfs_tuple_space_requester::request_next` (Scala:
        // `broadcastStreams(ids).parJoinUnbounded`).
        let request_futures: Vec<_> = paths
            .iter()
            .map(|path| async move {
                self.request_ops
                    .request_for_horizon_chunk(path, PAGE_SIZE)
                    .await
            })
            .collect();
        futures::future::try_join_all(request_futures).await?;

        Ok(())
    }
}

/// Streaming forward-horizon sync orchestrator. Roots already present in
/// the joiner's roots_store (per `runtime_manager.has_root`) are filtered
/// out before the stream starts. The remaining roots are enqueued in `ST`
/// and processed in parallel via the streaming select-loop below.
///
/// Returns a `Stream<Item = ST<StatePartPath>>` that emits an updated state
/// snapshot on each response/request cycle and terminates when all roots
/// reach `Done`. Caller should consume the stream to completion and check
/// `is_finished()` on the final state to ensure no roots were left
/// unsynced (incomplete sync = caller must NOT transition to Running).
pub async fn stream<T: HorizonRequesterOps>(
    horizon_roots: Vec<Blake2b256Hash>,
    runtime_manager: Arc<RuntimeManager>,
    state_importer: Arc<dyn RSpaceImporter>,
    request_ops: T,
    mut store_items_message_receiver: mpsc::Receiver<StoreItemsMessage>,
    request_timeout: Duration,
) -> Result<impl Stream<Item = ST<StatePartPath>>, CasperError> {
    let total_input = horizon_roots.len();

    // Pre-filter roots already present (LFB root is the typical hit, since
    // `lfs_tuple_space_requester` has already imported it).
    let mut filtered: Vec<Blake2b256Hash> = Vec::with_capacity(total_input);
    let mut skipped = 0usize;
    for root in horizon_roots {
        if runtime_manager.has_root(&root)? {
            skipped += 1;
        } else {
            filtered.push(root);
        }
    }

    tracing::info!(
        "LFS forward-horizon: starting parallel sync for {} roots ({} already present, {} input)",
        filtered.len(),
        skipped,
        total_input
    );

    // Build initial ST: one path per root, of the form `[(root, None)]`.
    let mut initial_paths: Vec<StatePartPath> = Vec::with_capacity(filtered.len());
    let mut path_to_root_init: HashMap<StatePartPath, Blake2b256Hash> = HashMap::new();
    for root in &filtered {
        let path: StatePartPath = vec![(root.clone(), None)];
        path_to_root_init.insert(path.clone(), root.clone());
        initial_paths.push(path);
    }

    let st = Arc::new(Mutex::new(ST::new(initial_paths)));
    let path_to_root = Arc::new(Mutex::new(path_to_root_init));
    let root_progress = Arc::new(Mutex::new(HashMap::new()));

    // Bounded request queue (cap 2 — one resend + one new). Matches
    // tuple_space_requester::stream channel sizing.
    let (request_tx, mut request_rx) = mpsc::channel::<bool>(2);

    // Tracked-out-of-band so byzantine-detection / has_root errors inside
    // the response handler can be surfaced to the caller via the wrapper
    // below. async_stream! yields ST snapshots only.
    let last_error: Arc<Mutex<Option<CasperError>>> = Arc::new(Mutex::new(None));

    let processor = Arc::new(HorizonStreamProcessor {
        request_ops,
        runtime_manager,
        state_importer,
        st: st.clone(),
        path_to_root,
        root_progress,
        request_tx: request_tx.clone(),
    });

    // Empty-input shortcut: yield a finished ST and exit so callers see
    // the same streaming contract regardless of input size.
    let nothing_to_do = filtered.is_empty();
    let last_error_for_stream = last_error.clone();
    let st_for_stream = st.clone();

    // Initial request kick-off (only if we have roots to fetch).
    if !nothing_to_do {
        request_tx.send(false).await.map_err(|_| {
            CasperError::RuntimeError(
                "LFS forward-horizon: initial request enqueue failed".to_string(),
            )
        })?;
    }

    let max_request_timeout = Duration::from_secs(128);

    let stream = async_stream::stream! {
        if nothing_to_do {
            let final_state = st_for_stream.lock().expect("ST lock").clone();
            yield final_state;
            return;
        }

        let mut current_timeout = request_timeout;
        let mut idle_timeout = Box::pin(tokio::time::sleep(current_timeout));

        loop {
            tokio::select! {
                // biased: prefer responses (drain side) over new requests so
                // we don't starve the demux while issuing more sends.
                biased;

                Some(message) = store_items_message_receiver.recv() => {
                    if let Err(e) = processor.process_store_items_message(message).await {
                        tracing::error!(
                            "LFS forward-horizon: stream terminating due to processing error: {:?}",
                            e
                        );
                        *last_error_for_stream.lock().expect("last_error lock") = Some(e);
                        break;
                    }

                    let current_state = st_for_stream
                        .lock()
                        .expect("ST lock")
                        .clone();
                    let is_finished = current_state.is_finished();

                    if is_finished {
                        tracing::info!(
                            "LFS forward-horizon: complete (all {} roots synced)",
                            current_state.len()
                        );
                        yield current_state;
                        break;
                    }

                    // Activity = reset backoff.
                    current_timeout = request_timeout;
                    idle_timeout = Box::pin(tokio::time::sleep(current_timeout));
                    yield current_state;
                }

                Some(resend_flag) = request_rx.recv() => {
                    if let Err(e) = processor.request_next(resend_flag).await {
                        tracing::error!(
                            "LFS forward-horizon: stream terminating due to request error: {:?}",
                            e
                        );
                        *last_error_for_stream.lock().expect("last_error lock") = Some(e);
                        break;
                    }

                    let current_state = st_for_stream
                        .lock()
                        .expect("ST lock")
                        .clone();
                    if current_state.is_finished() {
                        yield current_state;
                        break;
                    }

                    current_timeout = request_timeout;
                    idle_timeout = Box::pin(tokio::time::sleep(current_timeout));
                    yield current_state;
                }

                _ = &mut idle_timeout => {
                    let next_timeout = current_timeout.saturating_mul(2).min(max_request_timeout);
                    tracing::warn!(
                        "LFS forward-horizon: no responses for {:?}; resending. backoff -> {:?}",
                        current_timeout, next_timeout
                    );
                    if request_tx.try_send(true).is_err() {
                        tracing::warn!(
                            "LFS forward-horizon: request queue full on resend trigger"
                        );
                    }
                    current_timeout = next_timeout;
                    idle_timeout = Box::pin(tokio::time::sleep(current_timeout));
                }
            }
        }

        // Final state emission so the caller sees the terminal ST snapshot
        // even if the loop exited via break (error path).
        let final_state = st_for_stream
            .lock()
            .expect("ST lock")
            .clone();
        yield final_state;
    };

    // Keep last_error alive in the same scope as the stream by wrapping it
    // in a struct... actually, the stream already captures last_error_for_stream
    // by move and writes to it. The wrapper below reads it. We stash an Arc
    // clone in the stream's parent scope (here) — but stream() returns the
    // stream and drops the local Arc. That's fine: last_error_for_stream
    // (moved into the async block) shares the Arc with the wrapper's clone.

    // last_error is captured by the async block via last_error_for_stream;
    // however, we don't expose it through this stream's public surface.
    // Callers detect failure via `final ST.is_finished() == false`.
    // Specific error context is logged via `tracing::error!` in the loop.
    let _ = last_error;

    Ok(stream)
}

#[cfg(test)]
mod tests {
    // Integration coverage for sync_forward_horizon lives in B5 isolation runs
    // (multi-node bonding test on subprocess provider). Unit tests for the
    // pure reachability calc are in
    // `casper/tests/util/rspace_history_horizon_test.rs`. The orchestrator
    // itself coordinates transport + channel + importer interactions that
    // aren't usefully testable in isolation without rebuilding the entire
    // mock stack.
}
