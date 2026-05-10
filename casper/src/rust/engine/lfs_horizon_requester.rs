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

/// Closure type for "is this root already in the joiner's roots store?".
/// Decoupled from `RuntimeManager` so the orchestrator can be unit-tested
/// against a fake roots store without spinning up the runtime stack.
pub type HasRootFn = Arc<dyn Fn(&Blake2b256Hash) -> Result<bool, CasperError> + Send + Sync>;

struct HorizonStreamProcessor<T: HorizonRequesterOps> {
    request_ops: T,
    has_root: HasRootFn,
    state_importer: Arc<dyn RSpaceImporter>,
    st: Arc<Mutex<ST<StatePartPath>>>,
    /// Maps each chunk path to the SET of roots paginating through it.
    /// Initial paths `[(root, None)]` map to `{root}`. When two roots'
    /// pagination chains converge on a shared cursor (radix tries derived
    /// from sibling chain states routinely share suffix paths), the cursor
    /// maps to BOTH roots; one wire request satisfies both, and the
    /// terminal completion fires `set_root` for each.
    path_to_root: Arc<Mutex<HashMap<StatePartPath, HashSet<Blake2b256Hash>>>>,
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
            tracing::debug!("LFS forward-horizon: ignoring chunk for unknown/stale path");
            return Ok(());
        }

        // Find which roots this chunk's pagination chain belongs to.
        // Multiple roots may share the same cursor — the chunk's content
        // satisfies all of them in one shot.
        let roots: Vec<Blake2b256Hash> = {
            let map = self.path_to_root.lock().expect("path_to_root lock");
            match map.get(&start_path) {
                Some(set) => set.iter().cloned().collect(),
                None => Vec::new(),
            }
        };
        if roots.is_empty() {
            tracing::warn!(
                "LFS forward-horizon: received chunk for path with no root mapping; ignoring"
            );
            return Ok(());
        }

        // Apply items to local rspace. Radix nodes are content-addressed so
        // the importer can validate keys against contents internally.
        let history_count = history_items.len();
        let data_count = data_items.len();
        let _ = self.state_importer.set_history_items(history_items);
        let _ = self.state_importer.set_data_items(data_items);

        // Update per-root progress for every root sharing this chunk.
        {
            let mut progress_map = self.root_progress.lock().expect("root_progress lock");
            for root in &roots {
                let entry = progress_map.entry(root.clone()).or_default();
                entry.chunk_count += 1;
                entry.total_history += history_count;
                entry.total_data += data_count;
            }
        }

        let is_terminal = last_path == start_path;

        if is_terminal {
            // Byzantine signal: terminal cursor on first chunk + no data
            // means the peer doesn't have this root. Fail loud if ANY root
            // sharing this terminal saw only an empty first chunk — that's
            // still an unrecoverable signal even if other roots happened to
            // have legitimate data on a prior path.
            {
                let progress_map = self.root_progress.lock().expect("root_progress lock");
                for root in &roots {
                    let progress = progress_map.get(root).cloned().unwrap_or_default();
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
                }
            }

            // Record the root tag in roots_store and verify the import
            // reconstructed each expected root.
            for root in &roots {
                self.state_importer.set_root(root);
                let now_have = (self.has_root)(root)?;
                if !now_have {
                    return Err(CasperError::RuntimeError(format!(
                        "LFS forward-horizon: root {} not in store after import; \
                         peer shipped invalid data",
                        root
                    )));
                }
                let progress = {
                    let progress_map = self.root_progress.lock().expect("root_progress lock");
                    progress_map.get(root).cloned().unwrap_or_default()
                };
                tracing::debug!(
                    "LFS forward-horizon: completed root {} ({} chunks, {} history, {} data)",
                    root,
                    progress.chunk_count,
                    progress.total_history,
                    progress.total_data
                );
            }

            // Mark the path Done and free its multi-root mapping. ST
            // tracks paths, not roots, so a single done() retires the
            // shared chunk for all associated roots.
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
            // Non-terminal: continuation cursor inherits the full root set.
            // If `last_path` is already tracked (another root's chunk
            // already converged on the same cursor), merge into the
            // existing set so its terminal completion fires set_root for
            // every root, not just the first writer.
            {
                let mut map = self.path_to_root.lock().expect("path_to_root lock");
                let entry = map.entry(last_path.clone()).or_default();
                for root in &roots {
                    entry.insert(root.clone());
                }
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
    has_root: HasRootFn,
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
        if has_root(&root)? {
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
    // Each initial path uniquely identifies its root, so the singleton
    // sets here will only ever grow when pagination cursors converge.
    let mut initial_paths: Vec<StatePartPath> = Vec::with_capacity(filtered.len());
    let mut path_to_root_init: HashMap<StatePartPath, HashSet<Blake2b256Hash>> = HashMap::new();
    for root in &filtered {
        let path: StatePartPath = vec![(root.clone(), None)];
        let mut set = HashSet::new();
        set.insert(root.clone());
        path_to_root_init.insert(path.clone(), set);
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
        has_root,
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
    use super::*;

    // ── ST<Key> state machine ────────────────────────────────────────────
    //
    // Mirrors the `Init → Requested → Received → Done` lifecycle that the
    // streaming orchestrator drives every horizon root through. These tests
    // pin the algebraic semantics so future changes to the orchestrator or
    // the `lfs_tuple_space_requester` parallel structure can't silently
    // diverge from the contract the stream loop assumes.

    #[test]
    fn st_new_initializes_all_entries_as_init() {
        let st: ST<i32> = ST::new(vec![1, 2, 3]);
        assert_eq!(st.len(), 3);
        assert_eq!(st.done_count(), 0);
        assert!(!st.is_finished());
    }

    #[test]
    fn st_add_inserts_new_keys_as_init_and_leaves_existing() {
        // Existing key in Requested state must NOT be reset to Init by add().
        let st: ST<i32> = ST::new(vec![1]);
        let (st, _) = st.get_next(false); // 1 -> Requested
        let mut new_keys = HashSet::new();
        new_keys.insert(2);
        new_keys.insert(1); // existing
        let st = st.add(new_keys);
        // 2 should be Init now; 1 should still be Requested → only 2 picked.
        let (_, requested) = st.get_next(false);
        assert_eq!(requested, vec![2]);
    }

    #[test]
    fn st_get_next_without_resend_picks_only_init() {
        let st: ST<i32> = ST::new(vec![1, 2]);
        let (st, batch) = st.get_next(false);
        let mut sorted = batch.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2]);
        // Calling again without resend yields nothing: both are Requested.
        let (_, batch2) = st.get_next(false);
        assert!(batch2.is_empty());
    }

    #[test]
    fn st_get_next_with_resend_picks_init_and_requested() {
        let st: ST<i32> = ST::new(vec![1]);
        let (st, _) = st.get_next(false); // 1 -> Requested
        let (_, batch) = st.get_next(true); // resend: pick Requested again
        assert_eq!(batch, vec![1]);
    }

    #[test]
    fn st_received_valid_only_on_requested_or_init() {
        let st: ST<i32> = ST::new(vec![1, 2]);
        // Init: received() accepts (handles fast in-memory test scenarios
        // where responses arrive before the request loop runs).
        let (st, valid) = st.received(1);
        assert!(valid);
        // Requested: received() accepts.
        let (st, _) = st.get_next(false); // 2 -> Requested
        let (st, valid) = st.received(2);
        assert!(valid);
        // Already-Received: received() rejects (idempotent).
        let (_, valid) = st.received(1);
        assert!(!valid);
    }

    #[test]
    fn st_done_only_valid_on_received() {
        let st: ST<i32> = ST::new(vec![1]);
        // Init -> done() is a no-op (key NOT advanced).
        let st = st.done(1);
        assert!(!st.is_finished());
        // After received() -> done() advances.
        let (st, _) = st.received(1);
        let st = st.done(1);
        assert!(st.is_finished());
    }

    #[test]
    fn st_is_finished_only_when_all_done() {
        let st: ST<i32> = ST::new(vec![1, 2]);
        let (st, _) = st.received(1);
        let st = st.done(1);
        assert!(!st.is_finished()); // 2 still Init
        let (st, _) = st.received(2);
        let st = st.done(2);
        assert!(st.is_finished());
    }

    #[test]
    fn st_done_count_matches_done_entries() {
        let st: ST<i32> = ST::new(vec![1, 2, 3]);
        let (st, _) = st.received(1);
        let st = st.done(1);
        let (st, _) = st.received(2);
        let st = st.done(2);
        assert_eq!(st.done_count(), 2);
        assert_eq!(st.len() - st.done_count(), 1);
    }

    // ── stream() behavioral tests with mock HorizonRequesterOps ─────────
    //
    // These tests construct the orchestrator with a mock network layer so
    // we can assert on:
    //   - skip-already-present pre-filter (mock has_root returns true)
    //   - terminal cursor handling (single-chunk happy path)
    //   - byzantine peer detection (terminal+empty on first chunk)
    //   - pagination (non-terminal cursor enqueues continuation)
    //   - parallel fan-out (mock counts max simultaneous in-flight requests)
    //
    // The mock RSpaceImporter is a no-op since the orchestrator only uses
    // it as a passive sink for items we feed via canned responses.

    use futures::StreamExt;
    use rspace_plus_plus::rspace::shared::trie_importer::TrieImporter;
    use rspace_plus_plus::rspace::state::rspace_importer::RSpaceImporter;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use tokio::sync::mpsc as test_mpsc;

    /// No-op RSpaceImporter — orchestrator just calls set_history_items /
    /// set_data_items / set_root with the bytes we feed it; tests don't
    /// inspect what was imported, only what the stream did with the calls.
    struct NoopImporter;

    impl TrieImporter for NoopImporter {
        fn set_history_items(&self, _data: Vec<(Blake2b256Hash, Vec<u8>)>) -> () {}
        fn set_data_items(&self, _data: Vec<(Blake2b256Hash, Vec<u8>)>) -> () {}
        fn set_root(&self, _key: &Blake2b256Hash) -> () {}
    }

    impl RSpaceImporter for NoopImporter {
        fn get_history_item(&self, _hash: Blake2b256Hash) -> Option<Vec<u8>> {
            None
        }
    }

    /// Recording RSpaceImporter — captures every root passed to `set_root`.
    /// Pair with a `has_root` closure that reads from the same set so the
    /// orchestrator's post-import verification reflects what was recorded.
    struct RecordingImporter {
        recorded: Arc<Mutex<HashSet<Blake2b256Hash>>>,
    }

    impl RecordingImporter {
        fn new() -> (Self, Arc<Mutex<HashSet<Blake2b256Hash>>>) {
            let recorded = Arc::new(Mutex::new(HashSet::new()));
            (
                Self {
                    recorded: recorded.clone(),
                },
                recorded,
            )
        }
    }

    impl TrieImporter for RecordingImporter {
        fn set_history_items(&self, _data: Vec<(Blake2b256Hash, Vec<u8>)>) -> () {}
        fn set_data_items(&self, _data: Vec<(Blake2b256Hash, Vec<u8>)>) -> () {}
        fn set_root(&self, key: &Blake2b256Hash) -> () {
            self.recorded.lock().unwrap().insert(key.clone());
        }
    }

    impl RSpaceImporter for RecordingImporter {
        fn get_history_item(&self, _hash: Blake2b256Hash) -> Option<Vec<u8>> {
            None
        }
    }

    /// Mock HorizonRequesterOps: records every send and tracks max
    /// simultaneous in-flight requests. Each send increments the in-flight
    /// counter and returns immediately; the test loop is responsible for
    /// feeding canned responses and decrementing. For the parallel-fan-out
    /// test we observe `max_in_flight` AFTER request_next has dispatched
    /// the batch but BEFORE responses arrive.
    struct MockOps {
        sends: Arc<Mutex<Vec<StatePartPath>>>,
        in_flight: Arc<AtomicUsize>,
        max_in_flight: Arc<AtomicUsize>,
    }

    impl MockOps {
        fn new() -> Self {
            Self {
                sends: Arc::new(Mutex::new(Vec::new())),
                in_flight: Arc::new(AtomicUsize::new(0)),
                max_in_flight: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn handles(&self) -> (Arc<Mutex<Vec<StatePartPath>>>, Arc<AtomicUsize>) {
            (self.sends.clone(), self.max_in_flight.clone())
        }
    }

    #[async_trait]
    impl HorizonRequesterOps for MockOps {
        async fn request_for_horizon_chunk(
            &self,
            path: &StatePartPath,
            _page_size: i32,
        ) -> Result<(), CasperError> {
            self.sends.lock().unwrap().push(path.clone());
            let now = self.in_flight.fetch_add(1, AtomicOrdering::SeqCst) + 1;
            self.max_in_flight.fetch_max(now, AtomicOrdering::SeqCst);
            // Simulate the request being "outstanding" — the test loop
            // decrements via a shared counter when it feeds a response.
            // We don't decrement here because the response handler
            // is what unblocks the next request cycle.
            Ok(())
        }
    }

    fn hash_for(seed: u8) -> Blake2b256Hash {
        let mut bytes = vec![0u8; 32];
        bytes[0] = seed;
        Blake2b256Hash::from_bytes(bytes)
    }

    fn make_response(
        start_path: StatePartPath,
        last_path: StatePartPath,
        history_item: Option<Blake2b256Hash>,
    ) -> StoreItemsMessage {
        let history_items = history_item
            .map(|h| vec![(h, prost::bytes::Bytes::from_static(b"x"))])
            .unwrap_or_default();
        StoreItemsMessage {
            start_path,
            last_path,
            history_items,
            data_items: vec![],
        }
    }

    /// Helper to drain an `impl Stream` to its final `ST`.
    async fn drain<S>(stream: S) -> Option<ST<StatePartPath>>
    where
        S: Stream<Item = ST<StatePartPath>>,
    {
        let mut stream = Box::pin(stream);
        let mut last = None;
        while let Some(st) = stream.next().await {
            last = Some(st);
        }
        last
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_skips_already_present_roots() {
        // has_root returns true for all roots → filtered set is empty.
        // Stream yields a finished ST and exits without any request.
        let r1 = hash_for(1);
        let r2 = hash_for(2);
        let has_root: HasRootFn = Arc::new(|_| Ok(true));
        let importer: Arc<dyn RSpaceImporter> = Arc::new(NoopImporter);
        let ops = MockOps::new();
        let (sends, _) = ops.handles();
        let (_tx, rx) = test_mpsc::channel::<StoreItemsMessage>(2);

        let stream = stream(
            vec![r1, r2],
            has_root,
            importer,
            ops,
            rx,
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        let final_st = drain(stream).await.expect("stream yields final state");
        assert!(final_st.is_finished());
        assert_eq!(final_st.len(), 0);
        assert!(
            sends.lock().unwrap().is_empty(),
            "no requests should have been sent for already-present roots"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_completes_on_terminal_cursor_with_data() {
        // Single root, single chunk. Response is terminal (last_path ==
        // start_path) and contains one history item → root marked Done.
        let root = hash_for(7);
        let importer: Arc<dyn RSpaceImporter> = Arc::new(NoopImporter);
        // has_root flips: false on the pre-filter call (so the root gets
        // added to ST), then true on the post-import verification (so the
        // import is accepted as having reconstructed the root).
        let post_import_has_root: HasRootFn = {
            let r = root.clone();
            let imported = Arc::new(Mutex::new(false));
            Arc::new(move |q| {
                if *q == r && *imported.lock().unwrap() {
                    Ok(true)
                } else {
                    let was = *imported.lock().unwrap();
                    *imported.lock().unwrap() = true;
                    Ok(was)
                }
            })
        };
        let ops = MockOps::new();
        let (tx, rx) = test_mpsc::channel::<StoreItemsMessage>(4);

        // Pre-feed the response.
        let start = vec![(root.clone(), None)];
        tx.send(make_response(
            start.clone(),
            start.clone(),
            Some(hash_for(8)),
        ))
        .await
        .unwrap();
        // Drop tx so the receiver eventually sees channel-closed if loop
        // doesn't terminate via is_finished.
        drop(tx);

        let stream = stream(
            vec![root],
            post_import_has_root,
            importer,
            ops,
            rx,
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        let final_st = drain(stream).await.expect("stream yields final state");
        assert!(
            final_st.is_finished(),
            "single-root happy path must mark all paths Done; final state: {:?}",
            final_st
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_detects_byzantine_peer_terminal_empty_first_chunk() {
        // Single root. Response is terminal AND empty (no history items,
        // no data items) on the FIRST chunk. This is the contract for
        // "peer doesn't have this root" — orchestrator must error out.
        let root = hash_for(5);
        let has_root: HasRootFn = Arc::new(|_| Ok(false));
        let importer: Arc<dyn RSpaceImporter> = Arc::new(NoopImporter);
        let ops = MockOps::new();
        let (tx, rx) = test_mpsc::channel::<StoreItemsMessage>(4);

        let start = vec![(root.clone(), None)];
        tx.send(make_response(start.clone(), start.clone(), None))
            .await
            .unwrap();
        drop(tx);

        let stream = stream(
            vec![root],
            has_root,
            importer,
            ops,
            rx,
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        let final_st = drain(stream).await.expect("stream yields final state");
        // Byzantine signal causes the loop to break before marking Done →
        // final ST is NOT finished. Caller's loud-fail check then triggers.
        assert!(
            !final_st.is_finished(),
            "byzantine peer (terminal+empty) must NOT mark the root Done"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_paginates_non_terminal_responses() {
        // Single root, two chunks. Response 1 is non-terminal: last_path
        // points at a continuation. Response 2 is terminal at that
        // continuation. Orchestrator must enqueue Response 2's start_path
        // (= Response 1's last_path) as a new Init entry, then fan a
        // request out for it, then mark Done on terminal arrival.
        //
        // Responses are fed by a separate task with a small yield between
        // sends so the orchestrator's `request_next` cycle can fire for the
        // continuation path before response 2 lands. Without this, the
        // biased-response select drains both responses before any request
        // ever ships and the test would assert against an empty sends log.
        let root = hash_for(9);
        let cont_path = vec![(hash_for(10), Some(0u8))];
        let post_import_has_root: HasRootFn = {
            let r = root.clone();
            let imported = Arc::new(Mutex::new(false));
            Arc::new(move |q| {
                if *q == r && *imported.lock().unwrap() {
                    Ok(true)
                } else {
                    let was = *imported.lock().unwrap();
                    *imported.lock().unwrap() = true;
                    Ok(was)
                }
            })
        };
        let importer: Arc<dyn RSpaceImporter> = Arc::new(NoopImporter);
        let ops = MockOps::new();
        let (sends, _) = ops.handles();
        let (tx, rx) = test_mpsc::channel::<StoreItemsMessage>(8);

        let start = vec![(root.clone(), None)];
        let resp1 = make_response(start.clone(), cont_path.clone(), Some(hash_for(11)));
        let resp2 = make_response(cont_path.clone(), cont_path.clone(), Some(hash_for(12)));

        let feeder_sends = sends.clone();
        let feeder_start = start.clone();
        let feeder_cont = cont_path.clone();
        let feeder = tokio::spawn(async move {
            // Wait until the initial start_path has actually been requested.
            loop {
                if feeder_sends.lock().unwrap().contains(&feeder_start) {
                    break;
                }
                tokio::task::yield_now().await;
            }
            tx.send(resp1).await.unwrap();
            // Wait for the continuation path to be requested (proves the
            // orchestrator enqueued it after processing response 1).
            loop {
                if feeder_sends.lock().unwrap().contains(&feeder_cont) {
                    break;
                }
                tokio::task::yield_now().await;
            }
            tx.send(resp2).await.unwrap();
            drop(tx);
        });

        let stream = stream(
            vec![root],
            post_import_has_root,
            importer,
            ops,
            rx,
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        let final_st = drain(stream).await.expect("stream yields final state");
        feeder.await.unwrap();
        assert!(
            final_st.is_finished(),
            "after 2 chunks (paginated) the root should be Done"
        );
        let sent_paths = sends.lock().unwrap().clone();
        assert!(sent_paths.contains(&start), "initial start_path was sent");
        assert!(
            sent_paths.contains(&cont_path),
            "continuation path (= response 1's last_path) was enqueued and sent"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stream_fans_out_in_parallel() {
        // Four roots. Don't feed any responses. Observe `max_in_flight`
        // after the request loop dispatches the initial batch. With
        // try_join_all the orchestrator dispatches all 4 requests
        // simultaneously before any response would be needed.
        let roots: Vec<Blake2b256Hash> = (1..=4u8).map(hash_for).collect();
        let has_root: HasRootFn = Arc::new(|_| Ok(false));
        let importer: Arc<dyn RSpaceImporter> = Arc::new(NoopImporter);
        let ops = MockOps::new();
        let (_, max_in_flight) = ops.handles();
        let (_tx, rx) = test_mpsc::channel::<StoreItemsMessage>(8);

        // Launch the stream and let it pump the initial request cycle.
        let stream = stream(
            roots.clone(),
            has_root,
            importer,
            ops,
            rx,
            Duration::from_secs(1),
        )
        .await
        .unwrap();
        // Pull one item (the post-request_next snapshot) which proves the
        // initial fan-out completed. We don't drain to completion because
        // there are no responses.
        let mut stream = Box::pin(stream);
        let _first = stream.next().await;

        let observed = max_in_flight.load(AtomicOrdering::SeqCst);
        assert_eq!(
            observed, 4,
            "orchestrator must dispatch all 4 root requests in parallel; \
             max_in_flight={}",
            observed
        );
    }

    /// Regression: two roots whose first chunks return the SAME continuation
    /// cursor must BOTH receive `set_root` after the shared continuation
    /// terminates. Without per-path multi-root tracking, the orchestrator's
    /// HashMap<Path, Hash> mapping silently drops all but the last writer,
    /// `set_root` fires for only one root, and the other is left with an
    /// imported trie but no roots-store tag — causing later `reset(root)` to
    /// fail with `RootRepositoryDivergence`. Radix-trie pagination shares
    /// suffix cursors across roots derived from sibling chain states, so
    /// this collision is the common case, not an edge case.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stream_completes_all_roots_when_first_chunks_share_last_path() {
        let r1 = hash_for(20);
        let r2 = hash_for(21);
        let shared_cont = vec![(hash_for(99), Some(0u8))];

        let (importer_inner, recorded) = RecordingImporter::new();
        let importer: Arc<dyn RSpaceImporter> = Arc::new(importer_inner);

        // has_root: read from the recorded set so post-import verify sees
        // what set_root just wrote. Pre-filter sees nothing → both roots
        // enter the orchestrator.
        let recorded_for_has_root = recorded.clone();
        let has_root: HasRootFn =
            Arc::new(move |q| Ok(recorded_for_has_root.lock().unwrap().contains(q)));

        let ops = MockOps::new();
        let (sends, _) = ops.handles();
        let (tx, rx) = test_mpsc::channel::<StoreItemsMessage>(8);

        let start1 = vec![(r1.clone(), None)];
        let start2 = vec![(r2.clone(), None)];

        // Both first chunks are non-terminal and end at the same cursor.
        let resp1 = make_response(start1.clone(), shared_cont.clone(), Some(hash_for(101)));
        let resp2 = make_response(start2.clone(), shared_cont.clone(), Some(hash_for(102)));
        // The shared continuation terminates with one item (proves it's not
        // an empty-byzantine signal).
        let resp3 = make_response(
            shared_cont.clone(),
            shared_cont.clone(),
            Some(hash_for(103)),
        );

        let feeder_sends = sends.clone();
        let feeder_start1 = start1.clone();
        let feeder_start2 = start2.clone();
        let feeder_cont = shared_cont.clone();
        let feeder = tokio::spawn(async move {
            // Wait until both initial start_paths have been requested before
            // feeding either response, so the orchestrator has both roots in
            // ST when it processes them.
            loop {
                let both_sent = {
                    let s = feeder_sends.lock().unwrap();
                    s.contains(&feeder_start1) && s.contains(&feeder_start2)
                };
                if both_sent {
                    break;
                }
                tokio::task::yield_now().await;
            }
            tx.send(resp1).await.unwrap();
            tx.send(resp2).await.unwrap();
            // Wait for the shared continuation to be requested.
            loop {
                if feeder_sends.lock().unwrap().contains(&feeder_cont) {
                    break;
                }
                tokio::task::yield_now().await;
            }
            tx.send(resp3).await.unwrap();
            drop(tx);
        });

        let stream = stream(
            vec![r1.clone(), r2.clone()],
            has_root,
            importer,
            ops,
            rx,
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        let final_st = drain(stream).await.expect("stream yields final state");
        feeder.await.unwrap();

        assert!(
            final_st.is_finished(),
            "orchestrator should mark all paths Done; final state: {:?}",
            final_st
        );

        let recorded = recorded.lock().unwrap();
        assert!(
            recorded.contains(&r1),
            "set_root must fire for r1 (was silently dropped by HashMap collision); recorded = {:?}",
            recorded
        );
        assert!(
            recorded.contains(&r2),
            "set_root must fire for r2; recorded = {:?}",
            recorded
        );
        assert_eq!(
            recorded.len(),
            2,
            "exactly 2 roots should have been recorded; recorded = {:?}",
            recorded
        );
    }
}
