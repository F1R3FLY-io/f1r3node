> Last updated: 2026-04-29

# Crate: casper (Consensus Layer)

**Path**: `casper/`

CBC Casper consensus: block creation, validation, DAG management, safety oracle, and finalization.

## Core Traits

```rust
pub trait Casper {
    // Snapshot of current consensus state
    fn get_snapshot() -> Result<CasperSnapshot, CasperError>;
    // Block validation
    fn validate_block(block: &BlockMessage, snapshot: &CasperSnapshot)
        -> Result<BlockStatus, CasperError>;
    // Deploy management
    fn add_deploy(deploy: Signed<DeployData>) -> Result<(), CasperError>;
}

pub trait MultiParentCasper: Casper {
    // DAG operations
    fn dag() -> &BlockDagKeyValueStorage;
    // Block proposal
    fn propose(snapshot: &CasperSnapshot) -> Result<ProposeResult, CasperError>;
}

pub trait Engine: Send + Sync {
    // State machine progression
    fn handle_message(msg: CasperMessage) -> Result<(), CasperError>;
}
```

## CasperSnapshot

State captured at discrete block intervals:

```rust
pub struct CasperSnapshot {
    pub dag: KeyValueDagRepresentation,
    pub last_finalized_block: BlockHash,
    pub lca: BlockHash,
    pub tips: Vec<BlockHash>,
    pub parents: Vec<BlockMessage>,
    pub justifications: DashSet<Justification>,
    pub invalid_blocks: HashMap<BlockHash, Validator>,
    /// Signatures of deploys seen in the ancestry window above LCA.
    pub deploys_in_scope: Arc<DashSet<Bytes>>,
    /// Signatures of deploys that appeared in a merge block's
    /// `rejected_deploys` list within the ancestry window. Intersects
    /// with `deploys_in_scope` when a deploy was executed in one block
    /// and rejected during a descendant merge; the block creator uses
    /// this set to know which in-scope deploys are eligible for
    /// re-inclusion via the rejected-deploy buffer.
    pub rejected_in_scope: Arc<DashSet<Bytes>>,
    pub max_block_num: i64,
    pub max_seq_nums: DashMap<Validator, u64>,
    pub on_chain_state: OnChainCasperState,
}
```

## Block Creation Flow

1. **Deploy selection** (`prepare_user_deploys`):
   - Read unfinalized deploys from storage
   - Pull recovered deploys from the rejected-deploy buffer (sigs that
     a prior multi-parent merge conflict-rejected — their effects never
     landed in canonical state and they are eligible for re-inclusion)
   - Filter by validity window (block number, expiration timestamp)
   - Exclude deploys already in scope (prevent duplication), with one
     exception: sigs in `casper_snapshot.rejected_in_scope` are NOT
     excluded — those were conflict-rejected by a descendant merge and
     are the recovery candidates pulled from the buffer above
   - Remove expired deploys
   - Apply the same `rejected_in_scope` exemption to
     `collect_self_chain_deploy_sigs` so a recovered sig is not dropped
     just because it lives in the proposer's own prior block
   - **Adaptive deploy cap**: EMA-based controller dynamically adjusts per-block deploy count to maintain a 1-second latency target. Parameters are hardcoded (target: 1000ms, min cap: 1, small batch bypass: 3 deploys, backlog floor enabled with trigger: 2, divisor: 2, min: 2, max: 8). Small batches bypass the cap. A backlog floor mechanism prevents deploy starvation when many deploys are pending.

2. **System deploy preparation**:
   - Slashing deploys (punish equivocators)
   - Close block deploys (finalize state)

3. **Block assembly**:
   - Header: version, timestamp, sender, block number, post-state hash
   - Body: state, processed deploys, rejected deploys, system deploys
   - Justifications, bonds cache
   - Block hash via Blake2b256
   - Sign with validator's Secp256k1 key
   - **Timestamp hardening**: Guards against i64 overflow in timestamp conversion. If current time < max parent timestamp (clock skew), clamps to parent timestamp to avoid `InvalidTimestamp` validation errors.

4. **Self-validation**:
   - Pre/post-state hash mismatches return `BlockError::BlockException` instead of panicking (replaced `assert!` with proper error returns)

## Block Validation

**`BlockProcessor` pipeline**:
1. Check interest (shard match, version, not old)
2. Format and signature validation
3. Dependency resolution (fetch missing parents)
4. Full validation with CasperSnapshot:
   - Block number progression
   - Sender weight verification
   - Repeat deploy detection
   - Parent validity
   - Justification consistency
   - Bonds cache correctness
   - Equivocation checks (simple + neglected)
   - Timestamp/expiration validation
   - Deploy execution and state hash verification

**`BlockStatus`**: `Valid(ValidBlock)` or `Error(BlockError)`

**`InvalidBlock`**: 25+ variants covering equivocation, format, signature, timing, sequence, deploy validity issues.

## Fork Choice (LMD GHOST)

**Estimator** implements Latest Message Driven Greedy Heaviest Observed Subtree:
1. Calculate Lowest Common Ancestor (LCA) of all latest messages
2. Score each latest message from LCA downward
3. Rank and select non-conflicting subset with highest score
4. Constraints: `max_number_of_parents`, `max_parent_depth`

## Safety Oracle (Clique Oracle)

Computes normalized fault tolerance between -1.0 and 1.0:
1. Get weight map (stakes) from main parent
2. Filter validators agreeing on target message
3. Compute maximum clique of non-disagreeing validators
4. Formula: `(2 * maxCliqueWeight - totalStake) / totalStake`
5. If agreeing weight <= 50%, return MIN_FAULT_TOLERANCE (-1.0)

## Finalization

**Finalizer** scoped search from last finalized block (LFB) to tips:
1. Find blocks with >50% stake agreement via main parent chain
2. Execute Clique Oracle on candidates
3. Output first block exceeding fault tolerance threshold, along with its computed FT value
4. Cache the normalized FT in `BlockMetadata.fault_tolerance_value` for the directly finalized block and all indirectly finalized ancestors
5. Propagate FT to all previously-finalized blocks whose cached value is lower (`propagate_ft_to_finalized_blocks`). This covers orphaned branches in the multi-parent DAG and ensures all finalized blocks converge toward FT=1.0 as later rounds produce higher agreement.
6. Guarded by `FinalizationInProgress` atomic bool (prevents snapshot creation during finalization)

**FT caching**: The block API returns the cached FT for finalized blocks instead of recomputing via the clique oracle. Cached FT is monotonically non-decreasing — it only increases as later finalization rounds propagate higher values. Bulk endpoints (`get_blocks`, `show_main_chain`, `get_blocks_by_heights`) use a single DAG snapshot per response for internal consistency.

## Equivocation Detection

| Type | Description | Action |
|------|-------------|--------|
| Simple | Direct double-sign at same sequence number | Rejected |
| Admissible | Block conflicts but referenced via justification | Allowed |
| Ignorable | Standalone conflict | Rejected, not slashed |
| Neglected | Failure to slash known equivocation | Block invalid |

## Engine State Machine

- `Initializing` -- Waiting for approved block
- `GenesisValidator` -- Genesis ceremony
- `GenesisCeremony` -- Multi-sig approval
- `Running` -- Normal consensus

## Merging

`merging/dag_merger.rs` -- Multi-parent state merging for blocks with multiple parents.

### Parent State Merge

When a block has multiple parents (selected by the fork choice rule), the node must compute a merged post-state before executing new deploys. The merge procedure:

1. **Find the LCA** (Lowest Common Ancestor) of the parent blocks in the DAG.
2. **Determine visible blocks** -- all blocks between the LCA and the parents (exclusive of LCA, inclusive of parents).
3. **Run ConflictSetMerger** -- collects deploys from visible blocks, detects conflicts (deploys touching overlapping channels), and resolves them deterministically.

### LCA-Scoped Merge

The merge scope is limited to blocks at or above the LCA. Blocks below the LCA are common ancestors whose state is already reflected in the LCA's post-state -- replaying them would be redundant and expensive. Because the LCA is derived purely from DAG structure (parent pointers and block heights), every validator computes the same LCA for the same set of parent blocks.

### Determinism Constraint

The merge scope cannot rely on local finalization status because different validators may have temporarily different finalized views. A validator that has finalized block B and one that has not must still compute the same merge result for identical parent sets. Using block height and LCA (both derived from the immutable DAG) ensures this.

**Deterministic ordering**: Merge paths in `conflict_set_merger.rs` and casper-buffer eviction enforce deterministic tie-breaks to ensure consistent behavior across nodes.

### Mergeable Channels

Not every overlapping channel touch is a conflict. Some channels carry data with commutative update semantics — two concurrent writes can be combined rather than one rejected. The merge engine tags such channels with a `MergeType` (defined in `rspace++/src/rspace/merger/merging_logic.rs`), and the rholang interpreter detects them at evaluation time via `is_mergeable_channel` (`rholang/src/rust/interpreter/reduce.rs`):

| `MergeType` | Channel pattern | Combine rule |
|-------------|-----------------|--------------|
| `IntegerAdd` | Vault balances, gas accumulators, per-purse counters | Sum the deltas across chains |
| `BitmaskOr` | Registry `TreeHashMap` interior-node bitmaps (`@(*bitmaskTag, node, *storeToken)`) | OR-merge the bitmaps |

When the conflict-set merger inspects a shared channel, it looks up the channel's tag against this table. If a `MergeType` is found, the deploys are merged rather than treated as conflicting. If not, ordinary conflict resolution applies (one deploy is kept, the other rejected).

`BitmaskOr` was added to handle a class of failure where two registry inserts from sibling blocks both touched the same `TreeHashMap` interior node. Without bitmask merging, one of the inserts would be rejected at multi-parent merge — even though the inserts were at different keys and logically commute. The regression is captured at unit level by `casper/tests/multi_node/bridge_contract_concurrent_merge.rs`.

To diagnose a suspected merge rejection, run with `RUST_LOG=f1r3fly.merge.tag_check=trace` to see which channels match a `MergeType` and which do not.

#### Pitfalls when authoring contracts that use mergeable-tagged channels

Mergeable-tagged channels rely on a contract-maintained singleton-Datum invariant: at any observation point the channel holds zero or one Datum. Registry.rho upholds this with the lock pattern `for (@val <- @chan) { @chan!(newVal) }` — the consume removes the existing Datum, the contract publishes a fresh one. Two situations break the invariant and silently corrupt the contract's own state without breaking consensus:

- **Replicated sends (`!!`) on a mergeable-tagged channel.** A persistent Datum is not removed by the lock-acquire consume, so the contract's release `@chan!(newVal)` adds a second Datum alongside the persistent one. Each subsequent lock cycle adds another. The numeric reader's multi-value path (`get_number_channel` in `casper/src/rust/rholang/runtime.rs`) then OR-folds (or, for `IntegerAdd`, picks max of) all the Datums on every read. Determinism holds — every validator sees the same growing multi-Datum state — but the contract's effective value diverges from any single write. Use `!` (linear send) on mergeable-tagged channels.

- **Mixing the lock pattern with `<<-` peeks that don't acquire.** The `<<-` peek reads without consuming. If the contract uses `<<-` to read and then `!` to write without the linear-consume step in between, two concurrent reads can both observe the same pre-state and both publish, again leaving multiple Datums.

These are contract-author footguns, not runtime errors — the runtime can't tell intended-singleton from intended-multiset. If you build a contract on top of a mergeable-tagged channel, model the lifetime explicitly and prefer the Registry.rho lock pattern (`for (@val <- @chan) { @chan!(newVal) }`) for any read-modify-write step.

### Performance

Merge cost is O(visible_blocks^2 x deploys^2) for the conflict resolution phase, dominated by pairwise conflict detection across deploys in the visible block set. LCA scoping keeps the visible block count bounded, preventing merge cost from degrading under sustained load.

### Fallback

If the visible block count exceeds `MAX_PARENT_MERGE_SCOPE_BLOCKS` (512) or the LCA distance exceeds `MAX_LCA_DISTANCE_BLOCKS` (256), the merge falls back to the latest parent's post-state. This caps worst-case merge latency at the cost of discarding deploys from non-selected parents, which will be re-proposed in subsequent blocks.

## Deploy Replay

Deploys within a block are evaluated sequentially (matching Scala's `traverse`, not
`parTraverse`). Each deploy uses a soft checkpoint for rollback on error. The play path
records event logs; the replay path uses `ReplayRSpace` with the recorded event log as
an oracle to force the same COMMs regardless of evaluation order.

The `RuntimeManager` coordinates play (`compute_state`) and replay (`replay_compute_state`):
- Play: evaluates user deploys + system deploys, creates checkpoint, returns state hash
- Replay: rigs `ReplayRSpace` with play's event log, re-evaluates, verifies state hash match

All `RuntimeManager` methods and `RhoRuntime` methods are async, with `.await` on
ISpace operations throughout the call chain.

## Tests

Feature-gated `test_utils` module (`#[cfg(feature = "test-utils")]`) provides test infrastructure: `helper/` (test_node, block_generator, block_dag_storage_fixture, block_util, bonding_util, no_ops_casper_effect) and `util/` (genesis_builder, test_mocks, rholang/resources, comm/transport_layer_test_impl). Integration tests for block_report_api and reporting_casper.

All interpreter-level tests use `#[tokio::test(flavor = "multi_thread", worker_threads = 4)]`
to match the production multi-threaded runtime, ensuring parallel `tokio::spawn` evaluation
of Rholang Par branches is exercised during testing.

**See also:** [casper/ crate README](../../casper/README.md) | [Consensus Protocol](./CONSENSUS_PROTOCOL.md) | [Byzantine Fault Tolerance](./BYZANTINE_FAULT_TOLERANCE.md) | [Synchrony Constraint](./SYNC_CONSTRAINT.md)

[← Back to docs index](../README.md)
