> Last updated: 2026-05-14

# State Merging

Multi-parent state merging combines RSpace post-states from sibling blocks into a single base state for the next block's deploys. This document covers the merge pipeline end-to-end and the two layers of mergeability that interact at the seam.

For the higher-level consensus context, see [CONSENSUS_PROTOCOL.md §6](./CONSENSUS_PROTOCOL.md#6-state-merging-multi-parent).

## Why merging is necessary

In a multi-parent DAG, different validators can include different deploys in their concurrent blocks. A merge block with parents `P1` and `P2` needs a combined post-state that incorporates effects from both — minus anything that genuinely conflicts. Single-parent chains avoid this problem at the cost of throughput; multi-parent merging is the throughput-vs-coordination trade.

## Two layers of mergeability

The merger enforces two distinct mergeability contracts, layered:

### Layer 1 — CSP mergeability

Encoded in [`merging_logic::conflicts()`](../../rspace++/src/rspace/merger/merging_logic.rs) and [`ChannelChange::combine`](../../rspace++/src/rspace/merger/channel_change.rs). Channels are CSP-style multisets — two distinct linear sends on the same channel both survive into the merged state. This is the correct semantic for bag-style channels (e.g., a queue receiving messages from multiple producers).

The canonical reference for which event pairs commute under this layer is the pairwise event-class matrix (legend: `!`=send, `!!`=persistent send, `4`=for, `C`=contract, `P`=peek, `X`=no match). Cells marked ✓ commute as the matrix describes; cells marked X must be flagged as conflicts. Layer-1 implementation tracks this via `produces_*` / `consumes_*` sets on `EventLogIndex` and the four checks in `conflicts()`.

### Layer 2 — Number-channel single-value contract

A channel whose identity matches an entry in the runtime's `mergeable_tags` registry is bound by an additional contract: **at any observation point, the channel holds at most one Datum**. This contract is imposed *on top of* layer 1 and overrides multiset semantics for tagged channels.

Two registered tags ([`rholang/src/rust/interpreter/merging/mergeable_tags.rs`](../../rholang/src/rust/interpreter/merging/mergeable_tags.rs)):

| Tag identity | Channel pattern | Commutative merge rule |
|---|---|---|
| `NonNegativeNumber` | Vault balances, gas accumulators, per-purse counters | Sum deltas across branches (`IntegerAdd`) |
| `BitmaskOr` tag | `TreeHashMap` interior-node bitmaps (`@(*bitmaskTag, node, *storeToken)`) | Bitwise OR of bitmaps |

Tag membership is decided by **channel identity** (head Par of the tuple matches a registered tag), not by the channel's runtime value at any moment. The runtime detects identity at evaluation time via [`is_mergeable_channel`](../../rholang/src/rust/interpreter/reduce.rs).

## The contract seam

A channel can be identity-tagged (under layer 2) but not always have a commutative-merge representation in a given deploy:

- **Empty end-of-deploy state.** A deploy that consumes a Number channel without writing back leaves it empty; the runtime can't derive a single-Int diff.
- **Non-numeric end-of-deploy value.** TreeHashMap leaves are tagged with the `bitmaskTag` prefix (same identity family as the interior bitmap nodes) but store `Map` values rather than `Int`s. A `Map` value can't go through the BitmaskOr commutative-merge.
- **Multi-Datum value mid-deploy.** A bug or contract-author footgun can leave more than one Datum on a tagged channel; the runtime [sanitises]( ../../casper/src/rust/rholang/runtime.rs) at read time (OR-fold for BitmaskOr, max for IntegerAdd) and logs a `mergeable_channel.sanitize` WARN.

In all three cases the channel **remains under the single-value contract** but does not appear in that deploy's commutative-merge map (`number_channels_data`). The contract is still in force — distinct writes from sibling branches must be resolved through conflict detection, not silently combined.

## The merger pipeline (end-to-end)

```
deploy execution
   │
   │ is_mergeable_channel  (per produce/consume on each channel)
   ▼
eval_result.mergeable : HashMap<Par, MergeType>          ── all identity-tagged channels touched
   │
   │ get_number_channels_data
   ▼
MergeableChsForDeploy
   ├── commutative      : NumberChannelsEndVal           ── subset with single-Int end value
   └── identity_tagged  : HashableSet<Blake2b256Hash>    ── full set, contract membership
   │
   │ convert_number_channels_to_diff (against pre-state)
   ▼
MergeableChsForDeploy { commutative: NumberChannelsDiff, identity_tagged: ... }
   │
   │ block_index::create_event_log_index
   ▼
EventLogIndex
   ├── produces_linear / persistent / consumed / ...     ── layer-1 event sets
   ├── produces_mergeable / consumes_mergeable           ── filtered to channels in number_channels_data
   ├── number_channels_data                              ── commutative-merge diffs
   └── identity_tagged_channels                          ── contract membership (for conflicts() check #4)
   │
   │ dag_merger::merge
   ▼
resolve_conflicts                                        ── runs conflicts() pairwise, picks winners
   │
   │ compute_merged_state
   ▼
combined StateChange + all_mergeable_channels (commutative-merged across branches)
   │
   │ compute_trie_actions
   │   per channel:
   │     ├ if in all_mergeable_channels: calculate_number_channel_merge  ── 1 datum (commutative-folded)
   │     └ else:                          make_trie_action                ── multiset combine
   ▼
HotStoreTrieAction list → new state root
```

## `conflicts()` — the four checks

[`merging_logic::conflicts(a, b)`](../../rspace++/src/rspace/merger/merging_logic.rs) returns the set of channel hashes on which event logs `a` and `b` are non-mergeable. Four checks; their union is the conflict set.

### Check 1 — races for the same I/O event

Both branches' `produces_consumed` (or `consumes_produced`) intersect on the same produce (or consume). The same I/O event was destroyed by COMM in both branches — a race. Mergeable produces/consumes (those whose channel is in `number_channels_data` for both branches) are excluded: they commute by tag.

### Check 2 — potential cross-branch COMMs

A produce created and not destroyed in branch `a` matches a consume created and not destroyed in branch `b` (or vice versa). Applying both branches' diffs would trigger a COMM that neither branch saw locally, producing state that neither validator computed individually.

### Check 3 — produce touches base join

A produce in either branch lands on a channel that participates in a join at the LCA's base state. Joins read across multiple channels; applying produces from both branches could trigger continuations that neither branch saw, with the same problem as check 2 at higher arity.

### Check 4 — identity-tagged channels with non-commutative pending writes

Both branches leave a pending produce on the same identity-tagged channel (`identity_tagged_channels` intersect) where neither produce is in `produces_mergeable` — i.e., the channel lacks a commutative-merge representation in at least one branch. Without this check, the writes flow through `ChannelChange::combine` and the multiset-union semantic violates the layer-2 single-value contract.

This is the check that closes the seam. It is the structural equivalent of saying: *"the matrix (layer 1) is the formal spec for CSP commutativity; the single-value contract (layer 2) takes precedence over the matrix on identity-tagged channels."*

## What happens after a conflict is detected

[`ConflictSetMerger`](../../casper/src/rust/merging/conflict_set_merger.rs) consumes the conflict map and runs cost-optimal rejection: among the conflicting branches, find the subset to reject that minimises total cost (`Σ deploy.cost`) while removing all conflicts. The chosen survivors flow into `compute_merged_state` and contribute their diffs to the merged post-state.

Rejected deploys land in the `KeyValueRejectedDeployBuffer`. The next proposer's [`prepare_user_deploys`](../../casper/src/rust/blocks/proposer/block_creator.rs) reads from the buffer first, re-including rejected deploys with fresh signatures. So conflict-rejection is not "deploy is lost" — it is "deploy is deferred to a subsequent block where it doesn't conflict". This is the recovery path established by PR #488.

## When the dispatch falls through to the multiset path

In `compute_trie_actions`, the dispatch on each channel hash:

- **In `all_mergeable_channels`** (some branch contributed a commutative diff) → `calculate_number_channel_merge`. Produces exactly one Datum using the fold rule for the channel's `MergeType`.
- **Not in `all_mergeable_channels`** → `make_trie_action`. Multiset semantics: `init ∖ removed ∪ added`. Correct for bag-style (layer-1-only) channels.

The contract-seam bug arose when a channel was layer-2 contract-bound (identity-tagged) but absent from `all_mergeable_channels` for every branch (none had a single-Int end value). The dispatch fell through to the multiset path and silently produced a multi-Datum post-state. The next propose reading the channel via [`convert_to_read_number`](../../rholang/src/rust/interpreter/merging/rholang_merging_logic.rs) raised `Number channel ... has N pre-state values; single-value invariant violated` and the shard wedged.

Check #4 in `conflicts()` prevents this by ensuring at most one branch's contribution can ever reach `compute_trie_actions` for a contract-bound channel without a commutative representation.

## `ChannelChange::combine` — why multiset-union is correct (under layer 1)

[`ChannelChange::combine`](../../rspace++/src/rspace/merger/channel_change.rs) performs multiset union of `added` and `removed` lists. This is correct for layer 1 — two `ch!(x)` writes from sibling branches *should* both survive on a bag channel. The dev framing that "combine is a bug" applies to *layer-2 reachability*: a contract-bound channel should never reach `combine` in a state where multiset semantics could violate its contract. With check #4 in place, that's now guaranteed structurally — `combine` keeps its layer-1-correct behaviour, and the contract is enforced upstream.

## Determinism constraints

The merge result must be byte-identical across every validator that observes the same parent set. This requires:

- **Merge scope from DAG structure, not finalization.** Validators may have different finalized views; LCA + visible blocks is the DAG-derived scope ([§Merge Scope](./README.md#lca-scoped-merge)).
- **Deterministic ordering** in `conflict_set_merger.rs` and casper-buffer eviction — same conflict map produces the same chosen-rejection set across nodes.
- **Stable identity for `EventLogIndex`** — its `Ord` impl uses sorted views of internal sets (including the new `identity_tagged_channels`) for deterministic comparison regardless of underlying `HashSet` iteration order.

## Replay symmetry

Replay re-derives `MergeableChsForDeploy` from the play side's deterministic deploy execution. Both `runtime.rs::get_number_channels_data` and its replay-side counterpart compute the same `commutative` and `identity_tagged` outputs from the same `HashMap<Par, MergeType>` input. The persistence path (`save_mergeable_channels` / `load_mergeable_channels` via `DeployMergeableData`) carries both fields through the mergeable-store so replay-loaded `BlockIndex` builds get the same `identity_tagged` coverage as freshly-computed ones.

## Performance bounds

- Conflict detection: O(visible_blocks² × deploys²) for pairwise event-log comparison.
- LCA scoping keeps `visible_blocks` bounded.
- Fallback: if `visible_blocks > MAX_PARENT_MERGE_SCOPE_BLOCKS (512)` or `LCA distance > MAX_LCA_DISTANCE_BLOCKS (256)`, the merge degrades to the latest parent's post-state. Deploys from non-selected parents land in the rejected-deploy buffer and are re-included by a later proposer.

## Pitfalls when authoring contracts on mergeable-tagged channels

The single-value contract is a *contract-maintained* invariant — the runtime enforces it at read time, but contracts are responsible for upholding the singleton-Datum shape. See [casper/README.md §Pitfalls](./README.md#pitfalls-when-authoring-contracts-that-use-mergeable-tagged-channels) for the contract-author guidance (`!!` on tagged channels, `<<-` peek interleaving).

## Key source files

| File | Role |
|---|---|
| [`rspace++/src/rspace/merger/merging_logic.rs`](../../rspace++/src/rspace/merger/merging_logic.rs) | `conflicts()`, `MergeType`, `MergeableChsForDeploy`, `combine_mergeable_value` |
| [`rspace++/src/rspace/merger/event_log_index.rs`](../../rspace++/src/rspace/merger/event_log_index.rs) | `EventLogIndex` (per-deploy event + tag-membership summary) |
| [`rspace++/src/rspace/merger/channel_change.rs`](../../rspace++/src/rspace/merger/channel_change.rs) | `ChannelChange::combine` (layer-1 multiset union) |
| [`rspace++/src/rspace/merger/state_change_merger.rs`](../../rspace++/src/rspace/merger/state_change_merger.rs) | `compute_trie_actions` (per-channel dispatch) |
| [`casper/src/rust/merging/dag_merger.rs`](../../casper/src/rust/merging/dag_merger.rs) | Top-level merge entry; wires the conflict-detect + resolve + compute pipeline |
| [`casper/src/rust/merging/conflict_set_merger.rs`](../../casper/src/rust/merging/conflict_set_merger.rs) | `resolve_conflicts` (cost-optimal rejection), `compute_merged_state` |
| [`casper/src/rust/rholang/runtime.rs`](../../casper/src/rust/rholang/runtime.rs) | `get_number_channels_data` (populates `MergeableChsForDeploy` from runtime tag map) |
| [`rholang/src/rust/interpreter/merging/rholang_merging_logic.rs`](../../rholang/src/rust/interpreter/merging/rholang_merging_logic.rs) | `calculate_number_channel_merge`, `convert_to_read_number` (single-value invariant check), `DeployMergeableData` (persistence shape) |
| [`rholang/src/rust/interpreter/merging/mergeable_tags.rs`](../../rholang/src/rust/interpreter/merging/mergeable_tags.rs) | `default_mergeable_tags` registry |
| [`rholang/src/rust/interpreter/reduce.rs`](../../rholang/src/rust/interpreter/reduce.rs) | `is_mergeable_channel` (identity check at evaluation time) |

**See also:** [Consensus Protocol §6](./CONSENSUS_PROTOCOL.md#6-state-merging-multi-parent) | [casper/ README §Merging](./README.md#merging) | [rspace/ README §Merger](../rspace/README.md#merger-consensus-support)

[← Back to docs index](../README.md)
