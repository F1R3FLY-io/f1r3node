# Casper CBC Consensus Protocol

> How blocks go from proposal to finalization, and why each step exists.

## Summary

F1r3fly uses **CBC Casper** (Correct-by-Construction Casper), a proof-of-stake consensus protocol based on [Ethereum's CBC Casper research](https://github.com/ethereum/research/blob/master/papers/CasperTFG/CasperTFG.pdf).

**Properties:**
- **Multi-parent DAG** — each block can reference multiple parents (one per validator). Forks are merged, not discarded.
- **Mathematical finality** — blocks are finalized when the clique oracle proves >2/3 of stake agrees and will never disagree. Deterministic, not probabilistic.
- **Concurrent execution** — RSpace tuple space enables parallel deploy processing within blocks.

**Pipeline:**
1. Deploy arrives → pool
2. Heartbeat or deploy signal triggers proposal
3. Snapshot: select parents via LMD GHOST fork choice, compute justifications and deploy scope
4. Create block: select deploys, execute in RSpace, compute state hash, sign
5. Self-validate, broadcast to peers
6. Receivers: replay deploys, verify state hash, check equivocations
7. Valid → insert into DAG → trigger finalization
8. Finalizer: clique oracle computes fault tolerance → if FT > threshold, block finalized

**Abstraction boundaries** — what is consensus-specific vs reusable:

| Consensus-specific | Consensus-agnostic (reusable) |
|---|---|
| Fork choice — LMD GHOST (`estimator.rs`) | DAG storage (`BlockDagKeyValueStorage`) |
| Safety oracle — Clique Oracle (`clique_oracle.rs`) | Block persistence (`KeyValueBlockStore`) |
| Finalization — FT threshold (`finalizer.rs`) | Deploy pool (`KeyValueDeployStorage`) |
| Synchrony constraint (`synchrony_constraint_checker.rs`) | Contract execution (`RhoRuntime` / `ReplayRSpace`) |
| Equivocation detection (`equivocation_detector.rs`) | P2P transport (`TransportLayer`) |
| Pre-proposal constraint checks | Engine trait (`Arc<dyn MultiParentCasper>`) |
| | Block creation/assembly, signing |
| | Validation steps 1-3, 6 (format, signature, deps, replay) |

---

## Contents

- [1. Node Startup & Genesis](#1-node-startup--genesis)
- [2. Block Proposal](#2-block-proposal)
- [3. Block Propagation](#3-block-propagation)
- [4. Block Validation](#4-block-validation)
- [5. Fork Choice (LMD GHOST)](#5-fork-choice-lmd-ghost)
- [6. State Merging (Multi-Parent)](#6-state-merging-multi-parent)
- [7. Finalization (Clique Oracle)](#7-finalization-clique-oracle)
- [8. Liveness (Heartbeat Proposer)](#8-liveness-heartbeat-proposer)
- [9. Equivocation & Slashing](#9-equivocation--slashing)
- [10. Configuration](#10-configuration)
- [11. Known Limitations](#11-known-limitations)
- [Source File Map](#source-file-map)

---

## 1. Node Startup & Genesis

### Engine State Machine

The consensus engine operates through a state machine defined in `engine.rs`:

```
GenesisCeremonyMaster ──┐
                        ├──→ Initializing ──→ Running
GenesisValidator ───────┘
```

| State | Role | Transitions to |
|-------|------|----------------|
| `GenesisCeremonyMaster` | One node coordinates genesis. Collects `required_signatures` block approvals from genesis validators. | Initializing (once approved) |
| `GenesisValidator` | Other validators. Send `UnapprovedBlock`, wait for `ApprovedBlock`. | Initializing (on receipt) |
| `Initializing` | Stores approved block, creates `MultiParentCasper` instance from storage. | Running |
| `Running` | Active consensus. Handles blocks, requests, proposals. | Terminal |

### Genesis Ceremony

Genesis creates the first block containing:
- Initial validator bonds (from `bonds.txt`)
- Initial wallet balances (from `wallets.txt`)
- Shard configuration (fault tolerance threshold, synchrony constraint threshold — **locked in forever**)
- System contracts (PoS, vault, registry)

**Why this matters**: The synchrony constraint threshold and fault tolerance threshold are on-chain parameters written into the genesis block's state. Changing them requires a new genesis (new network).

### Key Design Point

The engine uses `Arc<dyn MultiParentCasper>` for dynamic dispatch. The `Running` state holds a trait object, not a concrete type.

---

## 2. Block Proposal

Proposals are triggered by the [heartbeat proposer](#8-liveness-heartbeat-proposer) or by deploy arrival (when auto-propose is enabled).

### Step 1: Acquire Snapshot

`MultiParentCasperImpl::get_snapshot()` captures the consensus state at proposal time:

1. **Get latest messages** from all bonded validators (one block per validator)
2. **Filter out** slashed/invalid validators
3. **Select parents** via [fork choice](#5-fork-choice-lmd-ghost) (LMD GHOST)
   - One parent per validator, deduplicated
   - Limited by `max_number_of_parents` and `max_parent_depth`
4. **Compute LCA** (Lowest Common Ancestor) of selected parents — bounds the [merge scope](#6-state-merging-multi-parent)
5. **Build justifications**: Each bonded validator's latest message hash
6. **Compute deploy scope**: BFS traversal within `deploy_lifespan` window to find all deploys already included in ancestor blocks

**Why snapshot first?** The snapshot is immutable once created. This prevents race conditions — the proposal works against a consistent view of the DAG even as new blocks arrive concurrently.

**Guard**: If finalization is in progress (`finalization_in_progress` flag), snapshot creation fails. This prevents proposing against a stale view that the finalizer is about to advance.

### Step 2: Check Constraints

Before building a block, the proposer verifies:

| Constraint | What it checks | Why |
|------------|---------------|-----|
| Active validator | Sender is in bonded validator set with non-zero stake | Only bonded validators can propose |
| Synchrony constraint | Other validators have produced recent blocks (see [section 8](#8-liveness-heartbeat-proposer)) | Prevents isolation attacks |
| Height constraint | Block height < LFB height + threshold | Prevents runaway chain growth |

### Step 3: Select Deploys

`block_creator::prepare_user_deploys()`:

1. Read unfinalized deploys from `KeyValueDeployStorage`
2. **Filter**: Not future (`valid_after_block_number`), not expired by height, not expired by time
3. **Exclude**: Deploys already in scope (prevents re-inclusion across branches)
4. **Sort deterministically**: `(valid_after_block_number, timestamp, signature)` — every validator selects the same deploys in the same order
5. **Cap**: `max_user_deploys_per_block`
6. **Adaptive cap**: EMA-based controller targets 1-second block creation latency. When blocks take longer, cap decreases. Small batches bypass the cap entirely. A backlog floor prevents deploy starvation.

**Why deterministic ordering?** All validators must select identical deploys for identical parent sets, or state hashes diverge and blocks get rejected.

### Step 4: Execute Deploys

For each selected deploy:
1. Execute Rholang via `RhoRuntime` (play runtime)
2. RSpace produce/consume operations with phlogiston cost metering
3. `create_soft_checkpoint()` between deploys (isolates effects)

Then execute system deploys:
- `SlashDeploy`: Penalize known equivocators
- `CloseBlockDeploy`: Finalize block state, update bonds

Finally: `create_checkpoint()` produces the post-state hash.

### Step 5: Assemble & Sign

- Header: version, timestamp, sender, block number (max parent + 1), sequence number
- Body: pre-state hash, post-state hash, processed deploys, rejected deploys, system deploys
- Justifications: One per bonded validator
- Bonds cache: Current validator set with stakes
- Hash the block via Blake2b256
- Sign with validator's Secp256k1 key

**Timestamp hardening**: If current time < max parent timestamp (clock skew), the timestamp is clamped to the parent's timestamp. This prevents `InvalidTimestamp` validation errors.

### Step 6: Self-Validate

The proposer validates its own block before broadcasting. If pre/post-state hashes don't match expectations, the block is rejected as `BlockException` (not panicked).

---

## 3. Block Propagation

1. Proposer broadcasts `BlockHashMessage` to all connected peers
2. Peers that don't have the block send `BlockRequest`
3. Proposer streams the full `BlockMessage` to requesting peer
4. Receiving peer enters the [validation pipeline](#4-block-validation)

The block retriever (`block_retriever.rs`) handles missing dependencies:
- Tracks pending requests per block hash
- Implements retry budgets, cooldowns, and quarantine for stuck requests
- Deduplicates requests to avoid flooding

---

## 4. Block Validation

`BlockProcessor` runs an 8-step pipeline on received blocks:

### Step 1: Interest Check
- Already in DAG or casper buffer? Skip.
- Shard ID and version match the approved block? Required.
- Block number >= approved block number? Required (no ancient blocks).

### Step 2: Format & Signature
- Verify cryptographic signature (Secp256k1, Schnorr, or FROST)
- Check field validity: hash length, timestamp within ±15s of local time, required fields present

### Step 3: Dependency Resolution
- All parent blocks must be in the DAG
- If missing: store block in **casper buffer** (max ~16K entries), request missing parents from peers
- Casper buffer tracks retry attempts per dependency and quarantines blocks after budget exhaustion

### Step 4: Snapshot Computation
- Recompute `CasperSnapshot` with the block's actual parents as tips
- This ensures validation uses the same state the proposer had

### Step 5: Block Summary
- Structural consistency: block number progression, sender weight, justification format

### Step 6: Checkpoint Validation (Deploy Replay)
- **Replay every deploy** via `ReplayRSpace` (replay runtime, not play runtime)
- Verify computed post-state hash matches the block's claimed hash
- This is the most expensive step — it proves the proposer executed correctly

### Step 7: Equivocation Checks
- **Simple equivocation**: Block creator's latest message should match their creator justification. If not:
  - `AdmissibleEquivocation`: Block was requested as a dependency — store as invalid but keep in DAG for tracking
  - `IgnorableEquivocation`: Block arrived unsolicited — drop entirely
- **Neglected equivocation**: Block justifies a known equivocator without slashing them — block is invalid

### Step 8: Deploy & State Validation
- Deploys are within scope, not duplicated
- Phlogiston price meets minimum
- Invalid block tracking applied

### On Success
- Block inserted into DAG
- Latest messages updated for block's sender
- Children index updated
- **Finalization triggered** asynchronously (single-flight guard)

---

## 5. Fork Choice (LMD GHOST)

**Latest Message Driven Greedy Heaviest Observed Subtree** — the algorithm that selects which blocks to build on.

### Algorithm (`estimator.rs`)

1. **Collect latest messages**: One block per bonded validator
2. **Filter**: Remove messages from slashed validators; ignore messages >1000 blocks old
3. **Compute LCA**: Lowest Common Ancestor of all latest messages (iterative LUCA-many algorithm)
4. **Score**: BFS from each latest message up to LCA. Each validator's stake weight flows down through the main-parent chain.
5. **Rank recursively**: Starting from LCA, greedily pick the highest-scored child. Repeat until no higher-scored descendants exist.
6. **Apply depth filter**: Main parent (rank 1) always included. Secondary parents filtered to within `max_parent_depth` of main parent.

### Why LMD GHOST?

- Selects by **weight** (stake), not longest chain — a validator with 51% stake immediately wins fork choice
- **History-independent** after the LCA — only recent messages matter
- Supports **multi-parent** selection — the ranked list becomes the parent set for the next block

---

## 6. State Merging (Multi-Parent)

When a block has multiple parents, their RSpace states must be merged before executing new deploys. This is the key difference from single-parent chains.

### Why Merge?

In a multi-parent DAG, different validators may have included different deploys in their blocks. Block B with parents P1 and P2 needs a combined state that includes effects from both P1 and P2 (minus conflicts).

### Algorithm (`dag_merger.rs` + `conflict_set_merger.rs`)

1. **Identify visible blocks**: All blocks between the LCA and the parents (exclusive of LCA, inclusive of parents)
2. **Collect deploys**: Extract user deploys from all visible blocks
3. **Detect conflicts**: Branches conflict if they contain the **same user deploy ID** (not content — just the deploy signature)
4. **Resolve**: `ConflictSetMerger` selects the highest-value subset of non-conflicting deploys. Dependents of rejected deploys are also rejected.
5. **Merge**: Replay selected deploys via RSpace merger to compute combined post-state

### Determinism Constraint

The merge scope is derived entirely from DAG structure (parent pointers and block heights), not from local finalization state. Two validators with different finalization views must compute the same merge result for identical parent sets. This is why the LCA (not the LFB) bounds the scope.

### Performance Bounds

- Merge cost: O(visible_blocks^2 x deploys^2) for conflict resolution
- LCA scoping keeps visible_blocks bounded
- **Fallback**: If visible_blocks > 512 or LCA distance > 256, falls back to latest parent's post-state (discards non-selected parent deploys — they'll be re-proposed)

### System Deploys

System deploys (`SlashDeploy`, `CloseBlockDeploy`) are deterministic and non-conflicting. They are not subject to conflict resolution.

---

## 7. Finalization (Clique Oracle)

Finalization determines when a block is **mathematically irreversible**. Once finalized, the block's state is committed and deploy effects are permanent.

### Trigger

Finalization runs asynchronously after each valid block is added to the DAG. A single-flight guard (`finalizer_task_in_progress`) prevents concurrent runs. A `finalization_in_progress` flag prevents snapshot creation during finalization (avoids stale proposals).

### Algorithm (`finalizer.rs`)

1. **Scope**: BFS from latest messages down the main-parent chain to the current LFB (Last Finalized Block)
2. **Candidate filtering**: Only blocks with >50% stake agreement (quick filter before expensive clique computation)
3. **Clique Oracle** for each candidate:
   - Build an agreement graph: edge between validators A and B if they "never eventually see disagreement" about the target block
   - Find the maximum weighted clique (largest fully-connected subgraph by stake)
   - Compute fault tolerance: `FT = (2 * clique_weight - total_stake) / total_stake`
4. **Finalization**: If `FT > fault_tolerance_threshold`, block is finalized
5. **LFB advancement**: Update last finalized block, clean up deploy storage, emit `BlockFinalised` event

### "Never Eventually See Disagreement"

Two validators A and B agree on block T if:
- A's latest message is in T's main-parent chain
- B's latest message is in T's main-parent chain
- Walking B's self-justification chain from B's latest back to A's view of B reveals no messages that disagree with T

This is a **permanent** agreement — once two validators are in a clique for block T, no future messages can break it.

### Work Budgets

The finalizer operates under time budgets to avoid blocking the proposer. Cooperative yield: every 8 iterations, yield 1ms to avoid starving other tasks.

### Fault Tolerance Values

| FT Value | Meaning | Finalized at FTT=0.0? | FTT=0.33? |
|----------|---------|----------------------|-----------|
| 1.0 | All stake agrees | Yes | Yes |
| 0.67 | 5/6 of stake | Yes | Yes |
| 0.33 | 2/3 of stake | Yes | No (strict >) |
| 0.0 | Exactly 50% | No | No |
| -1.0 | No majority | No | No |

---

## 8. Liveness (Heartbeat Proposer)

The heartbeat proposer (`node/src/rust/instances/heartbeat_proposer.rs`) ensures the chain makes progress even without user deploys.

### Trigger Logic

The heartbeat runs a loop that races between:
- **Timer**: `check_interval` (default: 5s)
- **Signal**: Deploy received (wakes immediately)

### Decision Tree

On each heartbeat tick:

1. **Frontier chase**: Is my latest block behind the DAG tip? → Propose (catch up)
2. **Pending deploys**: Are there unfinalized deploys AND LFB lag exceeds threshold? → Propose
3. **Stale LFB recovery**: Is LFB older than `max_lfb_age` AND regular recovery is throttled? → Leader-only proposal (deterministic leader selection prevents N competing recovery blocks)
4. **Self-propose cooldown**: Don't propose more often than the configured cooldown

### Why Heartbeat Matters

Without heartbeat, a shard with no user deploys would never advance finalization. The heartbeat produces empty blocks that allow validators to build justifications and the clique oracle to detect agreement. It also recovers from synchrony constraint deadlocks via the stale-LFB bypass.

### Synchrony Recovery

When the synchrony constraint blocks proposals:
1. Detect stall: no progress for the configured stall window
2. Allow bypass after cooldown
3. Limited bypass budget before requiring another stall window
4. Alternative: finalized-baseline mode uses LFB height instead of tip height

---

## 9. Equivocation & Slashing

### Equivocation Types

| Type | What happened | Detection | Action |
|------|--------------|-----------|--------|
| Simple | Validator created two blocks at same sequence number | Creator justification != latest message | Block rejected |
| Admissible | Equivocating block needed as dependency by another block | Same as simple, but block is in dependency chain | Stored as invalid in DAG for tracking |
| Ignorable | Equivocating block arrived unsolicited | Same as simple, not needed as dependency | Dropped entirely |
| Neglected | Validator had evidence of equivocation but didn't slash | Justifications reference known equivocator | Block rejected, validator penalized |

### Slashing Flow

1. Equivocation detected during block validation
2. `EquivocationRecord` created and stored persistently
3. Next block from any honest validator includes `SlashDeploy` system deploy
4. `SlashDeploy` executes PoS contract to remove equivocator from bonds
5. Equivocator loses entire stake

### Two-Level Slashing

- **Level 1**: Direct equivocator — loses entire stake
- **Level 2**: Validator that neglected to report equivocation — also loses stake

This makes collusion economically irrational: both parties get slashed.

---

## 10. Configuration

All consensus parameters are defined in HOCON configuration files:

- **Built-in defaults**: [`node/src/main/resources/defaults.conf`](../../node/src/main/resources/defaults.conf) — every available option and its default value
- **Shard override**: [`docker/conf/default.conf`](../../docker/conf/default.conf) — minimal overrides for multi-validator shard
- **Standalone override**: [`docker/conf/standalone-dev.conf`](../../docker/conf/standalone-dev.conf) — standalone mode with instant finalization

Operator config files are minimal overrides — HOCON's fallback semantics merge them on top of the built-in defaults automatically.

**Genesis-locked parameters** (cannot change after network creation): `fault-tolerance-threshold` and `synchrony-constraint-threshold` are written into the genesis block's on-chain state. Changing them requires a new genesis.

See also: [Consensus Configuration Guide](https://github.com/F1R3FLY-io/system-integration/blob/main/docs/consensus-configuration.md) — FTT and synchrony threshold semantics, finalization formula, recommended values per validator set size.

---

## 11. Known Limitations

See [F1R3FLY-io/f1r3node issues](https://github.com/F1R3FLY-io/f1r3node/issues) for current open issues related to consensus.

---

## Source File Map

### Core Consensus
| File | Role |
|------|------|
| `casper/src/rust/casper.rs` | `Casper` and `MultiParentCasper` trait definitions |
| `casper/src/rust/multi_parent_casper_impl.rs` | Main implementation: snapshot, propose, validate, finalize |
| `casper/src/rust/casper_conf.rs` | `CasperConf`, `HeartbeatConf` configuration structs |

### Engine
| File | Role |
|------|------|
| `casper/src/rust/engine/engine.rs` | State machine: GenesisValidator → Initializing → Running |
| `casper/src/rust/engine/running.rs` | Message handling in Running state |
| `casper/src/rust/engine/genesis_ceremony_master.rs` | Genesis ceremony coordination |
| `casper/src/rust/engine/genesis_validator.rs` | Genesis validator participation |
| `casper/src/rust/engine/approve_block_protocol.rs` | Genesis approval collection |

### Block Lifecycle
| File | Role |
|------|------|
| `casper/src/rust/blocks/proposer/proposer.rs` | Proposal orchestration, constraint checks |
| `casper/src/rust/blocks/proposer/block_creator.rs` | Deploy selection, block assembly, adaptive cap |
| `casper/src/rust/blocks/block_processor.rs` | 8-step validation pipeline, casper buffer |
| `casper/src/rust/validate.rs` | Individual validation rules |

### Fork Choice & Safety
| File | Role |
|------|------|
| `casper/src/rust/estimator.rs` | LMD GHOST fork choice |
| `casper/src/rust/safety/clique_oracle.rs` | Clique oracle, fault tolerance computation |
| `casper/src/rust/finality/finalizer.rs` | Finalization search with work budgets |
| `casper/src/rust/synchrony_constraint_checker.rs` | Synchrony constraint + recovery bypass |
| `casper/src/rust/equivocation_detector.rs` | Equivocation types and detection |

### Merging
| File | Role |
|------|------|
| `casper/src/rust/merging/dag_merger.rs` | Multi-parent state merge |
| `casper/src/rust/merging/conflict_set_merger.rs` | Deploy conflict resolution |

### Liveness
| File | Role |
|------|------|
| `node/src/rust/instances/heartbeat_proposer.rs` | Heartbeat-driven proposals, stale recovery |

### Storage (consensus-agnostic)
| File | Role |
|------|------|
| `block-storage/src/rust/dag/block_dag_key_value_storage.rs` | DAG structure, latest messages, metadata |
| `block-storage/src/rust/key_value_block_store.rs` | Block persistence |
| `block-storage/src/rust/deploy/key_value_deploy_storage.rs` | Deploy pool |

### Configuration
| File | Role |
|------|------|
| `node/src/main/resources/defaults.conf` | HOCON defaults for all parameters |

---

**See also:** [Casper Module Overview](./README.md) | [Byzantine Fault Tolerance](./BYZANTINE_FAULT_TOLERANCE.md) | [Synchrony Constraint](./SYNC_CONSTRAINT.md) | [Data Flows](../data-flows/)

[← Back to docs index](../README.md)
