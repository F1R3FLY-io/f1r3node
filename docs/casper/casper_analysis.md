# Casper Block DAG Analysis: Orphaned Blocks Issue

## Executive Summary

This document provides a detailed analysis of how Casper handles block DAG structure, parent selection, finalization, and storage. The analysis reveals the mechanisms by which blocks can become unreferenced (orphaned) and how finalization propagates through the DAG.

---

## 1. BLOCK DAG STRUCTURE

### 1.1 Core DAG Representation

**File:** `/block-storage/src/main/scala/coop/rchain/blockstorage/dag/BlockDagKeyValueStorage.scala`

The BlockDAG is maintained through multiple interconnected index structures:

- **DAG Set** (line 40): `Set[BlockHash]` - All blocks currently in the DAG
- **Latest Messages Map** (line 41): `Map[Validator, BlockHash]` - Latest message per validator
- **Child Map** (line 42): `Map[BlockHash, Set[BlockHash]]` - Parent-to-children relationships
- **Height Map** (line 43): `SortedMap[Long, Set[BlockHash]]` - Blocks grouped by block number
- **Invalid Blocks Set** (line 44): `Set[BlockMetadata]` - Explicitly invalid blocks
- **Last Finalized Block** (line 45): `BlockHash` - Current finalization frontier

### 1.2 Block Metadata Structure

**File:** `/block-storage/src/main/scala/coop/rchain/blockstorage/dag/BlockMetadataStore.scala:50-58`

Each block metadata includes:
- Block hash
- Parents: Set of parent block hashes (multiple parents in multi-parent Casper)
- Block number: Height in the DAG
- Weight map: Bonds/stake per validator
- Finalization status flags

### 1.3 How the DAG is Maintained During Block Insertion

**File:** `/block-storage/src/main/scala/coop/rchain/blockstorage/dag/BlockDagKeyValueStorage.scala:161-259`

When a block is inserted:

```
Block Insertion Flow:
1. Check if block sender is valid (lines 171-173)
2. Determine if block should become the new latest message for its sender (lines 183-194)
   - Only if sequence number >= current latest message
3. Add block metadata to index (line 224)
4. Add deploy indices for quick lookup (line 228)
5. Update invalid blocks index if invalid (line 231)
6. Update latest messages for sender and newly bonded validators (lines 239-246)
7. Record finalization if approved block (line 249)
```

### 1.4 Parent Reference Resolution

**File:** `/block-storage/src/main/scala/coop/rchain/blockstorage/dag/BlockMetadataStore.scala:150-160`

When a block is added to the DAG:

```scala
// Update children relation map
val blockChilds = block.parents.map((_, Set(block.hash))) + ((block.hash, Set()))
val newChildMap = blockChilds.foldLeft(state.childMap) {
  case (acc, (key, newChildren)) =>
    val currChildren = acc.getOrElse(key, Set.empty[BlockHash])
    acc.updated(key, currChildren ++ newChildren)
}
```

**Key Point:** Each parent now knows about this block as a child. This is how the DAG maintains forward pointers for ancestor traversal.

---

## 2. PARENT SELECTION & LMD GHOST ESTIMATOR

### 2.1 Estimator Overview

**File:** `/casper/src/main/scala/coop/rchain/casper/Estimator.scala`

The Estimator implements LMD GHOST (Latest Message Driven GHOST) to select parents for new blocks.

**Function Flow (lines 40-76):**

```scala
tips(dag, genesis) -> tips(dag, genesis, latestMessagesHashes)
```

The algorithm:

1. **Get Latest Messages** (line 46): Fetch the current set of latest message hashes per validator
2. **Filter Invalid** (line 61): Remove validators marked as invalid
3. **Calculate LCA** (lines 63-67): Find lowest common ancestor of all latest messages
4. **Build Scores Map** (line 69): Score each block based on validator weight support
5. **Rank Fork Choices** (line 71): Recursively rank blocks by score and their descendants
6. **Filter Deep Parents** (line 73): Optionally limit parent depth based on block number delta
7. **Take Top N** (line 75): Return up to `maxNumberOfParents` highest-scored parents

### 2.2 Scoring Mechanism

**File:** `/casper/src/main/scala/coop/rchain/casper/Estimator.scala:127-170`

The `buildScoresMap` function:

```scala
def buildScoresMap(
    blockDag: BlockDagRepresentation[F],
    latestMessagesHashes: Map[Validator, BlockHash],
    lowestCommonAncestor: BlockHash
): F[Map[BlockHash, Long]]
```

For each validator's latest message:
- Traverse backwards through main parent chain from latest message to LCA (lines 149-151)
- Weight each block on that path by validator's stake (line 155)
- Accumulate weights in the score map (line 156)

**Result:** Blocks on more validators' main chains have higher scores.

### 2.3 Fork Choice Ranking

**File:** `/casper/src/main/scala/coop/rchain/casper/Estimator.scala:172-189`

The `rankForkchoices` function recursively ranks and promotes children:

```scala
private def replaceBlockHashWithChildren(
    b: BlockHash,
    blockDag: BlockDagRepresentation[F],
    scores: Map[BlockHash, Long]
): F[List[BlockHash]]
```

**Key Logic (lines 200-214):**

```scala
blockDag
  .children(b)  // Get children from child map
  .map(
    maybeChildren =>
      maybeChildren
        .flatMap { children =>
          toNonEmptyList(children.filter(scores.contains))  // Only children that were scored
        }
        .getOrElse(List(b))  // If no scored children, return block itself
  )
```

**CRITICAL ISSUE:** Only children that appear in the `scores` map are considered scored. Children are only in the scores map if they are ancestors of at least one latest message.

### 2.4 Maximum Parent Depth Constraint

**File:** `/casper/src/main/scala/coop/rchain/casper/Estimator.scala:78-95`

If `maxParentDepthOpt` is set:

```scala
case Some(maxParentDepth) =>
  val mainHash +: secondaryHashes = rankedLatestHashes
  // Keep only secondary parents within maxParentDepth of main parent
  secondaryParents.filter(p => maxBlockNumber - p.blockNum <= maxParentDepth)
```

**Purpose:** Prevent referencing blocks that are too far in the past (may have low relevance).

---

## 3. HOW BLOCKS BECOME UNREFERENCED (ORPHANED)

### 3.1 Root Cause: Latest Message Update Mechanism

**File:** `/block-storage/src/main/scala/coop/rchain/blockstorage/dag/BlockDagKeyValueStorage.scala:183-194`

A block becomes the new latest message for its sender **only if**:

```scala
def shouldAddAsLatest: F[Boolean] =
  latestMessagesIndex
    .get1(block.sender)
    .flatMap(_.traverse(blockMetadataIndex.getUnsafe))
    .map(_.map(_.seqNum))
    .map(lmSeqNumOpt => lmSeqNumOpt.isEmpty || lmSeqNumOpt.exists(block.seqNum >= _))
```

**Logic:** Block becomes latest message if:
- No existing latest message for validator exists, OR
- Block's sequence number >= existing latest message's sequence number

### 3.2 Orphaning Scenario: Rapid Block Arrival

**Scenario:**

```
Time T1: Block A (seqNum=1) arrives from Validator V
         -> A becomes latestMessage[V]
         -> New blocks can now reference A in justifications

Time T2: Block B (seqNum=2) arrives from Validator V
         -> B becomes latestMessage[V] (replaces A)
         -> Estimator recalculates tips based on new latestMessages

Time T3: New blocks created by other validators reference the NEW tips
         -> These tips are scored based on latest messages including B
         -> A may no longer be in the scoring calculation
         -> Estimator's rankForkchoices may not traverse through A
         -> A becomes unreferenced in terms of tip selection
```

### 3.3 Why Unreferenced Blocks Matter: Estimator Children Filtering

**File:** `/casper/src/main/scala/coop/rchain/casper/Estimator.scala:200-214`

```scala
toNonEmptyList(children.filter(scores.contains))  // ONLY scored children
```

If Block A (old latestMessage):
1. Is not an ancestor of any new latestMessages
2. Has no children in the scores map
3. Will not be promoted during fork choice ranking

**Result:** A is no longer in the selected parent set, even though it exists in the DAG.

### 3.4 Storage Persistence: Blocks Are NOT Deleted

**File:** `/block-storage/src/main/scala/coop/rchain/blockstorage/KeyValueBlockStore.scala`

Blocks are stored in persistent key-value stores:
- Block metadata in `blockMetadataIndex`
- Full block data in block store
- Deploy indices

**Critical:** Unreferenced blocks REMAIN in storage and can still be finalized through the finalization oracle.

---

## 4. JUSTIFICATION CREATION (Block Proposal)

### 4.1 CasperSnapshot Preparation

**File:** `/casper/src/main/scala/coop/rchain/casper/MultiParentCasperImpl.scala:233-242`

When creating a new block, justifications are built from latest messages:

```scala
justifications <- {
  for {
    lms <- dag.latestMessages  // Get current latest messages
    r = lms.toList
      .map {
        case (validator, blockMetadata) => Justification(validator, blockMetadata.blockHash)
      }
      .filter(j => onChainState.bondsMap.keySet.contains(j.validator))  // Only bonded validators
  } yield r.toSet
}
```

**Key Points:**
1. Justifications reference the CURRENT latest messages
2. Only bonded validators' latest messages are included
3. If Block A is latest message at time T, and Block B becomes latest message at T+dt, newly created blocks reference B, not A

### 4.2 Block Creation Process

**File:** `/casper/src/main/scala/coop/rchain/casper/blocks/proposer/BlockCreator.scala:38-177`

When creating a block:

```scala
def create(s: CasperSnapshot[F], validatorIdentity: ValidatorIdentity): F[BlockCreatorResult]
```

The snapshot contains:
- `parents`: Selected parents from estimator (lines 47, 222-224)
- `justifications`: Current latest messages from bonded validators (lines 48, 233-242)

Parents are selected from tips computed by the estimator at snapshot time.

---

## 5. FINALIZATION LOGIC

### 5.1 How Blocks Can Be Finalized

**File:** `/casper/src/main/scala/coop/rchain/casper/finality/Finalizer.scala:91-186`

The finalization process:

1. **Build Agreement Stream** (lines 105-140):
   - Start with latest messages agreeing on themselves
   - Traverse down main parent chains
   - Record validator agreements at each level

2. **Filter Unorphanable Blocks** (lines 160-162):
   ```scala
   def cannotBeOrphaned(
       messageWeightMap: WeightMap,
       agreeingWeightMap: WeightMap
   ): Boolean = {
     activeStakeAgreeing > activeStakeTotal.toFloat / 2
   }
   ```
   Only blocks with >50% stake agreement qualify

3. **Compute Fault Tolerance** (lines 165-174):
   - Use CliqueOracle to determine fault tolerance score
   - First block exceeding threshold becomes finalized (lines 176-178)

### 5.2 Finalization Without Justifications

**CRITICAL INSIGHT:** Finalization does NOT require a block to be:
- Referenced in any justifications
- A parent of recently created blocks
- In the current set of tips

**File:** `/casper/src/main/scala/coop/rchain/casper/finality/Finalizer.scala:115-124`

Finalization traverses the DAG using **main parent chains**:

```scala
val nextF = layer
  .traverse {
    case (v, message) =>
      message.parents.headOption.traverse(dag.lookupUnsafe)
```

The algorithm walks backwards through `parents.head` (main parent), independent of:
- Current latest messages
- Current tips
- Estimator fork choice

**Result:** Unreferenced blocks CAN be finalized if they lie on the main chain path from a finalization ancestor to a validator's latest message.

### 5.3 Main Parent Chain Concept

**File:** `/block-storage/src/main/scala/coop/rchain/blockstorage/dag/BlockDagRepresentationSyntax.scala:104-115`

```scala
def mainParentChain(h: BlockHash, stopAtHeight: Long = 0)(
    implicit sync: Sync[F]
): Stream[F, BlockHash] =
  Stream.unfoldEval(h) { message =>
    lookupUnsafe(message).map(
      meta =>
        if (meta.blockNum <= stopAtHeight)
          none[(BlockHash, BlockHash)]
        else
          meta.parents.headOption.map(v => (v, v))
    )
  }
```

The main parent chain is formed by following `parents.head` backwards.

---

## 6. BLOCK STORAGE & RETRIEVAL

### 6.1 Block Storage Strategy

**File:** `/block-storage/src/main/scala/coop/rchain/blockstorage/BlockStore.scala`

Blocks are stored through multiple indices:

1. **Block Metadata Store** (BlockHash -> BlockMetadata)
2. **Block Message Store** (BlockHash -> BlockMessage)
3. **Deploy Index** (DeployId -> BlockHash)
4. **Latest Messages Index** (Validator -> BlockHash)
5. **Invalid Blocks Index** (BlockHash -> BlockMetadata)

### 6.2 Block Retrieval

All blocks can be retrieved by hash:

```scala
dag.lookup(blockHash)       // Get BlockMetadata
BlockStore[F].get(blockHash) // Get full BlockMessage
```

**Unreferenced blocks ARE retrievable** - they're in storage but not in the fork-choice path.

### 6.3 Finalization Effect on Storage

**File:** `/casper/src/main/scala/coop/rchain/casper/MultiParentCasperImpl.scala:119-137`

When blocks are finalized:

```scala
def processFinalised(finalizedSet: Set[BlockHash]): F[Unit] =
  finalizedSet.toList.traverse { h =>
    for {
      // Remove block deploys from persistent store
      deploysRemoved <- DeployStorage[F].remove(deploys)
      // Remove block index from cache
      _ <- BlockIndex.cache.remove(h).pure
      // Remove block post-state mergeable channels
      _ <- RuntimeManager[F].getMergeableStore.delete(stateHash)
    } yield ()
  }
```

Finalization triggers cleanup of:
- Deploy history
- Block indices from cache
- Mergeable channel data

But the block metadata remains in the DAG storage.

---

## 7. FORK CHOICE & "MAIN CHAIN" DETERMINATION

### 7.1 Fork Choice Calculation

**File:** `/casper/src/main/scala/coop/rchain/casper/Estimator.scala:55-76`

The current fork choice at any point is:

```scala
def tips(dag, genesis, latestMessagesHashes): ForkChoice =
  for {
    invalidLatestMessages <- dag.invalidLatestMessages(latestMessagesHashes)
    filteredLatestMessages = latestMessagesHashes -- invalidLatestMessages.keys
    lca <- calculateLCA(dag, genesis, filteredLatestMessages)
    scoresMap <- buildScoresMap(dag, filteredLatestMessages, lca)
    rankedLatestMessages <- rankForkchoices(List(lca), dag, scoresMap)
    rankedShallowHashes <- filterDeepParents(rankedLatestMessages, dag)
  } yield ForkChoice(rankedShallowHashes.take(maxNumberOfParents), lca)
```

**ForkChoice contains:**
- `tips`: Selected parent hashes (up to maxNumberOfParents)
- `lca`: Lowest common ancestor of all latest messages

### 7.2 Identifying the "Main Chain"

**File:** `/block-storage/src/main/scala/coop/rchain/blockstorage/dag/BlockDagRepresentationSyntax.scala:104-115`

The main chain is determined by following `parents.head`:

```scala
def isInMainChain(ancestor: BlockHash, descendant: BlockHash): F[Boolean] =
  mainParentChain(descendant).filter(_ == ancestor).head.compile.last
    .map(_.isDefined)
```

A block is in the main chain if it can be reached by following first parents backwards from a descendant.

### 7.3 Safety Oracle & Finalization Path

**File:** `/casper/src/main/scala/coop/rchain/casper/safety/CliqueOracle.scala:117-133`

The safety oracle determines finalization by:

1. Finding validators that agree on a target message
2. Checking if these validators form a "clique" (no disagreements visible)
3. Computing fault tolerance: `(maxCliqueWeight * 2 - totalStake) / totalStake`

**File:** `/casper/src/main/scala/coop/rchain/casper/safety/CliqueOracle.scala:63-95`

Two validators have no disagreement if:
- Their latest messages are both in the main chain with the target message
- No self-justification between them shows disagreement on the target

---

## 8. CRITICAL FINDINGS: ORPHANED BLOCKS ISSUE

### 8.1 How Blocks Become Orphaned

**Mechanism:**

1. **Block Creation:** Validator creates Block A with seqNum=1
   - Becomes latestMessage[Validator] immediately
   - Included in justifications for newly created blocks

2. **Quick Replacement:** Validator creates Block B with seqNum=2
   - Becomes new latestMessage[Validator]
   - A is no longer the latest message

3. **Fork Choice Recalculation:**
   - New blocks reference B in justifications, not A
   - Estimator scores blocks based on agreement with B
   - If no descendant of A reaches any new tips, A has no scored children

4. **Estimator Exclusion:**
   - `replaceBlockHashWithChildren` filters for `scores.contains`
   - If A's children aren't scored, A isn't promoted to tips
   - A becomes "unreferenced" in terms of parent selection

### 8.2 Why This Is Not Catastrophic (Yet)

**File:** `/casper/src/main/scala/coop/rchain/casper/finality/Finalizer.scala:91-186`

1. **Blocks Remain in Storage:** A is still in DAG storage
2. **Finalization Still Works:** A can still be finalized through main parent chains
3. **Blocks Can Be Retrieved:** A can be looked up by hash and retrieved

**However:** Unreferenced blocks may:
- Never receive new justifications
- Become harder to reason about in fork analysis
- Accumulate in memory if not pruned

### 8.3 Potential Issues with Current Design

**Issue 1: Estimator Latest Message Dependency**

The estimator's fork-choice relies entirely on `dag.latestMessages`. If a validator's sequence advances faster than descendants are added, older messages can fall off the scored set.

**File:** `/casper/src/main/scala/coop/rchain/casper/Estimator.scala:200-214`

```scala
children.filter(scores.contains)  // Only scored children = descendants of latest messages
```

**Issue 2: Rapid Sequence Number Progression**

If a validator issues blocks rapidly:
- Block N becomes latest message
- Block N+1 issued immediately after
- Block N never becomes a parent because N+1 replaces it before N's descendants are scored
- Result: N is orphaned

**Issue 3: MaxParentDepth Filtering**

**File:** `/casper/src/main/scala/coop/rchain/casper/Estimator.scala:78-95`

Even if a block is scored, it can be filtered if too deep:

```scala
secondaryParents.filter(p => maxBlockNumber - p.blockNum <= maxParentDepth)
```

If `maxBlockNumber - p.blockNum > maxParentDepth`, the block is excluded from parents even if well-scored.

---

## 9. BLOCKS REFERENCED IN KEY LOGIC LOCATIONS

### 9.1 Where Latest Messages Are Used

| Location | Purpose | File |
|----------|---------|------|
| Estimator tips calculation | Fork choice | `Estimator.scala:46, 58` |
| Justification creation | Block references | `MultiParentCasperImpl.scala:235` |
| Finalizer agreement stream | Finalization | `Finalizer.scala:106-124` |
| Scoring map building | Weight accumulation | `Estimator.scala:162-169` |
| LCA calculation | Common ancestor | `Estimator.scala:103-105` |

### 9.2 Where Blocks Are Stored

| Index | Purpose | File |
|-------|---------|------|
| DAG Set | All block membership | `BlockDagKeyValueStorage.scala:142` |
| Block Metadata Store | Block information | `BlockMetadataStore.scala` |
| Child Map | Forward pointers | `BlockMetadataStore.scala:156-160` |
| Latest Messages Index | Current tips per validator | `BlockDagKeyValueStorage.scala:141` |
| Height Map | Blocks by number | `BlockMetadataStore.scala:143-144` |
| Invalid Blocks | Slashed blocks | `BlockDagKeyValueStorage.scala:145` |

### 9.3 Where Finalization Queries

| Query | Purpose | File |
|-------|---------|------|
| `dag.isFinalized(blockHash)` | Check finalization | `BlockDagRepresentationSyntax.scala:76-77` |
| `dag.mainParentChain(h)` | Traverse lineage | `BlockDagRepresentationSyntax.scala:104-115` |
| `dag.ancestors()` | Find unfinalized parents | `BlockDagRepresentationSyntax.scala:163-175` |
| `dag.descendants()` | Find children | `BlockDagRepresentationSyntax.scala:151-161` |

---

## 10. RECOMMENDATIONS FOR INVESTIGATION

### 10.1 Diagnose Orphaned Blocks

1. **Query the DAG** for blocks that:
   - Are not in `latestMessages` (orphaned from tip perspective)
   - But exist in `DAG storage`
   - And check if they have unfinalized descendants

2. **Trace the lineage**:
   - Use `mainParentChain()` to verify ancestor path
   - Check if the block has valid children

3. **Analyze scores**:
   - Recompute `buildScoresMap()` and check if orphaned blocks appear
   - Verify if their children are scored

### 10.2 Monitor System Health

- Track ratio of: `(DAG blocks) / (latest messages)`
- If ratio grows unbounded, orphaning is occurring
- Monitor `blockNum` distribution across validators

### 10.3 Potential Fixes

1. **Extend parent selection window:**
   - Keep more blocks as candidates in `buildScoresMap()`
   - Don't filter children so aggressively

2. **Implement block affinity:**
   - Prefer children of existing parents
   - Reduce volatility in parent set changes

3. **Enhanced latestMessage tracking:**
   - Keep historical latest messages
   - Reference recent blocks even if superceded

---

## 11. FILE REFERENCE SUMMARY

### Core Estimator/Fork-Choice
- `/casper/src/main/scala/coop/rchain/casper/Estimator.scala`

### Block Proposal & Justifications
- `/casper/src/main/scala/coop/rchain/casper/blocks/proposer/BlockCreator.scala`
- `/casper/src/main/scala/coop/rchain/casper/MultiParentCasperImpl.scala`

### Finalization
- `/casper/src/main/scala/coop/rchain/casper/finality/Finalizer.scala`
- `/casper/src/main/scala/coop/rchain/casper/safety/CliqueOracle.scala`

### DAG Storage
- `/block-storage/src/main/scala/coop/rchain/blockstorage/dag/BlockDagKeyValueStorage.scala`
- `/block-storage/src/main/scala/coop/rchain/blockstorage/dag/BlockMetadataStore.scala`
- `/block-storage/src/main/scala/coop/rchain/blockstorage/dag/BlockDagRepresentationSyntax.scala`

### DAG Operations
- `/casper/src/main/scala/coop/rchain/casper/util/DagOperations.scala`
- `/casper/src/main/scala/coop/rchain/casper/util/ProtoUtil.scala`

---

## Conclusion

The Casper implementation maintains a complete DAG where all blocks remain accessible in storage. However, **blocks can become unreferenced from the estimator's fork-choice calculation** if they are quickly superseded in the latest-message indices and their descendants don't propagate through the latest-message scoring system.

Critically, **unreferenced blocks CAN still be finalized** through the finalization oracle, which works independently of the estimator by traversing main parent chains. This means orphaned blocks remain part of the system and can accumulate if the conditions for quick replacement occur frequently.

The system is designed to handle this gracefully, but monitoring for orphaning patterns is recommended to detect performance or consistency issues.

