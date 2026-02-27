# Invalid Block Handling Issue

## Problem Statement

When a validator produces an invalid block but continues producing subsequent blocks, the current implementation may incorrectly build on their newer blocks while only filtering the invalid one. The correct behavior should be to **refuse to build on any blocks from a validator after they produce an invalid block**.

## Current Behavior (Incorrect)

### Scenario

1. Validator V produces block B1 (valid)
2. Validator V produces block B2 (invalid) - marked invalid, added to `invalidBlocksSet`
3. Validator V produces block B3 (valid) - becomes `latestMessage[V]`
4. When selecting parents, we call `invalidLatestMessages(latestMessageHashes)`
5. This checks: "Is B3 in `invalidBlocksSet`?" - **No, it's not**
6. B3 passes the filter and is used as a parent
7. **Problem**: B3 builds on B2 (the invalid block), so including B3 means implicitly accepting the invalid chain

### Code Analysis

**`invalidLatestMessages` in `BlockDagRepresentationSyntax.scala:76-83`:**
```scala
def invalidLatestMessages(latestMessagesHashes: Map[Validator, BlockHash])(
    implicit sync: Sync[F]
): F[Map[Validator, BlockHash]] =
  dag.invalidBlocks.map { invalidBlocks =>
    latestMessagesHashes.filter {
      case (_, blockHash) => invalidBlocks.map(_.blockHash).contains(blockHash)
    }
  }
```

This only checks if the validator's **current latest message** is in the invalid set. It does NOT check if ANY ancestor of the latest message is invalid.

**Parent selection in `MultiParentCasperImpl.scala:130-137`:**
```scala
for {
  lmh        <- dag.latestMessageHashes
  invalidLms <- dag.invalidLatestMessages(lmh)
  validLms   = lmh -- invalidLms.keys
} yield
  if (validLms.isEmpty) IndexedSeq(approvedBlock.blockHash)
  else validLms.values.toSet.toIndexedSeq
```

This filters out validators whose latest message is invalid, but not validators who have an invalid block in their ancestry chain.

## Expected Behavior (Correct)

Once a validator produces an invalid block:
1. That validator should be **slashed** (this already happens via `SlashDeploy`)
2. Their bond should be reduced/removed (handled by PoS contract)
3. **All subsequent blocks from that validator should be rejected**
4. Parent selection should use the validator's **last valid block**, not their latest block

### What Needs to Change

1. **Track "last valid block" per validator**: When a block is marked invalid, we need to record that validator's last valid block (the parent of the invalid block in their self-justification chain).

2. **Modified parent selection**: Instead of using `latestMessageHashes` directly, we need:
   ```
   For each validator:
     If validator has any invalid block in their chain:
       Use their last valid block (before the first invalid block)
     Else:
       Use their latest message
   ```

3. **Ancestor checking**: When filtering parents, check if any block in the validator's chain (from genesis to latest) is in the invalid set.

## Implementation Options

### Option A: Track Last Valid Block Index

Add a new index: `Map[Validator, BlockHash]` for "lastValidBlock" per validator.

When marking a block as invalid:
1. Find the validator's self-justification (previous block from same validator)
2. Store that as their "lastValidBlock"
3. All subsequent blocks from this validator are automatically excluded because we use lastValidBlock instead of latestMessage

**Pros**: Efficient lookups
**Cons**: Requires schema change, migration

### Option B: Ancestor Chain Check

When selecting parents, for each validator's latest message:
1. Walk the self-justification chain backward
2. Check if any block in the chain is in `invalidBlocksSet`
3. If yes, find the last valid block and use that instead

**Pros**: No schema change needed
**Cons**: More expensive at runtime (chain traversal)

### Option C: Eager Marking

When a block is marked invalid, also mark all descendant blocks from the same validator as invalid.

**Pros**: Simple filter logic stays the same
**Cons**: Requires updating multiple blocks when one is invalid

## Recommended Approach

**Option A (Track Last Valid Block Index)** is recommended because:
- It's the most efficient at query time
- Parent selection is called frequently
- The storage overhead is minimal (one entry per slashed validator)

## Related Code Locations

| File | Function | Relevance |
|------|----------|-----------|
| `BlockDagRepresentationSyntax.scala:68-83` | `invalidLatestMessages` | Current filtering logic |
| `BlockDagRepresentationSyntax.scala:91-99` | `selfJustificationChain` | Walking validator's block chain |
| `BlockDagKeyValueStorage.scala:230-231` | `insert` | Where invalid blocks are recorded |
| `MultiParentCasperImpl.scala:125-137` | `estimator` | Parent selection |
| `MultiParentCasperImpl.scala:266-273` | `getSnapshot` | Parent selection for block creation |
| `BlockCreator.scala:106-130` | `prepareSlashingDeploys` | Slashing logic |

## Testing Considerations

A test should verify:
1. Validator V creates block B1 (valid)
2. Validator V creates block B2 (invalid)
3. Validator V creates block B3 (valid, building on B2)
4. When another validator creates a block, B3 should NOT be a parent
5. Only B1 (or earlier valid blocks) should be considered as V's contribution

## Related Work

- Slashing deploys are already created via `prepareSlashingDeploys` for validators with invalid latest messages
- The PoS contract handles bond reduction
- This issue is about the **parent selection** logic, not the slashing mechanism itself

---

*Created: December 2025*
*Status: Analysis complete, implementation pending*
