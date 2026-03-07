# Casper Synchrony Constraint

## Overview

The **synchrony constraint** is a Byzantine fault-tolerant mechanism in RChain's Casper consensus protocol that prevents validators from proposing blocks too far ahead of the rest of the network. It enforces that a validator must wait to see blocks from other validators (weighted by stake) before being allowed to propose a new block.

This constraint is designed to ensure **network synchrony** and prevent scenarios where a validator with high stake could dominate block production or fork the network by proposing blocks without proper coordination with other validators.

## The Problem It Solves

### Byzantine Fault Tolerance and Network Synchrony

In a Byzantine fault-tolerant (BFT) consensus system, validators must coordinate their actions to maintain agreement even in the presence of faulty or malicious nodes. The synchrony constraint addresses several critical issues:

1. **Prevents Validator Isolation**: Without this constraint, a single validator could continuously propose blocks on their own chain without waiting for other validators, potentially creating a fork that diverges from the main network.

2. **Enforces Stake-Weighted Participation**: In a Proof-of-Stake system, security comes from having a majority of stake participating honestly. The synchrony constraint ensures that blocks are only created when enough stake (other than the proposer's own) has recently participated.

3. **Maintains DAG Coherence**: Casper uses a DAG (Directed Acyclic Graph) structure where blocks reference multiple parents through justifications. The synchrony constraint ensures validators are building on a coherent DAG with recent participation from other validators.

4. **Prevents Equivocation Exploitation**: Malicious validators could attempt to create conflicting blocks. By requiring recent blocks from other validators, the constraint makes such attacks more difficult.

### Design Rationale

The synchrony constraint implements the principle that **validators should not propose in isolation**. In a healthy network with N validators, before a validator proposes block B_n+1, they should have seen evidence that other validators (weighted by stake) have produced blocks since their last proposal at B_n.

This ensures:
- Network liveness depends on validator participation
- Finalization can progress (requires multiple validators to build on blocks)
- The DAG structure reflects genuine network consensus, not solo activity

## Implementation

### Location

- **Primary Implementation**: `/Users/leaf/Pyrofex/firefly/f1r3fly/casper/src/main/scala/coop/rchain/casper/SynchronyConstraintChecker.scala`
- **Error Type**: `/Users/leaf/Pyrofex/firefly/f1r3fly/casper/src/main/scala/coop/rchain/casper/blocks/proposer/ProposeResult.scala` (lines 23, 41)
- **Used By**: `/Users/leaf/Pyrofex/firefly/f1r3fly/casper/src/main/scala/coop/rchain/casper/blocks/proposer/Proposer.scala` (lines 184-190)

### The Algorithm

The synchrony constraint checker performs the following steps when a validator attempts to propose a block:

#### 1. Retrieve Validator's Last Proposed Block (Lines 48-51)

```scala
s.dag.latestMessageHash(validator).flatMap {
  case Some(lastProposedBlockHash) =>
    for {
      lastProposedBlockMeta <- s.dag.lookupUnsafe(lastProposedBlockHash)
```

The checker first finds the validator's most recent block in the DAG.

#### 2. Calculate Validators Seen Since Last Proposal (Lines 69, 19-35)

```scala
seenSenders <- calculateSeenSendersSince(lastProposedBlockMeta, s.dag)

private def calculateSeenSendersSince(
    lastProposed: BlockMetadata,
    dag: BlockDagRepresentation[F]
): F[Set[Validator]] =
  for {
    latestMessages <- dag.latestMessageHashes
    seenSendersSince = lastProposed.justifications.flatMap {
      case Justification(validator, latestBlockHash) =>
        if (validator != lastProposed.sender && latestMessages(validator) != latestBlockHash) {
          // Since we would have fetched missing justifications initially, it can only mean
          // that we have received at least one new block since then
          Some(validator)
        } else {
          None
        }
    }.toSet
  } yield seenSendersSince
```

**Key Logic**: For each validator V in the last proposed block's justifications:
- If V is not the proposer themselves (`validator != lastProposed.sender`)
- AND V's current latest message hash differs from what was in the justifications (`latestMessages(validator) != latestBlockHash`)
- THEN we have seen at least one new block from V

This identifies which validators have produced new blocks since the last proposal.

#### 3. Get Active Validators and Weight Map (Lines 62-67)

```scala
activeValidators <- runtimeManager.getActiveValidators(mainParentStateHash)

validatorWeightMap = mainParentMeta.weightMap.filter {
  case (validator, _) => activeValidators.contains(validator)
}
```

The checker retrieves the current set of active validators from the PoS (Proof-of-Stake) contract and filters the weight map to only include active validators.

#### 4. Calculate Stake Weights (Lines 70, 75-76)

```scala
sendersWeight = seenSenders.toList.flatMap(validatorWeightMap.get).sum

validatorOwnStake = validatorWeightMap.getOrElse(validator, 0L)
otherValidatorsWeight = validatorWeightMap.values.sum - validatorOwnStake
```

- `sendersWeight`: Total stake of validators who have produced new blocks
- `validatorOwnStake`: The proposer's own stake
- `otherValidatorsWeight`: Total stake of all validators except the proposer

#### 5. Calculate Synchrony Constraint Value (Lines 78-80)

```scala
synchronyConstraintValue = if (otherValidatorsWeight == 0) 1
else sendersWeight.toDouble / otherValidatorsWeight
```

**The Formula**:
```
synchronyConstraintValue = sendersWeight / otherValidatorsWeight
```

Where:
- `sendersWeight` = sum of stakes of validators who have produced new blocks since last proposal
- `otherValidatorsWeight` = total stake of all validators except the proposer

**Special Case**: If `otherValidatorsWeight == 0` (only one validator in the network), the value is set to 1.0 to allow the solo validator to propose.

#### 6. Check Against Threshold (Lines 87-90)

```scala
if (synchronyConstraintValue >= synchronyConstraintThreshold)
  CheckProposeConstraintsResult.success
else
  NotEnoughNewBlocks
```

If the calculated value meets or exceeds the configured threshold, the validator is allowed to propose. Otherwise, the proposal is rejected with `NotEnoughNewBlocks`.

#### 7. Genesis Exception (Lines 93-96)

```scala
latestBlockIsGenesis = lastProposedBlockMeta.blockNum == 0
allowedToPropose <- if (latestBlockIsGenesis)
                     CheckProposeConstraintsResult.success.pure[F]
                   else checkConstraint
```

If the validator's last block was the genesis block (block number 0), they are allowed to propose without checking the constraint. This enables the first block after genesis.

## Configuration

### Parameter

```hocon
casper {
  synchrony-constraint-threshold = 0.67
}
```

**Location**: `/Users/leaf/Pyrofex/firefly/f1r3fly/node/src/main/resources/defaults.conf` (line 239)

**Type**: `Double` (0.0 to 1.0)

**Default Value**: 0.67 (67%)

### What Different Values Mean

The threshold represents the **minimum fraction of other validators' stake** (excluding the proposer's own stake) that must have produced new blocks since the validator's last proposal.

- **threshold = 0.0**: Requires 0% of other stake to have seen new blocks (effectively disabled)
- **threshold = 0.33**: Requires 33% of other stake to have produced new blocks
- **threshold = 0.5**: Requires 50% of other stake (simple majority)
- **threshold = 0.67**: Requires 67% of other stake (Byzantine fault tolerance threshold)
- **threshold = 1.0**: Requires 100% of other stake (all validators must have produced new blocks)

### Byzantine Fault Tolerance

The default value of **0.67 (2/3 majority)** aligns with the Byzantine fault tolerance threshold used in many BFT consensus protocols. This value ensures that:

1. At least 2/3 of the stake (excluding the proposer) must have participated since the last proposal
2. This matches the typical BFT assumption that < 1/3 of validators can be faulty
3. Blocks are only created when there's strong evidence of network-wide participation

## Known Issues: Single Validator Networks

### The Problem

**Even with `synchrony-constraint-threshold = 0.0`, validators in very small networks get the `NotEnoughNewBlocks` error and their blocks are orphaned forever.**

This is a critical issue for:
- Development/test environments with 1 validator
- Small private networks with 2 validators
- Validators that temporarily lose connectivity to others

### Root Cause Analysis

The issue occurs due to how the synchrony constraint checker calculates "seen senders":

#### Line 27 Critical Condition

```scala
if (validator != lastProposed.sender && latestMessages(validator) != latestBlockHash) {
```

This condition has two parts:
1. `validator != lastProposed.sender` - Exclude the proposer themselves
2. `latestMessages(validator) != latestBlockHash` - Check if validator's latest message has changed

#### Single Validator Scenario

In a network with only **one validator**:

1. **First block after genesis**: Validator V proposes block B1
   - B1's justifications contain only V's own justification (to genesis or previous block)
   - `calculateSeenSendersSince` checks B1's justifications
   - For each justification (only V's own):
     - Check: `validator != lastProposed.sender` → `V != V` → **FALSE**
   - Result: `seenSenders = empty set` (because validator excludes themselves)

2. **Calculate weights**:
   - `sendersWeight = 0` (no senders in the empty set)
   - `otherValidatorsWeight = 0` (total stake minus own stake, but only one validator exists)

3. **Calculate synchrony constraint value** (line 79):
   ```scala
   synchronyConstraintValue = if (otherValidatorsWeight == 0) 1
   else sendersWeight.toDouble / otherValidatorsWeight
   ```
   - Since `otherValidatorsWeight == 0`, value is set to **1.0**

4. **Check against threshold**:
   - `synchronyConstraintValue (1.0) >= threshold (0.0)` → **TRUE**
   - **Should pass**, but...

#### Wait, Why Does It Still Fail?

The special case at line 79 (`if (otherValidatorsWeight == 0) 1`) should handle single validators correctly!

Let's examine more carefully:

#### The Real Issue: Active Validators Filter

At lines 62-67, the code filters validators by the **active validators** list from PoS:

```scala
activeValidators <- runtimeManager.getActiveValidators(mainParentStateHash)

validatorWeightMap = mainParentMeta.weightMap.filter {
  case (validator, _) => activeValidators.contains(validator)
}
```

**Potential Issue 1**: If the validator is not in the active validators list (e.g., during rotation, expired bond, etc.), then:
- `validatorWeightMap` will be empty or won't contain the validator
- `validatorOwnStake = validatorWeightMap.getOrElse(validator, 0L)` → 0
- `otherValidatorsWeight = 0 - 0 = 0` → triggers special case
- Should still work...

#### The Real Root Cause: Two-Validator Network

The issue is most pronounced in a **two-validator network**:

1. Validators V1 and V2 both bonded
2. V1 proposes block B1
   - Justifications: [V1 → B0, V2 → B0] (both justified to previous block B0)
3. V1 tries to propose block B2
   - Check: Has V2 produced a new block since B1?
   - V2's latest message is still B0 (same as in B1's justifications)
   - Result: `seenSenders = {}` (empty)
   - `sendersWeight = 0`
   - `validatorOwnStake = stake(V1)` (e.g., 100)
   - `otherValidatorsWeight = total_stake - stake(V1)` (e.g., 200 - 100 = 100)
   - `synchronyConstraintValue = 0 / 100 = 0.0`
   - Check: `0.0 >= 0.0` → **TRUE** (should pass)

Wait, even with threshold 0.0, this should pass!

#### The Actual Bug: Threshold Comparison

Let's look at the exact comparison again (line 87):

```scala
if (synchronyConstraintValue >= synchronyConstraintThreshold)
```

With `threshold = 0.0` and `value = 0.0`:
- `0.0 >= 0.0` → **TRUE** ✓

This should work! So why are users reporting failures?

#### The Real Problem: Configuration Not Being Applied

The most likely issue is that **the configuration is not being loaded or applied correctly**:

1. The threshold might be read from a different config file
2. The threshold might be overridden by command-line arguments
3. The threshold might not be propagated correctly through the code

Looking at line 44:
```scala
val synchronyConstraintThreshold = s.onChainState.shardConf.synchronyConstraintThreshold
```

The threshold comes from `onChainState.shardConf` - this is **on-chain configuration** stored in the genesis block or PoS contract, not just the node's configuration file!

**Critical Finding**: The synchrony constraint threshold is stored **on-chain** as part of the shard configuration. Simply changing the configuration file **after genesis** will not affect the constraint - it was locked in at genesis time!

### Why 0.0 Doesn't Work as Expected

Even if you set `synchrony-constraint-threshold = 0.0` in your configuration file:

1. **After genesis creation**: The threshold value was already written into the genesis block's state
2. **At runtime**: The checker reads from `s.onChainState.shardConf.synchronyConstraintThreshold` (on-chain value)
3. **Result**: Your configuration file change is ignored; the genesis value is used

To truly use threshold 0.0, you must:
- Set it in the configuration **before creating the genesis block**
- Or modify the genesis block data
- Or create a new network with a new genesis

### Additional Issue: The Self-Exclusion Logic

Even with threshold properly set to 0.0, there's a philosophical issue in line 27:

```scala
if (validator != lastProposed.sender && ...)
```

This explicitly excludes the proposer from the "seen senders" calculation. This means:

- **In a 1-validator network**: The validator can never see "other" validators
  - The special case at line 79 handles this by setting value to 1.0 when otherValidatorsWeight == 0

- **In a 2-validator network with no activity from the other validator**:
  - Validator V1 cannot propose again until V2 produces a block
  - Even with threshold 0.0, if V2 is offline/stuck, V1 cannot progress
  - This creates a **liveness failure**

## Recommendations for Different Network Sizes

### Single Validator (Standalone Development)

**Problem**: Single validator networks should work but may encounter issues if:
- Genesis was created with threshold > 0.0
- Validator is not in active validators list
- On-chain configuration differs from node configuration

**Solution**:
```hocon
casper {
  synchrony-constraint-threshold = 0.0

  genesis-block-data {
    # Ensure only one validator in bonds
    bonds-file = genesis/bonds.txt
  }
}
```

**Critical**: Set this **before** creating genesis. If genesis already exists with a different threshold, you must:
1. Delete the data directory (`rm -rf ~/.rnode/`)
2. Recreate genesis with the correct configuration
3. Restart the network

**Standalone Mode**: Use `-s` or `standalone = true` flag, which sets:
- `required-signatures = 0` (auto-approve genesis)
- `ceremony-master-mode = true` (create genesis if not found)

### Small Networks (2-4 Validators)

**Recommended**: `synchrony-constraint-threshold = 0.0`

With very few validators, the synchrony constraint can cause liveness issues:
- If one validator goes offline, others may be blocked
- With threshold 0.67 and 3 validators: need 2/3 of others, which could be just 1 validator
- Better to rely on other consensus safety mechanisms

**Alternative**: `synchrony-constraint-threshold = 0.5`

Requires simple majority participation but less stringent than default.

### Medium Networks (5-10 Validators)

**Recommended**: `synchrony-constraint-threshold = 0.5`

Balances liveness with safety:
- Ensures majority of stake is active
- Less likely to cause stalls than 0.67
- Still prevents individual validators from dominating

### Large Networks (10+ Validators)

**Recommended**: `synchrony-constraint-threshold = 0.67` (default)

The default Byzantine fault tolerance threshold:
- Assumes up to 1/3 of validators can be faulty/offline
- Ensures strong consensus before allowing proposals
- Aligns with BFT assumptions
- Sufficient validator diversity to maintain liveness

### Production Networks

**Recommended**: `synchrony-constraint-threshold = 0.67`

**Considerations**:
- Monitor validator participation rates
- Ensure at least 67% of stake is actively validating
- Set up alerting for validators that fall behind
- Plan for validator rotation and maintenance windows

## Examples and Scenarios

### Example 1: Three Validators with Default Threshold

**Setup**:
- Validators: V1 (stake: 100), V2 (stake: 150), V3 (stake: 50)
- Total stake: 300
- Threshold: 0.67

**Scenario**: V1 wants to propose a new block

**Calculation**:
- V1's last block was B_n at block number 100
- B_n's justifications included: [V1 → B_99, V2 → B_98, V3 → B_97]
- Current latest messages: [V1 → B_n, V2 → B_101, V3 → B_97]

**Analysis**:
1. Seen senders since B_n:
   - V2: Latest changed from B_98 to B_101 ✓ (V2 has produced new blocks)
   - V3: Latest still B_97 ✗ (V3 has not produced new blocks)
   - Result: `seenSenders = {V2}`

2. Calculate weights:
   - `sendersWeight = stake(V2) = 150`
   - `validatorOwnStake = stake(V1) = 100`
   - `otherValidatorsWeight = 300 - 100 = 200`

3. Calculate synchrony value:
   - `synchronyConstraintValue = 150 / 200 = 0.75`

4. Check threshold:
   - `0.75 >= 0.67` → **TRUE** ✓
   - **Result**: V1 is allowed to propose

**Interpretation**: V1 has seen 75% of other validators' stake produce new blocks, exceeding the 67% threshold.

### Example 2: Same Network, V1 Blocked

**Scenario**: Neither V2 nor V3 have produced blocks since V1's last proposal

**Analysis**:
1. Seen senders: `seenSenders = {}` (empty)
2. Weights:
   - `sendersWeight = 0`
   - `otherValidatorsWeight = 200`
3. Synchrony value: `0 / 200 = 0.0`
4. Check: `0.0 >= 0.67` → **FALSE** ✗
5. **Result**: `NotEnoughNewBlocks` - V1 must wait

**Interpretation**: V1 has not seen any new blocks from other validators, so it cannot propose. This prevents V1 from creating a long chain without coordination.

### Example 3: Two Validators, One Offline

**Setup**:
- Validators: V1 (stake: 100), V2 (stake: 100)
- Threshold: 0.67
- V2 is offline

**Scenario**: V1 proposes B1, then tries to propose B2

**Analysis**:
1. Seen senders: V2 has not produced any blocks → `{}`
2. Weights: `sendersWeight = 0`, `otherValidatorsWeight = 100`
3. Synchrony value: `0 / 100 = 0.0`
4. Check: `0.0 >= 0.67` → **FALSE** ✗
5. **Result**: **Network halts** - V1 cannot propose without V2

**Critical Issue**: With threshold 0.67 and only 2 validators, if one goes offline, the network **loses liveness**. This is why small networks should use threshold 0.0 or 0.5.

### Example 4: Single Validator with Threshold 0.0

**Setup**:
- Validators: V1 (stake: 100)
- Threshold: 0.0 (in genesis)

**Scenario**: V1 proposes blocks continuously

**Analysis**:
1. Seen senders: Only V1 exists, self is excluded → `{}`
2. Weights: `sendersWeight = 0`, `otherValidatorsWeight = 0`
3. Special case triggered (line 79):
   - `synchronyConstraintValue = 1.0` (because otherValidatorsWeight == 0)
4. Check: `1.0 >= 0.0` → **TRUE** ✓
5. **Result**: V1 can always propose

**Success Case**: The special case logic correctly handles single validators.

### Example 5: Weighted Stake Scenario

**Setup**:
- V1 (stake: 10), V2 (stake: 500), V3 (stake: 40), V4 (stake: 50)
- Total: 600
- Threshold: 0.67

**Scenario**: V1 wants to propose

**Analysis**:
- Other validators' stake: 600 - 10 = 590
- Need to see: 590 * 0.67 = 395.3 in stake

**Case A**: Only V2 has produced a new block
- `sendersWeight = 500`
- `synchronyConstraintValue = 500 / 590 = 0.847`
- `0.847 >= 0.67` → **TRUE** ✓
- **Result**: V1 can propose (V2's large stake is sufficient)

**Case B**: Only V3 and V4 have produced new blocks
- `sendersWeight = 40 + 50 = 90`
- `synchronyConstraintValue = 90 / 590 = 0.153`
- `0.153 >= 0.67` → **FALSE** ✗
- **Result**: V1 cannot propose (insufficient stake seen)

**Interpretation**: The constraint is **stake-weighted**, not validator-count-weighted. A single validator with high stake can satisfy the constraint.

## Troubleshooting

### Symptoms: "Proposal failed: Must wait for more blocks from other validators"

**Check 1: What's the actual threshold?**

Look in the logs for the line (around line 82-85):
```
Seen X senders with weight Y out of total Z (A out of B needed)
```

Where:
- A = `synchronyConstraintValue` (calculated)
- B = `synchronyConstraintThreshold` (configured)

**Check 2: Is threshold properly set in genesis?**

The threshold is locked into the genesis block. Check your genesis creation:

```bash
# Check configuration used for genesis
cat ~/.rnode/genesis/rnode.conf | grep synchrony-constraint-threshold
```

**Check 3: How many active validators?**

The constraint only considers **active validators** from the PoS contract:

```bash
# Query active validators via gRPC
grpcurl -plaintext -d '{"depth": 1}' localhost:40401 casper.v1.ProposeService.getBlocks
```

### Fix: Recreate Genesis with Correct Threshold

If your genesis was created with the wrong threshold:

1. **Stop all nodes**
2. **Delete data directories**: `rm -rf ~/.rnode/`
3. **Update configuration** (before genesis):
   ```hocon
   casper {
     synchrony-constraint-threshold = 0.0  # or desired value
   }
   ```
4. **Recreate genesis**: Start bootstrap node first to create new genesis
5. **Restart network**: Nodes will use new genesis with correct threshold

### Workaround: Ensure Multiple Validators Propose

If you cannot recreate genesis, ensure validator activity:

1. **In development**: Set up automated proposing on multiple validators
2. **Monitor**: Check that validators are producing blocks regularly
3. **Alert**: Set up monitoring for validators falling behind
4. **Restart**: If a validator gets stuck, check network connectivity to others

## Future Improvements

### Potential Enhancements

1. **Dynamic Threshold Adjustment**
   - Allow threshold to be adjusted via governance/voting
   - Adapt threshold based on network size and health

2. **Liveness vs. Safety Trade-off**
   - Implement degraded mode: if no blocks seen for N minutes, reduce threshold temporarily
   - Balance Byzantine fault tolerance with liveness guarantees

3. **Better Single-Validator Handling**
   - Auto-detect single-validator scenarios
   - Provide clear warnings in logs
   - Bypass constraint entirely if only one active validator

4. **Configuration Validation**
   - Warn at startup if threshold seems too high for network size
   - Recommend threshold based on number of bonded validators
   - Validate configuration before genesis creation

5. **Monitoring and Observability**
   - Expose synchrony constraint metrics (Prometheus/Grafana)
   - Log detailed information about which validators are seen/not seen
   - Create alerts for validators approaching constraint limits

## Summary

The Casper synchrony constraint is a critical safety mechanism that:

- **Enforces network synchrony** by requiring validators to see recent blocks from others before proposing
- **Prevents isolation attacks** where validators create long chains alone
- **Implements Byzantine fault tolerance** through stake-weighted participation thresholds
- **Uses the formula**: `synchronyConstraintValue = sendersWeight / otherValidatorsWeight`
- **Compares against threshold**: Must have `value >= threshold` to propose

**Key Takeaways**:

1. **Threshold is locked in at genesis time** - changing config files later has no effect
2. **Default 0.67 is for BFT in large networks** - too high for small networks
3. **Small networks should use 0.0 or 0.5** - to prevent liveness failures
4. **Single validators need special case** - handled by code but requires threshold 0.0 in genesis
5. **Configuration must be correct before genesis creation** - cannot be changed after without recreating network

**For Developers**:
- Always set `synchrony-constraint-threshold = 0.0` for standalone/single-validator development
- Delete data directory and recreate genesis if threshold was wrong
- Monitor validator activity to ensure constraint doesn't cause stalls
- Consider implementing heartbeat or auto-propose mechanisms for test networks

**For Production**:
- Use default 0.67 for networks with 10+ validators
- Monitor that at least 67% of stake is actively proposing
- Set up alerting for validators falling behind
- Plan maintenance windows to avoid dropping below threshold
