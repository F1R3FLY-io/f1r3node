# Byzantine Fault Tolerance in F1R3FLY

## Overview

Byzantine Fault Tolerance (BFT) is the ability of a distributed system to reach consensus despite the presence of malicious actors (Byzantine nodes). Unlike crash faults where nodes simply stop functioning, Byzantine faults include arbitrary malicious behavior: lying, equivocating (sending conflicting messages), or colluding to attack the network.

F1r3fly achieves Byzantine Fault Tolerance through a layered defense architecture that combines prevention, detection, finalization, and punishment mechanisms. This document details each layer and the mathematical guarantees they provide.

---

## Architecture Overview

F1r3fly's BFT implementation consists of four integrated defense layers:

1. **Synchrony Constraint** (Prevention Layer) - Prevents validators from operating in isolation
2. **Equivocation Detection** (Detection Layer) - Identifies conflicting messages from validators
3. **Safety Oracle** (Finalization Layer) - Determines mathematical finality through clique detection
4. **Slashing** (Punishment Layer) - Economically penalizes Byzantine behavior

Each layer is designed to be independent yet complementary, providing defense-in-depth against Byzantine attacks.

---

## 1. Synchrony Constraint - Prevention Layer

### Purpose

The synchrony constraint prevents validators from proposing blocks without sufficient network coordination. This mechanism stops isolation attacks where a malicious validator attempts to create a long private chain or dominate block production.

### Mechanism

Before a validator can propose a new block, they must demonstrate they have seen recent blocks from a sufficient fraction of other validators' stake. This ensures that block production requires active network participation rather than solo operation.

### Implementation

**Formula:**
```scala
synchronyConstraintValue = sendersWeight / otherValidatorsWeight

// Requirement:
synchronyConstraintValue >= synchronyConstraintThreshold
```

Where:
- `sendersWeight` = Total stake of validators who have produced new blocks since proposer's last block
- `otherValidatorsWeight` = Total stake of all validators excluding the proposer
- `synchronyConstraintThreshold` = Configured threshold (default: 0.67)

**Code Location:** `casper/src/main/scala/coop/rchain/casper/SynchronyConstraintChecker.scala`

### Example

Consider a network with three validators:
- V1 (stake: 100), V2 (stake: 150), V3 (stake: 50)
- Total stake: 300
- Threshold: 0.67

When V1 wants to propose:
- Other validators' stake: 300 - 100 = 200
- V2 has produced new blocks (weight: 150)
- V3 has not produced new blocks

Calculation:
```
synchronyConstraintValue = 150 / 200 = 0.75
0.75 >= 0.67 → ALLOWED ✓
```

If neither V2 nor V3 had produced blocks:
```
synchronyConstraintValue = 0 / 200 = 0.0
0.0 >= 0.67 → BLOCKED ✗
```

### Attack Prevention

**Without Synchrony Constraint:**
- Malicious validator could create 100+ blocks in isolation
- Private chain attacks become trivial
- Network could partition without detection

**With Synchrony Constraint:**
- Validator blocked after first block without network participation
- Must coordinate with ≥67% of other validators' stake
- Isolation attacks mathematically prevented

### Configuration

```hocon
casper {
  synchrony-constraint-threshold = 0.67  // 2/3 majority (BFT threshold)
}
```

**Recommended Values:**
- **Large networks (10+ validators):** 0.67 (default, BFT optimal)
- **Medium networks (5-10 validators):** 0.5 (simple majority)
- **Small networks (2-4 validators):** 0.0-0.5 (prevent liveness issues)
- **Single validator (development):** 0.0 (no constraint needed)

**Note:** The threshold value is locked into the genesis block and cannot be changed without recreating the network.

---

## 2. Equivocation Detection - Detection Layer

### What is Equivocation?

Equivocation is the creation of two conflicting blocks at the same sequence number by a single validator. This is the fundamental Byzantine fault in blockchain consensus.

**Example:**
```
        Block A (seq: 5) "Alice has 100 REV"
       /
Validator V1
       \
        Block B (seq: 5) "Alice has 0 REV"  ← EQUIVOCATION
```

### Simple Equivocation Detection

The system detects equivocations by comparing a block's creator justification with the known latest message from that validator.

**Implementation:**
```scala
maybeLatestMessageOfCreatorHash <- dag.latestMessageHash(block.sender)
maybeCreatorJustification       = creatorJustificationHash(block)
isNotEquivocation               = maybeCreatorJustification == maybeLatestMessageOfCreatorHash
```

**Code Location:** `casper/src/main/scala/coop/rchain/casper/EquivocationDetector.scala:32-34`

### Equivocation Types

**1. Admissible Equivocation**
- Needed as a dependency by another block in the DAG
- Added to the DAG but marked as invalid
- Recorded in the equivocation tracker for future detection
- Enables neglected equivocation detection

**2. Ignorable Equivocation**
- Not needed as a dependency by any block
- Rejected immediately without adding to DAG
- Logged but no further action required

**3. Neglected Equivocation**
- Validator had sufficient evidence to detect equivocation but failed to slash
- Most insidious form - represents collusion between validators
- Both the equivocator and the validator who ignored it are punished

### Neglected Equivocation Detection

#### The Problem

Byzantine validators could collude by agreeing to ignore each other's equivocations. Without detection of this behavior, groups of validators could cooperate to attack the network without consequences.

#### The Algorithm

The system maintains an `EquivocationRecord` for each detected equivocation:

```scala
case class EquivocationRecord(
  equivocator: Validator,
  equivocationBaseBlockSeqNum: SequenceNumber,
  equivocationDetectedBlockHashes: Set[BlockHash]
)
```

When processing a new block, the system checks three conditions:

**Case 1: Block has sufficient information + contains equivocator in justification**
- Validator saw the equivocation but ignored it
- **Action:** Slash the block creator for neglecting equivocation

**Case 2: Block has sufficient information + properly excluded equivocator**
- Validator correctly handled the equivocation
- **Action:** Update equivocation record, no penalty

**Case 3: Block lacks sufficient information**
- Validator couldn't have known about equivocation
- **Action:** No action taken

**Code Location:** `models/src/main/scala/coop/rchain/models/EquivocationRecord.scala:20-24`

#### Example Scenario

```
Timeline:
1. Validator V1 equivocates (creates Block A and Block B at seq 5)
2. Validator V2 receives both blocks (has complete evidence)
3. V2 proposes Block C with V1 in justifications (should have excluded V1)
4. Network detects V2 had sufficient information to slash V1 but didn't
5. Result: V2 is slashed for neglecting the equivocation
```

This mechanism creates strong incentives against collusion: helping Byzantine validators escape detection results in punishment.

#### Determining "Sufficient Information"

The algorithm determines if a validator had enough information to detect an equivocation by:

1. Examining the validator's justifications
2. Following the justification chain to accumulate evidence
3. Checking if multiple conflicting blocks are reachable
4. Verifying if stake thresholds make the equivocation detectable

If the validator's justification set contains or can reach both equivocating blocks, they had sufficient information and must have properly handled it.

---

## 3. Safety Oracle - Finalization Layer

### Purpose

The Safety Oracle determines when blocks achieve mathematical finality using clique detection algorithms. This provides deterministic finality guarantees rather than probabilistic confirmation.

### Theoretical Foundation

The implementation is based on Ethereum's CBC Casper research and the clique oracle algorithm.

**Core Principle:**
> "If validators in a clique see each other agreeing on block B and cannot see each other disagreeing on B, and the clique has more than half of the validators by weight, then no messages external to the clique can raise the scores these validators assign to a competing candidate to be higher than the score they assign to B."

**Source:** [Casper the Friendly Finality Gadget](https://github.com/ethereum/research/blob/master/papers/CasperTFG/CasperTFG.pdf)

**Code Location:** `casper/src/main/scala/coop/rchain/casper/safety/CliqueOracle.scala`

### Algorithm Overview

The clique oracle computes normalized fault tolerance through the following steps:

**Step 1: Identify Agreeing Validators**
```scala
agreeingValidators = validators.filter { v =>
  v.latestMessage.isInMainChain(targetBlock)
}
```

**Step 2: Build Agreement Graph**

Two validators form an edge in the agreement graph if they never see disagreement about the target block:

```scala
edges = for {
  (validatorA, validatorB) <- agreeingValidators.combinations(2)
  if neverEventuallySeeDisagreement(validatorA, validatorB, targetBlock)
  if neverEventuallySeeDisagreement(validatorB, validatorA, targetBlock)
} yield (validatorA, validatorB)
```

**Step 3: Find Maximum Clique**

Compute the maximum clique (largest fully-connected subgraph) weighted by validator stakes:

```scala
maxCliqueWeight = findMaximumCliqueByWeight(edges, stakes)
```

**Step 4: Calculate Normalized Fault Tolerance**

```scala
totalStake = allValidatorStakes.sum
normalizedFaultTolerance = (maxCliqueWeight * 2 - totalStake) / totalStake
```

**Step 5: Determine Finality**

```scala
if (normalizedFaultTolerance >= faultToleranceThreshold) {
  // Block is mathematically finalized
}
```

### Agreement Graph Construction

#### Never Eventually See Disagreement

Two validators A and B are considered to never see disagreement if validator A's view of validator B's justification chain contains no messages that disagree with the target block.

**Implementation:**
```scala
def neverEventuallySeeDisagreement(
  validatorA: Validator,
  validatorB: Validator, 
  targetBlock: BlockHash
): F[Boolean] = {
  // Get B's latest message
  lmB = B.latestMessage
  
  // Get B's message as seen by A (from A's justifications)
  lmAjB = A.latestMessage.justificationFor(B)
  
  // Walk B's self-justification chain from lmB back to lmAjB
  disagreements = selfJustificationChain(lmB, stopAt = lmAjB).filter { msg =>
    !isInMainChain(targetBlock, msg)
  }
  
  // True if no disagreements found
  return disagreements.isEmpty
}
```

**Visualization:**
```
    A    B
    *    *  <- lmB (B's latest message)
     \   *
      \  *
       \ *
        \*
         *  <- lmAjB (B's message as seen by A)
```

The algorithm checks if any of B's self-justifications between `lmB` and `lmAjB` disagree with the target block. If none disagree, then A and B never eventually see disagreement.

### Normalized Fault Tolerance Calculation

**Formula:**
```
normalizedFaultTolerance = (maxCliqueWeight * 2 - totalStake) / totalStake
```

**Range:** -1.0 to 1.0

**Interpretation:**

| Value | Meaning |
|-------|---------|
| 1.0   | All stake agrees, perfect consensus |
| 0.67  | 5/6 of stake in agreement (default threshold) |
| 0.5   | 3/4 of stake in agreement |
| 0.0   | Exactly 50% stake in agreement |
| -1.0  | No clique >50% stake, block potentially orphaned |

**Example Calculation:**

Given:
- Total stake: 600
- Maximum clique weight: 500

Calculation:
```
normalizedFaultTolerance = (500 * 2 - 600) / 600
                        = (1000 - 600) / 600
                        = 400 / 600
                        = 0.667

If threshold = 0.67 → Block is FINALIZED ✓
```

### Early Finalization Check

As an optimization, if agreeing weight ≤ totalStake / 2, the algorithm immediately returns `-1.0` (minimum fault tolerance) since a block cannot be finalized without majority support:

```scala
if (agreeingWeightMap.values.sum <= totalStake / 2) {
  return SafetyOracle.MIN_FAULT_TOLERANCE  // -1.0
}
```

### Byzantine Fault Tolerance Threshold

The default threshold of **0.67 (2/3)** aligns with the classical Byzantine fault tolerance requirement:

- Assumes up to 1/3 of validators can be Byzantine (malicious or faulty)
- Requires 2/3+ honest validators to finalize
- Provides mathematical safety guarantees against Byzantine attacks

With this threshold:
- Byzantine validators control < 1/3 stake → Cannot prevent finalization
- Byzantine validators control < 1/3 stake → Cannot create conflicting finalized blocks

---

## 4. Slashing - Punishment Layer

### Purpose

Slashing provides economic penalties for Byzantine behavior. Validators who equivocate or neglect to report equivocations forfeit their staked tokens, making attacks economically irrational.

### Slashing Mechanism

When an invalid block is detected with equivocation:

```scala
case InvalidBlock.AdmissibleEquivocation =>
  // 1. Create equivocation record
  val equivocationRecord = EquivocationRecord(
    equivocator = block.sender,
    equivocationBaseBlockSeqNum = block.seqNum - 1,
    equivocationDetectedBlockHashes = Set.empty
  )
  
  // 2. Add to equivocation tracker
  tracker.insertEquivocationRecord(equivocationRecord)
  
  // 3. Mark block as invalid in DAG
  BlockDagStorage.insert(block, invalid = true)
  
  // 4. Remove from buffer
  CasperBufferStorage.remove(block.blockHash)
```

**Code Location:** `casper/src/main/scala/coop/rchain/casper/MultiParentCasperImpl.scala:447-471`

### Stake Forfeiture

Validators are removed from the bond map (active validator set) when equivocations are detected. This removal represents the loss of their staked tokens:

```scala
maybeEquivocatingValidatorBond = bonds(block).find(_.validator == equivocatingValidator)

maybeEquivocatingValidatorBond match {
  case Some(Bond(_, stake)) =>
    // Validator still bonded, equivocation detected
    // Will be slashed when properly excluded from future blocks
    
  case None =>
    // Validator removed from bonds - equivocation acknowledged
    // Stake has been forfeited
    EquivocationDetected
}
```

**Code Location:** `casper/src/main/scala/coop/rchain/casper/EquivocationDetector.scala:152-168`

The actual stake forfeiture is enforced through the Proof-of-Stake smart contract deployed on-chain.

**PoS Contract Location:** `casper/src/main/resources/PoS.rhox`

### Two-Level Slashing

F1r3fly implements a two-tier slashing system:

**Level 1: Direct Equivocators**
- Validators who create conflicting blocks
- Lose entire staked amount
- Immediately removed from active validator set

**Level 2: Neglected Equivocators**
- Validators who had evidence of equivocation but failed to properly handle it
- Also lose staked amount (or portion thereof)
- Removed from active validator set

### Incentive Structure

The slashing mechanism creates the following incentive structure:

| Action | Outcome | Incentive |
|--------|---------|-----------|
| Honest validation | Keep stake + earn rewards | ✓ Profitable |
| Create equivocation | Lose entire stake | ✗ Devastating loss |
| Ignore equivocation | Lose entire stake | ✗ Devastating loss |
| Report equivocation | Keep stake + maintain network | ✓ Rational choice |
| Collude with Byzantine validators | Both parties lose stake | ✗ Mutual destruction |

This structure ensures that:
1. **Honest validation is the only profitable strategy**
2. **Byzantine behavior results in financial loss**
3. **Collusion is economically irrational** (both parties punished)
4. **Network security is economically incentivized**

---

## 5. Byzantine Fault Tolerance Guarantees

### Safety Properties

F1r3fly provides the following safety guarantees under the assumption that Byzantine validators control less than 1/3 of total stake:

**1. No Conflicting Finality**
- Honest validators cannot finalize conflicting blocks
- Requires >2/3 stake to achieve finality
- If Byzantine validators <1/3, impossible to create conflicting finalized blocks
- Mathematical guarantee, not probabilistic

**2. No Silent Corruption**
- All equivocations are eventually detected
- Neglected equivocations punish colluding validators  
- Detection mechanisms operate continuously
- No Byzantine behavior can remain hidden indefinitely

**3. No Isolation Attacks**
- Synchrony constraint prevents solo chain creation
- Must coordinate with ≥67% of network stake
- Cannot create long private forks
- Network partition attacks detected and prevented

**4. Finality is Irreversible**
- Once a block achieves finality threshold, it cannot be reverted
- Safety oracle computation provides mathematical proof
- No probabilistic risk of reversal (unlike longest-chain rules)

### Liveness Properties

Under network synchrony assumptions and honest majority, F1r3fly guarantees:

**1. Chain Growth**
- New blocks will be produced
- Block production rate depends on validator coordination
- No deadlock conditions under honest majority

**2. Finalization Progress**
- Last Finalized Block (LFB) advances over time
- As validators coordinate, cliques form
- Safety oracle automatically detects and advances finality
- No manual intervention required

**3. Transaction Inclusion**
- Valid transactions eventually included in blocks
- Invalid transactions rejected deterministically
- No censorship possible with honest majority
- Deploy lifespan prevents indefinite pending

### Mathematical Guarantees

F1r3fly provides **formal mathematical guarantees** under the following assumptions:

**Assumptions:**
1. Byzantine validators control < 1/3 of total stake
2. Network is eventually synchronous (messages delivered within bounded time)
3. Cryptographic assumptions hold (Ed25519 signatures are secure)
4. Honest validators follow protocol

**Guarantees:**
```
P(conflicting finalized blocks) = 0      (impossible)
P(equivocation detected | occurs) = 1    (always caught)
P(finality reversal | finalized) = 0     (impossible)
P(liveness | honest majority) = 1        (guaranteed progress)
```

### Comparison to Other Consensus Mechanisms

#### Bitcoin (Longest Chain / Nakamoto Consensus)

| Property | Bitcoin | F1r3fly |
|----------|---------|---------|
| Detection | Implicit (longer chain wins) | Explicit equivocation detection |
| Prevention | PoW makes forks expensive | Synchrony constraint + clique oracle |
| Finality | Probabilistic (never 100%) | Mathematical (deterministic) |
| BFT | No (vulnerable to 51% attack) | Yes (tolerates <1/3 Byzantine) |
| Equivocation Cost | PoW resources wasted | Entire stake forfeited |
| Finality Time | ~60 minutes (6 confirmations) | Variable, typically <5 minutes |

#### Ethereum 2.0 (Casper FFG)

| Property | Ethereum 2.0 | F1r3fly |
|----------|--------------|---------|
| Detection | Explicit slashing conditions | Explicit + neglected equivocation |
| Prevention | Attestation rules + checkpoints | Synchrony constraint + clique oracle |
| Finality | Economic (2 epochs, ~13 min) | Mathematical (per-block via oracle) |
| BFT | Yes (<1/3 Byzantine) | Yes (<1/3 Byzantine) |
| Structure | Single-parent chain + committees | Multi-parent DAG |
| Equivocation Cost | Partial stake (depends on type) | Entire stake forfeited |
| Throughput | Limited by slot times | Parallel block production in DAG |

#### Tendermint / PBFT-style Consensus

| Property | Tendermint | F1r3fly |
|----------|------------|---------|
| Detection | Byzantine voting detection | Explicit equivocation tracking |
| Prevention | 2/3 voting threshold | Synchrony + clique oracle |
| Finality | Immediate (within block) | Computed via safety oracle |
| BFT | Yes (<1/3 Byzantine) | Yes (<1/3 Byzantine) |
| Structure | Single-parent chain | Multi-parent DAG |
| Throughput | Sequential (one proposer) | Parallel (multi-parent) |
| Validator Set | Typically smaller (<100) | Scales to larger sets (100+) |

### F1r3fly's Unique Advantages

**1. Parallel Block Production with BFT**
- Most BFT systems use sequential block production (single proposer per round)
- F1r3fly's multi-parent DAG allows concurrent proposals
- Maintains BFT guarantees while enabling parallelism

**2. Multi-Layer Equivocation Detection**
- Direct equivocation detection (standard)
- Neglected equivocation detection (unique to F1r3fly)
- Prevents validator collusion through two-level slashing

**3. Flexible Finality**
- Per-block finality computation (not checkpoint-based)
- Continuous finality advancement as network coordinates
- No waiting for epoch boundaries

**4. Economic Security Model**
- Total stake forfeiture for equivocation
- Slashing for neglecting to report equivocations  
- Strong incentives against collusion
- Economically rational to behave honestly

---

## 6. Implementation Details

### Block Validation Pipeline

Byzantine fault tolerance checks are integrated into the block validation pipeline:

```scala
// Validation sequence in MultiParentCasperImpl
for {
  _ <- Validate.blockSummary(block)
  _ <- Validate.checkpoint(block, dag)  
  _ <- Validate.bondsCache(block)
  _ <- Validate.neglectedInvalidBlock(block, dag)
  
  // BFT-specific validations
  _ <- EquivocationDetector.checkNeglectedEquivocationsWithUpdate(block, dag, genesis)
  _ <- EquivocationDetector.checkEquivocations(depDag, block, dag)
  
  _ <- Validate.phloPrice(block)
  _ <- processBlock(block)
} yield ValidBlock
```

**Code Location:** `casper/src/main/scala/coop/rchain/casper/MultiParentCasperImpl.scala:342-382`

### Equivocation Tracker Storage

The system maintains persistent storage for equivocation records:

```scala
trait EquivocationsTracker[F[_]] {
  def equivocationRecords: F[Set[EquivocationRecord]]
  
  def insertEquivocationRecord(record: EquivocationRecord): F[Unit]
  
  def updateEquivocationRecord(
    record: EquivocationRecord,
    blockHash: BlockHash
  ): F[Unit]
}
```

**Storage Location:** `block-storage/src/main/scala/coop/rchain/blockstorage/dag/EquivocationTrackerStore.scala`

### Safety Oracle Integration

The safety oracle is invoked during:

1. **Block Information Queries** - Computing fault tolerance for block display
2. **Finalization Detection** - Determining when blocks achieve finality
3. **Last Finalized Block Advancement** - Updating the finalization frontier

```scala
def getBlockInfo[F[_]: Monad: SafetyOracle: BlockStore](
  block: BlockMessage
)(implicit casper: MultiParentCasper[F]): F[BlockInfo] = {
  for {
    dag <- casper.blockDag
    normalizedFaultTolerance <- SafetyOracle[F].normalizedFaultTolerance(
      dag, 
      block.blockHash
    )
    initialFault <- casper.normalizedInitialFault(block.weightMap)
    faultTolerance = normalizedFaultTolerance - initialFault
  } yield BlockInfo(block, faultTolerance)
}
```

**Code Location:** `casper/src/main/scala/coop/rchain/casper/api/BlockAPI.scala:570-593`

### Synchrony Constraint Integration

The synchrony constraint is checked during block proposal:

```scala
def checkSynchronyConstraint[F[_]](
  validator: Validator,
  dag: BlockDagRepresentation[F],
  state: CasperState
): F[ProposeResult] = {
  val threshold = state.onChainState.shardConf.synchronyConstraintThreshold
  
  for {
    lastProposed <- dag.latestMessageHash(validator)
    seenSenders <- calculateSeenSendersSince(lastProposed, dag)
    sendersWeight = seenSenders.flatMap(state.weightMap.get).sum
    validatorOwnStake = state.weightMap.getOrElse(validator, 0L)
    otherValidatorsWeight = state.weightMap.values.sum - validatorOwnStake
    
    synchronyValue = if (otherValidatorsWeight == 0) 1.0
                     else sendersWeight.toDouble / otherValidatorsWeight
                     
    result = if (synchronyValue >= threshold) Success
             else NotEnoughNewBlocks(synchronyValue, threshold)
  } yield result
}
```

---

## 7. Testing and Verification

### Test Coverage

The Byzantine fault tolerance implementation includes comprehensive test coverage:

**Test Locations:**
- `casper/src/test/scala/coop/rchain/casper/batch1/` - Core consensus tests
- `casper/src/test/scala/coop/rchain/casper/batch2/` - Extended validation tests  
- `casper/src/test/scala/coop/rchain/casper/batch2/CliqueOracleTest.scala` - Safety oracle tests

### Critical Test Scenarios

**Equivocation Detection Tests:**
- Simple equivocation (conflicting blocks at same sequence)
- Neglected equivocation (validator ignores known equivocation)
- Multiple equivocators
- Equivocation chain detection

**Safety Oracle Tests:**
- Clique formation with varying stake distributions
- Agreement graph construction
- Fault tolerance calculation accuracy
- Edge cases (single validator, equal stakes, network partitions)

**Synchrony Constraint Tests:**
- Single validator networks
- Multi-validator coordination
- Threshold boundary conditions
- Genesis block exceptions

### Property-Based Testing

The implementation includes property-based tests that verify:

1. **Safety Properties:**
   - No conflicting blocks can both be finalized
   - Equivocations are always detected
   - Slashing is deterministic

2. **Liveness Properties:**
   - Block production continues under honest majority
   - Finality advances over time
   - No deadlock conditions

3. **Byzantine Resistance:**
   - <1/3 Byzantine validators cannot prevent finality
   - <1/3 Byzantine validators cannot create conflicting finalized blocks
   - Collusion attempts result in mutual slashing

---

## 8. Configuration and Tuning

### Fault Tolerance Threshold

```hocon
casper {
  fault-tolerance-threshold = 0.0
}
```

The fault tolerance threshold determines how much additional fault tolerance (beyond initial faults) is required for finality.

**Default:** 0.0 (finality as soon as normalized fault tolerance is positive)

**Range:** 0.0 to 1.0

Higher values require stronger consensus before declaring finality.

### Synchrony Constraint Threshold  

```hocon
casper {
  synchrony-constraint-threshold = 0.67
}
```

**Default:** 0.67 (2/3 majority)

**Network Size Recommendations:**
- **Large (10+ validators):** 0.67 (optimal BFT)
- **Medium (5-10 validators):** 0.5 (simple majority)
- **Small (2-4 validators):** 0.0-0.5 (avoid liveness issues)
- **Single validator:** 0.0 (no constraint)

**Important:** This value is locked into the genesis block and cannot be changed without recreating the network from genesis.

### Minimum Phlo Price

```hocon  
casper {
  min-phlo-price = 1
}
```

Validators that accept deploys with prices below this threshold can be slashed. This prevents spam and ensures network sustainability.

---

## 9. Attack Scenarios and Defenses

### Attack: Long-Range Attack

**Description:** Validator with historical keys attempts to create alternative history from old blocks.

**Defense:**
- Synchrony constraint requires recent participation from current validators
- Safety oracle requires cliques from current stake distribution  
- Historical keys have no current stake, cannot form valid cliques
- Attack fails at synchrony constraint check

### Attack: Nothing-at-Stake

**Description:** Validators vote on multiple conflicting branches since there's no cost.

**Defense:**
- Equivocation detection identifies conflicting votes
- Slashing mechanism makes voting on multiple branches costly (lose entire stake)
- Neglected equivocation detection prevents validators from ignoring each other's multiple votes
- Economic cost makes nothing-at-stake irrational

### Attack: Validator Collusion

**Description:** Group of validators coordinate to ignore each other's Byzantine behavior.

**Defense:**
- Neglected equivocation detection specifically targets this attack
- If validator had evidence but didn't slash, they are themselves slashed
- Two-level slashing makes collusion mutually destructive
- Requires >2/3 stake to avoid detection (same as direct Byzantine attack)

### Attack: Censorship

**Description:** Validators refuse to include certain transactions.

**Defense:**
- Multi-parent DAG allows different validators to include transactions in parallel branches
- Honest validators (>2/3) can include censored transactions
- Deploy lifespan prevents indefinite censorship
- Merkle proofs allow detection of censorship

### Attack: DDoS on Validators  

**Description:** Network flooding to prevent validator participation.

**Defense:**
- Synchrony constraint adapts to actual participation levels
- Liveness maintained with >2/3 participation
- Network can finalize blocks despite DDoS on minority validators
- Rate limiting and connection management at network layer

### Attack: Sybil Attack

**Description:** Attacker creates many identities to gain influence.

**Defense:**
- Proof-of-Stake: Influence proportional to stake, not identity count
- Bond requirement: Must stake tokens to become validator
- Economic barrier: Creating many identities requires proportional stake
- Slashing applies per-stake, making Sybil identities expensive

---

## 10. Future Enhancements

### Graduated Slashing

Potential enhancement to implement proportional slashing based on:
- Severity of equivocation
- Whether equivocation was accidental or malicious
- Validator's historical behavior
- Network impact of the Byzantine behavior

### Dynamic Threshold Adjustment

Allow threshold parameters to be adjusted via on-chain governance:
- Synchrony constraint threshold
- Fault tolerance threshold  
- Slashing amounts
- Based on network size and observed behavior

### Optimistic Finality

Add optimistic finality detection:
- Fast finality assumption under normal operation
- Full clique oracle computation only during disputes
- Reduces computational overhead while maintaining security

### Cross-Shard Byzantine Detection

When sharding is implemented:
- Detect Byzantine behavior across shard boundaries
- Cross-shard equivocation detection
- Unified slashing across shards

---

## 11. References

### Academic Papers

1. **CBC Casper:** Zamfir, V. "Casper the Friendly Finality Gadget" - Ethereum Research
   - https://github.com/ethereum/research/blob/master/papers/CasperTFG/CasperTFG.pdf

2. **Safety Oracles:** Ethereum CBC Casper Simulator - Clique Oracle Implementation  
   - https://github.com/ethereum/cbc-casper/blob/0.2.0/casper/safety_oracles/clique_oracle.py

3. **Byzantine Fault Tolerance:** Lamport, L., Shostak, R., Pease, M. "The Byzantine Generals Problem"
   - ACM Transactions on Programming Languages and Systems, 1982

### Implementation References

**Core Implementation Files:**
- `casper/src/main/scala/coop/rchain/casper/EquivocationDetector.scala`
- `casper/src/main/scala/coop/rchain/casper/safety/CliqueOracle.scala`
- `casper/src/main/scala/coop/rchain/casper/SynchronyConstraintChecker.scala`
- `casper/src/main/scala/coop/rchain/casper/SafetyOracle.scala`
- `models/src/main/scala/coop/rchain/models/EquivocationRecord.scala`

**Documentation:**
- `docs/casper/SYNC_CONSTRAINT.md` - Detailed synchrony constraint documentation
- `CLAUDE/casper/DESIGN.md` - Casper consensus protocol design overview

---

## Summary

F1r3fly implements robust Byzantine Fault Tolerance through a multi-layered defense architecture:

1. **Synchrony Constraint** prevents isolation attacks by requiring network coordination
2. **Equivocation Detection** (including neglected equivocation) identifies Byzantine behavior and collusion  
3. **Safety Oracle** provides mathematical finality through clique detection algorithms
4. **Slashing** creates economic disincentives for Byzantine behavior

These mechanisms work together to provide:
- **Mathematical finality** (not probabilistic)
- **Byzantine fault tolerance** (tolerates <1/3 malicious stake)  
- **Collusion resistance** (through multi-layer detection)
- **Economic security** (via comprehensive slashing)

The system achieves Byzantine Fault Tolerance while maintaining the scalability benefits of the multi-parent DAG structure, enabling parallel block production without sacrificing security guarantees.

