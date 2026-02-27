# Casper Consensus Protocol Design

## About this Document

This is a Claude Code generated document based on a task to examine the codebase and produce a summary of the Casper algorithm. **It is not authoritative documentation; This document is intended to be used as context for AI coding assistants.** If you make changes to casper, please have your coding assistant update this file, so that others can benefit from improved context.

## Overview

Casper is RChain's proof-of-stake consensus protocol implementation that uses a multi-parent DAG (Directed Acyclic Graph) structure rather than a traditional single-parent blockchain. This design enables parallel block production and higher throughput while maintaining Byzantine fault tolerance.

## Core Architecture

### Multi-Parent DAG Structure

Unlike traditional blockchains where each block has a single parent, Casper blocks can reference multiple parents through "justifications". This creates a DAG structure that:

- Allows parallel block production by different validators
- Enables concurrent execution paths that can later merge
- Maintains partial ordering while supporting high throughput
- Preserves all relationships even after finalization (no linearization)

### Key Components

1. **MultiParentCasper** (`MultiParentCasperImpl.scala`) - Main consensus implementation handling:
   - Block creation and validation
   - Deploy (transaction) processing
   - Fork choice through the Estimator
   - Finalization using safety oracles

2. **Engine** (`engine/` directory) - State machine managing consensus lifecycle:
   - `Initializing` → `Running` states
   - Handles block synchronization and initial bootstrap
   - Manages transition from genesis to active consensus

3. **Safety Oracle** (`safety/` directory) - Implements clique-based finalization:
   - Detects when sufficient validators agree on blocks
   - Determines finalization based on fault tolerance thresholds
   - Tracks finalized block advancement through the DAG

4. **Estimator** - Fork choice rule implementation:
   - LMD GHOST (Latest Message Driven - Greedy Heaviest Observed SubTree)
   - Determines main chain and parent selection
   - Handles tie-breaking in fork scenarios

## Block Production Protocol

### Triggering Mechanisms

Casper uses a **manual/triggered block production model** rather than automatic scheduling:

1. **Manual API Trigger** - Blocks produced when explicitly requested via:
   - `BlockAPI.createBlock` endpoint
   - gRPC `ProposeService` interface

2. **Deploy-triggered Proposing** - Optional automatic proposal after deploy submission:
   - Configured via `triggerPropose` parameter
   - Enables responsive block production based on transaction flow

### Proposal Process

The `Proposer` class orchestrates block creation:

1. **Constraint Checking**:
   - Verify validator is bonded and active
   - Check synchrony constraints (sufficient stake has seen recent blocks)
   - Validate finalized height constraints

2. **Block Assembly**:
   - Collect pending deploys from storage
   - Add slashing deploys for equivocators
   - Include system deploys (e.g., CloseBlockDeploy)
   - Return `NoNewDeploys` if nothing to include (no empty blocks)

3. **Validation**:
   - Self-validate the created block before broadcasting
   - Ensures block follows all consensus rules

4. **Distribution**:
   - Store block locally
   - Update Casper state
   - Broadcast block hash to peers
   - Publish events for monitoring

## Finalization Mechanism

### Safety-Based Finalization

Finalization in Casper **preserves the DAG structure** - blocks remain in their multi-parent relationships even after finalization:

1. **Clique Detection**: The safety oracle identifies when enough validators (based on stake weight) have built upon a block

2. **Fault Tolerance Threshold**: Blocks become finalized when they have support exceeding the configured fault tolerance (typically 2/3 of stake)

3. **Last Finalized Block (LFB)**: Tracks the frontier of finalized blocks advancing through the DAG

4. **Concurrent Finalization**: Multiple blocks at the same height can be finalized if on compatible branches

### Finalization Effects

When blocks are finalized:
- Deploys are removed from pending storage
- Block indexes are cleared from cache
- Mergeable channel data is cleaned up
- Events are published for monitoring
- **DAG structure is preserved** (no linearization occurs)

## State Management and RSpace Integration

### RuntimeManager Interface

Casper interacts with RSpace (Rholang's tuple space storage) through the `RuntimeManager`:

```scala
trait RuntimeManager[F[_]] {
  def computeState(startHash: StateHash)(
    terms: Seq[Signed[DeployData]],
    systemDeploys: Seq[SystemDeploy],
    blockData: BlockData,
    invalidBlocks: Map[BlockHash, Validator]
  ): F[(StateHash, Seq[ProcessedDeploy], Seq[ProcessedSystemDeploy])]
  
  def replayComputeState(startHash: StateHash)(...)
  def getData(hash: StateHash)(channel: Par): F[Seq[Par]]
  def computeBonds(startHash: StateHash): F[Seq[Bond]]
  def getActiveValidators(startHash: StateHash): F[Seq[Validator]]
}
```

### State Hash Management

- Each block contains a `postStateHash` (Blake2b256 hash of RSpace trie root)
- State transitions: `preStateHash` → execute deploys → `postStateHash`
- History repository maintains all historical states as immutable tries

### Execution Flow

1. Load parent state from RSpace history
2. Execute deploys sequentially in RSpace runtime
3. Track state changes and storage operations
4. Checkpoint to obtain new state hash
5. Store mergeable channel data for conflict resolution
6. Clean up finalized data

## Conflict Resolution and Merging

### Handling Parallel Branches

When parallel blocks with potentially conflicting deploys converge, Casper employs sophisticated merging:

### Conflict Detection (`DagMerger.scala`)

Conflicts are detected when deployments:
- Share the same deploy ID (duplicates)
- Access the same channels in incompatible ways
- Violate numeric channel constraints (e.g., insufficient balance)

### Resolution Algorithm (`ConflictSetMerger.scala`)

The merger uses a **cost-optimal rejection algorithm**:

1. **Branch Formation**: Group deployments into branches without cross-dependencies
2. **Conflict Mapping**: Identify conflicts between branches
3. **Rejection Options**: Calculate different ways to resolve conflicts
4. **Optimal Selection**: Choose resolution minimizing total cost of rejected deploys
5. **State Merging**: Combine non-conflicting changes into new merged state

### Mergeable Channels

Special handling for numeric channels (REV balances):
- Changes tracked as diffs rather than absolute values
- Stored separately in `MergeableStore`
- Enables parallel balance modifications
- Overflow checking prevents invalid states

### Determinism Guarantees

The rejection process ensures deterministic outcomes through:
- Cost-based selection (minimize rejected deploy costs)
- Tie-breaking rules (branch size, lexicographic ordering)
- Dependency tracking (cascade rejection of dependent deploys)

## Consensus Properties

### Safety Properties

- **Agreement**: Honest validators cannot finalize conflicting blocks
- **Finality**: Finalized blocks cannot be reverted
- **Slashing**: Equivocators are detected and punished

### Liveness Properties

- **Chain Growth**: New blocks can be produced given honest majority
- **Finalization Progress**: LFB advances under synchrony assumptions
- **Deploy Inclusion**: Valid deploys eventually included (within lifespan)

### Performance Features

- **Parallel Execution**: Multiple validators produce blocks concurrently
- **Efficient Merging**: Conflicts resolved at convergence points
- **No Empty Blocks**: Blocks only created when deploys available
- **Partial Ordering**: Only conflicting operations require ordering

## Testing Infrastructure

The `casper` module includes comprehensive testing:

### Test Organization

- **103 test files** organized for parallel execution
- `batch1/` and `batch2/` - Main test suites grouped by execution time
- `slowcooker/` - Long-running generative property-based tests
- Component-specific test directories for focused testing

### Test Framework Features

- `TestNode` - Complete test node with mocked transport
- Network simulation for multi-node scenarios
- Effect-based testing using cats-effect
- Property-based testing for consensus properties
- Resource management and cleanup

### Test Coverage

- Consensus safety and liveness
- Finalization progression
- Bonding and staking mechanisms
- Deploy lifecycle management
- Network communication protocols
- Conflict resolution and merging
- State management and queries

## Configuration

Key parameters in `CasperShardConf`:

- **Fault Tolerance Threshold** - Stake weight required for finalization
- **Deploy Lifespan** - Blocks until deploy expiration
- **Min/Max Phlo Price** - Transaction fee bounds
- **Bond Min/Max** - Staking limits
- **Parent Depth Limit** - Maximum DAG depth to consider
- **Finalization Rate** - How often to run finalization algorithm

## Integration Points

### External Dependencies

- **RSpace** - Tuple space for state storage
- **Rholang** - Smart contract execution environment
- **Block Storage** - Persistent block and DAG storage
- **Transport Layer** - P2P network communication
- **Cryptography** - Blake2b256 hashing, Secp256k1 signatures

### APIs and Interfaces

- gRPC services for block proposals and queries
- HTTP API for deploy submission
- Event streaming for monitoring
- Metrics collection for observability

## Design Rationale

### Why DAG over Chain?

- **Higher Throughput**: Parallel block production
- **Reduced Latency**: No waiting for single proposer
- **Efficient Finalization**: Concurrent finalization of compatible blocks
- **Natural Sharding**: DAG structure supports cross-shard references

### Why Manual Block Production?

- **Explicit Control**: Operators decide when to propose
- **Testing Friendly**: Deterministic test scenarios
- **Resource Efficiency**: No wasted empty blocks
- **Flexible Deployment**: Adaptable to different network topologies

### Why Cost-Based Rejection?

- **Economic Incentives**: Users motivated to avoid conflicts
- **Deterministic**: Same rejection across all nodes
- **Fair**: Minimizes total economic loss
- **Efficient**: Optimizes for maximum deploy inclusion

## Future Considerations

Potential areas for enhancement:

- Automatic block production scheduling
- Cross-shard communication protocols  
- Light client support with DAG proofs
- Enhanced merge algorithms for complex conflicts
- Dynamic validator set management
- Improved state synchronization methods
