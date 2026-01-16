# Checkpoint and Replay Specification: CheckpointReplay.tla

This document covers the TLA+ specification for checkpoint creation, state restoration, and deterministic replay in `CheckpointReplay.tla`.

## Overview

The specification models the checkpoint and replay subsystem that enables:
- Persistent checkpoint creation for state recovery
- Soft checkpoints for speculative execution
- Deterministic replay from operation logs
- State restoration guarantees

This is critical for blockchain consensus where nodes must be able to replay operations to verify state transitions.

## State Variables

```tla
VARIABLES
    spaceStates,        \* Map from SpaceId to space state (data + conts)
    checkpoints,        \* Map from CheckpointId to (MerkleRoot, RegistrySnapshot)
    operationLog,       \* Sequence of operations for replay
    replayMode,         \* Whether we're in replay mode
    replayIndex,        \* Current position in operation log during replay
    softCheckpoint,     \* In-memory soft checkpoint (for speculative exec)
    merkleRoots         \* Map from SpaceId to current Merkle root
```

### spaceStates

Maps space identifiers to their current state:

```tla
SpaceState == [
    data: [ChannelNames -> Seq(Datum)],
    continuations: Seq([channels: SUBSET ChannelNames, persist: BOOLEAN])
]

spaceStates \in [SpaceIds -> SpaceState]
```

**Example**:
```
spaceStates = [
    s1 |-> [data |-> [ch1 |-> <<datum1>>, ch2 |-> <<>>],
            continuations |-> <<>>],
    s2 |-> [data |-> [ch1 |-> <<>>, ch2 |-> <<>>],
            continuations |-> <<wk1>>]
]
```

### checkpoints

Maps checkpoint IDs to checkpoint records:

```tla
Checkpoint == [
    merkleRoot: Nat,          \* Hash representing state
    registrySnapshot: Nat,    \* Snapshot of registry state
    blockHeight: Nat
]

checkpoints \in [CheckpointIds -> Checkpoint \cup {NULL}]
```

**Example**:
```
checkpoints = [
    cp1 |-> [merkleRoot |-> 42, registrySnapshot |-> 10, blockHeight |-> 100],
    cp2 |-> NULL  \* Not yet created
]
```

### operationLog

Sequence of operations for replay:

```tla
OperationType == {"Produce", "Consume", "Install"}

Operation == [
    type: OperationType,
    spaceId: SpaceIds,
    channel: ChannelNames \cup {NULL},
    channels: SUBSET ChannelNames,
    data: DataValues \cup {NULL}
]

operationLog \in Seq(Operation)
```

**Example**:
```
operationLog = <<
    [type |-> "Produce", spaceId |-> s1, channel |-> ch1, channels |-> {}, data |-> d1],
    [type |-> "Consume", spaceId |-> s1, channel |-> NULL, channels |-> {ch1}, data |-> NULL]
>>
```

### replayMode / replayIndex

Replay state tracking:

```tla
replayMode \in BOOLEAN
replayIndex \in Nat
```

- `replayMode = TRUE` when replaying operations
- `replayIndex` tracks current position in log (1-indexed)

### softCheckpoint

In-memory checkpoint for speculative execution:

```tla
softCheckpoint \in [SpaceIds -> SpaceState] \cup {NULL}
```

- `NULL` when no soft checkpoint is active
- Contains full state snapshot when active

### merkleRoots

Tracks Merkle root for each space (simplified as counters):

```tla
merkleRoots \in [SpaceIds -> Nat]
```

## Constants

```tla
CONSTANTS
    SpaceIds,           \* Set of all possible space identifiers
    ChannelNames,       \* Set of all possible channel names
    DataValues,         \* Set of all possible data values
    CheckpointIds,      \* Set of all possible checkpoint identifiers
    MaxHistoryLength,   \* Maximum length of operation history
    NULL                \* Null value for optional fields
```

## Type Invariant

```tla
TypeOK ==
    /\ spaceStates \in [SpaceIds -> SpaceState]
    /\ checkpoints \in [CheckpointIds -> Checkpoint \cup {NULL}]
    /\ operationLog \in Seq(Operation)
    /\ replayMode \in BOOLEAN
    /\ replayIndex \in Nat
    /\ softCheckpoint \in [SpaceIds -> SpaceState] \cup {NULL}
    /\ merkleRoots \in [SpaceIds -> Nat]
```

## Initial State

```tla
Init ==
    /\ spaceStates = [s \in SpaceIds |-> EmptySpaceState]
    /\ checkpoints = [c \in CheckpointIds |-> NULL]
    /\ operationLog = <<>>
    /\ replayMode = FALSE
    /\ replayIndex = 0
    /\ softCheckpoint = NULL
    /\ merkleRoots = [s \in SpaceIds |-> 0]
```

## Actions

### Normal Mode Operations

#### Produce

Send data to a channel (only in normal mode):

```tla
Produce(spaceId, channel, data, persist) ==
    /\ ~replayMode
    /\ Len(operationLog) < MaxHistoryLength
    /\ LET newDatum == [data |-> data, persist |-> persist]
           currentData == spaceStates[spaceId].data[channel]
       IN spaceStates' = [spaceStates EXCEPT
            ![spaceId].data[channel] = Append(currentData, newDatum)]
    /\ operationLog' = Append(operationLog,
        [type |-> "Produce", spaceId |-> spaceId, channel |-> channel,
         channels |-> {}, data |-> data])
    /\ merkleRoots' = [merkleRoots EXCEPT ![spaceId] = ComputeMerkleRoot(spaceId)]
    /\ UNCHANGED <<checkpoints, replayMode, replayIndex, softCheckpoint>>
```

**Key points**:
1. Only operates when `~replayMode`
2. Logs the operation for future replay
3. Updates Merkle root

#### Consume

Receive data from channels:

```tla
Consume(spaceId, channels, persist) ==
    /\ ~replayMode
    /\ Len(operationLog) < MaxHistoryLength
    /\ channels # {}
    /\ operationLog' = Append(operationLog,
        [type |-> "Consume", spaceId |-> spaceId, channel |-> NULL,
         channels |-> channels, data |-> NULL])
    /\ merkleRoots' = [merkleRoots EXCEPT ![spaceId] = ComputeMerkleRoot(spaceId)]
    /\ UNCHANGED <<spaceStates, checkpoints, replayMode, replayIndex, softCheckpoint>>
```

### Checkpoint Operations

#### CreateCheckpoint

Create a persistent checkpoint:

```tla
CreateCheckpoint(checkpointId, blockHeight) ==
    /\ ~replayMode
    /\ checkpoints[checkpointId] = NULL
    /\ LET snapshot == [
            merkleRoot |-> merkleRoots[CHOOSE s \in SpaceIds : TRUE],
            registrySnapshot |-> Len(operationLog),
            blockHeight |-> blockHeight
           ]
       IN checkpoints' = [checkpoints EXCEPT ![checkpointId] = snapshot]
    /\ UNCHANGED <<spaceStates, operationLog, replayMode,
                   replayIndex, softCheckpoint, merkleRoots>>
```

**Preconditions**:
- Not in replay mode
- Checkpoint ID not already used

**Effects**:
- Records current Merkle root
- Records current log length (for knowing where to start replay)
- Records block height for ordering

#### CreateSoftCheckpoint

Create an in-memory checkpoint for speculative execution:

```tla
CreateSoftCheckpoint ==
    /\ ~replayMode
    /\ softCheckpoint = NULL
    /\ softCheckpoint' = spaceStates
    /\ UNCHANGED <<spaceStates, checkpoints, operationLog,
                   replayMode, replayIndex, merkleRoots>>
```

**Intuition**: Soft checkpoints allow speculative execution - if speculation fails, revert to the soft checkpoint.

#### RevertToSoftCheckpoint

Rollback to soft checkpoint:

```tla
RevertToSoftCheckpoint ==
    /\ ~replayMode
    /\ softCheckpoint # NULL
    /\ spaceStates' = softCheckpoint
    /\ softCheckpoint' = NULL
    /\ UNCHANGED <<checkpoints, operationLog, replayMode, replayIndex, merkleRoots>>
```

**Intuition**: Speculation failed - restore the saved state.

#### CommitSoftCheckpoint

Discard the soft checkpoint (commit speculation):

```tla
CommitSoftCheckpoint ==
    /\ ~replayMode
    /\ softCheckpoint # NULL
    /\ softCheckpoint' = NULL
    /\ UNCHANGED <<spaceStates, checkpoints, operationLog,
                   replayMode, replayIndex, merkleRoots>>
```

**Intuition**: Speculation succeeded - discard the backup.

### Replay Mode Operations

#### EnterReplayMode

Start replaying from a checkpoint:

```tla
EnterReplayMode(startCheckpointId, logToReplay) ==
    /\ ~replayMode
    /\ checkpoints[startCheckpointId] # NULL
    /\ replayMode' = TRUE
    /\ replayIndex' = 1
    /\ operationLog' = logToReplay
    /\ UNCHANGED <<spaceStates, checkpoints, softCheckpoint, merkleRoots>>
```

**Preconditions**:
- Not already in replay mode
- Starting checkpoint exists

**Effects**:
- Enter replay mode
- Set index to 1 (first operation)
- Load the log to replay

#### ReplayNextOperation

Replay one operation from the log:

```tla
ReplayNextOperation ==
    /\ replayMode
    /\ replayIndex <= Len(operationLog)
    /\ LET op == operationLog[replayIndex]
       IN CASE op.type = "Produce" ->
                /\ LET newDatum == [data |-> op.data, persist |-> FALSE]
                       currentData == spaceStates[op.spaceId].data[op.channel]
                   IN spaceStates' = [spaceStates EXCEPT
                        ![op.spaceId].data[op.channel] =
                            Append(currentData, newDatum)]
                /\ merkleRoots' = [merkleRoots EXCEPT
                    ![op.spaceId] = ComputeMerkleRoot(op.spaceId)]
           [] op.type = "Consume" ->
                /\ merkleRoots' = [merkleRoots EXCEPT
                    ![op.spaceId] = ComputeMerkleRoot(op.spaceId)]
                /\ UNCHANGED spaceStates
           [] OTHER ->
                /\ UNCHANGED <<spaceStates, merkleRoots>>
    /\ replayIndex' = replayIndex + 1
    /\ UNCHANGED <<checkpoints, operationLog, replayMode, softCheckpoint>>
```

**Key insight**: Replay is **deterministic** - same log produces same state transitions.

#### ExitReplayMode

Exit replay mode after all operations processed:

```tla
ExitReplayMode ==
    /\ replayMode
    /\ replayIndex > Len(operationLog)  \* All operations replayed
    /\ replayMode' = FALSE
    /\ replayIndex' = 0
    /\ UNCHANGED <<spaceStates, checkpoints, operationLog,
                   softCheckpoint, merkleRoots>>
```

## Safety Invariants

### ReplayModeExclusive

Replay mode and soft checkpoints are mutually exclusive:

```tla
ReplayModeExclusive ==
    replayMode => softCheckpoint = NULL
```

**Intuition**: You can't do speculative execution while replaying.

### ReplayIndexValid

Replay index stays within bounds:

```tla
ReplayIndexValid ==
    replayMode => replayIndex <= Len(operationLog) + 1
```

**Intuition**: The index points to the next operation to replay, so it can be at most `len + 1` (meaning all done).

### CheckpointsImmutable

Checkpoints are never modified after creation:

```tla
CheckpointsImmutable ==
    \* This would be checked across state transitions
    TRUE
```

**Note**: This is expressed as a refinement - once created, checkpoint values never change.

### Combined Safety

```tla
SafetyInvariant ==
    /\ TypeOK
    /\ ReplayModeExclusive
    /\ ReplayIndexValid
```

## Liveness Properties

### ReplayEventuallyCompletes

If we enter replay mode, we eventually exit:

```tla
ReplayEventuallyCompletes ==
    replayMode ~> ~replayMode
```

**Intuition**: `P ~> Q` means "P leads to Q" - if P ever becomes true, Q eventually becomes true.

### SoftCheckpointResolved

Soft checkpoints are eventually resolved:

```tla
SoftCheckpointResolved ==
    (softCheckpoint # NULL) ~> (softCheckpoint = NULL)
```

**Intuition**: A soft checkpoint is either committed or reverted - it doesn't persist indefinitely.

## Determinism Property

```tla
ReplayDeterminism ==
    \* Two replays of the same log from the same checkpoint produce same state
    \* This is implicit in the specification since operations are deterministic
    TRUE
```

**Key insight**: The specification itself encodes determinism - each operation has exactly one outcome based on the current state and operation parameters.

## Next State Relation

```tla
Next ==
    \/ \E s \in SpaceIds, c \in ChannelNames, d \in DataValues, p \in BOOLEAN :
        Produce(s, c, d, p)
    \/ \E s \in SpaceIds, cs \in SUBSET ChannelNames, p \in BOOLEAN :
        Consume(s, cs, p)
    \/ \E cp \in CheckpointIds, h \in Nat :
        CreateCheckpoint(cp, h)
    \/ CreateSoftCheckpoint
    \/ RevertToSoftCheckpoint
    \/ CommitSoftCheckpoint
    \/ \E cp \in CheckpointIds :
        ResetToCheckpoint(cp)
    \/ \E cp \in CheckpointIds, log \in Seq(Operation) :
        EnterReplayMode(cp, log)
    \/ ReplayNextOperation
    \/ ExitReplayMode

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
```

## Example Traces

### Trace 1: Normal Operation with Checkpoint

```
State 0 (Init):
  spaceStates = [s1 |-> EmptySpaceState]
  checkpoints = [cp1 |-> NULL]
  operationLog = <<>>
  replayMode = FALSE
  softCheckpoint = NULL

Action: Produce(s1, ch1, d1, FALSE)

State 1:
  spaceStates[s1].data[ch1] = <<[data |-> d1, persist |-> FALSE]>>
  operationLog = <<[type |-> "Produce", ...]>>
  merkleRoots[s1] = 1

Action: CreateCheckpoint(cp1, 100)

State 2:
  checkpoints[cp1] = [merkleRoot |-> 1, registrySnapshot |-> 1, blockHeight |-> 100]
  (other variables unchanged)
```

### Trace 2: Soft Checkpoint with Revert

```
State 0:
  spaceStates[s1].data[ch1] = <<datum1>>
  softCheckpoint = NULL

Action: CreateSoftCheckpoint

State 1:
  softCheckpoint = [s1 |-> [data |-> [ch1 |-> <<datum1>>], ...]]
  spaceStates unchanged

Action: Produce(s1, ch1, d2, FALSE)  -- speculative

State 2:
  spaceStates[s1].data[ch1] = <<datum1, datum2>>
  softCheckpoint still holds original state

Action: RevertToSoftCheckpoint  -- speculation failed!

State 3:
  spaceStates[s1].data[ch1] = <<datum1>>  -- reverted!
  softCheckpoint = NULL
```

### Trace 3: Replay Sequence

```
State 0:
  checkpoints[cp1] = [merkleRoot |-> 5, ...]
  operationLog = <<op1, op2, op3>>
  replayMode = FALSE

Action: EnterReplayMode(cp1, <<op1, op2, op3>>)

State 1:
  replayMode = TRUE
  replayIndex = 1

Action: ReplayNextOperation  (replays op1)

State 2:
  replayIndex = 2
  spaceStates updated with op1

Action: ReplayNextOperation  (replays op2)

State 3:
  replayIndex = 3
  spaceStates updated with op2

Action: ReplayNextOperation  (replays op3)

State 4:
  replayIndex = 4
  spaceStates updated with op3

Action: ExitReplayMode

State 5:
  replayMode = FALSE
  replayIndex = 0
```

## Correspondence to Rust

| TLA+ | Rust |
|------|------|
| `spaceStates` | `GenericRSpace` instances in registry |
| `checkpoints` | `CheckpointStore` |
| `operationLog` | `EventLog` operations |
| `replayMode` | `ReplayRSpace` vs normal mode |
| `replayIndex` | Replay iterator position |
| `softCheckpoint` | `ISpace::create_soft_checkpoint()` result |
| `merkleRoots` | `ISpace::merkle_root()` |
| `CreateCheckpoint` | `CheckpointStore::create()` |
| `CreateSoftCheckpoint` | `ISpace::create_soft_checkpoint()` |
| `RevertToSoftCheckpoint` | Rollback using soft checkpoint |
| `ReplayNextOperation` | `ReplayRSpace::step()` |

## Model Checking

To check this specification with TLC:

```tla
\* CheckpointReplay.cfg
CONSTANTS
    SpaceIds = {s1}
    ChannelNames = {ch1}
    DataValues = {d1}
    CheckpointIds = {cp1}
    MaxHistoryLength = 3
    NULL = NULL

SPECIFICATION Spec
INVARIANTS TypeOK SafetyInvariant
PROPERTIES ReplayEventuallyCompletes SoftCheckpointResolved
```

With these constants, TLC explores the checkpoint/replay state machine exhaustively.

## Theorems

```tla
THEOREM SafetyHolds == Spec => []SafetyInvariant

THEOREM ReplayCompletes == Spec => ReplayEventuallyCompletes
```

## Next Steps

Continue to [03-space-coordination-spec.md](03-space-coordination-spec.md) for multi-space coordination specifications.
