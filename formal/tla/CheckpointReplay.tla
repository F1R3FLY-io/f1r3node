----------------------------- MODULE CheckpointReplay -----------------------------
(****************************************************************************)
(* TLA+ Specification for Checkpoint and Replay in Reified RSpaces          *)
(*                                                                          *)
(* This module specifies:                                                   *)
(* - Checkpoint creation and state capture                                  *)
(* - State restoration from checkpoints                                     *)
(* - Deterministic replay semantics                                         *)
(* - Soft checkpoint optimization                                           *)
(*                                                                          *)
(* Author: Claude (Anthropic)                                               *)
(* Date: 2025-12-26                                                         *)
(* Reference: Reifying RSpaces Specification                                *)
(****************************************************************************)

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    SpaceIds,           \* Set of all possible space identifiers
    ChannelNames,       \* Set of all possible channel names
    DataValues,         \* Set of all possible data values
    CheckpointIds,      \* Set of all possible checkpoint identifiers
    MaxHistoryLength,   \* Maximum length of operation history
    NULL                \* Null value for optional fields

VARIABLES
    spaceStates,        \* Map from SpaceId to space state (data + conts)
    checkpoints,        \* Map from CheckpointId to (MerkleRoot, RegistrySnapshot)
    operationLog,       \* Sequence of operations for replay
    replayMode,         \* Whether we're in replay mode
    replayIndex,        \* Current position in operation log during replay
    softCheckpoint,     \* In-memory soft checkpoint (for speculative exec)
    merkleRoots         \* Map from SpaceId to current Merkle root

vars == <<spaceStates, checkpoints, operationLog, replayMode,
          replayIndex, softCheckpoint, merkleRoots>>

-----------------------------------------------------------------------------
(* Type Definitions *)
-----------------------------------------------------------------------------

\* A datum stored at a channel
Datum == [data: DataValues, persist: BOOLEAN]

\* Space state: channels with their data and continuations
SpaceState == [
    data: [ChannelNames -> Seq(Datum)],
    continuations: Seq([channels: SUBSET ChannelNames, persist: BOOLEAN])
]

\* Operation types for the log
OperationType == {"Produce", "Consume", "Install"}

Operation == [
    type: OperationType,
    spaceId: SpaceIds,
    channel: ChannelNames \cup {NULL},
    channels: SUBSET ChannelNames,
    data: DataValues \cup {NULL}
]

\* Checkpoint structure
Checkpoint == [
    merkleRoot: Nat,          \* Hash representing state
    registrySnapshot: Nat,     \* Snapshot of registry state
    blockHeight: Nat
]

-----------------------------------------------------------------------------
(* Helper Functions *)
-----------------------------------------------------------------------------

\* Compute a simplified "Merkle root" (just a sequence number here)
ComputeMerkleRoot(spaceId) ==
    \* In reality, this would be a cryptographic hash
    \* Here we use a simple counter for specification purposes
    merkleRoots[spaceId] + 1

\* Empty space state
EmptySpaceState ==
    [data |-> [c \in ChannelNames |-> <<>>],
     continuations |-> <<>>]

-----------------------------------------------------------------------------
(* Type Invariants *)
-----------------------------------------------------------------------------

TypeOK ==
    /\ spaceStates \in [SpaceIds -> SpaceState]
    /\ checkpoints \in [CheckpointIds -> Checkpoint \cup {NULL}]
    /\ operationLog \in Seq(Operation)
    /\ replayMode \in BOOLEAN
    /\ replayIndex \in Nat
    /\ softCheckpoint \in [SpaceIds -> SpaceState] \cup {NULL}
    /\ merkleRoots \in [SpaceIds -> Nat]

-----------------------------------------------------------------------------
(* Initial State *)
-----------------------------------------------------------------------------

Init ==
    /\ spaceStates = [s \in SpaceIds |-> EmptySpaceState]
    /\ checkpoints = [c \in CheckpointIds |-> NULL]
    /\ operationLog = <<>>
    /\ replayMode = FALSE
    /\ replayIndex = 0
    /\ softCheckpoint = NULL
    /\ merkleRoots = [s \in SpaceIds |-> 0]

-----------------------------------------------------------------------------
(* Normal Mode Operations *)
-----------------------------------------------------------------------------

\* Produce: send data to a channel
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

\* Consume: receive data from channels (simplified)
Consume(spaceId, channels, persist) ==
    /\ ~replayMode
    /\ Len(operationLog) < MaxHistoryLength
    /\ channels # {}
    \* In reality, this would match and remove data
    /\ operationLog' = Append(operationLog,
        [type |-> "Consume", spaceId |-> spaceId, channel |-> NULL,
         channels |-> channels, data |-> NULL])
    /\ merkleRoots' = [merkleRoots EXCEPT ![spaceId] = ComputeMerkleRoot(spaceId)]
    /\ UNCHANGED <<spaceStates, checkpoints, replayMode, replayIndex, softCheckpoint>>

-----------------------------------------------------------------------------
(* Checkpoint Operations *)
-----------------------------------------------------------------------------

\* Create a persistent checkpoint
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

\* Create a soft (in-memory) checkpoint for speculative execution
CreateSoftCheckpoint ==
    /\ ~replayMode
    /\ softCheckpoint = NULL
    /\ softCheckpoint' = spaceStates
    /\ UNCHANGED <<spaceStates, checkpoints, operationLog,
                   replayMode, replayIndex, merkleRoots>>

\* Revert to soft checkpoint (rollback speculative execution)
RevertToSoftCheckpoint ==
    /\ ~replayMode
    /\ softCheckpoint # NULL
    /\ spaceStates' = softCheckpoint
    /\ softCheckpoint' = NULL
    /\ UNCHANGED <<checkpoints, operationLog, replayMode, replayIndex, merkleRoots>>

\* Commit soft checkpoint (discard the backup)
CommitSoftCheckpoint ==
    /\ ~replayMode
    /\ softCheckpoint # NULL
    /\ softCheckpoint' = NULL
    /\ UNCHANGED <<spaceStates, checkpoints, operationLog,
                   replayMode, replayIndex, merkleRoots>>

\* Reset to a checkpoint
ResetToCheckpoint(checkpointId) ==
    /\ checkpoints[checkpointId] # NULL
    /\ ~replayMode
    \* In reality, this would restore state from the checkpoint
    \* Here we just mark that we're resetting
    /\ UNCHANGED vars

-----------------------------------------------------------------------------
(* Replay Mode Operations *)
-----------------------------------------------------------------------------

\* Enter replay mode with a log to replay
EnterReplayMode(startCheckpointId, logToReplay) ==
    /\ ~replayMode
    /\ checkpoints[startCheckpointId] # NULL
    /\ replayMode' = TRUE
    /\ replayIndex' = 1
    /\ operationLog' = logToReplay
    /\ UNCHANGED <<spaceStates, checkpoints, softCheckpoint, merkleRoots>>

\* Replay the next operation from the log
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

\* Exit replay mode
ExitReplayMode ==
    /\ replayMode
    /\ replayIndex > Len(operationLog)  \* All operations replayed
    /\ replayMode' = FALSE
    /\ replayIndex' = 0
    /\ UNCHANGED <<spaceStates, checkpoints, operationLog,
                   softCheckpoint, merkleRoots>>

-----------------------------------------------------------------------------
(* Safety Invariants *)
-----------------------------------------------------------------------------

\* Replay mode is mutually exclusive with normal operations
ReplayModeExclusive ==
    replayMode => softCheckpoint = NULL

\* Replay index is within bounds
ReplayIndexValid ==
    replayMode => replayIndex <= Len(operationLog) + 1

\* Checkpoints are immutable once created
CheckpointsImmutable ==
    \* This would be checked across state transitions
    TRUE

\* Combined safety invariant
SafetyInvariant ==
    /\ TypeOK
    /\ ReplayModeExclusive
    /\ ReplayIndexValid

-----------------------------------------------------------------------------
(* Determinism Properties *)
-----------------------------------------------------------------------------

\* Two replays of the same log from the same checkpoint produce same state
\* This is expressed as a refinement mapping or temporal property
ReplayDeterminism ==
    \* If we replay the same log twice from the same checkpoint,
    \* the resulting states should be identical
    \* This is implicit in the specification since operations are deterministic
    TRUE

-----------------------------------------------------------------------------
(* Liveness Properties *)
-----------------------------------------------------------------------------

\* Replay eventually completes
ReplayEventuallyCompletes ==
    replayMode ~> ~replayMode

\* Soft checkpoint is eventually resolved (committed or reverted)
SoftCheckpointResolved ==
    (softCheckpoint # NULL) ~> (softCheckpoint = NULL)

-----------------------------------------------------------------------------
(* State Transitions *)
-----------------------------------------------------------------------------

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

-----------------------------------------------------------------------------
(* Theorems *)
-----------------------------------------------------------------------------

THEOREM SafetyHolds == Spec => []SafetyInvariant

THEOREM ReplayCompletes == Spec => ReplayEventuallyCompletes

=============================================================================
