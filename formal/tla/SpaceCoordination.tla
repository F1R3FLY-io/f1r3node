----------------------------- MODULE SpaceCoordination -----------------------------
(****************************************************************************)
(* TLA+ Specification for Multi-Space Coordination in Reified RSpaces       *)
(*                                                                          *)
(* This module specifies:                                                   *)
(* - Space registry operations (register, lookup, routing)                  *)
(* - Channel ownership semantics                                            *)
(* - Use block stack management                                             *)
(* - Cross-space join prevention                                            *)
(*                                                                          *)
(* Author: Claude (Anthropic)                                               *)
(* Date: 2025-12-26                                                         *)
(* Reference: Reifying RSpaces Specification                                *)
(****************************************************************************)

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    SpaceIds,           \* Set of all possible space identifiers
    ChannelNames,       \* Set of all possible channel names
    ProcessIds,         \* Set of all possible process identifiers
    MaxUseBlockDepth,   \* Maximum nesting depth for use blocks
    NULL                \* Null value for optional fields

VARIABLES
    spaces,             \* Map from SpaceId to SpaceConfig
    channelOwnership,   \* Map from ChannelName to SpaceId
    useBlockStacks,     \* Map from ProcessId to Seq(SpaceId)
    defaultSpace        \* The global default space

vars == <<spaces, channelOwnership, useBlockStacks, defaultSpace>>

-----------------------------------------------------------------------------
(* Type Definitions *)
-----------------------------------------------------------------------------

SpaceQualifier == {"Default", "Temp", "Seq"}

InnerCollectionType == {"Bag", "Queue", "Stack", "Set", "Cell"}

OuterStorageType == {"HashMap", "PathMap", "Array", "Vector", "HashSet"}

SpaceConfig == [
    qualifier: SpaceQualifier,
    innerType: InnerCollectionType,
    outerType: OuterStorageType
]

-----------------------------------------------------------------------------
(* Type Invariants *)
-----------------------------------------------------------------------------

TypeOK ==
    /\ spaces \in [SpaceIds -> SpaceConfig \cup {NULL}]
    /\ channelOwnership \in [ChannelNames -> SpaceIds \cup {NULL}]
    /\ useBlockStacks \in [ProcessIds -> Seq(SpaceIds)]
    /\ defaultSpace \in SpaceIds \cup {NULL}

-----------------------------------------------------------------------------
(* Initial State *)
-----------------------------------------------------------------------------

Init ==
    /\ spaces = [s \in SpaceIds |-> NULL]
    /\ channelOwnership = [c \in ChannelNames |-> NULL]
    /\ useBlockStacks = [p \in ProcessIds |-> <<>>]
    /\ defaultSpace = NULL

-----------------------------------------------------------------------------
(* Space Registry Operations *)
-----------------------------------------------------------------------------

(* Register a new space with given configuration *)
RegisterSpace(spaceId, config) ==
    /\ spaces[spaceId] = NULL  \* Space not already registered
    /\ spaces' = [spaces EXCEPT ![spaceId] = config]
    /\ UNCHANGED <<channelOwnership, useBlockStacks, defaultSpace>>

(* Set the global default space *)
SetDefaultSpace(spaceId) ==
    /\ spaces[spaceId] # NULL  \* Space must exist
    /\ defaultSpace' = spaceId
    /\ UNCHANGED <<spaces, channelOwnership, useBlockStacks>>

(* Register a channel with a specific space *)
RegisterChannel(channel, spaceId) ==
    /\ spaces[spaceId] # NULL       \* Space must exist
    /\ channelOwnership[channel] = NULL  \* Channel not already registered
    /\ channelOwnership' = [channelOwnership EXCEPT ![channel] = spaceId]
    /\ UNCHANGED <<spaces, useBlockStacks, defaultSpace>>

-----------------------------------------------------------------------------
(* Use Block Operations *)
-----------------------------------------------------------------------------

(* Push a space onto the use block stack for a process *)
PushUseBlock(processId, spaceId) ==
    /\ spaces[spaceId] # NULL  \* Space must exist
    /\ Len(useBlockStacks[processId]) < MaxUseBlockDepth
    /\ useBlockStacks' = [useBlockStacks EXCEPT
        ![processId] = Append(@, spaceId)]
    /\ UNCHANGED <<spaces, channelOwnership, defaultSpace>>

(* Pop a space from the use block stack *)
PopUseBlock(processId) ==
    /\ Len(useBlockStacks[processId]) > 0
    /\ useBlockStacks' = [useBlockStacks EXCEPT
        ![processId] = SubSeq(@, 1, Len(@) - 1)]
    /\ UNCHANGED <<spaces, channelOwnership, defaultSpace>>

(* Get the effective default space for a process *)
EffectiveDefaultSpace(processId) ==
    IF Len(useBlockStacks[processId]) > 0
    THEN useBlockStacks[processId][Len(useBlockStacks[processId])]
    ELSE defaultSpace

-----------------------------------------------------------------------------
(* Channel Routing *)
-----------------------------------------------------------------------------

(* Resolve which space a channel belongs to *)
ResolveSpace(channel, processId) ==
    IF channelOwnership[channel] # NULL
    THEN channelOwnership[channel]
    ELSE EffectiveDefaultSpace(processId)

(* Check if all channels in a set belong to the same space *)
SameSpace(channels, processId) ==
    LET resolvedSpaces == {ResolveSpace(c, processId) : c \in channels}
    IN Cardinality(resolvedSpaces) <= 1

-----------------------------------------------------------------------------
(* Safety Invariants *)
-----------------------------------------------------------------------------

(* No orphaned channels - every registered channel has an existing space *)
NoOrphanedChannels ==
    \A c \in ChannelNames :
        channelOwnership[c] # NULL => spaces[channelOwnership[c]] # NULL

(* Use block stacks only contain registered spaces *)
UseBlocksValid ==
    \A p \in ProcessIds :
        \A i \in 1..Len(useBlockStacks[p]) :
            spaces[useBlockStacks[p][i]] # NULL

(* Default space, if set, must be registered *)
DefaultSpaceValid ==
    defaultSpace # NULL => spaces[defaultSpace] # NULL

(* Combined safety invariant *)
SafetyInvariant ==
    /\ TypeOK
    /\ NoOrphanedChannels
    /\ UseBlocksValid
    /\ DefaultSpaceValid

-----------------------------------------------------------------------------
(* Cross-Space Join Prevention *)
-----------------------------------------------------------------------------

(* A join pattern is valid only if all channels resolve to the same space *)
ValidJoinPattern(channels, processId) ==
    SameSpace(channels, processId)

(* Invariant: A consume operation with multiple channels requires same-space *)
(* Note: This is an action guard, not a state invariant *)
ConsumeMultiChannelGuard(channels, processId) ==
    Cardinality(channels) > 1 => ValidJoinPattern(channels, processId)

-----------------------------------------------------------------------------
(* Seq Channel Mobility Restriction *)
-----------------------------------------------------------------------------

(* Seq-qualified channels cannot be sent to other processes *)
IsSeqChannel(channel) ==
    LET spaceId == channelOwnership[channel]
    IN spaceId # NULL /\ spaces[spaceId] # NULL /\
       spaces[spaceId].qualifier = "Seq"

(* Send operation must not include Seq channels *)
ValidSendChannels(channels) ==
    \A c \in channels : ~IsSeqChannel(c)

-----------------------------------------------------------------------------
(* State Transitions *)
-----------------------------------------------------------------------------

Next ==
    \/ \E s \in SpaceIds, cfg \in SpaceConfig : RegisterSpace(s, cfg)
    \/ \E s \in SpaceIds : SetDefaultSpace(s)
    \/ \E c \in ChannelNames, s \in SpaceIds : RegisterChannel(c, s)
    \/ \E p \in ProcessIds, s \in SpaceIds : PushUseBlock(p, s)
    \/ \E p \in ProcessIds : PopUseBlock(p)

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* Liveness Properties *)
-----------------------------------------------------------------------------

(* Eventually, a default space is set (if we want to enforce this) *)
EventuallyHasDefaultSpace ==
    <>(defaultSpace # NULL)

(* Use blocks are eventually balanced (for terminating processes) *)
(* This would require process lifecycle to be modeled *)

-----------------------------------------------------------------------------
(* Theorems to Check *)
-----------------------------------------------------------------------------

THEOREM SafetyHolds == Spec => []SafetyInvariant

=============================================================================
