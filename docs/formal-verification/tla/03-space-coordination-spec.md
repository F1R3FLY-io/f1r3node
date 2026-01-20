# Space Coordination Specification: SpaceCoordination.tla

This document covers the TLA+ specification for multi-space coordination in `SpaceCoordination.tla`.

## Overview

The specification models the registry and coordination layer for multiple RSpace instances:
- Space registration and configuration
- Channel ownership tracking
- Use block stack management (dynamic space scoping)
- Cross-space join prevention
- Seq channel mobility restrictions

This ensures that channels are properly routed to their owning spaces and that joins don't cross space boundaries.

## State Variables

```tla
VARIABLES
    spaces,             \* Map from SpaceId to SpaceConfig
    channelOwnership,   \* Map from ChannelName to SpaceId
    useBlockStacks,     \* Map from ProcessId to Seq(SpaceId)
    defaultSpace        \* The global default space
```

### spaces

Maps space identifiers to their configurations:

```tla
SpaceQualifier == {"Default", "Temp", "Seq"}

InnerCollectionType == {"Bag", "Queue", "Stack", "Set", "Cell"}

OuterStorageType == {"HashMap", "PathMap", "Array", "Vector", "HashSet"}

SpaceConfig == [
    qualifier: SpaceQualifier,
    innerType: InnerCollectionType,
    outerType: OuterStorageType
]

spaces \in [SpaceIds -> SpaceConfig \cup {NULL}]
```

**Example**:
```
spaces = [
    s1 |-> [qualifier |-> "Default", innerType |-> "Bag", outerType |-> "HashMap"],
    s2 |-> [qualifier |-> "Temp", innerType |-> "Queue", outerType |-> "Vector"],
    s3 |-> NULL  \* Not yet registered
]
```

### channelOwnership

Maps channels to their owning space:

```tla
channelOwnership \in [ChannelNames -> SpaceIds \cup {NULL}]
```

**Example**:
```
channelOwnership = [
    ch1 |-> s1,   \* ch1 owned by s1
    ch2 |-> s2,   \* ch2 owned by s2
    ch3 |-> NULL  \* ch3 uses effective default
]
```

### useBlockStacks

Per-process stacks of active use blocks:

```tla
useBlockStacks \in [ProcessIds -> Seq(SpaceIds)]
```

**Example**:
```
useBlockStacks = [
    p1 |-> <<s2, s1>>,  \* p1 is in: use s1 { use s2 { ... } }
    p2 |-> <<s1>>,      \* p2 is in: use s1 { ... }
    p3 |-> <<>>         \* p3 has no active use blocks
]
```

The stack represents nested `use` blocks - the top (last element) is the current effective space.

### defaultSpace

The global default space:

```tla
defaultSpace \in SpaceIds \cup {NULL}
```

## Constants

```tla
CONSTANTS
    SpaceIds,           \* Set of all possible space identifiers
    ChannelNames,       \* Set of all possible channel names
    ProcessIds,         \* Set of all possible process identifiers
    MaxUseBlockDepth,   \* Maximum nesting depth for use blocks
    NULL                \* Null value for optional fields
```

## Type Invariant

```tla
TypeOK ==
    /\ spaces \in [SpaceIds -> SpaceConfig \cup {NULL}]
    /\ channelOwnership \in [ChannelNames -> SpaceIds \cup {NULL}]
    /\ useBlockStacks \in [ProcessIds -> Seq(SpaceIds)]
    /\ defaultSpace \in SpaceIds \cup {NULL}
```

## Initial State

```tla
Init ==
    /\ spaces = [s \in SpaceIds |-> NULL]
    /\ channelOwnership = [c \in ChannelNames |-> NULL]
    /\ useBlockStacks = [p \in ProcessIds |-> <<>>]
    /\ defaultSpace = NULL
```

All spaces unregistered, no channel ownership, no active use blocks, no default space.

## Actions

### Space Registry Operations

#### RegisterSpace

Register a new space with configuration:

```tla
RegisterSpace(spaceId, config) ==
    /\ spaces[spaceId] = NULL  \* Space not already registered
    /\ spaces' = [spaces EXCEPT ![spaceId] = config]
    /\ UNCHANGED <<channelOwnership, useBlockStacks, defaultSpace>>
```

**Precondition**: Space ID not already registered.

**Example**:
```
Before: spaces[s1] = NULL
Action: RegisterSpace(s1, [qualifier |-> "Default", innerType |-> "Bag", outerType |-> "HashMap"])
After:  spaces[s1] = [qualifier |-> "Default", innerType |-> "Bag", outerType |-> "HashMap"]
```

#### SetDefaultSpace

Set the global default space:

```tla
SetDefaultSpace(spaceId) ==
    /\ spaces[spaceId] # NULL  \* Space must exist
    /\ defaultSpace' = spaceId
    /\ UNCHANGED <<spaces, channelOwnership, useBlockStacks>>
```

**Precondition**: Space must be registered.

#### RegisterChannel

Associate a channel with a space:

```tla
RegisterChannel(channel, spaceId) ==
    /\ spaces[spaceId] # NULL       \* Space must exist
    /\ channelOwnership[channel] = NULL  \* Channel not already registered
    /\ channelOwnership' = [channelOwnership EXCEPT ![channel] = spaceId]
    /\ UNCHANGED <<spaces, useBlockStacks, defaultSpace>>
```

**Preconditions**:
- Space must be registered
- Channel not already owned

### Use Block Operations

#### PushUseBlock

Enter a `use` block for a process:

```tla
PushUseBlock(processId, spaceId) ==
    /\ spaces[spaceId] # NULL  \* Space must exist
    /\ Len(useBlockStacks[processId]) < MaxUseBlockDepth
    /\ useBlockStacks' = [useBlockStacks EXCEPT
        ![processId] = Append(@, spaceId)]
    /\ UNCHANGED <<spaces, channelOwnership, defaultSpace>>
```

**Intuition**: When a process enters `use mySpace { ... }`, push `mySpace` onto that process's stack.

**Example**:
```
Before: useBlockStacks[p1] = <<s1>>
Action: PushUseBlock(p1, s2)
After:  useBlockStacks[p1] = <<s1, s2>>
```

#### PopUseBlock

Exit a `use` block:

```tla
PopUseBlock(processId) ==
    /\ Len(useBlockStacks[processId]) > 0
    /\ useBlockStacks' = [useBlockStacks EXCEPT
        ![processId] = SubSeq(@, 1, Len(@) - 1)]
    /\ UNCHANGED <<spaces, channelOwnership, defaultSpace>>
```

**Precondition**: Stack must not be empty.

### Helper Functions

#### EffectiveDefaultSpace

Get the current default space for a process:

```tla
EffectiveDefaultSpace(processId) ==
    IF Len(useBlockStacks[processId]) > 0
    THEN useBlockStacks[processId][Len(useBlockStacks[processId])]
    ELSE defaultSpace
```

**Intuition**: If inside a use block, use that space. Otherwise, use the global default.

**Example**:
```
useBlockStacks[p1] = <<s1, s2>>
EffectiveDefaultSpace(p1) = s2  \* Top of stack

useBlockStacks[p2] = <<>>
defaultSpace = s1
EffectiveDefaultSpace(p2) = s1  \* Falls back to global default
```

#### ResolveSpace

Determine which space a channel belongs to:

```tla
ResolveSpace(channel, processId) ==
    IF channelOwnership[channel] # NULL
    THEN channelOwnership[channel]
    ELSE EffectiveDefaultSpace(processId)
```

**Resolution order**:
1. If channel has explicit owner, use that
2. Otherwise, use effective default space for this process

#### SameSpace

Check if all channels resolve to the same space:

```tla
SameSpace(channels, processId) ==
    LET resolvedSpaces == {ResolveSpace(c, processId) : c \in channels}
    IN Cardinality(resolvedSpaces) <= 1
```

**Intuition**: For a join pattern, all channels must belong to the same space.

## Safety Invariants

### NoOrphanedChannels

Every registered channel has an existing space:

```tla
NoOrphanedChannels ==
    \A c \in ChannelNames :
        channelOwnership[c] # NULL => spaces[channelOwnership[c]] # NULL
```

**Intuition**: You can't delete a space while channels still point to it.

### UseBlocksValid

Use block stacks only contain registered spaces:

```tla
UseBlocksValid ==
    \A p \in ProcessIds :
        \A i \in 1..Len(useBlockStacks[p]) :
            spaces[useBlockStacks[p][i]] # NULL
```

**Intuition**: Every space on every use block stack must exist.

### DefaultSpaceValid

The default space, if set, must be registered:

```tla
DefaultSpaceValid ==
    defaultSpace # NULL => spaces[defaultSpace] # NULL
```

### Combined Safety

```tla
SafetyInvariant ==
    /\ TypeOK
    /\ NoOrphanedChannels
    /\ UseBlocksValid
    /\ DefaultSpaceValid
```

## Cross-Space Join Prevention

### ValidJoinPattern

A join pattern is valid only if all channels resolve to the same space:

```tla
ValidJoinPattern(channels, processId) ==
    SameSpace(channels, processId)
```

### ConsumeMultiChannelGuard

Multi-channel consume requires same-space:

```tla
ConsumeMultiChannelGuard(channels, processId) ==
    Cardinality(channels) > 1 => ValidJoinPattern(channels, processId)
```

**Intuition**: If you're consuming from multiple channels (a join), they must all be in the same space. Cross-space joins are prohibited.

**Example**:
```
channelOwnership[ch1] = s1
channelOwnership[ch2] = s2

ValidJoinPattern({ch1, ch2}, p1) = FALSE  \* Different spaces!

channelOwnership[ch3] = s1

ValidJoinPattern({ch1, ch3}, p1) = TRUE   \* Same space
```

## Seq Channel Mobility Restrictions

### IsSeqChannel

Check if a channel belongs to a Seq-qualified space:

```tla
IsSeqChannel(channel) ==
    LET spaceId == channelOwnership[channel]
    IN spaceId # NULL /\ spaces[spaceId] # NULL /\
       spaces[spaceId].qualifier = "Seq"
```

### ValidSendChannels

Seq channels cannot be sent to other processes:

```tla
ValidSendChannels(channels) ==
    \A c \in channels : ~IsSeqChannel(c)
```

**Intuition**: Seq-qualified channels are non-mobile - they cannot leave the process that created them.

## Next State Relation

```tla
Next ==
    \/ \E s \in SpaceIds, cfg \in SpaceConfig : RegisterSpace(s, cfg)
    \/ \E s \in SpaceIds : SetDefaultSpace(s)
    \/ \E c \in ChannelNames, s \in SpaceIds : RegisterChannel(c, s)
    \/ \E p \in ProcessIds, s \in SpaceIds : PushUseBlock(p, s)
    \/ \E p \in ProcessIds : PopUseBlock(p)

Spec == Init /\ [][Next]_vars
```

## Liveness Properties

### EventuallyHasDefaultSpace

Eventually a default space is set:

```tla
EventuallyHasDefaultSpace ==
    <>(defaultSpace # NULL)
```

**Note**: This is optional - the specification works without a default space if all channels are explicitly registered.

## Example Traces

### Trace 1: Basic Registration

```
State 0 (Init):
  spaces = [s1 |-> NULL, s2 |-> NULL]
  channelOwnership = [ch1 |-> NULL, ch2 |-> NULL]
  defaultSpace = NULL

Action: RegisterSpace(s1, [qualifier |-> "Default", innerType |-> "Bag", outerType |-> "HashMap"])

State 1:
  spaces[s1] = [qualifier |-> "Default", innerType |-> "Bag", outerType |-> "HashMap"]

Action: SetDefaultSpace(s1)

State 2:
  defaultSpace = s1

Action: RegisterChannel(ch1, s1)

State 3:
  channelOwnership[ch1] = s1
```

### Trace 2: Use Block Nesting

```
State 0:
  spaces = [s1 |-> cfg1, s2 |-> cfg2]
  useBlockStacks[p1] = <<>>

Action: PushUseBlock(p1, s1)

State 1:
  useBlockStacks[p1] = <<s1>>
  EffectiveDefaultSpace(p1) = s1

Action: PushUseBlock(p1, s2)

State 2:
  useBlockStacks[p1] = <<s1, s2>>
  EffectiveDefaultSpace(p1) = s2  \* Now s2!

Action: PopUseBlock(p1)

State 3:
  useBlockStacks[p1] = <<s1>>
  EffectiveDefaultSpace(p1) = s1  \* Back to s1

Action: PopUseBlock(p1)

State 4:
  useBlockStacks[p1] = <<>>
  EffectiveDefaultSpace(p1) = defaultSpace
```

### Trace 3: Channel Resolution

```
Configuration:
  spaces = [s1 |-> cfg1, s2 |-> cfg2]
  channelOwnership = [ch1 |-> s1, ch2 |-> s2, ch3 |-> NULL]
  defaultSpace = s1
  useBlockStacks[p1] = <<s2>>

Resolution:
  ResolveSpace(ch1, p1) = s1   \* Explicit owner
  ResolveSpace(ch2, p1) = s2   \* Explicit owner
  ResolveSpace(ch3, p1) = s2   \* No owner, use effective default (s2 from use block)

  SameSpace({ch1, ch2}, p1) = FALSE  \* s1 ≠ s2
  SameSpace({ch2, ch3}, p1) = TRUE   \* Both resolve to s2
```

### Trace 4: Cross-Space Join Prevention

```
Configuration:
  channelOwnership = [ch1 |-> s1, ch2 |-> s2]

Attempt: Consume on {ch1, ch2}
  ResolveSpace(ch1, p1) = s1
  ResolveSpace(ch2, p1) = s2
  SameSpace({ch1, ch2}, p1) = FALSE
  ValidJoinPattern({ch1, ch2}, p1) = FALSE  ✗ INVALID

Consume would be rejected - cross-space join not allowed.
```

## Correspondence to Rust

| TLA+ | Rust |
|------|------|
| `spaces` | `Registry::spaces` map |
| `channelOwnership` | `Registry::channel_owners` |
| `useBlockStacks` | `Registry::use_block_stack` per process |
| `defaultSpace` | `Registry::default_space` |
| `SpaceQualifier` | `SpaceQualifier` enum |
| `SpaceConfig` | `SpaceConfig` struct |
| `RegisterSpace` | `Registry::register_space()` |
| `SetDefaultSpace` | `Registry::set_default_space()` |
| `RegisterChannel` | `Registry::register_channel()` |
| `PushUseBlock` | `Registry::push_use_block()` |
| `PopUseBlock` | `Registry::pop_use_block()` |
| `EffectiveDefaultSpace` | `Registry::effective_default_space()` |
| `ResolveSpace` | `Registry::resolve_space()` |
| `ValidJoinPattern` | Join validation in `consume()` |
| `IsSeqChannel` | `ChannelQualifier::Seq` check |

## Model Checking

To check this specification with TLC:

```tla
\* SpaceCoordination.cfg
CONSTANTS
    SpaceIds = {s1, s2}
    ChannelNames = {ch1, ch2}
    ProcessIds = {p1}
    MaxUseBlockDepth = 3
    NULL = NULL

SPECIFICATION Spec
INVARIANTS TypeOK SafetyInvariant
```

With these constants, TLC explores the space coordination state machine exhaustively.

## Theorem

```tla
THEOREM SafetyHolds == Spec => []SafetyInvariant
```

## Design Notes

### Why Prevent Cross-Space Joins?

Cross-space joins would require:
1. Atomic operations across multiple spaces
2. Distributed locking or 2PC
3. Complex rollback semantics

By requiring all join channels to be in the same space, we keep the semantics simple and avoid distributed coordination.

### Why Use Block Stacks?

The stack model matches Rholang's scoping semantics:

```rholang
use space1 {
  // EffectiveDefaultSpace = space1
  use space2 {
    // EffectiveDefaultSpace = space2
  }
  // EffectiveDefaultSpace = space1 again
}
```

### Why Seq Channels Are Immobile

Seq-qualified spaces have sequential (non-concurrent) semantics. Sending a Seq channel to another process would violate the sequential access guarantee.

## Next Steps

This completes the TLA+ documentation. See also:
- [00-introduction.md](00-introduction.md) - TLA+ overview
- [01-generic-rspace-spec.md](01-generic-rspace-spec.md) - Core tuple space operations
- [02-checkpoint-replay-spec.md](02-checkpoint-replay-spec.md) - Checkpoint and replay
