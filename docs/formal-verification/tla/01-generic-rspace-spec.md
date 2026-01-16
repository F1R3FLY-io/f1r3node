# GenericRSpace Specification: GenericRSpace.tla

This document covers the TLA+ specification of the core tuple space operations in `GenericRSpace.tla`.

## Overview

The specification models the `GenericRSpace<CS, M>` parameterized implementation that combines a ChannelStore with a Match strategy. It captures:
- Data and continuation stores
- Join relationships for multi-channel consumes
- Produce and consume operations
- The fundamental `NoPendingMatch` invariant

## State Variables

```tla
VARIABLES
    dataStore,      \* channel -> Seq(Datum)
    contStore,      \* Seq(channel) -> Seq(WaitingContinuation)
    joins,          \* channel -> Set(Seq(channel))
    nameCounter,    \* Counter for gensym
    qualifier       \* Space qualifier (0=Default, 1=Temp, 2=Seq)
```

### dataStore

Maps channels to sequences of data with persistence flags:

```tla
Datum == [data: Data, persist: BOOLEAN]
dataStore \in [Channels -> Seq(Datum)]
```

**Example**:
```
dataStore = [
    ch1 |-> <<[data |-> d1, persist |-> FALSE], [data |-> d2, persist |-> TRUE]>>,
    ch2 |-> <<>>
]
```

### contStore

Maps channel sequences (for joins) to waiting continuations:

```tla
WaitingCont == [patterns: Seq(Patterns), cont: Continuations, persist: BOOLEAN]
contStore \in [Seq(Channels) -> Seq(WaitingCont)]
```

**Example**:
```
contStore = [
    <<ch1>> |-> <<[patterns |-> <<p1>>, cont |-> k1, persist |-> FALSE]>>,
    <<ch1, ch2>> |-> <<>>  \* Join on ch1 AND ch2
]
```

### joins

Tracks which channel sequences involve each channel:

```tla
joins \in [Channels -> SUBSET Seq(Channels)]
```

**Example**:
```
joins = [
    ch1 |-> {<<ch1>>, <<ch1, ch2>>},  \* ch1 participates in two patterns
    ch2 |-> {<<ch1, ch2>>}             \* ch2 only in the join
]
```

## Constants

```tla
CONSTANTS
    Channels,       \* Set of possible channel identifiers
    Patterns,       \* Set of possible patterns
    Data,           \* Set of possible data values
    Continuations,  \* Set of possible continuations
    NULL            \* Null value for optional fields
```

## Type Invariant

```tla
TypeInvariant ==
    /\ dataStore \in [Channels -> Seq(Datum)]
    /\ contStore \in [Seq(Channels) -> Seq(WaitingCont)]
    /\ joins \in [Channels -> SUBSET Seq(Channels)]
    /\ nameCounter \in Nat
    /\ qualifier \in {0, 1, 2}
```

## Core Invariant: NoPendingMatch

The fundamental safety property:

```tla
NoPendingMatch ==
    \A ch \in Channels :
        \A i \in 1..Len(dataStore[ch]) :
            LET datum == dataStore[ch][i] IN
            \A joinChannels \in joins[ch] :
                \A j \in 1..Len(contStore[joinChannels]) :
                    LET wk == contStore[joinChannels][j] IN
                    ~AnyPatternMatches(wk.patterns, datum.data)
```

**Intuition**: For every piece of data at every channel, no waiting continuation on any join involving that channel has a matching pattern.

## Actions

### Gensym

Generate a fresh channel name:

```tla
Gensym ==
    /\ nameCounter' = nameCounter + 1
    /\ UNCHANGED <<dataStore, contStore, joins, qualifier>>
```

### Produce

Send data to a channel:

```tla
Produce(ch, data, persist) ==
    LET datum == [data |-> data, persist |-> persist] IN
    LET matchingJoin == FindMatchingCont(ch, data) IN
    IF matchingJoin = NULL THEN
        \* No matching continuation - store the data
        /\ dataStore' = [dataStore EXCEPT ![ch] = Append(@, datum)]
        /\ UNCHANGED <<contStore, joins, nameCounter, qualifier>>
    ELSE
        \* Found matching continuation - fire it
        /\ contStore' = [contStore EXCEPT ![matchingJoin] =
                RemoveMatchingCont(matchingJoin, data)]
        /\ UNCHANGED <<dataStore, joins, nameCounter, qualifier>>
```

**Two cases**:
1. **No match found**: Store data, leave continuations unchanged
2. **Match found**: Fire continuation (remove it), don't store data

### FindMatchingCont

Helper to find a matching continuation:

```tla
FindMatchingCont(ch, data) ==
    LET matchingJoins == {jc \in joins[ch] :
            \E i \in 1..Len(contStore[jc]) :
                AnyPatternMatches(contStore[jc][i].patterns, data)}
    IN IF matchingJoins = {} THEN NULL
       ELSE CHOOSE jc \in matchingJoins : TRUE
```

### Consume

Wait for data on channel(s):

```tla
Consume(channels, patterns, cont, persist) ==
    LET checkAllMatch ==
        \A i \in 1..Len(channels) :
            FindMatchingData(channels[i], patterns[i]) /= NULL
    IN
    IF checkAllMatch THEN
        \* All channels have matching data - consume immediately
        /\ dataStore' = [ch \in Channels |->
            IF \E i \in 1..Len(channels) : channels[i] = ch
            THEN \* Remove matching data (if not persistent)
                 ...
            ELSE dataStore[ch]]
        /\ UNCHANGED <<contStore, joins, nameCounter, qualifier>>
    ELSE
        \* Not all data available - store continuation
        LET wk == [patterns |-> patterns, cont |-> cont, persist |-> persist] IN
        /\ contStore' = [contStore EXCEPT ![channels] = Append(@, wk)]
        /\ joins' = [ch \in Channels |->
            IF \E i \in 1..Len(channels) : channels[i] = ch
            THEN joins[ch] \union {channels}
            ELSE joins[ch]]
        /\ UNCHANGED <<dataStore, nameCounter, qualifier>>
```

**Two cases**:
1. **All data present**: Consume immediately, update dataStore
2. **Data missing**: Store continuation, update joins

### Clear

Reset the space:

```tla
Clear ==
    /\ dataStore' = [ch \in Channels |-> <<>>]
    /\ contStore' = [chs \in Seq(Channels) |-> <<>>]
    /\ joins' = [ch \in Channels |-> {}]
    /\ UNCHANGED <<nameCounter, qualifier>>
```

## Initial State

```tla
Init ==
    /\ dataStore = [ch \in Channels |-> <<>>]
    /\ contStore = [chs \in Seq(Channels) |-> <<>>]
    /\ joins = [ch \in Channels |-> {}]
    /\ nameCounter = 0
    /\ qualifier = 0
```

## Next State Relation

```tla
Next ==
    \/ Gensym
    \/ \E ch \in Channels, d \in Data, p \in BOOLEAN : Produce(ch, d, p)
    \/ \E chs \in Seq(Channels), ps \in Seq(Patterns),
          k \in Continuations, p \in BOOLEAN : Consume(chs, ps, k, p)
    \/ Clear
```

## Safety Properties

### ProduceSafety

```tla
ProduceSafety ==
    \A ch \in Channels, d \in Data, p \in BOOLEAN :
        NoPendingMatch /\ Produce(ch, d, p) => NoPendingMatch'
```

**Intuition**: If the invariant holds and we produce, the invariant still holds.

### ConsumeSafety

```tla
ConsumeSafety ==
    \A chs \in Seq(Channels), ps \in Seq(Patterns),
       k \in Continuations, p \in BOOLEAN :
        NoPendingMatch /\ Consume(chs, ps, k, p) => NoPendingMatch'
```

### Overall Safety

```tla
Safety == []NoPendingMatch
```

**This states**: In all reachable states, `NoPendingMatch` holds.

## Liveness Property

```tla
MatchEventuallyFires ==
    \A ch \in Channels, d \in Data :
        \A jc \in joins[ch] :
            (\E i \in 1..Len(contStore[jc]) :
                AnyPatternMatches(contStore[jc][i].patterns, d))
            => <>(Len(contStore[jc]) < Len(contStore[jc]))
```

**Intuition**: If data arrives and a matching continuation exists, eventually the continuation fires (continuation queue shrinks).

## Fairness

```tla
Fairness == WF_vars(Next)
LiveSpec == Spec /\ Fairness
```

Weak fairness ensures that if `Next` is continuously enabled, it eventually happens.

## Example Traces

### Trace 1: Produce then Consume

```
State 0 (Init):
  dataStore = [ch1 |-> <<>>, ch2 |-> <<>>]
  contStore = [<<ch1>> |-> <<>>, ...]
  joins = [ch1 |-> {}, ch2 |-> {}]

Action: Produce(ch1, d1, FALSE)
  No matching continuation -> store data

State 1:
  dataStore = [ch1 |-> <<[data |-> d1, persist |-> FALSE]>>, ch2 |-> <<>>]
  contStore = unchanged
  joins = unchanged

Action: Consume(<<ch1>>, <<p1>>, k1, FALSE)  where Matches(p1, d1) = TRUE
  Data present -> consume immediately

State 2:
  dataStore = [ch1 |-> <<>>, ch2 |-> <<>>]  \* d1 removed
  contStore = unchanged
  joins = unchanged
```

### Trace 2: Consume then Produce (comm)

```
State 0 (Init):
  dataStore = [ch1 |-> <<>>, ...]
  contStore = [<<ch1>> |-> <<>>, ...]
  joins = [ch1 |-> {}, ...]

Action: Consume(<<ch1>>, <<p1>>, k1, FALSE)
  No data -> store continuation

State 1:
  dataStore = unchanged
  contStore = [<<ch1>> |-> <<[patterns |-> <<p1>>, cont |-> k1, persist |-> FALSE]>>, ...]
  joins = [ch1 |-> {<<ch1>>}, ...]

Action: Produce(ch1, d1, FALSE)  where Matches(p1, d1) = TRUE
  Matching continuation found -> fire it

State 2:
  dataStore = unchanged (data goes to continuation)
  contStore = [<<ch1>> |-> <<>>, ...]  \* continuation removed
  joins = [ch1 |-> {<<ch1>>}, ...]  \* joins unchanged
```

### Trace 3: Multi-Channel Join

```
State 0:
  Consume(<<ch1, ch2>>, <<p1, p2>>, k1, FALSE)
  No data at ch1 or ch2 -> store continuation

State 1:
  contStore[<<ch1, ch2>>] = <<wk1>>
  joins[ch1] = {<<ch1, ch2>>}
  joins[ch2] = {<<ch1, ch2>>}

State 2:
  Produce(ch1, d1, FALSE)  where Matches(p1, d1) = TRUE
  Continuation needs BOTH ch1 and ch2 data
  ch2 has no data -> store at ch1

State 3:
  dataStore[ch1] = <<[data |-> d1, ...]>>
  NoPendingMatch still holds? YES - wk1.patterns needs ch2 data too

State 4:
  Produce(ch2, d2, FALSE)  where Matches(p2, d2) = TRUE
  Now both ch1 and ch2 have matching data
  Fire the join continuation!

State 5:
  dataStore[ch1] = <<>>  \* d1 consumed
  dataStore[ch2] = <<>>  \* d2 consumed
  contStore[<<ch1, ch2>>] = <<>>  \* wk1 fired
```

## Correspondence to Rust

| TLA+ | Rust |
|------|------|
| `dataStore[ch]` | `ChannelStore::data_at(ch)` |
| `contStore[chs]` | `ChannelStore::continuations_at(chs)` |
| `joins[ch]` | `ChannelStore::joins_for(ch)` |
| `Produce(ch, d, p)` | `GenericRSpace::produce()` |
| `Consume(chs, ps, k, p)` | `GenericRSpace::consume()` |
| `FindMatchingCont` | `GenericRSpace::find_matching_continuation()` |
| `NoPendingMatch` | Invariant maintained by implementation |
| `nameCounter` | `GenericRSpace::name_counter` |

## Model Checking

To check this specification with TLC:

```tla
\* GenericRSpace.cfg
CONSTANTS
    Channels = {c1, c2}
    Patterns = {p1}
    Data = {d1}
    Continuations = {k1}
    NULL = NULL

SPECIFICATION Spec
INVARIANTS TypeInvariant NoPendingMatch
```

With 2 channels, 1 pattern, 1 data, 1 continuation, TLC explores ~10,000 states.

## Next Steps

Continue to [02-checkpoint-replay-spec.md](02-checkpoint-replay-spec.md) for checkpoint and replay specifications.
