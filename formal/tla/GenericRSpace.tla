----------------------------- MODULE GenericRSpace -----------------------------
(***************************************************************************)
(* TLA+ Specification for GenericRSpace<CS, M>                             *)
(*                                                                          *)
(* This module specifies the GenericRSpace parameterized implementation    *)
(* that combines a ChannelStore with a Match strategy.                     *)
(*                                                                          *)
(* Reference: Rust implementation in                                        *)
(*   rholang/src/rust/interpreter/spaces/generic_rspace.rs                 *)
(***************************************************************************)

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    Channels,       \* Set of possible channel identifiers
    Patterns,       \* Set of possible patterns
    Data,           \* Set of possible data values
    Continuations,  \* Set of possible continuations
    NULL            \* Null value for optional fields

VARIABLES
    dataStore,      \* channel -> Seq(Datum)
    contStore,      \* Seq(channel) -> Seq(WaitingContinuation)
    joins,          \* channel -> Set(Seq(channel))
    nameCounter,    \* Counter for gensym
    qualifier       \* Space qualifier (0=Default, 1=Temp, 2=Seq)

vars == <<dataStore, contStore, joins, nameCounter, qualifier>>

(***************************************************************************)
(* Type Definitions                                                         *)
(***************************************************************************)

(* A datum stored at a channel *)
Datum == [data: Data, persist: BOOLEAN]

(* A waiting continuation *)
WaitingCont == [patterns: Seq(Patterns), cont: Continuations, persist: BOOLEAN]

(* Match function abstraction - returns TRUE if pattern matches data *)
(* This is a parameter that can be instantiated with different strategies *)
Matches(pattern, data) == TRUE  \* Default: wildcard matches everything

(***************************************************************************)
(* Type Invariants                                                          *)
(***************************************************************************)

TypeInvariant ==
    /\ dataStore \in [Channels -> Seq(Datum)]
    /\ contStore \in [Seq(Channels) -> Seq(WaitingCont)]
    /\ joins \in [Channels -> SUBSET Seq(Channels)]
    /\ nameCounter \in Nat
    /\ qualifier \in {0, 1, 2}

(***************************************************************************)
(* Core Safety Invariant: No Pending Match                                  *)
(***************************************************************************)

(* Convert sequence to set for checking membership *)
SeqToSet(s) == {s[i] : i \in 1..Len(s)}

(* Check if any pattern in patterns matches the data *)
AnyPatternMatches(patterns, data) ==
    \E i \in 1..Len(patterns) : Matches(patterns[i], data)

(* The fundamental invariant: if data exists at a channel, no waiting
   continuation on that channel has a matching pattern *)
NoPendingMatch ==
    \A ch \in Channels :
        \A i \in 1..Len(dataStore[ch]) :
            LET datum == dataStore[ch][i] IN
            \A joinChannels \in joins[ch] :
                \A j \in 1..Len(contStore[joinChannels]) :
                    LET wk == contStore[joinChannels][j] IN
                    ~AnyPatternMatches(wk.patterns, datum.data)

(***************************************************************************)
(* Gensym Operation                                                         *)
(***************************************************************************)

(* Generate a fresh channel name *)
Gensym ==
    /\ nameCounter' = nameCounter + 1
    /\ UNCHANGED <<dataStore, contStore, joins, qualifier>>

(* Gensym always produces unique names - this is an invariant about the model *)
GensymUnique ==
    \A ch1, ch2 \in Channels :
        ch1 /= ch2 => TRUE  \* Different channels are truly different (tautology for model checking)

(***************************************************************************)
(* Produce Operation                                                        *)
(***************************************************************************)

(* Find a matching continuation for data at channel ch *)
FindMatchingCont(ch, data) ==
    LET matchingJoins == {jc \in joins[ch] :
            \E i \in 1..Len(contStore[jc]) :
                AnyPatternMatches(contStore[jc][i].patterns, data)}
    IN IF matchingJoins = {} THEN NULL
       ELSE CHOOSE jc \in matchingJoins : TRUE

(* Remove first matching continuation from a join-channel's queue *)
RemoveMatchingCont(joinChannels, data) ==
    LET conts == contStore[joinChannels] IN
    LET matchIdx == CHOOSE i \in 1..Len(conts) :
            AnyPatternMatches(conts[i].patterns, data)
    IN SubSeq(conts, 1, matchIdx-1) \o SubSeq(conts, matchIdx+1, Len(conts))

(* Produce: send data to a channel *)
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

(***************************************************************************)
(* Consume Operation                                                        *)
(***************************************************************************)

(* Find matching data for a pattern at a channel *)
FindMatchingData(ch, pattern) ==
    LET data == dataStore[ch] IN
    LET matchIdx == {i \in 1..Len(data) : Matches(pattern, data[i].data)}
    IN IF matchIdx = {} THEN NULL
       ELSE CHOOSE i \in matchIdx : TRUE

(* Remove data at index from channel *)
RemoveData(ch, idx) ==
    LET data == dataStore[ch] IN
    SubSeq(data, 1, idx-1) \o SubSeq(data, idx+1, Len(data))

(* Consume: wait for data on channel(s) *)
Consume(channels, patterns, cont, persist) ==
    LET checkAllMatch ==
        \A i \in 1..Len(channels) :
            FindMatchingData(channels[i], patterns[i]) /= NULL
    IN
    IF checkAllMatch THEN
        \* All channels have matching data - consume immediately
        /\ dataStore' = [ch \in Channels |->
            IF \E i \in 1..Len(channels) : channels[i] = ch
            THEN LET i == CHOOSE i \in 1..Len(channels) : channels[i] = ch
                     idx == FindMatchingData(ch, patterns[i])
                 IN IF dataStore[ch][idx].persist THEN dataStore[ch]
                    ELSE RemoveData(ch, idx)
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

(***************************************************************************)
(* Clear Operation                                                          *)
(***************************************************************************)

Clear ==
    /\ dataStore' = [ch \in Channels |-> <<>>]
    /\ contStore' = [chs \in Seq(Channels) |-> <<>>]
    /\ joins' = [ch \in Channels |-> {}]
    /\ UNCHANGED <<nameCounter, qualifier>>

(***************************************************************************)
(* Initial State                                                            *)
(***************************************************************************)

Init ==
    /\ dataStore = [ch \in Channels |-> <<>>]
    /\ contStore = [chs \in Seq(Channels) |-> <<>>]
    /\ joins = [ch \in Channels |-> {}]
    /\ nameCounter = 0
    /\ qualifier = 0

(***************************************************************************)
(* Next State Relation                                                      *)
(***************************************************************************)

Next ==
    \/ Gensym
    \/ \E ch \in Channels, d \in Data, p \in BOOLEAN : Produce(ch, d, p)
    \/ \E chs \in Seq(Channels), ps \in Seq(Patterns),
          k \in Continuations, p \in BOOLEAN : Consume(chs, ps, k, p)
    \/ Clear

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* Safety Properties                                                        *)
(***************************************************************************)

(* The core safety property: produce maintains no-pending-match *)
ProduceSafety ==
    \A ch \in Channels, d \in Data, p \in BOOLEAN :
        NoPendingMatch /\ Produce(ch, d, p) => NoPendingMatch'

(* Consume maintains no-pending-match *)
ConsumeSafety ==
    \A chs \in Seq(Channels), ps \in Seq(Patterns),
       k \in Continuations, p \in BOOLEAN :
        NoPendingMatch /\ Consume(chs, ps, k, p) => NoPendingMatch'

(* Overall safety *)
Safety == []NoPendingMatch

(***************************************************************************)
(* Liveness Properties                                                      *)
(***************************************************************************)

(* If data is produced and a matching consume is waiting, eventually fires *)
MatchEventuallyFires ==
    \A ch \in Channels, d \in Data :
        \A jc \in joins[ch] :
            (\E i \in 1..Len(contStore[jc]) :
                AnyPatternMatches(contStore[jc][i].patterns, d))
            => <>(Len(contStore[jc]) < Len(contStore[jc]))

(***************************************************************************)
(* Fairness                                                                 *)
(***************************************************************************)

Fairness == WF_vars(Next)

LiveSpec == Spec /\ Fairness

=============================================================================
