# Concurrent RSpace: Maximizing Parallelism in Rholang's Tuple Space

## Background

RSpace is the tuple space at the core of the Rholang runtime. It mediates all inter-process communication through two operations — `produce` (deposit a datum on a channel) and `consume` (register a continuation waiting for data) — with a COMM event firing when both sides match. Rholang's Par operator (`|`) composes processes concurrently; the runtime evaluates each branch as an asynchronous future. Join patterns (`for(@x <- A & @y <- B) { P }`) create cross-channel dependencies by consuming from multiple channels atomically.

For consensus, evaluation must be deterministic: all validators must produce identical state hashes, event logs, and phlogiston (gas) costs. The play/replay model requires that an observer replaying a block reproduces the exact COMM events and costs recorded by the block creator. A COST_MISMATCH — any difference in total phlogiston — causes the observer to reject the block.

## 1. Problem: Global Serialization

The `rust/dev` baseline uses `futures::future::join_all` for concurrent Par evaluation. However, `join_all` delivers no actual concurrency because two global mutexes serialize all RSpace operations:

```
    ┌─────────────┐      ┌──────────────────────────────────────────┐
    │ Interpreter │      │ Arc<tokio::sync::Mutex<dyn ISpace>>      │
    │             │      │ (global lock — all futures wait here)    │
    │ eval(Par)   │      │                                          │
    │     │       │      │ ┌──────────────────────────────────────┐ │
    │     ▼       │      │ │ InMemHotStore                        │ │
    │ join_all    │      │ │                                      │ │
    │ ┌──┬──┬──┐  │      │ │ ┌──────────────────────────────────┐ │ │
    │ │f₀│f₁│f₂│──┼──────▶ │ │ Mutex<HotStoreState>             │ │ │
    │ └──┴──┴──┘  │      │ │ │  ┌─────────────────────────────┐ │ │ │
    │  (all block │      │ │ │  │ DashMap DashMap DashMap...  │ │ │ │
    │   on Mutex) │      │ │ │  │ (unused concurrency)        │ │ │ │
    │             │      │ │ │  └─────────────────────────────┘ │ │ │
    └─────────────┘      │ │ └──────────────────────────────────┘ │ │
                         │ └──────────────────────────────────────┘ │
                         └──────────────────────────────────────────┘
```

Every produce and consume — even on independent channels — waits for the same lock. The DashMap concurrent hashmaps inside the hot store provide per-shard RwLocks, but a `Mutex<HotStoreState>` wrapper defeats this entirely.

The Scala node avoided this through `TwoStepLock` with per-channel `Semaphore`s (`MultiLock.scala`) and `Ref[F, HotStoreState]` for atomic state updates. The Rust port collapsed this into a single global mutex.

| Concern             | Scala Node                                                           | `rust/dev` Baseline                      | Refactored Rust                                       |
|---------------------|----------------------------------------------------------------------|------------------------------------------|-------------------------------------------------------|
| Hot store state     | `Ref[F, HotStoreState]` — atomic snapshots                           | `Mutex<HotStoreState>` wrapping DashMaps | DashMap accessed directly — per-shard RwLocks         |
| Per-channel locking | `TwoStepLock` + `MultiLock` — per-channel `Semaphore`s via `TrieMap` | None — single global mutex               | `DashMap<u64, Mutex<()>>` — per-channel-group mutexes |
| Interpreter access  | `F[_]: Concurrent` — shared monadic access                           | `Arc<tokio::sync::Mutex<dyn ISpace>>`    | `Arc<Box<dyn ISpace>>` — no lock                      |
| ISpace mutability   | `F[_]` effect type — no `&mut self`                                  | `&mut self` on all 12 methods            | `&self` with interior mutability                      |
| Eval loop           | `parTraverseSafe`                                                    | `join_all` — serialized by global mutex  | `FuturesUnordered` + two-phase dispatch               |
| Candidate ordering  | `Random.shuffle`                                                     | `thread_rng().shuffle`                   | `sort_by(source.hash)` — deterministic                |
| Cost accounting     | `Ref[F, Cost]` — atomic                                              | `Arc<Mutex<Cost>>` — TOCTOU bug          | `AtomicI64` + CAS loop                                |

## 2. Refactored Architecture

Six phases remove the serialization, each independently testable:

```
    ┌─────────────┐
    │ Interpreter │
    │             │
    │ eval(Par)   │
    │     │       │
    │     ▼       │
    │ FuturesUn-  │     ┌─────────────────────────────────────────────┐
    │ ordered     │     │ Arc<Box<dyn ISpace>>  (no Mutex)            │
    │ ┌──┬──┬──┐  │     │                                             │
    │ │f₀│f₁│f₂│──┼─────▶ ┌─────────────────────────────────────────┐ │
    │ └──┴──┴──┘  │     │ │ ConcurrentHotStore  (no Mutex)          │ │
    │             │     │ │  DashMap ─── per-shard RwLocks ─── ◄──┐ │ │
    │ Deferred    │     │ │  DashMap ─── per-shard RwLocks ─── ◄──┤ │ │
    │ Queue       │     │ │  DashMap ─── per-shard RwLocks ─── ◄──┘ │ │
    │ ┌────────┐  │     │ └─────────────────────────────────────────┘ │
    │ │ Phase 2│  │     │                                             │
    │ │(bodies)│  │     │ channel_locks: DashMap<hash, Mutex>         │
    │ └────────┘  │     │ (per-group, joins only)                     │
    └─────────────┘     └─────────────────────────────────────────────┘
```

## 3. Phase 1 — Lock-Free Cost Accounting

The cost manager used `Arc<Mutex<Cost>>` with a two-step check-and-deduct that had a TOCTOU (time-of-check-to-time-of-use) race: between unlocking after deduction and re-locking for verification, another thread could deduct past zero:

```
    Thread A                         Thread B
    ────────                         ────────
    lock()
    read cost = 100
    deduct 60 → cost = 40
    unlock()
                                     lock()
                                     read cost = 40
                                     deduct 50 → cost = −10
                                     unlock()
    lock()
    read cost = −10 → error!
    (too late — Thread B already overspent)
```

Replaced with `AtomicI64` + CAS (compare-and-swap) loop. CAS is a hardware atomic instruction (`CMPXCHG` on x86, `LDXR`/`STXR` on ARM) that couples the check and deduction into a single indivisible operation. This is the Rust-native equivalent of Scala's `Ref[F].modify`.

```
    ╔═══════════════════════════════════════════════════════════╗
    ║  ALGORITHM: Lock-Free Cost Charge                         ║
    ╠═══════════════════════════════════════════════════════════╣
    ║                                                           ║
    ║  ── value is an AtomicI64 (remaining phlogiston) ──       ║
    ║                                                           ║
    ║  procedure CHARGE(amount):                                ║
    ║  ┌─ loop:                                                 ║
    ║  │    current ← LOAD(value, Acquire)                      ║
    ║  │    if current < 0 then                                 ║
    ║  │      return OutOfPhlogistonsError                      ║
    ║  │    new_value ← current − amount                        ║
    ║  │    if CAS(value, expected=current, desired=new_value)  ║
    ║  │      if new_value < 0 then                             ║
    ║  │        return OutOfPhlogistonsError                    ║
    ║  │      return Ok                                         ║
    ║  └─ retry with fresh load                                 ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
```

Additionally, COMM cost accounting differed depending on which side triggered the COMM (produce vs consume fired different refund paths). The fix normalizes all COMMs to produce-triggered semantics in `charging_rspace.rs`, making the total cost commutative:

  Σᵢ cost(opᵢ) = Σᵢ cost(op_σ(i))  ∀ permutations σ

## 4. Phase 2 — Exposing DashMap Concurrency

DashMap partitions entries into shards, each with its own RwLock. Operations on different shards proceed without contention:

```
    DashMap<Channel, Vec<Datum>>
    ┌──────────────────────────────────────────────────┐
    │  Shard 0          Shard 1          Shard 2       │
    │  ┌──────────┐    ┌──────────┐    ┌──────────┐    │
    │  │ RwLock   │    │ RwLock   │    │ RwLock   │    │
    │  │ ┌──────┐ │    │ ┌──────┐ │    │ ┌──────┐ │    │
    │  │ │ ch_A │ │    │ │ ch_B │ │    │ │ ch_C │ │    │
    │  │ │ ch_D │ │    │ │ ch_E │ │    │ │ ch_F │ │    │
    │  │ └──────┘ │    │ └──────┘ │    │ └──────┘ │    │
    │  └──────────┘    └──────────┘    └──────────┘    │
    └──────────────────────────────────────────────────┘
```

The `Mutex<HotStoreState>` wrapper was removed — hot store methods now access DashMaps directly. The `HistoryStoreCache` wrapper was also removed.

```
    Before: self.hot_store_state.lock().unwrap().data.get(channel)
                 ▲                                  ▲
            global Mutex                   DashMap shard lock
            (serializes all)                (per-shard, wasted)

    After:  self.state.data.get(channel)
                            ▲
                     DashMap shard lock
                     (per-shard, utilized)
```

## 5. Phase 3 — Interior Mutability

The `ISpace` trait required `&mut self` on all methods, forcing exclusive access via the global mutex. Changed all 12 methods to `&self` using interior mutability:

| Field                | Before                            | After                        |
|----------------------|-----------------------------------|------------------------------|
| `event_log`          | `Vec<Event>`                      | `Arc<Mutex<Vec<Event>>>`     |
| `produce_counter`    | `BTreeMap<Produce, i32>`          | `Arc<Mutex<BTreeMap<...>>>`  |
| `history_repository` | `Arc<Box<dyn HistoryRepository>>` | `Arc<RwLock<Arc<Box<...>>>>` |
| `store`              | `Arc<Box<dyn HotStore>>`          | `Arc<RwLock<Arc<Box<...>>>>` |

`RwLock` on `history_repository` and `store` allows concurrent reads during produce/consume, with exclusive access only during `create_checkpoint` and `reset` (called between deploys).

## 6. Phase 4 — Per-Channel-Group Locks

Most Rholang channels are private unforgeable names with no contention. Join patterns create cross-channel dependencies requiring atomicity:

```
    Thread 1 (produce @A)            Thread 2 (produce @B)
    ─────────────────────            ─────────────────────
    read @A: has continuation
    read @B: no data yet            read @B: has continuation
      → store data on @A            read @A: no data yet
                                      → store data on @B

    Result: Both stored data. Neither fired the COMM.
    The join continuation starves.
```

Per-channel-group locks solve this with ordered acquisition (preventing deadlock):

```
    ╔═══════════════════════════════════════════════════════════╗
    ║  ALGORITHM: Channel Group Lock Acquisition                ║
    ╠═══════════════════════════════════════════════════════════╣
    ║                                                           ║
    ║  ── channel_locks is a DashMap<u64, Arc<Mutex<()>>> ──    ║
    ║                                                           ║
    ║  procedure LOCK_CHANNEL_GROUP(channels):                  ║
    ║    hashes ← [HASH(ch) for ch in channels]                 ║
    ║    SORT(hashes)                                           ║
    ║    key ← HASH(hashes)                                     ║
    ║    lock ← channel_locks.GET_OR_INSERT(key, new Mutex)     ║
    ║    return lock.ACQUIRE()                                  ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
```

```
    ┌────────────────────────────────────────────────────┐
    │  Channel Operations and Their Lock Requirements    │
    │                                                    │
    │  priv_1!(data)           → DashMap shard lock only │
    │  for(@x <- priv_2){ P }  → DashMap shard lock only │
    │                                                    │
    │  for(@x <- A & @y <- B){ P }                       │
    │    produce(@A, v)  → channel_group_lock({A,B})     │
    │    produce(@B, v)  → channel_group_lock({A,B})     │
    │                      (same lock, serialized)       │
    └────────────────────────────────────────────────────┘
```

This is simpler than Scala's two-step lock (which acquires Phase A locks on initial channels, discovers join groups, then acquires Phase B locks on discovered channels). The Rust design hashes the entire channel group in one step.

## 7. Phase 5 — Removing the Interpreter Lock

With ISpace methods taking `&self` (Phase 3) and per-channel-group locks handling concurrency (Phase 4), the interpreter-level `Arc<tokio::sync::Mutex<dyn ISpace>>` is redundant:

```
    Before:                              After:

    Arc<tokio::sync::Mutex<              Arc<Box<dyn ISpace>>
         Box<dyn ISpace>>>                    │
              │                          self.space.produce(...)
    self.space.try_lock().unwrap()
              │
    space_locked.produce(...)
              │
    drop(space_locked)
```

Removed 19 `.try_lock().unwrap()` call sites across `reduce.rs`, `rho_runtime.rs`, `contract_call.rs`, and `interpreter.rs`.

## 8. Phase 6 — Content-Hash Ordering, FuturesUnordered, Two-Phase Dispatch

### 8.1 Content-Hash Candidate Ordering

Both Scala (`Random.shuffle`) and `rust/dev` (`thread_rng().shuffle`) randomize candidate ordering for fairness. This breaks consensus under concurrent evaluation — different shuffle seeds produce different COMMs.

Replaced with deterministic sorting by `source.hash` (Blake2b256). The avalanche property of cryptographic hashes ensures uniform distribution across the ordering space — providing fairness without randomness.

```
    ╔═══════════════════════════════════════════════════════════╗
    ║  ALGORITHM: Content-Hash Deterministic Ordering           ║
    ╠═══════════════════════════════════════════════════════════╣
    ║                                                           ║
    ║  procedure ORDER_BY_HASH(candidates, hash_fn):            ║
    ║    indexed ← [(c, i) for i, c in ENUMERATE(candidates)]   ║
    ║    SORT(indexed, key = λ(c, _). hash_fn(c))               ║
    ║    return indexed                                         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
```

### 8.2 FuturesUnordered

Replaced the eval loop's `join_all` with `FuturesUnordered`, which polls whichever future is ready next:

```
    Sequential for-loop:
    ── f₀ ──────────────────────────────────▶ complete
                  ── f₁ ────────────────────▶ complete
                                ── f₂ ──────▶ complete

    FuturesUnordered:
    ── f₀ ──── yield ──────── resume ───────▶ complete
         ── f₁ ──── yield ──────── resume ──▶ complete
              ── f₂ ────────────────────────▶ complete
```

With Phases 1–5 complete, produce/consume no longer block on a global lock, so concurrent Par branches make real progress on independent channels.

### 8.3 The Body Interleaving Problem

Naively replacing the loop with `FuturesUnordered` caused a −644 phlogiston COST_MISMATCH. When a COMM fires, the continuation body is evaluated inline via `dispatch → eval()`. This body evaluation yields at `.await` points, allowing other futures to interleave. RSpace (play) and ReplayRSpace (replay) have different code paths with different yield points, producing different interleaving and different costs:

```
    FuturesUnordered WITHOUT two-phase (broken):
    ════════════════════════════════════════════════════════════

    ── f₀ (recv) ─── COMM fires ─── dispatch(body₀) ─────────┐
                     ── f₁ (send) ───────────────────────────┤ ◄─ interleaves!
    ── f₀ body₀ continues ───────────────────────────────────┤
                     ── f₁ COMM body₁ ───────────────────────┤
                                                             ▼
    Measured: consistent −644 phlogiston COST_MISMATCH.
```

### 8.4 Two-Phase Dispatch

The fix separates matching (concurrent) from body evaluation (sequential):

```
    FuturesUnordered WITH two-phase dispatch (correct):
    ════════════════════════════════════════════════════════════

    Phase 1 — concurrent matching:
    ── f₀ (recv) ─── COMM ─── defer(body₀) ──────────────────▶ done
                     ── f₁ (send) ─── COMM ─── defer(body₁) ─▶ done

    Phase 2 — sequential body dispatch:
    ── body₀ ────────────────────────────────────────▶ complete
                     ── body₁ ───────────────────────▶ complete
                                                           ──▶ done
```

```
    ╔═══════════════════════════════════════════════════════════╗
    ║  ALGORITHM: Two-Phase Concurrent Evaluation               ║
    ╠═══════════════════════════════════════════════════════════╣
    ║                                                           ║
    ║  procedure EVAL_INNER(par, env, rand):                    ║
    ║    terms ← COLLECT_TERMS(par)                             ║
    ║    futures ← [EVAL_TERM(t, env, SPLIT(rand, i))           ║
    ║               for i, t in ENUMERATE(terms)]               ║
    ║                                                           ║
    ║    ── Phase 1: concurrent matching ──                     ║
    ║    deferred ← new SharedQueue()                           ║
    ║    SET_DEFERRED_MODE(deferred)                            ║
    ║    results ← FUTURES_UNORDERED(futures)                   ║
    ║    CLEAR_DEFERRED_MODE()                                  ║
    ║                                                           ║
    ║    ── Phase 2: sequential body dispatch ──                ║
    ║    bodies ← DRAIN_AND_SORT(deferred, by insertion index)  ║
    ║    for body in bodies:                                    ║
    ║      DISPATCH(body)                                       ║
    ║      ── dispatch calls EVAL_INNER recursively ──          ║
    ║      ── (creates its own Phase 1/Phase 2 cycle) ──        ║
    ║                                                           ║
    ║    return AGGREGATE_ERRORS(results)                       ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
```

Nested `eval` calls from Phase 2 dispatch create their own two-phase cycle:

```
    eval_inner (outer)
    ├── Phase 1: FuturesUnordered
    │   ├── produce → COMM → defer(body₀)
    │   └── consume → COMM → defer(body₁)
    │
    └── Phase 2: sequential dispatch
        ├── body₀ → eval_inner (nested, own cycle)
        │   ├── Phase 1: FuturesUnordered
        │   │   └── produce → COMM → defer(body₀₀)
        │   └── Phase 2: dispatch body₀₀
        │
        └── body₁ → eval_inner (nested, own cycle)
            ├── Phase 1: FuturesUnordered
            │   └── consume → COMM → defer(body₁₀)
            └── Phase 2: dispatch body₁₀
```

### 8.5 What Runs in Parallel

```
    ┌──────────────────────────────────────────────────────────┐
    │                    Concurrent (Phase 1)                  │
    │                                                          │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
    │  │ produce  │  │ consume  │  │ produce  │  │ consume  │  │
    │  │ (ch_A)   │  │ (ch_B)   │  │ (ch_C)   │  │ (ch_D)   │  │
    │  │          │  │          │  │          │  │          │  │
    │  │ pattern  │  │ pattern  │  │ pattern  │  │ pattern  │  │
    │  │ matching │  │ matching │  │ matching │  │ matching │  │
    │  │          │  │          │  │          │  │          │  │
    │  │ hot store│  │ hot store│  │ hot store│  │ hot store│  │
    │  │ read/    │  │ read/    │  │ read/    │  │ read/    │  │
    │  │ write    │  │ write    │  │ write    │  │ write    │  │
    │  │          │  │          │  │          │  │          │  │
    │  │ atomic   │  │ atomic   │  │ atomic   │  │ atomic   │  │
    │  │ cost     │  │ cost     │  │ cost     │  │ cost     │  │
    │  │ charge   │  │ charge   │  │ charge   │  │ charge   │  │
    │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
    ├──────────────────────────────────────────────────────────┤
    │                   Sequential (Phase 2)                   │
    │                                                          │
    │  ┌─────────────────┐  ┌─────────────────┐                │
    │  │ body₀ dispatch  │  │ body₁ dispatch  │                │
    │  │ (recursive eval)│→ │ (recursive eval)│→ done          │
    │  └─────────────────┘  └─────────────────┘                │
    └──────────────────────────────────────────────────────────┘
```

Produce/consume matching (pattern matching, hot store reads/writes, DashMap shard locks, atomic cost accounting) runs concurrently. Only COMM body callbacks — which recurse into `eval()` and create arbitrary new state — are sequenced.

## 9. Determinism Guarantees

**State hash.** The history trie is content-addressed. The root hash depends only on the set of stored key-value pairs, not insertion order:

```
H({(k₁,v₁), …, (kₙ,vₙ)}) = H({(k_σ(1),v_σ(1)), …, (k_σ(n),v_σ(n))})  ∀ permutations σ
```

**Cost.** The CAS-based cost manager makes the total cost equal to the sum of all charges regardless of interleaving. Cost normalization makes COMM cost identical regardless of which side fires.

**Events.** Content-hash ordering ensures the same pending datums and continuations always produce the same COMM match.

**RNG.** `Blake2b512Random` is split by term index (determined at Par construction, not evaluation time). The COMM dispatcher merges continuation and data random states — same inputs regardless of which side fires.

## 10. Summary

| Phase | `rust/dev` Baseline                      | Refactored                                  | Parallelism Gained                                    | Scala Equivalent                                      |
|-------|------------------------------------------|---------------------------------------------|-------------------------------------------------------|-------------------------------------------------------|
| 1     | `Arc<Mutex<Cost>>` with TOCTOU           | `AtomicI64` + CAS loop                      | Concurrent cost charges without lock contention       | `Ref[F, Cost]`                                        |
| 2     | `Mutex<HotStoreState>` wrapping DashMaps | DashMaps accessed directly                  | Per-shard DashMap concurrency                         | `Ref[F, HotStoreState]` with immutable `Map`s         |
| 3     | `&mut self` on all ISpace methods        | `&self` + interior mutability               | Shared concurrent access to RSpace                    | `F[_]: Concurrent` effect type                        |
| 4     | Global mutex (no per-channel locking)    | Per-channel-group `Mutex<()>` via `DashMap` | Independent channels fully parallel; joins serialized | `TwoStepLock` + `MultiLock` with `Semaphore[F]`       |
| 5     | `Arc<tokio::sync::Mutex<dyn ISpace>>`    | `Arc<Box<dyn ISpace>>`                      | Direct `&self` access, no global bottleneck           | (N/A — Scala never had this)                          |
| 6     | `join_all` (serialized by global mutex)  | `FuturesUnordered` + two-phase dispatch     | Concurrent matching with deterministic body dispatch  | `parTraverseSafe` (Scala avoids interleaving via STM) |
