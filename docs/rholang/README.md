> Last updated: 2026-04-14

# Crate: rholang (Interpreter/Reducer)

**Path**: `rholang/`

The Rholang language interpreter: parsing, normalization, evaluation, cost accounting, and system processes.

## Compilation Pipeline

```
Rholang Source Code
    |
    | rholang_parser::RholangParser::parse()
    v
Vec<AnnProc> (Annotated AST)
    |
    | normalize_ann_proc()
    v
Par (Normalized, sorted canonical form)
    |
    | ParSortMatcher::sort_match()
    v
Par (Deterministic ordering for consensus)
```

**Compiler entry**: `Compiler::source_to_adt(source)` or `source_to_adt_with_normalizer_env(source, env)`

## Normalization

Transforms parsed AST into canonical `Par` form with De Bruijn indices.

**Process normalizers** (each handles one Rholang construct):

| Normalizer | Rholang Syntax | Output |
|-----------|---------------|--------|
| `normalize_p_send` | `chan!(data)` | `Send { chan, data, persistent }` |
| `normalize_p_input` | `for(x <- chan) { body }` | `Receive { binds, body }` |
| `normalize_p_par` | `p \| q` | `Par { processes }` |
| `normalize_p_new` | `new x in { body }` | `New { bindings, body }` |
| `normalize_p_eval` | `*chan` | `Expr { EVar }` |
| `normalize_p_match` | `match expr { case ... }` | `Match { cases }` |
| `normalize_p_bundle` | `bundle+{ ... }` | `Bundle { body, flags }` |
| `normalize_p_contr` | `contract ch(x) = body` | Contract extraction |
| `normalize_p_let` | `let x = val in body` | Inline substitution |
| `normalize_p_if` | `if(cond) { ... } else { ... }` | Conditional Par |
| `normalize_p_ground` | `"str"`, `42`, `true` | `Expr { GString/GInt/GBool }` |
| `normalize_p_conjunction` | `/\` | `Connective { AND }` |
| `normalize_p_disjunction` | `\/` | `Connective { OR }` |
| `normalize_p_negation` | `~` | `Connective { NOT }` |
| `normalize_p_method` | `.method(args)` | `Expr { EMethod }` |
| `normalize_p_send_sync` | `chan!!(data)` | Synchronous send |
| `normalize_p_matches` | `matches` | Matches expression |
| `normalize_p_collect` | `collect` | Collection expression |
| `normalize_p_simple_type` | type literals | Simple type |
| `normalize_p_var` | `x` | Variable reference |
| `normalize_p_var_ref` | `=x` | Variable reference (explicit) |

**Binding management**: `BoundMapChain` tracks nested variable scopes; `FreeMap` detects unbound variables. Validation rejects top-level free variables, wildcards, or connectives.

## Evaluation (Reducer)

**`DebruijnInterpreter::eval(par, env, rand)`** is the core evaluator:

```
eval_inner():
  1. Group par terms: Sends, Receives, News, Matches, Bundles, Exprs
  2. tokio::spawn each term as an independent task (matching Scala's parTraverse)
  3. For each term type:
     - Send  -> space.produce(channel, data, persistent).await
     - Receive -> space.consume(channels, patterns, continuation, persistent, peeks).await
     - New -> create unforgeable names via Blake2b512Random, eval body
     - Match -> evaluate target, try each case pattern
     - Bundle -> evaluate in restricted scope
     - Expr -> evaluate arithmetic/logic/collection operations
  4. If produce/consume returns a match -> dispatch continuation
  5. Await all spawned tasks, aggregate errors (parTraverseSafe pattern)
```

**Parallel evaluation**: Par branches (`A | B | C`) are dispatched as independent
tokio tasks via `tokio::spawn`, matching Scala's `parTraverse` for true parallel
execution. Per-channel FIFO locks in RSpace (`tokio::sync::Mutex`) ensure
deterministic COMM matching. The `ReplayRSpace` event log oracle forces the same
COMMs during replay regardless of scheduling order.

**Stack safety**: `StackGrowingFuture` wraps recursive calls with `stacker::maybe_grow()` to dynamically expand the thread stack (1MB red zone).

**Variable substitution**: `Substitute` trait replaces De Bruijn-indexed variables. Each substitution is charged `O(term.encoded_len())` phlogiston.

**Arithmetic overflow safety**: Division and negation operations guard against i64 overflow. `i64::MIN / -1` and `i64::MIN.neg()` return `InterpreterError::ReduceError("Arithmetic overflow in ...")` as soft deploy failures rather than panics.

## Cost Accounting (Phlogiston)

Every operation is metered to prevent resource exhaustion:

```rust
pub struct CostManager {
    state: Arc<Mutex<Cost>>,
    log: Arc<Mutex<VecDeque<Cost>>>,
    max_log_entries: usize,
}
```

- `charge(cost)` -- Subtract from remaining budget via `lock()` (blocking, thread-safe under concurrent `tokio::spawn` tasks); returns `OutOfPhlogistonsError` if exhausted. Uses saturating arithmetic to prevent overflow with `i64::MAX` budgets.
- `set(cost)` -- Reset balance
- `get()` -- Query remaining

**Cost table** (representative values):

| Operation | Cost |
|-----------|------|
| Arithmetic (add, sub, mul, div, mod) | 3-9 phlos |
| Boolean AND/OR | 2 phlos |
| Comparison | 3 phlos |
| Collection lookup/add/remove | 3 phlos each |
| `hex_to_bytes` | O(string_length) |
| `produce` | O(channel_size + data_size) |
| `consume` | O(channels * patterns * continuation) |
| Parsing | O(source_code_length) |
| Substitution | O(result_term_size) |

## System Processes (Built-in Channels)

Rholang operations implemented in Rust, accessible via fixed unforgeable channel names:

| Channel | Byte ID | Purpose |
|---------|---------|---------|
| `stdout` | 0 | Standard output |
| `stderr` | 2 | Error output |
| `ed25519_verify` | 4 | Ed25519 signature verification |
| `sha256_hash` | 5 | SHA-256 hashing |
| `keccak256_hash` | 6 | Keccak-256 hashing |
| `blake2b256_hash` | 7 | Blake2b-256 hashing |
| `secp256k1_verify` | 8 | Secp256k1 signature verification |
| `get_block_data` | 10 | Read current block info |
| `get_invalid_blocks` | 11 | Invalid block list |
| `vault_address` | 12 | Vault address derivation |
| `deployer_id_ops` | 13 | Deployer identity operations |
| `reg_lookup` | 14 | Registry lookup |
| `reg_insert_random` | 15 | Registry insert (random URI) |
| `reg_insert_signed` | 16 | Registry insert (signed) |
| `gpt4` | 20 | OpenAI GPT-4 (non-deterministic) |
| `dalle3` | 21 | OpenAI DALL-E 3 (non-deterministic) |
| `text_to_audio` | 22 | OpenAI TTS (non-deterministic) |
| `grpc_tell` | 25 | External gRPC client |
| `dev_null` | 26 | Discard output |
| `abort` | 27 | Abort computation |
| `ollama_chat` | 28 | Ollama chat (non-deterministic) |
| `ollama_generate` | 29 | Ollama text generation (non-deterministic) |
| `ollama_models` | 30 | Ollama model list |
| `deploy_data` | 31 | Deploy data access |

Non-deterministic processes (OpenAI, Ollama) are specially marked for replay handling during consensus.

### TTS System Process

The `text_to_audio` channel (byte ID 22) invokes OpenAI text-to-speech via `create_audio_speech`:

```rust
pub async fn create_audio_speech(
    &self,
    input: &str,
    output_path: &str,
) -> Result<Vec<u8>, InterpreterError>
```

The implementation writes audio to a UUID-based temp filename (`audio_{uuid}.mp3`) to prevent race conditions when multiple TTS calls run concurrently. It reads the file back with `tokio::fs::read` (async, non-blocking) and cleans up the temp file with `tokio::fs::remove_file` after reading. The caller wraps the returned bytes as `RhoByteArray::create_par(bytes)`, produces the output on the ack channel, and returns it.

### NonDeterministicProcessFailure Error Pattern

All 6 non-deterministic processes (GPT-4, DALL-E 3, TTS, Ollama chat, Ollama generate, Ollama models) follow a uniform two-stage error pattern that preserves output for replay:

**Stage 1 -- Service call failure**: The external service call itself fails (network error, API error, etc.). No output has been produced yet, so there is nothing to preserve:

```rust
Err(InterpreterError::NonDeterministicProcessFailure {
    cause: Box::new(e),
    output_not_produced: vec![],
})
```

**Stage 2 -- Produce call failure**: The service call succeeded and output was generated, but the subsequent `produce` into RSpace fails. The output is serialized so it can be replayed:

```rust
Err(InterpreterError::NonDeterministicProcessFailure {
    cause: Box::new(e),
    output_not_produced: output.iter().map(|p| p.encode_to_vec()).collect(),
})
```

During replay, `DispatchType::FailedNonDeterministicCall` triggers `Produce::with_error()` to mark the produce event as failed, ensuring the replay matches the original execution trace.

## Dispatch

**`RholangAndScalaDispatcher`** routes continuations:
- `DeterministicCall` -- Standard Rholang evaluation
- `NonDeterministicCall(Vec<Vec<u8>>)` -- External service with cached output
- `FailedNonDeterministicCall(err)` -- Failed external call
- `Skip` -- No-op

## ChargingRSpace

Wraps any `ISpace` to add cost metering:
```rust
pub fn charging_rspace<T: ISpace>(space: T, cost: _cost) -> impl ISpace
```
Charges `storage_cost_produce` or `storage_cost_consume` before each operation.

When a COMM fires, `handle_result` applies order-independent charging:
1. Refund ALL pre-charged storage costs unconditionally (no `triggered_by` identity check)
2. Charge unified `comm_event_storage_cost(N)` where N = consume's channel count
3. Re-charge for persistent items that remain in the tuplespace

This ensures the same COMM produces identical cost regardless of which operation
(produce or consume) triggered it — required for deterministic cost under
parallel `tokio::spawn` evaluation.

## RhoRuntime

```rust
pub trait RhoRuntime: Send + Sync {
    async fn evaluate(&mut self, term: &str, cost: Cost, env: HashMap<String, Par>,
                      rand: Blake2b512Random) -> Result<EvaluateResult, InterpreterError>;
    async fn inj(&mut self, par: Par, env: &Env<Par>, rand: Blake2b512Random)
        -> Result<(), InterpreterError>;
    async fn create_checkpoint(&mut self) -> Checkpoint;
    async fn create_soft_checkpoint(&mut self) -> SoftCheckpoint<...>;
    async fn reset(&mut self, root: &Blake2b256Hash) -> Result<(), InterpreterError>;
    // ... other checkpoint/state operations (all async)
}
```

`RhoRuntimeImpl` is the concrete implementation connecting the interpreter to RSpace.
The `RhoISpace` type is `Arc<Box<dyn ISpace<...> + Send + Sync>>` — no outer mutex,
as the async ISpace provides its own interior mutability and per-channel locking.

## Pattern Matching

**`Matcher`** implements `Match<BindPattern, ListParWithRandom>` for RSpace.

**`SpatialMatcherContext`** performs spatial matching on `Par` terms:
- Variable bindings (unification)
- Logical connectives (AND, OR, NOT patterns)
- Structural patterns with remainder
- Uses `maximum_bipartite_match` for optimal matching of unordered sets

## Registry

`registry.rs` -- URI-based name registry using base32 encoding with CRC14 checksum.
`registry_bootstrap.rs` -- Built-in registry contracts installed at genesis.

**Legacy URI aliases**: For backward compatibility with older clients (e.g., rust-client), the following `rho:rchain:*` URIs are aliased to their canonical `rho:system:*` equivalents:
- `rho:rchain:pos` -> `rho:system:pos` (PoS contract)
- `rho:rchain:revVault` -> `rho:vault:system` (SystemVault)
- `rho:rchain:deployId` -> `rho:system:deployId` (deploy context)
- `rho:rchain:deployerId` -> `rho:system:deployerId` (deploy context)

Aliases are resolved at the registry and normalizer level. Usage is logged at debug level (`f1r3fly.legacy-uri`).

## FFI (lib.rs)

C interface for Scala JNA interop:
- `create_rho_runtime()`, `evaluate()`, `inj()`, `create_soft_checkpoint()`, `revert_to_soft_checkpoint()`
- Memory management: `rholang_get_allocated_bytes()`, `rholang_deallocate_memory()`

## Tests

Test suites in `tests/`: `interpreter_spec.rs`, `reduce_spec.rs`, `substitute_test.rs`, `crypto_channels_spec.rs`, `abort_spec.rs`, `deploy_data_spec.rs`, `demo_verification.rs`, `getsubtrie_spec.rs`, `setsubtrie_spec.rs`, `replay_memory_profile_spec.rs`, `ollama_integration_test.rs`, `openai_service_spec.rs`, `zipper_*_spec.rs` (3 files), `accounting/cost_accounting_spec.rs`, `accounting/non_deterministic_processes_spec.rs`, `matcher/match_test.rs`.

The `non_deterministic_processes_spec.rs` suite contains 9 replay consistency tests using the `evaluate_and_replay()` helper, which follows a play-checkpoint-rig-replay-verify cycle: evaluate the Rholang term, create a soft checkpoint, configure mock services to fail, replay from the checkpoint, and verify the replay trace matches the original. Tests cover GPT-4, DALL-E 3, TTS, and gRPC tell for both success and error cases.

**See also:** [rholang/ crate README](../../rholang/README.md) | [Rholang Tutorial](./rholangtut.md) | [Pattern Matching](./rholangmatchingtut.md) | [Ollama Integration](./ollama.md)

[← Back to docs index](../README.md)
