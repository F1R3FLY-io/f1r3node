> Last updated: 2026-03-23

# Crate: models

**Path**: `models/`

Central type definitions for the entire system. All protobuf-generated types, domain wrappers, Rholang AST, and deterministic collection types.

## Build System

`build.rs` compiles `.proto` files via `tonic-prost-build`. Proto sources are in `src/main/protobuf/` (12 compiled protos + `scalapb.proto`). Custom derives are applied to `rhoapi` messages: `serde::Serialize`, `serde::Deserialize`, `utoipa::ToSchema`, `Eq`, `Ord`, `PartialOrd`, `#[repr(C)]`. Post-processing removes auto-derived `PartialEq` from types needing custom equality (to exclude irrelevant fields).

## Block Domain Types

```rust
pub struct BlockHash(pub Vec<u8>);    // 32-byte block identifier
pub struct StateHash(pub Vec<u8>);    // 32-byte post-state hash
pub struct Validator(pub Vec<u8>);    // 65-byte public key

pub struct BlockMetadata {
    pub block_hash: BlockHash,
    pub parents: Vec<BlockHash>,
    pub sender: Validator,
    pub justifications: Vec<Justification>,
    pub weight_map: BTreeMap<Validator, i64>,
    pub block_number: i64,
    pub sequence_number: i32,
    pub invalid: bool,
    pub directly_finalized: bool,
    pub finalized: bool,
}
```

**`EquivocationRecord`** -- Tracks validator double-signing:
- `equivocator: Validator`
- `equivocation_base_block_seq_num: i32`
- `equivocation_detected_block_hashes: BTreeSet<BlockHash>`

## Deploy & Transfer Types (Protobuf)

**`DeployInfo`** (proto `DeployServiceCommon.proto`) -- Per-deploy metadata in block responses:
- Fields: `deployer`, `term`, `timestamp`, `sig`, `sigAlgorithm`, `phloPrice`, `phloLimit`, `validAfterBlockNumber`, `cost`, `errored`, `systemDeployError`
- `transfers: repeated TransferInfo` -- Inline transfer data populated by the block enricher

**`TransferInfo`** (proto) -- REV transfer extracted from deploy execution:
- `fromAddr: string` -- Sender address
- `toAddr: string` -- Recipient address
- `amount: int64` -- Amount in dust (smallest unit)
- `success: bool` -- Whether the transfer succeeded
- `failReason: string` -- Error message if success is false

## Casper Protocol Messages

**`CasperMessage`** enum wraps all consensus messages:
- `BlockHashMessage`, `BlockMessage`
- `ApprovedBlockCandidate`, `ApprovedBlock`, `BlockApproval`
- `BlockRequest`, `ForkChoiceTipRequest`
- `HasBlock`, `HasBlockRequest`
- `StoreItemsMessageRequest`, `StoreItemsMessage` (state sync)

**`ToPacket` trait** -- Converts proto messages to routing `Packet`s for network serialization.

## Rholang AST (Protobuf-generated `rhoapi` module)

The core process calculus representation:

| Type | Description |
|------|-------------|
| `Par` | Parallel composition (top-level container for all process terms) |
| `Send` | Send operation: `chan!(data)` |
| `Receive` | Receive/for: `for(x <- chan) { body }` |
| `New` | New scope: `new x in { body }` |
| `Match` | Pattern match: `match expr { case ... }` |
| `Bundle` | Access control: read-only, write-only, or both |
| `Expr` | Expressions via `ExprInstance` enum (literals, arithmetic, collections, methods) |
| `Var` | Variables via `VarInstance` enum (BoundVar, FreeVar, Wildcard) |
| `GUnforgeable` | Unforgeable names via `UnfInstance` (GPrivate, GDeployId, GDeployerId, GSysAuthToken) |
| `Connective` | Logical operators (AND, OR, NOT) for pattern matching |
| `TaggedContinuation` | Stored continuation (ParBody or ScalaBodyRef) |
| `PCost` | Phlogiston cost tracking per operation |

**Custom PartialEq/Hash**: Implemented manually on `Par`, `Send`, `Receive`, `New`, `Match`, `Bundle`, `Expr`, etc. to exclude auto-derived fields and ensure semantic equality.

## Sorted Collections (Deterministic Ordering)

Critical for consensus -- all nodes must process data in identical order.

**`SortedParHashSet`** -- HashSet<Par> + Vec<Par> for canonical iteration:
- `insert()`, `remove()`, `union()`, `contains()`
- Maintained via `Ordering::sort_pars()`

**`SortedParMap`** -- HashMap<Par, Par> + Vec<(Par, Par)> for canonical iteration:
- `insert()`, `extend()`, `remove()`, `apply()`, `get()`, `keys()`

**`ParSet`** / **`ParMap`** -- Wrappers adding metadata:
- `connective_used: bool`, `locally_free: Vec<u8>`, `remainder: Option<Var>`

**Sorter module** (`rholang/sorter/`): 10 sort matchers implementing `Sortable<T>` trait with `sort_match(term) -> ScoredTerm<T>` for deterministic canonical ordering of all Rholang term types (par, send, receive, new, match, bundle, connective, expr, unforgeable, var). Supporting modules: `sortable.rs` (trait), `score_tree.rs` (scoring), `ordering.rs`.

## Tests

Test files: `scored_term_sort_test.rs`, `sorted_par_map_test.rs`, `var_sort_matcher_test.rs`, `par_sort_matcher_test.rs`, `json_encoder_test.rs`, `sorted_par_hash_set_test.rs`, `pathmap_integration_tests.rs`.

## PathMap Integration

Supports indexed Rholang structures:
- `RholangPathMap = PathMap<Par>` type alias
- `par_to_path()` -- Converts Par to byte segment path via S-expression encoding
- `SExpr` encoding with compact byte tags (NewVar, Symbol, VarRef, Arity)
- `RholangReadZipper` / `RholangWriteZipper` -- Navigation cursors

## Utility Types

- `ParExt` trait -- Extractors: `get_g_string()`, `get_g_int()`, `get_g_bool()`, `get_e_tuple_body()`
- `BundleOps` -- Bundle merge (AND semantics for flags), display
- `StringOps` -- Hex decoding compatible with Scala `Base16.decode()`
- `NormalizerEnv` -- Deploy execution context (`rho:system:deployId`, `rho:system:deployerId`)
- `ParToSExpr` -- Process-to-S-expression conversion for PathMap paths

**See also:** [models/ crate README](../../models/README.md)

[← Back to docs index](../README.md)
