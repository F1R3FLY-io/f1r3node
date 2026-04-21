> Last updated: 2026-04-21

# Crate: node (Orchestrator/Entry Point)

**Path**: `node/`

Main binary. Manages node lifecycle, configuration, gRPC/HTTP servers, CLI, and diagnostics.

## Boot Sequence

```
main()
  -> init_json_logging()
  -> Options::try_parse() (clap CLI)
  -> IF "run" subcommand:
       configuration::builder::build()  (HOCON + CLI merge)
       check_host(), check_ports(), load_private_key_from_file()
       initialize_diagnostics()  (Prometheus, InfluxDB, Zipkin, Sigar)
       node_runtime::start()
         -> NodeIdentifier from TLS certificate
         -> setup_node_program()
              -> Initialize LMDB stores (block, DAG, casper buffer, deploy, eval, play, replay, reporting)
              -> Create RuntimeManager (play/replay) with history
              -> Create Estimator, ValidatorIdentity
              -> Create block processor queue (mpsc::unbounded_channel)
              -> Create proposer queue (oneshot channels)
              -> Create API services and server instances
         -> Spawn concurrent tasks via JoinSet:
              CasperLoop, UpdateForkChoiceLoop, EngineInit, CasperLaunch,
              BlockProcessorInstance, ProposerInstance, HeartbeatProposer, ServersInstances
         -> Monitor tasks, graceful shutdown on SIGTERM/SIGINT
  -> ELSE (CLI subcommand):
       run_cli() -> route to deploy/propose/repl/keygen/status/etc.
```

## Configuration

**Config precedence** (highest wins):
1. CLI arguments (`--native-token-name=X`, `--network-id=Y`, etc.)
2. Config file (`rnode.conf` in data directory)
3. Default config (`defaults.conf` baked into the binary)

**Config build pipeline** (`configuration/mod.rs::build()`):
1. Load `defaults.conf` via HOCON → `HoconLoader::load_file()`
2. Merge user config on top (if `rnode.conf` exists in data dir) → `default_config.load_file(config_file)`
3. Resolve HOCON substitutions (e.g. `protocol-client.network-id = ${protocol-server.network-id}`)
4. Deserialize merged HOCON into `NodeConf` struct → `merged_config.resolve()`
5. Apply CLI overrides → `node_conf.override_config_values(options)` (via `config_mapper.rs`)
6. Validate → `validate_config(&node_conf)` (e.g. native token non-empty, decimals ≤ 18, quorum ≤ keys)

**Important**: HOCON substitutions resolve in step 3, before CLI overrides in step 5. A CLI flag like `--network-id` must override both `protocol_server.network_id` AND `protocol_client.network_id` because the substitution `${protocol-server.network-id}` already resolved to the HOCON default.

**`NodeConf`** fields:
- `protocol_server` -- P2P (port 40400, network-id, TLS)
- `protocol_client` -- Bootstrap peer, network-id (must match server), timeouts
- `peers_discovery` -- Kademlia (port 40404)
- `api_server` -- gRPC external (40401), internal (40402), HTTP (40403), admin (40405)
- `storage` -- Data directory (default `~/.rnode`)
- `casper` -- Validator key, parents, finalization, heartbeat, genesis block data (bonds, wallets, native token metadata)
- `metrics` -- Prometheus, InfluxDB, Zipkin, Sigar toggles
- `dev` -- Dev mode, deployer private key
- `openai` -- LLM integration settings

### CLI Flag Overrides

The following boolean flags override HOCON configuration at startup. CLI flags always take precedence.

| Flag | HOCON Target | Description |
|------|-------------|-------------|
| `--ceremony-master-mode` | `casper.genesis_ceremony.ceremony_master_mode = true` | Enable ceremony master mode (creates genesis block if none found) |
| `--enable-mergeable-channel-gc` | `casper.enable_mergeable_channel_gc = true` | Enable mergeable channel garbage collection |
| `--disable-mergeable-channel-gc` | `casper.enable_mergeable_channel_gc = false` | Disable mergeable channel GC (takes precedence over `--enable-mergeable-channel-gc`) |
| `--heartbeat-enabled` | `casper.heartbeat_conf.enabled = true` | Enable heartbeat block proposing for liveness |
| `--heartbeat-disabled` | `casper.heartbeat_conf.enabled = false` | Disable heartbeat proposing (takes precedence over `--heartbeat-enabled`) |
| `--native-token-name` | `casper.genesis_block_data.native_token_name` | Native token display name (genesis-locked) |
| `--native-token-symbol` | `casper.genesis_block_data.native_token_symbol` | Native token ticker symbol (genesis-locked) |
| `--native-token-decimals` | `casper.genesis_block_data.native_token_decimals` | Native token decimal places, 0-18 (genesis-locked) |

**Precedence rules for paired flags**: When both an enable and disable flag are provided for the same setting, the disable flag wins. The config mapper evaluates `--disable-*` after `--enable-*`, so the disable always takes final effect.

### Config Mapping

CLI flags are applied to the parsed `NodeConf` by `config_mapper.rs`:

- `--ceremony-master-mode` unconditionally sets `casper.genesis_ceremony.ceremony_master_mode = true`.
- `--disable-mergeable-channel-gc` / `--enable-mergeable-channel-gc` override `casper.enable_mergeable_channel_gc`. The disable flag is checked first; only if it is absent does the enable flag apply.
- `--heartbeat-disabled` / `--heartbeat-enabled` follow the same pattern for `casper.heartbeat_conf.enabled`.

## gRPC Services

| Service | Port | Methods |
|---------|------|---------|
| **DeployService** | 40401 (external) | `do_deploy`, `show_main_chain`, `get_blocks`, `get_block`, `find_deploy`, `exploratory_deploy`, `last_finalized_block`, `is_finalized`, `bond_status`, `get_data_at_name`, `listen_for_data_at_name` (deprecated), `listen_for_continuation_at_name`, `status`, `machine_verifiable_dag`, `visualize_dag`, `get_blocks_by_heights`, `get_event_by_hash` |
| **ProposeService** | 40402 (internal) | `propose`, `propose_result` |
| **ReplService** | 40402 (internal) | `run` (single command), `eval` (full program) |
| **LspService** | 40402 (internal) | `validate` (Rholang syntax diagnostics) |
| **Transport (P2P)** | 40400 | `packet_handler`, `streamed_blob_handler` |
| **Kademlia RPC** | 40404 | `ping`, `lookup` |

## HTTP REST API

| Port | Purpose |
|------|---------|
| 40403 | Public REST (deploy, blocks, finalization, transactions, status) via Axum |
| 40405 | Admin (propose, propose_result) |

**`/api/status`** returns node identity, network membership, and native token metadata:

```json
{
  "version": {"api": "1", "node": "..."},
  "address": "rnode://...",
  "networkId": "testnet",
  "shardId": "root",
  "peers": 4,
  "nodes": 4,
  "minPhloPrice": 1,
  "nativeTokenName": "F1R3CAP",
  "nativeTokenSymbol": "F1R3",
  "nativeTokenDecimals": 8,
  "peerList": [...]
}
```

## Rholang Type System (RhoExpr)

API responses from `explore-deploy`, `data-at-name`, and related endpoints return Rholang values as `RhoExpr` — a JSON-serializable representation of all Rholang types.

### Supported types

| Category | RhoExpr variant | JSON example |
|----------|----------------|-------------|
| **Primitives** | | |
| Boolean | `ExprBool` | `{"ExprBool": {"data": true}}` |
| Integer | `ExprInt` | `{"ExprInt": {"data": 42}}` |
| String | `ExprString` | `{"ExprString": {"data": "hello"}}` |
| URI | `ExprUri` | `{"ExprUri": {"data": "rho:io:stdout"}}` |
| Bytes | `ExprBytes` | `{"ExprBytes": {"data": "0a1b2c"}}` |
| **Extended numerics** | | |
| Float (f64) | `ExprFloat` | `{"ExprFloat": {"data": 3.14}}` |
| BigInt | `ExprBigInt` | `{"ExprBigInt": {"data": "12345678901234567890"}}` |
| BigRational | `ExprBigRat` | `{"ExprBigRat": {"numerator": "1", "denominator": "3"}}` |
| FixedPoint | `ExprFixedPoint` | `{"ExprFixedPoint": {"value": "31415", "scale": 4}}` |
| **Collections** | | |
| Tuple | `ExprTuple` | `{"ExprTuple": {"data": [...]}}` |
| List | `ExprList` | `{"ExprList": {"data": [...]}}` |
| Set | `ExprSet` | `{"ExprSet": {"data": [...]}}` |
| Map | `ExprMap` | `{"ExprMap": {"data": {"key": ...}}}` |
| Par (parallel) | `ExprPar` | `{"ExprPar": {"data": [...]}}` |
| **Unforgeable names** | | |
| Private | `ExprUnforg` | `{"ExprUnforg": {"data": {"UnforgPrivate": {"data": "hex..."}}}}` |
| Deploy ID | `ExprUnforg` | `{"ExprUnforg": {"data": {"UnforgDeploy": {"data": "hex..."}}}}` |
| Deployer ID | `ExprUnforg` | `{"ExprUnforg": {"data": {"UnforgDeployer": {"data": "hex..."}}}}` |
| System auth | `ExprUnforg` | `{"ExprUnforg": {"data": "UnforgSysAuthToken"}}` |
| **Bundle** | `ExprBundle` | `{"ExprBundle": {"data": ..., "read": true, "write": false}}` |
| **Operators** | | |
| Arithmetic | `ExprPlus`, `ExprMinus`, `ExprMult`, `ExprDiv`, `ExprMod` | `{"ExprPlus": {"left": ..., "right": ...}}` |
| Comparison | `ExprLt`, `ExprLte`, `ExprGt`, `ExprGte`, `ExprEq`, `ExprNeq` | `{"ExprEq": {"left": ..., "right": ...}}` |
| Logical | `ExprNot`, `ExprNeg`, `ExprAnd`, `ExprOr` | `{"ExprAnd": {"left": ..., "right": ...}}` |
| String | `ExprConcat`, `ExprInterpolate`, `ExprDiff` | `{"ExprConcat": {"left": ..., "right": ...}}` |
| **Other** | | |
| Pattern match | `ExprMatches` | `{"ExprMatches": {"target": ..., "pattern": ...}}` |
| Method call | `ExprMethod` | `{"ExprMethod": {"target": ..., "name": "method", "args": [...]}}` |
| Variable | `ExprVar` | `{"ExprVar": {"index": 0}}` |
| Process | `ExprUnknown` | `{"ExprUnknown": {"type_name": "Process"}}` |
| Unknown | `ExprUnknown` | `{"ExprUnknown": {"type_name": "..."}}` |

### Design

- **No silent drops**: every Rholang type has a representation. Unknown future types render as `ExprUnknown` with a type name — never silently disappear from responses.
- **Map keys**: any RhoExpr can be a map key. Primitives use natural string representation; complex types are serialized to JSON strings.
- **Extended numerics**: `BigInt`, `BigRat`, and `FixedPoint` are represented as decimal strings (not binary) for client readability. `Float` is IEEE 754 f64.
- **Process-level constructs** (sends, receives, new bindings) are represented as `ExprUnknown { type_name: "Process" }` rather than full AST serialization. These are rarely returned by data queries.
- **Deploy not found**: returns HTTP 404 (not 400) so clients can distinguish "not yet in block" from "invalid request."

### Key files

- `api/web_api.rs` — `RhoExpr` enum, `expr_from_par_proto()`, `expr_from_expr_proto()`, `unforg_from_proto()`, `extract_key_from_expr()`

**See also:** [Exploratory Deploy](exploratory-deploy.md)

## WebSocket Events

The `/ws/events` endpoint on the HTTP port (40403) streams real-time node events. See [websocket-events.md](websocket-events.md) for full documentation.

9 event types are streamed: 3 block lifecycle (`block-created`, `block-added`, `block-finalised`), 4 genesis ceremony (`sent-unapproved-block`, `block-approval-received`, `sent-approved-block`, `approved-block-received`), and 2 node lifecycle (`entered-running-state`, `node-started`).

Events published during startup are buffered and replayed to clients that connect after the node is running. The buffer is sealed when engine initialization completes.

## Error Handling & Shutdown

`handle_unrecoverable_errors()` in `node_runtime.rs` is the top-level error boundary. Any `Err` from `NodeRuntime::main()` is caught, logged via `tracing::error!`, and the process exits with code 1. This covers:
- Config validation failures (empty token name, invalid decimals)
- Genesis ceremony failures (required signatures not met)
- Token metadata verification mismatch (joiner config disagrees with on-chain state)
- Any runtime panic or unrecoverable error

The error chain propagates cleanly: `verify_token_metadata_matches_config → Err(CasperError) → ? in casper_launch.launch() → ? in NodeRuntime::main() → handle_unrecoverable_errors → process::exit(1)`. Destructors fire in order; no mid-async process::exit calls.

## API Server Startup

`bind_tcp_listener_with_retry()` in `servers_instances.rs` handles `AddrInUse` resilience for HTTP/Admin servers: 60 attempts with 500ms delay between retries.

`APIServers::build()` in `api_servers.rs` constructs all gRPC services (Repl, Propose, Deploy, LSP) with shared dependencies (engine cell, block store, connections). `WebApiImpl` in `web_api.rs` handles the HTTP REST layer and caches config-derived values (network-id, shard-id, min-phlo-price, native token metadata) for fast `/api/status` responses without per-request config reads.

## Transfer Extraction

Transfer data (from/to/amount/success) is extracted from block execution reports and inlined on `DeployInfo` for `get_block` and `last_finalized_block` responses.

### Architecture

Transfers are extracted from `BlockReportAPI`, which replays blocks using `ReportingRspace` to capture full COMM event data. Results are cached in `ReportStore` — each block is replayed once, then served from cache forever.

```
API handler (get_block / last_finalized_block)
  → BlockReportAPI.block_report(hash, false)
    → ReportStore check (cached? → return immediately)
    → ReportingCasper.trace(block) → full replay → cache in ReportStore
  → extract_transfers_from_report(&report, &transfer_unforgeable)
    → scan COMM events on transfer_unforgeable channel
    → parse from/to/amount/success from produce data
  → populate DeployInfo.transfers / DeployInfoSerde.transfers
```

### Behavior by node type

| Node type | HTTP `transfers` field | gRPC `transfers` / `transfersAvailable` |
|-----------|----------------------|----------------------------------------|
| **Readonly** | `"transfers": [...]` (populated) or `"transfers": []` (no transfers) | `transfers: [...]`, `transfersAvailable: true` |
| **Validator** | Field **omitted** (block replay unavailable) | `transfers: []`, `transfersAvailable: false` |

- HTTP uses `Option<Vec<TransferInfoSerde>>` with `skip_serializing_if = "Option::is_none"` — field absent when `None`
- gRPC uses `repeated TransferInfo` (always present, may be empty) + `bool transfersAvailable` to distinguish

### Key files

- `web/block_info_enricher.rs` — `extract_transfers_from_report()` standalone function, `find_transfers_in_report()` per-deploy scanner
- `web/transaction.rs` — `transfer_unforgeable()` (computes transfer channel Par from SystemVault.rho), `helpers` module for parsing produce event data
- `api/web_api.rs` — `WebApiImpl.enrich_transfers()` for HTTP path
- `api/deploy_grpc_service_v1.rs` — `DeployGrpcServiceV1Impl.enrich_proto_transfers()` for gRPC path
- `runtime/setup.rs` — wires `BlockReportAPI` + `transfer_unforgeable` into API services, proactive cache on finalization events

### Proactive caching

On finalization, a background task calls `block_report_api.block_report(hash, false)` to pre-warm `ReportStore`. On validators this is a no-op (block report rejected). On readonly nodes, the first API query for a block hits the pre-warmed cache.

## Find Deploy Retry

Both gRPC and REST APIs retry `find_deploy` on `DeployNotFoundError`:

| API | Retry Interval | Max Attempts |
|-----|----------------|--------------|
| gRPC | 100ms | 80 |
| REST | 50ms | 1 |

These values are hardcoded (previously configurable via `F1R3_*` env vars, removed in v0.4.10).

## Runtime Instances

**`BlockProcessorInstance`** -- Receives blocks, validates, applies to DAG. Semaphore-bounded parallelism. Re-queues on `FinalizationInProgress`.

**`ProposerInstance`** -- Dequeues proposal requests. Non-blocking locking (try_lock). 5-minute timeout for stuck proposals. Min-interval between proposals is 250ms (hardcoded).

**`HeartbeatProposer`** -- Periodic proposals for network liveness. Operator-tunable heartbeat settings (enabled, check-interval, max-lfb-age, self-propose-cooldown) are in `defaults.conf` under the `casper.heartbeat` section. The following behavioral parameters are hardcoded:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Frontier chase max lag | 0 | Max lag permitting frontier-chase proposals |
| Pending deploy max lag | 20 | Lag threshold above which pending deploy proposals throttle |
| Deploy recovery max lag | 64 | Lag threshold for deploy recovery mode |
| Stale recovery min interval | 12000ms | Min interval for stale-LFB recovery proposals |
| Deploy finalization grace | 25000ms | Grace period bypassing min-interval during deploy finalization |

**Deploy grace window**: When a deploy is proposed or finalization-critical parents observed, a grace window opens (default 25s) that allows proposals which would normally be blocked by cooldown/interval constraints.

**Stale LFB leader-only recovery**: Deterministic leader selection allows one validator to propose when LFB is stale but regular recovery is throttled (`lag in [1, threshold)`).

## Diagnostics

`initialize_diagnostics()` sets up:
- Prometheus (`/metrics` HTTP endpoint)
- InfluxDB (HTTP batch and/or UDP reporters)
- Zipkin (OpenTelemetry distributed tracing)
- Sigar (CPU, memory, disk system metrics)

## CLI Subcommands

| Command | Purpose |
|---------|---------|
| `run` | Start node |
| `eval FILE` | Execute Rholang file |
| `repl` | Interactive REPL |
| `deploy PHLO_LIMIT PHLO_PRICE ...` | Deploy contract |
| `propose` | Trigger block proposal |
| `show-block HASH` | Display block |
| `show-blocks DEPTH` | Recent blocks |
| `visualize-dag DEPTH` | DAG structure |
| `keygen PATH` | Generate key pair |
| `last-finalized-block` | Latest finalized block |
| `is-finalized HASH` | Check finalization |
| `bond-status KEY` | Validator bond query |
| `data-at-name NAME` | RSpace data subscription |
| `cont-at-name NAMES` | RSpace continuation subscription |
| `status` | Node status |

## Tests

Integration tests in `tests/`: `rho_trie_traverser_test.rs`. Inline tests in `block_info_enricher.rs` (2 unit tests for transfer extraction logic).

**See also:** [node/ crate README](../../node/README.md) | [Docker Setup](../../docker/README.md)

[← Back to docs index](../README.md)
