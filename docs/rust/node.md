> Last updated: 2026-03-23

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
1. CLI arguments
2. Config file (`rnode.conf` in data directory)
3. Default config (`defaults.conf`)

**`NodeConf`** fields:
- `protocol_server` -- P2P (port 40400, network-id, TLS)
- `protocol_client` -- Bootstrap peer, timeouts
- `peers_discovery` -- Kademlia (port 40404)
- `api_server` -- gRPC external (40401), internal (40402), HTTP (40403), admin (40405)
- `storage` -- Data directory (default `~/.rnode`)
- `casper` -- Validator key, parents, finalization, heartbeat
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

**Precedence rules for paired flags**: When both an enable and disable flag are provided for the same setting, the disable flag wins. The config mapper evaluates `--disable-*` after `--enable-*`, so the disable always takes final effect.

### Config Mapping

CLI flags are applied to the parsed `NodeConf` by `config_mapper.rs`:

- `--ceremony-master-mode` unconditionally sets `casper.genesis_ceremony.ceremony_master_mode = true`.
- `--disable-mergeable-channel-gc` / `--enable-mergeable-channel-gc` override `casper.enable_mergeable_channel_gc`. The disable flag is checked first; only if it is absent does the enable flag apply.
- `--heartbeat-disabled` / `--heartbeat-enabled` follow the same pattern for `casper.heartbeat_conf.enabled`.

## gRPC Services

| Service | Port | Methods |
|---------|------|---------|
| **DeployService** | 40401 (external) | `do_deploy`, `show_main_chain`, `get_blocks`, `get_block`, `find_deploy`, `exploratory_deploy`, `last_finalized_block`, `is_finalized`, `bond_status`, `listen_for_data_at_name`, `listen_for_continuation_at_name`, `status`, `machine_verifiable_dag`, `visualize_dag`, `get_blocks_by_heights` |
| **ProposeService** | 40402 (internal) | `propose`, `propose_result` |
| **ReplService** | 40402 (internal) | `run` (single command), `eval` (full program) |
| **LspService** | 40402 (internal) | `validate` (Rholang syntax diagnostics) |
| **Transport (P2P)** | 40400 | `packet_handler`, `streamed_blob_handler` |
| **Kademlia RPC** | 40404 | `ping`, `lookup` |

## HTTP REST API

| Port | Purpose |
|------|---------|
| 40403 | Public REST (deploy, blocks, finalization, transactions) via Axum |
| 40405 | Admin (propose, propose_result) |

## API Server Startup

`bind_tcp_listener_with_retry()` in `servers_instances.rs` handles `AddrInUse` resilience for HTTP/Admin servers: 60 attempts with 500ms delay between retries.

## Transfer Enrichment Pipeline

Inline transfer data on `DeployInfo` for `get_block` and `last_finalized_block` responses:

1. **`BlockEnricher` trait** (`web/block_info_enricher.rs`) -- Async trait for enriching `BlockInfo` with transfer data. Single method: `enrich(&self, BlockInfo) -> BlockInfo`.

2. **`CacheTransactionEnricher`** -- Concrete implementation backed by `CacheTransactionAPI`. Extracts block hash, calls `get_transaction()`, maps `UserDeploy` transactions to `TransferInfo`, populates `DeployInfo.transfers`.

3. **`CacheTransactionAPI`** (`web/transaction.rs`) -- Two-level caching: persistent LMDB store + `DashMap`-based in-flight request deduplication via `Shared<BoxFuture>`. Cache miss triggers extraction from `BlockReportAPI`, result stored persistently.

4. **Proactive caching** (`setup.rs`) -- Subscribes to `BlockFinalised` events. Background task spawns extraction with `Semaphore`-bounded concurrency (limit 8). Small race window where client may call `get_block` before cache is populated; `CacheTransactionAPI` handles this by computing on demand.

5. **REST integration** -- `WebApiImpl` holds `Arc<dyn BlockEnricher>`. `get_block()` and `last_finalized_block()` call `block_enricher.enrich()` before serialization.

6. **gRPC integration** -- `DeployGrpcServiceV1Impl` holds `Arc<dyn BlockEnricher>`. Same enrichment on `get_block` and `last_finalized_block` responses.

## Find Deploy Retry

Both gRPC and REST APIs retry `find_deploy` on `DeployNotFoundError`:

| API | Env: Interval | Default | Env: Max Attempts | Default |
|-----|---------------|---------|-------------------|---------|
| gRPC | `F1R3_GRPC_FIND_DEPLOY_RETRY_INTERVAL_MS` | 100ms | `F1R3_GRPC_FIND_DEPLOY_MAX_ATTEMPTS` | 80 |
| REST | `F1R3_FIND_DEPLOY_RETRY_INTERVAL_MS` | 50ms | `F1R3_FIND_DEPLOY_MAX_ATTEMPTS` | 1 |

## Runtime Instances

**`BlockProcessorInstance`** -- Receives blocks, validates, applies to DAG. Semaphore-bounded parallelism. Re-queues on `FinalizationInProgress`.

**`ProposerInstance`** -- Dequeues proposal requests. Non-blocking locking (try_lock). 5-minute timeout for stuck proposals. Min-interval between proposals configurable via `F1R3_PROPOSER_MIN_INTERVAL_MS` (default 250ms).

**`HeartbeatProposer`** -- Periodic proposals for network liveness with env-configurable behavior:

| Env Var | Default | Purpose |
|---------|---------|---------|
| `F1R3_HEARTBEAT_FRONTIER_CHASE_MAX_LAG` | 0 | Max lag permitting frontier-chase proposals |
| `F1R3_HEARTBEAT_PENDING_DEPLOY_MAX_LAG` | 20 | Lag threshold above which pending deploy proposals throttle |
| `F1R3_HEARTBEAT_DEPLOY_RECOVERY_MAX_LAG` | 64 | Lag threshold for deploy recovery mode |
| `F1R3_HEARTBEAT_SELF_PROPOSE_COOLDOWN_MS` | 0 | Cooldown between validator's own proposals |
| `F1R3_HEARTBEAT_STALE_RECOVERY_MIN_INTERVAL_MS` | 12000 | Min interval for stale-LFB recovery proposals |
| `F1R3_HEARTBEAT_DEPLOY_FINALIZATION_GRACE_MS` | 25000 | Grace period bypassing min-interval during deploy finalization |

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

Integration tests in `tests/`: `transaction_api_test.rs` (end-to-end transaction API), `rho_trie_traverser_test.rs`. Inline tests in `block_info_enricher.rs` (3 unit tests for enrichment logic).

[← Back to overview](./README.md)
