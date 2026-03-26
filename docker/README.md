# F1R3FLY Docker Network

## Quick Start (Docker Only)

Pull the latest image and start a multi-validator shard:

```bash
docker compose -f shard.yml pull
docker compose -f shard.yml up -d
```

Wait for genesis (~2-3 minutes). All validators must transition to Running state:
```bash
docker compose -f shard.yml logs 2>&1 | grep "Making a transition to Running state"
```

Once all validators report Running, press `Ctrl+C`. The network is ready.

**Follow logs:**
```bash
# All nodes
docker compose -f shard.yml logs -f

# Specific node
docker compose -f shard.yml logs -f validator1
docker compose -f shard.yml logs -f boot
docker compose -f shard.yml logs -f readonly
```

**Stop:**
```bash
docker compose -f shard.yml down
```

**Stop and wipe all data (fresh restart):**
```bash
docker compose -f shard.yml down -v
```

## Build from Source

Requires Nix or a Rust toolchain. Build a local Docker image:
```bash
./node/docker-commands.sh build-local
```

Then start with the local image:
```bash
F1R3FLY_RUST_IMAGE=f1r3fly-rust-node:local docker compose -f shard.yml up -d
```

## Standalone Node (Single Validator)

For local development with instant finalization:
```bash
docker compose -f standalone.yml up -d
docker compose -f standalone.yml logs -f
docker compose -f standalone.yml down -v
```

## Compose Files

| File | Description |
|------|-------------|
| `shard.yml` | Full shard: bootstrap + 3 validators + observer |
| `standalone.yml` | Single standalone node for development |
| `validator4.yml` | Additional validator joining existing shard |
| `observer.yml` | Additional read-only node joining existing shard |
| `shard-monitoring.yml` | Prometheus + Grafana + cAdvisor overlay |

## Configuration

All compose files use 2 shared config files. Per-role behavior is controlled via CLI flags in compose commands.

| Config File | Used By | Purpose |
|-------------|---------|---------|
| `conf/default.conf` | All shard roles | Shared shard defaults |
| `conf/standalone-dev.conf` | Standalone | `standalone = true`, instant finalization |

CLI flags used per role:

| Flag | Used by |
|------|---------|
| `--ceremony-master-mode` | Bootstrap |
| `--heartbeat-disabled` | Bootstrap, Observer |
| `--required-signatures N` | Bootstrap |
| `--genesis-validator` | Validators 1-3 |

Key settings in `default.conf`:
- `fault-tolerance-threshold = 0.99` (near-unanimous finalization)
- `synchrony-constraint-threshold = 0` (no synchrony gate on proposals)
- `enable-mergeable-channel-gc = true`
- `heartbeat.enabled = true` (overridden via `--heartbeat-disabled` for bootstrap/observer)

## Port Mapping

| Node | Protocol | gRPC Ext | gRPC Int | HTTP | Discovery | Admin |
|------|----------|----------|----------|------|-----------|-------|
| Bootstrap | 40400 | 40401 | 40402 | 40403 | 40404 | 40405 |
| Validator1 | 40410 | 40411 | 40412 | 40413 | 40414 | 40415 |
| Validator2 | 40420 | 40421 | 40422 | 40423 | 40424 | 40425 |
| Validator3 | 40430 | 40431 | 40432 | 40433 | 40434 | 40435 |
| Validator4 | 40440 | 40441 | 40442 | 40443 | 40444 | 40445 |
| Observer | 40450 | 40451 | 40452 | 40453 | 40454 | 40455 |

## Monitoring

Start the monitoring stack after the shard is running:

```bash
docker compose -f shard-monitoring.yml up -d    # Start
docker compose -f shard-monitoring.yml down      # Stop
```

| Component | URL | Description |
|---|---|---|
| Prometheus | http://localhost:9090 | Metrics, targets, recording rules |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |
| cAdvisor | http://localhost:8080 | Container CPU/memory/IO metrics |

Prometheus uses DNS-based service discovery — only running nodes get scraped (no false DOWN targets for standalone or partial shard).

## CI/Smoke Testing

Automated startup SLA check:
```bash
./scripts/ci/check-casper-init-sla.sh docker/shard.yml 180
```

Debug bundle on failure:
```bash
./scripts/ci/collect-casper-init-artifacts.sh docker/shard.yml /tmp/casper-init-artifacts
```

## Smoke Test

Verify the shard is working end-to-end using the [rust-client](https://github.com/F1R3FLY-io/rust-client) smoke test:

```bash
# Clone rust-client (if not already present)
git clone https://github.com/F1R3FLY-io/rust-client.git ../rust-client

# Run smoke test against the running shard (default: localhost:40411)
cd ../rust-client
./scripts/smoke_test.sh localhost 40412 40413 40452
```

The smoke test builds the rust-client binary and runs 30+ commands covering:
- Deploy / propose / finalize workflow
- Token transfers with finalization verification
- Node status, blocks, metrics
- PoS queries (epoch info, validator status)
- Block streaming (watch-blocks)
- Load testing with concurrent transfers

Results are logged to `logs/smoke_test_*.log` with pass/fail counters.

## Genesis Configuration

### Wallets (genesis/wallets.txt)
- **Bootstrap Node** - Initial REV balance for network operations
- **Validator 1-3** - Funded for transaction fees and operations

### Bonds (genesis/bonds.txt)
- **Validator 1** - Bonded with 1000 stake
- **Validator 2** - Bonded with 1000 stake
- **Validator 3** - Bonded with 1000 stake

Bootstrap and Validator 4 are **not** in bonds.txt and do not validate.

## Interact with Node

Rust client: https://github.com/F1R3FLY-io/rust-client

## Wallet Information

### Standalone / Bootstrap Node
- **Private Key**: `5f668a7ee96d944a4494cc947e4005e172d7ab3461ee5538f1f2a45a835e9657`
- **Public Key**: `04ffc016579a68050d655d55df4e09f04605164543e257c8e6df10361e6068a5336588e9b355ea859c5ab4285a5ef0efdf62bc28b80320ce99e26bb1607b3ad93d`
- **ETH**: `fac7dde9d0fa1df6355bd1382fe75ba0c50e8840`
- **REV**: `1111AtahZeefej4tvVR6ti9TJtv8yxLebT31SCEVDCKMNikBk5r3g`

### Validator 1
- **Private Key**: `357cdc4201a5650830e0bc5a03299a30038d9934ba4c7ab73ec164ad82471ff9`
- **Public Key**: `04fa70d7be5eb750e0915c0f6d19e7085d18bb1c22d030feb2a877ca2cd226d04438aa819359c56c720142fbc66e9da03a5ab960a3d8b75363a226b7c800f60420`
- **ETH**: `a77c116ce0ebe1331487638233bb52ba6b277da7`
- **REV**: `111127RX5ZgiAdRaQy4AWy57RdvAAckdELReEBxzvWYVvdnR32PiHA`

### Validator 2
- **Private Key**: `2c02138097d019d263c1d5383fcaddb1ba6416a0f4e64e3a617fe3af45b7851d`
- **Public Key**: `04837a4cff833e3157e3135d7b40b8e1f33c6e6b5a4342b9fc784230ca4c4f9d356f258debef56ad4984726d6ab3e7709e1632ef079b4bcd653db00b68b2df065f`
- **ETH**: `df00c6395a23e9b2b8780de9a93c9522512947c3`
- **REV**: `111129p33f7vaRrpLqK8Nr35Y2aacAjrR5pd6PCzqcdrMuPHzymczH`

### Validator 3
- **Private Key**: `b67533f1f99c0ecaedb7d829e430b1c0e605bda10f339f65d5567cb5bd77cbcb`
- **Public Key**: `0457febafcc25dd34ca5e5c025cd445f60e5ea6918931a54eb8c3a204f51760248090b0c757c2bdad7b8c4dca757e109f8ef64737d90712724c8216c94b4ae661c`
- **ETH**: `ca778c4ecf5c6eb285a86cedd4aaf5167f4eae13`
- **REV**: `1111LAd2PWaHsw84gxarNx99YVK2aZhCThhrPsWTV7cs1BPcvHftP`

### Validator 4
- **Private Key**: `5ff3514bf79a7d18e8dd974c699678ba63b7762ce8d78c532346e52f0ad219cd`
- **Public Key**: `04d26c6103d7269773b943d7a9c456f9eb227e0d8b1fe30bccee4fca963f4446e3385d99f6386317f2c1ad36b9e6b0d5f97bb0a0041f05781c60a5ebca124a251d`
- **ETH**: `0cab9328d6d896e5159a1f70bc377e261ded7414`
- **REV**: `1111La6tHaCtGjRiv4wkffbTAAjGyMsVhzSUNzQxH1jjZH9jtEi3M`
