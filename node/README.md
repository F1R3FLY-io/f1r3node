# Node

Main entry point for the F1r3fly blockchain node. Orchestrates all subsystems: consensus (Casper), networking (comm), contract execution (Rholang/RSpace), storage, gRPC/HTTP APIs, and diagnostics.

## Building

```bash
cargo build --release -p node
cargo build --profile dev -p node   # debug mode
```

## Running

### Docker (Recommended)

See [docker/README.md](../docker/README.md) for complete Docker setup.

```bash
# Standalone (single node, instant finalization)
docker compose -f docker/standalone.yml up -d

# Multi-validator shard (bootstrap + 3 validators + observer)
docker compose -f docker/shard.yml up -d
```

### Local Development

See [run-local/README.md](../run-local/README.md) for local development setup.

```bash
# Standalone mode
cargo run --release -p node -- run -s \
  --host 127.0.0.1 \
  --validator-private-key <KEY> \
  --allow-private-addresses
```

## Ports

| Port | Service |
|------|---------|
| 40400 | Protocol (P2P) |
| 40401 | gRPC External |
| 40402 | gRPC Internal |
| 40403 | HTTP API |
| 40404 | Kademlia Discovery |
| 40405 | Admin HTTP API |

## Configuration

The node uses HOCON configuration with fallback semantics. Operator configs are minimal overrides on top of built-in defaults.

- [defaults.conf](src/main/resources/defaults.conf) — Built-in defaults with all available options
- [docker/conf/default.conf](../docker/conf/default.conf) — Shard override
- [docker/conf/standalone-dev.conf](../docker/conf/standalone-dev.conf) — Standalone override

## Testing

```bash
cargo test -p node
cargo test --release -p node
```

## Documentation

- [Node Module Overview](../docs/node/README.md) — Binary entry point, gRPC/HTTP servers, CLI, diagnostics
- [Docker Setup](../docker/README.md) — Docker compose for shard, standalone, and monitoring
