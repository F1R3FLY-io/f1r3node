# F1r3fly

> A decentralized, economic, censorship-resistant, public compute infrastructure and blockchain that hosts and executes smart contracts with trustworthy, scalable, concurrent proof-of-stake consensus.

## Table of Contents

- [What is F1r3fly?](#what-is-f1r3fly)
- [Security Notice](#note-on-the-use-of-this-software)
- [Installation](#installation)
  - [Docker (Recommended)](#docker-recommended)
  - [Source (Development)](#source-development)
- [Building](#building)
- [Running](#running)
  - [Docker Network](#docker-network-recommended)
  - [Local Development](#local-development-node)
- [Usage](#usage)
  - [F1r3fly Rust Client](#f1r3fly-rust-client)
  - [F1r3drive](#f1r3drive)
  - [Smoke Test](#smoke-test)
  - [Evaluating Rholang Contracts](#evaluating-rholang-contracts)
- [Configuration](#configuration-file)
- [Troubleshooting](#troubleshooting)
- [Support & Community](#support--community)
- [Known Issues & Reporting](#caveats-and-filing-issues)
- [License](#licence-information)

## What is F1r3fly?

F1r3fly is an open-source blockchain platform that provides:

- **Decentralized compute infrastructure** - Censorship-resistant public blockchain
- **Smart contract execution** - Hosts and executes programs (smart contracts)
- **Scalable consensus** - Proof-of-stake consensus with content delivery
- **Concurrent processing** - Trustworthy and scalable concurrent execution

### Getting Started

- **Production/Testing**: Use [Docker](#docker-recommended) to run nodes
- **Development**: Install via [Nix](#nix-recommended) or [manually](#manual-install)
- **Community**: Join the [F1r3fly Discord](https://discord.gg/NN59aFdAHM)
- **Testnet**: Public testnet access coming soon

### Node Implementations

This repository contains two node implementations on different branches:

| Implementation | Branch | Status | Description |
|---|---|---|---|
| **Rust Node** | [`rust/dev`](https://github.com/F1R3FLY-io/f1r3fly/tree/rust/dev) | **Production (recommended)** | Pure Rust, no JVM dependency |
| Scala Node | `main` (this branch) | Maintenance | Original Scala implementation |

> You are on the **`main` branch** (Scala node). For new deployments, the [`rust/dev` branch](https://github.com/F1R3FLY-io/f1r3fly/tree/rust/dev) is recommended.

## Note on the use of this software
This code has not yet completed a security review. We strongly recommend that you do not use it in production or to transfer items of material value. We take no responsibility for any loss you may incur through the use of this code.

## Installation

### Docker (Recommended)

```bash
# Pull the latest image
docker pull f1r3flyindustries/f1r3fly-scala-node:latest

# Start a standalone node (for development)
docker compose -f docker/standalone.yml up

# Or start a multi-validator network (for testing consensus)
docker compose -f docker/shard.yml up
```

| Port | Service | Description |
|------|---------|-------------|
| 40400 | Protocol Server | Main blockchain protocol |
| 40401 | gRPC External | External gRPC API |
| 40402 | gRPC Internal | Internal gRPC API |
| 40403 | HTTP API | REST/HTTP API endpoints |
| 40404 | Peer Discovery | Node discovery service |
| 40405 | Admin | Admin/metrics endpoint |

Data persists in `docker/data/`. Fresh start: `docker compose -f docker/shard.yml down && rm -rf docker/data/`

Docker Compose loads `docker/.env` automatically — it contains node credentials and optional tuning. See [docker/.env.example](docker/.env.example) for all available variables.

See [docker/README.md](docker/README.md) for shard setup, validator bonding, and network configuration.

### Source (Development)

#### Nix (Recommended)

Nix provides a reproducible environment with all dependencies pinned (JDK 17, SBT, Rust, BNFC, JFlex, protobuf).

1. Install [Nix](https://nixos.org/download/) and [direnv](https://direnv.net/#basic-installation)
2. Clone and enter the environment:

```bash
git clone <repository-url>
cd f1r3fly
direnv allow
```

If you encounter `error: experimental Nix feature 'nix-command' is disabled`:
```bash
mkdir -p ~/.config/nix
echo "experimental-features = flakes nix-command" > ~/.config/nix/nix.conf
direnv allow
```

#### Manual Install

Install the following:

- **Java 17** (OpenJDK/Temurin)
- **SBT** (Scala Build Tool)
- **Rust** (stable toolchain): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **System packages:**

Debian/Ubuntu:
```bash
sudo apt install autoconf cmake curl git jflex libtool make protobuf-compiler sbt unzip
```

macOS (Homebrew):
```bash
brew install autoconf cmake git libtool make protobuf sbt
```

**BNFC** (parser generator, from Haskell):
```bash
# Via cabal
cabal install alex happy BNFC
```

## Building

### Quick Commands

```bash
# Fat JAR for local development
sbt ";compile ;project node ;assembly ;project rchain"

# Docker image
docker context use default && sbt ";compile ;project node ;Docker/publishLocal ;project rchain"

# Docker image (multiplatform: amd64 + arm64)
docker context use default && MULTI_ARCH=true sbt ";compile ;project node ;Docker/publishLocal ;project rchain"

# Clean build
sbt "clean"
```

### SBT Tips

- Keep SBT running: use `sbt` shell for faster subsequent commands
- Project-specific builds: `sbt "project node" compile`

## Running

### Docker Network (Recommended)

**Standalone** (single-validator, for development):
```bash
docker compose -f docker/standalone.yml up -d
docker compose -f docker/standalone.yml logs -f
docker compose -f docker/standalone.yml down
```

**Shard** (3 validators + bootstrap + observer):
```bash
docker compose -f docker/shard.yml up -d

# Wait for genesis (~2-3 min)
docker compose -f docker/shard.yml logs -f --tail=500 | grep "Making a transition to Running state"
# Ctrl+C once all validators report Running

docker compose -f docker/shard.yml logs -f         # Follow logs
docker compose -f docker/shard.yml down             # Stop
docker compose -f docker/shard.yml down -v          # Stop and wipe data
```

**Observer** (read-only, requires running shard):
```bash
docker compose -f docker/observer.yml up
```

**Monitoring** (Prometheus + Grafana + cAdvisor, requires running shard):
```bash
docker compose -f docker/shard-monitoring.yml up -d
```

| Component | URL | Description |
|---|---|---|
| Prometheus | http://localhost:9090 | Metrics, targets, recording rules |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |
| cAdvisor | http://localhost:8080 | Container CPU/memory/IO metrics |

See [docker/README.md](docker/README.md) for details.

### Local Development Node

Local runs do **not** load `docker/.env` — configuration comes from HOCON config at `~/.rnode/rnode.conf`. Environment variables for AI services must be set in your shell.

After [building from source](#building):

```bash
java -Djna.library.path=./rust_libraries/release \
  --add-opens java.base/sun.security.util=ALL-UNNAMED \
  --add-opens java.base/java.nio=ALL-UNNAMED \
  --add-opens java.base/sun.nio.ch=ALL-UNNAMED \
  -jar node/target/scala-2.12/rnode-assembly-1.0.0-SNAPSHOT.jar \
  run -s --no-upnp --allow-private-addresses --synchrony-constraint-threshold=0.0
```

Fresh start: `rm -rf ~/.rnode/`

To enable AI services locally, set env vars before running:
```bash
OPENAI_ENABLED=true OPENAI_SCALA_CLIENT_API_KEY=sk-... java -Djna.library.path=...
```

## Usage

### F1r3fly Rust Client

CLI for interacting with F1r3fly nodes: deploy, propose, transfer, bond validators, check health.

Repository: [F1R3FLY-io/rust-client](https://github.com/F1R3FLY-io/rust-client)

```bash
git clone https://github.com/F1R3FLY-io/rust-client.git
cd rust-client
cargo build --release

# Deploy a Rholang contract
cargo run -- deploy -f ./rho_examples/stdout.rho

# Check network status
cargo run -- status
```

### F1r3drive

FUSE-based file system backed by the F1r3fly blockchain.

Repository: [F1R3FLY-io/f1r3drive](https://github.com/F1R3FLY-io/f1r3drive)

### Smoke Test

Verify the shard end-to-end using the [rust-client](https://github.com/F1R3FLY-io/rust-client) smoke test:

```bash
cd rust-client
./scripts/smoke_test.sh localhost 40412 40413 40452
```

### Evaluating Rholang Contracts

```bash
sbt ";compile ;stage"
./node/target/universal/stage/bin/rnode \
  eval ./rholang/examples/tut-ai.rho
```

Explore `rholang/examples/` for sample contracts.

### System Packages

<details>
<summary>Debian/Ubuntu</summary>

Requires `java17-runtime-headless`:

```bash
sudo apt update
sudo apt install ./rnode_X.Y.Z_all.deb
rnode run -s
```

Paths: `/usr/bin/rnode`, `/usr/share/rnode/rnode.jar`

</details>

<details>
<summary>RedHat/Fedora</summary>

Requires `java-17-openjdk`:

```bash
sudo dnf install ./rnode-X.Y.Z-1.noarch.rpm
rnode run -s
```

Paths: `/usr/bin/rnode`, `/usr/share/rnode/rnode.jar`

</details>

## Multi-Service Orchestration

For running F1r3fly alongside other ecosystem services (Embers, F1R3Sky, monitoring), see the [system-integration](https://github.com/F1R3FLY-io/system-integration) repository. It provides `shardctl` CLI, shared configs, integration tests, and Docker Compose orchestration.

## AI Services

F1r3fly nodes expose AI capabilities as Rholang system processes. These are available to smart contracts at runtime.

| Rholang Process | Provider | Description |
|---|---|---|
| `rho:ai:gpt4` | OpenAI | GPT-4 text completion |
| `rho:ai:dalle3` | OpenAI | DALL-E 3 image generation |
| `rho:ai:textToAudio` | OpenAI | Text-to-speech audio |
| `rho:ai:ollama:chat` | Ollama (local) | Chat completion via local Ollama |
| `rho:ai:ollama:generate` | Ollama (local) | Text generation via local Ollama |
| `rho:ai:ollama:models` | Ollama (local) | List available local models |

AI services are disabled by default. Enable via environment variables (Docker) or HOCON config (local). See [docker/.env.example](docker/.env.example) for all available env vars and [defaults.conf](node/src/main/resources/defaults.conf) for HOCON config reference.

When AI is disabled, contracts using `rho:ai:*` processes will fail at deploy time.

## Configuration File

- **Default location**: Data directory (`~/.rnode/`)
- **Custom location**: `--config-file <path>`
- **Format**: [HOCON](https://github.com/lightbend/config/blob/master/HOCON.md)

Reference configs:
- [defaults.conf](node/src/main/resources/defaults.conf) - All available options
- [docker/conf/standalone-dev.conf](docker/conf/standalone-dev.conf) - Standalone development config
- [docker/conf/default.conf](docker/conf/default.conf) - Shard config

## Troubleshooting

### SBT Build Issues

```bash
rm -rf ~/.cache/coursier/
sbt clean
```

### Node Compilation (StackOverflow)

```bash
sbt
sbt:rchain> project node
sbt:node> compile
```

### Docker Issues

```bash
docker context use default
docker system prune -a
```

### Port Conflicts

```bash
lsof -i :40400-40404
kill -9 <PID>
```

## Support & Community

- [F1r3fly Discord](https://discord.gg/NN59aFdAHM) - Tutorials, discussions, support
- [GitHub Issues](https://github.com/F1R3FLY-io/f1r3fly/issues) - Bug reports and feature requests

## Caveats and Filing Issues

This F1r3fly repository is under active development. Report issues at [GitHub Issues](https://github.com/F1R3FLY-io/f1r3fly/issues/new/choose).

Include: F1r3fly version, OS, steps to reproduce, expected vs actual behavior, logs.

## Acknowledgements

We use [YourKit](https://www.yourkit.com/) to profile F1r3fly performance.

## License Information

F1r3fly is licensed under the **Apache License 2.0**.
