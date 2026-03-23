# F1r3fly

> A decentralized, economic, censorship-resistant, public compute infrastructure and blockchain that hosts and executes smart contracts with trustworthy, scalable, concurrent proof-of-stake consensus.

## Table of Contents

- [What is F1r3fly?](#what-is-f1r3fly)
- [Security Notice](#note-on-the-use-of-this-software)
- [Installation](#installation)
  - [Docker (Recommended)](#docker-recommended)
  - [Source (Development)](#source-development)
- [Building](#building)
  - [Docker Image](#building-rust-node-docker-image)
  - [Cargo](#building-from-source)
- [Running](#running)
  - [Docker Network](#docker-network-recommended)
  - [Local Development](#local-development)
- [Usage](#usage)
  - [F1r3fly Rust Client](#f1r3fly-rust-client)
  - [Smoke Test](#smoke-test)
  - [F1r3flyFS](#f1r3flyfs)
- [Configuration](#configuration-file)
- [Troubleshooting](#troubleshooting)
- [Support & Community](#support--community)
- [Known Issues & Reporting](#caveats-and-filing-issues)
- [Legacy Scala/Hybrid Node](#legacy-scalahybrid-node)
- [License](#licence-information)

## What is F1r3fly?

F1r3fly is an open-source blockchain platform that provides:

- **Decentralized compute infrastructure** - Censorship-resistant public blockchain
- **Smart contract execution** - Hosts and executes programs (smart contracts)
- **Scalable consensus** - Proof-of-stake consensus with content delivery
- **Concurrent processing** - Trustworthy and scalable concurrent execution

### Getting Started

- **Production/Testing**: Use [Docker](#docker-recommended) to run nodes
- **Development**: Install build tools [manually](#manual-install) or via [Nix](#nix-alternative)
- **Community**: Join the [F1r3fly Discord](https://discord.gg/NN59aFdAHM)
- **Testnet**: Public testnet access coming soon

### Node Implementations

This repository contains two node implementations on different branches:

| Implementation | Branch | Status | Description |
|---|---|---|---|
| **Rust Node** | `rust/dev` (default) | **Production (recommended)** | Pure Rust, no JVM dependency |
| Scala Node | [`main`](https://github.com/F1R3FLY-io/f1r3fly/tree/main) | Maintenance | Original implementation, see [legacy section](#legacy-scalahybrid-node) |

> You are on the **`rust/dev` branch** (Rust node).

## Note on the use of this software
A security review of this code is underway. If you are looking for production ready deployment of this codebase, please contact F1r3fly at f1r3fly.ceo \<at\> gmail \<dot\> com. F1r3fly takes no responsibility for material or financial loss under the terms of the Apache 2.0 license.

## Installation

### Docker (Recommended)

```bash
# Pull the latest image
docker pull f1r3flyindustries/f1r3fly-rust-node:latest

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

Data persists in named Docker volumes. Fresh start: `docker compose -f docker/standalone.yml down -v`

Docker Compose loads `docker/.env` automatically — it contains node credentials and optional tuning. See [docker/.env.example](docker/.env.example) for all available variables.

See [docker/README.md](docker/README.md) for shard setup, validator bonding, and network configuration.

### Source (Development)

#### Manual Install

Install the following build dependencies:

**Rust toolchain:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
```

**System packages:**

Debian/Ubuntu:
```bash
sudo apt install autoconf cmake curl git libtool make protobuf-compiler unzip pkg-config libssl-dev
```

macOS (Homebrew):
```bash
brew install autoconf cmake git libtool make protobuf openssl pkg-config
```

**Optional tools:**
- [`just`](https://github.com/casey/just) - Task runner for build commands
- [`grpcurl`](https://github.com/fullstorydev/grpcurl) - gRPC CLI for testing

#### Nix (Alternative)

Nix provides a reproducible environment with all dependencies pinned. Install [Nix](https://nixos.org/download/) and [direnv](https://direnv.net/#basic-installation), then:

```bash
direnv allow
```

If you encounter `error: experimental Nix feature 'nix-command' is disabled`:
```bash
mkdir -p ~/.config/nix
echo "experimental-features = flakes nix-command" > ~/.config/nix/nix.conf
direnv allow
```

## Building

### Building Rust Node Docker Image

```bash
# Using helper script (recommended)
./node/docker-commands.sh build-local

# Direct Docker build
docker build -f node/Dockerfile -t f1r3flyindustries/f1r3fly-rust-node:latest .

# Multi-architecture (amd64 + arm64)
docker buildx build --platform linux/amd64,linux/arm64 \
  -f node/Dockerfile -t f1r3flyindustries/f1r3fly-rust-node:latest --push .
```

Image details: `rust:bookworm` builder, `debian:bookworm-slim` runtime, pure Rust binary. Tags: `f1r3fly-rust-node:local` (local build), `f1r3flyindustries/f1r3fly-rust-node:latest` (published).

### Building from Source

```bash
# Release build
cargo build --release -p node

# Debug build (faster compile)
cargo build -p node
```

Or using `just`:

```bash
just build              # Release build
just build-debug        # Debug build
```

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

**Build and run local image:**
```bash
./node/docker-commands.sh build-local
F1R3FLY_RUST_IMAGE=f1r3fly-rust-node:local docker compose -f docker/shard.yml up
```

### Local Development

Run the Rust node locally without Docker. Local runs do **not** load `docker/.env` — configuration comes from the HOCON config file (`run-local/conf/standalone.conf`). Environment variables for AI services must be set in your shell.

```bash
# Build and run standalone node
just run-standalone

# Or manually
just setup-standalone
just build
./target/release/node run -s --config-file run-local/conf/standalone.conf
```

| Command | Description |
|---------|-------------|
| `just build` | Build node in release mode |
| `just build-debug` | Build node in debug mode (faster compile) |
| `just run-standalone` | Build and run standalone node |
| `just run-standalone-debug` | Run in debug mode |
| `just setup-standalone` | Set up data directory only |
| `just clean-standalone` | Remove node data (fresh start) |
| `just help` | Show node CLI help |

Configuration: [`run-local/conf/standalone.conf`](run-local/conf/standalone.conf). See [`run-local/README.md`](run-local/README.md) for details.

To enable AI services locally, set env vars before running:
```bash
OPENAI_ENABLED=true OPENAI_API_KEY=sk-... just run-standalone
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

### Rholang CLI

Standalone CLI for executing and compiling Rholang programs (no running node required). See [rholang/README.md](rholang/README.md) for full documentation.

```bash
# Build
cargo build --release --bin rholang-cli

# Execute a Rholang program
./target/release/rholang-cli examples/hello.rho

# Compile to AST
./target/release/rholang-cli --format ast examples/hello.rho
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

### F1r3flyFS

Distributed file system built on F1r3fly: [F1r3flyFS Repository](https://github.com/F1R3FLY-io/f1r3flyfs#f1r3flyfs)

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

### Rust Build Issues

```bash
# Update toolchain
rustup update stable

# Clean and rebuild
cargo clean
cargo build --release -p node
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

### Nix Issues

```bash
nix-garbage-collect
direnv reload
```

## Support & Community

- [F1r3fly Discord](https://discord.gg/NN59aFdAHM) - Tutorials, discussions, support
- [GitHub Issues](https://github.com/F1R3FLY-io/f1r3fly/issues) - Bug reports and feature requests

## Caveats and Filing Issues

This F1r3fly repository is under active development. Report issues at [GitHub Issues](https://github.com/F1R3FLY-io/f1r3fly/issues/new).

Include: F1r3fly version, OS, steps to reproduce, expected vs actual behavior, logs.

## Legacy Scala/Hybrid Node

<details>
<summary>Scala node and hybrid node documentation (deprecated)</summary>

The Scala node (`main` branch) and hybrid Scala+Rust node are deprecated in favor of the pure Rust node. For new deployments, use the Rust node on `rust/dev`.

### Building Legacy Hybrid Node

Requires [Nix development environment](#nix-alternative):

```bash
# Build hybrid Docker image
docker context use default && sbt ";compile ;project node ;Docker/publishLocal ;project rchain"

# Build fat JAR
sbt ";compile ;project node ;assembly ;project rchain"
```

### Running Legacy Node

```bash
java -Djna.library.path=./rust_libraries/release \
  --add-opens java.base/sun.security.util=ALL-UNNAMED \
  --add-opens java.base/java.nio=ALL-UNNAMED \
  --add-opens java.base/sun.nio.ch=ALL-UNNAMED \
  -jar node/target/scala-2.12/rnode-assembly-1.0.0-SNAPSHOT.jar \
  run -s --no-upnp --allow-private-addresses --synchrony-constraint-threshold=0.0
```

### System Packages (Legacy)

<details>
<summary>Debian/Ubuntu</summary>

Requires `java17-runtime-headless`:

```bash
sudo apt update
sudo apt install ./rnode_X.Y.Z_all.deb
rnode run -s
```

</details>

<details>
<summary>RedHat/Fedora</summary>

Requires `java-17-openjdk`:

```bash
sudo dnf install ./rnode-X.Y.Z-1.noarch.rpm
rnode run -s
```

</details>

### SBT Tips

- Keep SBT running: use `sbt` shell for faster subsequent commands
- Project-specific builds: `sbt "project node" compile`
- Clean build: `sbt clean` then `rm -rf ~/.cache/coursier/`

### Evaluating Rholang Contracts (Legacy)

```bash
sbt ";compile ;stage"
./node/target/universal/stage/bin/rnode \
  -Djna.library.path=./rust_libraries/release \
  eval ./rholang/examples/tut-ai.rho
```

</details>

## Acknowledgements

We use [YourKit](https://www.yourkit.com/) to profile F1r3fly performance.

## License Information

F1r3fly is licensed under the **Apache License 2.0**.
