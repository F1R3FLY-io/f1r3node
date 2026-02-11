# F1r3fly

> A decentralized, economic, censorship-resistant, public compute infrastructure and blockchain that hosts and executes smart contracts with trustworthy, scalable, concurrent proof-of-stake consensus.

## Table of Contents

- [What is F1r3fly?](#what-is-f1r3fly)
- [Security Notice](#note-on-the-use-of-this-software)
- [Installation](#installation)
  - [Docker (Recommended)](#docker-recommended)
  - [Source (Development)](#source-development)
  - [macOS](#macos)
  - [System Packages (Legacy)](#system-packages-legacy---hybrid-node)
- [Building](#building)
  - [Building Rust Node Docker Image](#building-rust-node-docker-image-recommended)
- [Running](#running)
  - [Docker Network](#docker-network-recommended)
  - [Local Development (Pure Rust)](#local-development-pure-rust)
- [Usage](#usage)
  - [F1r3fly Rust Client](#f1r3fly-rust-client)
  - [Evaluating Rholang Contracts](#evaluating-rholang-contracts)
  - [F1r3flyFS](#f1r3flyfs)
- [Configuration](#configuration-file)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Support & Community](#support--community)
- [Known Issues & Reporting](#caveats-and-filing-issues)
- [Acknowledgements](#acknowledgements)
- [License](#licence-information)

## What is F1r3fly?

F1r3fly is an open-source blockchain platform that provides:

- **Decentralized compute infrastructure** - Censorship-resistant public blockchain
- **Smart contract execution** - Hosts and executes programs (smart contracts)
- **Scalable consensus** - Proof-of-stake consensus with content delivery
- **Concurrent processing** - Trustworthy and scalable concurrent execution

### Getting Started

- **Development**: Install locally using [Nix and direnv](#source) for a complete development environment
- **Production**: Use [Docker](#docker) or [system packages](#debianubuntu) for running nodes
- **Community**: Join the [F1r3fly Discord](https://discord.gg/NN59aFdAHM) for tutorials, documentation, and project information
- **Testnet**: Public testnet access coming soon

### Node Implementations

F1r3fly has two node implementations in this repository on different branches:

| Implementation | Branch                                                    | Status                       | Description                                                            |
| -------------- | --------------------------------------------------------- | ---------------------------- | ---------------------------------------------------------------------- |
| **Rust Node**  | `rust/dev` (default)                                      | **Production (recommended)** | Pure Rust implementation with better performance and no JVM dependency |
| Scala Node     | [`main`](https://github.com/F1R3FLY-io/f1r3fly/tree/main) | Stable                       | Original Scala implementation, being phased out in favor of Rust       |

> **Note**: You are on the **`rust/dev` branch** (Rust node). For the Scala implementation, switch to the [`main` branch](https://github.com/F1R3FLY-io/f1r3fly/tree/main).

## Note on the use of this software
A security review of this code is underway. If you are looking for production ready deployment of this codebase, please contact F1r3fly at  f1r3fly.ceo \<at\> gmail \<dot\> com. F1r3fly takes no responsibility for material or financial loss under the terms of the Apache 2.0 license.

## Installation

### Docker (Recommended)

**Pure Rust node - recommended for production and testing**

#### Quick Start

```bash
# Pull the latest image
docker pull f1r3flyindustries/f1r3fly-rust-node:latest

# Start a standalone node (simplest - for development)
docker compose -f docker/standalone.yml up

# Or start a multi-validator network (for testing consensus)
docker compose -f docker/shard-with-autopropose.yml up
```

#### Port Configuration

| Port  | Service         | Description              |
| ----- | --------------- | ------------------------ |
| 40400 | Protocol Server | Main blockchain protocol |
| 40401 | gRPC External   | External gRPC API        |
| 40402 | gRPC Internal   | Internal gRPC API        |
| 40403 | HTTP API        | REST/HTTP API endpoints  |
| 40404 | Peer Discovery  | Node discovery service   |

#### Data Persistence

The network automatically creates a `docker/data/` directory for blockchain state.

**Fresh Start** - Reset to genesis:
```bash
docker compose -f docker/standalone.yml down
rm -rf docker/data/
docker compose -f docker/standalone.yml up
```

#### Resources

- [Docker Hub - f1r3fly-rust-node](https://hub.docker.com/r/f1r3flyindustries/f1r3fly-rust-node)
- [Docker Setup Guide](docker/README.md) - Complete setup, validator bonding, and network configuration

### Source (Development)

**For development and contributing to F1r3fly**

#### Prerequisites

1. **Install Nix**: https://nixos.org/download/
2. **Install direnv**: https://direnv.net/#basic-installation

#### Setup

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

> The initial setup will compile all libraries (takes a few minutes).

### macOS

**Docker is recommended for macOS**

```bash
# Install Docker Desktop for Mac, then:
docker compose -f docker/standalone.yml up
```

Native macOS packages are not currently available.

### System Packages (Legacy - Hybrid Node)

> **Note**: These packages install the legacy hybrid Scala+Rust node which requires Java. For new deployments, use [Docker](#docker-recommended) with the pure Rust node.

<details>
<summary>Debian/Ubuntu (Legacy)</summary>

**Dependency**: `java17-runtime-headless`

```bash
# Download from GitHub Releases
sudo apt update
sudo apt install ./rnode_X.Y.Z_all.deb
rnode run -s
```

**Paths**: `/usr/bin/rnode`, `/usr/share/rnode/rnode.jar`

</details>

<details>
<summary>RedHat/Fedora (Legacy)</summary>

**Dependency**: `java-17-openjdk`

```bash
# Download from GitHub Releases
sudo dnf install ./rnode-X.Y.Z-1.noarch.rpm
rnode run -s
```

**Paths**: `/usr/bin/rnode`, `/usr/share/rnode/rnode.jar`

</details>

## Building

**Prerequisites**: [Development environment setup](#source)

### Quick Commands

```bash
# Pure Rust Docker image (recommended)
./node/docker-commands.sh build-local

# Clean build
sbt "clean"

# Legacy: Fat JAR for local development (hybrid Scala + Rust)
sbt ";compile ;project node ;assembly ;project rchain"

# Legacy: Hybrid Docker image
docker context use default && sbt ";compile ;project node ;Docker/publishLocal ;project rchain"
```

### Building Rust Node Docker Image (Recommended)

**Build the Rust node Docker image directly using Docker (no SBT required)**

The Rust node Docker image is built from a multi-stage Dockerfile located at `node/Dockerfile`. This creates a pure Rust binary image without Java dependencies.

#### Quick Start

**Using helper script** (recommended):
```bash
# Source the helper script
source node/docker-commands.sh

# Build for local use (faster, single architecture)
docker_build_local

# Build for production (single architecture)
docker_build

# Build multi-architecture (amd64 + arm64, requires Docker buildx)
MULTI_ARCH=1 docker_build
```

**Direct execution**:
```bash
# Build locally
./node/docker-commands.sh build-local

# Build for production
./node/docker-commands.sh build

# Build multi-architecture
MULTI_ARCH=1 ./node/docker-commands.sh build
```

#### Manual Docker Build

**From workspace root**:
```bash
# Build for current architecture
docker build -f node/Dockerfile -t f1r3flyindustries/f1r3fly-rust-node:latest .

# Build for specific architecture
docker buildx build --platform linux/amd64 -f node/Dockerfile \
  -t f1r3flyindustries/f1r3fly-rust-node:amd64 .

# Multi-architecture build (requires buildx)
docker buildx build --platform linux/amd64,linux/arm64 \
  -f node/Dockerfile \
  -t f1r3flyindustries/f1r3fly-rust-node:latest \
  --push .
```

#### Image Details

- **Base images**: `rust:bookworm` (builder), `debian:12-slim` (runtime)
- **Binary**: Pure Rust `node` binary with all dependencies linked
- **Ports**: Exposes 40400-40404 (protocol, gRPC external/internal, HTTP API, discovery)
- **Healthcheck**: Uses `grpcurl` and `curl` with `jq` for service validation
- **User**: Runs as `daemon` user
- **Entrypoint**: `/opt/docker/bin/docker-entrypoint.sh` (automatically sets `--profile=docker`)

#### Publishing

```bash
# Using helper script
source node/docker-commands.sh
docker_publish
DRONE_BUILD_NUMBER=123 docker_publish_drone

# Direct execution
./node/docker-commands.sh publish
DRONE_BUILD_NUMBER=123 ./node/docker-commands.sh publish-drone
```

🐳 **Docker Setup**: [docker/README.md](docker/README.md)

### SBT Tips

- **Keep SBT running**: Use `sbt` shell for faster subsequent commands
- **Project-specific builds**: `sbt "project node" compile`
- **Parallel compilation**: Automatic with modern SBT

### Building Hybrid Docker Image (Legacy)

> **Note**: The hybrid node is deprecated. Use the pure Rust node for production deployments.

After setting up the [development environment](#source), build the hybrid Docker image:

**Native Build**:
```bash
docker context use default && sbt ";compile ;project node ;Docker/publishLocal ;project rchain"
```

**Cross-Platform Build** (AMD64 + ARM64):
```bash
docker context use default && MULTI_ARCH=true sbt ";compile ;project node ;Docker/publishLocal ;project rchain"
```

Both create: `f1r3flyindustries/f1r3fly-hybrid-node:latest`

## Running

### Docker Network (Recommended)

**Standalone Node** (simplest - for development):
```bash
# Single-validator node with instant finalization
docker compose -f docker/standalone.yml up

# Start in background
docker compose -f docker/standalone.yml up -d

# View logs
docker compose -f docker/standalone.yml logs -f

# Stop the node
docker compose -f docker/standalone.yml down
```

**Multi-Validator Network** (for testing consensus):
```bash
# Start the shard network (3 validators + bootstrap + observer)
docker compose -f docker/shard-with-autopropose.yml up

# Start in background
docker compose -f docker/shard-with-autopropose.yml up -d

# View logs
docker compose -f docker/shard-with-autopropose.yml logs -f

# Stop the network
docker compose -f docker/shard-with-autopropose.yml down
```

**Fresh Start**: Reset to genesis state:
```bash
# Stop network and remove all blockchain data
docker compose -f docker/standalone.yml down  # or shard-with-autopropose.yml
rm -rf docker/data/
docker compose -f docker/standalone.yml up    # or shard-with-autopropose.yml
```

**Observer Node** (optional - read-only access):
```bash
# Start observer (requires running shard network)
docker compose -f docker/observer.yml up
```

### Build and Run Rust Node

**Build the Rust node image and start the cluster**:

```bash
# Build the Rust node image and start standalone
./node/docker-commands.sh build-local && cd docker && docker-compose -f standalone.yml up

# Or start the full shard network
./node/docker-commands.sh build-local && cd docker && docker-compose -f shard-with-autopropose.yml up
```

**If image is already built**:
```bash
cd docker && docker-compose -f standalone.yml up
# OR
cd docker && docker-compose -f shard-with-autopropose.yml up
```

### Local Development (Pure Rust)

**Run the Rust node locally without Docker** - ideal for development and debugging.

#### Prerequisites

- [Development environment](#source-development) (Nix + direnv)
- `just` command runner (included in nix flake)

#### Quick Start

```bash
# Enter the development environment
direnv allow  # or: nix develop

# Build and run standalone node
just run-standalone
```

#### Available Commands

Run `just` to see all available commands:

| Command | Description |
|---------|-------------|
| `just build` | Build node in release mode |
| `just build-debug` | Build node in debug mode (faster compile) |
| `just run-standalone` | Build and run standalone node |
| `just run-standalone-debug` | Run in debug mode |
| `just setup-standalone` | Set up data directory only |
| `just clean-standalone` | Remove node data (fresh start) |
| `just help` | Show node CLI help |
| `just run-help` | Show 'run' subcommand options |

#### Configuration

Local configuration files are in [`run-local/`](run-local/):

```
run-local/
├── conf/standalone.conf     # Node configuration
├── genesis/standalone/      # Genesis files (bonds, wallets)
└── data/standalone/         # Node data (gitignored)
```

See [`run-local/README.md`](run-local/README.md) for details.

#### Fresh Start

```bash
just clean-standalone
just run-standalone
```

---

### Local Development Node (Legacy - Hybrid)

> **Note**: This uses the legacy hybrid Scala+Rust node. For production, use the pure Rust Docker image.

After [building from source](#building):

```bash
java -Djna.library.path=./rust_libraries/release \
  --add-opens java.base/sun.security.util=ALL-UNNAMED \
  --add-opens java.base/java.nio=ALL-UNNAMED \
  --add-opens java.base/sun.nio.ch=ALL-UNNAMED \
  -jar node/target/scala-2.12/rnode-assembly-1.0.0-SNAPSHOT.jar \
  run -s --no-upnp --allow-private-addresses --synchrony-constraint-threshold=0.0
```

**Fresh Start**: `rm -rf ~/.rnode/`

## Usage

### F1r3fly Rust Client

**Modern Rust-based CLI for interacting with F1r3fly nodes**

The F1r3fly Rust Client provides a comprehensive command-line interface for blockchain operations:

| Feature                | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| **Deploy**             | Upload Rholang code to F1r3fly nodes                         |
| **Propose**            | Create new blocks containing deployed code                   |
| **Full Deploy**        | Deploy + propose in a single operation                       |
| **Deploy & Wait**      | Deploy with automatic finalization checking                  |
| **Exploratory Deploy** | Execute Rholang without blockchain commitment (read-only)    |
| **Transfer**           | Send REV tokens between addresses                            |
| **Bond Validator**     | Add new validators to the network                            |
| **Network Health**     | Check validator status and network consensus                 |
| **Key Management**     | Generate public keys and key pairs for blockchain identities |

🔗 **Repository**: [F1R3FLY-io/rust-client](https://github.com/F1R3FLY-io/rust-client)

**Installation**:
```bash
git clone https://github.com/F1R3FLY-io/rust-client.git
cd rust-client
cargo build --release
```

**Quick Example**:
```bash
# Deploy a Rholang contract
cargo run -- deploy -f ./rho_examples/stdout.rho

# Check network status
cargo run -- status
```

### Evaluating Rholang Contracts

**Prerequisites**: [Running node](#running)

#### Quick Evaluation

1. **Build the evaluator**:
   ```bash
   sbt ";compile ;stage"
   ```

2. **Evaluate a contract**:
   ```bash
   ./node/target/universal/stage/bin/rnode \
     -Djna.library.path=./rust_libraries/release \
     eval ./rholang/examples/tut-ai.rho
   ```

#### Example Contracts

Explore the `rholang/examples/` directory for sample contracts and tutorials.

### F1r3flyFS

**Distributed file system built on F1r3fly**

F1r3flyFS provides a simple, fast file system interface on top of the F1r3fly blockchain.

🔗 **Project**: [F1r3flyFS Repository](https://github.com/F1R3FLY-io/f1r3flyfs#f1r3flyfs)

## Troubleshooting

### Common Issues and Solutions

#### Nix Issues

**Problem**: Unable to load `flake.nix` or general Nix problems
```bash
nix-garbage-collect
```

#### SBT Build Issues

**Problem**: Build failures or dependency issues
```bash
# Clear coursier cache
rm -rf ~/.cache/coursier/

# Clean SBT
sbt clean
```

#### Node Compilation Issues

**Problem**: StackOverflow error during compilation
```bash
# Option 1: Direct compile
sbt "node/compile"

# Option 2: Use SBT shell (recommended)
sbt
sbt:rchain> project node
sbt:node> compile
```

#### Rust Library Issues

**Problem**: Rust compilation or library loading errors
```bash
# Clean Rust libraries
./scripts/clean_rust_libraries.sh

# Reset Rust toolchain
rustup default stable
```

#### Docker Issues

**Problem**: Docker build failures
```bash
# Reset Docker context
docker context use default

# Clean Docker system
docker system prune -a
```

#### Port Conflicts

**Problem**: "Port already in use" errors
```bash
# Find processes using F1r3fly ports
lsof -i :40400-40404

# Kill specific process
kill -9 <PID>
```

## Configuration File

**Customize RNode behavior with HOCON configuration**

### Configuration Options

- **Default location**: Data directory (usually `~/.rnode/`)
- **Custom location**: Use `--config-file <path>` command line option
- **Format**: [HOCON](https://github.com/lightbend/config/blob/master/HOCON.md) (Human-Optimized Config Object Notation)

### Reference Configuration

- **All available options**: [defaults.conf](node/src/main/resources/defaults.conf)
- **Standalone example**: [docker/conf/standalone-dev.conf](docker/conf/standalone-dev.conf) - Ready-to-use development configuration

## Development

**Contributing to F1r3fly? Start here!**

### Development Setup

1. **Environment**: Follow [source installation](#source) instructions
2. **Docker**: Use `docker/shard-with-autopropose.yml` for testing
3. **Client**: Use [F1r3fly Rust Client](https://github.com/F1R3FLY-io/rust-client) for interaction

### Contribution Workflow

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test locally
4. Submit pull request

🐳 **Docker Guide**: [docker/README.md](docker/README.md) - Complete Docker setup, validator bonding, and network configuration

## Support & Community

### Discord Community

Join the F1r3fly community for real-time support, tutorials, and project updates:

🌐 **[F1r3fly Discord](https://discord.gg/NN59aFdAHM)**

**Available Resources**:
- Project tutorials and documentation
- Development planning and discussions
- Events calendar and announcements
- Community support and Q&A

### Getting Help

1. **Documentation**: Start with this README
2. **Troubleshooting**: Check the [troubleshooting section](#troubleshooting)
3. **Community**: Ask questions in Discord
4. **Issues**: Report bugs in GitHub Issues

## Caveats and Filing Issues

### Known Issues

⚠️ **[Beta-release Software**: This F1r3fly repository is under active development](#note-on-the-use-of-this-software)

**Current Issue Trackers**:
- **F1r3fly Issues**: [GitHub Issues](https://github.com/F1R3FLY-io/f1r3fly/issues)
- **RChain Legacy Issues**: [Legacy Bug Reports](https://github.com/rchain/rchain/issues?q=is%3Aopen+is%3Aissue+label%3Abug)

### Filing Bug Reports

**Report Issues**: [Create New Issue](https://github.com/F1R3FLY-io/f1r3fly/issues/new)

**Include in your report**:
- F1r3fly version
- Operating system and version
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs or error messages

## Acknowledgements

**Performance Profiling**: 
We use [YourKit](https://www.yourkit.com/) to profile F1r3fly performance. YourKit supports open source projects with their full-featured Java and .NET profilers.

**Tools**:
- [YourKit Java Profiler](https://www.yourkit.com/java/profiler/)
- [YourKit .NET Profiler](https://www.yourkit.com/.net/profiler/)

## License Information

F1r3fly is licensed under the **Apache License 2.0**.
