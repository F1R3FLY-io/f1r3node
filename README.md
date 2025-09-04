# F1r3fly

> A decentralized, economic, censorship-resistant, public compute infrastructure and blockchain that hosts and executes smart contracts with trustworthy, scalable, concurrent proof-of-stake consensus.

## Table of Contents

- [What is F1r3fly?](#what-is-f1r3fly)
- [Security Notice](#note-on-the-use-of-this-software)
- [Installation](#installation)
  - [Source (Development)](#source)
  - [Docker](#docker)
  - [Debian/Ubuntu](#debianubuntu)
  - [RedHat/Fedora](#redhatfedora)
  - [macOS](#macos)
- [Building](#building)
- [Running](#running)
- [Usage](#usage)
  - [Node CLI](#node-cli)
  - [Evaluating Rholang Contracts](#evaluating-rholang-contracts)
  - [F1r3flyFS](#f1r3flyfs)
- [Configuration](#configuration-file)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
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

## Note on the use of this software
This code has not yet completed a security review. We strongly recommend that you do not use it in production or to transfer items of material value. We take no responsibility for any loss you may incur through the use of this code.

## Installation

### Source

**Recommended for development and contributing to F1r3fly**

#### Prerequisites

1. **Install Nix**: https://nixos.org/download/
   - Learn more: [How Nix Works](https://nixos.org/guides/how-nix-works/)

2. **Install direnv**: https://direnv.net/#basic-installation
   - Learn more: [direnv documentation](https://direnv.net/)

#### Setup Steps

3. **Clone and setup the repository**:
   ```bash
   git clone <repository-url>
   cd f1r3fly-build
   direnv allow
   ```

   **Troubleshooting**: If you encounter the error:
   ```
   error: experimental Nix feature 'nix-command' is disabled
   ```

   **Solution**:
   ```bash
   # Create Nix config directory
   mkdir -p ~/.config/nix
   
   # Add experimental features
   echo "experimental-features = flakes nix-command" > ~/.config/nix/nix.conf
   
   # Try again
   direnv allow
   ```

   > 📝 **Note**: The initial setup will compile all libraries (takes a few minutes). Your development environment will be ready after completion.

### Docker

**Recommended for production deployments and testing**

#### Quick Start

Run a complete F1r3fly network with automatic block production:

```bash
# Start the shard network (recommended)
docker compose -f docker/shard-with-autopropose.yml up
```

#### Port Configuration

| Port  | Service              | Description                    |
|-------|---------------------|--------------------------------|
| 40400 | Protocol Server     | Main blockchain protocol       |
| 40401 | gRPC External       | External gRPC API              |
| 40402 | gRPC Internal       | Internal gRPC API              |
| 40403 | HTTP API            | REST/HTTP API endpoints        |
| 40404 | Peer Discovery      | Node discovery service         |

#### Advanced Options

**Data Persistence**: The shard network automatically creates a `docker/data/` directory for blockchain state.

**Fresh Start**: Remove data directory to reset to genesis:
```bash
docker compose -f docker/shard-with-autopropose.yml down
rm -rf docker/data/
docker compose -f docker/shard-with-autopropose.yml up
```

**Additional Resources**: [Docker Hub Repository](https://hub.docker.com/r/f1r3flyindustries/f1r3fly-rust-node)

#### Building Docker Images Locally

After setting up the [development environment](#source), build Docker images:

**Native Build** (Recommended - 3-5x faster):
```bash
docker context use default && sbt ";compile ;project node ;Docker/publishLocal ;project rchain"
```

**Cross-Platform Build** (Production - AMD64 + ARM64):
```bash
docker context use default && MULTI_ARCH=true sbt ";compile ;project node ;Docker/publishLocal ;project rchain"
```

Both create: `f1r3flyindustries/f1r3fly-rust-node:latest`

### Debian/Ubuntu

**Pre-built packages for Debian-based systems**

F1r3fly provides Debian packages with RNode binary and Rust libraries. **Dependency**: `java17-runtime-headless`

#### Installation Steps

1. **Download**: Visit [GitHub Releases](https://github.com/F1R3FLY-io/f1r3fly/releases) and download `rnode_X.Y.Z_all.deb`

2. **Install**:
   ```bash
   sudo apt update
   sudo apt install ./rnode_X.Y.Z_all.deb
   ```
   > Replace `X.Y.Z` with the actual version number

3. **Run**:
   ```bash
   rnode run -s
   ```

**Installation Paths**:
- Binary: `/usr/bin/rnode`
- JAR: `/usr/share/rnode/rnode.jar`

#### Build Package Locally (Optional)

```bash
# Setup development environment first
     sbt "project node;debian:packageBin"

# Package location: node/target/rnode_X.Y.Z_all.deb
     ```

### RedHat/Fedora

**Pre-built packages for RPM-based systems**

F1r3fly provides RPM packages with RNode binary and Rust libraries. **Dependency**: `java-17-openjdk`

#### Installation Steps

1. **Download**: Visit [GitHub Releases](https://github.com/F1R3FLY-io/f1r3fly/releases) and download `rnode-X.Y.Z-1.noarch.rpm`

2. **Install**:
   ```bash
   # For Fedora/newer systems
   sudo dnf install ./rnode-X.Y.Z-1.noarch.rpm
   
   # For CentOS/older systems
     sudo yum install ./rnode-X.Y.Z-1.noarch.rpm
     ```
   > Replace `X.Y.Z` with the actual version number

3. **Run**:
   ```bash
   rnode run -s
   ```

**Installation Paths**:
- Binary: `/usr/bin/rnode`
- JAR: `/usr/share/rnode/rnode.jar`

#### Build Package Locally (Optional)

```bash
# Setup development environment first
     sbt "project node;rpm:packageBin"

# Package location: node/target/rpm/RPMS/noarch/rnode-X.Y.Z-1.noarch.rpm
     ```

### macOS

**Currently experimental - Docker or source build recommended**

macOS native packages are not yet available. Choose one of these options:

#### Option 1: Docker (Recommended)
Use the [Docker installation method](#docker) for the best macOS experience.

#### Option 2: Build from Source
1. Set up the [development environment](#source) using Nix and direnv
2. Build and run:
   ```bash
   # Build fat JAR
   sbt ";compile ;project node ;assembly ;project rchain"
   
   # Run RNode
   java -Djna.library.path=./rust_libraries/release \
     --add-opens java.base/sun.security.util=ALL-UNNAMED \
     --add-opens java.base/java.nio=ALL-UNNAMED \
     --add-opens java.base/sun.nio.ch=ALL-UNNAMED \
     -jar node/target/scala-2.12/rnode-assembly-1.0.0-SNAPSHOT.jar run -s
   ```

> 🔮 **Future**: Native macOS packages may be added in upcoming releases.

## Building

**Prerequisites**: [Development environment setup](#source)

### Quick Commands

```bash
# Fat JAR for local development
sbt ";compile ;project node ;assembly ;project rchain"

# Docker image (native - faster for development)
docker context use default && sbt ";compile ;project node ;Docker/publishLocal ;project rchain"

# Docker image (cross-platform - for production)
docker context use default && MULTI_ARCH=true sbt ";compile ;project node ;Docker/publishLocal ;project rchain"

# Clean build
sbt "clean"
```

### SBT Tips

- **Keep SBT running**: Use `sbt` shell for faster subsequent commands
- **Project-specific builds**: `sbt "project node" compile`
- **Parallel compilation**: Automatic with modern SBT

🐳 **Docker Setup**: [docker/README.md](docker/README.md)

## Running

### Docker Network (Recommended)

**F1r3fly Shard with Auto-propose** - Complete blockchain network with automatic block production:

```bash
# Start the shard network (recommended)
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
docker compose -f docker/shard-with-autopropose.yml down
rm -rf docker/data/
docker compose -f docker/shard-with-autopropose.yml up
```

**Observer Node** (optional - read-only access):
```bash
# Start observer (requires running shard network)
docker compose -f docker/observer.yml up
```

### Local Development Node

After [building from source](#fat-jar-local-development):

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

| Feature | Description |
|---------|-------------|
| **Deploy** | Upload Rholang code to F1r3fly nodes |
| **Propose** | Create new blocks containing deployed code |
| **Full Deploy** | Deploy + propose in a single operation |
| **Deploy & Wait** | Deploy with automatic finalization checking |
| **Exploratory Deploy** | Execute Rholang without blockchain commitment (read-only) |
| **Transfer** | Send REV tokens between addresses |
| **Bond Validator** | Add new validators to the network |
| **Network Health** | Check validator status and network consensus |
| **Key Management** | Generate public keys and key pairs for blockchain identities |

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

View all available options: [defaults.conf](node/src/main/resources/defaults.conf)

### Example Configuration

```hocon
# Basic node configuration
standalone = false

# Protocol server settings
protocol-server {
  network-id = "testnet"
  port = 40400
}

# Bootstrap configuration
protocol-client {
  network-id = "testnet"
  bootstrap = "rnode://de6eed5d00cf080fc587eeb412cb31a75fd10358@52.119.8.109?protocol=40400&discovery=40404"
}

# Peer discovery
peers-discovery {
  port = 40404
}

# API server configuration
api-server {
  host = "my-rnode.domain.com"
  port-grpc-external = 40401
  port-grpc-internal = 40402
  port-http = 40403
  port-admin-http = 40405
}

# Storage settings
storage {
  data-dir = "/my-data-dir"
}

# Casper consensus configuration
casper {
  fault-tolerance-threshold = 1
  shard-name = "root"
  finalization-rate = 1
}

# Metrics and monitoring
metrics {
  prometheus = false
  influxdb = false
  influxdb-udp = false
  zipkin = false
  sigar = false
}

# Development mode
dev-mode = false
```

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

1. **Documentation**: Start with this README and the [Node CLI docs](node-cli/README.md)
2. **Troubleshooting**: Check the [troubleshooting section](#troubleshooting)
3. **Community**: Ask questions in Discord
4. **Issues**: Report bugs in GitHub Issues

## Caveats and Filing Issues

### Known Issues

⚠️ **Pre-release Software**: F1r3fly is under active development

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
