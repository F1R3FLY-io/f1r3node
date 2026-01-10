# RNode - F1R3FLY.io Blockchain Platform

## Project Context
- RNode is a decentralized blockchain platform developed by F1R3FLY.io that implements four distinct consensus mechanisms using the Rholang programming language
- This is a **multi-language blockchain project** with Scala for node infrastructure and Rust for performance-critical components
- The platform provides concurrent smart contract execution with Byzantine Fault Tolerant operations
- If the user does not provide enough information with their prompts, ask the user to clarify before executing the task
- Follow F1R3FLY.io's documentation-first development methodology for all features

## Platform Requirements
- **Multi-Platform Support**: Linux, macOS (Intel & Apple Silicon), with Docker containerization
- **Nix with Direnv** - Primary development environment management (`nix develop` or `direnv allow`)
- **Scala 2.12+** - JVM-based node infrastructure with SBT build system
- **Rust 1.70+** - Performance components with Cargo build system
- **Docker** - For deployment and testing (`docker compose -f docker/shard.yml up`)

## Architecture Overview

### Multi-Consensus Design
RNode implements **four consensus mechanisms**, all implemented in Rholang:
1. **Cordial Miners** - Cooperative, energy-efficient mining approach
2. **Casper CBC** - Byzantine Fault Tolerant consensus with mathematical safety proofs
3. **RGB Partially Synchronized State Machines** - Client-side validation with Bitcoin anchoring
4. **Casanova** - Adaptive consensus for high-performance scenarios

### Core Components
- **Node** (Scala) - Main node process and lifecycle management
- **Casper** (Scala) - Multi-consensus mechanisms and validator logic
- **Comm** (Scala + Rust) - P2P networking with custom TLS validation
- **Rholang** (Rust) - Concurrent programming language interpreter
- **RSpace** (Rust) - High-performance tuple space storage (LMDB backend)
- **Bitcoin Anchor** (Rust) - Layer 1 anchoring with RGB protocol integration
- **Node CLI** (Rust) - Command-line interface for node interaction
- **Crypto** (Scala + Rust) - Cryptographic primitives and utilities

## Environment Setup
```bash
# Primary setup (Nix/Direnv recommended)
direnv allow  # Enters development shell with all dependencies

# Manual Nix setup
nix develop

# Verify environment
sbt compile  # Scala components
cargo build  # Rust components
```

## Development Commands

### Core Node Operations
```bash
# Build project
sbt ";compile ;project node ;Docker/publishLocal ;project rchain"

# Start development network
docker compose -f docker/shard.yml up

# Clean build
sbt clean
./scripts/clean_rust_libraries.sh

# Run tests
./scripts/run_rust_tests.sh  # Rust tests
sbt test                     # Scala tests
```

### Smart Contract Development
```bash
# Deploy Rholang contract
cargo run -- deploy -f contract.rho --private-key $PRIVATE_KEY

# Propose block (validators only)
cargo run -- propose --private-key $VALIDATOR_KEY

# Check block finalization
cargo run -- is-finalized -b $BLOCK_HASH

# Read-only contract execution
cargo run -- exploratory-deploy -f query.rho

# Generate key pairs
cargo run -- generate-key-pair --save
```

### Network Operations
```bash
# Node status
cargo run -- status

# Monitor logs
tail -f ~/.rnode/rnode.log

# Delete data (fresh start)
./scripts/delete_data.sh
```

## Code Style and Standards

### Scala Components
- **Style**: Functional programming with Akka actors
- **Build**: SBT with multi-project structure
- **Testing**: ScalaTest with property-based testing
- **Concurrency**: Actor model for concurrent operations

### Rust Components  
- **Style**: Zero-cost abstractions, ownership model
- **Build**: Cargo with workspace management
- **Testing**: Built-in test framework with proptest
- **Performance**: Optimize for concurrent execution

### Rholang Smart Contracts
- **Paradigm**: Process calculus based concurrent programming
- **Communication**: Channel-based message passing
- **Security**: Unforgeable names and capability-based security
- **Pattern Matching**: Structural pattern matching on channels

### General Guidelines
- **No Comments**: Unless explicitly requested by user
- **Documentation-First**: All features begin with documentation
- **Security-First**: Never log private keys, validate all inputs
- **Multi-Consensus Aware**: Design for consensus mechanism switching

## F1R3FLY.io Documentation Standards

### Documentation Structure
```
docs/
├── requirements/           # User stories, business requirements, acceptance criteria
├── specifications/         # Technical specs, API definitions, data schemas  
├── architecture/          # System design, ADRs, diagrams, patterns
├── api/                   # API documentation and examples
└── ToDos.md              # Current project status and active tasks
```

### Document Types
- **Requirements**: `US-*` (User Stories), `BR-*` (Business Requirements), `AC-*` (Acceptance Criteria)
- **Specifications**: `SPEC-*` (Technical), `API-*` (APIs), `INT-*` (Integration)
- **Architecture**: `ADR-*` (Architecture Decision Records)

## Current Development Status

### Completed Components
- [x] Multi-language build system (SBT + Cargo)
- [x] Nix/Direnv development environment
- [x] Basic Casper CBC consensus (Scala implementation)
- [x] Rholang language interpreter (Rust)
- [x] RSpace tuple space storage (Rust)
- [x] P2P networking with TLS validation
- [x] Node CLI tools (Rust)
- [x] Bitcoin anchoring with RGB protocol integration
- [x] Docker containerization

### Active Development
- [ ] **Cordial Miners consensus** - Implement in Rholang
- [ ] **RGB PSSM consensus** - Client-side validation framework
- [ ] **Casanova consensus** - Adaptive consensus algorithm
- [ ] **Multi-consensus switching** - Runtime consensus selection
- [ ] **Enhanced gas accounting** - Complete phlogiston system
- [ ] **Performance optimization** - Concurrent execution improvements

## Smart Contract Development

### Rholang Language Features
```rholang
// Process creation and parallel composition
new channel in {
  channel!("Hello") |  // Send
  for (msg <- channel) { // Receive
    // Process message
  }
}

// Pattern matching
for (@{"action": "transfer", "amount": amount} <- requestChan) {
  // Handle transfer request
}

// Unforgeable names for security
new unforgeableName in {
  // This name is cryptographically unique
}
```

### Example Smart Contract Workflow
```bash
# 1. Write contract in Rholang
cat > hello.rho << 'EOF'
new stdout(`rho:io:stdout`) in {
  stdout!("Hello, RNode!")
}
EOF

# 2. Deploy contract
cargo run -- deploy -f hello.rho --private-key $PRIVATE_KEY

# 3. Propose block to include deploy
cargo run -- propose --private-key $VALIDATOR_KEY

# 4. Wait for finalization
cargo run -- is-finalized -b $BLOCK_HASH
```

## Testing Strategy

### Comprehensive Testing
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Property-Based Tests**: Test invariants across consensus mechanisms
- **Performance Tests**: Benchmark concurrent execution
- **Security Tests**: Validate cryptographic operations

### Testing Commands
```bash
# Rust components
cargo test                    # Unit tests
cargo test --release         # Performance tests
./scripts/run_rust_tests.sh  # Full test suite

# Scala components  
sbt test                     # Unit tests
sbt "project casper" test    # Specific project tests

# Integration tests
cd integration-tests && python -m pytest
```

## Security Considerations

### Blockchain Security
- **Consensus Safety**: Multiple consensus mechanisms with BFT guarantees
- **Cryptographic Security**: Ed25519 and Secp256k1 signatures
- **Network Security**: TLS 1.3 for all P2P communications
- **Economic Security**: Validator bonding and slashing mechanisms

### Smart Contract Security
- **Capability Security**: Object capabilities through unforgeable names
- **Resource Limits**: Phlogiston gas system prevents resource exhaustion
- **Isolation**: Contracts execute in isolated environments
- **Deterministic Execution**: Consistent results across all nodes

### Development Security
- **Private Key Safety**: Never log or expose private keys
- **Input Validation**: Validate all user inputs and state transitions
- **Error Handling**: Secure error messages without information leakage
- **Dependency Management**: Regular security audits of dependencies

## Git and Version Control

### File Management
- **DO NOT** run `git add`, `git rm`, `git mv`, or `git commit` commands
- **Instruct users** to commit changes when needed
- **What to Commit**:
  - ✅ Source code, tests, documentation, build files
  - ❌ Generated files, logs, private keys, build artifacts

### Branch Strategy
- **Main Branch**: `main` - stable releases
- **Development**: Feature branches for new consensus mechanisms
- **Releases**: Tagged releases for versions
- **Hotfixes**: Critical security patches

## Common Development Tasks

### Adding New Consensus Mechanism
1. **Requirements**: Document in `docs/requirements/`
2. **Specification**: Define protocol in `docs/specifications/`
3. **Architecture**: Create ADR in `docs/architecture/decisions/`
4. **Implementation**: Write Rholang contracts
5. **Testing**: Comprehensive test suite
6. **Integration**: Add to consensus switching framework

### Performance Optimization
1. **Profiling**: Use built-in profilers for hotspots
2. **Benchmarking**: Measure before and after changes
3. **Concurrent Design**: Leverage RSpace parallelism
4. **Memory Management**: Optimize for LMDB storage
5. **Network Optimization**: Reduce P2P message overhead

### Debugging Guidelines
- **Logs**: Check `~/.rnode/rnode.log` for detailed logs
- **REPL**: Use `sbt console` for interactive debugging
- **Exploratory Deploy**: Test contracts without blockchain commitment
- **Network Issues**: Verify P2P connectivity and TLS certificates
- **State Inspection**: Use RSpace debugging tools

## Troubleshooting

### Environment Issues
```bash
# Nix problems
nix-garbage-collect
direnv reload

# SBT build problems  
rm -rf ~/.cache/coursier/
sbt clean

# Rust problems
./scripts/clean_rust_libraries.sh
rustup default stable
```

### Runtime Issues
- **Out of Memory**: Increase JVM heap size in node configuration
- **Network Partition**: Check P2P connectivity and firewall rules
- **Consensus Stuck**: Verify validator bonds and network synchronization
- **Contract Execution**: Check phlogiston limits and contract syntax

## Project Specifics
- **Multi-Consensus Focus**: All consensus mechanisms implemented in Rholang
- **Performance Critical**: RSpace and networking components optimized in Rust
- **Security First**: Comprehensive security model with capability-based contracts
- **Enterprise Ready**: Docker deployment with monitoring and metrics
- **Bitcoin Integration**: L1 anchoring through RGB protocol for additional security
- **Developer Experience**: Comprehensive CLI tools and development environment