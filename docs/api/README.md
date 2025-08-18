# RNode API Documentation

This directory contains comprehensive API documentation for the RNode platform.

## API Overview

RNode provides multiple API interfaces for different use cases:

### 1. gRPC APIs
- **Node API**: Core node operations and queries
- **Deploy API**: Smart contract deployment and execution
- **Consensus API**: Consensus mechanism control and monitoring
- **Network API**: P2P network operations and status

### 2. REST APIs  
- **HTTP Gateway**: RESTful interface for web applications
- **WebSocket Events**: Real-time blockchain events
- **Metrics API**: Node performance and health metrics

### 3. Command Line Interface
- **Node CLI**: Direct node interaction and administration
- **Developer Tools**: Contract development and debugging utilities

## API Categories

### Core Node Operations
- Node status and health checks
- Block retrieval and validation
- Transaction processing and queries
- State inspection and debugging

### Smart Contract Platform
- Contract deployment and invocation
- Rholang code execution
- Gas/cost estimation and tracking
- Contract state queries

### Multi-Consensus Management
- Consensus mechanism selection
- Consensus switching protocols
- Validator operations and bonding
- Network condition monitoring

### Network & P2P
- Peer discovery and management
- Block propagation control
- Network partition handling
- Communication security

## API Standards

### Authentication
- Public key based authentication
- Digital signature verification
- Rate limiting and access control
- API key management for services

### Error Handling
- Standardized error codes and messages
- Detailed error context and debugging info
- Graceful degradation strategies
- Retry mechanisms and timeouts

### Data Formats
- Protocol Buffers for efficient serialization
- JSON for human-readable APIs
- Binary formats for high-performance operations
- Standardized timestamps and encoding

## Getting Started

### Prerequisites
- Running RNode instance
- Valid authentication credentials
- Network connectivity to node
- API client libraries (optional)

### Quick Start
```bash
# Check node status
curl -X GET http://localhost:40403/api/status

# Deploy a contract (run from node-cli folder or https://github.com/F1R3FLY-io/rust-client)
cargo run -- deploy -f contract.rho --private-key $KEY

# Query blockchain state
curl -X GET http://localhost:40403/api/blocks/latest
```

### API Clients
- **Rust**: Official node-cli implementation
- **JavaScript/TypeScript**: Web and Node.js applications
- **Python**: Integration testing and scripting
- **Go**: Enterprise applications

## Available APIs

- **[Node Operations](node-operations.md)** - Core node functionality
- **[Smart Contracts](smart-contracts.md)** - Contract deployment and execution
- **[Consensus APIs](consensus-apis.md)** - Multi-consensus management
- **[Network APIs](network-apis.md)** - P2P networking operations
- **[WebSocket Events](websocket-events.md)** - Real-time event streaming
- **[CLI Reference](cli-reference.md)** - Command-line interface guide

## Examples and Tutorials

- **[Basic Usage Examples](examples/)** - Common API usage patterns
- **[Contract Development](tutorials/contract-development.md)** - Smart contract workflow
- **[Network Administration](tutorials/network-admin.md)** - Node operation guides
- **[Integration Patterns](tutorials/integration.md)** - Application integration

## API Versioning

RNode APIs follow semantic versioning:
- **v1.x**: Current stable API
- **v2.x**: Next generation API (under development)
- **experimental**: Cutting-edge features (unstable)

## Support and Community

- **Documentation Issues**: GitHub repository issues
- **API Questions**: F1R3FLY Discord community
- **Feature Requests**: GitHub feature request template
- **Security Issues**: security@f1r3fly.io