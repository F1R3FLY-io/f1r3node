# Node Operations API

## Overview
The Node Operations API provides access to core RNode functionality including status monitoring, block queries, and system administration.

> **Note**: All `cargo run` commands in this document should be executed from the `node-cli` folder or using the maintained version at https://github.com/F1R3FLY-io/rust-client

## Base URL
```
gRPC: localhost:40402 (internal) / localhost:40401 (external)
HTTP: localhost:40403
```

## Authentication
All operations require valid authentication using:
- Private key signing for administrative operations
- Public API access for read-only queries

## Core Operations

### Node Status

#### Get Node Status
Returns current node status and health information.

**gRPC**
```protobuf
service NodeService {
  rpc GetStatus(StatusRequest) returns (StatusResponse);
}

message StatusResponse {
  string version = 1;
  int64 uptime = 2;
  string consensus_mode = 3;
  int32 peer_count = 4;
  int64 latest_block_number = 5;
  string node_id = 6;
  bool sync_status = 7;
}
```

**HTTP**
```http
GET /api/status

Response:
{
  "version": "1.0.0",
  "uptime": 86400000,
  "consensus_mode": "casper_cbc",
  "peer_count": 12,
  "latest_block_number": 1000,
  "node_id": "rnode://1234...abcd",
  "sync_status": true
}
```

**CLI**
```bash
cargo run -- status
```

### Block Operations

#### Get Block by Hash
Retrieve a specific block by its hash.

**gRPC**
```protobuf
service BlockService {
  rpc GetBlock(GetBlockRequest) returns (Block);
}

message GetBlockRequest {
  string block_hash = 1;
}
```

**HTTP**
```http
GET /api/blocks/{block_hash}

Response:
{
  "block_hash": "0x1234...abcd",
  "header": {
    "parent_hash_list": ["0xparent1", "0xparent2"],
    "timestamp": 1640000000000,
    "version": 1
  },
  "body": {
    "deploys": [...],
    "post_state_hash": "0xstate123"
  }
}
```

#### Get Latest Block
Retrieve the most recent finalized block.

**HTTP**
```http
GET /api/blocks/latest
```

**CLI**
```bash
cargo run -- latest-block
```

#### Get Block Range
Retrieve multiple blocks within a specified range.

**HTTP**
```http
GET /api/blocks?from=100&to=110

Response:
{
  "blocks": [...],
  "total_count": 10
}
```

### Deploy Operations

#### Deploy Contract
Deploy a Rholang contract to the blockchain.

**gRPC**
```protobuf
service DeployService {
  rpc Deploy(DeployRequest) returns (DeployResponse);
}

message DeployRequest {
  string rholang_code = 1;
  string private_key = 2;
  int64 phlo_limit = 3;
  int64 phlo_price = 4;
  int64 nonce = 5;
}
```

**HTTP**
```http
POST /api/deploy
Content-Type: application/json

{
  "rholang_code": "new stdout(`rho:io:stdout`) in { stdout!(\"Hello World\") }",
  "private_key": "0x1234...abcd",
  "phlo_limit": 100000,
  "phlo_price": 1,
  "nonce": 42
}

Response:
{
  "deploy_id": "0xdeploy123",
  "status": "pending",
  "cost": 1500
}
```

**CLI**
```bash
cargo run -- deploy -f contract.rho --private-key $PRIVATE_KEY
```

#### Propose Block
Propose a new block containing pending deploys.

**CLI**
```bash
cargo run -- propose --private-key $VALIDATOR_KEY
```

### Query Operations

#### Exploratory Deploy  
Execute Rholang code without committing to blockchain (read-only).

**CLI**
```bash
cargo run -- exploratory-deploy -f query.rho
```

#### Get Deploy Status
Check the status of a deployed contract.

**HTTP**
```http
GET /api/deploys/{deploy_id}/status

Response:
{
  "deploy_id": "0xdeploy123",
  "status": "finalized",
  "block_hash": "0xblock456",
  "cost": 1500,
  "error_message": null
}
```

### System Administration

#### Get Node Configuration
Retrieve current node configuration.

**HTTP**
```http
GET /api/admin/config

Response:
{
  "network_id": "mainnet",
  "validator_key": "0xvalidator...",
  "consensus_mode": "casper_cbc",
  "max_peers": 50,
  "data_dir": "/var/lib/rnode"
}
```

#### Shutdown Node
Gracefully shutdown the node (requires admin privileges).

**HTTP**
```http
POST /api/admin/shutdown
Authorization: Signature {signature}
```

## Error Response Format

All APIs return standardized error responses:

```json
{
  "error": {
    "code": "INVALID_DEPLOY",
    "message": "Deploy validation failed",
    "details": {
      "field": "rholang_code",
      "reason": "Syntax error at line 5"
    }
  }
}
```

## Common Error Codes

- `INVALID_REQUEST`: Malformed request parameters
- `AUTHENTICATION_FAILED`: Invalid or missing authentication
- `DEPLOY_VALIDATION_FAILED`: Contract code validation error
- `INSUFFICIENT_PHLO`: Not enough gas for execution
- `BLOCK_NOT_FOUND`: Requested block does not exist
- `NODE_NOT_READY`: Node is not ready to process requests
- `CONSENSUS_ERROR`: Consensus mechanism error

## Rate Limiting

API endpoints are rate limited to prevent abuse:
- Public endpoints: 100 requests/minute
- Authenticated endpoints: 1000 requests/minute
- Administrative endpoints: 10 requests/minute

## Examples

### Deploy and Query Contract
```bash
# Deploy a simple contract
DEPLOY_ID=$(cargo run -- deploy -f hello.rho --private-key $KEY | grep deploy_id)

# Propose a block
cargo run -- propose --private-key $VALIDATOR_KEY

# Check deploy status
curl -X GET "http://localhost:40403/api/deploys/$DEPLOY_ID/status"

# Query the result
cargo run -- exploratory-deploy -f query.rho
```

### Monitor Node Status
```bash
# Continuous monitoring
while true; do
  curl -s http://localhost:40403/api/status | jq .
  sleep 5
done
```