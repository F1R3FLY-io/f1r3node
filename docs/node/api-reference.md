# API Reference

## Ports

| Port | Protocol | Service |
|------|----------|---------|
| 40401 | gRPC | DeployService (external) |
| 40402 | gRPC | ProposeService (internal) |
| 40403 | HTTP | Public REST API |
| 40405 | HTTP | Admin REST API |

---

## HTTP REST API (port 40403)

### `GET /api/status`

Node identity, network membership, and operational state. Available on all node types.

**Parameters:** None

```bash
curl http://localhost:40403/api/status
```

```json
{
  "version": {"api": "1", "node": "F1r3node Rust 0.4.10 ()"},
  "address": "rnode://1e780e5d...@rnode.bootstrap?protocol=40400&discovery=40404",
  "networkId": "testnet",
  "shardId": "root",
  "peers": 4,
  "nodes": 4,
  "minPhloPrice": 1,
  "peerList": [
    {"address": "rnode://...", "nodeId": "a5aec03d...", "host": "rnode.validator3", "protocolPort": 40400, "discoveryPort": 40404, "isConnected": true}
  ],
  "nativeTokenName": "F1R3CAP",
  "nativeTokenSymbol": "F1R3",
  "nativeTokenDecimals": 8,
  "lastFinalizedBlockNumber": 28,
  "isValidator": false,
  "isReadOnly": false,
  "isReady": true,
  "currentEpoch": 2,
  "epochLength": 10
}
```

| Field | Type | Description |
|-------|------|-------------|
| `version` | object | API and node version |
| `address` | string | Node's rnode:// address |
| `networkId` | string | Network identifier |
| `shardId` | string | Shard identifier |
| `peers` | int | Connected peer count |
| `nodes` | int | Discovered node count |
| `minPhloPrice` | int | Minimum phlogiston price for deploys |
| `peerList` | array | Detailed peer info with connection status |
| `nativeTokenName` | string | Full token name from genesis |
| `nativeTokenSymbol` | string | Token ticker symbol |
| `nativeTokenDecimals` | int | Decimal places (dust per token = 10^decimals) |
| `lastFinalizedBlockNumber` | int | LFB block number. `-1` if casper not yet initialized |
| `isValidator` | bool | `true` if the node can propose blocks |
| `isReadOnly` | bool | `true` if running in read-only mode |
| `isReady` | bool | `true` after engine enters Running state |
| `currentEpoch` | int | `lastFinalizedBlockNumber / epochLength` |
| `epochLength` | int | Blocks per epoch, from genesis configuration |

---

### Block Endpoints

All block endpoints support `?view=full|summary`. Single-item endpoints default to **full** (includes deploys). List endpoints default to **summary** (block headers only, deploys omitted).

#### `GET /api/block/{hash}`

Get a block by hash. Default: full (with deploys).

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `hash` | path | yes | Block hash (hex) |
| `view` | query | no | `full` (default) or `summary` |

```bash
curl http://localhost:40403/api/block/3bfdf56f...
curl "http://localhost:40403/api/block/3bfdf56f...?view=summary"
```

Full response includes `blockInfo` (header with `isFinalized`) + `deploys` array. Summary omits `deploys`.

#### `GET /api/last-finalized-block`

Get the last finalized block. Default: full.

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `view` | query | no | `full` (default) or `summary` |

```bash
curl http://localhost:40403/api/last-finalized-block
curl "http://localhost:40403/api/last-finalized-block?view=summary"
```

#### `GET /api/blocks/{depth}`

Get recent blocks by depth. Default: summary.

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `depth` | path | yes | Number of block heights to return |
| `view` | query | no | `summary` (default) or `full` |

```bash
curl http://localhost:40403/api/blocks/5
curl "http://localhost:40403/api/blocks/5?view=full"
```

#### `GET /api/blocks/{start}/{end}`

Get blocks by height range. Default: summary.

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `start` | path | yes | Start block height (inclusive) |
| `end` | path | yes | End block height (inclusive) |
| `view` | query | no | `summary` (default) or `full` |

```bash
curl http://localhost:40403/api/blocks/100/110
```

#### `GET /api/blocks`

Get the most recent block. Default: summary.

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `view` | query | no | `summary` (default) or `full` |

```bash
curl http://localhost:40403/api/blocks
```

#### `GET /api/is-finalized/{hash}`

Check if a block is finalized. Returns `true` or `false`.

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `hash` | path | yes | Block hash (hex) |

```bash
curl http://localhost:40403/api/is-finalized/3bfdf56f...
```

---

### Deploy Endpoints

#### `POST /api/deploy`

Submit a signed deploy to the network. Validator nodes only.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `data.term` | string | yes | Rholang source code |
| `data.timestamp` | int | yes | Deploy timestamp (ms since epoch) |
| `data.phloPrice` | int | yes | Phlogiston price per unit |
| `data.phloLimit` | int | yes | Maximum phlogiston to consume |
| `data.validAfterBlockNumber` | int | yes | Deploy valid after this block number |
| `data.shardId` | string | yes | Target shard (e.g. `"root"`) |
| `deployer` | string | yes | Deployer public key (hex) |
| `signature` | string | yes | Deploy signature (hex) |
| `sigAlgorithm` | string | yes | Signature algorithm (`"secp256k1"`) |

```bash
curl -X POST http://localhost:40413/api/deploy \
  -H 'Content-Type: application/json' \
  -d '{
    "data": {
      "term": "new stdout(`rho:io:stdout`) in { stdout!(42) }",
      "timestamp": 1700000000000,
      "phloPrice": 10,
      "phloLimit": 100000,
      "validAfterBlockNumber": 0,
      "shardId": "root"
    },
    "deployer": "04abc...",
    "signature": "3044...",
    "sigAlgorithm": "secp256k1"
  }'
```

#### `GET /api/deploy/{id}`

Get deploy execution details.

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `id` | path | yes | Deploy ID (hex signature) |
| `view` | query | no | `full` (default) or `summary` |

```bash
curl http://localhost:40403/api/deploy/abc123...
curl "http://localhost:40403/api/deploy/abc123...?view=summary"
```

**Full response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `deployId` | string | Deploy signature ID |
| `blockHash` | string | Containing block hash |
| `blockNumber` | int | Containing block number |
| `timestamp` | int | Block timestamp |
| `cost` | int | Phlogiston consumed |
| `errored` | bool | Whether execution failed |
| `isFinalized` | bool | Whether the containing block is finalized |
| `deployer` | string | Deployer public key (full only) |
| `term` | string | Rholang source (full only) |
| `systemDeployError` | string | System deploy error message (full only) |
| `phloPrice` | int | Phlo price (full only) |
| `phloLimit` | int | Phlo limit (full only) |
| `sigAlgorithm` | string | Signature algorithm (full only) |
| `validAfterBlockNumber` | int | Valid-after constraint (full only) |
| `transfers` | array/null | Transfer list or null on validators (full only) |

**Summary** returns only: `deployId`, `blockHash`, `blockNumber`, `timestamp`, `cost`, `errored`, `isFinalized`.

#### `GET /api/prepare-deploy`

Get the next valid `validAfterBlockNumber` for deploy construction.

**Parameters:** None

```bash
curl http://localhost:40403/api/prepare-deploy
```

```json
{"names": [], "seqNumber": 20}
```

---

### Exploratory Deploy

Execute Rholang code in read-only mode. No block is created, no phlo is consumed. **Readonly nodes only.**

#### `POST /api/explore-deploy`

Execute against the latest block state.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `term` | string | yes | Rholang source code |

```bash
curl -X POST http://localhost:40453/api/explore-deploy \
  -H 'Content-Type: application/json' \
  -d '{"term": "new ret in { ret!(42) }"}'
```

#### `POST /api/explore-deploy-by-block-hash`

Execute against a specific block's post-state.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `term` | string | yes | Rholang source code |
| `blockHash` | string | yes | Block hash to execute against |
| `usePreStateHash` | bool | no | Use pre-state instead of post-state (default: false) |

```bash
curl -X POST http://localhost:40453/api/explore-deploy-by-block-hash \
  -H 'Content-Type: application/json' \
  -d '{"term": "new ret in { ret!(42) }", "blockHash": "3bfdf56f...", "usePreStateHash": false}'
```

---

### Data Query

#### `POST /api/data-at-name-by-block-hash`

Query data at a Rholang name in a specific block's post-state.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `par` | object | yes | Rholang `Par` (protobuf JSON) identifying the channel |
| `blockHash` | string | yes | Block hash to query against |
| `usePreStateHash` | bool | no | Use pre-state instead of post-state (default: false) |

```bash
curl -X POST http://localhost:40453/api/data-at-name-by-block-hash \
  -H 'Content-Type: application/json' \
  -d '{
    "par": {"unforgeables": [{"g_private_body": {"id": "..."}}]},
    "blockHash": "3bfdf56f...",
    "usePreStateHash": false
  }'
```

---

### High-Level Query Endpoints

Convenience endpoints wrapping exploratory deploy or genesis config. Unless noted, **readonly nodes only**.

#### `GET /api/balance/{address}`

Vault balance for a wallet address.

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `address` | path | yes | REV address (Base58, starts with `1111`) — not raw hex public key |
| `block_hash` | query | no | Block hash to query against (defaults to LFB) |

```bash
curl http://localhost:40453/api/balance/11112BpS5mG8...
```

```json
{"address": "11112BpS5mG8...", "balance": 1000000, "blockNumber": 42, "blockHash": "3bfdf56f..."}
```

#### `GET /api/registry/{uri}`

Registry lookup. Returns the registered data unwrapped from the registry's `(true, data)` tuple.

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `uri` | path | yes | Registry URI (e.g. `rho:id:abc...`) |
| `block_hash` | query | no | Block hash to query against (defaults to LFB) |

```bash
curl http://localhost:40453/api/registry/rho:id:abc...
```

```json
{"uri": "rho:id:abc...", "data": [<RhoExpr>], "blockNumber": 42, "blockHash": "3bfdf56f..."}
```

#### `GET /api/validators`

Active validator set from the PoS contract (`getBonds`).

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `block_hash` | query | no | Block hash to query against (defaults to LFB) |

```bash
curl http://localhost:40453/api/validators
```

```json
{
  "validators": [
    {"publicKey": "04837a4cff...", "stake": 1000},
    {"publicKey": "04fa70d7be...", "stake": 1000},
    {"publicKey": "0457febafc...", "stake": 1000}
  ],
  "totalStake": 3000,
  "blockNumber": 30,
  "blockHash": "6bb5892d..."
}
```

#### `GET /api/validator/{pubkey}`

Status of a specific validator — whether bonded and current stake. Queries the PoS contract (`getBonds`).

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `pubkey` | path | yes | Validator public key (hex) |
| `block_hash` | query | no | Block hash to query against (defaults to LFB) |

```bash
curl http://localhost:40453/api/validator/04837a4cff...
```

Bonded validator:
```json
{"publicKey": "04837a4cff...", "isBonded": true, "stake": 1000, "blockNumber": 4, "blockHash": "7701282c..."}
```

Unknown key:
```json
{"publicKey": "aaaa", "isBonded": false, "stake": null, "blockNumber": 4, "blockHash": "7701282c..."}
```

#### `GET /api/bond-status/{pubkey}`

Check if a public key is bonded. HTTP equivalent of gRPC `bondStatus`. Uses `BlockAPI::bond_status` directly — **available on all node types** (no exploratory deploy).

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `pubkey` | path | yes | Validator public key (hex) |

```bash
curl http://localhost:40403/api/bond-status/04837a4cff...
```

```json
{"publicKey": "04837a4cff...", "isBonded": true}
```

#### `GET /api/epoch`

Current epoch info. Uses cached genesis config — no exploratory deploy required. **Available on all node types.**

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `block_hash` | query | no | Block hash to derive epoch from (defaults to LFB) |

```bash
curl http://localhost:40403/api/epoch
```

```json
{
  "currentEpoch": 3,
  "epochLength": 10,
  "quarantineLength": 10,
  "blocksUntilNextEpoch": 10,
  "lastFinalizedBlockNumber": 30,
  "blockHash": "6bb5892d..."
}
```

#### `GET /api/epoch/rewards`

Current epoch rewards from the PoS contract (`getCurrentEpochRewards`). Returns a map of validator public keys to their accumulated rewards.

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `block_hash` | query | no | Block hash to query against (defaults to LFB) |

```bash
curl http://localhost:40453/api/epoch/rewards
```

```json
{
  "rewards": {
    "ExprMap": {
      "data": {
        "04837a4cff...": {"ExprInt": {"data": 0}},
        "04fa70d7be...": {"ExprInt": {"data": 0}}
      }
    }
  },
  "blockNumber": 3,
  "blockHash": "2ee3df7f..."
}
```

#### `POST /api/estimate-cost`

Estimate phlogiston cost of Rholang code without committing. Runs exploratory deploy and returns only the cost.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `term` | string | yes | Rholang source code |

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `block_hash` | query | no | Block hash to execute against (defaults to LFB) |

```bash
curl -X POST http://localhost:40453/api/estimate-cost \
  -H 'Content-Type: application/json' \
  -d '{"term": "new ret in { ret!(42) }"}'
```

```json
{"cost": 204, "blockNumber": 3, "blockHash": "2ee3df7f..."}
```

---

### Admin API (port 40405)

#### `POST /api/propose`

Propose a new block containing pending deploys. Validator nodes only.

**Parameters:** None

```bash
curl -X POST http://localhost:40415/api/propose
```

---

## gRPC API

### DeployService (port 40401)

| Method | Request | Response | Description |
|--------|---------|----------|-------------|
| `doDeploy` | `DeployDataProto` | `DeployResponse` | Submit a signed deploy. Validates phlo price, shard ID, signature, expiration. Triggers auto-propose if enabled |
| `getBlock` | `BlockQuery` | `BlockResponse` | Get block by hash. Returns `BlockInfo` (header + deploys). Transfers enriched on readonly |
| `getBlocks` | `BlocksQuery` | `stream BlockInfoResponse` | Get recent blocks by depth. Streaming. Returns `LightBlockInfo` (headers only) |
| `showMainChain` | `BlocksQuery` | `stream BlockInfoResponse` | Walk the main chain path from tip. Streaming. Returns `LightBlockInfo` |
| `getBlocksByHeights` | `BlocksQueryByHeight` | `stream BlockInfoResponse` | Get blocks in a height range. Streaming. Clamped by `api_max_blocks_limit` |
| `lastFinalizedBlock` | `LastFinalizedBlockQuery` | `LastFinalizedBlockResponse` | Get the last finalized block. Returns full `BlockInfo` with deploys |
| `isFinalized` | `IsFinalizedQuery` | `IsFinalizedResponse` | Check if a block is finalized. Returns bool |
| `findDeploy` | `FindDeployQuery` | `FindDeployResponse` | Find block containing a deploy. Retries up to 80x with 100ms intervals while deploy propagates through DAG |
| `getDataAtName` | `DataAtNameByBlockQuery` | `RhoDataResponse` | Query data at a Rholang name in a specific block's post-state. Takes `Par` + block hash + `usePreStateHash` |
| `listenForContinuationAtName` | `ContinuationAtNameQuery` | `ContinuationAtNameResponse` | Find processes waiting to receive on given channel names. Returns matching patterns and continuation bodies |
| `exploratoryDeploy` | `ExploratoryDeployQuery` | `ExploratoryDeployResponse` | Execute Rholang read-only. No block created, no phlo consumed. Returns result `Par`s, block context, and cost. Readonly only |
| `bondStatus` | `BondStatusQuery` | `BondStatusResponse` | Check if a public key is bonded. Takes public key bytes, returns bool. HTTP: `GET /api/bond-status/{pubkey}` |
| `previewPrivateNames` | `PrivateNamePreviewQuery` | `PrivateNamePreviewResponse` | Generate unforgeable names from deployer key + timestamp. Allows clients to compute signatures over names before deploying. Max 1024 names |
| `getEventByHash` | `ReportQuery` | `EventInfoResponse` | Get full block execution trace — every COMM/produce/consume event per deploy and system deploy. Takes block hash + `forceReplay` flag. Used for debugging and auditing |
| `visualizeDag` | `VisualizeDagQuery` | `stream VisualizeBlocksResponse` | DAG visualization in DOT format. Takes depth + startBlockNumber + showJustificationLines |
| `machineVerifiableDag` | `MachineVerifyQuery` | `MachineVerifyResponse` | Machine-parseable DAG representation |
| `status` | `google.protobuf.Empty` | `StatusResponse` | Node status — version, address, peers, network, native token metadata, LFB number, isValidator, isReadOnly, isReady, epoch |

### ProposeService (port 40402)

| Method | Request | Response | Description |
|--------|---------|----------|-------------|
| `propose` | `ProposeQuery` | `ProposeResponse` | Propose a new block. `isAsync`: if true returns immediately, otherwise waits for result. Validator only |
| `proposeResult` | `ProposeResultQuery` | `ProposeResultResponse` | Get latest propose result. Blocks until current proposal completes if one is in progress |

Proto definitions: `models/src/main/protobuf/DeployServiceV1.proto`, `ProposeServiceV1.proto`, `DeployServiceCommon.proto`.

---

## Transfer Behavior

| Node type | `transfers` on deploy responses | `TransfersAvailable` WebSocket event |
|-----------|--------------------------------|--------------------------------------|
| **Readonly** | Populated array | Emitted after block report cache warming |
| **Validator** | `null` (omitted) | Not emitted |

`null` means transfers cannot be extracted on this node type (block replay requires readonly mode). An empty array `[]` means the deploy had no transfers.

---

## View Parameters

| Endpoint type | Default | Opt-in |
|---|---|---|
| Single item (`/block/{hash}`, `/last-finalized-block`, `/deploy/{id}`) | `full` | `?view=summary` |
| Lists (`/blocks`, `/blocks/{depth}`, `/blocks/{start}/{end}`) | `summary` | `?view=full` |

---

## WebSocket Events

See [websocket-events.md](websocket-events.md) for the full event catalog, payload format, and startup replay behavior.

```bash
wscat -c ws://localhost:40403/ws/events
```
