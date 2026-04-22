# Deployment Workflow

How to deploy Rholang contracts to the F1R3FLY Rust shard.

## Deploy Lifecycle

1. **Write** -- create a `.rho` file
2. **Deploy** -- submit the contract with phlogiston limit and deployer key
3. **Propose** -- validator includes the deploy in a block
4. **Finalize** -- block reaches finality through consensus
5. **Query** -- read results via exploratory deploy or data-at-name

## CLI Commands

### Deploy a Contract

```bash
cargo run --bin rholang-cli -- deploy \
  -f contract.rho \
  --private-key $PRIVATE_KEY \
  --phlo-limit 100000 \
  --phlo-price 1
```

Parameters:
- `-f` / `--file`: path to `.rho` file
- `--private-key`: deployer's private key (hex)
- `--phlo-limit`: maximum phlogiston to spend (default varies)
- `--phlo-price`: price per phlogiston unit (typically 1)

### Propose a Block

Validators include pending deploys into a new block:

```bash
cargo run --bin rholang-cli -- propose \
  --private-key $VALIDATOR_KEY
```

In auto-propose mode (default for single-validator networks), blocks are proposed automatically.

### Check Finalization

```bash
cargo run --bin rholang-cli -- is-finalized -b $BLOCK_HASH
```

Returns `true` once the block has been finalized by the consensus protocol.

### Exploratory Deploy

Execute a read-only contract without creating a block. Useful for querying state.

```bash
cargo run --bin rholang-cli -- exploratory-deploy \
  -f query.rho
```

The result includes the phlogiston cost and any data sent to `rho:io:stdout`.

## HTTP API

The node exposes an HTTP API (default port 40403).

### Deploy

```bash
curl -X POST http://localhost:40403/api/deploy \
  -H 'Content-Type: application/json' \
  -d '{
    "term": "new stdout(`rho:io:stdout`) in { stdout!(\"hello\") }",
    "phloLimit": 100000,
    "phloPrice": 1,
    "validAfterBlockNumber": -1
  }'
```

### Get Deploy Status

```bash
curl http://localhost:40403/api/deploy/$DEPLOY_ID
curl http://localhost:40403/api/deploy/$DEPLOY_ID?view=summary
```

**Views:**
- **`full`** (default): all fields — `deployId`, `blockHash`, `blockNumber`, `timestamp`, `cost`, `errored`, `isFinalized`, `deployer`, `term`, `systemDeployError`, `phloPrice`, `phloLimit`, `sigAlgorithm`, `validAfterBlockNumber`, `transfers`
- **`summary`**: core fields only — `deployId`, `blockHash`, `blockNumber`, `timestamp`, `cost`, `errored`, `isFinalized`. For lightweight polling.

**Transfers:** The `transfers` field is `null` on validator nodes (block replay unavailable) and a populated array on readonly nodes. `null` means transfers can't be extracted on this node type — query a readonly node for transfer details.

### Exploratory Deploy

```bash
curl -X POST http://localhost:40403/api/explore-deploy \
  -H 'Content-Type: application/json' \
  -d '{"term": "new ret(`rho:io:stdout`) in { ret!(42) }"}'
```

Response includes the phlogiston cost.

### Get Data at Name by Block Hash

```bash
curl -X POST http://localhost:40403/api/data-at-name-by-block-hash \
  -H 'Content-Type: application/json' \
  -d '{"par": {"unforgeables": [{"g_private_body": {"id": "..."}}]}, "blockHash": "abc123...", "usePreStateHash": false}'
```

## gRPC API

The node exposes gRPC services for programmatic access:

- `DeployService.doDeploy` -- submit a deploy
- `DeployService.getBlock` -- get block by hash
- `DeployService.findDeploy` -- find deploy by ID
- `ProposeService.propose` -- propose a block
- `DeployService.getDataAtName` -- query data on a channel

Python client (`pyf1r3fly`) wraps these for integration testing.

## WebSocket Events

The node streams block lifecycle events via WebSocket:

```
ws://localhost:40403/ws/events
```

Event types:
- `block-created` -- new block proposed
- `block-added` -- block added to DAG
- `block-finalised` -- block reached finality
- Genesis ceremony events
- Node lifecycle events

Events published during startup are buffered and replayed when clients connect.

## Deploy Result

After a deploy is included in a finalized block, the result contains:

| Field | Description |
|-------|-------------|
| `cost` | Phlogiston consumed |
| `errored` | Whether the deploy produced an error |
| `systemDeployError` | System-level error message (if any) |
| `blockNumber` | Block containing the deploy |

## Common Patterns

### Deploy and Wait for Result

```bash
# 1. Deploy
DEPLOY_ID=$(cargo run --bin rholang-cli -- deploy -f contract.rho --private-key $KEY)

# 2. Wait for block (if not auto-propose)
cargo run --bin rholang-cli -- propose --private-key $VALIDATOR_KEY

# 3. Check result
curl http://localhost:40403/api/deploy/$DEPLOY_ID?view=detail
```

### Query State After Deploy

Use exploratory deploy to read state without creating a new block:

```rho
// query.rho -- read the registry entry set by a previous deploy
new lookup(`rho:registry:lookup`), stdout(`rho:io:stdout`) in {
  new ret in {
    lookup!(`rho:id:my_service_uri`, *ret) |
    for (service <- ret) {
      new result in {
        service!({"action": "status"}, *result) |
        for (@status <- result) {
          stdout!(status)
        }
      }
    }
  }
}
```

### Generate Keys

```bash
cargo run --bin rholang-cli -- generate-key-pair --save
```

This creates a keypair file that can be used for deploys.

## Phlogiston Tips

- Start with `--phlo-limit 100000` for simple contracts
- Use `1000000` for complex contracts (registry operations, vault transfers)
- Check the deploy result's `cost` field to see actual consumption
- If you get `OutOfPhlogistonsError`, increase the limit
- See [Cost Model](13-cost-model.md) for detailed cost tables
