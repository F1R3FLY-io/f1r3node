# Task 3: Synthetic Deploy & Mempool Integration

## Architecture Reference

- [Layer 1 §1.3 — Automated Synthetic Deploy](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L160-L217)

---

## What to Implement

After a file is verified on disk (Task 2), automatically construct a Rholang deploy that registers the file on-chain, and push it into the Casper mempool. The client does NOT send a separate `doDeploy`.

### Technical Details

#### In `FileUploadAPI.scala`

1. **`metadataToDeployProto`** — maps `FileUploadMetadata` fields to `DeployDataProto`:
   - The **client** constructs the Rholang term and signs the full `DeployData` (same as `doDeploy`)
   - The client sends `term` + `sig` in the upload metadata
   - The server maps metadata → `DeployDataProto` for signature validation

2. **Synthetic Rholang term** (constructed by client):
   ```rholang
   new ret, file(`rho:io:file`) in {
     file!("register", "<fileHash>", <fileSize>, "<fileName>", *ret)
   }
   ```

3. **Deploy validation & mempool push** — in `DeployGrpcServiceV1.uploadFile`:
   - `DeployData.from(proto)` validates the client's signature (same path as `doDeploy`)
   - `BlockAPI.deploy(signed, ...)` pushes to the mempool
   - On failure: saved file is cleaned up (no orphans)

4. **Mempool admission** — same checks as `doDeploy`:
   - LMDB headroom check (1GB minimum free space via `KeyValueDeployStorage`)
   - Signature validation (client's key)
   - `shardId` matching
   - `validAfterBlockNumber` range check
   - Push into `DeployBuffer` / `DeployStorage`

5. **Return to client** — `FileUploadResult`:
   - `fileHash` = computed Blake2b hash
   - `deployId` = deploy signature hex (used with `findDeploy` / `isFinalized`)
   - `storagePhloCost` = computed cost
   - `totalPhloCharged` = total phlo

#### Modified File: `models/.../DeployServiceV1.proto`

- Added `string term = 12` to `FileUploadMetadata`
- Renamed `expectedFileHash` → `fileHash`

---

## Verification

### New Test: `SyntheticDeploySpec`

**File**: `node/src/test/scala/coop/rchain/node/api/SyntheticDeploySpec.scala`

| Test Case | Assertion |
|-----------|-----------|
| Map metadata → DeployDataProto | All 9 fields correctly mapped |
| Signature round-trip | `DeployData.from(proto)` succeeds with valid client signature |
| Tampered signature | `DeployData.from(proto)` returns `Left` on tampered sig |
| Empty term rejection | Upload rejected before file write |
| Mempool rejection: `shardId` mismatch | Error returned |
| `FileUploadResult.deployId` | Empty from API (filled by gRPC layer after sig validation) |
| `FileUploadResult.storagePhloCost` | Matches `fileSize` |
| Dedup path | `deployProto` still returned when file already exists |

```bash
sbt 'node/testOnly coop.rchain.node.api.SyntheticDeploySpec'
```

### Existing Tests to Verify (regression)

```bash
# BlockAPI changes must not break existing deploy submission tests
sbt 'casper/testOnly coop.rchain.casper.api.*'
sbt 'casper/testOnly coop.rchain.casper.batch1.*'
```

### Manual Verification

After uploading a file via gRPC:
```bash
# Use the returned deployId to track via existing RPCs
grpcurl -plaintext -d '{"deployId":"<deployId>"}' localhost:40401 ...DeployService/findDeploy
# Verify: returns LightBlockInfo once block is proposed
```

---

## Subtasks

- [x] Add `string term = 12` to `FileUploadMetadata` proto
- [x] Rename `expectedFileHash` → `fileHash` in proto
- [x] Implement `metadataToDeployProto` (maps metadata → `DeployDataProto`)
- [x] Implement `computeStorageCost` with overflow protection
- [x] Validate client signature via `DeployData.from(proto)` in gRPC layer
- [x] Push signed deploy to mempool via `BlockAPI.deploy`
- [x] Clean up saved file on deploy failure
- [x] Return `FileUploadResult` with `deployId`, cost fields
- [x] Unit tests for proto mapping, sig validation, and cost calculation
- [x] Unit tests for empty term rejection, shard mismatch, dedup path
