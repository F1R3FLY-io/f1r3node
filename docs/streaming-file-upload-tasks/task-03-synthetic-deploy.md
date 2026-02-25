# Task 3: Synthetic Deploy & Mempool Integration

## Architecture Reference

- [Layer 1 §1.3 — Automated Synthetic Deploy](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L160-L217)

---

## What to Implement

After a file is verified on disk (Task 2), automatically construct a Rholang deploy that registers the file on-chain, and push it into the Casper mempool. The client does NOT send a separate `doDeploy`.

### Technical Details

#### In `FileUploadAPI.scala`

1. **Cryptographic envelope** — prove the proposer validated the file:
   ```
   payload = "$fileHash:$fileSize"
   nodeSigHex = Base16.encode(casper.getValidator.get.sign(payload.getBytes))
   ```
   - This binds the file hash and size to the proposing node's identity
   - On replay (Task 6), the system process verifies this signature against `context.blockProposer`

2. **Synthetic Rholang term** construction:
   ```rholang
   new ret, file(`rho:io:file`) in {
     file!("register", "<fileHash>", <fileSize>, "<fileName>", "<nodeSigHex>", *ret)
   }
   ```

3. **Deploy construction** — `DeployDataProto` with:
   - `deployer` = from upload metadata (client's public key)
   - `timestamp` = from upload metadata
   - `sig` = re-computed over the new term + fields
   - `phloPrice`, `phloLimit`, `validAfterBlockNumber`, `shardId` = from metadata
   - `term` = the synthetic Rholang above

4. **Mempool push** — same path as `doDeploy`:
   - LMDB headroom check (1GB minimum free space via `KeyValueDeployStorage`)
   - Signature validation
   - `shardId` matching
   - `validAfterBlockNumber` range check
   - Push into `DeployBuffer` / `DeployStorage`

5. **Return to client** — `FileUploadResult`:
   - `fileHash` = computed Blake2b hash
   - `deployId` = deploy signature (used with `findDeploy` / `isFinalized`)
   - `storagePhloCost` = computed cost
   - `totalPhloCharged` = total phlo

#### Modified File: `node/src/main/.../BlockAPI.scala`

- Add `uploadFile` method that delegates to `FileUploadAPI` and calls `deploy()` internally

---

## Verification

### New Test: `SyntheticDeploySpec`

**File**: `node/src/test/scala/coop/rchain/node/api/SyntheticDeploySpec.scala`

| Test Case | Assertion |
|-----------|-----------|
| Construct synthetic term | Parse back with Rholang normalizer → all fields (`fileHash`, `fileSize`, `fileName`, `nodeSigHex`) correct |
| Cryptographic envelope | `nodeSigHex` is valid Ed25519 signature over `"$hash:$size"` (verify with BouncyCastle) |
| Mempool admission (happy path) | Mock `DeployBuffer.addDeploy` called with correct `DeployDataProto` |
| Mempool rejection: LMDB headroom < 1GB | Error returned, no deploy in buffer |
| Mempool rejection: `shardId` mismatch | Error returned |
| `FileUploadResult.deployId` | Equals the deploy signature bytes |
| `FileUploadResult.storagePhloCost` | Matches `fileSize × phloPerStorageByte` |

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

- [ ] Implement cryptographic envelope (`payload` + node signature)
- [ ] Implement synthetic Rholang term construction
- [ ] Implement `DeployDataProto` construction (with re-computed sig)
- [ ] Integrate with `DeployBuffer` / `DeployStorage` (same path as `doDeploy`)
- [ ] Add `uploadFile` method to `BlockAPI.scala`
- [ ] Return `FileUploadResult` with `deployId`, cost fields
- [ ] Unit tests for term construction and signature
- [ ] Unit tests for mempool admission edge cases
