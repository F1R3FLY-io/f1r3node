# Task 2: Streaming File Ingestion

## Architecture Reference

- [Layer 1 §1.1 — Chunk Flow](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L96-L117)
- [Layer 1 §1.2 — Interruption and Atomic Guarantee](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L119-L124)
- [Layer 1 §1.4 — Module Changes](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L219-L230)

---

## What to Implement

The core upload pipeline: accept a gRPC stream of `FileUploadChunk` messages, write bytes directly to disk (bypassing JVM heap), compute a streaming Blake2b-256 hash, and perform an atomic rename on completion. Handle deduplication and interruption cleanup.

### Technical Details

#### New File: `node/src/main/.../FileUploadAPI.scala`

1. **Metadata validation** (first chunk):
   - Verify `sig` over metadata fields using `deployer` public key
   - Validate `shardId` matches the node's shard
   - Validate `validAfterBlockNumber` range
   - Pre-validate phlo limits: `metadata.phloLimit >= BASE_REGISTER_PHLO + metadata.fileSize * phloPerStorageByte` (see Task 10)

2. **Dedup check**:
   - If `expectedFileHash` is provided and file `<data-dir>/file-replication/<hash>` exists → skip upload, jump to synthetic deploy (Task 3)

3. **Streaming disk write**:
   - Open temp file: `<data-dir>/file-replication/<txn-id>.tmp`
   - Use `java.nio.channels.FileChannel` with direct `ByteBuffer` allocation (off-heap, zero-copy)
   - Track `bytesReceived`; abort immediately if `bytesReceived > metadata.fileSize`
   - Feed every chunk into `org.bouncycastle.crypto.digests.Blake2bDigest` (streaming, 256-bit output)

4. **Finalization**:
   - Verify `bytesReceived == metadata.fileSize` at EOF
   - Finalize hash → `computedHash`
   - If `expectedFileHash` provided: verify `computedHash == expectedFileHash`
   - Atomic rename: `.tmp` → `<hash>`
   - If hash collision (file already exists): delete `.tmp`, reuse existing file

5. **Interruption cleanup** (`finally` block):
   - On gRPC stream error, client disconnect, or timeout: delete `.tmp` immediately
   - No deploy is created; no trace on disk

6. **Metadata persistence**:
   - Write `<hash>.meta.json` with `fileName`, `fileSize`, `uploaderPubKey`, `timestamp`
   - New file: `FileMetadata.scala` — data class + JSON serialization

#### Modified File: `node/src/main/.../DeployGrpcServiceV1.scala`

- Wire `uploadFile` RPC → calls `FileUploadAPI` using Monix `Observable.foldLeftF`

---

## Verification

### New Test: `FileUploadAPISpec`

**File**: `node/src/test/scala/coop/rchain/node/api/FileUploadAPISpec.scala`

ScalaTest FlatSpec + Monix Task (same pattern as existing `ExploratoryDeployAPITest`).

| Test Case | Assertion |
|-----------|-----------|
| Stream N chunks (1MB file) | Final file content matches input bytes |
| Stream N chunks (1MB file) | Computed Blake2b-256 hash matches `org.bouncycastle` reference |
| Interrupt stream midway (simulate gRPC cancel) | `.tmp` file deleted, no orphans in `file-replication/` |
| `bytesReceived > metadata.fileSize` | Immediate abort, `.tmp` deleted |
| `computedHash != expectedFileHash` | Rejection error, `.tmp` deleted |
| Upload duplicate (same hash exists on disk) | Second upload skips disk write, returns existing hash |
| `FileMetadata` JSON | Round-trip serialization matches original |

```bash
sbt 'node/testOnly coop.rchain.node.api.FileUploadAPISpec'
```

### Existing Tests to Verify (regression)

```bash
# Ensure existing gRPC/API tests still pass after DeployGrpcServiceV1 changes
sbt 'casper/testOnly coop.rchain.casper.api.*'
```

### Manual Verification

```bash
# Stream a 100MB file via grpcurl (after node is running)
grpcurl -plaintext -d @ localhost:40401 coop.rchain.casper.protocol.DeployService/uploadFile < test_upload.bin
# Verify: response contains fileHash, file exists at <data-dir>/file-replication/<hash>
ls -la <data-dir>/file-replication/
```

---

## Subtasks

- [ ] Create `FileUploadAPI.scala` skeleton with trait/interface
- [ ] Implement metadata validation (signature, shardId, block number range)
- [ ] Implement deduplication check (`expectedFileHash` → disk lookup)
- [ ] Implement streaming disk write with `FileChannel` + direct `ByteBuffer`
- [ ] Implement streaming Blake2b-256 hashing (BouncyCastle)
- [ ] Implement `bytesReceived > fileSize` guard (abort on overflow)
- [ ] Implement EOF verification (`bytesReceived == metadata.fileSize`)
- [ ] Implement hash verification (`computedHash == expectedFileHash`)
- [ ] Implement atomic rename (`.tmp` → `<hash>`)
- [ ] Implement hash collision handling (delete `.tmp`, reuse existing)
- [ ] Implement interruption cleanup (`finally` / error handler)
- [ ] Create `FileMetadata.scala` data class + JSON serde
- [ ] Write `<hash>.meta.json` on successful upload
- [ ] Wire `uploadFile` in `DeployGrpcServiceV1.scala` (Monix `Observable`)
- [ ] Unit tests for all above cases
