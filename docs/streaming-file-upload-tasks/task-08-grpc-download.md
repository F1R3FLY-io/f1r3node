# Task 8: gRPC Download (Observer-Only)

## Architecture Reference

- [Layer 3 §3.4 — gRPC Download](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L470-L566)

---

## What to Implement

A server-streaming `downloadFile` gRPC endpoint that streams file bytes to clients. Only available on read-only (observer) nodes. Supports resume via byte offset.

### Technical Details

#### Modified File: `node/src/main/.../DeployGrpcServiceV1.scala`

1. Wire `downloadFile` RPC → delegates to `BlockAPI.downloadFile`
2. Same wiring pattern as `exploratoryDeploy`

#### Modified File: `node/src/main/.../BlockAPI.scala`

1. **Observer gate**: `casper.getValidator.map(_.isEmpty)` — if node has a validator key, reject with `"File download can only be executed on a read-only RNode."`

2. **Path traversal prevention**: validate `request.fileHash` matches `^[a-f0-9]{64}$` — reject with `INVALID_ARGUMENT` if not a valid hash format

3. **File lookup**: resolve `fileReplicationDir / fileHash` — return `NOT_FOUND` if file doesn't exist

4. **Streaming response**:
   - First message: `FileDownloadChunk(metadata: FileDownloadMetadata(fileHash, fileSize))`
   - Subsequent messages: `FileDownloadChunk(data: bytes)` — 4MB chunks
   - Seek to `offset` before streaming (for resume support)
   - Use `java.nio.channels.FileChannel` with direct `ByteBuffer` for off-heap I/O

5. **Rate limiting**:
   - `max-concurrent-downloads-per-ip` (default: 4) — tracked via `ConcurrentHashMap[InetAddress, Semaphore]`
   - Reject with `RESOURCE_EXHAUSTED` when limit reached

---

## Verification

### New Test: `DownloadFileAPITest`

**File**: `casper/src/test/scala/coop/rchain/casper/api/DownloadFileAPITest.scala`

Same pattern as existing [ExploratoryDeployAPITest.scala](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/casper/src/test/scala/coop/rchain/casper/api/ExploratoryDeployAPITest.scala) — uses `TestNode.networkEff` with `withReadOnlySize = 1`.

| Test Case | Assertion |
|-----------|-----------|
| Download on observer node | Full file bytes match original |
| Download on bonded validator | Returns `"File download can only be executed on a read-only RNode."` |
| `fileHash = "../../etc/passwd"` | Returns `INVALID_ARGUMENT` |
| `fileHash = "abc123"` (too short) | Returns `INVALID_ARGUMENT` |
| Valid hash, file not on disk | Returns `NOT_FOUND` |
| Download with `offset > 0` | Stream starts at correct byte position |
| 5 concurrent downloads from same IP (limit=4) | 5th rejected with `RESOURCE_EXHAUSTED` |
| Concurrent downloads from different IPs | All accepted |

```bash
sbt 'casper/testOnly coop.rchain.casper.api.DownloadFileAPITest'
```

### Existing Tests to Verify (regression)

```bash
# Existing observer-related tests must still pass
sbt 'casper/testOnly coop.rchain.casper.api.ExploratoryDeployAPITest'
sbt 'casper/testOnly coop.rchain.casper.api.*'
```

### Manual Verification

```bash
# On a running 3-node network:
# Upload file to Validator A, then download from Observer C
grpcurl -plaintext -d '{"fileHash":"<hash>"}' observer:40401 ...DeployService/downloadFile
# Verify: streamed bytes match original file (sha256sum)

# Attempt download on Validator A → expect rejection
grpcurl -plaintext -d '{"fileHash":"<hash>"}' validatorA:40401 ...DeployService/downloadFile
```

---

## Subtasks

- [ ] Wire `downloadFile` RPC in `DeployGrpcServiceV1.scala`
- [ ] Implement `BlockAPI.downloadFile` with observer gate
- [ ] Implement `fileHash` format validation (regex)
- [ ] Implement file-to-stream logic (`FileChannel` + `ByteBuffer`)
- [ ] Implement `FileDownloadMetadata` as first message
- [ ] Implement 4MB chunking for data messages
- [ ] Implement `offset` seek for resume support
- [ ] Implement per-IP rate limiting with `ConcurrentHashMap` + `Semaphore`
- [ ] Unit tests for observer gate
- [ ] Unit tests for path traversal prevention
- [ ] Unit tests for streaming + resume
- [ ] Unit tests for rate limiting
