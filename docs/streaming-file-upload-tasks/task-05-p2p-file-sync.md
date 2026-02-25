# Task 5: P2P File Sync Protocol

## Architecture Reference

- [Layer 2 — Validator File Sync & Integrity](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L233-L323)

---

## What to Implement

When a validator receives a block containing file-registration deploys, it must check whether the referenced files exist locally. If not, it fetches them from the block proposer via a streaming P2P transfer, verifies integrity, and only then proceeds with block validation.

### Technical Details

#### Block Scan (in `MultiParentCasperImpl.scala`)

1. When a block arrives via P2P:
   - Scan all deploys in the block for `rho:io:file` "register" terms
   - Extract `fileHash` from each file-registration deploy
   - Check if `<data-dir>/file-replication/<hash>` exists locally
   - If all files present → proceed to normal Casper validation
   - If any missing → trigger file fetch from block proposer

#### P2P File Transfer (in `TransportLayer.scala` / `CommMessages.scala`)

2. **Request**: Send `FileRequest(fileHash)` to the block proposer peer
3. **Response**: The proposer streams back `FilePacket` messages:
   - First `FilePacket`: `fileSize` (total bytes)
   - Subsequent `FilePacket`s: `data` chunks (4MB each)

4. **Receiving validator**:
   - Opens `.tmp` file
   - Streams chunks to disk via direct I/O (same as Layer 1)
   - Tracks `bytesReceived`; **aborts immediately** if `bytesReceived > expectedFileSize` (prevents remote disk exhaustion)
   - Computes Blake2b-256 incrementally
   - At EOF: verifies hash matches `fileHash` from the block's deploy
   - On match: atomic rename `.tmp` → `<hash>`, proceed with block validation
   - On mismatch: delete `.tmp`, **reject the block**, negative `PeerScore`

5. **Serving side** (on the proposer node):
   - Handle incoming `FileRequest` by streaming the file from `file-replication/` directory
   - Chunk size: 4MB per `FilePacket`

---

## Verification

### Existing Test to Extend: `FileReplicationSpec`

**File**: [FileReplicationSpec.scala](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/casper/src/test/scala/coop/rchain/casper/engine/FileReplicationSpec.scala)

This test already exists and validates basic file replication between two mock nodes. **Extend it** with new test cases:

| Test Case (add to existing spec) | Assertion |
|-----------|-----------|
| _(existing)_ Replicate file from peer | File content + size match |
| **(new)** Block scan: extract file hashes | File-registration deploys → correct hash list |
| **(new)** Block scan: no file deploys | Returns empty list, no fetch triggered |
| **(new)** Hash mismatch on transfer | `.tmp` deleted, block rejected |
| **(new)** `bytesReceived > fileSize` | Connection dropped, `.tmp` deleted |
| **(new)** `PeerScore` penalty | Negative score assigned on corruption |

```bash
sbt 'casper/testOnly coop.rchain.casper.engine.FileReplicationSpec'
```

### New Integration Test: `FileAvailabilityValidationSpec`

**File**: `casper/src/test/scala/coop/rchain/casper/engine/FileAvailabilityValidationSpec.scala`

Uses `TestNode.networkEff` (same as `ExploratoryDeployAPITest`). Two nodes: upload on A, verify B fetches file before validating A's block.

```bash
sbt 'casper/testOnly coop.rchain.casper.engine.FileAvailabilityValidationSpec'
```

### Existing Tests to Verify (regression)

```bash
sbt 'casper/testOnly coop.rchain.casper.engine.*'
sbt 'casper/testOnly coop.rchain.casper.addblock.*'
```

---

## Subtasks

- [x] Implement block deploy scanning (extract file hashes from register terms)
- [x] Implement local file existence check
- [x] Add `FileRequest` / `FilePacket` handling to `TransportLayer.scala`
- [x] Implement file serving (proposer side): chunk and stream file
- [x] Implement file receiving: `.tmp` write, streaming hash, size guard
- [x] Implement hash verification at EOF (Blake2b-256, matching upload API)
- [x] Implement atomic rename on success
- [x] Implement cleanup + block rejection on failure
- [ ] Implement `PeerScore` penalty on hash mismatch
  > **Deferred**: No `PeerScore` abstraction exists in the codebase.
  > A TODO comment marks the exact location in `FileRequester.finalizeDownload`
  > where a negative score should be applied once the mechanism is added.
- [x] Wire scan + fetch into `MultiParentCasperImpl.addBlock` pipeline
- [x] Add `fileAvailability` check to `Validate.scala`
- [x] Unit tests for block scanning (`FileAvailabilitySpec`: 10 tests)
- [x] Unit tests for transfer protocol: success, hash mismatch, availability, size overflow
- [ ] Integration test: two-node file replication (`FileAvailabilityValidationSpec` with `TestNode.networkEff`)
  > **Deferred**: Requires significant test infrastructure wiring.
