# Task 14: P2P File Replication Bug Fix

## Problem

Cross-validator file downloads were failing: files uploaded to one validator were not downloadable from another. The `test_download_from_validator` integration test consistently failed.

---

## Root Cause Analysis

Debug logging revealed **three bugs** in the P2P file replication pipeline:

### Bug 1: Fire-and-Forget DA Callback

**Files**: `CasperLaunch.scala`, `GenesisCeremonyMaster.scala`

The `daFetchFiles` callback used `Concurrent[F].start(...)` to spawn a background fiber and immediately returned `List.empty[String]`. This caused `MultiParentCasperImpl` to see `stillMissing=0` and accept the block before the file actually arrived.

**Fix**: Replaced fire-and-forget with synchronous broadcast + `fileRequester.awaitFiles(hashes, timeout)`. The callback now:
1. Broadcasts `FileRequest` to all connected peers
2. Polls until files arrive (via `FilePacket` handler) or timeout expires
3. Returns the list of truly still-missing files

### Bug 2: Single-Peer Download Routing

**File**: `FileRequester.scala` — `requestFiles` method

`requestFiles` called `handleHasFile(peer, HasFile(hash))` for each peer. This method was designed for incoming P2P messages (indicating a peer *has* the file). When called locally for all peers:
- The first peer (bootstrap) triggered `startDownload` → sent `FileRequest` to bootstrap
- Subsequent peers were skipped (`isDownloading=true`)
- **Bootstrap didn't have the file** → the download stalled silently

**Fix**: Replaced with direct `FileRequest` broadcast to ALL peers. Each peer's `handleFileRequest` only responds if it has the file. Download state is registered once (idempotent), and `FileRequest` is sent to every peer.

### Bug 3: Race Condition in `handleFilePacket`

**File**: `FileRequester.scala` — `handleFilePacket` method

With the broadcast fix, multiple peers respond with `FilePacket` at offset=0. The non-atomic `downloads.get` + `downloads.update` allowed two threads to concurrently:
1. Both read `bytesReceived=0` (offset validation passes)
2. Both append data to the `.part` file (duplicate data)
3. Both finalize → hash verification fails (corrupted content)

**Fix**: Replaced with `downloads.modify` (atomic compare-and-swap). Only one thread claims each offset — the second thread sees the advanced `bytesReceived` and its modify returns `None`, cleanly rejecting the duplicate.

---

## Files Modified

| File | Change Type |
|------|-------------|
| `casper/.../CasperLaunch.scala` | DA callback: synchronous broadcast + `awaitFiles` |
| `casper/.../GenesisCeremonyMaster.scala` | DA callback: same fix as CasperLaunch |
| `casper/.../FileRequester.scala` | `requestFiles`: broadcast to all peers; `handleFilePacket`: atomic `Ref.modify` |

---

## Verification

All 15 file upload/download integration tests pass:

```
$ sc test test_file_upload --verbose
15 passed, 34 deselected in 427.51s (0:07:07)
All tests passed!
```

Key tests that were previously failing:
- `test_download_from_validator` — cross-validator download
- `test_download_from_all_validators` — multi-validator download
- `test_download_with_offset` — offset-based resume download

---

## Subtasks

- [x] Add debug logging to trace P2P file transfer flow
- [x] Identify root cause via log analysis
- [x] Fix DA callback: `awaitFiles` instead of fire-and-forget
- [x] Fix `requestFiles`: broadcast `FileRequest` to all peers
- [x] Fix `handleFilePacket`: atomic `Ref.modify` for race condition
- [x] Verify all 15 integration tests pass
- [x] Remove debug logging
`, "IsArtifact": false}
