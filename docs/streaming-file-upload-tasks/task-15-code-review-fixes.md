# Task 15: Code Review Fixes

## Problem

Comprehensive code review of the `feature/streaming-file-upload-registry` branch identified 18 issues across correctness, security, performance, and maintainability.

---

## Changes

### Critical (4)

| # | Issue | File | Fix |
|---|-------|------|-----|
| 1 | TOCTOU race in `requestFiles` | `FileRequester.scala` | `downloads.get` + `.update` → atomic `downloads.modify` |
| 2 | Double `toByteArray` (2× 4MB alloc) | `FileRequester.scala` | Extract once to local `val data` |
| 3 | Blocking `Files.exists` outside effect | `FileRequester.scala` | Wrapped in `Sync[F].delay` |
| 4 | Blocking `Files.deleteIfExists` | `DeployGrpcServiceV1.scala` | Wrapped in `Task.delay`, `.map` → `.flatMap` |

### Medium (6)

| # | Issue | File | Fix |
|---|-------|------|-----|
| 6 | `_sysAuthTokenCh` unbounded growth | `FileRegistry.rho` | One-shot `_tokenInitFlag` pattern |
| 7 | Global `@{["fileState", hash]}` channel | `FileRegistry.rho` | Unforgeable `fileStateCh` per handle; `ownerAuthKey` refactored to use `getInfo` |
| 8 | Missing path traversal guard | `FileDownloadAPI.scala` | `normalize().startsWith()` check after resolve |
| 9 | Mutable `var bytesStreamed` | `FileDownloadAPI.scala` | Replaced with `scan` accumulator |
| 10 | No `fileSize > 0` validation | `FileUploadAPI.scala` | Added `fileSize <= 0` + `fileHash` regex checks |
| 11 | `MissingFileData` slashability undocumented | `BlockStatus.scala` | Documented as intentionally not slashable |

### Minor (3)

| # | Issue | File | Fix |
|---|-------|------|-----|
| 12 | `ByteBuffer.allocate` per chunk | `FileDownloadAPI.scala` | Allocate once, reuse via `clear()` |
| 13 | `Files.size` outside effect | `FileRequester.scala` | Moved into `Sync[F].delay` using `channel.size()` |
| 15 | Dead `Validate.fileAvailability` | `Validate.scala` | Removed (17 lines) |

---

## Deferred

| # | Issue | Reason |
|---|-------|--------|
| 5 | `ipSemaphores` unbounded map | Needs TTL cache library decision (Caffeine/Guava) |
| 14 | `FileMetadata.fromJson` throws | Low risk — only used in catch-all contexts |
| 16 | Duplicate `toHex` methods | Cosmetic, cross-module dependency |
| 17 | Config units comments | Already documented in `defaults.conf` |
| 18 | `sender.bytes` as public key | Verified: always 65-byte uncompressed secp256k1 |

---

## Files Modified

| File | Change |
|------|--------|
| `casper/.../FileRequester.scala` | Fixes #1, #2, #3, #13 |
| `casper/.../BlockStatus.scala` | Fix #11 |
| `casper/.../Validate.scala` | Fix #15 |
| `casper/src/main/resources/FileRegistry.rho` | Fixes #6, #7 |
| `node/.../DeployGrpcServiceV1.scala` | Fix #4 |
| `node/.../FileDownloadAPI.scala` | Fixes #8, #9, #12 |
| `node/.../FileUploadAPI.scala` | Fix #10 |
| `node/.../FileUploadAPISpec.scala` | Updated test for new hash format validation |

---

## Verification

```
sbt compile                                          → ✅ clean (0 errors)
sbt casper/testOnly ...FileReplicationSpec            → ✅ 4/4 passed
sbt node/testOnly ...FileUploadAPISpec                → ✅ 10/10 passed
sbt node/testOnly ...FileDownloadAPISpec              → ✅ all passed
```

---

## Subtasks

- [x] Fix TOCTOU race in `FileRequester.requestFiles`
- [x] Fix double `toByteArray` allocation in `handleFilePacket`
- [x] Wrap blocking I/O in `FileRequester.handleFileRequest`
- [x] Wrap blocking `Files.deleteIfExists` in `DeployGrpcServiceV1`
- [x] Fix `_sysAuthTokenCh` unbounded growth in `FileRegistry.rho`
- [x] Replace global `@{["fileState", hash]}` with unforgeable channels
- [x] Add path traversal defense in `FileDownloadAPI`
- [x] Replace mutable `var bytesStreamed` with `scan`
- [x] Add `fileSize` and `fileHash` validation in `FileUploadAPI`
- [x] Document `MissingFileData` slashability decision
- [x] Reuse `ByteBuffer` in download stream
- [x] Move `Files.size` inside `Sync[F].delay`
- [x] Remove dead `Validate.fileAvailability` method
- [x] Update test for new hash format validation
