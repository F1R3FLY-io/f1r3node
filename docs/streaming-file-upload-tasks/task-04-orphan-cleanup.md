# Task 4: Orphan File Cleanup

## Architecture Reference

- [Layer 1 §1.2 — Orphan File Cleanup](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L126-L158)

---

## What to Implement

When a file-registration synthetic deploy is expelled from the mempool (expiration or eviction), the corresponding physical file on disk must be cleaned up to prevent storage leaks.

### Technical Details

#### Hook location: `BlockCreator.prepareUserDeploys` (expiration path only)

The `BlockCreator` already reads all pending deploys and partitions them into `valid` and `allExpiredDeploys`. After removing expired deploys from storage, we call `OrphanFileCleanup.cleanupOrphanedFiles` using these two pre-computed sets — **zero extra `readAll` calls**.

> **Design decision:** Finalized deploys (included in a block) are NOT orphans — the on-chain `FileRegistry` contract handles their lifecycle. Cleanup only runs on the **expiration** path.

1. **Detect file-registration deploys**:
   - Check if the removed deploy's `term` contains `rho:io:file`, `"register"`, and a 64-char hex hash
   - All three conditions must match to avoid false positives

2. **Cross-reference check**:
   - Before deleting the physical file, check if any deploy in the `valid` set (remaining pool) references the same `fileHash`
   - This prevents deleting a deduplicated file that another pending deploy still needs

3. **Physical cleanup**:
   - Delete `<data-dir>/file-replication/<hash>`
   - Delete `<hash>.meta.json`
   - Only if no other pending deploy references the same hash
   - IO wrapped in `Sync[F].delay`, safe no-op if files don't exist

4. **Configuration**:
   - `fileReplicationDir: Option[Path]` is a field on `CasperShardConf` (set via `CasperLaunch.of` → `Setup.scala`)
   - When `None`, cleanup is skipped (backward-compatible default)

---

## Verification

### New Test: `OrphanFileCleanupSpec`

**File**: `casper/src/test/scala/coop/rchain/casper/util/OrphanFileCleanupSpec.scala`

Uses temp directories + direct method calls (purely functional, no mocking infrastructure needed).

| Test Case | Assertion |
|-----------|-----------|
| File deploy term detected | `isFileRegistrationDeploy` returns `true` for `rho:io:file` + `"register"` pattern |
| Non-file deploy term | Returns `false` for regular Rholang terms |
| False-positive term (register + hash but no `rho:io:file`) | Returns `false` |
| Hash extraction | `extractFileHash` returns correct 64-char hex from term |
| Hash extraction on non-file term | Returns `None` |
| File deploy expired | Physical file + `.meta.json` deleted from `file-replication/` |
| Two deploys reference same hash → expire one | File NOT deleted (cross-reference check) |
| Expire last deploy for hash | File IS deleted |
| Non-file deploy expired | No filesystem side-effects |
| Files don't exist on disk | `deleteFileAndMeta` is safe no-op |

```bash
sbt 'casper/testOnly coop.rchain.casper.util.OrphanFileCleanupSpec'
```

### Existing Tests to Verify (regression)

```bash
sbt 'casper/testOnly coop.rchain.casper.blocks.proposer.BlockCreatorSpec'
sbt 'casper/testOnly coop.rchain.casper.batch1.*'
sbt 'casper/testOnly coop.rchain.casper.batch2.*'
```

---

## Subtasks

- [x] Identify the deploy removal hook (`BlockCreator.prepareUserDeploys`)
- [x] Implement file-registration deploy detection (term pattern matching with `rho:io:file`)
- [x] Implement cross-reference check (reuse `valid` set from `prepareUserDeploys`)
- [x] Implement physical file + meta.json deletion (IO wrapped in `Sync[F].delay`)
- [x] Wire cleanup logic into deploy expiration path
- [x] Add `fileReplicationDir` to `CasperShardConf` (threaded via `CasperLaunch` → `Setup`)
- [x] Unit tests for all edge cases (10 tests, all passing)
- [x] Regression tests verified (batch1, batch2, BlockCreatorSpec — 48 tests, all passing)
