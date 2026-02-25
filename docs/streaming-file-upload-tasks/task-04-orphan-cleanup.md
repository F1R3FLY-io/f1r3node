# Task 4: Orphan File Cleanup

## Architecture Reference

- [Layer 1 §1.2 — Orphan File Cleanup](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L126-L158)

---

## What to Implement

When a file-registration synthetic deploy is expelled from the mempool (expiration or eviction), the corresponding physical file on disk must be cleaned up to prevent storage leaks.

### Technical Details

#### Hook location: Deploy buffer eviction/expiration logic

The existing deploy buffer (`DeployBuffer` / `KeyValueDeployStorage`) already removes expired/evicted deploys. We need to tap into this removal event.

1. **Detect file-registration deploys**:
   - Check if the removed deploy's `term` matches the file registration pattern: `file!("register", ...)` on the `rho:io:file` system channel
   - Use simple string pattern matching or term parsing

2. **Cross-reference check**:
   - Before deleting the physical file, scan the remaining mempool for any other pending deploy that references the same `fileHash`
   - This prevents deleting a deduplicated file that another pending deploy still needs

3. **Physical cleanup**:
   - Delete `<data-dir>/file-replication/<hash>`
   - Delete `<hash>.meta.json`
   - Only if no other pending deploy references the same hash

4. **Trigger conditions**:
   - Deploy's `validAfterBlockNumber` window expired (block height advanced past range)
   - Deploy evicted from mempool (LMDB backpressure, mempool full)

---

## Verification

### New Test: `OrphanFileCleanupSpec`

**File**: `casper/src/test/scala/coop/rchain/casper/engine/OrphanFileCleanupSpec.scala`

Uses temp directories + mock deploy buffer (same pattern as existing `FileReplicationSpec` in the same package).

| Test Case | Assertion |
|-----------|-----------|
| File deploy expired | Physical file + `.meta.json` deleted from `file-replication/` |
| Two deploys reference same hash → expire one | File NOT deleted (cross-reference check) |
| Expire second (last) deploy | File IS deleted |
| File deploy evicted (LMDB backpressure) | Physical file cleaned up |
| Non-file deploy expired | No filesystem side-effects |
| File deploy term detection | `file!("register", ...)` pattern correctly identified |

```bash
sbt 'casper/testOnly coop.rchain.casper.engine.OrphanFileCleanupSpec'
```

### Existing Tests to Verify (regression)

```bash
# Deploy buffer changes must not break existing deploy lifecycle tests
sbt 'casper/testOnly coop.rchain.casper.batch1.*'
sbt 'casper/testOnly coop.rchain.casper.batch2.*'
```

---

## Subtasks

- [ ] Identify the deploy removal hook in the existing buffer code
- [ ] Implement file-registration deploy detection (term pattern matching)
- [ ] Implement cross-reference check (scan pending deploys for same hash)
- [ ] Implement physical file + meta.json deletion
- [ ] Wire cleanup logic into deploy expiration path
- [ ] Wire cleanup logic into deploy eviction path
- [ ] Unit tests for all edge cases
