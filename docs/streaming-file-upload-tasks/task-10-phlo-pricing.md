# Task 10: Storage-Proportional Phlo Pricing

## Architecture Reference

- [Layer 5 — Storage-Proportional Cost Accounting](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L785-L890)

---

## What to Implement

Ensure uploaders pay phlo strictly proportional to the physical file size. Enforced at two points: upload-time (early rejection) and block execution-time (consensus enforcement).

### Technical Details

#### Cost Formula

```
totalPhlo = BASE_REGISTER_PHLO + (fileSize × phloPerStorageByte)
```

| Component | Default | Example (10GB) |
|-----------|---------|----------------|
| `BASE_REGISTER_PHLO` | 300 | 300 phlo |
| `phloPerStorageByte` | 1 | 10,737,418,240 phlo |
| **Total** | — | ~10.7B phlo |

#### New File: `node/src/main/.../FileUploadCosts.scala`

1. Constants: `BASE_REGISTER_PHLO`, `DEFAULT_PHLO_PER_STORAGE_BYTE`
2. Calculation method: `totalRequired(fileSize, phloPerStorageByte) = BASE_REGISTER_PHLO + fileSize * phloPerStorageByte`

#### Enforcement Point 1: Upload-Time (in `FileUploadAPI.scala`)

3. Before accepting any data chunks:
   - Compute `totalRequired` from `metadata.fileSize` and config
   - If `metadata.phloLimit < totalRequired` → reject immediately with detailed error
   - File is not written; no deploy created; client gets fast feedback

#### Enforcement Point 2: Block Execution (in `FileSystemProcess.scala`)

4. During `register` handler:
   - `costAccounting.charge(Cost(fileSize * phloPerStorageByte))`
   - If phlo exhausted → `OutOfPhloError` → deploy fails, registration rolled back
   - This is the consensus-enforced check that validators agree on

5. The `fileSize` passed to the system process comes from the synthetic deploy term (deterministic). The `nodeSigHex` cryptographic envelope proves the proposer validated this size.

#### Downloads

6. Downloads are **free** — no phlo charged

#### Return to Client

7. `FileUploadResult` includes:
   - `storagePhloCost` = `fileSize × phloPerStorageByte`
   - `totalPhloCharged` = `BASE_REGISTER_PHLO + storagePhloCost`

---

## Verification

### New Test: `FileUploadCostSpec`

**File**: `node/src/test/scala/coop/rchain/node/api/FileUploadCostSpec.scala`

| Test Case | Assertion |
|-----------|-----------|
| `totalRequired(0, 1)` | `= 300` (base only) |
| `totalRequired(1024, 1)` | `= 1324` |
| `totalRequired(10GB, 1)` | `= 10_737_418_540` |
| Upload: `phloLimit=100, fileSize=1024` | Immediate rejection with cost error message |
| Upload: `phloLimit=2000, fileSize=1024` | Accepted (2000 ≥ 1324) |
| `FileUploadResult.storagePhloCost` | Matches `fileSize × phloPerStorageByte` |
| `FileUploadResult.totalPhloCharged` | Matches `BASE + storagePhloCost` |

```bash
sbt 'node/testOnly coop.rchain.node.api.FileUploadCostSpec'
```

### New Test: `StorageCostEnforcementSpec` (execution-time)

**File**: `casper/src/test/scala/coop/rchain/casper/StorageCostEnforcementSpec.scala`

Uses `TestNode` to verify phlo deduction during block execution.

| Test Case | Assertion |
|-----------|-----------|
| Register file, sufficient phlo | Deploy succeeds, phlo deducted = `fileSize × rate` |
| Register file, insufficient phlo | `OutOfPhloError`, deploy fails, registration rolled back |

```bash
sbt 'casper/testOnly coop.rchain.casper.StorageCostEnforcementSpec'
```

### Existing Tests to Verify (regression)

```bash
# Phlo charging changes are high-risk for existing cost accounting
sbt 'rholang/testOnly coop.rchain.rholang.interpreter.*'
```

---

## Subtasks

- [ ] Create `FileUploadCosts.scala` with constants and calculation method
- [ ] Implement upload-time phlo validation in `FileUploadAPI.scala`
- [ ] Implement execution-time phlo charging in `FileSystemProcess.scala`
- [ ] Add `storagePhloCost` / `totalPhloCharged` to `FileUploadResult` response
- [ ] Add CLI flag `--file-upload-phlo-per-storage-byte` to `Options.scala`
- [ ] Add config defaults to `defaults.conf`
- [ ] Unit tests for cost calculation
- [ ] Unit tests for upload-time rejection
- [ ] Unit tests for execution-time charging
- [ ] Integration test for end-to-end cost enforcement
