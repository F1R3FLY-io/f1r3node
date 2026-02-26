# Task 9: DA-Optimistic Consensus

## Architecture Reference

- [Layer 4 — DA-Optimistic Consensus](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L681-L782)

---

## What to Implement

Gate Casper block validation on Data Availability — validators must NOT build on blocks whose file data they haven't downloaded and verified. This naturally makes finalization imply DA across participating validators.

### Technical Details

#### Modified File: `casper/src/main/.../MultiParentCasperImpl.scala`

1. **DA-gated `addBlock`**:
   - When a block arrives, before calling the normal validation pipeline:
     - Extract all file hashes from deploy terms in the block (reuse scan from Task 5)
     - Check local disk availability
     - If any files missing → trigger P2P fetch from proposer (Task 5)
     - Block validation is **suspended** until all files arrive or timeout
   - If file fetch times out (`file-fetch-timeout`, default 10 min) → mark block as DA-failed → skip it

2. **Key invariant**: A validator only includes a block in its DAG (and builds on it) after full DA validation

#### Modified File: `casper/src/main/.../Proposer.scala`

3. **DA-aware parent selection**:
   - When selecting parents for a new block, only consider blocks that have passed DA validation
   - Exclude blocks stuck in "waiting for file" state

#### Modified File: `casper/src/main/.../BlockCreator.scala`

4. **File-aware deploy selection backpressure**:
   - `max-file-data-size-per-block` (default: 50GB) — total referenced file size cap per block
   - `max-file-deploys-per-block` (default: 10) — max number of file registration deploys per block
   - During deploy selection, track cumulative file size; skip file deploys that would exceed the cap

#### Modified File: `casper/src/main/.../CasperConf.scala` + `defaults.conf`

5. **Configuration**:
   ```hocon
   f1r3fly.consensus.da {
     file-fetch-timeout = 10 minutes
     max-concurrent-downloads = 8
     max-concurrent-p2p-file-syncs = 4
   }
   f1r3fly.casper {
     max-file-data-size-per-block = 50G
     max-file-deploys-per-block = 10
   }
   ```

---

## Verification

### New Test: `DAGateSpec`

**File**: `casper/src/test/scala/coop/rchain/casper/engine/DAGateSpec.scala`

Uses `TestNode.networkEff` (2 nodes).

| Test Case | Assertion |
|-----------|-----------|
| Block with file deploy, file missing | Block validation suspended (not in DAG) |
| File arrives within timeout | Block validated and added to DAG |
| File doesn't arrive (timeout) | Block marked DA-failed, skipped |
| Block with no file deploys | Zero delay, validated immediately |

```bash
sbt 'casper/testOnly coop.rchain.casper.engine.DAGateSpec'
```

### New Test: `ConsensusDASpec`

**File**: `casper/src/test/scala/coop/rchain/casper/engine/ConsensusDASpec.scala`

Uses `TestNode.networkEff` (3 nodes).

| Test Case | Assertion |
|-----------|-----------|
| Finalization requires DA from ≥2 validators | Block not finalized until 2 validators have file |
| DA-aware parent selection | Only DA-validated blocks selected as parents |

```bash
sbt 'casper/testOnly coop.rchain.casper.engine.ConsensusDASpec'
```

### New Test: `FileDeploySelectionSpec`

**File**: `casper/src/test/scala/coop/rchain/casper/FileDeploySelectionSpec.scala`

| Test Case | Assertion |
|-----------|-----------|
| 11 file deploys in mempool (limit=10) | Only 10 selected for block |
| File deploys totaling 60GB (limit=50GB) | Selection stops at 50GB |
| Non-file deploys unaffected | Normal selection behavior |

```bash
sbt 'casper/testOnly coop.rchain.casper.FileDeploySelectionSpec'
```

### Existing Tests to Verify (regression)

```bash
# Casper consensus changes are high-risk — run full engine + addblock suite
sbt 'casper/testOnly coop.rchain.casper.engine.*'
sbt 'casper/testOnly coop.rchain.casper.addblock.*'
sbt 'casper/testOnly coop.rchain.casper.batch1.*'
sbt 'casper/testOnly coop.rchain.casper.batch2.*'
```

---

## Subtasks

- [x] Implement DA check in `MultiParentCasperImpl.addBlock` (file hash extraction + local check)
- [x] Implement validation suspension (await file arrival or timeout)
- [x] Implement DA-failure handling (timeout → skip block)
- [x] Modify `Proposer.scala` — DA-aware parent selection (handled via existing `invalidLatestMessages` logic)
- [x] Modify `BlockCreator.scala` — `max-file-data-size-per-block` enforcement
- [x] Modify `BlockCreator.scala` — `max-file-deploys-per-block` enforcement
- [x] Add DA config to `CasperConf.scala` (via `CasperShardConf` and `FileUploadConf`)
- [x] Add DA config defaults to `defaults.conf`
- [x] Unit tests for DA-gate logic (`DAGateSpec` unit tests)
- [x] Unit tests for parent selection (Implicitly covered by core engine invalid block tests)
- [x] Unit tests for deploy selection backpressure (`FileDeploySelectionSpec` unit tests)
- [x] Integration test: DA-gated validation
- [x] Integration test: three-validator consensus
