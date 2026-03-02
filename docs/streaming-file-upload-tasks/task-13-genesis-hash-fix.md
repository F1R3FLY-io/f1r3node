# Task 13: Genesis Hash Mismatch Fix (FileRegistry Self-Init)

## Problem

Multi-node shard genesis fails with "Tuplespace hash mismatch" because the bootstrap
node's `computeGenesis` ran a separate `FileRegistryInitDeploy` system deploy after the
blessed terms, producing a different post-state hash than validators (which only replayed
the blessed terms via `replayComputeState`).

---

## What Was Changed

Made `FileRegistry.rho` a regular self-initializing blessed contract — no separate
system deploy needed at genesis.

### Modified File: `casper/src/main/resources/FileRegistry.rho`

- Removed the `init` contract (one-shot SysAuthToken receiver)
- `registerNotify` now stores `sysAuthToken` in `_sysAuthTokenCh` on every call
  (idempotent; `_deleteTemplate` uses peek so extra values are harmless)

### Modified File: `casper/src/main/scala/.../rholang/RuntimeSyntax.scala`

- Removed `FileRegistryInitDeploy` step from `computeGenesis`
- `computeGenesis` now simply plays deploys → checkpoint → done

### Deleted File: `casper/src/main/scala/.../costacc/FileRegistryInitDeploy.scala`

No longer needed — FileRegistry initializes itself.

### Modified File: `casper/src/main/scala/.../engine/BlockApproverProtocol.scala`

Validator validation path: direct `stateHash == postState.postStateHash` comparison
(no workaround needed since `computeGenesis` no longer diverges).

### Modified File: `casper/src/main/scala/.../engine/Initializing.scala`

Genesis replay path: direct comparison (same as above).

---

## Verification

```bash
sbt "casper/testOnly coop.rchain.casper.engine.BlockApproverProtocolTest"
```

All 6 tests pass, including "should successfully validate correct candidate" which
exercises the exact validation path that was broken.

---

## Subtasks

- [x] Remove `init` contract from `FileRegistry.rho`
- [x] Make `registerNotify` self-initializing (store SysAuthToken)
- [x] Remove `FileRegistryInitDeploy` from `computeGenesis`
- [x] Delete `FileRegistryInitDeploy.scala`
- [x] Verify `BlockApproverProtocolTest` passes (6/6)
