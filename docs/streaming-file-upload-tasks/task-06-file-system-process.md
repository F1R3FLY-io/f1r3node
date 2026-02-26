# Task 6: FileSystemProcess — System Process (`rho:io:file`)

## Architecture Reference

- [Layer 3 §3.1–3.2 — Architecture Overview & Per-File Registry](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L326-L377)
- [Layer 5 §5.3 — Block Execution Validation](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L843-L863)
- [Module Summary](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L893-L922)

---

## What to Implement

Add `rho:io:file` system process handlers (`register` and `delete`) inside the existing `SystemProcesses.scala`, following the same `Contract[F]` pattern used by `stdOut`, `vaultAddress`, `ollamaChat`, etc. Register the new channel in `FixedChannels`, `BodyRefs`, and the `Definition` list.

### Technical Details

#### Modified File: [SystemProcesses.scala](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/rholang/src/main/scala/coop/rchain/rholang/interpreter/SystemProcesses.scala)

1. **Add to trait `SystemProcesses[F[_]]`**:
   ```scala
   def fileRegister: Contract[F]
   def fileDelete: Contract[F]
   ```

2. **Add to `FixedChannels`** (next available byte after `DEPLOY_DATA = 31`):
   ```scala
   val FILE_IO: Par = byteName(32)
   ```

3. **Add to `BodyRefs`**:
   ```scala
   val FILE_REGISTER: Long = 30L
   val FILE_DELETE: Long   = 31L
   ```

4. **Add to `nonDeterministicCalls`** if applicable (delete interacts with filesystem).

5. **Implement `fileRegister: Contract[F]`** — handles `file!("register", fileHash, fileSize, fileName, nodeSigHex, *ret)`:
   - **Verify cryptographic envelope**: validate `nodeSigHex` over `"$fileHash:$fileSize"` using `context.blockProposer` public key (from `ProcessContext.blockData`). This proves the proposer physically verified the file exists on disk with the correct size.
   - **Charge storage phlo**: `costAccounting.charge(Cost(fileSize * phloPerStorageByte))`. If phlo is exhausted → `OutOfPhloError` → deploy fails, registration rolled back.
   - **Delegate to FileRegistry**: Natively inject a call to `FileRegistry!("register", fileHash, fileName, context.deployer, sysAuthToken, *ret)`. The `sysAuthToken` is passed so that only system processes can call register — user code cannot call `FileRegistry!("register", ...)` directly.
   - Returns `(true, fileHash)` on success

6. **Implement `fileDelete: Contract[F]`** — handles `file!("delete", fileHash, sysAuthToken, *ret)`:
   - **Verify `SysAuthToken`**: check that the token matches the one stored at genesis
   - **Physical file deletion**: delete `<data-dir>/file-replication/<hash>` from disk
   - **Note**: deletion is "blind" — only called by `FileRegistry._deleteTemplate` after all ref-count checks pass. The system process does NOT do ownership checks; that's the registry's job.
   - Returns `(true, fileHash)` on success; `(false, "File not found")` if file doesn't exist

7. **Wire into `Definition` list** — add `Definition[F]` entries for the `rho:io:file` channel (same pattern as `rho:io:stdout`, `rho:io:ollamaChat`, etc.)

#### Modified File: `rholang/src/main/.../Runtime.scala`

- Wire `fileReplicationDir: Path` into the `ProcessContext` or pass it via config so the `fileDelete` handler knows where files live.

#### Deterministic Replay Safety

- **Register**: Deterministic — file sync (Layer 2) guarantees the file is on disk before replay. The `nodeSigHex` verification uses `context.blockProposer` which is deterministic per block.
- **Delete**: Deterministic — only modifies on-chain state. Physical file removal happens asynchronously after finalization, not during execution.

---

## Verification

### New Test: `FileSystemProcessSpec`

**File**: `rholang/src/test/scala/coop/rchain/rholang/interpreter/FileSystemProcessSpec.scala`

Same pattern as existing [RuntimeSpec.scala](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/rholang/src/test/scala/coop/rchain/rholang/interpreter/RuntimeSpec.scala) — ScalaTest FlatSpec + `mkRuntime`.

| Test Case | Assertion |
|-----------|-----------|
| Register with valid envelope | Phlo charged (`CostAccounting.charge` called), `FileRegistry` invoked |
| Register with invalid `nodeSigHex` | Rejection error, no state change |
| Register with insufficient phlo | `OutOfPhloError`, deploy rolled back |
| Delete with valid `SysAuthToken` | File removed from `file-replication/` dir |
| Delete with invalid `SysAuthToken` | Rejection, file untouched |
| Delete non-existent file | Returns `(false, "File not found")` |
| `rho:io:file` channel registration | Channel exists in system process table |

```bash
sbt 'rholang/testOnly coop.rchain.rholang.interpreter.FileSystemProcessSpec'
```

### Existing Tests to Verify (regression)

```bash
# System process changes must not break existing interpreter tests
sbt 'rholang/testOnly coop.rchain.rholang.interpreter.RuntimeSpec'
sbt 'rholang/testOnly coop.rchain.rholang.interpreter.ReduceSpec'
```

---

## Subtasks

- [x] Add `fileRegister` / `fileDelete` to `SystemProcesses` trait
- [x] Add `FILE_IO` to `FixedChannels` (byte 32)
- [x] Add `FILE_REGISTER` / `FILE_DELETE` to `BodyRefs`
- [x] Implement `fileRegister` contract: cryptographic envelope verification (Secp256k1 over SHA-256(`"$hash:$size"`), wrapped in `Try` for robustness)
- [x] Implement `fileRegister` contract: `CostAccounting.charge(fileStorageCost(...))` for storage phlo
- [x] Implement `fileRegister` contract: delegate to `FileRegistry` with `sysAuthToken` via `FILE_REGISTRY_NOTIFY` fixed channel bridge
- [x] Implement `fileDelete` contract: `SysAuthToken` verification
- [x] Implement `fileDelete` contract: physical file deletion from disk
- [x] `fileDelete` marked as `nonDeterministicCalls` with replay branch (re-produces captured output, skips re-deletion)
- [x] Add `Definition[F]` entries for `rho:io:file` channel (two definitions: arity 6 for register, arity 4 for delete)
- [x] Wire `fileReplicationDir` into `ProcessContext` and thread through full `RhoRuntime` creation chain
- [x] Ensure `sysAuthToken` passed at genesis initialization via `FileRegistryInitDeploy` system deploy
- [x] Unit tests for register (invalid sig, malformed hex, graceful error output)
- [x] Unit tests for delete (valid `SysAuthToken` with file assertion `Files.exists == false`, invalid token, non-existent file)
- [x] `rho:io:file` read-only channel guard test
- [x] Storage cost charge integration tests (`StorageCostChargeSpec` — 5 tests)
- [x] Casper-level storage cost enforcement tests (`StorageCostEnforcementSpec` — 6 tests)
- [x] Regression: `RuntimeSpec` (6/6 pass)
- [x] Regression: `FileRegistrySpec` (8/8 pass)
- [x] Regression: `PoSSpec` (15/15 pass)

---

## Deferred Items (completed)

| Item | Resolution |
|------|------------|
| Delegate to `FileRegistry!("register", ...)` with `sysAuthToken` in `fileRegister` | ✅ Implemented via `FILE_REGISTRY_NOTIFY` fixed channel bridge — system process produces `(fileHash, fileName, deployerId, GSysAuthToken())`, `FileRegistry.rho` bridge consumer forwards to `register` API |
| Pass `sysAuthToken` at genesis so `FileRegistry` can call `rho:io:file` delete | ✅ Implemented via `FileRegistryInitDeploy` system deploy — runs during `computeGenesis` after blessed terms, calls `FileRegistry!("init", sysAuthToken)` |
