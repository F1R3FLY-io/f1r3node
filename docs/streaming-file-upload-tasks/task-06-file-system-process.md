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

- [ ] Add `fileRegister` / `fileDelete` to `SystemProcesses` trait
- [ ] Add `FILE_IO` to `FixedChannels` (byte 32)
- [ ] Add `FILE_REGISTER` / `FILE_DELETE` to `BodyRefs`
- [ ] Implement `fileRegister` contract: cryptographic envelope verification
- [ ] Implement `fileRegister` contract: `CostAccounting.charge()` for storage phlo
- [ ] Implement `fileRegister` contract: delegate to `FileRegistry` with `sysAuthToken`
- [ ] Implement `fileDelete` contract: `SysAuthToken` verification
- [ ] Implement `fileDelete` contract: physical file deletion from disk
- [ ] Add `Definition[F]` entries for `rho:io:file` channel
- [ ] Wire `fileReplicationDir` + config into `Runtime.scala` / `ProcessContext`
- [ ] Ensure `sysAuthToken` passed at genesis initialization
- [ ] Unit tests for register (happy path + failures)
- [ ] Unit tests for delete (happy path + failures)
