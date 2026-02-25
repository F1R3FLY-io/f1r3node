# Task 7: FileRegistry.rho — On-Chain Contract

## Architecture Reference

- [Layer 3 §3.2–3.6 — Per-File Registry, Ownership, Deletion](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L326-L647)
- [Existing contract file](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/casper/src/main/resources/FileRegistry.rho)

---

## What to Implement

The on-chain Rholang contract that manages file ownership via a `TreeHashMap` mapping `fileHash → fileHandle`. Provides AuthKey-gated delete with reference counting (deployers array) and `SysAuthToken` security gate on registration.

### Technical Details

#### File: `casper/src/main/resources/FileRegistry.rho`

Architecture mirrors **SystemVault.rho** (see vault/AuthKey pattern comparison in the arch doc).

1. **State shape**:
   ```rholang
   @{["fileState", "<fileHash>"]}!({ "deployers": [pubKey1, pubKey2], "name": "<fileName>" })
   ```

2. **`FileRegistry!("register", fileHash, fileName, ownerDeployerId, sysAuthToken, *ret)`**:
   - Requires `SysAuthToken` (only callable from system process, not user code)
   - Lookup `fileHash` in `fileMap` (TreeHashMap):
     - **Not found**: Create new `fileHandle` capability, insert into `fileMap`, create state channel with `deployers: [ownerDeployerId]`, `name: fileName`
     - **Found**: Read existing state, append `ownerDeployerId` to `deployers` array (dedup)
   - Return `(true, fileHash)`

3. **`FileRegistry!("ownerAuthKey", fileHash, ownerDeployerId, *ret)`**:
   - Issues an `AuthKey` with shape `(_fileRegistry, fileHash)` where `_fileRegistry` is an unforgeable private name
   - Only works within the same deploy context that owns the file
   - Returns `Nil` if the caller's deployer ID is not in the `deployers` array

4. **`FileRegistry!("lookup", fileHash, *ret)`**:
   - Returns the `fileHandle` bundle for the given hash
   - Uses `<<-` (peek) for non-consuming read → zero conflict in parallel reads

5. **`fileHandle!("delete", authKey, *ret)`** → `_deleteTemplate`:
   - Step 1: `AuthKey!("check", authKey, (_fileRegistry, fileHash), *authValidCh)` — verify identity
   - Step 2: Read `deployers` array from state channel, remove caller's pubKey
   - Step 3: If array becomes empty:
     - Read `_sysAuthTokenCh` (private, unforgeable)
     - Call `rho:io:file!("delete", fileHash, sysAuthToken, *sysDeleteCh)` — physical deletion
     - `TreeHashMap!("delete", fileMap, fileHash, *ackCh)` — remove from on-chain registry
     - Clean up state channel
   - If array NOT empty: write updated state back, return `(true, fileHash)`

6. **Genesis initialization**:
   - `FileRegistry` receives `SysAuthToken` during genesis
   - Stores it on a private unforgeable channel `_sysAuthTokenCh`
   - Registers itself at URI `rho:file:registry` via `rho:registry:insertSigned`

---

## Verification

### New Test: `FileRegistryTest.rho` (Rholang unit test)

**File**: `casper/src/test/resources/FileRegistryTest.rho`

Uses `RhoSpec` test framework (same pattern as existing [SystemVaultTest.rho](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/casper/src/test/resources/SystemVaultTest.rho), [AuthKeyTest.rho](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/casper/src/test/resources/AuthKeyTest.rho), [TreeHashMapTest.rho](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/casper/src/test/resources/TreeHashMapTest.rho)).

| Test Case | Assertion |
|-----------|-----------|
| Register new file | State channel contains `{ deployers: [pubKey], name: "test.txt" }` |
| Register same file, different deployer | Both deployers in array |
| Register same file, same deployer | No duplicate in array |
| `ownerAuthKey` for valid owner | Returns valid AuthKey |
| `ownerAuthKey` for non-owner | Returns `Nil` |
| Delete with valid AuthKey | Deployer removed from array |
| Delete last deployer | `rho:io:file!("delete", ...)` called, TreeHashMap entry removed |
| Delete with invalid AuthKey | Returns `(false, "Invalid AuthKey")` |
| Lookup (non-consuming read) | Data still available after lookup |

### New Scala Test Runner: `FileRegistrySpec`

**File**: `casper/src/test/scala/coop/rchain/casper/FileRegistrySpec.scala`

```scala
class FileRegistrySpec extends RhoSpec(
  CompiledRholangSource("FileRegistryTest.rho", NormalizerEnv.Empty),
  Seq.empty,
  10.minutes
)
```

```bash
sbt 'casper/testOnly coop.rchain.casper.FileRegistrySpec'
```

### Existing Tests to Verify (regression)

```bash
# Contract changes must not break existing genesis / system contract tests
sbt 'casper/testOnly coop.rchain.casper.SystemContractInitializationSpec'
sbt 'casper/testOnly coop.rchain.casper.genesis.*'
```

---

## Subtasks

- [ ] Define `_fileRegistry` unforgeable private name
- [ ] Implement `_sysAuthTokenCh` storage (genesis init)
- [ ] Implement `register` operation (new file path: create handle + state)
- [ ] Implement `register` operation (existing file path: append deployer)
- [ ] Implement deduplication guard (don't add same deployer twice)
- [ ] Implement `ownerAuthKey` operation (AuthKey issuance)
- [ ] Implement `lookup` operation (non-consuming read via `<<-`)
- [ ] Implement `_deleteTemplate` — AuthKey check
- [ ] Implement `_deleteTemplate` — deployer array removal
- [ ] Implement `_deleteTemplate` — array-empty path (system delete + TreeHashMap cleanup)
- [ ] Implement `_deleteTemplate` — array-not-empty path (write updated state)
- [ ] Register at `rho:file:registry` via `rho:registry:insertSigned`
- [ ] Genesis wiring: pass `SysAuthToken` during initialization
- [ ] Rholang unit tests
