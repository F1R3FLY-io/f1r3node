# Streaming File Upload and Distributed Data Availability

## Executive Summary

This document details the architecture for the F1R3FLY network's native support for large binary file handling (10GB+). Standard blockchain transactions and execution engines are not optimized for large BLOBs (Binary Large Objects) due to memory exhaustion (OOM), propagation delays, and deterministic execution suspensions.

To solve this, F1R3FLY introduces a **Five-Layer Architecture** that separates Data Availability (DA) from Execution, guaranteeing fast finalization, strict file integrity, and Automated Blockchain Connection without inflating the LMDB state database.

### Design Priorities

| # | Priority | Solution Layer |
|---|----------|---------------|
| 1 | Support 10GB files in a single upload | Layer 1: Streaming Ingestion with disk-direct I/O |
| 2 | Block validity: deploy connected to file | Layer 1: Automated synthetic deploy + Layer 2: File Sync |
| 3 | Client downloads file once block is created | Layer 3: `rho:io:file` system process |
| 4 | Fast finalization with DA guarantees | Layer 4: DA-Optimistic Consensus |
| 5 | Fair cost accounting for storage | Layer 5: Storage-Proportional Phlo Pricing |

> **Note:** Deploy lifecycle tracking (queued → executed → finalized) uses existing RPCs — `findDeploy(deployId)` returns the block containing the deploy, and `isFinalized(blockHash)` checks finality. No new API is needed.

### Supported Operations

| Operation | Mechanism | Description |
|-----------|-----------|-------------|
| **Upload** | `uploadFile` gRPC stream | Streaming ingestion + automatic blockchain registration |
| **Download** | `downloadFile` gRPC stream | Direct byte streaming with resume support (**observer-only**) |
| **Delete** | `doDeploy` via `FileRegistry` → `fileHandle!("delete", authKey, *ret)` | Owner-only delete — `FileRegistry` validates `AuthKey`, then calls `rho:io:file` (with `SysAuthToken`); removes hash and schedules physical file removal |
| **Update** | Delete old + Upload new | Content-addressed storage means new content = new hash. Client tracks version history. |

---

## Minimal Architecture Overview

1. **Ingestion (Layer 1)**: Files stream directly to the local disk, bypassing execution memory. A synthetic `Deploy` is automatically created to link the file to the blockchain.
2. **Sync (Layer 2)**: Validators fetch missing files from peers natively during block propagation, ensuring data exists before validation begins.
3. **Execution (Layer 3)**: The `FileRegistry.rho` contract maintains a **per-file registry map** in RSpace — each file hash maps to a metadata state containing the `fileName` and a map of `deployers` (owners). This structure provides **native zero-cost deduplication with reference counting**. The `rho:io:file` system process only executes physical disk deletion once the deployers map for a file becomes completely empty. The client downloads the file via gRPC when the block finalizes.
4. **Validation (Layer 4)**: The Casper consensus engine is strictly gated by data availability—blocks won't finalize until validators have verified the file data.
5. **Pricing (Layer 5)**: The uploader is charged execution Phlo strictly proportional to the physical size of the file to prevent storage bloat.

---

## Layer 1: Ingestion & Automated Block Connection

The ingestion layer handles the node's `DeployService` gRPC API via the `uploadFile` streamed endpoint. It safely saves multi-gigabyte files, ensures no partial uploads persist, and automatically generates the Rholang `Deploy` transaction that connects the file to the blockchain.

### 1.1 gRPC Streaming Protocol

Standard gRPC limits (`grpc-max-recv-message-size`, default 16MB in `Options.scala`) prevent large files from being sent in a single request. The `uploadFile` endpoint employs client-side streaming:

#### Protobuf Definition (`models/src/main/protobuf/DeployServiceV1.proto`)

```protobuf
service DeployService {
  // Existing RPCs ...
  rpc uploadFile(stream FileUploadChunk) returns (FileUploadResponse) {}
}

message FileUploadMetadata {
  bytes  deployer             = 1;  // ed25519 public key (same as DeployDataProto)
  int64  timestamp            = 2;  // millisecond epoch timestamp
  bytes  sig                  = 3;  // signature over metadata fields
  string sigAlgorithm         = 4;  // "ed25519"
  int64  phloPrice            = 5;  // phlo price for the synthetic deploy
  int64  phloLimit            = 6;  // phlo limit for the synthetic deploy
  int64  validAfterBlockNumber = 7; // block height validity constraint
  string shardId              = 8;  // shard identifier
  string fileName             = 9;  // original filename (informational)
  int64  fileSize             = 10; // total file size in bytes (for progress tracking)
  string fileHash              = 11; // Blake2b-256 pre-computed by client (for deduplication)
  string term                  = 12; // Rholang term signed by client (for on-chain registration)
}

message FileUploadChunk {
  oneof chunk {
    FileUploadMetadata metadata = 1;  // MUST be first message
    bytes data                  = 2;  // Subsequent binary chunks (1-4MB each)
  }
}

message FileUploadResponse {
  oneof message {
    servicemodelapi.ServiceError error = 1;
    FileUploadResult result            = 2;
  }
}

message FileUploadResult {
  string fileHash         = 1;  // Blake2b-256 content hash
  string deployId         = 2;  // Signature of the synthetic deploy (tracking ID)
  int64  storagePhloCost  = 3;  // Phlo charged for storage
  int64  totalPhloCharged = 4;  // Total phlo that will be deducted
}
```

**Critical**: The response includes `deployId` — this is the deploy signature that the client uses to track the file's lifecycle via the existing `findDeploy` and `isFinalized` RPCs.

#### Chunk Flow

```
Client                                Node (DeployGrpcServiceV1)
  │                                      │
  ├─► FileUploadChunk(metadata)  ──────► │  1. Pre-validate phlo limits against exact `metadata.fileSize` & signature
  │                                      │  2. Check if `fileHash` exists on disk
  │                                      │  3A. If exists: Bypass upload, go to step 10
  │                                      │  3B. If not: Open temp file `<txn-id>.tmp` & hasher
  ├─► FileUploadChunk(data: 4MB) ──────► │  4. Write to disk & abort if `bytesReceived > metadata.fileSize`
  ├─► FileUploadChunk(data: 4MB) ──────► │  5. Feed bytes to streaming hasher
  │      ... (2500 chunks for 10GB) ...   │
  ├─► FileUploadChunk(data: 4MB) ──────► │  6. EOF detected (last chunk)
  │                                      │  7. Verify `bytesReceived == metadata.fileSize`
  │                                      │  8. Finalize hash, verify integrity matches `fileHash`
  │                                      │  9. Rename .tmp → <hash> (atomic)
  │                                      │ 10. Generate synthetic deploy (Section 1.3)
  │                                      │ 11. Push deploy to DeployBuffer (mempool)
  │  ◄── FileUploadResponse(hash, id) ──┤ 12. Return hash + deployId + cost to client
```

The node reads chunks natively and bypasses the JVM heap, writing directly to the underlying physical NVMe/SSD via `java.nio.channels.FileChannel` with direct `ByteBuffer` allocation. **There is no upper file size limit for the gRPC stream.**

### 1.2 Interruption and Atomic Guarantee

- If the gRPC stream terminates early (client disconnect, timeout, network failure): the `.tmp` file is **immediately deleted** on the server side via a `finally` block in the Monix `Observable` handler. No trace is left on disk. No transaction is created.
- The `uploadFile` handler continuously feeds incoming bytes into a `Blake2b-256` streaming hasher (`org.bouncycastle.crypto.digests.Blake2bDigest`), computing the hash incrementally without buffering the entire file.
- When the final chunk is received, the computed hash is finalized. Only then does the node perform the **atomic rename**: `<data-dir>/file-replication/<txn-id>.tmp` → `<data-dir>/file-replication/<hash>`.
- If a file with the same hash already exists (duplicate upload), the `.tmp` is deleted and the existing file is reused. The synthetic deploy is still created (idempotent file registration).

#### Orphan File Cleanup (Deploy Expiration / Eviction)

If the synthetic deploy is **never included in a block**, the uploaded file becomes an orphan. This occurs when:
- The deploy's `validAfterBlockNumber` window expires (block height advances past the validity range)
- The deploy is evicted from the mempool (e.g., mempool full, LMDB backpressure)

When the deploy buffer removes a file-registration deploy (expiration or eviction), it **deletes the associated file from disk**:

```
Deploy removed from mempool (expired / evicted)
        │
        ▼
┌────────────────────────────────────────────┐
│ Is this a file-registration deploy?         │
│ (term matches file!("register", …)          │
│  via `rho:io:file` system channel)          │
└──────────┬─────────────────────────────────┘
           │ Yes
           ▼
┌────────────────────────────────────────────┐
│ Extract fileHash from deploy term           │
│ Check: no other pending deploy references   │
│        the same fileHash                    │
└──────────┬─────────────────────────────────┘
           │ No other references
           ▼
┌────────────────────────────────────────────┐
│ Delete: <data-dir>/file-replication/<hash>  │
│ Delete: <hash>.meta.json                    │
└────────────────────────────────────────────┘
```

This prevents storage leaks from abandoned uploads. The check for other pending deploys referencing the same hash ensures that a duplicate upload's file is not deleted while another deploy still needs it.

### 1.3 Automated Synthetic Deploy (The Connection)

Once the file is atomically verified on disk, the node **automatically** creates the corresponding blockchain transaction. The client does not send a secondary `doDeploy` request.

#### Phlo Cost Validation (Storage-Proportional Pricing)

Before constructing the synthetic deploy, the node validates that the client's `phloLimit` is sufficient to cover the **storage-proportional cost** of the file (see Layer 5 for full details):

```scala
// In FileUploadAPI.scala — early cost validation before any file chunks are processed
val storagePhloCost = metadata.fileSize * config.phloPerStorageByte  // e.g., 1 phlo/byte
val totalPhloRequired = FileUploadCosts.BASE_REGISTER_PHLO + storagePhloCost

if (metadata.phloLimit < totalPhloRequired) {
  // Reject upload IMMEDIATELY before accepting any data chunks
  return FileUploadResponse(Error(
    s"Insufficient phloLimit: ${metadata.phloLimit}. " +
    s"Required: $totalPhloRequired ($BASE_REGISTER_PHLO base + $storagePhloCost storage for ${metadata.fileSize} bytes)"
  ))
}
```

#### Deploy Construction

The system internally constructs a `DeployDataProto`:

```scala
// The client constructs the Rholang term and signs the full DeployData.
// The node maps the upload metadata (including term + sig) to a DeployDataProto,
// validates the client's signature via DeployData.from(), and pushes to the mempool.
// This ensures deployer = client (not the node), so FileRegistry uses the client's identity.

// Client-side (SDK):
val syntheticTerm = s"""new ret, file(`rho:io:file`) in {
  file!("register", "$fileHash", $fileSize, "${metadata.fileName}", *ret)
}"""
val deployData = DeployData(term = syntheticTerm, timestamp = ..., phloPrice = ..., ...)
val signed = Signed(deployData, Secp256k1, clientPrivateKey)
// Client sends signed.sig, signed.sigAlgorithm, and term in FileUploadMetadata

// Server-side (FileUploadAPI → DeployGrpcServiceV1):
val proto = SyntheticDeploy.metadataToDeployProto(metadata) // maps metadata fields → DeployDataProto
val signed = DeployData.from(proto)  // validates client signature
BlockAPI.deploy(signed, ...)         // pushes to mempool — same path as doDeploy
```

This deploy is pushed directly into the node's Casper mempool (the `DeployBuffer` / `DeployStorage`). The deploy undergoes the same admission control as `doDeploy`:
- **Storage cost validation** (phloLimit ≥ base + fileSize × phloPerByte)
- LMDB headroom check (1GB minimum free space, per `KeyValueDeployStorage` backpressure)
- Signature validation
- `shardId` matching
- `validAfterBlockNumber` range check

**Result**: The client makes ONE call (`uploadFile`). If it succeeds, the file is on disk AND the transaction is in the mempool. The `deployId` returned allows tracking.

### 1.4 Module Changes for Layer 1

| Module | File(s) | Change Type | Description |
|--------|---------|-------------|-------------|
| `models` | `DeployServiceV1.proto` | **MODIFY** | Add `uploadFile` RPC, `FileUploadChunk`, `FileUploadMetadata`, `FileUploadResponse`, `FileUploadResult` messages |
| `node` | `DeployGrpcServiceV1.scala` | **MODIFY** | Implement gRPC streaming handler using Monix `Observable.foldLeftF` |
| `node` | `FileUploadAPI.scala` | **NEW** | Core logic: streaming write, Blake2b hashing, atomic rename, synthetic deploy construction |
| `node` | `FileMetadata.scala` | **NEW** | Data class for file metadata persistence (`<hash>.meta.json`) |
| `node` | `BlockAPI.scala` | **MODIFY** | Add `uploadFile` method that delegates to `FileUploadAPI` and calls `deploy()` internally |
| `node` | `Options.scala` | **MODIFY** | Add CLI flags: `--file-upload-chunk-size` (default 4MB), `--file-replication-dir` |
| `node` | `defaults.conf` | **MODIFY** | Add `file-upload` config section |

---

## Layer 2: Validator File Sync & Integrity

Once a file is correctly ingested, it must be available on all validators that need to execute blocks referencing it. Layer 2 ensures files are synced between validators with cryptographic integrity verification.

### 2.1 Direct Validator File Sync

When a validator receives a proposed block containing deploys that reference file hashes (via `file!("register", ...)` on the `rho:io:file` system channel), it checks whether the referenced files exist locally. If not, it fetches them directly from the block proposer via the existing P2P transport layer.

This is a **simple pull-based model**: the receiving validator knows which peer proposed the block and requests the missing file from that peer. No DHT discovery or multi-peer coordination is needed.

```
Block arrives from Validator A (P2P)
        │
        ▼
┌───────────────────┐
│  Scan block deploys│
│  for file references│
└────────┬──────────┘
         │
    ┌────┴─────┐
    │ All files │──── Yes ──► Execute block normally (Casper validation)
    │ on disk?  │
    └────┬─────┘
         │ No
         ▼
┌─────────────────────┐
│ Request missing files│
│ from block proposer  │───► Chunked streaming via P2P transport
│ (Validator A)        │
└────────┬────────────┘
         │
    ┌────┴─────┐
    │ Hash      │──── Mismatch ──► Delete .tmp, reject block
    │ verified? │
    └────┬─────┘
         │ Match
         ▼
┌─────────────────────┐
│ Rename .tmp → <hash>│
│ Execute block via    │
│ Casper validation    │
│ pipeline             │
└─────────────────────┘
```

**Key simplification**: The receiving validator does **not** build on (validate) the block until the file is available locally. This naturally gates block validation on data availability without a separate queue or complex state machine (see Layer 4 for consensus integration).

### 2.2 P2P File Transfer Protocol

File transfer uses a simple request/response streaming protocol over the existing `TransportLayer`:

```protobuf
// Addition to CommMessages

message FileRequest {
  bytes fileHash = 1;   // Blake2b-256 hash of requested file
}

message FilePacket {
  oneof content {
    int64 fileSize = 1;  // First message: total file size
    bytes data     = 2;  // Subsequent messages: file chunks (4MB each)
  }
}
```

The requesting validator:
1. Opens a `.tmp` file for the incoming data
2. Streams chunks to disk (same direct I/O as Layer 1, using **16MB chunks** by default), tracking `bytesReceived`
3. **Aborts immediately** if `bytesReceived > expectedFileSize` from the block's deploy — drops the connection and deletes `.tmp` (prevents remote disk exhaustion via infinite garbage streams)
4. Computes Blake2b-256 hash incrementally
5. Verifies hash matches the expected `fileHash` from the block's deploy
6. Performs atomic rename `.tmp` → `<hash>` on success
7. Deletes `.tmp` on hash mismatch (block is rejected)

### 2.3 Cryptographic Integrity Enforcement

If the calculated hash at EOF does not match the expected `fileHash` from the block's deploy:
- The `.tmp` file is immediately deleted
- The block is **rejected** — treated as invalid (the proposer sent a block referencing a file whose content doesn't match)
- The proposer may be scored negatively via the existing `PeerScore` mechanism

### 2.4 Module Changes for Layer 2

| Module | File(s) | Change Type | Description |
|--------|---------|-------------|-------------|
| `comm` | `CommMessages.scala` | **MODIFY** | Add `FileRequest` and `FilePacket` Protobuf message types |
| `comm` | `TransportLayer.scala` | **MODIFY** | Handle `FileRequest`/`FilePacket` P2P streaming |
| `casper` | `MultiParentCasperImpl.scala` | **MODIFY** | Check file availability before block validation; fetch from proposer if missing |
| `casper` | `Validate.scala` | **MODIFY** | Add `fileAvailability` validation check to the block validation pipeline |

---

## Layer 3: Execution — Per-File Registry & Reference Counting

Layer 3 uses an **on-chain `FileRegistry.rho` contract** that implements a per-file mapping (`fileHash -> { deployers: {id1: true, id2: true}, name: "filename.ext" }`). This structure provides native file deduplication, preserved filenames, and reference-counted physical deletion. The `rho:io:file` system process acts as a "dumb" I/O layer that only executes physical deletion when authorized by the registry.

> **Implementation caveat**: The `deployers` field uses a **Map** (`{pubKey: true}`) rather than a List (`[pubKey]`). Rholang Lists do not support `.contains()` (throws `MethodNotDefined`), so Maps are required for O(1) membership tests, `.delete(key)` removal, and `.size()` counting.

### 3.1 Architecture Overview

```text
┌────────────────────────────────────────────────────────────────┐
│              On-Chain (RSpace — FileRegistry.rho)              │
│                                                                │
│  fileMap: TreeHashMap[hash → fileHandle]                       │
│  state:   @{["fileState", hash]} → { deployers: {pk1: true},   │
│                                      name: "1.txt" }           │
│           ┌──────────────────────────┐                         │
│           ▼                          ▼                         │
│  register(hash, name)    delete(hash, callerPubKey, auth)      │
│ (add to deployers map)       (remove from deployers map)       │
│           │                          │                         │
│           │                     if array is empty:             │
└───────────┼──────────────────────────┼─────────────────────────┘
            ▼                          ▼
     ┌──────────────┐           ┌──────────────┐
     │ rho:io:file  │           │ rho:io:file  │
     │  "register"  │           │   "delete"   │
     │ (sys process)│           │ (sys process)│
     └──────┬───────┘           └──────┬───────┘
            │                          │
            ▼                          ▼
  ┌──────────────────────────────────────────┐
  │  Filesystem: <data-dir>/file-replication/ │
  │  <hash1>, <hash2>, ...                    │
  └──────────────────────────────────────────┘
```

### 3.2 Per-File Registry Map — On-Chain Storage

The system uses `FileRegistry.rho` to manage file ownership and access control. This directly resolves the pitfalls of a purely physical system (where one user deleting a deduplicated file destroys it for everyone else).

#### Deduplication & Reference Counting

If Alice and Bob upload the identical 10GB file:
1. **Layer 1** detects the hash exists and skips the physical upload (saving 10GB).
2. **Layer 3** calls `FileRegistry!("register")`, which reads the existing `fileState` and adds Bob's public key to the `deployers` map.
3. The registry state becomes: `{ "deployers": {AlicePk: true, BobPk: true}, "name": "ubuntu.iso" }`.
4. If Alice deletes the file using her `ownerAuthKey`, her key is removed from the array. The file is NOT physically deleted because Bob's key still references it.
5. Once Bob also deletes it (array length becomes 0), the registry passes the unforgeable `SysAuthToken` to `rho:io:file` to permanently nuke the physical file.

#### Channel Naming Convention

```rholang
@{["fileState", "<fileHash>"]}!( { "deployers": {pubKey1: true, ...}, "name": "<fileName>" } )
```

### 3.3 FileRegistry — Ownership & AuthKey Layer

The **`FileRegistry.rho`** Rholang contract sits above the system process, providing the encapsulated ownership model — structured almost identically to `SystemVault.rho`.

| **SystemVault** | **FileRegistry** |
|---|---|
| `vaultMap: TreeHashMap[address → vault]` | `fileMap: TreeHashMap[hash → fileHandle]` |
| `_systemVault` (private unforgeable) | `_fileRegistry` (private unforgeable shape secret) |
| `deployerAuthKey` → issues `AuthKey(shape=(_sv, addr))` | `ownerAuthKey` → issues `AuthKey(shape=(_fr, hash, pubKey))` |
| `vault!("transfer", target, amount, authKey, ret)` | `fileHandle!("delete", callerPubKey, authKey, ret)` |
| `_transferTemplate` — validates auth → moves purse tokens | `_deleteTemplate` — validates auth → drops ref. If ref=0 → system delete |
| Physical action: Mint/purse token transfer | Physical action: System process disk deletion |

#### `FileRegistry!("register", fileHash, fileName, ownerDeployerId, sysAuthToken, ret)`

Called by the `rho:io:file` system process immediately after file ingestion and Phlo cost deduction. It requires the `SysAuthToken` to prevent direct invocation from user Rholang code. Creates the per-file `fileHandle` capability and inserts it into `fileMap`. The owner's pubkey is added to the `deployers` map.

#### `FileRegistry!("ownerAuthKey", fileHash, ownerDeployerId, ret)`

Issues an `AuthKey` whose shape is `(_fileRegistry, fileHash)`. Since `_fileRegistry` is a private unforgeable name known only inside the contract, the key cannot be forged. Only works within the same deploy that owns the file. Returns `Nil` if the caller is not the owner.

#### `fileHandle!("delete", authKey, ret)` — The Delete Flow

Follows the same validation pipeline as `_transferTemplate`:

```
fileHandle!("delete", authKey, *ret)
         │
         ▼
_deleteTemplate(fileHash, authKey, ret)
         │
         ├─ Step 1 ─► AuthKey!("check", authKey, (_fileRegistry, fileHash), *authValidCh)
         │               │ false → ret!(false, "Invalid AuthKey")   ← short-circuit
         │               │ true  ↓
         ├─ Step 2 ─► for @storedToken <<- _sysAuthTokenCh  ← private, cannot be forged
         │               ↓
         ├─ Step 2b ─► rho:io:file!("delete", fileHash, storedToken, *sysDeleteCh)
         │               │             (system process validates SysAuthToken, then
         │               │              removes hash from deployer list,
         │               │              schedule physical file removal after finalization)
         │               │ false → ret!(false, "File not found")    ← already deleted?
         │               │ true  ↓
         └─ Step 3 ─► TreeHashMap!("delete", fileMap, fileHash, *ackCh)
                         │             (remove from on-chain registry)
                         ↓
                     ret!(true, fileHash)
```

#### Rholang Delete — Full Example

The client submits this deploy to delete a file they own:

```rholang
new rl(`rho:registry:lookup`), fileRegistryCh,
    deployData(`rho:deploy:data`), deployDataCh,
    authKeyCh, fileHandleCh, deleteCh
in {
  // 1. Resolve FileRegistry from the global registry
  rl!(`rho:id:m6rqma7yas7o6ieos45ai4dskmc6zugs9rmsp6i3zan8qe5hsfqsdt`, *fileRegistryCh) |
  for (@(_, FileRegistry) <- fileRegistryCh) {

    // 2. Get current deployerId from runtime (unforgeable, cannot be spoofed)
    deployData!(*deployDataCh) |
    for (_, deployerId, _ <- deployDataCh) {

      // 3. Obtain the owner AuthKey for this specific file
      //    (Returns Nil if caller is not the owner — deploy fails gracefully)
      FileRegistry!("ownerAuthKey", "<fileHash>", *deployerId, *authKeyCh) |
      for (@authKey <- authKeyCh) {
        if (authKey == Nil) { Nil }  // Not the owner — no-op
        else {
          // 4. Resolve the per-file handle (lookup returns the bundle directly)
          FileRegistry!("lookup", "<fileHash>", *fileHandleCh) |
          for (@fileHandle <- fileHandleCh) {

            // 5. Call delete — gated by AuthKey
            //    Mirrors: vault!("transfer", target, amount, authKey, *ret)
            @fileHandle!("delete", authKey, *deleteCh) |
            for (@result <- deleteCh) {
              // result = (true, fileHash) on success
              //          (false, errorMsg) on failure
              Nil
            }
          }
        }
      }
    }
  }
}
```

### 3.4 gRPC Download

The `downloadFile` gRPC endpoint streams file bytes directly to the client. Downloads are **gated by a finalization check**: the file must be registered in the `FileRegistry` contract at the Last Finalized Block's (LFB) post-state. This prevents downloading files from unfinalized, orphaned, or unpaid uploads.

> **Observer-only**: `downloadFile` may only be called on a read-only (observer) node. Calls to a bonded validator node are immediately rejected with `"File download can only be executed on a read-only f1r3node."` In dev mode (`devMode = true`), validator nodes are also allowed to serve downloads, but the finalization check is **still enforced**.

#### Protobuf Definition

`downloadFile` is declared in `DeployServiceV1.proto` alongside `exploratoryDeploy`:

```protobuf
service DeployService {
  // ... existing RPCs ...
  // Executes deploy as user deploy with immediate rollback and return result
  rpc exploratoryDeploy(ExploratoryDeployQuery) returns (ExploratoryDeployResponse) {}
  // Stream file bytes by content hash. Observer-only — rejected on validator nodes.
  rpc downloadFile(FileDownloadRequest) returns (stream FileDownloadChunk) {}
}

message FileDownloadRequest {
  string fileHash    = 1;  // Blake2b-256 hash
  int64  offset      = 2;  // Resume offset (0 = start)
}

message FileDownloadChunk {
  oneof chunk {
    FileDownloadMetadata metadata = 1;
    bytes data                    = 2;  // 4MB chunks
  }
}

message FileDownloadMetadata {
  string fileHash    = 1;
  int64  fileSize    = 2;
}
```

#### Download Logic

The handler in `DeployGrpcServiceV1.scala` delegates to `FileDownloadAPI.streamFile` with a **finalization checker** callback. Before streaming, the finalization checker executes an `exploratoryDeploy` against the LFB post-state to verify that the file hash is registered in the `FileRegistry` contract. This check is **always enforced** — `devMode` only opens the API to validator nodes but does not bypass finalization:

```scala
// In DeployGrpcServiceV1.scala — finalization checker (always active)
val finalizationChecker: String => Task[Boolean] = { fileHash: String =>
  BlockAPI.exploratoryDeploy[F](
    s"""new return, rl(`rho:registry:lookup`), fileRegistryCh in {
       |  rl!(`rho:id:m6rqma7yas7o6ieos45ai4dskmc6zugs9rmsp6i3zan8qe5hsfqsdt`, *fileRegistryCh) |
       |  for(@(_, FileRegistry) <- fileRegistryCh) {
       |    @FileRegistry!("lookup", "$fileHash", *return)
       |  }
       |}""".stripMargin,
    none[String],  // Use LFB (no specific block hash)
    false,         // Use post-state
    devMode
  ).toTask.map {
    case Right((pars, _)) => pars.nonEmpty && pars.exists(_ != Par())
    case Left(_)          => false
  }
}
```

```
downloadFile(request) {
  // Observer gate: reject if node has a validator key
  if (!isReadOnly && !devMode) → PERMISSION_DENIED

  // Path traversal prevention: strictly validate fileHash format
  if (!request.fileHash.matches("^[a-f0-9]{64}$"))
    → INVALID_ARGUMENT ("Invalid fileHash format")

  filePath = uploadDir / request.fileHash

  if (!filePath.exists)
    → NOT_FOUND

  // Finalization gate: query FileRegistry at LFB post-state
  if (!finalizationChecker(fileHash)) → NOT_FOUND ("file not found")

  → stream file from offset
}
```

The finalization check adds ~5-50ms of latency per download request (Rholang VM evaluation), which is negligible relative to the file transfer time (e.g., ~80 seconds for a 10GB file at 1Gbps).

#### Resume Support

The `offset` field enables interrupted download resumption. Client tracks bytes received locally and reconnects with `offset=bytesReceived`.

#### Rate Limiting

- `max-concurrent-downloads-per-ip` (default: 4) — prevents resource exhaustion
- Only files present on the local filesystem and registered in a finalized block are served
- Only available on observer (read-only) nodes

### 3.5 File Deletion

#### Deletion Flow

```text
Owner submits deploy:
  FileRegistry!("ownerAuthKey", fileHash, deployerId, *authKeyCh)
  fileHandle!("delete", callerPubKey, authKey, *ret)
        │
        ▼
┌──────────────────────────────────────┐
│ FileRegistry._deleteTemplate         │
│  1. AuthKey.check (identity gate)    │
│  2. Remove callerPubKey from array   │
└──────────┬───────────────────────────┘
           │
     Array empty?
      │        │
     No       Yes
      │        │
      ▼        ▼
┌─────────┐  ┌────────────────────────────────────┐
│ Return  │  │ Read _sysAuthTokenCh (private)     │
└─────────┘  │ rho:io:file!("delete", hash,       │
             │              sysAuthToken, *ret)   │
             │ TreeHashMap.delete (on-chain)      │
             │ State channel cleanup              │
             └────────────────────────────────────┘
```
           │
           ▼
┌────────────────────────────────────┐
│ Once block is finalized:           │
│   - Physical file deleted from     │
│     file-replication/ directory    │
│   - download → NOT_FOUND           │
└────────────────────────────────────┘
```

Once a block containing a `delete` deploy is **finalized**, the physical file (`<hash>`) is permanently removed from the `file-replication/` directory. No intermediate on-chain DELETED state is stored. If full historical replay from genesis is later required, archive nodes can be utilized.

### 3.6 Multi-Validator Shard Mode Considerations

The per-file registry map works correctly when multiple validators propose blocks in parallel and the Casper protocol merges them.

#### Conflict Analysis

Per-deployer channels provide natural conflict isolation between different uploaders:

| Operation | Conflict Potential | Explanation |
|-----------|-------------------|-------------|
| **lookup** (read) | ✅ None | Uses `<<-` (peek) — non-consuming read. Multiple validators can read the state in parallel |
| **register** (different files) | ✅ None | Different files use different state channels. Zero contention. |
| **register** (same file, concurrent) | ⚠️ Conflict | Two users uploading same duplicate file at exact same time -> modifies same `fileState` array. Casper's cost-optimal rejection resolves this deterministically |
| **delete** (same file) | ⚠️ Conflict | Modifies the same `fileState` array. First-wins semantics |

#### Deterministic Replay Safety

The `FileSystemProcess` system process performs **non-deterministic I/O** (disk reads). In shard mode, when Validator B replays a block proposed by Validator A:

1. **Register**: Deterministic — file sync (Layer 2) guarantees the file is on disk before replay begins
2. **Delete**: Deterministic — only modifies on-chain state (deployer's array inside the file state in RSpace). Physical file is removed asynchronously after finalization.

The key invariant: **all system process operations either read from verified-identical files (file sync) or from on-chain state (RSpace replay). No operation depends on node-local non-deterministic state.**

#### Parallel Block Scenario

```
Validator A proposes Block B1:
  - Deploy 1: register(fileX) by DeployerAlice

Validator B proposes Block B2 (in parallel):
  - Deploy 2: register(fileW) by DeployerBob   ← no conflict (different deployer channel)
  - Deploy 3: delete(fileY)  by DeployerAlice   ← conflict potential if Alice also in B1

Casper Merge Resolution:
  - fileX registration (Alice): ✅ included
  - fileW registration (Bob):   ✅ included (different channel)
  - fileY deletion (Alice):     ⚠️ resolved by Casper cost-optimal ordering
```

### 3.7 When Can the Client Download the File?

The file is downloadable **only after the block containing the `register` deploy is finalized**. The `downloadFile` endpoint enforces this by querying the `FileRegistry` contract at the Last Finalized Block's post-state via an `exploratoryDeploy`. If the file hash is not registered in the finalized state, the download is rejected with `PERMISSION_DENIED`.

Client workflow:

1. `uploadFile()` → receive `(fileHash, deployId)`
2. Poll `findDeploy(deployId)` — when it returns a `LightBlockInfo`, the deploy is in a block
3. Poll `isFinalized(blockHash)` — once `true`, the file is permanently available for download
4. Call `downloadFile(fileHash)` via gRPC to stream the file

### 3.8 Module Changes for Layer 3

| Module | File(s) | Change Type | Description |
|--------|---------|-------------|-------------|
| `rholang` | `SystemProcesses.scala` | **MODIFY** | `fileRegister` and `fileDelete` system process handlers for `rho:io:file` channel (charges storage phlo, blind physical disk delete — requires `SysAuthToken`); registers URN map entry |
| `rholang` | `Runtime.scala` | **MODIFY** | Wire `fileReplicationDir` into system processes |
| `casper` | `FileRegistry.rho` | **NEW** | On-chain ownership contract (`fileHash -> { deployers: [...] }`), array reference counting for deduplication, AuthKey-gated delete, `SysAuthToken` security gate |
| `models` | `DeployServiceV1.proto` | **MODIFY** | `downloadFile` RPC (observer-only, alongside `exploratoryDeploy`), `FileDownloadRequest`, `FileDownloadChunk`, `FileDownloadMetadata` |
| `node` | `DeployGrpcServiceV1.scala` | **MODIFY** | `downloadFile` handler — delegates to `BlockAPI.downloadFile` (same pattern as `exploratoryDeploy`) |
| `node` | `BlockAPI.scala` | **MODIFY** | `downloadFile` (observer gate via `casper.getValidator.map(_.isEmpty)`), `getDeployerHashList(pubKey)` queries deployer channel via RSpace |

### 3.9 Scaling Limits

| Resource | Limit | Mitigation |
|----------|-------|------------|
| **Disk** | Physical NVMe/SSD only | Files in `<data-dir>/file-replication/`, separate from LMDB |
| **LMDB MapSize** | Bypassed | Only per-file registry state in RSpace (not file bytes) |
| **File count** | OS inode/ulimit | XFS/ext4 with large inode ratio; `df -i` monitoring |
| **Download bandwidth** | Per-IP rate limit | `max-concurrent-downloads-per-ip` (default: 4) |
| **Registry size** | RSpace channel capacity | One channel per deployer; scales with number of unique uploaders |

---

## Layer 4: DA-Optimistic Consensus

Standard Casper finalization does not account for Data Availability. A block might be valid by consensus rules but unexecutable because a referenced file hasn't propagated yet. Layer 4 ensures validators only build on blocks whose file data they possess.

### 4.1 The Problem with Standard Casper + Large Files

Current Casper finalization works as follows:

1. Validators build blocks on top of each other
2. When sufficient stake (>50% by default) has transitively built upon a block, it is **finalized**
3. Finalization is purely DAG-based — it does not check whether validators actually *have* all the data needed to execute the block

**Failure scenario without DA-awareness:**
```
1. Validator A uploads 10GB file, proposes Block B1
2. Validators B and C build on B1 (they haven't downloaded the file yet)
3. B1 is finalized (>50% stake built on it)
4. Validators B and C try to validate B1 — but the 10GB file download takes 30 minutes
5. During those 30 minutes, B and C cannot validate any blocks that depend on B1
6. Network stalls
```

### 4.2 Solution: DA-Optimistic Validation

Casper finalization proceeds normally, but block **validation** is gated by data availability. A validator will not mark a block as "validated" until all referenced files are downloaded and verified.

- **Key invariant**: A validator only builds on blocks it has **fully validated** (including file availability). This means a validator's "latest message" always implies DA satisfaction.
- **Finalization implication**: Since finalization requires multiple validators to build on a block, and building requires DA, finalization **naturally** implies DA across the participating validators.

```
Validator A proposes B1 (with 10GB file)
  │
  ├─► Validator B: receives B1, starts downloading file
  │                Does NOT build on B1 until file verified ◄── KEY CHANGE
  │
  ├─► Validator C: receives B1, starts downloading file
  │                Does NOT build on B1 until file verified
  │
  │   ... (file downloads complete, e.g. 5 minutes) ...
  │
  ├─► Validator B: file verified → validates B1 → builds B2 on B1
  ├─► Validator C: file verified → validates B1 → builds B3 on B2
  │
  └─► B1 is finalized (B+C have built on it, >50% stake)
      At this point, ALL participating validators have the file ✓
```

**Impact on finalization speed:**

- **Without file uploads**: Zero impact. Casper proceeds exactly as today.
- **With file uploads**: Finalization is delayed by the time it takes for ≥50% of validator stake to download and verify the file. For a 10GB file on 1Gbps links (~80 seconds transfer), finalization adds ~80-120 seconds.

#### Configuration (`defaults.conf`)

File-sync timeout and DA backpressure settings are defined in the top-level `file-upload` section (not a nested `consensus.da` block):

```hocon
file-upload {
  # Maximum time to wait for P2P file transfers to complete before rejecting a block.
  # Tune based on your largest expected file and slowest network link:
  #   6 GB @ 100 Mbps  ≈  8 min → 30 minutes is safe
  #   6 GB @  10 Mbps  ≈ 80 min → 2 hours needed
  #  10 GB @  10 Mbps  ≈ 2.2 hr → increase to 3 hours
  file-sync-timeout = 2 hours

  # DA consensus: maximum total referenced file size (bytes) across
  # all file-registration deploys in a single block.
  max-file-data-size-per-block = 53687091200  # 50 GB

  # DA consensus: maximum number of file-registration deploys
  # allowed in a single block.
  max-file-deploys-per-block = 10
}
```

### 4.3 Deploy Selection Backpressure for File Deploys

The existing `maxDeployDataSizePerBlock` backpressure (default 64MB, hard cap 64MB) applies to the **Rholang term size**, not the underlying file size. A file registration deploy's term is only ~100 bytes regardless of file size.

However, we add a new backpressure dimension:

```hocon
f1r3fly {
  casper {
    # Maximum total referenced file size per block
    max-file-data-size-per-block = 50G

    # Maximum number of file registration deploys per block
    max-file-deploys-per-block = 10
  }
}
```

This prevents a single block from referencing so many large files that validators spend hours downloading data before they can validate it.

### 4.4 Module Changes for Layer 4

| Module | File(s) | Change Type | Description |
|--------|---------|-------------|-------------|
| `casper` | `MultiParentCasperImpl.scala` | **MODIFY** | Gate `addBlock` validation on file availability |
| `casper` | `Proposer.scala` | **MODIFY** | Only select parent blocks that have passed DA validation |
| `casper` | `CasperConf.scala` / `defaults.conf` | **MODIFY** | Add `consensus.da.*` config |
| `casper` | `BlockCreator.scala` | **MODIFY** | Add file-aware deploy selection limits |

---

## Layer 5: Storage-Proportional Cost Accounting

File uploads introduce significant real-world costs (disk storage, replication bandwidth, permanent occupancy) that standard Rholang execution phlo does not cover. Layer 5 ensures validators on the public shard are compensated proportionally to the data they store.

### 5.1 Cost Model

The total phlo cost of a file upload is:

```
totalPhlo = BASE_REGISTER_PHLO + (fileSize × phloPerStorageByte)
```

| Component | Formula | Example (10GB file) |
|-----------|---------|---------------------|
| **Base register cost** | Fixed: `300` phlo | 300 phlo |
| **Storage cost** | `fileSize × phloPerStorageByte` | 10,737,418,240 × 1 = ~10.7B phlo |
| **Total phlo required** | Sum | ~10.7B phlo |
| **Total cost to client** | `totalPhlo × phloPrice` | `10.7B × phloPrice` tokens |

The `phloPrice` (set by the client in `FileUploadMetadata`) determines the token cost per phlo unit, same as standard deploys. Validators prioritize higher-`phloPrice` file deploys when selecting deploys for a block (FIFO with price priority).

### 5.2 Configuration

All file-upload settings live in the top-level `file-upload` section of `defaults.conf`:

```hocon
file-upload {
  # 16 MB chunks for P2P file sync (default was 4 MB)
  chunk-size = 16777216

  # Sub-directory under data-dir where uploaded files are stored
  replication-dir = "file-replication"

  # Cost per byte of file stored (in phlo units)
  phlo-per-storage-byte = 1

  # Fixed base cost for the register operation (in phlo units)
  base-register-phlo = 300

  # Maximum simultaneous downloads from a single IP
  max-concurrent-downloads-per-ip = 4

  # Maximum time to wait for P2P file transfers before rejecting a block
  file-sync-timeout = 2 hours

  # Maximum size of a single uploaded file (bytes). Default: 10 GB
  max-file-size = 10737418240

  # Maximum entries in the per-IP download rate-limiter LRU cache
  max-download-cache-entries = 10000

  # DA consensus: max total referenced file size per block
  max-file-data-size-per-block = 53687091200  # 50 GB

  # DA consensus: max file-registration deploys per block
  max-file-deploys-per-block = 10
}
```

### 5.3 Enforcement Points

Cost is enforced at **two** points for defense-in-depth:

#### Point 1: Upload-Time Validation (Early Rejection)

In `FileUploadAPI.scala`, **before** the synthetic deploy is submitted to the mempool:

```scala
val totalRequired = config.baseRegisterPhlo + (fileSize * config.phloPerStorageByte)
if (metadata.phloLimit < totalRequired) {
  // Reject immediately — file is deleted, no deploy created
  cleanup(tmpFile)
  return Error(s"Insufficient phloLimit. Required: $totalRequired for $fileSize bytes")
}
```

This ensures the client gets fast feedback without wasting mempool space.

#### Point 2: Block Execution Validation (Consensus Enforcement)

During block execution, the `rho:io:file` `register` system process charges phlo through the standard Rholang cost accounting (`CostAccounting.charge()`):

```scala
// In SystemProcesses.scala — fileRegister handler
// Signature verification was removed (commit 4465d78); the system process
// trusts that the deploy was already validated by Layer 1 admission control.
def fileRegister(hash: String, size: Long, fileName: String): F[Par] = {
  val storageCost = Cost(size * phloPerStorageByte)

  for {
    // 1. Charge proportional storage costs (deducts from deploy's phloLimit)
    _ <- costAccounting.charge(storageCost)
    
    // 2. System process produces delegation data to FILE_REGISTRY_NOTIFY channel;
    //    FileRegistry.rho consumes it and records the on-chain registration.
    _ <- produceToFileRegistryNotify(hash, fileName, context.deployer, sysAuthToken) 
  } yield Par()
}
```

If the deploy runs out of phlo during the `charge` call, it fails with `OutOfPhloError` — the standard Rholang behavior. The file registration is rolled back, and the deploy is recorded as failed.

### 5.4 Cost Breakdown Visibility

The `FileUploadResult` includes cost information so clients can audit charges (see Layer 1 protobuf definition for full message).

### 5.5 What Download Costs

Downloads are **free** — no phlo is charged. The uploader's one-time storage fee covers the cost of making the file available to all nodes and clients.

| Operation | Cost | Payer |
|-----------|------|-------|
| `uploadFile` (10GB) | `300 + 10GB × phloPerByte` phlo | Uploader |
| `downloadFile` | Free | — |
| P2P replication | Free (validator obligation) | — |

### 5.6 Module Changes for Layer 5

| Module | File(s) | Change Type | Description |
|--------|---------|-------------|-------------|
| `node` | `FileUploadAPI.scala` | **MODIFY** | Add phlo cost validation before deploy submission |
| `node` | `FileUploadCosts.scala` | **NEW** | Constants and cost calculation logic (`BASE_REGISTER_PHLO`, `phloPerStorageByte`) |
| `node` | `Options.scala` | **MODIFY** | Add `--file-upload-phlo-per-storage-byte` CLI flag |
| `node` | `defaults.conf` | **MODIFY** | Add `file-upload.phlo-per-storage-byte` and `file-upload.base-register-phlo` |
| `rholang` | `SystemProcesses.scala` | **MODIFY** | Charge storage phlo via `CostAccounting.charge()` during `register` |
| `models` | `DeployServiceV1.proto` | **MODIFY** | Add `storagePhloCost` and `totalPhloCharged` fields to `FileUploadResult` |

---

## Module Summary — Full Change Inventory

### New Modules/Files

| File | Module | Purpose |
|------|--------|---------|
| `FileUploadAPI.scala` | `node` | Core upload logic: O(1) streaming state machine, hashing, synthetic deploy |
| `FileMetadata.scala` | `node` | File metadata data class and JSON persistence |
| `FileUploadCosts.scala` | `node` | Storage cost constants and calculation logic |
| `FileDownloadAPI.scala` | `node` | File download streaming logic with finalization check |
| `FileReplicationSetup.scala` | `casper` | DRY helper that wires `FileRequester` + `OrphanFileCleanup` into each engine state |
| `SystemProcesses.scala (fileRegister/fileDelete)` | `rholang` | System process handlers: register (charges phlo), delete (blind physical delete) |

### Modified Modules/Files

| File | Module | Key Changes |
|------|--------|-------------|
| `DeployServiceV1.proto` | `models` | `uploadFile`/`downloadFile` RPCs (both in `DeployService`; `downloadFile` is observer-only), file messages, `FileUploadResult` cost fields |
| `DeployGrpcServiceV1.scala` | `node` | Upload handler; `downloadFile` handler with `checkFileFinalized()` (delegates to `FileDownloadAPI`, observer-only gate) |
| `BlockAPI.scala` | `node` | `uploadFile` + `getDeployerHashList` methods |
| `Options.scala` | `node` | File upload/download CLI flags, `phlo-per-storage-byte` |
| `defaults.conf` | `node` | Consolidated `file-upload` config section (chunk-size, replication-dir, costs, rate-limits, DA backpressure, `file-sync-timeout`, `max-file-size`, `max-download-cache-entries`) |
| `model.scala` | `node` | `FileUploadConf` and `FileConf` case classes grouping file-related configuration |
| `CommMessages.scala` | `comm` | `FileRequest`, `FilePacket` P2P messages |
| `TransportLayer.scala` | `comm` | File request/response streaming handler |
| `MultiParentCasperImpl.scala` | `casper` | DA-gated block validation, file sync before execution; `deployLifespan = 250` (from 50) to prevent OrphanFileCleanup race during large P2P replication |
| `Proposer.scala` | `casper` | DA-aware parent selection |
| `Validate.scala` | `casper` | File availability validation check |
| `BlockCreator.scala` | `casper` | File-aware deploy selection backpressure (uses `foldLeft` instead of mutable `var`) |
| `CasperConf.scala` | `casper` | DA config; `FileConf` nested in `CasperShardConf` |
| `OrphanFileCleanup.scala` | `casper` | Tightened `FileHashPattern` regex; cleans up `.part` files on timeout |
| `SystemProcesses.scala` | `rholang` | Register `rho:io:file` channel; `fileRegister` no longer requires `nodeSigHex` |
| `Runtime.scala` | `rholang` | Wire file dir + `phloPerStorageByte` into system processes via `ProcessContext` |
| `NodeRuntime.scala` | `node` | Pass config to runtime |

---

## Verification Plan

### Automated Tests

1. **Unit: `FileUploadAPISpec`** — Chunk streaming, atomic `.tmp` cleanup on interruption, Blake2b hashing, synthetic deploy generation, `deployId` return.
2. **Unit: `FileUploadCostSpec`** — Verify phlo cost calculation: `base + fileSize × phloPerByte`. Verify rejection when `phloLimit` is insufficient. Verify cost fields in `FileUploadResult`.
3. **Unit: `FileSystemProcessSpec`** — Verify `register` charges storage phlo via `CostAccounting`. Verify ownership check on delete.
4. **Integration: `FileReplicationSpec`** — Two `TestNode` instances: upload on A, verify B fetches the file when processing A's block. Inject corruption, verify block rejection.
5. **Integration: `DAGateSpec`** — Propose block with file reference; verify block is held until file arrives; verify block executes after file download.
6. **Integration: `ConsensusDASpec`** — Three-validator network: verify DA-optimistic consensus prevents finalization until ≥2 validators have the file.
7. **Integration: `StorageCostEnforcementSpec`** — Upload file with insufficient `phloLimit` → verify rejection. Upload with sufficient `phloLimit` → verify deploy executes and phlo is deducted.
8. **Integration: `FileDeletionSpec`** — Upload file, verify readable. Owner deletes. Verify hash removed from deployer list. Verify physical file deleted after finalization. Verify `downloadFile` returns `NOT_FOUND`.
9. **Integration: `MultiValidatorRegistrySpec`** — Three-validator shard: concurrent file registrations by different deployers merge without conflict (different files) or resolve deterministically via Casper (same file). Verify file registry state consistency across all validators after merge.

### Manual Verification

1. Start a 3-node Docker network (Validator A, B, Observer C)
2. Upload a 100MB file to Validator A via `grpcurl` streaming
3. Verify `FileUploadResponse` returns `fileHash` + `deployId` + `storagePhloCost`
4. Call `findDeploy(deployId)` — verify it returns a `LightBlockInfo` once the deploy is in a block. Call `isFinalized(blockHash)` — verify it returns `true` after finalization.
5. Verify Validator B logs show: file download from A, Blake2b verification, block execution
6. Download file from Observer C via `downloadFile` gRPC — verify bytes match. Attempt the same call on Validator A or B — verify it returns `"File download can only be executed on a read-only RNode."`
7. Delete the file (owner deploy). Verify caller's ID is removed from the deployers array. Verify physical file is deleted after finalization. Verify `downloadFile` returns `NOT_FOUND`.
8. Repeat with 10GB file on a high-bandwidth test network — verify end-to-end finalization within target time

---

## Client Implementation Guide (General)

To integrate F1R3FLY's streaming file capabilities into a client application (e.g., Rust, Java, Python), follow these general gRPC flow patterns.

> **Key constant**: The `FileRegistry` contract lives at a fixed on-chain address:
> ```
> rho:id:m6rqma7yas7o6ieos45ai4dskmc6zugs9rmsp6i3zan8qe5hsfqsdt
> ```
> This URI is derived from the public key used to register the contract at genesis (see `FileRegistry.rho` header). It is **the same for all files, all deployers, and all nodes** — it is the contract address, not a file identifier. You never need to generate a new one.

### 1. Uploading a File (`uploadFile`)

Uploading a file requires generating a synthetic deploy on the client side, then opening a client-streaming gRPC call to `uploadFile`.

**Steps:**
1. **Compute File Identity**: Calculate the total bytes of the file (`fileSize`) and its Blake2b-256 hash (`fileHash`). File size must not exceed `max-file-size` (default 10 GB).
2. **Construct Rholang Term**: Construct the exact Rholang execution term that the node will place on-chain.
   ```rholang
   new ret, file(`rho:io:file`) in {
     file!("register", "<fileHash>", <fileSize>, "<fileName>", *ret)
   }
   ```
3. **Sign the Deploy**: Use the standard `DeployData` signing process for your language SDK to sign the term. This yields the public key (`deployer`), timestamp, signature (`sig`), and algorithm (`sigAlgorithm`).
4. **Construct `FileUploadMetadata`**: Map the deploy data and file properties into the initialization message.
   - Set `phloPrice`, `phloLimit`, `validAfterBlockNumber`, and `shardId`. Ensure `phloLimit` covers both `baseRegisterPhlo` (default 300) and `fileSize * phloPerStorageByte` (default 1).
   - Set `fileHash`, `fileSize`, `fileName`, and the plain-text `term`.
5. **Open gRPC Stream (`DeployService.uploadFile`)**:
   - **First Chunk**: Must contain solely the `FileUploadMetadata` as the `metadata` oneof field. Send this chunk immediately.
   - **Data Chunks**: Read the source file sequentially and send it in 1MB to 4MB chunks using the `data` oneof field.
6. **Complete Stream**: Close the sending side of the stream.
7. **Read Response**: The node will return a `FileUploadResponse`. On success, it contains `FileUploadResult` with the `fileHash` and the generated tracking `deployId`.

#### Python Example (using `f1r3fly` SDK)

```python
from f1r3fly.client import F1r3flyClient
from f1r3fly.util import blake2b_256_hex, create_file_upload_metadata

data = open("my-file.bin", "rb").read()
file_hash = blake2b_256_hex(data)
file_size = len(data)

metadata = create_file_upload_metadata(
    key=private_key,
    file_hash=file_hash,
    file_size=file_size,
    file_name="my-file.bin",
    phlo_price=1,
    phlo_limit=500_000_000,
    valid_after_block_no=current_block_number - 1,
    shard_id="root",
)

with F1r3flyClient(host, grpc_port) as client:
    result = client.upload_file(metadata, data)
    # result.fileHash  → Blake2b-256 content hash
    # result.deployId  → deploy signature for tracking
```

### 2. Tracking a Deploy (Finding the Block)

After uploading (or deleting), you receive a `deployId`. Use it to find which block contains the deploy and whether it has been finalized.

**Steps:**
1. Poll `DeployService.findDeploy(deployId)` until it returns a `LightBlockInfo` — this means the deploy has been included in a block.
2. Extract `blockHash` from the returned `LightBlockInfo`.
3. Poll `DeployService.isFinalized(blockHash)` until it returns `true` — this means the block is permanently committed.

> **Why does this matter?** Files are only downloadable after the block containing the `register` deploy is finalized. Delete operations also only take physical effect after finalization.

#### Python Example

```python
import time
from f1r3fly.client import F1r3flyClient, F1r3flyClientException

def wait_for_deploy_in_block(client, deploy_id, timeout=120):
    """Poll findDeploy until the deploy is included in a block."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            block_info = client.find_deploy(deploy_id)
            return block_info  # has .blockHash, .blockNumber, etc.
        except F1r3flyClientException:
            time.sleep(3)
    raise TimeoutError(f"Deploy not found in a block within {timeout}s")

def wait_for_finalization(client, block_hash, timeout=120):
    """Poll isFinalized until the block is finalized."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.is_finalized(block_hash):
            return True
        time.sleep(3)
    raise TimeoutError(f"Block not finalized within {timeout}s")

# Usage after upload:
with F1r3flyClient(host, grpc_port) as client:
    block_info = wait_for_deploy_in_block(client, result.deployId)
    wait_for_finalization(client, block_info.blockHash)
    # File is now downloadable!
```

### 3. Downloading a File (`downloadFile`)

Downloading works solely on *read-only (observer) nodes*. It is a server-streaming gRPC call. The file must be registered in a finalized block.

**Steps:**
1. **Open connection** to the observer node's external gRPC port.
2. **Send `FileDownloadRequest`**: Provide the `fileHash`. For interrupted downloads, you can provide an `offset` (resume byte index).
3. **Consume Response Stream (`stream FileDownloadChunk`)**:
   - Iterate over the incoming stream.
   - The first message may contain `metadata` (file total size/hash).
   - Subsequent messages contain `data` chunks (up to 4MB). Append these chunks locally to a file or stream.
4. **Completion**: The stream ends cleanly when all bytes are sent.

#### Python Example

```python
from f1r3fly.pb.DeployServiceV1_pb2 import FileDownloadRequest

with F1r3flyClient(observer_host, observer_grpc_port) as client:
    request = FileDownloadRequest(fileHash=file_hash, offset=0)
    response_stream = client._deploy_stub.downloadFile(request, timeout=600)

    chunks = []
    for chunk in response_stream:
        which = chunk.WhichOneof('chunk')
        if which == 'data':
            chunks.append(chunk.data)

    downloaded_bytes = b''.join(chunks)
```

### 4. Deleting a File (`doDeploy`)

File deletion uses the standard `doDeploy` gRPC endpoint — there is no special file deletion RPC. You submit a Rholang script that:

1. **Looks up** the `FileRegistry` contract from the on-chain registry (using its fixed URI)
2. **Obtains** your owner `AuthKey` (proves you are an owner of the file)
3. **Calls** `delete` on the file handle

> **Important:** Only the file's owner can delete it. If Alice uploaded a file, only a deploy signed by Alice's key can delete it. If multiple users uploaded the same file (deduplication), each user can remove their own reference. The physical file is only deleted after **all** owners have removed their references.

#### Full Rholang Delete Script

This is the exact script used by the integration tests (substitute `<fileHash>` with your file's Blake2b-256 hash):

```rholang
new rl(`rho:registry:lookup`), fileRegistryCh, deployData(`rho:deploy:data`), deployDataCh,
    authKeyCh, fileHandleCh, deleteCh, deployerIdOps(`rho:system:deployerId:ops`), pubKeyCh,
    stdout(`rho:io:stdout`)
in {
  // 1. Resolve FileRegistry from the on-chain registry (FIXED address — same for all files)
  rl!(`rho:id:m6rqma7yas7o6ieos45ai4dskmc6zugs9rmsp6i3zan8qe5hsfqsdt`, *fileRegistryCh) |
  for (@(_, FileRegistry) <- fileRegistryCh) {

    // 2. Get deploy context (unforgeable, bound by runtime to current deployer)
    deployData!(*deployDataCh) |
    for (_, deployerId, _ <- deployDataCh) {
      deployerIdOps!("pubKeyBytes", *deployerId, *pubKeyCh) |
      for (@pubKeyBytes <- pubKeyCh) {

        // 3. Obtain owner AuthKey (returns Nil if caller is not an owner)
        @FileRegistry!("ownerAuthKey", "<fileHash>", *deployerId, *authKeyCh) |
        for (@authKey <- authKeyCh) {
          if (authKey != Nil) {

            // 4. Look up the per-file handle
            @FileRegistry!("lookup", "<fileHash>", *fileHandleCh) |
            for (@fileHandle <- fileHandleCh) {
              if (fileHandle != Nil) {

                // 5. Delete — gated by AuthKey
                @fileHandle!("delete", pubKeyBytes, authKey, *deleteCh) |
                for (@deleteResult <- deleteCh) {
                  // deleteResult = (true, fileHash) on success
                  //                (false, errorMsg) on failure
                  stdout!(("DELETE_RESULT", deleteResult))
                }
              }
            }
          }
        }
      }
    }
  }
}
```

#### Python Example

```python
def make_delete_script(file_hash: str) -> str:
    """Generate a Rholang script that deletes a file via the FileRegistry contract."""
    return f"""
new rl(`rho:registry:lookup`), fileRegistryCh, deployData(`rho:deploy:data`), deployDataCh,
    authKeyCh, fileHandleCh, deleteCh, deployerIdOps(`rho:system:deployerId:ops`), pubKeyCh,
    stdout(`rho:io:stdout`)
in {{
  rl!(`rho:id:m6rqma7yas7o6ieos45ai4dskmc6zugs9rmsp6i3zan8qe5hsfqsdt`, *fileRegistryCh) |
  for (@(_, FileRegistry) <- fileRegistryCh) {{
    deployData!(*deployDataCh) |
    for (_, deployerId, _ <- deployDataCh) {{
      deployerIdOps!("pubKeyBytes", *deployerId, *pubKeyCh) |
      for (@pubKeyBytes <- pubKeyCh) {{
        @FileRegistry!("ownerAuthKey", "{file_hash}", *deployerId, *authKeyCh) |
        for (@authKey <- authKeyCh) {{
          if (authKey != Nil) {{
            @FileRegistry!("lookup", "{file_hash}", *fileHandleCh) |
            for (@fileHandle <- fileHandleCh) {{
              if (fileHandle != Nil) {{
                @fileHandle!("delete", pubKeyBytes, authKey, *deleteCh) |
                for (@deleteResult <- deleteCh) {{
                  stdout!(("DELETE_RESULT", deleteResult))
                }}
              }}
            }}
          }}
        }}
      }}
    }}
  }}
}}
"""

# Submit the delete deploy via the standard doDeploy endpoint
rho_script = make_delete_script(file_hash)
deploy_id = node.deploy_string(rho_script, owner_private_key, phlo_limit=500_000_000)

# Track the delete deploy the same way as an upload deploy
block_info = wait_for_deploy_in_block(client, deploy_id)
wait_for_finalization(client, block_info.blockHash)
# File is now deleted — downloadFile will return NOT_FOUND
```

### 5. Quick Reference — Full Lifecycle

| Step | Operation | gRPC Method | Key Input | Key Output |
|------|-----------|-------------|-----------|------------|
| 1 | Upload | `uploadFile` (client-streaming) | File bytes + metadata | `fileHash`, `deployId` |
| 2 | Track upload | `findDeploy` → `isFinalized` | `deployId` → `blockHash` | Block inclusion + finality |
| 3 | Download | `downloadFile` (server-streaming, **observer-only**) | `fileHash` | File bytes |
| 4 | Delete | `doDeploy` (standard deploy) | Rholang delete script | `deployId` |
| 5 | Track delete | `findDeploy` → `isFinalized` | `deployId` → `blockHash` | Physical file removed |
