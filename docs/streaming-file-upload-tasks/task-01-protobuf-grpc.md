# Task 1: Protobuf & gRPC Definitions

## Architecture Reference

- [Layer 1 §1.1 — gRPC Streaming Protocol](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L46-L95)
- [Layer 3 §3.4 — gRPC Download](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L470-L506)

---

## What to Implement

Define all Protobuf messages and gRPC service extensions needed by the upload and download flows. This task produces the compiled stubs that Tasks 2 and 8 consume.

### Technical Details

**File**: `models/src/main/protobuf/DeployServiceV1.proto`

#### Upload Messages

1. **`FileUploadMetadata`** — first message in the upload stream  
   - Fields: `deployer` (bytes), `timestamp` (int64), `sig` (bytes), `sigAlgorithm` (string), `phloPrice` (int64), `phloLimit` (int64), `validAfterBlockNumber` (int64), `shardId` (string), `fileName` (string), `fileSize` (int64), `fileHash` (string), `term` (string)

2. **`FileUploadChunk`** — `oneof chunk { FileUploadMetadata metadata = 1; bytes data = 2; }`

3. **`FileUploadResult`** — `fileHash` (string), `deployId` (string), `storagePhloCost` (int64), `totalPhloCharged` (int64)

4. **`FileUploadResponse`** — `oneof message { ServiceError error = 1; FileUploadResult result = 2; }`

#### Download Messages

5. **`FileDownloadRequest`** — `fileHash` (string), `offset` (int64)

6. **`FileDownloadMetadata`** — `fileHash` (string), `fileSize` (int64)

7. **`FileDownloadChunk`** — `oneof chunk { FileDownloadMetadata metadata = 1; bytes data = 2; }`

#### Service Extension

8. Add to `DeployService`:
   ```protobuf
   rpc uploadFile(stream FileUploadChunk) returns (FileUploadResponse) {}
   rpc downloadFile(FileDownloadRequest) returns (stream FileDownloadChunk) {}
   ```

#### P2P Messages (for Task 5)

9. **`FileRequest`** — `fileHash` (bytes)
10. **`FilePacket`** — `oneof content { int64 fileSize = 1; bytes data = 2; }`
    - File: `comm` module protobuf (alongside `CommMessages`)

---

## Verification

### Compile Check (automated — run when task is done)

```bash
# Must pass — proves all proto definitions are syntactically correct and stubs generate
sbt models/compile
sbt comm/compile
```

### New Test: Proto Round-Trip

**File**: `models/src/test/scala/coop/rchain/models/FileUploadProtoSpec.scala`

```scala
class FileUploadProtoSpec extends FlatSpec with Matchers {
  // Construct each message, serialize to bytes, deserialize, assert field equality
  "FileUploadMetadata" should "round-trip through protobuf serialization" in { ... }
  "FileUploadChunk" should "correctly encode oneof metadata variant" in { ... }
  "FileUploadChunk" should "correctly encode oneof data variant" in { ... }
  "FileUploadResult" should "preserve cost fields" in { ... }
  "FileDownloadRequest" should "preserve offset field" in { ... }
}
```

```bash
sbt 'models/testOnly coop.rchain.models.FileUploadProtoSpec'
```

### Manual Verification

- [ ] Inspect generated stubs in `models/target/` — confirm all message classes exist
- [ ] Verify field numbers and types match [architecture doc §1.1](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L52-L91) exactly

---

## Subtasks

- [ ] Add `FileUploadMetadata` message to `DeployServiceV1.proto`
- [ ] Add `FileUploadChunk` message with `oneof`
- [ ] Add `FileUploadResult` message (with cost fields)
- [ ] Add `FileUploadResponse` message
- [ ] Add `FileDownloadRequest` message
- [ ] Add `FileDownloadMetadata` message
- [ ] Add `FileDownloadChunk` message with `oneof`
- [ ] Add `uploadFile` RPC to `DeployService`
- [ ] Add `downloadFile` RPC to `DeployService`
- [ ] Add `FileRequest` / `FilePacket` to `comm` module protobuf
- [ ] Verify `sbt compile` for `models` and `comm`
- [ ] Round-trip serialization unit test
