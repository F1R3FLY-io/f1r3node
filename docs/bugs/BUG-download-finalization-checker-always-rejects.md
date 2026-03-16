# BUG: FileDownloadAPI finalization checker always rejects — `FileRegistry!("lookup")` returns Nil

**Filed:** 2026-03-15  
**Severity:** Critical — all file downloads are broken  
**Component:** `FileDownloadAPI` / `FileRegistry.rho` / genesis contracts  
**Affected:** All nodes (validators in dev-mode + readonly observers)

---

## Summary

`FileDownloadAPI.streamFile()` gates every download behind a **finalization check** that runs an `exploratoryDeploy` to query `FileRegistry!("lookup", hash)` on the Last Finalized Block's post-state. This check **always returns `false`**, causing every download to be rejected with:

```
[FileDownloadAPI] Rejected: file not in finalized registry, hash=a581c6ffa5...
```

The file IS physically present on disk (upload succeeds, synthetic deploy is included in a finalized block), but the on-chain `FileRegistry` TreeHashMap lookup returns an empty `Par` (Nil).

## Reproduction

1. Start a shard with `docker-compose.scala.yml`
2. Upload a file via gRPC `uploadFile` — succeeds, returns `fileHash` + `deployId`
3. Wait for deploy to be included in a block (`findDeploy` succeeds)
4. Wait for block finalization (`lastFinalizedBlock.blockNumber >= deploy block`)
5. Attempt `downloadFile(fileHash)` — **always rejected**

Even after 200+ blocks and LFB far past the upload block, the download is still rejected.

## Root Cause Analysis

### The finalization checker code

[DeployGrpcServiceV1.scala](file:///home/dev/system-integration/services/f1r3node/node/src/main/scala/coop/rchain/node/api/DeployGrpcServiceV1.scala#L514-L549):

```scala
private def checkFileFinalized(devMode: Boolean): String => Task[Boolean] = {
  fileHash: String =>
    BlockAPI.exploratoryDeploy[F](
      s"""new return, rl(`rho:registry:lookup`), fileRegistryCh in {
         |  rl!(`rho:id:m6rqma7yas7o6ieos45ai4dskmc6zugs9rmsp6i3zan8qe5hsfqsdt`, *fileRegistryCh) |
         |  for(@(_, FileRegistry) <- fileRegistryCh) {
         |    @FileRegistry!("lookup", "$fileHash", *return)
         |  }
         |}""".stripMargin,
      none[String],  // Use LFB
      false,         // Use post-state
      devMode
    ).toTask.map {
      case Right((pars, _)) =>
        val isReg = pars.nonEmpty && pars.exists(p => p != coop.rchain.models.Par())
        isReg
      case Left(err) => false
    }
}
```

### Observed debug output

```
[FileDownloadAPI-Debug] exploratoryDeploy pars size: 1, pars: List(Par(List(),List(),...,false))
[FileDownloadAPI-Debug] isReg resolved to: false
```

The `exploratoryDeploy` returns a **single empty Par** — meaning `rho:registry:lookup` resolved the URI but `FileRegistry!("lookup", hash)` returned `Nil` (no entry in the TreeHashMap).

### Possible causes (most to least likely)

1. **Registry URI mismatch.** The `rho:id:m6rqma7yas7o6...` URI in the checker was pre-computed from a specific key/nonce/timestamp combination (see [FileRegistry.rho header table](file:///home/dev/system-integration/services/f1r3node/casper/src/main/resources/FileRegistry.rho#L1-L15)). If the genesis `insertSigned` call uses different parameters (e.g., the Docker image was built from a different commit), the actual registry URI will differ, and the lookup will silently return Nil.

2. **File registration deploy not reaching FileRegistry.** The upload creates a synthetic deploy with term `rho:io:file!("register", hash, size, name, *ret)`. The `rho:io:file` system process validates the sig, charges phlo, then produces on `rho:io:file:registerNotify`. The `FileRegistry.rho` contract has a persistent listener on `registerNotify` that calls `FileRegistry!("register", ...)`. If the `registerNotify` channel binding didn't happen at genesis (e.g., `FileRegistry.rho` failed to compile or the channel wiring is wrong), the delegation bridge is broken.

3. **TreeHashMap state not persisted across blocks.** The `FileRegistry` uses `TreeHashMap!("init", 3, ...)` at genesis — this creates the map in the genesis block's post-state. If subsequent blocks' post-states don't correctly inherit or merge the TreeHashMap channels, lookups on the LFB post-state may see an empty map.

4. **`exploratoryDeploy` uses wrong state.** The checker calls `exploratoryDeploy` with `none[String]` (LFB) and `usePreStateHash = false` (post-state). If `exploratoryDeploy` evaluates patterns against a state that doesn't include the file registration deploy's effects, the lookup returns empty.

## Files involved

| File | Role |
|------|------|
| [DeployGrpcServiceV1.scala](file:///home/dev/system-integration/services/f1r3node/node/src/main/scala/coop/rchain/node/api/DeployGrpcServiceV1.scala) | `checkFileFinalized` — builds the `exploratoryDeploy` query |
| [FileDownloadAPI.scala](file:///home/dev/system-integration/services/f1r3node/node/src/main/scala/coop/rchain/node/api/FileDownloadAPI.scala) | `streamFile` — calls `finalizationChecker` before streaming |
| [FileUploadAPI.scala](file:///home/dev/system-integration/services/f1r3node/node/src/main/scala/coop/rchain/node/api/FileUploadAPI.scala) | `SyntheticDeploy` — builds the deploy proto from upload metadata |
| [FileRegistry.rho](file:///home/dev/system-integration/services/f1r3node/casper/src/main/resources/FileRegistry.rho) | On-chain registry contract (genesis deploy) |
| [SystemProcesses.scala](file:///home/dev/system-integration/services/f1r3node/rholang/src/main/scala/coop/rchain/rholang/interpreter/SystemProcesses.scala) | `rho:io:file` system process — handles `register` and `delete` |
| [Genesis.scala](file:///home/dev/system-integration/services/f1r3node/casper/src/main/scala/coop/rchain/casper/genesis/Genesis.scala) | Includes `StandardDeploys.fileRegistry(shardId)` in genesis deploys |
| [StandardDeploys.scala](file:///home/dev/system-integration/services/f1r3node/casper/src/main/scala/coop/rchain/casper/genesis/contracts/StandardDeploys.scala) | Defines `fileRegistry` deploy and `fileRegistryPk` |

## Impact

- **All file downloads are broken** — both from validators (dev-mode) and observers
- Upload works fine — files are saved to disk and deploys are included in blocks
- Cross-node file replication (P2P sync) works at the block/deploy level
- Integration tests for download cannot pass until this is fixed

## Suggested investigation steps

1. **Verify the URI:** Add a debug `exploratoryDeploy` that just does `rl!(\`rho:id:m6rqma7...\`, *ch) | for(@x <- ch) { return!(x) }` — does it return the `FileRegistry` bundle or Nil?

2. **Check registerNotify wiring:** Add logging to the `registerNotify` contract in `FileRegistry.rho` or check if the `rho:io:file:registerNotify` fixed channel is correctly bound in `RhoRuntime.scala`.

3. **Inspect genesis state:** Use `exploratoryDeploy` immediately after genesis (block #0) to verify `FileRegistry` exists at the expected URI.

4. **Compare image vs source:** Check if the Docker image `f1r3flyindustries/f1r3fly-scala-node` was built from the same commit that includes `FileRegistry.rho` in genesis.
