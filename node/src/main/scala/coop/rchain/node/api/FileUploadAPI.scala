package coop.rchain.node.api

import coop.rchain.casper.protocol.DeployDataProto
import coop.rchain.casper.protocol.deploy.v1._
import monix.eval.Task
import monix.reactive.Observable
import org.bouncycastle.crypto.digests.Blake2bDigest
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.{Files, Path, StandardCopyOption, StandardOpenOption}
import java.util.UUID

/**
  * Result from [[FileUploadAPI.processFileUpload]].
  *
  * @param result     gRPC response for the client (`deployId` initially empty;
  *                   filled by the gRPC layer after signature validation)
  * @param deployProto maps the upload metadata to a [[DeployDataProto]] ready
  *                   for signature validation via `DeployData.from()` and mempool push
  */
final case class FileUploadOutput(
    result: FileUploadResult,
    deployProto: Option[DeployDataProto]
)

object FileUploadAPI {

  // ---------------------------------------------------------------------------
  // Upload state machine — O(chunk-size) memory
  // ---------------------------------------------------------------------------

  /** ADT for the single-pass fold over the gRPC chunk stream. */
  private sealed trait UploadState

  /** No chunks received yet — expecting the metadata chunk first. */
  private case object WaitingForMetadata extends UploadState

  /**
    * Metadata validated, actively writing data chunks to [[channel]].
    *
    * @param metadata     validated upload metadata
    * @param channel      open `FileChannel` for the `.tmp` file
    * @param digest       streaming Blake2b-256 state
    * @param bytesWritten total bytes written so far
    * @param tempFile     path to the `.tmp` file
    * @param aborted      set `true` when `bytesWritten > maxBytes`, causing
    *                     remaining chunks to be skipped
    */
  private final case class WritingData(
      metadata: FileUploadMetadata,
      channel: FileChannel,
      digest: Blake2bDigest,
      bytesWritten: Long,
      tempFile: Path,
      aborted: Boolean
  ) extends UploadState

  /** A non-recoverable error was detected; remaining chunks are drained. */
  private final case class Failed(error: Throwable) extends UploadState

  /** File already on disk (dedup) — remaining data chunks are drained. */
  private final case class Dedup(metadata: FileUploadMetadata) extends UploadState

  // ---------------------------------------------------------------------------
  // Public entry point
  // ---------------------------------------------------------------------------

  def processFileUpload(
      chunks: Observable[FileUploadChunk],
      shardId: String,
      minPhloPrice: Long,
      isNodeReadOnly: Boolean,
      uploadDir: Path,
      phloPerStorageByte: Long = FileUploadCosts.DEFAULT_PHLO_PER_STORAGE_BYTE,
      baseRegisterPhlo: Long = FileUploadCosts.BASE_REGISTER_PHLO,
      maxFileSize: Long = 10L * 1024 * 1024 * 1024
  ): Task[FileUploadOutput] =
    Task.delay(Files.createDirectories(uploadDir)).flatMap { _ =>
      // Track last state so we can clean up on mid-stream Observable errors
      // (e.g. gRPC cancel) where foldLeftL never produces a final state.
      val lastState =
        new java.util.concurrent.atomic.AtomicReference[UploadState](WaitingForMetadata)

      // Single-pass fold: metadata is extracted from the first chunk,
      // all subsequent data chunks are written directly to a FileChannel.
      // Memory usage is O(chunk-size) regardless of total file size.
      chunks
        .foldLeftL(WaitingForMetadata: UploadState) { (state, chunk) =>
          val nextState = state match {
            // --- skip remaining chunks after an error or dedup -------------
            case _: Failed => state
            case _: Dedup  => state

            // --- first chunk: must be metadata -----------------------------
            case WaitingForMetadata =>
              chunk.chunk match {
                case FileUploadChunk.Chunk.Metadata(metadata) =>
                  validateMetadata(
                    metadata,
                    shardId,
                    minPhloPrice,
                    isNodeReadOnly,
                    phloPerStorageByte,
                    baseRegisterPhlo,
                    maxFileSize
                  ) match {
                    case Left(err) =>
                      Failed(new IllegalArgumentException(err))

                    case Right(_) =>
                      // Check for dedup — if file already on disk, skip writing
                      val fileHash = metadata.fileHash
                      if (fileHash.nonEmpty && Files.exists(uploadDir.resolve(fileHash))) {
                        Dedup(metadata)
                      } else {
                        val tempFile = uploadDir.resolve(s"${UUID.randomUUID().toString}.tmp")
                        try {
                          val channel = FileChannel.open(
                            tempFile,
                            StandardOpenOption.WRITE,
                            StandardOpenOption.CREATE_NEW
                          )
                          val digest = new Blake2bDigest(256)
                          WritingData(metadata, channel, digest, 0L, tempFile, aborted = false)
                        } catch {
                          case ex: Throwable => Failed(ex)
                        }
                      }
                  }

                case _ =>
                  Failed(new IllegalArgumentException("First chunk must be FileUploadMetadata"))
              }

            // --- data chunks: write to disk --------------------------------
            case wd: WritingData =>
              if (wd.aborted) wd
              else
                chunk.chunk match {
                  case FileUploadChunk.Chunk.Data(bytes) =>
                    val data     = bytes.toByteArray
                    val newTotal = wd.bytesWritten + data.length
                    if (newTotal > wd.metadata.fileSize)
                      wd.copy(bytesWritten = newTotal, aborted = true)
                    else {
                      // N.B. digest and channel are mutable — update() and
                      // write() mutate in place. copy() shares the same
                      // references, which is safe because foldLeftL is
                      // strictly sequential (old state is never reused).
                      wd.digest.update(data, 0, data.length)
                      val buf = ByteBuffer.wrap(data)
                      while (buf.hasRemaining) wd.channel.write(buf)
                      wd.copy(bytesWritten = newTotal)
                    }
                  case _ => wd // skip non-data chunks (e.g. stray metadata)
                }
          }
          lastState.set(nextState)
          nextState
        }
        .flatMap { finalState =>
          finalState match {
            // -- empty stream -----------------------------------------------
            case WaitingForMetadata =>
              Task.raiseError(new IllegalArgumentException("Stream is empty"))

            // -- dedup: file already on disk --------------------------------
            case Dedup(meta) =>
              Task.now(
                buildOutput(meta, meta.fileHash, phloPerStorageByte, baseRegisterPhlo)
              )

            // -- real error -------------------------------------------------
            case Failed(err) =>
              Task.raiseError(err)

            // -- normal completion ------------------------------------------
            case wd: WritingData =>
              // Close the channel first
              Task.delay(wd.channel.close()).flatMap { _ =>
                if (wd.aborted)
                  Task
                    .delay(Files.deleteIfExists(wd.tempFile))
                    .flatMap(
                      _ =>
                        Task.raiseError(
                          new IllegalArgumentException(
                            s"Upload aborted: received ${wd.bytesWritten} bytes exceeds declared fileSize ${wd.metadata.fileSize}"
                          )
                        )
                    )
                else {
                  val hashBytes = new Array[Byte](32)
                  wd.digest.doFinal(hashBytes, 0)
                  val computedHash = toHex(hashBytes)
                  finalizeUpload(
                    wd.metadata,
                    wd.tempFile,
                    uploadDir,
                    wd.bytesWritten,
                    computedHash,
                    phloPerStorageByte,
                    baseRegisterPhlo
                  )
                }
              }
          }
        }
        .onErrorHandleWith { err =>
          // Mid-stream Observable error (e.g. gRPC cancel): clean up any
          // open FileChannel and .tmp file tracked by the AtomicReference.
          // Both close() and deleteIfExists() are wrapped in try/catch so
          // a cleanup failure can never replace the original error.
          val cleanup = lastState.get() match {
            case wd: WritingData =>
              Task.delay {
                try wd.channel.close()
                catch { case _: Throwable => () }
                try {
                  Files.deleteIfExists(wd.tempFile); ()
                } catch { case _: Throwable => () }
              }
            case _ => Task.unit
          }
          cleanup.flatMap(_ => Task.raiseError(err))
        }
    }

  // ---------------------------------------------------------------------------
  // Validation
  // ---------------------------------------------------------------------------

  private def validateMetadata(
      metadata: FileUploadMetadata,
      nodeShardId: String,
      minPhloPrice: Long,
      isNodeReadOnly: Boolean,
      phloPerStorageByte: Long,
      baseRegisterPhlo: Long,
      maxFileSize: Long
  ): Either[String, Unit] =
    if (isNodeReadOnly)
      Left("Node is in read-only mode")
    else if (metadata.fileSize <= 0)
      Left(s"Invalid fileSize: ${metadata.fileSize} (must be > 0)")
    else if (metadata.fileSize > maxFileSize)
      Left(
        s"File too large: ${metadata.fileSize} bytes exceeds maximum ${maxFileSize} bytes " +
          s"(${maxFileSize / (1024 * 1024)} MB)"
      )
    else if (metadata.fileHash.nonEmpty && !metadata.fileHash.matches("^[a-f0-9]{64}$"))
      Left(s"Invalid fileHash format: must be 64 lowercase hex characters")
    else if (metadata.shardId != nodeShardId)
      Left(s"Invalid shardId: ${metadata.shardId} != $nodeShardId")
    else if (metadata.phloPrice < minPhloPrice)
      Left(s"Phlo price ${metadata.phloPrice} is lower than minimum $minPhloPrice")
    else if (metadata.term.isEmpty)
      Left("Missing required field: term (client must provide the signed Rholang term)")
    else {
      val totalRequired =
        FileUploadCosts.totalRequired(metadata.fileSize, phloPerStorageByte, baseRegisterPhlo)
      if (metadata.phloLimit < totalRequired)
        Left(
          s"Insufficient phlo: phloLimit=${metadata.phloLimit} < required=$totalRequired " +
            s"($baseRegisterPhlo base + ${metadata.fileSize} bytes × $phloPerStorageByte phlo/byte)"
        )
      else
        Right(())
    }

  // ---------------------------------------------------------------------------
  // Finalization
  // ---------------------------------------------------------------------------

  /**
    * Validates byte-count and optional hash, then atomically renames `.tmp` to its
    * final content-addressed name. Writes the `.meta.json` sidecar.
    */
  private def finalizeUpload(
      metadata: FileUploadMetadata,
      tempFile: Path,
      uploadDir: Path,
      bytesReceived: Long,
      computedHash: String,
      phloPerStorageByte: Long,
      baseRegisterPhlo: Long
  ): Task[FileUploadOutput] =
    if (bytesReceived != metadata.fileSize)
      Task
        .delay(Files.deleteIfExists(tempFile))
        .flatMap(
          _ =>
            Task.raiseError(
              new IllegalArgumentException(
                s"Size mismatch: received $bytesReceived, expected ${metadata.fileSize}"
              )
            )
        )
    else if (metadata.fileHash.nonEmpty && computedHash != metadata.fileHash)
      Task
        .delay(Files.deleteIfExists(tempFile))
        .flatMap(
          _ =>
            Task.raiseError(
              new IllegalArgumentException(
                s"Hash mismatch: computed $computedHash, expected ${metadata.fileHash}"
              )
            )
        )
    else
      Task.delay {
        val finalFile = uploadDir.resolve(computedHash)
        if (Files.exists(finalFile)) {
          Files.deleteIfExists(tempFile)
        } else {
          Files.move(tempFile, finalFile, StandardCopyOption.ATOMIC_MOVE)
        }

        // Write metadata sidecar
        val meta = FileMetadata(
          fileName = metadata.fileName,
          fileSize = metadata.fileSize,
          uploaderPubKey = toHex(metadata.deployer.toByteArray),
          timestamp = metadata.timestamp,
          hash = computedHash
        )
        val metaFile = uploadDir.resolve(s"$computedHash.meta.json")
        Files.write(metaFile, FileMetadata.toJson(meta).getBytes("UTF-8"))

        buildOutput(metadata, computedHash, phloPerStorageByte, baseRegisterPhlo)
      }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  /** Compute costs, build deploy proto, and package the [[FileUploadOutput]]. */
  private def buildOutput(
      metadata: FileUploadMetadata,
      fileHash: String,
      phloPerStorageByte: Long,
      baseRegisterPhlo: Long
  ): FileUploadOutput = {
    val (storageCost, totalCost) =
      SyntheticDeploy.computeStorageCost(metadata.fileSize, phloPerStorageByte, baseRegisterPhlo)

    val proto = SyntheticDeploy.metadataToDeployProto(metadata)

    FileUploadOutput(
      result = FileUploadResult(
        fileHash = fileHash,
        deployId = "", // filled by gRPC layer after sig validation
        storagePhloCost = storageCost,
        totalPhloCharged = totalCost
      ),
      deployProto = Some(proto)
    )
  }

  private def toHex(bytes: Array[Byte]): String =
    bytes.map("%02x".format(_)).mkString
}

/**
  * Pure helpers for the synthetic deploy that registers a file on-chain.
  *
  * The client constructs the Rholang term, signs the full `DeployData`, and
  * sends `term` + `sig` in the upload metadata. The server maps the metadata
  * to a [[DeployDataProto]], validates the client's signature via `DeployData.from()`,
  * and pushes the resulting `Signed[DeployData]` to the mempool.
  */
object SyntheticDeploy {

  /**
    * Maps [[FileUploadMetadata]] fields to a [[DeployDataProto]] that can be
    * validated and pushed through `DeployData.from()` → `BlockAPI.deploy()`.
    *
    * The client has already signed the deploy; we just reassemble the proto.
    */
  def metadataToDeployProto(metadata: FileUploadMetadata): DeployDataProto =
    DeployDataProto()
      .withDeployer(metadata.deployer)
      .withTerm(metadata.term)
      .withTimestamp(metadata.timestamp)
      .withSig(metadata.sig)
      .withSigAlgorithm(metadata.sigAlgorithm)
      .withPhloPrice(metadata.phloPrice)
      .withPhloLimit(metadata.phloLimit)
      .withValidAfterBlockNumber(metadata.validAfterBlockNumber)
      .withShardId(metadata.shardId)

  /**
    * Computes the phlo cost for storing `fileSize` bytes.
    *
    * Formula: `storagePhloCost = fileSize × phloPerStorageByte`,
    *          `totalPhloCharged = baseRegisterPhlo + storagePhloCost`.
    * Throws [[ArithmeticException]] on overflow.
    *
    * @return (storagePhloCost, totalPhloCharged)
    */
  def computeStorageCost(
      fileSize: Long,
      phloPerStorageByte: Long = FileUploadCosts.DEFAULT_PHLO_PER_STORAGE_BYTE,
      baseRegisterPhlo: Long = FileUploadCosts.BASE_REGISTER_PHLO
  ): (Long, Long) = {
    val storagePhloCost  = Math.multiplyExact(fileSize, phloPerStorageByte)
    val totalPhloCharged = Math.addExact(baseRegisterPhlo, storagePhloCost)
    (storagePhloCost, totalPhloCharged)
  }
}
