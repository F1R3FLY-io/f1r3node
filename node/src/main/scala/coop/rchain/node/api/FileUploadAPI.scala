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

  def processFileUpload(
      chunks: Observable[FileUploadChunk],
      shardId: String,
      minPhloPrice: Long,
      isNodeReadOnly: Boolean,
      uploadDir: Path,
      phloPerStorageByte: Long = FileUploadCosts.DEFAULT_PHLO_PER_STORAGE_BYTE,
      baseRegisterPhlo: Long = FileUploadCosts.BASE_REGISTER_PHLO
  ): Task[FileUploadOutput] =
    // Collect all chunks into a list first. gRPC streams are hot Observables —
    // calling headOptionL + drop(1) would create two separate subscriptions,
    // causing the second one to see zero data bytes. By materializing into a
    // list we get a single subscription and can safely split head/tail.
    chunks.toListL.flatMap { allChunks =>
      allChunks match {
        case Nil =>
          Task.raiseError(new IllegalArgumentException("Stream is empty"))

        case firstChunk :: dataChunks =>
          firstChunk.chunk match {
            case FileUploadChunk.Chunk.Metadata(metadata) =>
              validateMetadata(
                metadata,
                shardId,
                minPhloPrice,
                isNodeReadOnly,
                phloPerStorageByte,
                baseRegisterPhlo
              ) match {
                case Left(err) => Task.raiseError(new IllegalArgumentException(err))
                case Right(_) =>
                  processUpload(
                    metadata,
                    Observable.fromIterable(dataChunks),
                    uploadDir,
                    phloPerStorageByte,
                    baseRegisterPhlo
                  )
              }

            case _ =>
              Task.raiseError(
                new IllegalArgumentException("First chunk must be FileUploadMetadata")
              )
          }
      }
    }

  private def validateMetadata(
      metadata: FileUploadMetadata,
      nodeShardId: String,
      minPhloPrice: Long,
      isNodeReadOnly: Boolean,
      phloPerStorageByte: Long,
      baseRegisterPhlo: Long
  ): Either[String, Unit] =
    if (isNodeReadOnly)
      Left("Node is in read-only mode")
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

  private def processUpload(
      metadata: FileUploadMetadata,
      dataChunks: Observable[FileUploadChunk],
      uploadDir: Path,
      phloPerStorageByte: Long,
      baseRegisterPhlo: Long
  ): Task[FileUploadOutput] =
    Task.delay(Files.createDirectories(uploadDir)).flatMap { _ =>
      val fileHash = metadata.fileHash
      Task.delay(fileHash.nonEmpty && Files.exists(uploadDir.resolve(fileHash))).flatMap {
        case true =>
          // File already on disk (dedup) — still build deploy for on-chain registration
          Task.now(buildOutput(metadata, fileHash, phloPerStorageByte, baseRegisterPhlo))

        case false =>
          Task.delay(uploadDir.resolve(s"${UUID.randomUUID().toString}.tmp")).flatMap { tempFile =>
            streamToFileAndHash(dataChunks, tempFile, metadata.fileSize)
              .flatMap {
                case (bytesReceived, computedHash) =>
                  finalizeUpload(
                    metadata,
                    tempFile,
                    uploadDir,
                    bytesReceived,
                    computedHash,
                    phloPerStorageByte,
                    baseRegisterPhlo
                  )
              }
              .guarantee(
                Task.delay {
                  if (Files.exists(tempFile))
                    Files.deleteIfExists(tempFile)
                  ()
                }
              )
          }
      }
    }

  /**
    * Opens `tempFile` for writing, writes every data chunk via `FileChannel`, and
    * feeds the same bytes into a streaming `Blake2bDigest`.
    *
    * Aborts with an error if `bytesReceived > maxBytes`.
    *
    * @return (totalBytesWritten, Blake2b-256 hex hash)
    */
  private def streamToFileAndHash(
      chunks: Observable[FileUploadChunk],
      tempFile: Path,
      maxBytes: Long
  ): Task[(Long, String)] = {
    val openChannel = Task.delay(
      FileChannel.open(tempFile, StandardOpenOption.WRITE, StandardOpenOption.CREATE_NEW)
    )

    openChannel.bracket { channel =>
      val digest = new Blake2bDigest(256)
      chunks
        .foldLeftL((0L, false)) {
          case ((total, aborted), chunk) =>
            if (aborted) (total, aborted)
            else
              chunk.chunk match {
                case FileUploadChunk.Chunk.Data(bytes) =>
                  val data     = bytes.toByteArray
                  val newTotal = total + data.length
                  if (newTotal > maxBytes) (newTotal, true)
                  else {
                    digest.update(data, 0, data.length)
                    val buf = ByteBuffer.wrap(data)
                    while (buf.hasRemaining) channel.write(buf)
                    (newTotal, false)
                  }
                case _ => (total, aborted)
              }
        }
        .flatMap {
          case (totalBytes, aborted) =>
            if (aborted)
              Task.raiseError(
                new IllegalArgumentException(
                  s"Upload aborted: received $totalBytes bytes exceeds declared fileSize $maxBytes"
                )
              )
            else {
              val hashBytes = new Array[Byte](32)
              digest.doFinal(hashBytes, 0)
              val hex = toHex(hashBytes)
              Task.now((totalBytes, hex))
            }
        }
    } { channel =>
      Task.delay(channel.close())
    }
  }

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
      Task.raiseError(
        new IllegalArgumentException(
          s"Size mismatch: received $bytesReceived, expected ${metadata.fileSize}"
        )
      )
    else if (metadata.fileHash.nonEmpty && computedHash != metadata.fileHash)
      Task.raiseError(
        new IllegalArgumentException(
          s"Hash mismatch: computed $computedHash, expected ${metadata.fileHash}"
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
