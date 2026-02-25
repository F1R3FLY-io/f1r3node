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
      uploadDir: Path
  ): Task[FileUploadOutput] =
    chunks.headOptionL.flatMap {
      case None =>
        Task.raiseError(new IllegalArgumentException("Stream is empty"))

      case Some(firstChunk) =>
        firstChunk.chunk match {
          case FileUploadChunk.Chunk.Metadata(metadata) =>
            validateMetadata(metadata, shardId, minPhloPrice, isNodeReadOnly) match {
              case Left(err) => Task.raiseError(new IllegalArgumentException(err))
              case Right(_)  => processUpload(metadata, chunks.drop(1), uploadDir)
            }

          case _ =>
            Task.raiseError(
              new IllegalArgumentException("First chunk must be FileUploadMetadata")
            )
        }
    }

  private def validateMetadata(
      metadata: FileUploadMetadata,
      nodeShardId: String,
      minPhloPrice: Long,
      isNodeReadOnly: Boolean
  ): Either[String, Unit] =
    if (isNodeReadOnly)
      Left("Node is in read-only mode")
    else if (metadata.shardId != nodeShardId)
      Left(s"Invalid shardId: ${metadata.shardId} != $nodeShardId")
    else if (metadata.phloPrice < minPhloPrice)
      Left(s"Phlo price ${metadata.phloPrice} is lower than minimum $minPhloPrice")
    else if (metadata.term.isEmpty)
      Left("Missing required field: term (client must provide the signed Rholang term)")
    else
      Right(())

  private def processUpload(
      metadata: FileUploadMetadata,
      dataChunks: Observable[FileUploadChunk],
      uploadDir: Path
  ): Task[FileUploadOutput] =
    Task.delay(Files.createDirectories(uploadDir)).flatMap { _ =>
      val fileHash = metadata.fileHash
      Task.delay(fileHash.nonEmpty && Files.exists(uploadDir.resolve(fileHash))).flatMap {
        case true =>
          // File already on disk (dedup) — still build deploy for on-chain registration
          Task.now(buildOutput(metadata, fileHash))

        case false =>
          Task.delay(uploadDir.resolve(s"${UUID.randomUUID().toString}.tmp")).flatMap { tempFile =>
            streamToFileAndHash(dataChunks, tempFile, metadata.fileSize)
              .flatMap {
                case (bytesReceived, computedHash) =>
                  finalizeUpload(metadata, tempFile, uploadDir, bytesReceived, computedHash)
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
      computedHash: String
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

        buildOutput(metadata, computedHash)
      }

  /** Compute costs, build deploy proto, and package the [[FileUploadOutput]]. */
  private def buildOutput(
      metadata: FileUploadMetadata,
      fileHash: String
  ): FileUploadOutput = {
    val (storageCost, totalCost) =
      SyntheticDeploy.computeStorageCost(metadata.fileSize, metadata.phloPrice)

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
    * Formula: 1 phlo per byte for storage cost; total = cost × price.
    * Throws [[ArithmeticException]] on overflow.
    *
    * @return (storagePhloCost, totalPhloCharged)
    */
  def computeStorageCost(fileSize: Long, phloPrice: Long): (Long, Long) = {
    val storagePhloCost  = fileSize
    val totalPhloCharged = Math.multiplyExact(fileSize, phloPrice)
    (storagePhloCost, totalPhloCharged)
  }
}
