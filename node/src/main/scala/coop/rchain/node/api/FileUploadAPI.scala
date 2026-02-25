package coop.rchain.node.api

import coop.rchain.casper.protocol.deploy.v1._
import monix.eval.Task
import monix.reactive.Observable
import org.bouncycastle.crypto.digests.Blake2bDigest
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.{Files, Path, StandardCopyOption, StandardOpenOption}
import java.util.UUID

object FileUploadAPI {

  def processFileUpload(
      chunks: Observable[FileUploadChunk],
      shardId: String,
      minPhloPrice: Long,
      isNodeReadOnly: Boolean,
      uploadDir: Path
  ): Task[FileUploadResult] =
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
    else
      Right(())

  private def processUpload(
      metadata: FileUploadMetadata,
      dataChunks: Observable[FileUploadChunk],
      uploadDir: Path
  ): Task[FileUploadResult] =
    Task.delay(Files.createDirectories(uploadDir)).flatMap { _ =>
      // Fast-path: pre-verified dedup check
      val expectedHash = metadata.expectedFileHash
      Task.delay(expectedHash.nonEmpty && Files.exists(uploadDir.resolve(expectedHash))).flatMap {
        case true =>
          Task.now(
            FileUploadResult(
              fileHash = expectedHash,
              deployId = "",
              storagePhloCost = 0L,
              totalPhloCharged = 0L
            )
          )
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
              val hex = hashBytes.map("%02x".format(_)).mkString
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
  ): Task[FileUploadResult] =
    if (bytesReceived != metadata.fileSize)
      Task.raiseError(
        new IllegalArgumentException(
          s"Size mismatch: received $bytesReceived, expected ${metadata.fileSize}"
        )
      )
    else if (metadata.expectedFileHash.nonEmpty && computedHash != metadata.expectedFileHash)
      Task.raiseError(
        new IllegalArgumentException(
          s"Hash mismatch: computed $computedHash, expected ${metadata.expectedFileHash}"
        )
      )
    else
      Task.delay {
        val finalFile = uploadDir.resolve(computedHash)
        if (Files.exists(finalFile)) {
          // Hash collision — another upload beat us; discard the temp file
          Files.deleteIfExists(tempFile)
        } else {
          Files.move(tempFile, finalFile, StandardCopyOption.ATOMIC_MOVE)
        }

        // Write metadata sidecar
        val meta = FileMetadata(
          fileName = metadata.fileName,
          fileSize = metadata.fileSize,
          uploaderPubKey = metadata.deployer.toByteArray.map("%02x".format(_)).mkString,
          timestamp = metadata.timestamp,
          hash = computedHash
        )
        val metaFile = uploadDir.resolve(s"$computedHash.meta.json")
        Files.write(metaFile, FileMetadata.toJson(meta).getBytes("UTF-8"))

        FileUploadResult(
          fileHash = computedHash,
          deployId = "",
          storagePhloCost = 0L,
          totalPhloCharged = 0L
        )
      }
}
