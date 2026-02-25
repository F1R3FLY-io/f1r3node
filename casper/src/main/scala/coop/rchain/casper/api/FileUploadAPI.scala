package coop.rchain.casper.api

import cats.effect.Sync
import cats.syntax.all._
import coop.rchain.casper.protocol.deploy.v1._
import coop.rchain.shared.Log
import monix.eval.Task
import monix.reactive.Observable
import java.nio.file.{Files, Path, StandardCopyOption}
import java.security.MessageDigest

object FileUploadAPI {

  def processFileUpload[F[_]: Sync: Log](
      chunks: Observable[FileUploadChunk],
      networkId: String,
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
              case Right(_) =>
                val tempFile = Files.createTempFile("upload-", ".bin")
                for {
                  _    <- writeChunksToFile(chunks.drop(1), tempFile)
                  hash <- computeFileHash(tempFile)
                  _    <- saveMetadata(hash, metadata, uploadDir.resolve(s"$hash.meta.json"))
                  _    <- moveFile(tempFile, uploadDir.resolve(hash))
                } yield FileUploadResult(
                  fileHash = hash,
                  deployId = "",        // filled in by the gRPC handler after synthetic deploy creation
                  storagePhloCost = 0L, // filled in by the gRPC handler after cost calculation
                  totalPhloCharged = 0L
                )
            }
          case _ =>
            Task.raiseError(new IllegalArgumentException("First chunk must be FileUploadMetadata"))
        }
    }

  private def validateMetadata(
      metadata: FileUploadMetadata,
      nodeShardId: String,
      minPhloPrice: Long,
      isNodeReadOnly: Boolean
  ): Either[String, FileUploadMetadata] =
    if (isNodeReadOnly) Left("Node is in read-only mode")
    else if (metadata.shardId != nodeShardId)
      Left(s"Invalid shardId: ${metadata.shardId} != $nodeShardId")
    else if (metadata.phloPrice < minPhloPrice)
      Left(s"Phlo price ${metadata.phloPrice} is lower than minimum $minPhloPrice")
    else Right(metadata)

  private def writeChunksToFile(
      chunks: Observable[FileUploadChunk],
      tempFile: Path
  ): Task[Long] = {
    import java.io.FileOutputStream
    Task
      .eval(new FileOutputStream(tempFile.toFile))
      .bracket { fos =>
        chunks.foldLeftL(0L) { (totalBytes, chunk) =>
          chunk.chunk match {
            case FileUploadChunk.Chunk.Data(bytes) =>
              val data = bytes.toByteArray
              fos.write(data)
              totalBytes + data.length
            case _ => totalBytes
          }
        }
      } { fos =>
        Task.eval(fos.close())
      }
  }

  private def computeFileHash(file: Path): Task[String] =
    Task.eval {
      val digest = MessageDigest.getInstance("SHA-256")
      val bytes  = Files.readAllBytes(file)
      digest.update(bytes)
      digest.digest().map("%02x".format(_)).mkString
    }

  private def moveFile(tempFile: Path, finalFile: Path): Task[Unit] =
    Task.eval {
      Files.createDirectories(finalFile.getParent)
      Files.move(tempFile, finalFile, StandardCopyOption.REPLACE_EXISTING)
    }

  private def saveMetadata(hash: String, metadata: FileUploadMetadata, metaFile: Path): Task[Unit] =
    Task.eval {
      val json =
        s"""{"fileName":"${metadata.fileName}","fileSize":${metadata.fileSize},"hash":"$hash"}"""
      Files.write(metaFile, json.getBytes)
    }
}
