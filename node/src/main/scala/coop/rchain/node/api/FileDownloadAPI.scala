package coop.rchain.node.api

import coop.rchain.casper.protocol.deploy.v1._
import monix.eval.Task
import monix.reactive.Observable

import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.{Files, Path, StandardOpenOption}
import java.util.concurrent.{ConcurrentHashMap, Semaphore}

object FileDownloadAPI {

  /** Per-IP semaphore map for rate limiting. */
  private val ipSemaphores = new ConcurrentHashMap[String, Semaphore]()

  /** Regex for a valid Blake2b-256 hex hash (64 lowercase hex chars). */
  private val HashPattern = "^[a-f0-9]{64}$".r

  /**
    * Stream a file to the client as `FileDownloadChunk` messages.
    *
    * @param request            The download request containing `fileHash` and optional `offset`.
    * @param isNodeReadOnly     Whether this node is a read-only observer.
    * @param uploadDir          Directory where content-addressed files are stored.
    * @param ipAddress          Client IP address for rate limiting.
    * @param chunkSize          Chunk size for streaming (from config).
    * @param maxConcurrentPerIp Maximum concurrent downloads from one IP (from config).
    * @return An `Observable` starting with a metadata chunk, followed by data chunks.
    */
  def streamFile(
      request: FileDownloadRequest,
      isNodeReadOnly: Boolean,
      uploadDir: Path,
      ipAddress: String = "unknown",
      chunkSize: Int = 4 * 1024 * 1024,
      maxConcurrentPerIp: Int = 4
  ): Observable[FileDownloadChunk] =
    // Observer gate
    if (!isNodeReadOnly)
      Observable.raiseError(
        new IllegalArgumentException(
          "File download can only be executed on a read-only f1r3node."
        )
      )
    // Hash format validation
    else if (!HashPattern.pattern.matcher(request.fileHash).matches())
      Observable.raiseError(
        new IllegalArgumentException(s"INVALID_ARGUMENT: fileHash format invalid")
      )
    else {
      val filePath = uploadDir.resolve(request.fileHash)
      if (!Files.exists(filePath))
        Observable.raiseError(
          new IllegalArgumentException(s"NOT_FOUND: file ${request.fileHash} not found")
        )
      else
        acquireAndStream(filePath, request, ipAddress, chunkSize, maxConcurrentPerIp)
    }

  /**
    * Acquire a rate-limiting semaphore permit, then stream the file.
    * The permit is released when the Observable terminates (complete or error).
    */
  private def acquireAndStream(
      filePath: Path,
      request: FileDownloadRequest,
      ipAddress: String,
      chunkSize: Int,
      maxConcurrentPerIp: Int
  ): Observable[FileDownloadChunk] = {
    val sem = ipSemaphores.computeIfAbsent(ipAddress, _ => new Semaphore(maxConcurrentPerIp))
    if (!sem.tryAcquire())
      Observable.raiseError(
        new IllegalArgumentException(
          "RESOURCE_EXHAUSTED: too many concurrent downloads from this IP"
        )
      )
    else
      doStream(filePath, request.fileHash, request.offset, chunkSize)
        .guarantee(Task.delay(sem.release()))
  }

  /**
    * Produces the download stream: metadata message first, then data chunks.
    */
  private def doStream(
      filePath: Path,
      fileHash: String,
      offset: Long,
      chunkSize: Int
  ): Observable[FileDownloadChunk] = {
    val fileSize = Files.size(filePath)

    val metadataChunk = FileDownloadChunk(
      FileDownloadChunk.Chunk.Metadata(
        FileDownloadMetadata(fileHash = fileHash, fileSize = fileSize)
      )
    )

    val dataChunks = Observable
      .fromTask(
        Task.delay {
          val channel = FileChannel.open(filePath, StandardOpenOption.READ)
          if (offset > 0) channel.position(offset)
          channel
        }
      )
      .flatMap { channel =>
        Observable
          .repeatEvalF {
            Task.delay {
              val buf       = ByteBuffer.allocate(chunkSize)
              val bytesRead = channel.read(buf)
              buf.flip()
              (bytesRead, buf)
            }
          }
          .takeWhile { case (bytesRead, _) => bytesRead > 0 }
          .map {
            case (bytesRead, buf) =>
              val arr = new Array[Byte](bytesRead)
              buf.get(arr)
              FileDownloadChunk(
                FileDownloadChunk.Chunk.Data(com.google.protobuf.ByteString.copyFrom(arr))
              )
          }
          .guarantee(Task.delay(channel.close()))
      }

    Observable.cons(metadataChunk, dataChunks)
  }

  /**
    * Visible for testing: reset the per-IP semaphore map.
    */
  private[api] def resetRateLimiter(): Unit = ipSemaphores.clear()
}
