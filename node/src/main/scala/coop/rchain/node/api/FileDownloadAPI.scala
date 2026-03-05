package coop.rchain.node.api

import coop.rchain.casper.protocol.deploy.v1._
import monix.eval.Task
import monix.reactive.Observable

import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.{Files, Path, StandardOpenOption}
import java.util.concurrent.{ConcurrentHashMap, Semaphore}
import org.slf4j.LoggerFactory

object FileDownloadAPI {

  private val logger = LoggerFactory.getLogger("FileDownloadAPI")

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
      maxConcurrentPerIp: Int = 4,
      devMode: Boolean = false
  ): Observable[FileDownloadChunk] = {
    val hash = request.fileHash
    logger.info(
      "[FileDownloadAPI] Download request: hash={}..., offset={}, readOnly={}, devMode={}",
      hash.take(16),
      request.offset.toString,
      isNodeReadOnly.toString,
      devMode.toString
    )

    // Observer gate — in dev mode, allow downloads from any node
    if (!isNodeReadOnly && !devMode) {
      logger.warn("[FileDownloadAPI] Rejected: not read-only, hash={}...", hash.take(16))
      Observable.raiseError(
        new IllegalArgumentException(
          "File download can only be executed on a read-only f1r3node."
        )
      )
    }
    // Hash format validation
    else if (!HashPattern.pattern.matcher(hash).matches()) {
      logger.warn("[FileDownloadAPI] Rejected: invalid hash format '{}'", hash)
      Observable.raiseError(
        new IllegalArgumentException(s"INVALID_ARGUMENT: fileHash format invalid")
      )
    } else {
      val filePath = uploadDir.resolve(hash)
      if (!Files.exists(filePath)) {
        logger.debug(s"[FileDownloadAPI] File NOT FOUND: hash=${hash.take(16)}..., path=$filePath")
        Observable.raiseError(
          new IllegalArgumentException(s"NOT_FOUND: file ${hash} not found")
        )
      } else {
        val fileSize = Files.size(filePath)
        logger.info(
          "[FileDownloadAPI] File found: hash={}..., size={} bytes ({} MB)",
          hash.take(16),
          fileSize.toString,
          (fileSize / (1024 * 1024)).toString
        )
        acquireAndStream(filePath, request, ipAddress, chunkSize, maxConcurrentPerIp)
      }
    }
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
    if (!sem.tryAcquire()) {
      logger.warn(
        s"[FileDownloadAPI] Rate limited: hash=${request.fileHash.take(16)}..., ip=$ipAddress"
      )
      Observable.raiseError(
        new IllegalArgumentException(
          "RESOURCE_EXHAUSTED: too many concurrent downloads from this IP"
        )
      )
    } else {
      logger.debug(
        s"[FileDownloadAPI] Semaphore acquired: hash=${request.fileHash.take(16)}..., ip=$ipAddress"
      )
      doStream(filePath, request.fileHash, request.offset, chunkSize)
        .guarantee(Task.delay {
          sem.release()
          logger
            .debug("[FileDownloadAPI] Semaphore released: hash={}...", request.fileHash.take(16))
        })
    }
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
          logger.info(
            "[FileDownloadAPI] Streaming started: hash={}..., fileSize={}, offset={}, chunkSize={}",
            fileHash.take(16),
            fileSize.toString,
            offset.toString,
            chunkSize.toString
          )
          channel
        }
      )
      .flatMap { channel =>
        var bytesStreamed = offset
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
              bytesStreamed += bytesRead
              if (bytesStreamed % (100 * 1024 * 1024) < chunkSize) {
                logger.info(
                  s"[FileDownloadAPI] Progress: hash=${fileHash
                    .take(16)}..., ${bytesStreamed / (1024 * 1024)} MB / ${fileSize / (1024 * 1024)} MB"
                )
              }
              FileDownloadChunk(
                FileDownloadChunk.Chunk.Data(com.google.protobuf.ByteString.copyFrom(arr))
              )
          }
          .guarantee(Task.delay {
            channel.close()
            logger.info(
              s"[FileDownloadAPI] Streaming completed: hash=${fileHash.take(16)}..., totalBytes=$bytesStreamed"
            )
          })
      }

    Observable.cons(metadataChunk, dataChunks)
  }

  /**
    * Visible for testing: reset the per-IP semaphore map.
    */
  private[api] def resetRateLimiter(): Unit = ipSemaphores.clear()
}
