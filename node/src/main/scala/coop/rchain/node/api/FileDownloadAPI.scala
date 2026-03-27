package coop.rchain.node.api

import coop.rchain.casper.protocol.deploy.v1._
import monix.eval.Task
import monix.reactive.Observable

import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.{Files, Path, StandardOpenOption}
import java.util.concurrent.Semaphore
import java.util.{Collections, LinkedHashMap}
import org.slf4j.LoggerFactory
import coop.rchain.shared.FileHashValidation

object FileDownloadAPI {

  private val logger = LoggerFactory.getLogger("FileDownloadAPI")

  // Initialized once in streamFile (from config) to prevent race conditions.
  private val _maxIpEntries = new java.util.concurrent.atomic.AtomicInteger(-1)

  private def getMaxIpEntries(configMax: Int): Int = {
    _maxIpEntries.compareAndSet(-1, configMax)
    _maxIpEntries.get()
  }

  private lazy val ipSemaphores: java.util.Map[String, Semaphore] =
    Collections.synchronizedMap(
      new LinkedHashMap[String, Semaphore](128, 0.75f, true /* accessOrder */ ) {
        override def removeEldestEntry(
            eldest: java.util.Map.Entry[String, Semaphore]
        ): Boolean = size() > _maxIpEntries.get()
      }
    )

  /**
    * Stream a file to the client as `FileDownloadChunk` messages.
    *
    * The download is gated by a finalization check: the file must be registered
    * in the `FileRegistry` contract at the Last Finalized Block's post-state.
    * The finalization checker is mandatory and called on every download request.
    *
    * Guard order: observer gate → hash validation → path traversal → file exists
    *   → finalization check → rate limit → stream.
    * The finalization check runs before the rate-limit semaphore is acquired so
    * that the lightweight VM evaluation (~5-50 ms) does not consume a scarce
    * concurrency permit.
    *
    * @param request            The download request containing `fileHash` and optional `offset`.
    * @param isNodeReadOnly     Whether this node is a read-only observer.
    * @param uploadDir          Directory where content-addressed files are stored.
    * @param ipAddress          Client IP address for rate limiting.
    * @param chunkSize          Chunk size for streaming (from config).
    * @param maxConcurrentPerIp Maximum concurrent downloads from one IP (from config).
    * @param devMode            If true, allow downloads on validator nodes (not just observers).
    * @param maxCacheEntries    Maximum entries in the per-IP rate-limiter LRU cache (from config).
    * @param finalizationChecker A function that, given a file hash, returns `Task[Boolean]`
    *                            indicating whether the file is registered in a finalized block.
    *                            In production this executes an `exploratoryDeploy` against the
    *                            LFB post-state.
    * @return An `Observable` starting with a metadata chunk, followed by data chunks.
    */
  def streamFile(
      request: FileDownloadRequest,
      isNodeReadOnly: Boolean,
      uploadDir: Path,
      ipAddress: String = "unknown",
      chunkSize: Int = 4 * 1024 * 1024,
      maxConcurrentPerIp: Int = 4,
      devMode: Boolean = false,
      maxCacheEntries: Int = 10000,
      finalizationChecker: String => Task[Boolean]
  ): Observable[FileDownloadChunk] = {
    getMaxIpEntries(maxCacheEntries)
    val hash = request.fileHash
    logger.info(
      "[FileDownloadAPI] Download request: hash={}..., offset={}, readOnly={}, devMode={}",
      hash.take(16),
      request.offset.toString,
      isNodeReadOnly.toString,
      devMode.toString
    )

    // Observer gate — in dev mode, allow downloads from any node (including validators)
    if (!isNodeReadOnly && !devMode) {
      logger.warn("[FileDownloadAPI] Rejected: not read-only, hash={}...", hash.take(16))
      Observable.raiseError(
        new IllegalArgumentException(
          "File download can only be executed on a read-only f1r3node."
        )
      )
    }
    // Hash format validation (use shared utility)
    else if (!FileHashValidation.isValidFileHash(hash)) {
      logger.warn("[FileDownloadAPI] Rejected: invalid hash format '{}'", hash)
      Observable.raiseError(
        new IllegalArgumentException(s"INVALID_ARGUMENT: fileHash format invalid")
      )
    } else {
      val filePath = uploadDir.resolve(hash)
      // Defense in depth: verify resolved path is still under uploadDir
      if (!filePath.normalize().startsWith(uploadDir.normalize())) {
        logger.warn("[FileDownloadAPI] Rejected: path traversal attempt, hash={}", hash)
        Observable.raiseError(
          new IllegalArgumentException(
            s"INVALID_ARGUMENT: fileHash resolves outside upload directory"
          )
        )
      } else if (!Files.exists(filePath)) {
        logger.debug(s"[FileDownloadAPI] File NOT FOUND: hash=${hash.take(16)}..., path=$filePath")
        Observable.raiseError(
          new IllegalArgumentException(s"NOT_FOUND: file ${hash} not found")
        )
      } else {
        // ---- Finalization gate ------------------------------------------------
        // Always verify the file hash is registered in a finalized block.
        // The finalization check runs BEFORE the rate-limit semaphore is acquired
        // so the lightweight VM evaluation (~5-50 ms) does not hold a scarce permit.
        Observable
          .fromTask(finalizationChecker(hash))
          .flatMap { isFinalized =>
            if (isFinalized) {
              val fileSize = Files.size(filePath)
              logger.info(
                "[FileDownloadAPI] File found and finalized: hash={}..., size={} bytes ({} MB)",
                hash.take(16),
                fileSize.toString,
                (fileSize / (1024 * 1024)).toString
              )
              acquireAndStream(filePath, request, ipAddress, chunkSize, maxConcurrentPerIp)
            } else {
              logger.warn(
                "[FileDownloadAPI] Rejected: file not in finalized registry, hash={}...",
                hash.take(16)
              )
              Observable.raiseError(
                new IllegalArgumentException(
                  s"NOT_FOUND: file ${hash} not found"
                )
              )
            }
          }
      }
    }
  }

  /**
    * Acquire a rate-limiting semaphore permit, then stream the file.
    * The permit is released when the Observable terminates (complete or error)
    * via the outermost `.guarantee`.
    */
  private def acquireAndStream(
      filePath: Path,
      request: FileDownloadRequest,
      ipAddress: String,
      chunkSize: Int,
      maxConcurrentPerIp: Int
  ): Observable[FileDownloadChunk] = {
    val sem = ipSemaphores.synchronized {
      var s = ipSemaphores.get(ipAddress)
      if (s == null) {
        s = new Semaphore(maxConcurrentPerIp)
        ipSemaphores.put(ipAddress, s)
      }
      s
    }
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
    // Capture file size once to avoid duplicate syscalls
    val fileSize = Files.size(filePath)

    // Validate offset before streaming
    if (offset >= fileSize) {
      logger.warn(
        "[FileDownloadAPI] Rejected: offset={} >= fileSize={}, hash={}...",
        offset.toString,
        fileSize.toString,
        fileHash.take(16)
      )
      return Observable.raiseError(
        new IllegalArgumentException(
          s"INVALID_ARGUMENT: offset $offset is beyond file size $fileSize"
        )
      )
    }

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
        // Allocate one buffer outside the loop and reuse via clear()
        val buf = ByteBuffer.allocate(chunkSize)
        Observable
          .repeatEvalF {
            Task.delay {
              buf.clear()
              val bytesRead = channel.read(buf)
              buf.flip()
              bytesRead
            }
          }
          .takeWhile(_ > 0)
          // Use scan for immutable progress tracking instead of a mutable var
          .scan((offset, Option.empty[FileDownloadChunk])) {
            case ((streamed, _), bytesRead) =>
              val arr = new Array[Byte](bytesRead)
              buf.get(arr)
              val newStreamed = streamed + bytesRead
              if (newStreamed % (100 * 1024 * 1024) < chunkSize) {
                logger.info(
                  s"[FileDownloadAPI] Progress: hash=${fileHash
                    .take(16)}..., ${newStreamed / (1024 * 1024)} MB / ${fileSize / (1024 * 1024)} MB"
                )
              }
              val chunk = FileDownloadChunk(
                FileDownloadChunk.Chunk.Data(com.google.protobuf.ByteString.copyFrom(arr))
              )
              (newStreamed, Some(chunk))
          }
          .collect { case (_, Some(chunk)) => chunk }
          .guarantee(Task.delay {
            channel.close()
            logger.info(
              s"[FileDownloadAPI] Streaming completed: hash=${fileHash.take(16)}..."
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
