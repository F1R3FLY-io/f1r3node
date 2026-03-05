package coop.rchain.casper.engine

import cats.effect.concurrent.Ref
import cats.effect.{Concurrent, Sync, Timer}
import cats.syntax.all._
import coop.rchain.casper.protocol._
import coop.rchain.comm.PeerNode
import coop.rchain.comm.rp.Connect.RPConfAsk
import coop.rchain.comm.transport.TransportLayer
import coop.rchain.comm.syntax._
import coop.rchain.shared.{Log, Time}
import java.nio.file.{Path, StandardOpenOption}
import org.bouncycastle.crypto.digests.Blake2bDigest
import scala.concurrent.duration._

class FileRequester[F[_]: Concurrent: Log: Time: TransportLayer: RPConfAsk](
    dataDir: Path,
    chunkSize: Int = 4 * 1024 * 1024, // configurable, defaults to 4MB
    // Maximum time to wait for file transfers before giving up.
    // Tune based on expected file sizes and network bandwidth:
    //   6 GB @ 100 Mbps ≈ 8 min  → 30.minutes is safe
    //   6 GB @  10 Mbps ≈ 80 min → 2.hours needed
    //  10 GB @  10 Mbps ≈ 2.2 hr → increase to 3.hours
    syncTimeout: FiniteDuration = 2.hours
) extends CasperMessageProtocol {

  // State of file downloads: fileHash -> (Path, BytesReceived, ExpectedSize)
  case class DownloadState(
      tempPath: Path,
      digest: Blake2bDigest,
      bytesReceived: Long,
      expectedSize: Option[Long]
  )

  private[engine] val downloads = Ref.unsafe[F, Map[String, DownloadState]](Map.empty)
  private val completed         = Ref.unsafe[F, Set[String]](Set.empty)

  /**
    * Checks whether a file is available locally (either on disk or download completed).
    */
  def isFileAvailable(fileHash: String): F[Boolean] =
    for {
      onDisk     <- Sync[F].delay(java.nio.file.Files.exists(dataDir.resolve(fileHash)))
      isComplete <- completed.get.map(_.contains(fileHash))
    } yield onDisk || isComplete

  /**
    * Triggers downloads for multiple missing file hashes by broadcasting FileRequests
    * to the given peer. Each peer will respond with FilePacket only if it has the file.
    * Previously used handleHasFile which only started download from a single peer.
    */
  def requestFiles(peer: PeerNode, fileHashes: List[String]): F[Unit] =
    fileHashes.traverse_ { hash =>
      for {
        isCompleted <- completed.get.map(_.contains(hash))
        onDisk      <- Sync[F].delay(java.nio.file.Files.exists(dataDir.resolve(hash)))
        _ <- if (!isCompleted && !onDisk) {
              // Atomically ensure download state exists (prevents TOCTOU race
              // when daCallback broadcasts to multiple peers concurrently)
              downloads
                .modify { state =>
                  if (!state.contains(hash)) {
                    val tempPath = dataDir.resolve(s"$hash.part")
                    val digest   = new Blake2bDigest(256)
                    (state + (hash -> DownloadState(tempPath, digest, 0L, None)), true)
                  } else {
                    (state, false)
                  }
                }
                .flatMap { wasNew =>
                  if (wasNew)
                    Log[F].info(
                      s"[FileRequester] requestFiles: registered download for ${hash.take(16)}... from $peer"
                    )
                  else
                    Log[F].info(
                      s"[FileRequester] requestFiles: download already registered for ${hash
                        .take(16)}..., sending FileRequest to $peer"
                    )
                } *>
                // Always send FileRequest to this peer — peer will respond only if it has the file
                requestChunk(peer, hash, 0L)
            } else {
              Log[F].info(
                s"[FileRequester] requestFiles: skipping ${hash.take(16)}... (completed=$isCompleted, onDisk=$onDisk)"
              )
            }
      } yield ()
    }

  /**
    * Waits for a file to become available using a deadline-based approach.
    *
    * The poll interval (2 seconds) controls how often we check if the async download
    * has completed — it does NOT affect transfer speed (chunks stream independently).
    *
    * '''When to decrease''' (e.g. 500ms): latency-sensitive pipelines where blocks
    * carry small files (< 100 MB) and you want sub-second detection after download.
    * Trade-off: more frequent `Files.exists()` syscalls.
    *
    * '''When to increase''' (e.g. 5–10s): nodes handling many large files (> 1 GB)
    * concurrently, where reducing filesystem polling overhead matters more than
    * a few extra seconds of detection latency.
    *
    * Returns true if the file became available within the deadline, false otherwise.
    */
  def awaitFile(fileHash: String, timeout: FiniteDuration = syncTimeout): F[Boolean] = {
    val pollInterval = 2.seconds

    def poll(deadlineMs: Long): F[Boolean] =
      isFileAvailable(fileHash).flatMap {
        case true => true.pure[F]
        case false =>
          Sync[F].delay(System.currentTimeMillis()).flatMap { now =>
            if (now >= deadlineMs) false.pure[F]
            else Time[F].sleep(pollInterval) >> poll(deadlineMs)
          }
      }

    Sync[F].delay(System.currentTimeMillis() + timeout.toMillis).flatMap(poll)
  }

  /**
    * Waits for all given file hashes to become available within the timeout.
    * Uses a deadline-based approach suitable for multi-GB transfers.
    *
    * Poll interval tuning: same guidance as [[awaitFile]].
    * Returns the list of hashes that are still missing after the timeout.
    */
  def awaitFiles(
      fileHashes: List[String],
      timeout: FiniteDuration = syncTimeout
  ): F[List[String]] = {
    val pollInterval = 2.seconds

    def pollAll(remaining: List[String], deadlineMs: Long): F[List[String]] =
      if (remaining.isEmpty) List.empty[String].pure[F]
      else
        Sync[F].delay(System.currentTimeMillis()).flatMap { now =>
          if (now >= deadlineMs) remaining.pure[F]
          else
            remaining.filterA(h => isFileAvailable(h).map(!_)).flatMap { stillMissing =>
              if (stillMissing.isEmpty) List.empty[String].pure[F]
              else Time[F].sleep(pollInterval) >> pollAll(stillMissing, deadlineMs)
            }
        }

    Sync[F]
      .delay(System.currentTimeMillis() + timeout.toMillis)
      .flatMap(deadline => pollAll(fileHashes, deadline))
      .flatTap { stillMissing =>
        // Clean up stale download entries and orphaned .part files for timed-out downloads
        if (stillMissing.nonEmpty)
          stillMissing.traverse_ { hash =>
            downloads.modify(state => (state - hash, state.get(hash))).flatMap {
              case Some(ds) =>
                Sync[F].delay(java.nio.file.Files.deleteIfExists(ds.tempPath)) *>
                  Log[F].warn(
                    s"[FileRequester] Timeout cleanup: deleted ${ds.tempPath} for ${hash.take(16)}..."
                  )
              case None =>
                ().pure[F]
            }
          } else ().pure[F]
      }
  }

  def handleHasFile(peer: PeerNode, msg: HasFile): F[Unit] =
    for {
      isCompleted   <- completed.get.map(_.contains(msg.fileHash))
      onDisk        <- Sync[F].delay(java.nio.file.Files.exists(dataDir.resolve(msg.fileHash)))
      isDownloading <- downloads.get.map(_.contains(msg.fileHash))
      _ <- Log[F].info(
            s"[FileRequester] handleHasFile: hash=${msg.fileHash.take(16)}..., peer=$peer, " +
              s"completed=$isCompleted, onDisk=$onDisk, downloading=$isDownloading, dataDir=$dataDir"
          )
      _ <- if (!isCompleted && !onDisk && !isDownloading) {
            startDownload(peer, msg.fileHash)
          } else {
            Log[F].info(
              s"[FileRequester] Skipping download of ${msg.fileHash.take(16)}... (already handled)"
            )
          }
    } yield ()

  private def startDownload(peer: PeerNode, fileHash: String): F[Unit] =
    for {
      _ <- Log[F].info(
            s"[FileRequester] Starting download of file ${fileHash.take(16)}... from $peer"
          )
      tempPath = dataDir.resolve(s"$fileHash.part")
      digest   = new Blake2bDigest(256)
      _        <- downloads.update(_ + (fileHash -> DownloadState(tempPath, digest, 0L, None)))
      // Request first chunk
      _ <- requestChunk(peer, fileHash, 0L)
    } yield ()

  private def requestChunk(peer: PeerNode, fileHash: String, offset: Long): F[Unit] =
    Log[F].info(
      s"[FileRequester] Requesting chunk: hash=${fileHash.take(16)}..., offset=$offset, chunkSize=$chunkSize"
    ) *> {
      val req = FileRequest(fileHash, offset, chunkSize)
      TransportLayer[F].streamToPeer(peer, req.toProto)
    }

  def handleFilePacket(peer: PeerNode, msg: FilePacket): F[Unit] =
    Log[F].info(
      s"[FileRequester] handleFilePacket: hash=${msg.fileHash.take(16)}..., offset=${msg.offset}, " +
        s"dataSize=${msg.data.size}, eof=${msg.eof}, from=$peer"
    ) *> {
      // Atomically claim this offset: only one thread wins the race.
      // This prevents the corruption bug where two FilePacket responses at the same
      // offset both pass validation and both write to the .part file.
      downloads
        .modify { state =>
          state.get(msg.fileHash) match {
            case Some(ds) if msg.offset == ds.bytesReceived =>
              // Claim this offset by advancing bytesReceived atomically
              val newBytesReceived = ds.bytesReceived + msg.data.size
              val updated          = ds.copy(bytesReceived = newBytesReceived)
              (state + (msg.fileHash -> updated), Some((ds, newBytesReceived)))
            case _ =>
              // Either not found, or offset doesn't match (duplicate/late packet)
              (state, None)
          }
        }
        .flatMap {
          case Some((ds, newBytesReceived)) =>
            Log[F].info(
              s"[FileRequester] Packet progress: hash=${msg.fileHash.take(16)}..., " +
                s"bytesReceived=$newBytesReceived, expectedSize=${ds.expectedSize}"
            ) *> (
              // Size guard: abort if bytesReceived exceeds expected file size
              ds.expectedSize match {
                case Some(expected) if newBytesReceived > expected =>
                  Log[F].error(
                    s"File ${msg.fileHash} exceeded expected size ($newBytesReceived > $expected). " +
                      s"Aborting download and deleting .tmp."
                  ) >>
                    Sync[F].delay(java.nio.file.Files.deleteIfExists(ds.tempPath)) >>
                    downloads.update(_ - msg.fileHash)

                case _ =>
                  for {
                    _ <- Sync[F].delay {
                          // Append data to file and update digest
                          val data = msg.data.toByteArray
                          java.nio.file.Files.write(
                            ds.tempPath,
                            data,
                            StandardOpenOption.CREATE,
                            StandardOpenOption.APPEND
                          )
                          // IMPORTANT: `ds.digest` is a mutable Blake2bDigest object shared
                          // by reference with the updated map entry.
                          // The `modify` above updated `bytesReceived` but the `digest` field
                          // aliases the SAME JVM object in both the old `ds` and the new entry.
                          // This is safe because we hold exclusive ownership of this offset
                          // (only one thread wins the `modify`), but any refactor making
                          // DownloadState truly immutable must restructure this to update
                          // the digest inside the `modify` block.
                          ds.digest.update(data, 0, data.length)
                        }
                    _ <- if (msg.eof) {
                          finalizeDownload(
                            msg.fileHash,
                            ds.copy(bytesReceived = newBytesReceived)
                          )
                        } else {
                          requestChunk(peer, msg.fileHash, newBytesReceived)
                        }
                  } yield ()
              }
            )
          case None =>
            // Check if it's unknown or a duplicate
            downloads.get.flatMap { state =>
              state.get(msg.fileHash) match {
                case Some(ds) =>
                  Log[F].info(
                    s"[FileRequester] Ignoring duplicate packet for ${msg.fileHash.take(16)}...: " +
                      s"expected offset=${ds.bytesReceived}, got offset=${msg.offset}, from=$peer"
                  )
                case None =>
                  Log[F].warn(
                    s"[FileRequester] Received packet for UNKNOWN download ${msg.fileHash}"
                  )
              }
            }
        }
    }

  private def finalizeDownload(fileHash: String, ds: DownloadState): F[Unit] =
    for {
      hexDigest <- Sync[F].delay {
                    val hashBytes = new Array[Byte](32)
                    ds.digest.doFinal(hashBytes, 0)
                    hashBytes.map("%02x".format(_)).mkString
                  }
      _ <- if (hexDigest == fileHash) {
            val finalPath = dataDir.resolve(fileHash)
            Sync[F].delay(
              java.nio.file.Files
                .move(ds.tempPath, finalPath, java.nio.file.StandardCopyOption.REPLACE_EXISTING)
            ) >>
              completed.update(_ + fileHash) >>
              downloads.update(_ - fileHash) >>
              Log[F].info(s"File $fileHash download complete and verified.")
          } else {
            // TODO: When a PeerScore mechanism is added, apply a negative score penalty
            // to the serving peer here. This deters serving corrupted/tampered files.
            Log[F].error(
              s"File $fileHash download failed verification (calc: $hexDigest, exp: $fileHash). Deleting."
            ) >>
              Sync[F].delay(java.nio.file.Files.deleteIfExists(ds.tempPath)) >>
              downloads.update(_ - fileHash)
          }
    } yield ()

  def handleFileRequest(peer: PeerNode, msg: FileRequest): F[Unit] = {
    val filePath = dataDir.resolve(msg.fileHash)
    // Wrap Files.exists in Sync[F].delay to avoid blocking the calling fiber's thread
    Sync[F].delay(java.nio.file.Files.exists(filePath)).flatMap { exists =>
      Log[F].info(
        s"[FileRequester] handleFileRequest: hash=${msg.fileHash.take(16)}..., offset=${msg.offset}, " +
          s"chunkSize=${msg.chunkSize}, fileExists=$exists, from=$peer"
      ) *> (if (exists) {
              for {
                // Read chunk
                result <- Sync[F].delay {
                           val channel =
                             java.nio.channels.FileChannel.open(filePath, StandardOpenOption.READ)
                           val buf = java.nio.ByteBuffer.allocate(msg.chunkSize)
                           channel.position(msg.offset)
                           val read     = channel.read(buf)
                           val fileSize = channel.size()
                           channel.close()
                           val bytes = if (read > 0) {
                             buf.flip()
                             val arr = new Array[Byte](read)
                             buf.get(arr)
                             com.google.protobuf.ByteString.copyFrom(arr)
                           } else {
                             com.google.protobuf.ByteString.EMPTY
                           }
                           (bytes, fileSize)
                         }
                (bytes, fileSize) = result
                eof               = (msg.offset + bytes.size) >= fileSize
                _ <- Log[F].info(
                      s"[FileRequester] Serving chunk: hash=${msg.fileHash.take(16)}..., " +
                        s"offset=${msg.offset}, bytesRead=${bytes.size}, fileSize=$fileSize, eof=$eof"
                    )
                packet = FilePacket(msg.fileHash, msg.offset, bytes, eof)
                _      <- TransportLayer[F].streamToPeer(peer, packet.toProto)
              } yield ()
            } else {
              Log[F].warn(
                s"[FileRequester] Peer $peer requested file ${msg.fileHash} but NOT FOUND at $filePath"
              )
            })
    }
  }
}
