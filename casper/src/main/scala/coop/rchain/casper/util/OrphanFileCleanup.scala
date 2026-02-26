package coop.rchain.casper.util

import coop.rchain.casper.protocol.DeployData
import coop.rchain.crypto.signatures.Signed
import coop.rchain.shared.Log

import cats.effect.Sync
import cats.syntax.all._

import java.nio.file.{Files, Path}

/**
  * Utility for cleaning up orphaned files when file-registration deploys
  * are removed from the mempool (expiration, eviction, or finalization).
  *
  * A file-registration deploy has a term like:
  * {{{
  * new ret, file(`rho:io:file`) in { file!("register", "<64-char-hex-hash>", ...) }
  * }}}
  *
  * When such a deploy is removed, the physical file and its `.meta.json` sidecar
  * should be deleted — unless another pending deploy still references the same hash.
  */
object OrphanFileCleanup {

  // Matches a 64-character lowercase hex string (Blake2b-256 hash)
  private val FileHashPattern = """([a-f0-9]{64})""".r.unanchored

  /**
    * Checks whether a deploy's term matches the file-registration pattern.
    * Uses simple string matching: the term must reference the `rho:io:file`
    * system channel and contain `"register"` with a valid file hash.
    */
  def isFileRegistrationDeploy(d: DeployData): Boolean =
    d.term.contains("rho:io:file") &&
      d.term.contains("\"register\"") &&
      FileHashPattern.findFirstIn(d.term).isDefined

  /**
    * Extracts the 64-char hex file hash from a file-registration deploy term.
    * Returns `None` if the term does not contain a valid hash.
    */
  def extractFileHash(d: DeployData): Option[String] =
    FileHashPattern.findFirstIn(d.term)

  // Matches an integer immediately following the hash and a comma, e.g.
  // file!("register", "<hash>", 1048576, ...) → 1048576
  private val FileSizePattern =
    """[a-f0-9]{64}"\s*,\s*(\d+)""".r.unanchored

  /**
    * Extracts the file size from a file-registration deploy term.
    * Returns 0 if the term does not contain a recognizable file size.
    */
  def extractFileSize(d: DeployData): Long =
    FileSizePattern.findFirstMatchIn(d.term).map(_.group(1).toLong).getOrElse(0L)

  /**
    * Cleans up orphaned files for deploys that have been removed from the mempool.
    *
    * For each removed deploy that is a file-registration deploy:
    *   1. Extract the file hash from the deploy term
    *   2. Check if any remaining deploy in the pool references the same hash
    *   3. If no cross-reference exists, delete the physical file and `.meta.json`
    *
    * @param removedDeploys deploys that were just removed from the mempool
    * @param remainingDeploys deploys still in the mempool after removal
    * @param fileReplicationDir the `file-replication/` directory
    */
  def cleanupOrphanedFiles[F[_]: Sync: Log](
      removedDeploys: Iterable[Signed[DeployData]],
      remainingDeploys: Iterable[Signed[DeployData]],
      fileReplicationDir: Path
  ): F[Unit] = {
    // Collect all file hashes still referenced by remaining deploys
    val remainingHashes: Set[String] = remainingDeploys
      .filter(d => isFileRegistrationDeploy(d.data))
      .flatMap(d => extractFileHash(d.data))
      .toSet

    // For each removed file-registration deploy, delete if no cross-reference
    removedDeploys.toList
      .filter(d => isFileRegistrationDeploy(d.data))
      .flatMap(d => extractFileHash(d.data))
      .distinct
      .traverse_ { hash =>
        if (remainingHashes.contains(hash))
          Log[F].info(
            s"Orphan cleanup: skipping file $hash (still referenced by another deploy)"
          )
        else
          deleteFileAndMeta[F](fileReplicationDir, hash)
      }
  }

  /**
    * Deletes the physical file and its `.meta.json` sidecar from the given directory.
    * Logs the result. No-op if the files don't exist.
    */
  def deleteFileAndMeta[F[_]: Sync: Log](dir: Path, hash: String): F[Unit] =
    Sync[F]
      .delay {
        val dataFile    = dir.resolve(hash)
        val metaFile    = dir.resolve(s"$hash.meta.json")
        val dataDeleted = Files.deleteIfExists(dataFile)
        val metaDeleted = Files.deleteIfExists(metaFile)
        (dataDeleted, metaDeleted)
      }
      .flatMap {
        case (dataDeleted, metaDeleted) =>
          Log[F].info(
            s"Orphan cleanup: hash=$hash data=${if (dataDeleted) "deleted" else "not-found"} " +
              s"meta=${if (metaDeleted) "deleted" else "not-found"}"
          )
      }
}
