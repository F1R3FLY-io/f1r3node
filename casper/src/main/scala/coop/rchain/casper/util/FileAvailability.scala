package coop.rchain.casper.util

import cats.effect.Sync
import cats.syntax.all._
import coop.rchain.casper.protocol.{BlockMessage, DeployData}
import coop.rchain.casper.{BlockStatus, InvalidBlock, ValidBlockProcessing}
import coop.rchain.crypto.signatures.Signed
import coop.rchain.shared.Log

import java.nio.file.{Files, Path}

/**
  * Utility for checking file data availability before block validation.
  *
  * When a validator receives a block containing file-registration deploys, it must verify
  * that the referenced files exist locally before proceeding with validation. If any files
  * are missing, the validator should fetch them from the block proposer via P2P.
  */
object FileAvailability {

  /**
    * Extracts file hashes from all file-registration deploys in a block.
    *
    * A file-registration deploy has a term referencing `rho:io:file` with `"register"`
    * and a 64-character hex hash. Reuses the same pattern as [[OrphanFileCleanup]].
    *
    * @param block the block to scan
    * @return list of file hashes referenced by file-registration deploys
    */
  def extractFileHashes(block: BlockMessage): List[String] = {
    val allDeploys  = block.body.deploys.map(_.deploy)
    val fileDeploys = allDeploys.filter(d => OrphanFileCleanup.isFileRegistrationDeploy(d.data))
    fileDeploys.flatMap(d => OrphanFileCleanup.extractFileHash(d.data)).distinct.toList
  }

  /**
    * Checks whether all files referenced by a block's file-registration deploys
    * exist locally in the file replication directory.
    *
    * @param block the block to check
    * @param fileReplicationDir the `file-replication/` directory
    * @return `Right(Valid)` if all files are present or the block has no file deploys,
    *         `Left(MissingFileData)` if any file is missing
    */
  def checkFileAvailability[F[_]: Sync: Log](
      block: BlockMessage,
      fileReplicationDir: Path
  ): F[ValidBlockProcessing] = {
    val hashes = extractFileHashes(block)
    if (hashes.isEmpty)
      BlockStatus.valid.asRight[coop.rchain.casper.BlockError].pure[F]
    else {
      val missing = hashes.filterNot(h => Files.exists(fileReplicationDir.resolve(h)))
      if (missing.isEmpty)
        BlockStatus.valid.asRight[coop.rchain.casper.BlockError].pure[F]
      else
        Log[F]
          .warn(
            s"Block ${block.blockHash.toStringUtf8.take(10)}... " +
              s"references ${missing.size} missing file(s): ${missing.mkString(", ")}"
          )
          .as(BlockStatus.missingFileData.asLeft[coop.rchain.casper.ValidBlock])
    }
  }

  /**
    * Returns the list of file hashes from a block that are not yet present locally.
    * Wrapped in F[_] to avoid blocking I/O in a pure context.
    */
  def findMissingFiles[F[_]: Sync](block: BlockMessage, fileReplicationDir: Path): F[List[String]] =
    Sync[F].delay {
      val hashes = extractFileHashes(block)
      hashes.filterNot(h => Files.exists(fileReplicationDir.resolve(h)))
    }
}
