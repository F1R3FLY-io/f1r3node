package coop.rchain.shared

import java.nio.file.Path

/**
  * Centralised file-hash validation helpers shared by the P2P, gRPC, and
  * upload/download layers.
  *
  * A valid file hash is exactly 64 lowercase hex characters (Blake2b-256).
  */
object FileHashValidation {

  /** Pre-compiled regex — avoids re-compiling on every call. */
  val HashPattern = "^[a-f0-9]{64}$".r

  /** Returns `true` iff `hash` is exactly 64 lowercase hex chars. */
  def isValidFileHash(hash: String): Boolean =
    hash != null && HashPattern.findFirstMatchIn(hash).isDefined

  /**
    * Validates `hash` format *and* ensures the resolved path stays inside
    * `dataDir`, preventing path-traversal attacks.
    *
    * @return `Right(resolvedPath)` on success, `Left(reason)` on failure.
    */
  def validateFileHashPath(dataDir: Path, hash: String): Either[String, Path] =
    if (!isValidFileHash(hash))
      Left(s"Invalid file hash format: ${Option(hash).map(_.take(20)).getOrElse("null")}")
    else {
      val resolved   = dataDir.resolve(hash).normalize()
      val normalised = dataDir.normalize()
      if (resolved.startsWith(normalised)) Right(resolved)
      else Left(s"Path traversal detected: hash=$hash")
    }
}
