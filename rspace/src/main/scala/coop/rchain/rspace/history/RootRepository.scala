package coop.rchain.rspace.history

import cats.Applicative
import cats.syntax.all._
import cats.effect.Sync
import coop.rchain.rspace.hashing.Blake2b256Hash
import coop.rchain.rspace.history.History.emptyRootHash
import coop.rchain.shared.Log

import scala.Function._

class RootRepository[F[_]: Sync: Log](
    rootsStore: RootsStore[F]
) {
  private def unknownRootError(root: Blake2b256Hash) =
    new RuntimeException(s"unknown root: ${root.bytes.toHex.take(16)}...")

  def commit(root: Blake2b256Hash): F[Unit] =
    Log[F].debug(s"[RootRepository] commit $root") *>
      rootsStore.recordRoot(root)

  def currentRoot(): F[Blake2b256Hash] =
    rootsStore.currentRoot().flatMap {
      case None =>
        Log[F].debug(
          s"[RootRepository] currentRoot: empty store, recording $emptyRootHash"
        ) *> rootsStore.recordRoot(emptyRootHash).as(emptyRootHash)
      case Some(root) =>
        Log[F].debug(
          s"[RootRepository] currentRoot: $root"
        ) *> Applicative[F].pure(root)
    }

  def validateAndSetCurrentRoot(root: Blake2b256Hash): F[Unit] =
    rootsStore.validateAndSetCurrentRoot(root).flatMap {
      case None =>
        Log[F].error(
          s"[RootRepository] validateAndSetCurrentRoot FAILED: $root not in roots store"
        ) *> Sync[F].raiseError[Unit](unknownRootError(root))
      case Some(_) =>
        Log[F].debug(
          s"[RootRepository] validateAndSetCurrentRoot OK: $root"
        )
    }

}
