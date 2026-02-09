package coop.rchain.blockstorage.deploy

import cats.Functor
import cats.effect.Sync
import cats.syntax.all._
import com.google.protobuf.ByteString
import coop.rchain.blockstorage.dag.codecs.{codecDeploySignature, codecSignedDeployData}
import coop.rchain.casper.protocol.DeployData
import coop.rchain.crypto.signatures.Signed
import coop.rchain.shared.syntax._
import coop.rchain.store.{
  KeyValueStoreManager,
  KeyValueTypedStore,
  KeyValueTypedStoreCodec,
  LmdbKeyValueStore
}

final case class KeyValueDeployStorage[F[_]: Functor: Sync] private[deploy] (
    store: KeyValueTypedStore[F, ByteString, Signed[DeployData]],
    freeSpace: () => F[Long]
) extends DeployStorage[F] {
  def add(deploys: List[Signed[DeployData]]): F[Unit] =
    freeSpace().flatMap { space =>
      if (space < DeployStorage.MinFreeMemory)
        Sync[F].raiseError(
          new RuntimeException(
            s"Node is running low on storage space. Free space: $space, required: ${DeployStorage.MinFreeMemory}. Cannot accept new deploys."
          )
        )
      else
        store.put(deploys.map(d => (d.sig, d)))
    }

  def remove(deploys: List[Signed[DeployData]]): F[Int] =
    store.delete(deploys.map(_.sig))

  def readAll: F[Set[Signed[DeployData]]] =
    store.toMap.map(_.values.toSet)
}

object KeyValueDeployStorage {

  def apply[F[_]: Sync](kvm: KeyValueStoreManager[F]): F[KeyValueDeployStorage[F]] =
    for {
      store <- kvm.database("deploy_storage", codecDeploySignature, codecSignedDeployData)
      freeSpace = store match {
        case codec: KeyValueTypedStoreCodec[F, _, _] =>
          codec.store match {
            case lmdb: LmdbKeyValueStore[F] => () => lmdb.freeSpace
            case _                          => () => Runtime.getRuntime.freeMemory().pure[F]
          }
        case _ => () => Runtime.getRuntime.freeMemory().pure[F]
      }
    } yield KeyValueDeployStorage[F](store, freeSpace)

}
