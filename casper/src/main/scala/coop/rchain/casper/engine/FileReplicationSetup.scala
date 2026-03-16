package coop.rchain.casper.engine

import cats.effect.Sync
import cats.syntax.all._
import coop.rchain.casper.{CasperShardConf, PrettyPrinter}
import coop.rchain.casper.protocol.BlockMessage
import coop.rchain.casper.util.OrphanFileCleanup
import coop.rchain.comm.rp.Connect.{ConnectionsCell, RPConfAsk}
import coop.rchain.comm.transport.TransportLayer
import coop.rchain.shared.{Log, Time}

import java.nio.file.Path
import scala.concurrent.duration.FiniteDuration

/**
  * Shared setup for FileRequester and the DA-gating callback.
  */
object FileReplicationSetup {

  /**
    * DA-gating callback type: given a block and list of missing file hashes,
    * attempt to fetch them via P2P and return any still-missing hashes.
    */
  type DACallback[F[_]] = (BlockMessage, List[String]) => F[List[String]]

  /**
    * Creates a [[FileRequester]] and a DA-gating callback that broadcasts
    * file requests to all connected peers and awaits with a bounded timeout.
    *
    * @return (FileRequester, DACallback) tuple
    */
  def create[F[_]: cats.effect.Concurrent: Log: Time: TransportLayer: RPConfAsk: ConnectionsCell](
      conf: CasperShardConf
  ): F[(FileRequester[F], DACallback[F])] =
    for {
      dataDir <- Sync[F].delay {
                  val dir = conf.fileConf.fileReplicationDir.getOrElse(
                    java.nio.file.Paths.get("file-replication")
                  )
                  if (!dir.toFile.exists()) dir.toFile.mkdirs()
                  dir
                }
      fileRequester = new FileRequester[F](
        dataDir,
        conf.fileConf.fileChunkSize,
        conf.fileConf.fileSyncTimeout
      )
      daCallback: DACallback[F] = (block: BlockMessage, missingHashes: List[String]) =>
        for {
          _ <- Log[F].info(
                s"[FileReplicationSetup] DA-gate: ${missingHashes.size} file(s) missing for block " +
                  s"${PrettyPrinter.buildString(block.blockHash)}, triggering P2P fetch"
              )
          peers <- ConnectionsCell[F].read
          _ <- Log[F].info(
                s"[FileReplicationSetup] Requesting files from ${peers.size} peers"
              )
          _            <- peers.toList.traverse_(peer => fileRequester.requestFiles(peer, missingHashes))
          stillMissing <- fileRequester.awaitFiles(missingHashes, conf.fileConf.fileSyncTimeout)
          _ <- Log[F].info(
                s"[FileReplicationSetup] awaitFiles returned, stillMissing=${stillMissing.size}"
              )
          // Write .meta.json with deployId for each successfully downloaded file.
          // This enables the download finalization checker to verify via the DAG
          // (BlockAPI.findDeploy + isFinalized) instead of bypassing the check.
          downloaded = missingHashes.filterNot(stillMissing.contains)
          _ <- downloaded.traverse_ { hash =>
                val deployIdOpt = block.body.deploys.collectFirst {
                  case pd if OrphanFileCleanup.extractFileHash(pd.deploy.data).contains(hash) =>
                    pd.deploy.sig.toByteArray.map("%02x".format(_)).mkString
                }
                deployIdOpt.traverse_ { deployId =>
                  Sync[F].delay {
                    val metaPath = dataDir.resolve(s"$hash.meta.json")
                    val json     = s"""{"deployId":"$deployId"}"""
                    java.nio.file.Files.write(metaPath, json.getBytes("UTF-8"))
                  } *> Log[F].info(
                    s"[FileReplicationSetup] Wrote meta.json for ${hash.take(16)}... " +
                      s"with deployId=${deployId.take(16)}..."
                  )
                }
              }
        } yield stillMissing
    } yield (fileRequester, daCallback)
}
