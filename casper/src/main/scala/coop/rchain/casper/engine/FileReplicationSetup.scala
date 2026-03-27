package coop.rchain.casper.engine

import cats.effect.{Sync, Timer}
import cats.syntax.all._
import coop.rchain.casper.{CasperShardConf, PrettyPrinter}
import coop.rchain.casper.protocol.BlockMessage
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
  def create[F[_]: cats.effect.Concurrent: Log: Time: TransportLayer: RPConfAsk: ConnectionsCell: Timer](
      conf: CasperShardConf
  ): F[(FileRequester[F], DACallback[F])] =
    for {
      _ <- if (conf.fileConf.fileReplicationDir.isEmpty)
            Log[F].warn(
              "[FileReplicationSetup] No fileReplicationDir configured, " +
                "using relative path 'file-replication' — set fileReplicationDir in config for production!"
            )
          else Sync[F].unit
      dataDir <- Sync[F].delay {
                  val dir = conf.fileConf.fileReplicationDir
                    .getOrElse(java.nio.file.Paths.get("file-replication"))
                  if (!dir.toFile.exists()) dir.toFile.mkdirs()
                  dir
                }
      fileRequester = new FileRequester[F](
        dataDir,
        conf.fileConf.fileChunkSize,
        conf.fileConf.fileSyncTimeout,
        conf.fileConf.fileStallTimeout,
        conf.fileConf.fileMaxRetries,
        conf.fileConf.fileMaxBackoff
      )
      daCallback: DACallback[F] = (block: BlockMessage, missingHashes: List[String]) =>
        for {
          _ <- Log[F].info(
                s"[FileReplicationSetup] DA-gate: ${missingHashes.size} file(s) missing for block " +
                  s"${PrettyPrinter.buildString(block.blockHash)}, triggering P2P fetch"
              )
          peers <- ConnectionsCell[F].read
          // Limit to at most 3 peers to avoid flooding the network.
          // Prefer the block sender if identifiable, plus random fallbacks.
          selectedPeers = if (peers.size <= 3) peers.toList
          else scala.util.Random.shuffle(peers.toList).take(3)
          _ <- Log[F].info(
                s"[FileReplicationSetup] Requesting files from ${selectedPeers.size} of ${peers.size} peers"
              )
          _            <- selectedPeers.traverse_(peer => fileRequester.requestFiles(peer, missingHashes))
          stillMissing <- fileRequester.awaitFiles(missingHashes, conf.fileConf.fileSyncTimeout)
          _ <- Log[F].info(
                s"[FileReplicationSetup] awaitFiles returned, stillMissing=${stillMissing.size}"
              )
        } yield stillMissing
    } yield (fileRequester, daCallback)
}
