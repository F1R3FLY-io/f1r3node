package coop.rchain.node.web

import cats.effect.Sync
import cats.implicits._
import coop.rchain.comm.discovery.NodeDiscovery
import coop.rchain.comm.rp.Connect.{ConnectionsCell, RPConfAsk}
import org.http4s.HttpRoutes

object StatusInfo {

  final case class PeerInfo(
      address: String,
      nodeId: String,
      host: String,
      protocolPort: Int,
      discoveryPort: Int,
      isConnected: Boolean
  )

  final case class Status(
      address: String,
      version: String,
      peers: Int,
      nodes: Int,
      peerList: List[PeerInfo] = List.empty
  )

  def status[F[_]: Sync: ConnectionsCell: NodeDiscovery: RPConfAsk]: F[Status] =
    for {
      version <- Sync[F].delay(VersionInfo.get)
      peers   <- ConnectionsCell[F].read
      nodes   <- NodeDiscovery[F].peers
      rpConf  <- RPConfAsk[F].ask
    } yield {
      // Create a set of connected peer IDs for quick lookup
      val connectedIds = peers.map(_.id.key).toSet

      // Convert PeerNode to PeerInfo
      def peerNodeToInfo(peerNode: coop.rchain.comm.PeerNode, isConnected: Boolean): PeerInfo =
        PeerInfo(
          address = peerNode.toAddress,
          nodeId = peerNode.id.toString,
          host = peerNode.endpoint.host,
          protocolPort = peerNode.endpoint.tcpPort,
          discoveryPort = peerNode.endpoint.udpPort,
          isConnected = isConnected
        )

      // Combine discovered peers and active connections with deduplication
      val combinedPeers = nodes
        .map(node => node.id.key -> peerNodeToInfo(node, connectedIds.contains(node.id.key)))
        .toMap
        .values
        .toList

      Status(rpConf.local.toAddress, version, peers.length, nodes.length, combinedPeers)
    }

  def service[F[_]: Sync: ConnectionsCell: NodeDiscovery: RPConfAsk]: HttpRoutes[F] = {
    import io.circe.generic.auto._
    import io.circe.syntax._
    import org.http4s.circe.CirceEntityEncoder._
    val dsl = org.http4s.dsl.Http4sDsl[F]
    import dsl._

    HttpRoutes.of[F] {
      case GET -> Root => Ok(status.map(_.asJson))
    }
  }
}
