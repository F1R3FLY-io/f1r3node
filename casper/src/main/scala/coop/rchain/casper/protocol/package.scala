package coop.rchain.casper

import coop.rchain.comm.protocol.routing.Packet

package object protocol extends CasperMessageProtocol {
  import PacketTypeTag._

  def toCasperMessageProto(packet: Packet): PacketParseResult[CasperMessageProto] =
    PacketTypeTag
      .withNameOption(packet.typeId)
      .map {
        case BlockHashMessage         => convert[BlockHashMessage.type](packet)
        case BlockMessage             => convert[BlockMessage.type](packet)
        case ApprovedBlock            => convert[ApprovedBlock.type](packet)
        case ApprovedBlockRequest     => convert[ApprovedBlockRequest.type](packet)
        case BlockRequest             => convert[BlockRequest.type](packet)
        case HasBlockRequest          => convert[HasBlockRequest.type](packet)
        case HasBlock                 => convert[HasBlock.type](packet)
        case ForkChoiceTipRequest     => convert[ForkChoiceTipRequest.type](packet)
        case BlockApproval            => convert[BlockApproval.type](packet)
        case UnapprovedBlock          => convert[UnapprovedBlock.type](packet)
        case NoApprovedBlockAvailable => convert[NoApprovedBlockAvailable.type](packet)
        // Last finalized state messages
        case StoreItemsMessageRequest => convert[StoreItemsMessageRequest.type](packet)
        case StoreItemsMessage        => convert[StoreItemsMessage.type](packet)
        // File Replication
        case HasFile     => convert[HasFile.type](packet)
        case FileRequest => convert[FileRequest.type](packet)
        case FilePacket  => convert[FilePacket.type](packet)
      }
      .getOrElse(PacketParseResult.IllegalPacket(s"Unrecognized typeId: ${packet.typeId}"))

  @inline def convert[Tag <: PacketTypeTag](msg: Packet)(
      implicit fromPacket: FromPacket[Tag]
  ): PacketParseResult[fromPacket.To] = fromPacket.parseFrom(msg)

}
