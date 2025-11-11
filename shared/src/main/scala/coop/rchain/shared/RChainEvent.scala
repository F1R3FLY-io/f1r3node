package coop.rchain.shared

import cats.Applicative

final case class DeployEvent(id: String, cost: Long, deployer: String, errored: Boolean)

sealed trait RChainEvent {}

final case class BlockCreated(
    blockHash: String,
    parentHashes: List[String],
    justificationHashes: List[(String, String)],
    deploys: List[DeployEvent],
    creator: String,
    seqNum: Int
) extends RChainEvent

final case class BlockAdded(
    blockHash: String,
    parentHashes: List[String],
    justificationHashes: List[(String, String)],
    deploys: List[DeployEvent],
    creator: String,
    seqNum: Int
) extends RChainEvent

final case class BlockFinalised(
    blockHash: String,
    parentHashes: List[String],
    justificationHashes: List[(String, String)],
    deploys: List[DeployEvent],
    creator: String,
    seqNum: Int
) extends RChainEvent

object RChainEvent {
  def blockCreated(
      bs: String,
      parents: List[String],
      justifications: List[(String, String)],
      deploys: List[DeployEvent],
      creator: String,
      seqNum: Int
  ): RChainEvent =
    BlockCreated(bs, parents, justifications, deploys, creator, seqNum)

  def blockAdded(
      bs: String,
      parents: List[String],
      justifications: List[(String, String)],
      deploys: List[DeployEvent],
      creator: String,
      seqNum: Int
  ): RChainEvent =
    BlockAdded(bs, parents, justifications, deploys, creator, seqNum)

  def blockFinalised(
      bs: String,
      parents: List[String],
      justifications: List[(String, String)],
      deploys: List[DeployEvent],
      creator: String,
      seqNum: Int
  ): RChainEvent =
    BlockFinalised(bs, parents, justifications, deploys, creator, seqNum)
}

trait EventPublisher[F[_]] {
  def publish(e: => RChainEvent): F[Unit]
}

object EventPublisher {
  def apply[F[_]](implicit P: EventPublisher[F]): EventPublisher[F] = P

  def noop[F[_]: Applicative] = new EventPublisher[F] {
    override def publish(e: => RChainEvent): F[Unit] = Applicative[F].unit
  }
}
