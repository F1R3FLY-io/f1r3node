package coop.rchain.casper.merging

import cats.effect.Concurrent
import cats.syntax.all._
import com.google.protobuf.ByteString
import coop.rchain.rspace.hashing.Blake2b256Hash
import coop.rchain.rspace.history.HistoryRepository
import coop.rchain.rspace.merger._
import coop.rchain.rspace.syntax._

import scodec.bits.ByteVector

import java.util.Objects
import scala.util.Random

final case class DeployIdWithCost(id: ByteString, cost: Long)

/** index of deploys depending on each other inside a single block (state transition) */
final case class DeployChainIndex(
    deploysWithCost: Set[DeployIdWithCost],
    preStateHash: Blake2b256Hash,
    postStateHash: Blake2b256Hash,
    eventLogIndex: EventLogIndex,
    stateChanges: StateChange,
    private val hashCodeVal: Int
) {
  // equals and hash overrides are required to make conflict resolution faster, particularly rejection options calculation
  override def equals(obj: Any): Boolean = obj match {
    case that: DeployChainIndex => that.deploysWithCost == this.deploysWithCost
    case _                      => false
  }
  // caching hash code helps a lot to increase performance of computing rejection options
  // TODO mysterious speedup of merging benchmark when setting this to some fixed value
  override def hashCode(): Int = hashCodeVal
}

object DeployChainIndex {

  // Total ordering for deterministic processing across validators.
  // Primary sort by postStateHash, with tiebreakers by preStateHash and then
  // lexicographic comparison of sorted deploy IDs. This prevents ambiguity
  // when two deploy chains produce the same post-state hash.
  implicit val ord: Ordering[DeployChainIndex] = (a: DeployChainIndex, b: DeployChainIndex) => {
    val postCmp = Ordering[Blake2b256Hash].compare(a.postStateHash, b.postStateHash)
    if (postCmp != 0) postCmp
    else {
      val preCmp = Ordering[Blake2b256Hash].compare(a.preStateHash, b.preStateHash)
      if (preCmp != 0) preCmp
      else {
        // Tiebreak by sorted deploy IDs (unique per chain)
        val aIds   = a.deploysWithCost.toVector.map(d => ByteVector.view(d.id.toByteArray)).sorted
        val bIds   = b.deploysWithCost.toVector.map(d => ByteVector.view(d.id.toByteArray)).sorted
        val lenCmp = aIds.length.compareTo(bIds.length)
        if (lenCmp != 0) lenCmp
        else {
          aIds
            .zip(bIds)
            .map { case (ai, bi) => Ordering[ByteVector].compare(ai, bi) }
            .find(_ != 0)
            .getOrElse(0)
        }
      }
    }
  }

  def apply[F[_]: Concurrent, C, P, A, K](
      deploys: Set[DeployIndex],
      preStateHash: Blake2b256Hash,
      postStateHash: Blake2b256Hash,
      historyRepository: HistoryRepository[F, C, P, A, K]
  ): F[DeployChainIndex] = {

    val deploysWithCost = deploys.map(v => DeployIdWithCost(v.deployId, v.cost))
    val eventLogIndex   = deploys.map(_.eventLogIndex).toList.combineAll

    for {
      preHistoryReader  <- historyRepository.getHistoryReader(preStateHash)
      preStateReader    = preHistoryReader.readerBinary
      postHistoryReader <- historyRepository.getHistoryReader(postStateHash)
      postStateReader   = postHistoryReader.readerBinary

      stateChanges <- StateChange[F, C, P, A, K](
                       preStateReader = preStateReader,
                       postStateReader = postStateReader,
                       eventLogIndex,
                       historyRepository.getSerializeC
                     )
    } yield DeployChainIndex(
      deploysWithCost,
      preStateHash,
      postStateHash,
      eventLogIndex,
      stateChanges,
      Objects.hash(deploysWithCost.map(_.id).toSeq: _*)
    )
  }

  def random: Iterator[DeployChainIndex] =
    Iterator.continually[Int](Random.nextInt(10) + 1).map { size =>
      val deployIds = Range(0, size)
        .map(
          _ => ByteString.copyFrom(Array.fill(64)((scala.util.Random.nextInt(256) - 128).toByte))
        )
      DeployChainIndex(
        deployIds.map(id => DeployIdWithCost(id, 0)).toSet,
        Blake2b256Hash.fromByteArray(new Array[Byte](32)),
        Blake2b256Hash.fromByteArray(new Array[Byte](32)),
        EventLogIndex.empty,
        StateChange.empty,
        Objects.hash(deployIds.map(id => DeployIdWithCost(id, 0)).map(_.id): _*)
      )
    }
}
