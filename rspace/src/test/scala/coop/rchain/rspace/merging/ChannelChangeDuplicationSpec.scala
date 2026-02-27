package coop.rchain.rspace.merging

import cats.syntax.all._
import coop.rchain.rspace._
import coop.rchain.rspace.hashing.Blake2b256Hash
import coop.rchain.rspace.history.HistoryReaderBinary
import coop.rchain.rspace.internal.{Datum, WaitingContinuation}
import coop.rchain.rspace.merger.MergingLogic.NumberChannelsDiff
import coop.rchain.rspace.merger.{ChannelChange, StateChange, StateChangeMerger}
import coop.rchain.rspace.serializers.ScodecSerialize.{DatumB, JoinsB, WaitingContinuationB}
import coop.rchain.rspace.trace.{Consume, Produce}
import coop.rchain.shared.scalatestcontrib._
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Matchers}
import scodec.bits.ByteVector

import scala.collection.SortedSet

/**
  * Reproduces the ChannelChange.combine() duplication bug.
  *
  * When two sibling blocks (same parent/LCA) execute identical system deploys
  * on the same base state, they produce identical ChannelChange objects. The
  * combine operation concatenates their added/removed vectors without dedup,
  * causing StateChangeMerger.mkTrieAction to produce channels with duplicate
  * data.
  *
  * Trigger scenario (from CI):
  *   LCA state: channel has Seq(A)
  *   Block V2 (parent=LCA): CloseBlockDeploy removes A, adds B
  *   Block V3 (parent=LCA): CloseBlockDeploy removes A, adds B
  *   Combined: ChannelChange(added=Vector(B,B), removed=Vector(A,A))
  *   mkTrieAction formula: (Seq(A) diff Seq(A,A)) ++ Seq(B,B) = Seq(B,B)
  *
  */
class ChannelChangeDuplicationSpec extends FlatSpec with Matchers {

  private val datumA = ByteVector.fromValidHex("aa" * 32)
  private val datumB = ByteVector.fromValidHex("bb" * 32)

  private def mkHash(byte: Byte): Blake2b256Hash =
    Blake2b256Hash.fromByteArray(Array.fill(32)(byte))

  private val dummyHash = mkHash(0xff.toByte)

  "ChannelChange.combine" should "not duplicate when combining identical changes from sibling blocks" in {
    // Two sibling blocks both transition channel state: remove A, add B
    val change   = ChannelChange(added = Vector(datumB), removed = Vector(datumA))
    val combined = ChannelChange.combine(change, change)

    // Applying mkTrieAction formula: (init diff removed) ++ added
    // With init = Seq(A), correct result is Seq(B)
    val init         = Seq(datumA)
    val mergedResult = (init diff combined.removed) ++ combined.added
    mergedResult shouldBe Seq(datumB)
  }

  "StateChangeMerger.computeTrieActions" should "not duplicate data when merging identical sibling changes" in effectTest {
    val channelHash = mkHash(0x01)

    val baseReader = new StubHistoryReaderBinary(
      dataMap = Map(channelHash -> Seq(datumA))
    )

    // Two sibling blocks both change A -> B on the same channel
    val branchChange = StateChange(
      datumsChanges = Map(channelHash -> ChannelChange(Vector(datumB), Vector(datumA))),
      kontChanges = Map.empty,
      consumeChannelsToJoinSerializedMap = Map.empty
    )
    val combined = StateChange.combine(branchChange, branchChange)

    val mergeableChs: NumberChannelsDiff = Map.empty
    val noOverride = (
        _: Blake2b256Hash,
        _: ChannelChange[ByteVector],
        _: NumberChannelsDiff
    ) => Task.pure(Option.empty[HotStoreTrieAction])

    for {
      actions <- StateChangeMerger.computeTrieActions(
                  combined,
                  baseReader,
                  mergeableChs,
                  noOverride
                )
    } yield {
      actions.size shouldBe 1
      actions.head shouldBe a[TrieInsertBinaryProduce]
      val insert = actions.head.asInstanceOf[TrieInsertBinaryProduce]
      insert.hash shouldBe channelHash
      insert.data shouldBe Seq(datumB)
    }
  }

  private class StubHistoryReaderBinary(
      dataMap: Map[Blake2b256Hash, Seq[ByteVector]] = Map.empty,
      kontMap: Map[Blake2b256Hash, Seq[ByteVector]] = Map.empty,
      joinsMap: Map[Blake2b256Hash, Seq[ByteVector]] = Map.empty
  ) extends HistoryReaderBinary[Task, Any, Any, Any, Any] {

    override def getData(key: Blake2b256Hash): Task[Seq[DatumB[Any]]] =
      Task.pure(
        dataMap.getOrElse(key, Seq.empty).map { bv =>
          val source =
            Produce.fromHash(key, dummyHash, persistent = false, isDeterministic = true, Seq.empty)
          val datum: Datum[Any] = Datum(().asInstanceOf[Any], persist = false, source)
          DatumB(datum, bv)
        }
      )

    override def getContinuations(key: Blake2b256Hash): Task[Seq[WaitingContinuationB[Any, Any]]] =
      Task.pure(
        kontMap.getOrElse(key, Seq.empty).map { bv =>
          val source = Consume.fromHash(Seq(key), dummyHash, persistent = false)
          val wk: WaitingContinuation[Any, Any] =
            WaitingContinuation(
              Seq.empty[Any],
              ().asInstanceOf[Any],
              persist = false,
              SortedSet.empty[Int],
              source
            )
          WaitingContinuationB(wk, bv)
        }
      )

    override def getJoins(key: Blake2b256Hash): Task[Seq[JoinsB[Any]]] =
      Task.pure(
        joinsMap.getOrElse(key, Seq.empty).map { bv =>
          JoinsB(Seq.empty[Any], bv)
        }
      )
  }
}
