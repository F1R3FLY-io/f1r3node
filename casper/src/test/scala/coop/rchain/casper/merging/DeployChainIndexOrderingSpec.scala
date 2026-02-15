package coop.rchain.casper.merging

import com.google.protobuf.ByteString
import coop.rchain.rspace.hashing.Blake2b256Hash
import coop.rchain.rspace.merger.{EventLogIndex, StateChange}
import org.scalatest.{FlatSpec, Matchers}

/**
  * Tests that DeployChainIndex ordering provides a total, deterministic order.
  * This is critical for merge determinism across validators.
  */
class DeployChainIndexOrderingSpec extends FlatSpec with Matchers {

  private def mkHash(byte: Byte): Blake2b256Hash =
    Blake2b256Hash.fromByteArray(Array.fill(32)(byte))

  private def mkDeployId(byte: Byte): ByteString =
    ByteString.copyFrom(Array.fill(64)(byte))

  private def mkIndex(
      postState: Byte,
      preState: Byte,
      deployIds: Byte*
  ): DeployChainIndex =
    DeployChainIndex(
      deploysWithCost = deployIds.map(b => DeployIdWithCost(mkDeployId(b), 0L)).toSet,
      preStateHash = mkHash(preState),
      postStateHash = mkHash(postState),
      eventLogIndex = EventLogIndex.empty,
      stateChanges = StateChange.empty,
      hashCodeVal = deployIds.hashCode()
    )

  "DeployChainIndex ordering" should "order by postStateHash primarily" in {
    val a = mkIndex(postState = 1, preState = 0, 10)
    val b = mkIndex(postState = 2, preState = 0, 10)
    Ordering[DeployChainIndex].compare(a, b) should be < 0
  }

  it should "use preStateHash as tiebreaker when postStateHash matches" in {
    val a = mkIndex(postState = 1, preState = 1, 10)
    val b = mkIndex(postState = 1, preState = 2, 10)
    Ordering[DeployChainIndex].compare(a, b) should not be 0
  }

  it should "use deploy IDs as final tiebreaker" in {
    val a = mkIndex(postState = 1, preState = 1, 10)
    val b = mkIndex(postState = 1, preState = 1, 20)
    Ordering[DeployChainIndex].compare(a, b) should not be 0
  }

  it should "compare equal for identical instances" in {
    val a = mkIndex(postState = 1, preState = 1, 10, 20)
    val b = mkIndex(postState = 1, preState = 1, 10, 20)
    Ordering[DeployChainIndex].compare(a, b) shouldBe 0
  }

  it should "be antisymmetric" in {
    val a   = mkIndex(postState = 1, preState = 1, 10)
    val b   = mkIndex(postState = 1, preState = 1, 20)
    val cmp = Ordering[DeployChainIndex].compare(a, b)
    Ordering[DeployChainIndex].compare(b, a) shouldBe -cmp
  }

  it should "produce consistent sorted order for a collection" in {
    val items = List(
      mkIndex(postState = 3, preState = 1, 10),
      mkIndex(postState = 1, preState = 2, 20),
      mkIndex(postState = 2, preState = 1, 30),
      mkIndex(postState = 1, preState = 1, 40),
      mkIndex(postState = 1, preState = 1, 10)
    )
    val sorted1 = items.sorted
    val sorted2 = items.reverse.sorted
    sorted1.map(_.postStateHash) shouldBe sorted2.map(_.postStateHash)
    sorted1.map(_.preStateHash) shouldBe sorted2.map(_.preStateHash)
    sorted1.zip(sorted2).foreach {
      case (a, b) => Ordering[DeployChainIndex].compare(a, b) shouldBe 0
    }
  }
}
