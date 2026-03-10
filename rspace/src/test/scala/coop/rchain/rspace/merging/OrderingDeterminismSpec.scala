package coop.rchain.rspace.merging

import coop.rchain.rspace.hashing.Blake2b256Hash
import coop.rchain.rspace.merger.EventLogIndex
import coop.rchain.rspace.merger.MergingLogic._
import org.scalatest.{FlatSpec, Matchers}

/**
  * Tests that ordering implementations used in the merge layer are
  * deterministic and collision-free. These orderings are critical for
  * ensuring all validators compute identical merge results.
  */
class OrderingDeterminismSpec extends FlatSpec with Matchers {

  private def mkHash(byte: Byte): Blake2b256Hash =
    Blake2b256Hash.fromByteArray(Array.fill(32)(byte))

  "EventLogIndex ordering" should "distinguish instances with different numberChannelsData keys" in {
    val a = EventLogIndex.empty.copy(numberChannelsData = Map(mkHash(1) -> 100L))
    val b = EventLogIndex.empty.copy(numberChannelsData = Map(mkHash(2) -> 100L))
    Ordering[EventLogIndex].compare(a, b) should not be 0
  }

  it should "distinguish instances with same keys but different values" in {
    val a = EventLogIndex.empty.copy(numberChannelsData = Map(mkHash(1) -> 100L))
    val b = EventLogIndex.empty.copy(numberChannelsData = Map(mkHash(1) -> 200L))
    Ordering[EventLogIndex].compare(a, b) should not be 0
  }

  it should "compare equal for identical instances" in {
    val a = EventLogIndex.empty.copy(numberChannelsData = Map(mkHash(1) -> 100L))
    val b = EventLogIndex.empty.copy(numberChannelsData = Map(mkHash(1) -> 100L))
    Ordering[EventLogIndex].compare(a, b) shouldBe 0
  }

  it should "distinguish by entry count when keys differ in number" in {
    val a = EventLogIndex.empty.copy(numberChannelsData = Map(mkHash(1) -> 100L))
    val b = EventLogIndex.empty.copy(
      numberChannelsData = Map(mkHash(1) -> 100L, mkHash(2) -> 200L)
    )
    Ordering[EventLogIndex].compare(a, b) should not be 0
  }

  it should "be consistent across multiple comparisons" in {
    val a       = EventLogIndex.empty.copy(numberChannelsData = Map(mkHash(1) -> 100L))
    val b       = EventLogIndex.empty.copy(numberChannelsData = Map(mkHash(2) -> 50L))
    val result1 = Ordering[EventLogIndex].compare(a, b)
    val result2 = Ordering[EventLogIndex].compare(a, b)
    result1 shouldBe result2
    // Antisymmetry: compare(a,b) == -compare(b,a)
    Ordering[EventLogIndex].compare(b, a) shouldBe -result1
  }

  "computeRejectionOptions" should "produce deterministic results regardless of iteration" in {
    // Create a conflict map where multiple rejection options have equal cost
    val conflictMap = Map(
      1 -> Set(2, 3),
      2 -> Set(1, 3),
      3 -> Set(1, 2)
    )
    // Run multiple times to verify determinism
    val results = (1 to 10).map(_ => computeRejectionOptions(conflictMap))
    results.distinct.size shouldBe 1
  }

  it should "produce deterministic results with larger conflict maps" in {
    val conflictMap = Map(
      1 -> Set(2, 4),
      2 -> Set(1, 3),
      3 -> Set(2, 4),
      4 -> Set(1, 3)
    )
    val results = (1 to 10).map(_ => computeRejectionOptions(conflictMap))
    results.distinct.size shouldBe 1
  }
}
