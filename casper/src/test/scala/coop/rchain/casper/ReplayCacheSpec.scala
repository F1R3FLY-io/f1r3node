package coop.rchain.casper.util.rholang

import org.scalatest.{FlatSpec, Matchers}
import com.google.protobuf.ByteString
import coop.rchain.rspace.trace.Event
import scodec.bits.ByteVector

class ReplayCacheSpec extends FlatSpec with Matchers {

  behavior of "InMemoryReplayCache"

  it should "store and retrieve cached entries by key" in {
    val cache = new InMemoryReplayCache()
    val key   = ReplayCacheKey(ByteString.copyFromUtf8("parent"), ByteVector("sender".getBytes), 1L)
    val entry = ReplayCacheEntry(Vector.empty[Event], ByteString.copyFromUtf8("post-state"))

    cache.put(key, entry)
    cache.get(key) shouldBe Some(entry)
  }

  it should "miss for unknown key" in {
    val cache = new InMemoryReplayCache()
    val key   = ReplayCacheKey(ByteString.copyFromUtf8("unknown"), ByteVector("s".getBytes), 42L)
    cache.get(key) shouldBe None
  }

  it should "evict oldest entries when max size exceeded" in {
    val cache = new InMemoryReplayCache(maxEntries = 2)

    val k1 = ReplayCacheKey(ByteString.copyFromUtf8("p1"), ByteVector("a".getBytes), 1L)
    val k2 = ReplayCacheKey(ByteString.copyFromUtf8("p2"), ByteVector("b".getBytes), 2L)
    val k3 = ReplayCacheKey(ByteString.copyFromUtf8("p3"), ByteVector("c".getBytes), 3L)
    val e  = ReplayCacheEntry(Vector.empty[Event], ByteString.EMPTY)

    cache.put(k1, e)
    cache.put(k2, e)
    cache.put(k3, e)

    cache.get(k1) shouldBe None // evicted
    cache.get(k2) shouldBe Some(e)
    cache.get(k3) shouldBe Some(e)
  }

  it should "clear all entries" in {
    val cache = new InMemoryReplayCache()
    val key   = ReplayCacheKey(ByteString.copyFromUtf8("p"), ByteVector("s".getBytes), 5L)
    val entry = ReplayCacheEntry(Vector.empty[Event], ByteString.EMPTY)

    cache.put(key, entry)
    cache.clear()
    cache.get(key) shouldBe None
  }
}
