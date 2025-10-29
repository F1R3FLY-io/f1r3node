package coop.rchain.casper.util.rholang

import coop.rchain.rspace.trace.Event
import com.google.protobuf.ByteString
import scodec.bits.ByteVector

/** Cache key: parent state + block identity (sender, seqNum).
  * Using (sender, seqNum) avoids ambiguity across forks.
  */
final case class ReplayCacheKey(
    parentState: ByteString,
    senderPk: ByteVector,
    seqNum: Long
)

final case class ReplayCacheEntry(
    eventLog: Vector[Event],
    postState: ByteString
)

trait ReplayCache {
  def get(k: ReplayCacheKey): Option[ReplayCacheEntry]
  def put(k: ReplayCacheKey, v: ReplayCacheEntry): Unit
  def clear(): Unit
}

/** Simple in-memory LRU (thread-safe). */
final class InMemoryReplayCache(maxEntries: Int = 1024) extends ReplayCache {
  private val m =
    new java.util.LinkedHashMap[ReplayCacheKey, ReplayCacheEntry](maxEntries, 0.75f, true) {
      override def removeEldestEntry(e: java.util.Map.Entry[ReplayCacheKey, ReplayCacheEntry]) =
        this.size() > maxEntries
    }

  override def get(k: ReplayCacheKey): Option[ReplayCacheEntry] = this.synchronized {
    Option(m.get(k))
  }
  override def put(k: ReplayCacheKey, v: ReplayCacheEntry): Unit = this.synchronized {
    m.put(k, v); ()
  }
  override def clear(): Unit = this.synchronized(m.clear())
}
