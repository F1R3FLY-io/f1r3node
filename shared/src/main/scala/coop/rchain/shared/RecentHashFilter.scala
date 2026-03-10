package coop.rchain.shared

import scala.collection.mutable

/** Simple synchronized LRU filter to suppress redundant gossip messages by hash. */
final class RecentHashFilter(maxEntries: Int = 8192) {
  private val set = mutable.LinkedHashSet.empty[String]

  /** Returns true if this hash has been seen recently. */
  def seenBefore(hash: String): Boolean = this.synchronized {
    val exists = set.contains(hash)
    if (!exists) {
      set.add(hash)
      // trim oldest entries if over capacity
      while (set.size > maxEntries) {
        val oldest = set.headOption
        oldest.foreach(set.remove)
      }
    }
    exists
  }

  def size: Int = this.synchronized(set.size)
}
