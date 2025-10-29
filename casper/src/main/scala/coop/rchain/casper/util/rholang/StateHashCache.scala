package coop.rchain.casper.util.rholang

import com.google.protobuf.ByteString

final class StateHashCache(maxEntries: Int = 128) {
  private val map =
    new java.util.LinkedHashMap[ByteString, ByteString](maxEntries, 0.75f, true) {
      override def removeEldestEntry(
          e: java.util.Map.Entry[ByteString, ByteString]
      ) = this.size() > maxEntries
    }

  def get(pre: ByteString): Option[ByteString]     = synchronized { Option(map.get(pre)) }
  def put(pre: ByteString, post: ByteString): Unit = synchronized { map.put(pre, post) }
  def clear(): Unit                                = synchronized { map.clear() }
}
