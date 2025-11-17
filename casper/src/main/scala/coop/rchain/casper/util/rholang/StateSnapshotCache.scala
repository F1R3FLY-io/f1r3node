package coop.rchain.casper.util.rholang

final case class StateSnapshotEntry(bytes: Array[Byte])

trait StateSnapshotCache {
  def get(hash: com.google.protobuf.ByteString): Option[StateSnapshotEntry]
  def put(hash: com.google.protobuf.ByteString, entry: StateSnapshotEntry): Unit
  def clear(): Unit
}

final class InMemoryStateSnapshotCache(maxEntries: Int = 8) extends StateSnapshotCache {
  private val map =
    new java.util.LinkedHashMap[com.google.protobuf.ByteString, StateSnapshotEntry](
      maxEntries,
      0.75f,
      true
    ) {
      override def removeEldestEntry(
          e: java.util.Map.Entry[com.google.protobuf.ByteString, StateSnapshotEntry]
      ) =
        this.size() > maxEntries
    }

  override def get(k: com.google.protobuf.ByteString): Option[StateSnapshotEntry] =
    this.synchronized {
      Option(map.get(k))
    }

  override def put(k: com.google.protobuf.ByteString, v: StateSnapshotEntry): Unit =
    this.synchronized {
      map.put(k, v)
    }

  override def clear(): Unit = this.synchronized(map.clear())
}
