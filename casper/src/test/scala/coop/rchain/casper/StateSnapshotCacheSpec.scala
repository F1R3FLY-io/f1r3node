package coop.rchain.casper

import coop.rchain.casper.util.rholang.{InMemoryStateSnapshotCache, StateSnapshotEntry}
import com.google.protobuf.ByteString
import org.scalatest.{FunSpec, Matchers}

class StateSnapshotCacheSpec extends FunSpec with Matchers {
  describe("InMemoryStateSnapshotCache") {
    it("should store and restore snapshots") {
      val cache = new InMemoryStateSnapshotCache()
      val hash  = ByteString.copyFrom(Array[Byte](1, 2, 3))
      val data  = Array[Byte](4, 5, 6)

      cache.put(hash, StateSnapshotEntry(data))
      cache.get(hash).map(_.bytes.toSeq) shouldBe Some(data.toSeq)
    }
  }
}
