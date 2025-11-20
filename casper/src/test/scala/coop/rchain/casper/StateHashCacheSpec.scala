package coop.rchain.casper

import coop.rchain.casper.util.rholang.StateHashCache
import com.google.protobuf.ByteString
import org.scalatest.{FunSpec, Matchers}

class StateHashCacheSpec extends FunSpec with Matchers {
  describe("StateHashCache") {
    it("should cache and retrieve post-state hashes") {
      val cache = new StateHashCache()
      val pre   = ByteString.copyFromUtf8("A")
      val post  = ByteString.copyFromUtf8("B")
      cache.put(pre, post)
      cache.get(pre) shouldBe Some(post)
    }
  }
}
