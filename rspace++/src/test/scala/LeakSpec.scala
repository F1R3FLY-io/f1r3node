package rspacePlusPlus

import org.scalatest.wordspec.AnyWordSpec
import org.scalatest.matchers.should.Matchers
import coop.rchain.models.{Expr, Par}
import coop.rchain.models.Expr.ExprInstance
import coop.rchain.models.rspace_plus_plus_types.ChannelsProto
import rspacePlusPlus.JNAInterfaceLoader.INSTANCE

class LeakSpec extends AnyWordSpec with Matchers {

  "Native JNA calls" should {
    "not leak memory for hashChannel/hashChannels" in {
      INSTANCE.reset_allocated_bytes()

      // Non-empty Par to ensure non-zero serialization size
      val channel = Par(exprs = Seq(Expr(ExprInstance.GInt(1))))
      (1 to 1000).foreach { _ =>
        val _ = JNAInterfaceLoader.hashChannel(channel)
      }

      val bytesAfterHashChannel = INSTANCE.get_allocated_bytes()

      // hashChannels path (reuse single element in seq for simplicity)
      val channelsProto = ChannelsProto(channels = Seq(channel))
      (1 to 1000).foreach { _ =>
        val _ = JNAInterfaceLoader.hashChannels(Seq(channelsProto))
      }

      val bytesAfterHashChannels = INSTANCE.get_allocated_bytes()

      // Both should be zero if deallocation is correct
      bytesAfterHashChannel shouldBe 0L
      bytesAfterHashChannels shouldBe 0L
    }
  }
}
