package coop.rchain.rholang

import org.scalatest.{FlatSpec, Matchers}
import coop.rchain.models._
import coop.rchain.models.rholang_scala_rust_types._
import coop.rchain.rholang.JNAInterfaceLoader.RHOLANG_RUST_INSTANCE
import com.sun.jna.Memory

class LeakSpec extends FlatSpec with Matchers {
  "Rholang native JNA calls" should "not leak memory on source_to_adt" in {
    RHOLANG_RUST_INSTANCE.rholang_reset_allocated_bytes()

    val sourceParams = SourceToAdtParams("0", Map.empty)
    val spBytes      = sourceParams.toByteArray
    val spPtr        = new Memory(spBytes.length.toLong)
    spPtr.write(0, spBytes, 0, spBytes.length)

    val resultPtr = RHOLANG_RUST_INSTANCE.source_to_adt(spPtr, spBytes.length)
    assert(resultPtr != null)
    val len = resultPtr.getInt(0)

    // ensure result parses
    Par.parseFrom(resultPtr.getByteArray(4, len))

    // deallocate returned buffer (accounts for 4-byte prefix)
    RHOLANG_RUST_INSTANCE.rholang_deallocate_memory(resultPtr, len + 4)

    RHOLANG_RUST_INSTANCE.rholang_get_allocated_bytes() shouldBe 0L
  }
}
