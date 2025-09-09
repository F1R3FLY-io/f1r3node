package rspacePlusPlus

import com.sun.jna.Pointer
import coop.rchain.models.{Expr, Par}
import coop.rchain.models.Expr.ExprInstance

/**
  * Example demonstrating how to use the memory logging functionality
  * in JNAInterfaceLoader for investigating memory usage.
  */
object MemoryLoggingExample {

  def demonstrateMemoryLogging(): Unit = {
    println("=== Memory Logging Example ===")

    // Reset memory counter
    JNAInterfaceLoader.INSTANCE.reset_allocated_bytes()
    println(s"Initial allocated bytes: ${JNAInterfaceLoader.INSTANCE.get_allocated_bytes()}")

    // Example 1: Using the original methods (now with built-in logging)
    println("\n--- Using original methods with built-in logging ---")
    val channel = Par(exprs = Seq(Expr(ExprInstance.GInt(42))))
    val hash1   = JNAInterfaceLoader.hashChannel(channel)
    println(s"Hash result: ${hash1}")

    // Example 2: Using the explicit logging wrapper methods
    println("\n--- Using explicit logging wrapper methods ---")
    val channels = Seq(channel)
    val hash2    = JNAInterfaceLoader.hashChannels(channels)
    println(s"Hash result: ${hash2}")

    // Example 3: Memory leak detection pattern
    println("\n--- Memory leak detection pattern ---")
    detectMemoryLeaks()

    // Final memory state
    val finalBytes = JNAInterfaceLoader.INSTANCE.get_allocated_bytes()
    println(s"\nFinal allocated bytes: $finalBytes")
    if (finalBytes == 0) {
      println("✅ No memory leaks detected!")
    } else {
      println(s"⚠️  Potential memory leak: $finalBytes bytes not deallocated")
    }
  }

  private def detectMemoryLeaks(): Unit = {
    val iterations = 100
    val channel    = Par(exprs = Seq(Expr(ExprInstance.GInt(123))))

    println(s"Running $iterations iterations of hashChannel...")

    for (i <- 1 to iterations) {
      val beforeBytes = JNAInterfaceLoader.INSTANCE.get_allocated_bytes()
      val hash        = JNAInterfaceLoader.hashChannel(channel)
      val afterBytes  = JNAInterfaceLoader.INSTANCE.get_allocated_bytes()

      if (i % 20 == 0) {
        println(s"  Iteration $i: ${beforeBytes} -> ${afterBytes} bytes")
      }
    }

    val finalBytes = JNAInterfaceLoader.INSTANCE.get_allocated_bytes()
    println(s"After $iterations iterations: $finalBytes bytes allocated")
  }

  def main(args: Array[String]): Unit =
    demonstrateMemoryLogging()
}
