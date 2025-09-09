package coop.rchain.rholang

import com.sun.jna.{Memory, Pointer}

/**
  * Example demonstrating how to use the memory logging functionality
  * in JNAInterfaceLoader for investigating Rholang memory usage.
  */
object MemoryLoggingExample {

  def demonstrateRholangMemoryLogging(): Unit = {
    println("=== Rholang Memory Logging Example ===")

    // Reset memory counter
    JNAInterfaceLoader.RHOLANG_RUST_INSTANCE.rholang_reset_allocated_bytes()
    println(
      s"Initial allocated bytes: ${JNAInterfaceLoader.RHOLANG_RUST_INSTANCE.rholang_get_allocated_bytes()}"
    )

    // Example 1: Using the explicit logging wrapper methods
    println("\n--- Using explicit logging wrapper methods ---")

    // Note: These are examples - you would need actual runtime and rspace pointers
    // val runtimePtr = createRuntime() // You would create this
    // val rspacePtr = createRSpace()   // You would create this

    // Example of how to use the logging methods:
    // val result = JNAInterfaceLoader.evaluate_with_logging(runtimePtr, paramsPtr, paramsLen)
    // val checkpoint = JNAInterfaceLoader.create_checkpoint_with_logging(runtimePtr)
    // val hotChanges = JNAInterfaceLoader.get_hot_changes_with_logging(runtimePtr)

    println("Example usage patterns:")
    println("  JNAInterfaceLoader.evaluate_with_logging(runtimePtr, paramsPtr, paramsLen)")
    println("  JNAInterfaceLoader.create_checkpoint_with_logging(runtimePtr)")
    println("  JNAInterfaceLoader.get_hot_changes_with_logging(runtimePtr)")
    println("  JNAInterfaceLoader.rholang_deallocate_memory_with_logging(ptr, len)")

    // Example 2: Memory leak detection pattern
    println("\n--- Memory leak detection pattern ---")
    detectRholangMemoryLeaks()

    // Final memory state
    val finalBytes = JNAInterfaceLoader.RHOLANG_RUST_INSTANCE.rholang_get_allocated_bytes()
    println(s"\nFinal allocated bytes: $finalBytes")
    if (finalBytes == 0) {
      println("✅ No memory leaks detected!")
    } else {
      println(s"⚠️  Potential memory leak: $finalBytes bytes not deallocated")
    }
  }

  private def detectRholangMemoryLeaks(): Unit = {
    val iterations = 50

    println(s"Running $iterations iterations of memory tracking...")

    for (i <- 1 to iterations) {
      val beforeBytes = JNAInterfaceLoader.RHOLANG_RUST_INSTANCE.rholang_get_allocated_bytes()

      // Simulate some operations that might allocate memory
      // In real usage, you would call actual Rholang methods here

      val afterBytes = JNAInterfaceLoader.RHOLANG_RUST_INSTANCE.rholang_get_allocated_bytes()

      if (i % 10 == 0) {
        println(s"  Iteration $i: ${beforeBytes} -> ${afterBytes} bytes")
      }
    }

    val finalBytes = JNAInterfaceLoader.RHOLANG_RUST_INSTANCE.rholang_get_allocated_bytes()
    println(s"After $iterations iterations: $finalBytes bytes allocated")
  }

  def demonstrateMemoryTracking(): Unit = {
    println("\n=== Memory Tracking Demonstration ===")

    // Reset and track memory
    JNAInterfaceLoader.RHOLANG_RUST_INSTANCE.rholang_reset_allocated_bytes()

    // Track memory usage for different operations
    val operations = List(
      "evaluate",
      "create_checkpoint",
      "get_hot_changes",
      "bootstrap_registry",
      "source_to_adt"
    )

    operations.foreach { operation =>
      val beforeBytes = JNAInterfaceLoader.RHOLANG_RUST_INSTANCE.rholang_get_allocated_bytes()
      println(s"Before $operation: $beforeBytes bytes")

      // In real usage, you would call the actual method here
      // For example: JNAInterfaceLoader.evaluate_with_logging(...)

      val afterBytes = JNAInterfaceLoader.RHOLANG_RUST_INSTANCE.rholang_get_allocated_bytes()
      val delta      = afterBytes - beforeBytes
      println(s"After $operation: $afterBytes bytes (Δ$delta)")
    }
  }

  def main(args: Array[String]): Unit = {
    demonstrateRholangMemoryLogging()
    demonstrateMemoryTracking()
  }
}
