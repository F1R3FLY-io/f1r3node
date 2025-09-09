package coop.rchain.rholang

import com.sun.jna.{Library, Memory, Native, Pointer}

/**
  * The JNA interface for Rholang Rust
  */
trait JNAInterface extends Library {

  /* RHO RUNTIME */

  def evaluate(runtime_ptr: Pointer, params_ptr: Pointer, params_bytes_len: Int): Pointer

  def inj(runtime_ptr: Pointer, params_ptr: Pointer, params_bytes_len: Int): Unit

  def create_soft_checkpoint(runtime_ptr: Pointer): Pointer

  def revert_to_soft_checkpoint(
      runtime_ptr: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Unit

  def create_checkpoint(runtime_ptr: Pointer): Pointer

  def consume_result(runtime_ptr: Pointer, params_ptr: Pointer, params_bytes_len: Int): Pointer

  def reset(
      runtime_ptr: Pointer,
      root_pointer: Pointer,
      root_bytes_len: Int
  ): Int

  def get_data(
      rspace: Pointer,
      channel_pointer: Pointer,
      channel_bytes_len: Int
  ): Pointer

  def get_joins(
      rspace: Pointer,
      channel_pointer: Pointer,
      channel_bytes_len: Int
  ): Pointer

  def get_waiting_continuations(
      rspace: Pointer,
      channels_pointer: Pointer,
      channels_bytes_len: Int
  ): Pointer

  def set_block_data(runtime_ptr: Pointer, params_ptr: Pointer, params_bytes_len: Int): Unit

  def set_invalid_blocks(runtime_ptr: Pointer, params_ptr: Pointer, params_bytes_len: Int): Unit

  def get_hot_changes(runtime_ptr: Pointer): Pointer

  def set_cost_to_max(runtime_ptr: Pointer): Unit

  /* REPLAY RHO RUNTIME */

  def rig(runtime_ptr: Pointer, log_pointer: Pointer, log_bytes_len: Int): Unit

  def check_replay_data(runtime_ptr: Pointer): Unit

  /* ADDITIONAL */

  def bootstrap_registry(runtime_ptr: Pointer): Unit

  def create_runtime(rspace_ptr: Pointer, params_ptr: Pointer, params_bytes_len: Int): Pointer

  def create_runtime_with_test_framework(
      rspace_ptr: Pointer,
      params_ptr: Pointer,
      params_bytes_len: Int
  ): Pointer

  def create_replay_runtime(
      replay_space_ptr: Pointer,
      params_ptr: Pointer,
      params_bytes_len: Int
  ): Pointer

  def source_to_adt(params_ptr: Pointer, params_bytes_len: Int): Pointer

  // Leak tracking and deallocator
  def rholang_deallocate_memory(ptr: Pointer, len: Int): Unit
  def rholang_get_allocated_bytes(): Long
  def rholang_reset_allocated_bytes(): Unit
}

object JNAInterfaceLoader {
  // Load raw native instance
  private val RAW_RHOLANG_RUST_INSTANCE: JNAInterface =
    try {
      Native.load("rholang", classOf[JNAInterface])
    } catch {
      case e: UnsatisfiedLinkError =>
        throw new RuntimeException(
          s"Failed to load library 'rholang' from path '${System.getProperty("jna.library.path")}'",
          e
        )
    }

  // Logging proxy that wraps all calls and prints memory before/after
  private class LoggingJNAInterface(delegate: JNAInterface) extends JNAInterface {
    private def beforeBytes(): Long = delegate.rholang_get_allocated_bytes()
    private def afterAndLog(operation: String, before: Long): Unit =
      logMemoryUsage(operation, before)

    override def evaluate(
        runtime_ptr: Pointer,
        params_ptr: Pointer,
        params_bytes_len: Int
    ): Pointer = {
      val before = beforeBytes()
      val result = delegate.evaluate(runtime_ptr, params_ptr, params_bytes_len)
      afterAndLog("evaluate", before)
      result
    }

    override def inj(runtime_ptr: Pointer, params_ptr: Pointer, params_bytes_len: Int): Unit = {
      val before = beforeBytes()
      delegate.inj(runtime_ptr, params_ptr, params_bytes_len)
      afterAndLog("inj", before)
    }

    override def create_soft_checkpoint(runtime_ptr: Pointer): Pointer = {
      val before = beforeBytes()
      val result = delegate.create_soft_checkpoint(runtime_ptr)
      afterAndLog("create_soft_checkpoint", before)
      result
    }

    override def revert_to_soft_checkpoint(
        runtime_ptr: Pointer,
        payload_pointer: Pointer,
        payload_bytes_len: Int
    ): Unit = {
      val before = beforeBytes()
      delegate.revert_to_soft_checkpoint(runtime_ptr, payload_pointer, payload_bytes_len)
      afterAndLog("revert_to_soft_checkpoint", before)
    }

    override def create_checkpoint(runtime_ptr: Pointer): Pointer = {
      val before = beforeBytes()
      val result = delegate.create_checkpoint(runtime_ptr)
      afterAndLog("create_checkpoint", before)
      result
    }

    override def consume_result(
        runtime_ptr: Pointer,
        params_ptr: Pointer,
        params_bytes_len: Int
    ): Pointer = {
      val before = beforeBytes()
      val result = delegate.consume_result(runtime_ptr, params_ptr, params_bytes_len)
      afterAndLog("consume_result", before)
      result
    }

    override def reset(
        runtime_ptr: Pointer,
        root_pointer: Pointer,
        root_bytes_len: Int
    ): Int = {
      val before = beforeBytes()
      val result = delegate.reset(runtime_ptr, root_pointer, root_bytes_len)
      afterAndLog("reset", before)
      result
    }

    override def get_data(
        rspace: Pointer,
        channel_pointer: Pointer,
        channel_bytes_len: Int
    ): Pointer = {
      val before = beforeBytes()
      val result = delegate.get_data(rspace, channel_pointer, channel_bytes_len)
      afterAndLog("get_data", before)
      result
    }

    override def get_joins(
        rspace: Pointer,
        channel_pointer: Pointer,
        channel_bytes_len: Int
    ): Pointer = {
      val before = beforeBytes()
      val result = delegate.get_joins(rspace, channel_pointer, channel_bytes_len)
      afterAndLog("get_joins", before)
      result
    }

    override def get_waiting_continuations(
        rspace: Pointer,
        channels_pointer: Pointer,
        channels_bytes_len: Int
    ): Pointer = {
      val before = beforeBytes()
      val result = delegate.get_waiting_continuations(rspace, channels_pointer, channels_bytes_len)
      afterAndLog("get_waiting_continuations", before)
      result
    }

    override def set_block_data(
        runtime_ptr: Pointer,
        params_ptr: Pointer,
        params_bytes_len: Int
    ): Unit = {
      val before = beforeBytes()
      delegate.set_block_data(runtime_ptr, params_ptr, params_bytes_len)
      afterAndLog("set_block_data", before)
    }

    override def set_invalid_blocks(
        runtime_ptr: Pointer,
        params_ptr: Pointer,
        params_bytes_len: Int
    ): Unit = {
      val before = beforeBytes()
      delegate.set_invalid_blocks(runtime_ptr, params_ptr, params_bytes_len)
      afterAndLog("set_invalid_blocks", before)
    }

    override def get_hot_changes(runtime_ptr: Pointer): Pointer = {
      val before = beforeBytes()
      val result = delegate.get_hot_changes(runtime_ptr)
      afterAndLog("get_hot_changes", before)
      result
    }

    override def set_cost_to_max(runtime_ptr: Pointer): Unit = {
      val before = beforeBytes()
      delegate.set_cost_to_max(runtime_ptr)
      afterAndLog("set_cost_to_max", before)
    }

    override def rig(runtime_ptr: Pointer, log_pointer: Pointer, log_bytes_len: Int): Unit = {
      val before = beforeBytes()
      delegate.rig(runtime_ptr, log_pointer, log_bytes_len)
      afterAndLog("rig", before)
    }

    override def check_replay_data(runtime_ptr: Pointer): Unit = {
      val before = beforeBytes()
      delegate.check_replay_data(runtime_ptr)
      afterAndLog("check_replay_data", before)
    }

    override def bootstrap_registry(runtime_ptr: Pointer): Unit = {
      val before = beforeBytes()
      delegate.bootstrap_registry(runtime_ptr)
      afterAndLog("bootstrap_registry", before)
    }

    override def create_runtime(
        rspace_ptr: Pointer,
        params_ptr: Pointer,
        params_bytes_len: Int
    ): Pointer = {
      val before = beforeBytes()
      val result = delegate.create_runtime(rspace_ptr, params_ptr, params_bytes_len)
      afterAndLog("create_runtime", before)
      result
    }

    override def create_runtime_with_test_framework(
        rspace_ptr: Pointer,
        params_ptr: Pointer,
        params_bytes_len: Int
    ): Pointer = {
      val before = beforeBytes()
      val result =
        delegate.create_runtime_with_test_framework(rspace_ptr, params_ptr, params_bytes_len)
      afterAndLog("create_runtime_with_test_framework", before)
      result
    }

    override def create_replay_runtime(
        replay_space_ptr: Pointer,
        params_ptr: Pointer,
        params_bytes_len: Int
    ): Pointer = {
      val before = beforeBytes()
      val result = delegate.create_replay_runtime(replay_space_ptr, params_ptr, params_bytes_len)
      afterAndLog("create_replay_runtime", before)
      result
    }

    override def source_to_adt(params_ptr: Pointer, params_bytes_len: Int): Pointer = {
      val before = beforeBytes()
      val result = delegate.source_to_adt(params_ptr, params_bytes_len)
      afterAndLog("source_to_adt", before)
      result
    }

    override def rholang_deallocate_memory(ptr: Pointer, len: Int): Unit = {
      val before = beforeBytes()
      delegate.rholang_deallocate_memory(ptr, len)
      afterAndLog("rholang_deallocate_memory", before)
    }

    // This one should NOT log to avoid recursion when measuring memory
    override def rholang_get_allocated_bytes(): Long = delegate.rholang_get_allocated_bytes()

    override def rholang_reset_allocated_bytes(): Unit = {
      val before = beforeBytes()
      delegate.rholang_reset_allocated_bytes()
      afterAndLog("rholang_reset_allocated_bytes", before)
    }
  }

  // Exposed instance that logs on every call
  val RHOLANG_RUST_INSTANCE: JNAInterface = new LoggingJNAInterface(RAW_RHOLANG_RUST_INSTANCE)

  // Memory logging helper
  private def logMemoryUsage(operation: String, beforeBytes: Long): Unit = {
    val afterBytes = RHOLANG_RUST_INSTANCE.rholang_get_allocated_bytes()
    val delta      = afterBytes - beforeBytes
    val deltaStr   = if (delta >= 0) s"+$delta" else s"$delta"
    println(s"[RHOLANG_MEMORY] $operation: ${beforeBytes} -> ${afterBytes} bytes (Δ$deltaStr)")
  }

  // Wrapper methods maintained for compatibility; they now delegate to the logging instance
  def evaluate_with_logging(
      runtime_ptr: Pointer,
      params_ptr: Pointer,
      params_bytes_len: Int
  ): Pointer =
    RHOLANG_RUST_INSTANCE.evaluate(runtime_ptr, params_ptr, params_bytes_len)

  def inj_with_logging(runtime_ptr: Pointer, params_ptr: Pointer, params_bytes_len: Int): Unit =
    RHOLANG_RUST_INSTANCE.inj(runtime_ptr, params_ptr, params_bytes_len)

  def create_soft_checkpoint_with_logging(runtime_ptr: Pointer): Pointer =
    RHOLANG_RUST_INSTANCE.create_soft_checkpoint(runtime_ptr)

  def revert_to_soft_checkpoint_with_logging(
      runtime_ptr: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Unit =
    RHOLANG_RUST_INSTANCE.revert_to_soft_checkpoint(runtime_ptr, payload_pointer, payload_bytes_len)

  def create_checkpoint_with_logging(runtime_ptr: Pointer): Pointer =
    RHOLANG_RUST_INSTANCE.create_checkpoint(runtime_ptr)

  def consume_result_with_logging(
      runtime_ptr: Pointer,
      params_ptr: Pointer,
      params_bytes_len: Int
  ): Pointer =
    RHOLANG_RUST_INSTANCE.consume_result(runtime_ptr, params_ptr, params_bytes_len)

  def reset_with_logging(
      runtime_ptr: Pointer,
      root_pointer: Pointer,
      root_bytes_len: Int
  ): Int =
    RHOLANG_RUST_INSTANCE.reset(runtime_ptr, root_pointer, root_bytes_len)

  def get_data_with_logging(
      rspace: Pointer,
      channel_pointer: Pointer,
      channel_bytes_len: Int
  ): Pointer =
    RHOLANG_RUST_INSTANCE.get_data(rspace, channel_pointer, channel_bytes_len)

  def get_joins_with_logging(
      rspace: Pointer,
      channel_pointer: Pointer,
      channel_bytes_len: Int
  ): Pointer =
    RHOLANG_RUST_INSTANCE.get_joins(rspace, channel_pointer, channel_bytes_len)

  def get_waiting_continuations_with_logging(
      rspace: Pointer,
      channels_pointer: Pointer,
      channels_bytes_len: Int
  ): Pointer =
    RHOLANG_RUST_INSTANCE.get_waiting_continuations(rspace, channels_pointer, channels_bytes_len)

  def set_block_data_with_logging(
      runtime_ptr: Pointer,
      params_ptr: Pointer,
      params_bytes_len: Int
  ): Unit =
    RHOLANG_RUST_INSTANCE.set_block_data(runtime_ptr, params_ptr, params_bytes_len)

  def set_invalid_blocks_with_logging(
      runtime_ptr: Pointer,
      params_ptr: Pointer,
      params_bytes_len: Int
  ): Unit =
    RHOLANG_RUST_INSTANCE.set_invalid_blocks(runtime_ptr, params_ptr, params_bytes_len)

  def get_hot_changes_with_logging(runtime_ptr: Pointer): Pointer =
    RHOLANG_RUST_INSTANCE.get_hot_changes(runtime_ptr)

  def set_cost_to_max_with_logging(runtime_ptr: Pointer): Unit =
    RHOLANG_RUST_INSTANCE.set_cost_to_max(runtime_ptr)

  def rig_with_logging(runtime_ptr: Pointer, log_pointer: Pointer, log_bytes_len: Int): Unit =
    RHOLANG_RUST_INSTANCE.rig(runtime_ptr, log_pointer, log_bytes_len)

  def check_replay_data_with_logging(runtime_ptr: Pointer): Unit =
    RHOLANG_RUST_INSTANCE.check_replay_data(runtime_ptr)

  def bootstrap_registry_with_logging(runtime_ptr: Pointer): Unit =
    RHOLANG_RUST_INSTANCE.bootstrap_registry(runtime_ptr)

  def create_runtime_with_logging(
      rspace_ptr: Pointer,
      params_ptr: Pointer,
      params_bytes_len: Int
  ): Pointer =
    RHOLANG_RUST_INSTANCE.create_runtime(rspace_ptr, params_ptr, params_bytes_len)

  def create_runtime_with_test_framework_with_logging(
      rspace_ptr: Pointer,
      params_ptr: Pointer,
      params_bytes_len: Int
  ): Pointer =
    RHOLANG_RUST_INSTANCE.create_runtime_with_test_framework(
      rspace_ptr,
      params_ptr,
      params_bytes_len
    )

  def create_replay_runtime_with_logging(
      replay_space_ptr: Pointer,
      params_ptr: Pointer,
      params_bytes_len: Int
  ): Pointer =
    RHOLANG_RUST_INSTANCE.create_replay_runtime(replay_space_ptr, params_ptr, params_bytes_len)

  def source_to_adt_with_logging(params_ptr: Pointer, params_bytes_len: Int): Pointer =
    RHOLANG_RUST_INSTANCE.source_to_adt(params_ptr, params_bytes_len)

  def rholang_deallocate_memory_with_logging(ptr: Pointer, len: Int): Unit =
    RHOLANG_RUST_INSTANCE.rholang_deallocate_memory(ptr, len)
}
