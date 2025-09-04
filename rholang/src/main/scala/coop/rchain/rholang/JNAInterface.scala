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
  val RHOLANG_RUST_INSTANCE: JNAInterface =
    try {
      Native.load("rholang", classOf[JNAInterface])
    } catch {
      case e: UnsatisfiedLinkError =>
        throw new RuntimeException(
          s"Failed to load library 'rholang' from path '${System.getProperty("jna.library.path")}'",
          e
        )
    }
}
