package rspacePlusPlus

import com.sun.jna.{Library, Memory, Native, Pointer}
import coop.rchain.models.{BindPattern, ListParWithRandom, Par, TaggedContinuation}
import coop.rchain.rspace.hashing.Blake2b256Hash
import coop.rchain.models.rspace_plus_plus_types.HashProto

/**
  * The JNA interface for Rust RSpace++
  */
trait JNAInterface extends Library {
  def space_new(storePath: String): Pointer
  def space_new_replay(rspace: Pointer): Pointer
  def space_print(rspace: Pointer): Unit
  def space_clear(rspace: Pointer): Unit

  def spatial_match_result(
      payload_pointer: Pointer,
      target_bytes_len: Int,
      pattern_bytes_len: Int
  ): Pointer

  def produce(
      rspace: Pointer,
      payload_pointer: Pointer,
      channel_bytes_len: Int,
      data_bytes_len: Int,
      persist: Boolean
  ): Pointer

  def consume(
      rspace: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Pointer

  def install(
      rspace: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Pointer

  def create_checkpoint(rspace: Pointer): Pointer

  def reset_rspace(
      rspace: Pointer,
      root_pointer: Pointer,
      root_bytes_len: Int
  ): Int

  def get_data(
      rspace: Pointer,
      channel_pointer: Pointer,
      channel_bytes_len: Int
  ): Pointer

  def get_waiting_continuations(
      rspace: Pointer,
      channels_pointer: Pointer,
      channels_bytes_len: Int
  ): Pointer

  def get_joins(
      rspace: Pointer,
      channel_pointer: Pointer,
      channel_bytes_len: Int
  ): Pointer

  def to_map(rspace: Pointer): Pointer

  def spawn(rspace: Pointer): Pointer

  def create_soft_checkpoint(rspace: Pointer): Pointer

  def revert_to_soft_checkpoint(
      rspace: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Unit

  /* HistoryRepo */

  def history_repo_root(rspace: Pointer): Pointer

  /* Exporter */

  def get_nodes(
      rspace: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Pointer

  def get_history_and_data(
      rspace: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Pointer

  /* Importer */

  def validate_state_items(
      rspace: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Unit

  def set_history_items(
      rspace: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Unit

  def set_data_items(
      rspace: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Unit

  def set_root(
      rspace: Pointer,
      root_pointer: Pointer,
      root_bytes_len: Int
  ): Unit

  def get_history_item(
      rspace: Pointer,
      hash_pointer: Pointer,
      hash_bytes_len: Int
  ): Pointer

  /* HistoryReader */

  def history_reader_root(
      rspace: Pointer,
      state_hash_pointer: Pointer,
      state_hash_bytes_len: Int
  ): Pointer

  def get_history_data(
      rspace: Pointer,
      payload_pointer: Pointer,
      state_hash_bytes_len: Int,
      key_bytes_len: Int
  ): Pointer

  def get_history_waiting_continuations(
      rspace: Pointer,
      payload_pointer: Pointer,
      state_hash_bytes_len: Int,
      key_bytes_len: Int
  ): Pointer

  def get_history_joins(
      rspace: Pointer,
      payload_pointer: Pointer,
      state_hash_bytes_len: Int,
      key_bytes_len: Int
  ): Pointer

  /* ReplayRSpace */

  def replay_produce(
      rspace: Pointer,
      payload_pointer: Pointer,
      channel_bytes_len: Int,
      data_bytes_len: Int,
      persist: Boolean
  ): Pointer

  def replay_consume(
      rspace: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Pointer

  def replay_create_checkpoint(rspace: Pointer): Pointer

  def replay_clear(rspace: Pointer): Unit

  def replay_spawn(replay_rspace_ptr: Pointer): Pointer

  /* IReplayRSpace */

  def rig(rspace: Pointer, log_pointer: Pointer, log_bytes_len: Int): Unit

  def check_replay_data(rspace: Pointer): Unit

  /* Helper Functions */

  def hash_channel(channel_pointer: Pointer, channel_bytes_len: Int): Pointer

  def hash_channels(channels_pointer: Pointer, channels_bytes_len: Int): Pointer

  def deallocate_memory(ptr: Pointer, len: Int): Unit

  // Leak detection helpers in the native lib
  def get_allocated_bytes(): Long
  def reset_allocated_bytes(): Unit
}

trait ByteArrayConvertible {
  def toByteArray: Array[Byte]
}

object JNAInterfaceLoader {
  val INSTANCE: JNAInterface =
    try {
      Native.load("rspace_plus_plus_rhotypes", classOf[JNAInterface])
    } catch {
      case e: UnsatisfiedLinkError =>
        throw new RuntimeException(
          s"Failed to load library 'rspace_plus_plus_rhotypes' from path '${System.getProperty("jna.library.path")}'",
          e
        )
    }

  // Memory logging helper
  private def logMemoryUsage(operation: String, beforeBytes: Long): Unit = {
    val afterBytes = INSTANCE.get_allocated_bytes()
    val delta      = afterBytes - beforeBytes
    val deltaStr   = if (delta >= 0) s"+$delta" else s"$delta"
    println(s"[MEMORY] $operation: ${beforeBytes} -> ${afterBytes} bytes (Δ$deltaStr)")
  }

  // Wrapper methods with memory logging
  def space_new_with_logging(storePath: String): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.space_new(storePath)
    logMemoryUsage("space_new", beforeBytes)
    result
  }

  def space_new_replay_with_logging(rspace: Pointer): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.space_new_replay(rspace)
    logMemoryUsage("space_new_replay", beforeBytes)
    result
  }

  def space_print_with_logging(rspace: Pointer): Unit = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    INSTANCE.space_print(rspace)
    logMemoryUsage("space_print", beforeBytes)
  }

  def space_clear_with_logging(rspace: Pointer): Unit = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    INSTANCE.space_clear(rspace)
    logMemoryUsage("space_clear", beforeBytes)
  }

  def spatial_match_result_with_logging(
      payload_pointer: Pointer,
      target_bytes_len: Int,
      pattern_bytes_len: Int
  ): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.spatial_match_result(payload_pointer, target_bytes_len, pattern_bytes_len)
    logMemoryUsage("spatial_match_result", beforeBytes)
    result
  }

  def produce_with_logging(
      rspace: Pointer,
      payload_pointer: Pointer,
      channel_bytes_len: Int,
      data_bytes_len: Int,
      persist: Boolean
  ): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result =
      INSTANCE.produce(rspace, payload_pointer, channel_bytes_len, data_bytes_len, persist)
    logMemoryUsage("produce", beforeBytes)
    result
  }

  def consume_with_logging(
      rspace: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.consume(rspace, payload_pointer, payload_bytes_len)
    logMemoryUsage("consume", beforeBytes)
    result
  }

  def install_with_logging(
      rspace: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.install(rspace, payload_pointer, payload_bytes_len)
    logMemoryUsage("install", beforeBytes)
    result
  }

  def create_checkpoint_with_logging(rspace: Pointer): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.create_checkpoint(rspace)
    logMemoryUsage("create_checkpoint", beforeBytes)
    result
  }

  def reset_rspace_with_logging(
      rspace: Pointer,
      root_pointer: Pointer,
      root_bytes_len: Int
  ): Int = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.reset_rspace(rspace, root_pointer, root_bytes_len)
    logMemoryUsage("reset_rspace", beforeBytes)
    result
  }

  def get_data_with_logging(
      rspace: Pointer,
      channel_pointer: Pointer,
      channel_bytes_len: Int
  ): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.get_data(rspace, channel_pointer, channel_bytes_len)
    logMemoryUsage("get_data", beforeBytes)
    result
  }

  def get_waiting_continuations_with_logging(
      rspace: Pointer,
      channels_pointer: Pointer,
      channels_bytes_len: Int
  ): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.get_waiting_continuations(rspace, channels_pointer, channels_bytes_len)
    logMemoryUsage("get_waiting_continuations", beforeBytes)
    result
  }

  def get_joins_with_logging(
      rspace: Pointer,
      channel_pointer: Pointer,
      channel_bytes_len: Int
  ): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.get_joins(rspace, channel_pointer, channel_bytes_len)
    logMemoryUsage("get_joins", beforeBytes)
    result
  }

  def to_map_with_logging(rspace: Pointer): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.to_map(rspace)
    logMemoryUsage("to_map", beforeBytes)
    result
  }

  def spawn_with_logging(rspace: Pointer): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.spawn(rspace)
    logMemoryUsage("spawn", beforeBytes)
    result
  }

  def create_soft_checkpoint_with_logging(rspace: Pointer): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.create_soft_checkpoint(rspace)
    logMemoryUsage("create_soft_checkpoint", beforeBytes)
    result
  }

  def revert_to_soft_checkpoint_with_logging(
      rspace: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Unit = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    INSTANCE.revert_to_soft_checkpoint(rspace, payload_pointer, payload_bytes_len)
    logMemoryUsage("revert_to_soft_checkpoint", beforeBytes)
  }

  def history_repo_root_with_logging(rspace: Pointer): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.history_repo_root(rspace)
    logMemoryUsage("history_repo_root", beforeBytes)
    result
  }

  def get_nodes_with_logging(
      rspace: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.get_nodes(rspace, payload_pointer, payload_bytes_len)
    logMemoryUsage("get_nodes", beforeBytes)
    result
  }

  def get_history_and_data_with_logging(
      rspace: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.get_history_and_data(rspace, payload_pointer, payload_bytes_len)
    logMemoryUsage("get_history_and_data", beforeBytes)
    result
  }

  def validate_state_items_with_logging(
      rspace: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Unit = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    INSTANCE.validate_state_items(rspace, payload_pointer, payload_bytes_len)
    logMemoryUsage("validate_state_items", beforeBytes)
  }

  def set_history_items_with_logging(
      rspace: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Unit = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    INSTANCE.set_history_items(rspace, payload_pointer, payload_bytes_len)
    logMemoryUsage("set_history_items", beforeBytes)
  }

  def set_data_items_with_logging(
      rspace: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Unit = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    INSTANCE.set_data_items(rspace, payload_pointer, payload_bytes_len)
    logMemoryUsage("set_data_items", beforeBytes)
  }

  def set_root_with_logging(
      rspace: Pointer,
      root_pointer: Pointer,
      root_bytes_len: Int
  ): Unit = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    INSTANCE.set_root(rspace, root_pointer, root_bytes_len)
    logMemoryUsage("set_root", beforeBytes)
  }

  def get_history_item_with_logging(
      rspace: Pointer,
      hash_pointer: Pointer,
      hash_bytes_len: Int
  ): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.get_history_item(rspace, hash_pointer, hash_bytes_len)
    logMemoryUsage("get_history_item", beforeBytes)
    result
  }

  def history_reader_root_with_logging(
      rspace: Pointer,
      state_hash_pointer: Pointer,
      state_hash_bytes_len: Int
  ): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.history_reader_root(rspace, state_hash_pointer, state_hash_bytes_len)
    logMemoryUsage("history_reader_root", beforeBytes)
    result
  }

  def get_history_data_with_logging(
      rspace: Pointer,
      payload_pointer: Pointer,
      state_hash_bytes_len: Int,
      key_bytes_len: Int
  ): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result =
      INSTANCE.get_history_data(rspace, payload_pointer, state_hash_bytes_len, key_bytes_len)
    logMemoryUsage("get_history_data", beforeBytes)
    result
  }

  def get_history_waiting_continuations_with_logging(
      rspace: Pointer,
      payload_pointer: Pointer,
      state_hash_bytes_len: Int,
      key_bytes_len: Int
  ): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result = INSTANCE.get_history_waiting_continuations(
      rspace,
      payload_pointer,
      state_hash_bytes_len,
      key_bytes_len
    )
    logMemoryUsage("get_history_waiting_continuations", beforeBytes)
    result
  }

  def get_history_joins_with_logging(
      rspace: Pointer,
      payload_pointer: Pointer,
      state_hash_bytes_len: Int,
      key_bytes_len: Int
  ): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result =
      INSTANCE.get_history_joins(rspace, payload_pointer, state_hash_bytes_len, key_bytes_len)
    logMemoryUsage("get_history_joins", beforeBytes)
    result
  }

  def replay_produce_with_logging(
      rspace: Pointer,
      payload_pointer: Pointer,
      channel_bytes_len: Int,
      data_bytes_len: Int,
      persist: Boolean
  ): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result =
      INSTANCE.replay_produce(rspace, payload_pointer, channel_bytes_len, data_bytes_len, persist)
    logMemoryUsage("replay_produce", beforeBytes)
    result
  }

  def replay_consume_with_logging(
      rspace: Pointer,
      payload_pointer: Pointer,
      payload_bytes_len: Int
  ): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.replay_consume(rspace, payload_pointer, payload_bytes_len)
    logMemoryUsage("replay_consume", beforeBytes)
    result
  }

  def replay_create_checkpoint_with_logging(rspace: Pointer): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.replay_create_checkpoint(rspace)
    logMemoryUsage("replay_create_checkpoint", beforeBytes)
    result
  }

  def replay_clear_with_logging(rspace: Pointer): Unit = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    INSTANCE.replay_clear(rspace)
    logMemoryUsage("replay_clear", beforeBytes)
  }

  def replay_spawn_with_logging(replay_rspace_ptr: Pointer): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.replay_spawn(replay_rspace_ptr)
    logMemoryUsage("replay_spawn", beforeBytes)
    result
  }

  def rig_with_logging(rspace: Pointer, log_pointer: Pointer, log_bytes_len: Int): Unit = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    INSTANCE.rig(rspace, log_pointer, log_bytes_len)
    logMemoryUsage("rig", beforeBytes)
  }

  def check_replay_data_with_logging(rspace: Pointer): Unit = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    INSTANCE.check_replay_data(rspace)
    logMemoryUsage("check_replay_data", beforeBytes)
  }

  def hash_channel_with_logging(channel_pointer: Pointer, channel_bytes_len: Int): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.hash_channel(channel_pointer, channel_bytes_len)
    logMemoryUsage("hash_channel", beforeBytes)
    result
  }

  def hash_channels_with_logging(channels_pointer: Pointer, channels_bytes_len: Int): Pointer = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    val result      = INSTANCE.hash_channels(channels_pointer, channels_bytes_len)
    logMemoryUsage("hash_channels", beforeBytes)
    result
  }

  def deallocate_memory_with_logging(ptr: Pointer, len: Int): Unit = {
    val beforeBytes = INSTANCE.get_allocated_bytes()
    INSTANCE.deallocate_memory(ptr, len)
    logMemoryUsage("deallocate_memory", beforeBytes)
  }

  def hashChannel[C](channel: C): Blake2b256Hash =
    channel match {
      case value: { def toByteArray(): Array[Byte] } => {
        val channelBytes = value.toByteArray

        val payloadMemory = new Memory(channelBytes.length.toLong)
        payloadMemory.write(0, channelBytes, 0, channelBytes.length)

        val beforeBytes = INSTANCE.get_allocated_bytes()
        val hashResultPtr = INSTANCE.hash_channel(
          payloadMemory,
          channelBytes.length
        )
        logMemoryUsage("hashChannel", beforeBytes)

        // Not sure if these lines are needed
        // Need to figure out how to deallocate each memory instance
        payloadMemory.clear()

        if (hashResultPtr != null) {
          val resultByteslength = hashResultPtr.getInt(0)

          try {
            val resultBytes = hashResultPtr.getByteArray(4, resultByteslength)
            val hashProto   = HashProto.parseFrom(resultBytes)
            val hash =
              Blake2b256Hash.fromByteArray(hashProto.hash.toByteArray)

            hash

          } catch {
            case e: Throwable =>
              println("Error during scala hashChannel operation: " + e)
              throw e
          } finally {
            // Deallocate full buffer length including 4-byte length prefix
            val beforeDealloc = INSTANCE.get_allocated_bytes()
            INSTANCE.deallocate_memory(hashResultPtr, resultByteslength + 4)
            logMemoryUsage("hashChannel_deallocate", beforeDealloc)
          }
        } else {
          println("hashResultPtr is null")
          throw new RuntimeException("hashResultPtr is null")
        }
      }
      case _ => throw new IllegalArgumentException("Type does not have a toByteArray method")
    }

  def hashChannels[C](channels: Seq[C]): Blake2b256Hash =
    channels.head match {
      case value: { def toByteArray(): Array[Byte] } => {
        val channelsBytes = value.toByteArray

        val payloadMemory = new Memory(channelsBytes.length.toLong)
        payloadMemory.write(0, channelsBytes, 0, channelsBytes.length)

        val beforeBytes = INSTANCE.get_allocated_bytes()
        val hashResultPtr = INSTANCE.hash_channels(
          payloadMemory,
          channelsBytes.length
        )
        logMemoryUsage("hashChannels", beforeBytes)

        // Not sure if these lines are needed
        // Need to figure out how to deallocate each memory instance
        payloadMemory.clear()

        if (hashResultPtr != null) {
          val resultByteslength = hashResultPtr.getInt(0)

          try {
            val resultBytes = hashResultPtr.getByteArray(4, resultByteslength)
            val hashProto   = HashProto.parseFrom(resultBytes)
            val hash =
              Blake2b256Hash.fromByteArray(hashProto.hash.toByteArray)

            hash

          } catch {
            case e: Throwable =>
              println("Error during scala hashChannels operation: " + e)
              throw e
          } finally {
            // Deallocate full buffer length including 4-byte length prefix
            val beforeDealloc = INSTANCE.get_allocated_bytes()
            INSTANCE.deallocate_memory(hashResultPtr, resultByteslength + 4)
            logMemoryUsage("hashChannels_deallocate", beforeDealloc)
          }
        } else {
          println("hashResultPtr is null")
          throw new RuntimeException("hashResultPtr is null")
        }
      }
      case _ => {
        println("\nType does not have a toByteArray method")
        throw new IllegalArgumentException("Type does not have a toByteArray method")
      }
    }
}
