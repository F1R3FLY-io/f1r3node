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

  def hashChannel[C](channel: C): Blake2b256Hash =
    channel match {
      case value: { def toByteArray(): Array[Byte] } => {
        val channelBytes = value.toByteArray

        val payloadMemory = new Memory(channelBytes.length.toLong)
        payloadMemory.write(0, channelBytes, 0, channelBytes.length)

        val hashResultPtr = INSTANCE.hash_channel(
          payloadMemory,
          channelBytes.length
        )

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
            INSTANCE.deallocate_memory(hashResultPtr, resultByteslength + 4)
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

        val hashResultPtr = INSTANCE.hash_channels(
          payloadMemory,
          channelsBytes.length
        )

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
            INSTANCE.deallocate_memory(hashResultPtr, resultByteslength + 4)
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
