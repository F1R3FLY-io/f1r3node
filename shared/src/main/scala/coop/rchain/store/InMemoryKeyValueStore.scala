package coop.rchain.store

import cats.effect.Sync
import scodec.bits.ByteVector

import java.nio.ByteBuffer
import scala.collection.concurrent.TrieMap

final case class InMemoryKeyValueStore[F[_]: Sync]() extends KeyValueStore[F] {

  val state = TrieMap[ByteBuffer, ByteVector]()

  override def get[T](keys: Seq[ByteBuffer], fromBuffer: ByteBuffer => T): F[Seq[Option[T]]] =
    Sync[F].delay {
      // println("\ninMemState get: " + state);
      // println("\ninMemState get keys: " + keys);
      val res = keys.map { key =>
        val value = state.get(key)
        // println(s"\nRetrieved value for key $key: $value")
        value.map(_.toByteBuffer).map(fromBuffer)
      }
      // println("\nresults in get: " + res)
      res
    }

  override def put[T](kvPairs: Seq[(ByteBuffer, T)], toBuffer: T => ByteBuffer): F[Unit] =
    Sync[F].delay {
      // println("\nhit put in memKV");
      // println("\ninMemState before put: " + state);
      kvPairs
        .foreach {
          case (k, v) =>
            state.put(k, ByteVector(toBuffer(v)))
        }
      // println("\ninMemState after put: " + state)
    }

  override def delete(keys: Seq[ByteBuffer]): F[Int] =
    Sync[F].delay(keys.map(state.remove).count(_.nonEmpty))

  override def iterate[T](f: Iterator[(ByteBuffer, ByteBuffer)] => T): F[T] =
    Sync[F].delay {
      val iter = state.toIterator.map { case (k, v) => (k, v.toByteBuffer) }
      f(iter)
    }

  override def nonEmpty: F[Boolean] = Sync[F].delay(state.nonEmpty)

  def clear(): Unit = state.clear()

  def numRecords(): Int = state.size

  def sizeBytes(): Long =
    state.map { case (byteBuffer, byteVector) => byteBuffer.capacity + byteVector.size }.sum

}
