package coop.rchain.rspace

import coop.rchain.shared.Serialize
import scodec.bits.ByteVector

import java.nio.charset.StandardCharsets
import scala.util.Try

package object util {

  def unpackSeq[C, P, K, R](
      v: Seq[Option[(ContResult[C, P, K], Seq[Result[C, R]], Option[Any])]]
  ): Seq[Option[(K, Seq[R], Option[Any])]] =
    v.map(unpackOption)

  def unpackOption[C, P, K, R](
      v: Option[(ContResult[C, P, K], Seq[Result[C, R]], Option[Any])]
  ): Option[(K, Seq[R], Option[Any])] =
    v.map(unpackTuple)

  def unpackTuple[C, P, K, R](
      v: (ContResult[C, P, K], Seq[Result[C, R]], Option[Any])
  ): (K, Seq[R], Option[Any]) =
    v match {
      case (ContResult(continuation, _, _, _, _), data, previous) =>
        (continuation, data.map(_.matchedDatum), previous)
    }

  def unpackOptionWithPeek[C, P, K, R](
      v: Option[(ContResult[C, P, K], Seq[Result[C, R]], Option[Any])]
  ): Option[(K, Seq[(C, R, R, Boolean)], Boolean, Option[Any])] =
    v.map(unpackTupleWithPeek)

  def unpackTupleWithPeek[C, P, K, R](
      v: (ContResult[C, P, K], Seq[Result[C, R]], Option[Any])
  ): (K, Seq[(C, R, R, Boolean)], Boolean, Option[Any]) =
    v match {
      case (ContResult(continuation, _, _, _, peek), data, previous) =>
        (
          continuation,
          data.map(d => (d.channel, d.matchedDatum, d.removedDatum, d.persistent)),
          peek,
          previous
        )
    }

  /**
    * Extracts a continuation from a produce result
    */
  def getK[A, K](t: Option[(K, A, Option[Any])]): K =
    t.map(_._1).get

  /** Runs a continuation with the accompanying data
    */
  def runK[T](e: Option[((T) => Unit, T, Option[Any])]): Unit =
    e.foreach { case (k, data, _) => k(data) }

  /** Runs a list of continuations with the accompanying data
    */
  def runKs[T](t: Seq[Option[((T) => Unit, T, Option[Any])]]): Unit =
    t.foreach { case Some((k, data, _)) => k(data); case None => () }

  @SuppressWarnings(Array("org.wartremover.warts.Return"))
  def veccmp(a: ByteVector, b: ByteVector): Int = {
    val c = a.length - b.length
    if (c != 0) {
      c.toInt
    } else {
      for (i <- 0L until a.length) {
        //indexed access of two ByteVectors can be not fast enough,
        //however it is used by ByteVector creators (see === implementation)
        val ai = a(i)
        val bi = b(i)
        if (ai != bi) {
          return (ai & 0xFF) - (bi & 0xFF)
        }
      }
      0
    }
  }

  val ordByteVector: Ordering[ByteVector] = (a: ByteVector, b: ByteVector) => veccmp(a, b)

  val stringSerialize: Serialize[String] = new Serialize[String] {

    def encode(a: String): ByteVector =
      ByteVector.view(a.getBytes(StandardCharsets.UTF_8))

    def decode(bytes: ByteVector): Either[Throwable, String] =
      Try.apply(new String(bytes.toArray, StandardCharsets.UTF_8)).toEither
  }
}
