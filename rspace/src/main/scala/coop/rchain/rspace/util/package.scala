package coop.rchain.rspace

import coop.rchain.rspace.trace.Produce
import coop.rchain.shared.Serialize
import scodec.bits.ByteVector

import java.nio.charset.StandardCharsets
import scala.util.Try

package object util {

  def unpackProduceSeq[C, P, K, R](
      v: Seq[Option[(ContResult[C, P, K], Seq[Result[C, R]], Produce)]]
  ): Seq[Option[(K, Seq[R], Produce)]] =
    v.map(unpackProduceOption)

  def unpackSeq[C, P, K, R](
                             v: Seq[Option[(ContResult[C, P, K], Seq[Result[C, R]])]]
                           ): Seq[Option[(K, Seq[R])]] =
    v.map(unpackOption)

  def unpackProduceOption[C, P, K, R](
      v: Option[(ContResult[C, P, K], Seq[Result[C, R]], Produce)]
  ): Option[(K, Seq[R], Produce)] =
    v.map(unpackProduceTuple)

  def unpackOption[C, P, K, R](
                                v: Option[(ContResult[C, P, K], Seq[Result[C, R]])]
                              ): Option[(K, Seq[R])] =
    v.map(unpackTuple)

  def unpackProduceTuple[C, P, K, R](
      v: (ContResult[C, P, K], Seq[Result[C, R]], Produce)
  ): (K, Seq[R], Produce) =
    v match {
      case (ContResult(continuation, _, _, _, _), data, previous) =>
        (continuation, data.map(_.matchedDatum), previous)
    }

  def unpackTuple[C, P, K, R](
                               v: (ContResult[C, P, K], Seq[Result[C, R]])
                             ): (K, Seq[R]) =
    v match {
      case (ContResult(continuation, _, _, _, _), data) =>
        (continuation, data.map(_.matchedDatum))
    }

  def unpackOptionWithPeek[C, P, K, R](
      v: Option[(ContResult[C, P, K], Seq[Result[C, R]])]
  ): Option[(K, Seq[(C, R, R, Boolean)], Boolean)] =
    v.map(unpackTupleWithPeek)

  def unpackTupleWithPeek[C, P, K, R](
      v: (ContResult[C, P, K], Seq[Result[C, R]])
  ): (K, Seq[(C, R, R, Boolean)], Boolean) =
    v match {
      case (ContResult(continuation, _, _, _, peek), data) =>
        (
          continuation,
          data.map(d => (d.channel, d.matchedDatum, d.removedDatum, d.persistent)),
          peek
        )
    }

  /**
    * Extracts a continuation from a produce result
    */
  def getK[A, K](t: Option[(K, A)]): K =
    t.map(_._1).get
  def getProduceK[A, K](t: Option[(K, A, Produce)]): K =
    t.map(_._1).get

  /** Runs a continuation with the accompanying data
    */
  def runK[T](e: Option[((T) => Unit, T)]): Unit =
    e.foreach { case (k, data) => k(data) }
  def runProduceK[T](e: Option[((T) => Unit, T, Produce)]): Unit =
    e.foreach { case (k, data, _) => k(data) }

  /** Runs a list of continuations with the accompanying data
    */
  def runKs[T](t: Seq[Option[((T) => Unit, T)]]): Unit =
    t.foreach { case Some((k, data)) => k(data); case None => () }
  def runProduceKs[T](t: Seq[Option[((T) => Unit, T, Produce)]]): Unit =
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
