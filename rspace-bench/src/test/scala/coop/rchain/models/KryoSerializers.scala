package coop.rchain.models

import java.nio.ByteBuffer

import scala.reflect.ClassTag

import com.esotericsoftware.kryo.{Kryo, Serializer}
import com.esotericsoftware.kryo.io._
import coop.rchain.models.Var.VarInstance
import org.objenesis.strategy.StdInstantiatorStrategy

trait Serialize2ByteBuffer[A] {
  def encode(a: A): ByteBuffer
  def decode(bytes: ByteBuffer): A

}

class DefaultSerializer[T](implicit tag: ClassTag[T]) extends Serializer[T] {

  def defaultSerializer(kryo: Kryo): Serializer[T] =
    kryo
      .getDefaultSerializer(tag.runtimeClass)
      .asInstanceOf[Serializer[T]]

  override def write(kryo: Kryo, output: Output, e: T): Unit =
    defaultSerializer(kryo).write(kryo, output, e)

  override def read(
      kryo: Kryo,
      input: Input,
      `type`: Class[_ <: T]
  ): T = defaultSerializer(kryo).read(kryo, input, `type`)

}

object KryoSerializers {

  object ParMapSerializer extends Serializer[ParMap] {
    import ParMapTypeMapper._

    override def write(kryo: Kryo, output: Output, parMap: ParMap): Unit =
      // Handle empty ParMap specially to avoid Kryo serialization issues with empty EMap.kvs
      if (parMap.ps.sortedList.isEmpty) {
        // Create an EMap with an empty but properly initialized list
        val emptyEMap = EMap(
          Seq.empty,
          parMap.locallyFree.value,
          parMap.connectiveUsed,
          parMap.remainder
        )
        kryo.writeObject(output, emptyEMap)
      } else {
        kryo.writeObject(output, parMapToEMap(parMap))
      }

    override def read(kryo: Kryo, input: Input, `type`: Class[_ <: ParMap]): ParMap =
      emapToParMap(kryo.readObject(input, classOf[EMap]))
  }

  object ParSetSerializer extends Serializer[ParSet] {
    import ParSetTypeMapper._

    override def write(kryo: Kryo, output: Output, parSet: ParSet): Unit =
      kryo.writeObject(output, parSetToESet(parSet))

    override def read(kryo: Kryo, input: Input, `type`: Class[_ <: ParSet]): ParSet =
      esetToParSet(kryo.readObject(input, classOf[ESet]))
  }

  object EMapSerializer extends Serializer[EMap] {
    import scala.collection.immutable.BitSet

    override def write(kryo: Kryo, output: Output, emap: EMap): Unit = {
      // Write kvs list - handle empty list specially
      output.writeInt(emap.kvs.size, true)
      emap.kvs.foreach(kv => kryo.writeObject(output, kv))
      // Write locallyFree BitSet as a collection of integers
      val bits = emap.locallyFree.toSeq
      output.writeInt(bits.size, true)
      bits.foreach(bit => output.writeInt(bit, true))
      output.writeBoolean(emap.connectiveUsed)
      // Write remainder
      if (emap.remainder.isDefined) {
        output.writeBoolean(true)
        kryo.writeObject(output, emap.remainder.get)
      } else {
        output.writeBoolean(false)
      }
    }

    override def read(kryo: Kryo, input: Input, `type`: Class[_ <: EMap]): EMap = {
      val kvsSize = input.readInt(true)
      val kvs = if (kvsSize == 0) {
        Seq.empty
      } else {
        (0 until kvsSize).map(_ => kryo.readObject(input, classOf[KeyValuePair])).toSeq
      }
      // Read locallyFree BitSet from collection of integers
      val bitsSize = input.readInt(true)
      val locallyFree = if (bitsSize == 0) {
        BitSet.empty
      } else {
        BitSet((0 until bitsSize).map(_ => input.readInt(true)): _*)
      }
      val connectiveUsed = input.readBoolean()
      val remainder = if (input.readBoolean()) {
        Some(kryo.readObject(input, classOf[Var]))
      } else {
        None
      }
      EMap(kvs, locallyFree, connectiveUsed, remainder)
    }
  }

  def emptyReplacingSerializer[T](thunk: T => Boolean, replaceWith: T)(implicit tag: ClassTag[T]) =
    new DefaultSerializer[T] {
      override def read(
          kryo: Kryo,
          input: Input,
          `type`: Class[_ <: T]
      ): T = {
        val read = super.read(kryo, input, `type`)
        if (thunk(read))
          replaceWith
        else read
      }
    }

  val TaggedContinuationSerializer =
    emptyReplacingSerializer[TaggedContinuation](_.taggedCont.isEmpty, TaggedContinuation())

  val VarSerializer =
    emptyReplacingSerializer[Var](_.varInstance.isEmpty, Var())

  val ExprSerializer =
    emptyReplacingSerializer[Expr](_.exprInstance.isEmpty, Expr())

  val UnfSerializer =
    emptyReplacingSerializer[GUnforgeable](_.unfInstance.isEmpty, GUnforgeable())

  val ConnectiveSerializer =
    emptyReplacingSerializer[Connective](_.connectiveInstance.isEmpty, Connective())

  val NoneSerializer: DefaultSerializer[None.type] =
    emptyReplacingSerializer[None.type](_.isEmpty, None)

  val kryo = new Kryo()
  kryo.register(classOf[ParMap], ParMapSerializer)
  kryo.register(classOf[ParSet], ParSetSerializer)
  kryo.register(classOf[EMap], EMapSerializer)
  kryo.register(classOf[TaggedContinuation], TaggedContinuationSerializer)
  kryo.register(classOf[Var], VarSerializer)
  kryo.register(classOf[Expr], ExprSerializer)
  kryo.register(classOf[GUnforgeable], UnfSerializer)
  kryo.register(classOf[Connective], ConnectiveSerializer)
  kryo.register(None.getClass, NoneSerializer)

  kryo.setRegistrationRequired(false)
  // Support deserialization of classes without no-arg constructors
  kryo.setInstantiatorStrategy(new StdInstantiatorStrategy())

  def serializer[A](of: Class[A]): Serialize2ByteBuffer[A] = new Serialize2ByteBuffer[A] {

    private[this] val noSizeLimit = -1

    override def encode(gnat: A): ByteBuffer = {
      val output = new ByteBufferOutput(1024, noSizeLimit)
      kryo.writeObject(output, gnat)
      output.close()

      val buf = output.getByteBuffer
      buf.flip()
      buf
    }

    override def decode(bytes: ByteBuffer): A = {
      val input = new ByteBufferInput(bytes)
      val res   = kryo.readObject(input, of)
      input.close()
      res
    }

  }
}
