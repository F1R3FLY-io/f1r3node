package rspacePlusPlus

import cats.Applicative
import coop.rchain.models.{BindPattern, ListParWithRandom, Par, TaggedContinuation}
import coop.rchain.models.rspace_plus_plus_types.{ConsumeResult, ProduceResult}
import coop.rchain.shared.Log
import cats.effect.{Concurrent, Sync}
import cats.syntax.all._
import com.sun.jna.{Memory, Native, Pointer}
import java.nio.file.Files
import cats.effect.ContextShift
import scala.concurrent.ExecutionContext
import coop.rchain.metrics.Metrics
import coop.rchain.shared.Serialize
import coop.rchain.rspace.trace.Produce
import scala.collection.SortedSet
import coop.rchain.rspace.trace.Consume
import coop.rchain.rspace.Checkpoint
import coop.rchain.models.rspace_plus_plus_types.CheckpointProto
import coop.rchain.rspace.hashing.Blake2b256Hash
import coop.rchain.models.rspace_plus_plus_types.IOEventProto
import coop.rchain.rspace.trace.COMM
import coop.rchain.rspace.ContResult
import coop.rchain.rspace.Result
import coop.rchain.models.rspace_plus_plus_types.ConsumeParams
import coop.rchain.models.rspace_plus_plus_types.SortedSetElement
import rspacePlusPlus.JNAInterfaceLoader.{INSTANCE => INSTANCE}

class ReplayRSpacePlusPlus[F[_]: Concurrent: ContextShift: Log: Metrics, C, P, A, K](
    rspacePointer: Pointer
)(
    implicit
    serializeC: Serialize[Par],
    serializeP: Serialize[BindPattern],
    serializeA: Serialize[ListParWithRandom],
    serializeK: Serialize[TaggedContinuation],
    scheduler: ExecutionContext
) extends RSpaceOpsPlusPlus[F](rspacePointer)
    with IReplaySpacePlusPlus[F, Par, BindPattern, ListParWithRandom, TaggedContinuation] {

  protected def logF: Log[F] = Log[F]
  def syncF: Sync[F]         = Sync[F]

  protected[this] override def lockedConsume(
      channels: Seq[C],
      patterns: Seq[P],
      continuation: K,
      persist: Boolean,
      peeks: SortedSet[Int],
      consumeRef: Consume
  ): F[MaybeConsumeResult] =
    for {
      result <- Sync[F].delay {
                 // println(s"\nhit consume in scala ${}");
                 //  dataLogger.debug("hit consume in scala")

                 val consumeParams = ConsumeParams(
                   channels,
                   patterns,
                   Some(continuation),
                   persist,
                   peeks.map(SortedSetElement(_)).toSeq
                 )
                 val consumeParamsBytes = consumeParams.toByteArray

                 val payloadMemory = new Memory(consumeParamsBytes.length.toLong)
                 payloadMemory.write(0, consumeParamsBytes, 0, consumeParamsBytes.length)

                 val consumeResultPtr = INSTANCE.replay_consume(
                   rspacePointer,
                   payloadMemory,
                   consumeParamsBytes.length
                 )

                 // Clear parameter buffer (Memory is GC-managed)
                 payloadMemory.clear()

                 //  val jsonString = consumeResultPtr.getString(0)
                 //  println("\njsonString: " + jsonString)

                 //  if (jsonString != "") {
                 if (consumeResultPtr != null) {
                   val resultByteslength = consumeResultPtr.getInt(0)
                   try {
                     // println("\nresultByteslength: " + resultByteslength)
                     val resultBytes = consumeResultPtr.getByteArray(4, resultByteslength)
                     // println("resultBytes length: " + resultBytes.length)
                     val consumeResult = ConsumeResult.parseFrom(resultBytes)

                     //  val actionResult = ActionResult.parseFrom(jsonString.getBytes())

                     val contResult = consumeResult.contResult.get
                     val results    = consumeResult.results

                     Some(
                       (
                         ContResult(
                           continuation = contResult.continuation.get,
                           persistent = contResult.persistent,
                           channels = contResult.channels,
                           patterns = contResult.patterns,
                           peek = contResult.peek
                         ),
                         results.map(
                           r =>
                             Result(
                               channel = r.channel.get,
                               matchedDatum = r.matchedDatum.get,
                               removedDatum = r.removedDatum.get,
                               persistent = r.persistent
                             )
                         )
                       )
                     )
                   } catch {
                     case e: Throwable =>
                       println("Error during scala consume operation: " + e)
                       throw e
                   } finally {
                     INSTANCE.deallocate_memory(consumeResultPtr, resultByteslength + 4)
                   }
                 } else {
                   //  println("\nreturning None because ptr was null")
                   None
                 }
               }
    } yield result

  protected[this] override def lockedProduce(
      channel: C,
      data: A,
      persist: Boolean,
      produceRef: Produce
  ): F[MaybeProduceResult] =
    for {
      result <- Sync[F].delay {
                 //  println("\nHit scala produce, data: " + data)
                 val channelBytes       = channel.toByteArray
                 val channelBytesLength = channelBytes.length
                 val dataBytes          = data.toByteArray
                 val dataBytesLength    = dataBytes.length

                 val payloadSize   = channelBytesLength.toLong + dataBytesLength.toLong
                 val payloadMemory = new Memory(payloadSize)

                 payloadMemory.write(0, channelBytes, 0, channelBytesLength)
                 payloadMemory.write(channelBytesLength.toLong, dataBytes, 0, dataBytesLength)

                 val produceResultPtr = INSTANCE.replay_produce(
                   rspacePointer,
                   payloadMemory,
                   channelBytesLength,
                   dataBytesLength,
                   persist
                 )

                 // Clear parameter buffer (Memory is GC-managed)
                 payloadMemory.clear()

                 if (produceResultPtr != null) {
                   val resultByteslength = produceResultPtr.getInt(0)

                   try {
                     val resultBytes   = produceResultPtr.getByteArray(4, resultByteslength)
                     val produceResult = ProduceResult.parseFrom(resultBytes)
                     val contResult    = produceResult.contResult.get
                     val results       = produceResult.results
                     val produce       = produceResult.produce.get

                     Some(
                       (
                         ContResult(
                           continuation = contResult.continuation.get,
                           persistent = contResult.persistent,
                           channels = contResult.channels,
                           patterns = contResult.patterns,
                           peek = contResult.peek
                         ),
                         results.map(
                           r =>
                             Result(
                               channel = r.channel.get,
                               matchedDatum = r.matchedDatum.get,
                               removedDatum = r.removedDatum.get,
                               persistent = r.persistent
                             )
                         ),
                         Produce(
                           channelsHash =
                             Blake2b256Hash.fromByteArray(produce.channelHash.toByteArray),
                           hash = Blake2b256Hash.fromByteArray(produce.hash.toByteArray),
                           persistent = produce.persistent,
                           isDeterministic = produce.isDeterministic,
                           outputValue = produce.outputValue.map(_.toByteArray)
                         )
                       )
                     )
                   } catch {
                     case e: Throwable =>
                       println("Error during scala produce operation: " + e)
                       throw e
                   } finally {
                     INSTANCE.deallocate_memory(produceResultPtr, resultByteslength + 4)
                   }
                 } else {
                   None
                 }
               }
    } yield result

  override def createCheckpoint(): F[Checkpoint] = {
    for {
      result <- Sync[F].delay {
                 //  println("\nhit scala createCheckpoint")

                 val checkpointResultPtr = INSTANCE.replay_create_checkpoint(
                   rspacePointer
                 )

                 if (checkpointResultPtr != null) {
                   val resultByteslength = checkpointResultPtr.getInt(0)

                   try {
                     val resultBytes     = checkpointResultPtr.getByteArray(4, resultByteslength)
                     val checkpointProto = CheckpointProto.parseFrom(resultBytes)
                     val checkpointRoot  = checkpointProto.root

                     //  println(
                     //    "\nscala createCheckpoint root: " + Blake2b256Hash
                     //      .fromByteArray(checkpointRoot.toByteArray)
                     //  )

                     //TODO: No need to parse log here as it will always be empty
                     val checkpointLogProto = checkpointProto.log
                     val checkpointLog = checkpointLogProto.map {
                       case eventProto if eventProto.eventType.isComm =>
                         val commProto = eventProto.eventType.comm.get
                         val consume   = commProto.consume
                         val produces = commProto.produces.map { produceProto =>
                           Produce(
                             channelsHash =
                               Blake2b256Hash.fromByteArray(produceProto.channelHash.toByteArray),
                             hash = Blake2b256Hash.fromByteArray(produceProto.hash.toByteArray),
                             persistent = produceProto.persistent,
                             isDeterministic = produceProto.isDeterministic,
                             outputValue = produceProto.outputValue.map(_.toByteArray)
                           )
                         }
                         val peeks = commProto.peeks.map(_.value).to[SortedSet]
                         val timesRepeated = commProto.timesRepeated.map { entry =>
                           val produceProto = entry.key.get
                           val produce = Produce(
                             channelsHash =
                               Blake2b256Hash.fromByteArray(produceProto.channelHash.toByteArray),
                             hash = Blake2b256Hash.fromByteArray(produceProto.hash.toByteArray),
                             persistent = produceProto.persistent,
                             isDeterministic = produceProto.isDeterministic,
                             outputValue = produceProto.outputValue.map(_.toByteArray)
                           )
                           produce -> entry.value
                         }.toMap
                         COMM(
                           consume = Consume(
                             channelsHashes = consume.get.channelHashes
                               .map(bs => Blake2b256Hash.fromByteArray(bs.toByteArray)),
                             hash = Blake2b256Hash.fromByteArray(consume.get.hash.toByteArray),
                             persistent = consume.get.persistent
                           ),
                           produces = produces,
                           peeks = peeks,
                           timesRepeated = timesRepeated
                         )
                       case eventProto if eventProto.eventType.isIoEvent =>
                         val ioEventProto = eventProto.eventType.ioEvent.get
                         ioEventProto.ioEventType match {
                           case IOEventProto.IoEventType.Produce(produceProto) =>
                             Produce(
                               channelsHash =
                                 Blake2b256Hash.fromByteArray(produceProto.channelHash.toByteArray),
                               hash = Blake2b256Hash.fromByteArray(produceProto.hash.toByteArray),
                               persistent = produceProto.persistent,
                               isDeterministic = produceProto.isDeterministic,
                               outputValue = produceProto.outputValue.map(_.toByteArray)
                             )
                           case IOEventProto.IoEventType.Consume(consumeProto) =>
                             Consume(
                               channelsHashes = consumeProto.channelHashes
                                 .map(bs => Blake2b256Hash.fromByteArray(bs.toByteArray)),
                               hash = Blake2b256Hash.fromByteArray(consumeProto.hash.toByteArray),
                               persistent = consumeProto.persistent
                             )
                           case _ =>
                             throw new RuntimeException("Unknown IOEvent type")
                         }
                       case _ =>
                         throw new RuntimeException("Unknown Event type")
                     }

                     //  println("\n log in replayCreateCheckpoint: " + checkpointLog.length)

                     Checkpoint(
                       root = Blake2b256Hash.fromByteArray(checkpointRoot.toByteArray),
                       log = checkpointLog
                     )
                     //  Checkpoint(
                     //    root = Blake2b256Hash.fromByteArray(checkpointRoot.toByteArray),
                     //    log = Seq.empty[Event]
                     //  )

                   } catch {
                     case e: Throwable =>
                       println("Error during scala createCheckpoint operation: " + e)
                       throw e
                   } finally {
                     INSTANCE.deallocate_memory(checkpointResultPtr, resultByteslength + 4)
                   }
                 } else {
                   println(
                     "Error during createCheckpoint operation: Checkpoint pointer from rust was null"
                   )
                   throw new RuntimeException("Checkpoint pointer from rust was null")
                 }
               }
    } yield result
  }

  override def clear(): F[Unit] = Sync[F].delay { INSTANCE.replay_clear(rspacePointer) }

  def spawn: F[IReplaySpacePlusPlus[F, Par, BindPattern, ListParWithRandom, TaggedContinuation]] =
    for {
      result <- Sync[F].delay {
                 val rspace = INSTANCE.replay_spawn(
                   rspacePointer
                 )

                 new ReplayRSpacePlusPlus[
                   F,
                   Par,
                   BindPattern,
                   ListParWithRandom,
                   TaggedContinuation
                 ](rspace)
               }
    } yield result

  def getRspacePointer: Pointer =
    rspacePointer
}

object ReplayRSpacePlusPlus {

  /**
    * Creates [[ReplayRSpace]] from [[HistoryRepository]] and [[HotStore]].
    */
  // def apply[F[_]: Concurrent: ContextShift: Log: Metrics, C, P, A, K](
  //     storePath: String
  // )(
  //     implicit
  //     serializeC: Serialize[Par],
  //     serializeP: Serialize[BindPattern],
  //     serializeA: Serialize[ListParWithRandom],
  //     serializeK: Serialize[TaggedContinuation],
  //     scheduler: ExecutionContext
  // ): F[ReplayRSpacePlusPlus[F, C, P, A, K]] =
  //   Sync[F].delay {
  //     val replayRspacePointer = INSTANCE.space_new(storePath);
  //     new ReplayRSpacePlusPlus[F, C, P, A, K](replayRspacePointer)
  //   }

}
