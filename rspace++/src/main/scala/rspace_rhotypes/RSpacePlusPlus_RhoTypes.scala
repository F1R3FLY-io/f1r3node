package rspacePlusPlus

import cats.Applicative
import cats.implicits._
import com.sun.jna.{Memory, Native, Pointer}
import coop.rchain.models.rspace_plus_plus_types.{
  ChannelsProto,
  ConsumeParams,
  ConsumeProto,
  ConsumeResult,
  DatumProto,
  DatumsProto,
  FreeMapProto,
  HotStoreStateProto,
  InstallParams,
  JoinProto,
  JoinsProto,
  ProduceCounterMapEntry,
  ProduceResult,
  SoftCheckpointProto,
  SortedSetElement,
  StoreStateContMapEntry,
  StoreStateDataMapEntry,
  StoreStateInstalledContMapEntry,
  StoreStateInstalledJoinsMapEntry,
  StoreStateJoinsMapEntry,
  StoreToMapResult,
  WaitingContinuationProto,
  WaitingContinuationsProto
}
import coop.rchain.models.{BindPattern, ListParWithRandom, Par, TaggedContinuation}
import scala.collection.SortedSet
import coop.rchain.rspace.{ContResult, Result}
import coop.rchain.rspace.trace.{Consume, Produce}
import coop.rchain.rspace.{Checkpoint, SoftCheckpoint}
import coop.rchain.rspace.internal.{Datum, Row, WaitingContinuation}
import coop.rchain.rspace.hashing.Blake2b256Hash
import coop.rchain.shared.Log
import cats.effect.{Concurrent, Sync}
import coop.rchain.models.rspace_plus_plus_types.CheckpointProto
import coop.rchain.rspace.trace.Event
import coop.rchain.rspace.HotStoreState

import scala.collection.immutable.Map
import com.google.protobuf.ByteString
import coop.rchain.rspace.state.RSpaceExporter
import coop.rchain.rspace.state.RSpaceImporter
import coop.rchain.shared.Serialize
import coop.rchain.rspace.history.HistoryReader
import coop.rchain.rspace.HotStoreTrieAction
import coop.rchain.rspace.HotStoreAction
import coop.rchain.rspace.history.History
import coop.rchain.state.TrieNode
import java.nio.ByteBuffer
import scodec.bits.ByteVector
import com.typesafe.scalalogging.Logger
import cats.effect.ContextShift
import scala.concurrent.ExecutionContext
import rspacePlusPlus.state.{RSpacePlusPlusExporter, RSpacePlusPlusImporter}
import rspacePlusPlus.history.{RSpacePlusPlusHistoryReader}
import coop.rchain.rspace.state.exporters.RSpaceExporterItems.StoreItems
import java.nio.file.Path
import coop.rchain.metrics.Metrics
import coop.rchain.rspace.history.HistoryReaderBase
import rspacePlusPlus.history.{RSpacePlusPlusHistoryReaderBase, RSpacePlusPlusHistoryReaderBinary}
import coop.rchain.rspace.serializers.ScodecSerialize.DatumB
import coop.rchain.rspace.serializers.ScodecSerialize.WaitingContinuationB
import coop.rchain.rspace.serializers.ScodecSerialize.JoinsB
import coop.rchain.rspace.trace.COMM
import coop.rchain.models.rspace_plus_plus_types.IOEventProto
import coop.rchain.models.rspace_plus_plus_types.EventProto
import coop.rchain.models.rspace_plus_plus_types.CommProto
import rspacePlusPlus.JNAInterfaceLoader.{INSTANCE => INSTANCE}

/**
  * This class contains predefined types for Channel, Pattern, Data, and Continuation - RhoTypes
  * These types (C, P, A, K) MUST MATCH the corresponding types on the Rust side in 'rspace_rhotypes/lib.rs'
  */
class RSpacePlusPlus_RhoTypes[F[_]: Concurrent: ContextShift: Log: Metrics](rspacePointer: Pointer)(
    implicit
    serializeC: Serialize[Par],
    serializeP: Serialize[BindPattern],
    serializeA: Serialize[ListParWithRandom],
    serializeK: Serialize[TaggedContinuation],
    scheduler: ExecutionContext
) extends RSpaceOpsPlusPlus[F](
      rspacePointer
    )
    with ISpacePlusPlus[F, Par, BindPattern, ListParWithRandom, TaggedContinuation] {

  // val jnaLibraryPath = System.getProperty("jna.library.path")
  // println(s"Current jna.library.path: $jnaLibraryPath")

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

                 val consumeResultPtr = INSTANCE.consume(
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

                 val produceResultPtr = INSTANCE.produce(
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

  def print(): Unit =
    INSTANCE.space_print(rspacePointer)

  def createCheckpoint(): F[Checkpoint] =
    for {
      result <- Sync[F].delay {
                 //  println("\nhit scala createCheckpoint")

                 val checkpointResultPtr = INSTANCE.create_checkpoint(
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

                     //  println("\n log: " + checkpointLog)

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

  def spawn: F[ISpacePlusPlus[F, C, P, A, K]] =
    for {
      result <- Sync[F].delay {
                 //  println("\nhit scala spawn")

                 val rspace = INSTANCE.spawn(
                   rspacePointer
                 )

                 new RSpacePlusPlus_RhoTypes[F](rspace)
               }
    } yield result

  def getRspacePointer: Pointer =
    rspacePointer
}

object RSpacePlusPlus_RhoTypes {

  def create[F[_]: Concurrent: ContextShift: Log: Metrics](storePath: String)(
      implicit
      serializeC: Serialize[Par],
      serializeP: Serialize[BindPattern],
      serializeA: Serialize[ListParWithRandom],
      serializeK: Serialize[TaggedContinuation],
      scheduler: ExecutionContext
  ): F[RSpacePlusPlus_RhoTypes[F]] =
    Sync[F].delay {
      val rspacePointer = INSTANCE.space_new(storePath);
      new RSpacePlusPlus_RhoTypes[F](rspacePointer)
    }

  def createWithReplay[F[_]: Concurrent: ContextShift: Log: Metrics, C, P, A, K](
      storePath: String
  )(
      implicit
      serializeC: Serialize[Par],
      serializeP: Serialize[BindPattern],
      serializeA: Serialize[ListParWithRandom],
      serializeK: Serialize[TaggedContinuation],
      scheduler: ExecutionContext
  ): F[(RSpacePlusPlus_RhoTypes[F], ReplayRSpacePlusPlus[F, C, P, A, K])] =
    Sync[F].delay {
      val rspacePointer       = INSTANCE.space_new(storePath);
      val replayRspacePointer = INSTANCE.space_new_replay(rspacePointer);
      (
        new RSpacePlusPlus_RhoTypes[F](rspacePointer),
        new ReplayRSpacePlusPlus[F, C, P, A, K](replayRspacePointer)
      )
    }

  def spatialMatchResult[F[_]: Concurrent: Log, C, P, A, K](
      target: Par,
      pattern: Par
  ): F[Option[(Map[Int, Par], Unit)]] =
    for {
      result <- Sync[F].delay {
                 val targetBytes        = target.toByteArray
                 val targetBytesLength  = targetBytes.length
                 val patternBytes       = pattern.toByteArray
                 val patternBytesLength = patternBytes.length

                 if (targetBytesLength == 0 && patternBytesLength == 0) {
                   // This case means target and pattern are empty
                   Some(Map.empty[Int, Par], ())
                 } else {
                   val payloadSize   = targetBytesLength.toLong + patternBytesLength.toLong
                   val payloadMemory = new Memory(payloadSize)

                   payloadMemory.write(0, targetBytes, 0, targetBytesLength)
                   payloadMemory.write(
                     targetBytesLength.toLong,
                     patternBytes,
                     0,
                     patternBytesLength
                   )

                   val spatialMatchResultPtr = INSTANCE.spatial_match_result(
                     payloadMemory,
                     targetBytesLength,
                     patternBytesLength
                   )

                   // Clear parameter buffer (Memory is GC-managed)
                   payloadMemory.clear()

                   if (spatialMatchResultPtr != null) {
                     val resultByteslength = spatialMatchResultPtr.getInt(0)

                     try {
                       val resultBytes  = spatialMatchResultPtr.getByteArray(4, resultByteslength)
                       val freeMapProto = FreeMapProto.parseFrom(resultBytes)

                       Some((freeMapProto.entries, ()))
                     } catch {
                       case e =>
                         println("Error during scala spatialMatchResult operation: " + e)
                         throw e
                     } finally {
                       INSTANCE.deallocate_memory(spatialMatchResultPtr, resultByteslength + 4)
                     }
                   } else {
                     None
                   }
                 }

               }
    } yield result
}
