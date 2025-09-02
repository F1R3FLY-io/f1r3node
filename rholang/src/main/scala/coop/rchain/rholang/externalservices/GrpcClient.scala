package coop.rchain.rholang.externalservices

import cats.effect.Concurrent
import cats.implicits._
import coop.rchain.casper.client.external.v1.ExternalCommunicationServiceV1GrpcMonix.ExternalCommunicationServiceStub
import coop.rchain.casper.client.external.v1.{
  ExternalCommunicationServiceV1GrpcMonix,
  UpdateNotificationResponse
}
import coop.rchain.casper.client.external.v1.UpdateNotificationResponse._
import coop.rchain.shared.{Log, LogSource}
import io.grpc.{ManagedChannel, ManagedChannelBuilder}

import java.util.concurrent.{ExecutorService, Executors}
import scala.concurrent.{ExecutionContext, Future}
import scala.util.control.NonFatal
import monix.execution.Scheduler

/**
  * Service interface for gRPC client operations.
  *
  * This trait abstracts gRPC client functionality to enable dependency injection
  * and testing. Implementations should handle connection management and error handling.
  */
trait GrpcClientService {

  /**
    * Initializes a gRPC client connection and sends a message.
    *
    * @param clientHost The hostname or IP address of the gRPC server
    * @param clientPort The port number of the gRPC server
    * @param payload The message payload to send
    * @tparam F The effect type (e.g., IO, Task)
    * @return Effect that completes when the message is sent
    */
  def initClientAndTell[F[_]: Concurrent: Log](
      clientHost: String,
      clientPort: Long,
      payload: String
  ): F[Unit]
}

/**
  * No-operation implementation of GrpcClientService.
  *
  * This implementation logs requests but doesn't perform actual gRPC calls.
  * Useful for testing and when gRPC functionality is disabled.
  */
class NoOpGrpcService extends GrpcClientService {

  implicit private val logSource: LogSource = LogSource(this.getClass)

  override def initClientAndTell[F[_]: Concurrent: Log](
      clientHost: String,
      clientPort: Long,
      payload: String
  ): F[Unit] =
    for {
      _      <- Log[F].debug(s"gRPC service is disabled - call to $clientHost:$clientPort ignored")
      result <- Concurrent[F].unit
    } yield result
}

/**
  * Production implementation of GrpcClientService with custom thread pool.
  *
  * This implementation uses a dedicated thread pool for gRPC operations instead
  * of the default pool. It properly handles asynchronous operations using
  * F.async and custom ExecutionContext.
  */
class RealGrpcService extends GrpcClientService {

  implicit private val logSource: LogSource        = LogSource(this.getClass)
  private val grpcExecutorService: ExecutorService = Executors.newFixedThreadPool(4)
  private val customScheduler: Scheduler = Scheduler(
    ExecutionContext.fromExecutor(grpcExecutorService)
  )

  // Shutdown executor on JVM shutdown
  sys.addShutdownHook {
    grpcExecutorService.shutdown()
  }

  override def initClientAndTell[F[_]: Concurrent: Log](
      clientHost: String,
      clientPort: Long,
      payload: String
  ): F[Unit] =
    for {
      _ <- Log[F].info(s"Initiating gRPC call to $clientHost:$clientPort")
      result <- Concurrent[F]
                 .async[UpdateNotificationResponse] { callback =>
                   try {
                     val channel: ManagedChannel =
                       ManagedChannelBuilder
                         .forAddress(clientHost, clientPort.toInt)
                         .usePlaintext()
                         .build

                     val stub: ExternalCommunicationServiceStub =
                       ExternalCommunicationServiceV1GrpcMonix.stub(channel)

                     val task = stub.sendNotification(
                       coop.rchain.casper.clients
                         .UpdateNotification(clientHost, clientPort.toInt, payload)
                     )

                     // Use the future directly from task.runToFuture without explicit Future wrapper
                     val taskFuture = task.runToFuture(customScheduler)
                     taskFuture.foreach { response =>
                       callback(Right(response))
                       // Clean up the channel after successful response
                       channel.shutdown()
                     }(customScheduler)
                     taskFuture.failed.foreach { error =>
                       callback(Left(error))
                       // Clean up the channel on error
                       channel.shutdown()
                     }(customScheduler)

                   } catch {
                     case NonFatal(error) =>
                       callback(Left(error))
                   }
                 }
                 .recoverWith {
                   case error =>
                     Log[F].warn(
                       s"gRPC call to $clientHost:$clientPort failed: ${error.getMessage}"
                     ) >>
                       Concurrent[F].raiseError(error)
                 }
      // Handle logging of results in the F effect context
      _ <- result.message match {
            case Message.Result(resultStr) =>
              Log[F].debug(s"gRPC message sent successfully. Result: $resultStr")
            case Message.Error(error) =>
              Log[F].warn(s"gRPC call returned error: ${error.messages.mkString(", ")}")
            case Message.Empty =>
              Log[F].info("gRPC call returned empty response")
          }
    } yield ()
}

object GrpcClientService {

  /**
    * Default production implementation of GrpcClientService.
    * Uses custom thread pool and proper async handling.
    */
  lazy val instance: GrpcClientService = new RealGrpcService

  /**
    * No-operation instance for testing and disabled scenarios.
    */
  lazy val noOpInstance: GrpcClientService = new NoOpGrpcService
}
