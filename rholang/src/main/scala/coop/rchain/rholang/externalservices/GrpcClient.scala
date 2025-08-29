package coop.rchain.rholang.externalservices

import cats.effect.{Concurrent, Sync}
import cats.implicits._
import coop.rchain.casper.client.external.v1.ExternalCommunicationServiceV1GrpcMonix.ExternalCommunicationServiceStub
import coop.rchain.casper.client.external.v1.{ExternalCommunicationServiceV1GrpcMonix}
import io.grpc.{ManagedChannel, ManagedChannelBuilder}
import com.typesafe.scalalogging.Logger
import java.util.concurrent.{ExecutorService, Executors}
import scala.concurrent.{ExecutionContext, Future}
import scala.util.control.NonFatal
import monix.execution.Scheduler
import monix.execution.Scheduler.Implicits.global

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
  def initClientAndTell[F[_]: Concurrent](
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

  override def initClientAndTell[F[_]: Concurrent](
      clientHost: String,
      clientPort: Long,
      payload: String
  ): F[Unit] =
    Concurrent[F].unit
}

/**
  * Production implementation of GrpcClientService with custom thread pool.
  *
  * This implementation uses a dedicated thread pool for gRPC operations instead
  * of the default pool. It properly handles asynchronous operations using
  * F.async and custom ExecutionContext.
  */
class RealGrpcService extends GrpcClientService {

  // Custom thread pool for gRPC operations - not using default pool as requested
  private val grpcExecutorService: ExecutorService = Executors.newFixedThreadPool(4)
  private val customScheduler: Scheduler = Scheduler(
    ExecutionContext.fromExecutor(grpcExecutorService)
  )

  // Shutdown executor on JVM shutdown
  sys.addShutdownHook {
    grpcExecutorService.shutdown()
  }

  override def initClientAndTell[F[_]: Concurrent](
      clientHost: String,
      clientPort: Long,
      payload: String
  ): F[Unit] =
    // Use F.async to properly handle asynchronous gRPC calls without runSyncUnsafe
    Concurrent[F].async[Unit] { callback =>
      val future = Future {
        val channel: ManagedChannel =
          ManagedChannelBuilder
            .forAddress(clientHost, clientPort.toInt)
            .usePlaintext()
            .build

        try {
          val stub: ExternalCommunicationServiceStub =
            ExternalCommunicationServiceV1GrpcMonix.stub(channel)

          val task = stub.sendNotification(
            coop.rchain.casper.clients.UpdateNotification(clientHost, clientPort.toInt, payload)
          )

          // Execute the task properly without runSyncUnsafe using custom scheduler
          val result = task.runToFuture(customScheduler)
          result.foreach { token =>
            println(s"gRPC message sent successfully. Token: $token")
            callback(Right(()))
          }(customScheduler)
          result.failed.foreach { error =>
            println(s"gRPC call failed: ${error.getMessage}")
            callback(Left(error))
          }(customScheduler)

        } catch {
          case NonFatal(error) =>
            println(s"Failed to initialize gRPC client: ${error.getMessage}")
            callback(Left(error))
        } finally {
          // Clean up the channel
          channel.shutdown()
        }
      }(customScheduler)

      future.failed.foreach { error =>
        println(s"Future execution failed: ${error.getMessage}")
        callback(Left(error))
      }(customScheduler)
    }
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
