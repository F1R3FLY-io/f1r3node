package coop.rchain.rholang.interpreter

import cats.Monad
import cats.effect.{Concurrent, Sync}
import cats.implicits._
import cats.syntax._
import coop.rchain.casper.client.external.v1.{
  ExternalCommunicationServiceV1GrpcMonix,
  UpdateNotificationResponse
}
import coop.rchain.casper.client.external.v1.ExternalCommunicationServiceV1GrpcMonix.ExternalCommunicationServiceStub
import coop.rchain.models.either.implicits.modelEitherSyntaxGrpModel
import coop.rchain.shared.syntax._
import coop.rchain.models.syntax._
import coop.rchain.monix.Monixable
import coop.rchain.shared.ThrowableOps.RichThrowable
import io.grpc.{Channel, ManagedChannel, ManagedChannelBuilder}
import monix.execution.{CancelableFuture, Scheduler}

object GrpcClient {

  def initClientAndTell[F[_]: Sync](
      clientHost: String,
      clientPort: Long,
      payload: String
  ): F[Unit] = {
    val channel: ManagedChannel =
      ManagedChannelBuilder
        .forAddress(clientHost, clientPort.toInt)
        .usePlaintext()
        .build

    val stub: ExternalCommunicationServiceStub =
      ExternalCommunicationServiceV1GrpcMonix.stub(channel)

    grpcTell(stub, clientHost, clientPort.toInt, payload)
  }

  protected def grpcTell[F[_]: Sync](
      externalCommunicationServiceStub: ExternalCommunicationServiceStub,
      clientHost: String,
      clientPort: Int,
      payload: String
  ): F[Unit] = {
    import monix.execution.Scheduler.Implicits.global

    val task = externalCommunicationServiceStub.sendNotification(
      coop.rchain.casper.clients.UpdateNotification(clientHost, clientPort, payload)
    )

    val token: CancelableFuture[UpdateNotificationResponse] = task.runToFuture(Scheduler.global)

    Sync[F]
      .delay(token.map { x =>
        println("Response from gRPC server: " + x)
        x
      })
      .map { token =>
        println("Waiting for response from gRPC server. Token: " + token)
      }
      .handleErrorWith { th =>
        Sync[F].raiseError[Unit](th)
      }

  }

}
