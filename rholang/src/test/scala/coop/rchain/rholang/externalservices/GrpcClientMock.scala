package coop.rchain.rholang.externalservices

import cats.effect.Concurrent
import coop.rchain.shared.Log

// Mock that succeeds on first grpcTell call only
class SingleGrpcClientMock(expectedHost: String, expectedPort: Int) extends GrpcClientService {
  @volatile private var callCount = 0

  private def isFirstCall: Boolean = {
    callCount += 1
    callCount == 1
  }

  override def initClientAndTell[F[_]: Concurrent: Log](
      clientHost: String,
      clientPort: Long,
      payload: String
  ): F[Unit] =
    if (isFirstCall && clientHost == expectedHost && clientPort == expectedPort)
      Concurrent[F].unit
    else
      Concurrent[F].raiseError(
        new RuntimeException("GrpcClient should be called only once with correct parameters")
      )

  def wasCalled: Boolean = callCount > 0
}

object GrpcClientMock {

  // Factory method for creating grpc mock
  def createSingleGrpcClientMock(expectedHost: String, expectedPort: Int): SingleGrpcClientMock =
    new SingleGrpcClientMock(expectedHost, expectedPort)
}
