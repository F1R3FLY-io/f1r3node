package coop.rchain.rholang.interpreter

import cats.effect.Sync
import coop.rchain.rholang.OpenAIServiceMock

/**
  * Test utilities for ExternalServices.
  * 
  * This object provides factory methods and mock implementations for testing
  * external service integrations. It keeps test-specific code separate from
  * production implementations.
  */
object ExternalServicesTestUtils {

  /**
    * Creates ExternalServices with mock implementations for testing.
    * 
    * @param openAIService Mock OpenAI service implementation (defaults to non-deterministic mock)
    * @param grpcClient Mock gRPC client service implementation (defaults to no-op mock)
    * @return ExternalServices instance with mock implementations
    */
  def forTesting(
      openAIService: OpenAIService = OpenAIServiceMock.nonDeterministicService,
      grpcClient: GrpcClientService = GrpcClientServiceMock.noOp
  ): ExternalServices = ExternalServices.create(
    openAIService = openAIService,
    grpcClient = grpcClient
  )

  /**
    * Creates ExternalServices with only OpenAI service mocked.
    * Useful when you want to test gRPC functionality but mock OpenAI.
    * 
    * @param openAIService Mock OpenAI service implementation
    * @return ExternalServices instance with mock OpenAI and real gRPC client
    */
  def withMockOpenAI(openAIService: OpenAIService): ExternalServices = ExternalServices.create(
    openAIService = openAIService,
    grpcClient = GrpcClientService.default
  )

  /**
    * Creates ExternalServices with only gRPC client mocked.
    * Useful when you want to test OpenAI functionality but mock gRPC.
    * 
    * @param grpcClient Mock gRPC client service implementation
    * @return ExternalServices instance with real OpenAI and mock gRPC client
    */
  def withMockGrpc(grpcClient: GrpcClientService): ExternalServices = ExternalServices.create(
    openAIService = OpenAIServiceImpl.instance,
    grpcClient = grpcClient
  )
}

/**
  * Mock implementations of GrpcClientService for testing.
  * 
  * These mocks provide different behaviors useful for various testing scenarios.
  */
object GrpcClientServiceMock {

  /**
    * A no-op mock that always succeeds without performing any action.
    * Useful for most tests where gRPC calls are not the focus.
    */
  lazy val noOp: GrpcClientService = NoOpGrpcClientService

  /**
    * Creates a mock that fails with a specified error.
    * Useful for testing error handling scenarios.
    * 
    * @param error The error to throw when the service is called
    * @return Mock service that always fails with the specified error
    */
  def failing(error: Throwable): GrpcClientService = new FailingGrpcClientService(error)

  /**
    * Creates a mock that validates the expected host and port, succeeding only on the first call.
    * Useful for testing exact call expectations.
    * 
    * @param expectedHost Expected hostname
    * @param expectedPort Expected port number
    * @return Mock service that validates parameters and tracks calls
    */
  def singleCall(expectedHost: String, expectedPort: Long): SingleCallGrpcClientService =
    new SingleCallGrpcClientService(expectedHost, expectedPort)

  // Private implementations

  private object NoOpGrpcClientService extends GrpcClientService {
    override def initClientAndTell[F[_]: Sync](
        clientHost: String,
        clientPort: Long,
        payload: String
    ): F[Unit] = Sync[F].unit
  }

  private class FailingGrpcClientService(error: Throwable) extends GrpcClientService {
    override def initClientAndTell[F[_]: Sync](
        clientHost: String,
        clientPort: Long,
        payload: String
    ): F[Unit] = Sync[F].raiseError(error)
  }

  /**
    * Mock that validates parameters and tracks call count.
    * Extends the interface to provide test introspection capabilities.
    */
  class SingleCallGrpcClientService(expectedHost: String, expectedPort: Long) extends GrpcClientService {
    @volatile private var callCount = 0

    private def isFirstCall: Boolean = {
      callCount += 1
      callCount == 1
    }

    override def initClientAndTell[F[_]: Sync](
        clientHost: String,
        clientPort: Long,
        payload: String
    ): F[Unit] = {
      if (isFirstCall && clientHost == expectedHost && clientPort == expectedPort) {
        Sync[F].unit
      } else if (callCount > 1) {
        Sync[F].raiseError(new RuntimeException("GrpcClient should be called only once"))
      } else {
        Sync[F].raiseError(new RuntimeException(s"Unexpected parameters: host=$clientHost, port=$clientPort"))
      }
    }

    /**
      * Test utility method to check if the service was called.
      * @return true if the service was called at least once
      */
    def wasCalled: Boolean = callCount > 0

    /**
      * Test utility method to get the call count.
      * @return number of times the service was called
      */
    def getCallCount: Int = callCount
  }
}

