package coop.rchain.rholang.interpreter

import cats.effect.Sync

/**
  * Dependency injection container for external service integrations.
  * 
  * This class encapsulates all external service dependencies used by the RhoLang
  * system processes, providing a clean separation of concerns and enabling easier
  * testing through dependency injection.
  *
  * @param openAIService Service for OpenAI API interactions (GPT-4, DALL-E 3, TTS)
  * @param grpcClient Service for gRPC client communications
  */
final case class ExternalServices(
    openAIService: OpenAIService,
    grpcClient: GrpcClientService
)

object ExternalServices {

  /**
    * Creates ExternalServices with production implementations.
    * 
    * This is the main factory method for creating the external services
    * container in production environments. It uses the actual OpenAI service
    * and gRPC client implementations.
    * 
    * @return ExternalServices instance with production implementations
    */
  def apply(): ExternalServices = ExternalServices(
    openAIService = OpenAIServiceImpl.instance,
    grpcClient = GrpcClientService.default
  )

  /**
    * Creates ExternalServices with custom implementations.
    * 
    * This method allows for dependency injection of custom service implementations,
    * useful for configuration-based initialization or advanced use cases.
    * 
    * @param openAIService Custom OpenAI service implementation
    * @param grpcClient Custom gRPC client service implementation  
    * @return ExternalServices instance with custom implementations
    */
  def create(
      openAIService: OpenAIService,
      grpcClient: GrpcClientService
  ): ExternalServices = ExternalServices(
    openAIService = openAIService,
    grpcClient = grpcClient
  )
}

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
  def initClientAndTell[F[_]: Sync](
      clientHost: String,
      clientPort: Long,
      payload: String
  ): F[Unit]
}

object GrpcClientService {

  /**
    * Default production implementation of GrpcClientService.
    * 
    * This implementation uses the existing GrpcClient to handle actual
    * network communication. It's a singleton to avoid resource leaks.
    */
  lazy val default: GrpcClientService = DefaultGrpcClientService
  
  /**
    * Concrete implementation that delegates to GrpcClient.
    */
  private object DefaultGrpcClientService extends GrpcClientService {
    override def initClientAndTell[F[_]: Sync](
        clientHost: String,
        clientPort: Long,
        payload: String
    ): F[Unit] = GrpcClient.initClientAndTell[F](clientHost, clientPort, payload)
  }
}
