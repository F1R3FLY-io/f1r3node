package coop.rchain.rholang.externalservices

trait ExternalServices {
  def openAIService: OpenAIService
  def grpcClient: GrpcClientService
  def ollamaService: OllamaService
}

case object RealExternalServices extends ExternalServices {
  override val openAIService: OpenAIService  = OpenAIServiceImpl.instance
  override val grpcClient: GrpcClientService = GrpcClientService.instance
  override val ollamaService: OllamaService  = OllamaServiceImpl.instance
}

case object NoOpExternalServices extends ExternalServices {
  override val openAIService: OpenAIService  = OpenAIServiceImpl.noOpInstance
  override val grpcClient: GrpcClientService = GrpcClientService.noOpInstance
  override val ollamaService: OllamaService  = new DisabledOllamaService
}

case object ObserverExternalServices extends ExternalServices {
  override val openAIService: OpenAIService  = OpenAIServiceImpl.noOpInstance
  override val grpcClient: GrpcClientService = GrpcClientService.noOpInstance
  override val ollamaService: OllamaService  = new DisabledOllamaService
}

object ExternalServices {

  /**
    * Selects the appropriate ExternalServices based on whether the node is a validator or observer.
    * Validators get full functionality (real grpc client and AI services), while observers get NoOp
    * grpc client and disabled AI services.
    *
    * @param isValidator true if the node has validator keys configured, false for observer nodes
    * @return appropriate ExternalServices implementation
    */
  def forNodeType(isValidator: Boolean): ExternalServices =
    if (isValidator) RealExternalServices else ObserverExternalServices
}
