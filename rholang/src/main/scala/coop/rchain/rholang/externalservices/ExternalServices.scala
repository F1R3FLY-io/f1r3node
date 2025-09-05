package coop.rchain.rholang.externalservices

trait ExternalServices {
  def openAIService: OpenAIService
  def grpcClient: GrpcClientService
}

case object RealExternalServices extends ExternalServices {
  override val openAIService: OpenAIService  = OpenAIServiceImpl.instance
  override val grpcClient: GrpcClientService = GrpcClientService.instance
}

case object NoOpExternalServices extends ExternalServices {
  override val openAIService: OpenAIService  = OpenAIServiceImpl.noOpInstance
  override val grpcClient: GrpcClientService = GrpcClientService.noOpInstance
}

case object ObserverExternalServices extends ExternalServices {
  override val openAIService: OpenAIService  = OpenAIServiceImpl.noOpInstance
  override val grpcClient: GrpcClientService = GrpcClientService.noOpInstance
}

object ExternalServices {

  /**
    * Selects the appropriate ExternalServices based on whether the node is a validator or observer.
    * Validators get full functionality (real grpc client), while observers get NoOp grpc client
    * but still have access to OpenAI services.
    *
    * @param isValidator true if the node has validator keys configured, false for observer nodes
    * @return appropriate ExternalServices implementation
    */
  def forNodeType(isValidator: Boolean): ExternalServices =
    if (isValidator) RealExternalServices else ObserverExternalServices
}
