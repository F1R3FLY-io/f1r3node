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
