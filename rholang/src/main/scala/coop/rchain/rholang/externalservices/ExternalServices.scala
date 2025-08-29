package coop.rchain.rholang.externalservices

trait ExternalServices {
  def openAIService: OpenAIService
  def grpcClient: GrpcClientService
}

case object RealExternalServices extends ExternalServices {
  override lazy val openAIService: OpenAIService  = OpenAIServiceImpl.instance
  override lazy val grpcClient: GrpcClientService = GrpcClientService.instance
}

case object NoOpExternalServices extends ExternalServices {
  override lazy val openAIService: OpenAIService  = OpenAIServiceImpl.noOpInstance
  override lazy val grpcClient: GrpcClientService = GrpcClientService.noOpInstance
}
