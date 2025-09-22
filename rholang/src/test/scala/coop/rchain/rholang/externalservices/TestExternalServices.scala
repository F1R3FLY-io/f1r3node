package coop.rchain.rholang.externalservices

import coop.rchain.rholang.externalservices.{
  ExternalServices,
  GrpcClientService,
  OllamaService,
  OpenAIService
}

case class TestExternalServices(
    openAIService: OpenAIService,
    grpcClient: GrpcClientService,
    ollamaService: OllamaService
) extends ExternalServices
