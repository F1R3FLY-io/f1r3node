package coop.rchain.rholang.externalservices

import coop.rchain.rholang.externalservices.{
  ExternalServices,
  GrpcClientService,
  NunetService,
  OllamaService,
  OpenAIService
}

case class TestExternalServices(
    openAIService: OpenAIService,
    grpcClient: GrpcClientService,
    ollamaService: OllamaService,
    nunetService: NunetService
) extends ExternalServices
