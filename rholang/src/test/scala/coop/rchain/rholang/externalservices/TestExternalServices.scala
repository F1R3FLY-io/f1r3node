package coop.rchain.rholang.externalservices

import coop.rchain.rholang.externalservices.{ExternalServices, GrpcClientService, OpenAIService}

case class TestExternalServices(openAIService: OpenAIService, grpcClient: GrpcClientService)
    extends ExternalServices
