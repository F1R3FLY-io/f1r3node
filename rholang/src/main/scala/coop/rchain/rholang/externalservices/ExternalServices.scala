package coop.rchain.rholang.externalservices

trait ExternalServices {
  def openAIService: OpenAIService
  def grpcClient: GrpcClientService
  def ollamaService: OllamaService
  def nunetService: NunetService
}

/** Factory for creating ExternalServices with configuration */
object ExternalServices {

  /**
    * Create ExternalServices based on node type and configuration.
    *
    * @param isValidator true if node has validator keys, false for observer
    * @param openai OpenAI configuration (optional)
    * @param ollama Ollama configuration (optional)
    * @param nunet Nunet configuration (optional)
    */
  def apply(
      isValidator: Boolean,
      openai: Option[OpenAIConf],
      ollama: Option[OllamaConf],
      nunet: Option[NunetConf]
  ): ExternalServices =
    if (isValidator)
      new RealExternalServices(openai, ollama, nunet)
    else
      new ObserverExternalServices()

  /**
    * Selects the appropriate ExternalServices based on whether the node is a validator or observer.
    * Validators get full functionality (real grpc client and AI services), while observers get NoOp
    * grpc client and disabled AI services.
    *
    * @deprecated Use apply(isValidator, openai, ollama, nunet) instead for proper configuration injection
    * @param isValidator true if the node has validator keys configured, false for observer nodes
    * @return appropriate ExternalServices implementation
    */
  @deprecated("Use apply(isValidator, openai, ollama, nunet) instead", "1.0.0")
  def forNodeType(isValidator: Boolean): ExternalServices =
    if (isValidator) RealExternalServices else ObserverExternalServices
}

/** External services for validator nodes - all services enabled based on config */
class RealExternalServices(
    openai: Option[OpenAIConf],
    ollama: Option[OllamaConf],
    nunet: Option[NunetConf]
) extends ExternalServices {
  import scala.concurrent.ExecutionContext.Implicits.global

  // Debug logging
  private val logger = com.typesafe.scalalogging.Logger[this.type]
  logger.info(s"RealExternalServices initializing with nunet config: $nunet")

  override val openAIService: OpenAIService = {
    openai match {
      case Some(conf) if conf.enabled => new OpenAIServiceImpl(conf)
      case _                          => OpenAIServiceImpl.noOpInstance
    }
  }

  override val grpcClient: GrpcClientService = GrpcClientService.instance

  override val ollamaService: OllamaService = {
    logger.info(s"Creating ollama service with config: $ollama")
    ollama match {
      case Some(conf) if conf.enabled => new OllamaServiceImpl(conf)
      case _                          => new DisabledOllamaService
    }
  }

  override val nunetService: NunetService = {
    logger.info(s"Creating nunet service with config: $nunet")
    nunet match {
      case Some(conf) if conf.enabled =>
        logger.info(s"Nunet is enabled with config: $conf")
        new NunetServiceImpl(conf)
      case Some(conf) =>
        logger.warn(s"Nunet config found but not enabled: $conf")
        new DisabledNunetService
      case None =>
        logger.warn("No nunet configuration provided")
        new DisabledNunetService
    }
  }
}

/** External services for observer nodes - all services disabled */
class ObserverExternalServices() extends ExternalServices {
  override val openAIService: OpenAIService  = OpenAIServiceImpl.noOpInstance
  override val grpcClient: GrpcClientService = GrpcClientService.noOpInstance
  override val ollamaService: OllamaService  = new DisabledOllamaService
  override val nunetService: NunetService    = new DisabledNunetService
}

// Keep old singletons for backwards compatibility during transition
@deprecated("Use new RealExternalServices(openai, ollama, nunet) instead", "1.0.0")
case object RealExternalServices extends ExternalServices {
  override val openAIService: OpenAIService  = OpenAIServiceImpl.instance
  override val grpcClient: GrpcClientService = GrpcClientService.instance
  override val ollamaService: OllamaService  = OllamaServiceImpl.instance
  override val nunetService: NunetService    = NunetServiceImpl.instance
}

@deprecated("Use new ObserverExternalServices() instead", "1.0.0")
case object NoOpExternalServices extends ExternalServices {
  override val openAIService: OpenAIService  = OpenAIServiceImpl.noOpInstance
  override val grpcClient: GrpcClientService = GrpcClientService.noOpInstance
  override val ollamaService: OllamaService  = new DisabledOllamaService
  override val nunetService: NunetService    = new DisabledNunetService
}

@deprecated("Use new ObserverExternalServices() instead", "1.0.0")
case object ObserverExternalServices extends ExternalServices {
  override val openAIService: OpenAIService  = OpenAIServiceImpl.noOpInstance
  override val grpcClient: GrpcClientService = GrpcClientService.noOpInstance
  override val ollamaService: OllamaService  = new DisabledOllamaService
  override val nunetService: NunetService    = new DisabledNunetService
}
