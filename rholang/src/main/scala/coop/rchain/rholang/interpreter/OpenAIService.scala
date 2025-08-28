package coop.rchain.rholang.interpreter

import akka.actor.ActorSystem
import akka.stream.Materializer
import akka.stream.scaladsl.Sink
import io.cequence.openaiscala.domain.{ModelId, UserMessage}
import io.cequence.openaiscala.domain.settings.{
  CreateChatCompletionSettings,
  CreateCompletionSettings,
  CreateImageSettings,
  CreateSpeechSettings,
  VoiceType
}
import io.cequence.openaiscala.service.{OpenAIService => CeqOpenAIService, OpenAIServiceFactory}
import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.Logger
import cats.effect.{Concurrent, Sync}
import cats.syntax.all._

import java.util.Locale
import java.util.concurrent.Executors
import scala.concurrent.{ExecutionContext, Future}
import scala.concurrent.Await
import scala.concurrent.duration._
import scala.util.control.NonFatal

trait OpenAIService {

  /** @return raw mp3 bytes */
  def ttsCreateAudioSpeech[F[_]](prompt: String)(
      implicit F: Concurrent[F]
  ): F[Array[Byte]]

  /** @return ULR to a generated image */
  def dalle3CreateImage[F[_]](prompt: String)(
      implicit F: Concurrent[F]
  ): F[String]

  def gpt4TextCompletion[F[_]](prompt: String)(
      implicit F: Concurrent[F]
  ): F[String]
}

class DisabledOpenAIService extends OpenAIService {

  private[this] val logger: Logger = Logger[this.type]

  def ttsCreateAudioSpeech[F[_]](prompt: String)(implicit F: Concurrent[F]): F[Array[Byte]] = {
    logger.debug("OpenAI service is disabled - ttsCreateAudioSpeech request ignored")
    F.raiseError(new UnsupportedOperationException("OpenAI service is disabled via configuration"))
  }

  def dalle3CreateImage[F[_]](prompt: String)(implicit F: Concurrent[F]): F[String] = {
    logger.debug("OpenAI service is disabled - dalle3CreateImage request ignored")
    F.raiseError(new UnsupportedOperationException("OpenAI service is disabled via configuration"))
  }

  def gpt4TextCompletion[F[_]](prompt: String)(implicit F: Concurrent[F]): F[String] = {
    logger.debug("OpenAI service is disabled - gpt4TextCompletion request ignored")
    F.raiseError(new UnsupportedOperationException("OpenAI service is disabled via configuration"))
  }
}

class OpenAIServiceImpl extends OpenAIService {

  private[this] val logger: Logger = Logger[this.type]

  implicit private val ec: ExecutionContext =
    ExecutionContext.fromExecutor(Executors.newCachedThreadPool())
  private val system                              = ActorSystem()
  implicit private val materializer: Materializer = Materializer(system)

  // Build OpenAI client at startup.
  // The API key is resolved in the following order:
  //   1. From Typesafe configuration path `openai.api-key` if defined (e.g. in `rnode.conf` or `defaults.conf`).
  //   2. Fallback to environment variable `OPENAI_SCALA_CLIENT_API_KEY` (to preserve backward compatibility).
  //   3. Throw an exception if the key is missing (crash the node at startup).
  private val openAIService: CeqOpenAIService = {
    val config = ConfigFactory.load()

    // Read from config if present.
    val apiKeyFromConfig: Option[String] =
      if (config.hasPath("openai.api-key")) Some(config.getString("openai.api-key")) else None

    // Fallback to env variable (legacy behaviour)
    val apiKey: Option[String] =
      apiKeyFromConfig.orElse(Option(System.getenv("OPENAI_SCALA_CLIENT_API_KEY")))

    apiKey match {
      case Some(key) if key.nonEmpty =>
        logger.info("OpenAI service initialized successfully")
        val service = OpenAIServiceFactory(key)

        // Validate API key before first call (configurable)
        validateApiKeyOrFail(service, config)

        service
      case _ =>
        throw new IllegalStateException(
          "OpenAI API key is not configured. Provide it via config path 'openai.api-key' " +
            "or env var OPENAI_SCALA_CLIENT_API_KEY."
        )
    }
  }

  // shutdown system before jvm shutdown
  sys.addShutdownHook {
    system.terminate()
  }

  def ttsCreateAudioSpeech[F[_]](prompt: String)(
      implicit F: Concurrent[F]
  ): F[Array[Byte]] = {
    val future: Future[Array[Byte]] = openAIService
      .createAudioSpeech(
        prompt,
        CreateSpeechSettings(
          model = ModelId.tts_1_1106,
          voice = VoiceType.shimmer
        )
      )
      .flatMap(
        response =>
          response.map(_.toByteBuffer.array()).runWith(Sink.fold(Array.emptyByteArray)(_ ++ _))
      )

    F.async[Array[Byte]] { cb =>
      future.onComplete {
        case scala.util.Success(response) =>
          logger.info("OpenAI createAudioSpeech request succeeded")
          cb(Right(response))
        case scala.util.Failure(e) =>
          logger.warn("OpenAI createAudioSpeech request failed", e)
          cb(Left(e))
      }
    }
  }

  /** Performs a lightweight call to validate that the API key works. Fails fast on errors. */
  private def validateApiKeyOrFail(
      service: CeqOpenAIService,
      config: com.typesafe.config.Config
  ): Unit = {
    val doValidate: Boolean =
      if (config.hasPath("openai.validate-api-key")) config.getBoolean("openai.validate-api-key")
      else true

    if (!doValidate) {
      logger.info("OpenAI API key validation is disabled by config 'openai.validate-api-key=false'")
      return
    }

    val timeoutSec: Int =
      if (config.hasPath("openai.validation-timeout-sec"))
        config.getInt("openai.validation-timeout-sec")
      else 15

    try {
      // Listing models is a free endpoint and suitable for key validation
      val models = Await.result(service.listModels, timeoutSec.seconds)
      logger.info(s"OpenAI API key validated (${models.size} models available)")
    } catch {
      case NonFatal(e) =>
        throw new IllegalStateException(
          "OpenAI API key validation failed. Check 'openai.api-key' or 'OPENAI_SCALA_CLIENT_API_KEY'.",
          e
        )
    }
  }

  def dalle3CreateImage[F[_]](prompt: String)(
      implicit F: Concurrent[F]
  ): F[String] = {
    val future: Future[String] = openAIService
      .createImage(
        prompt,
        CreateImageSettings(
          model = Some(ModelId.dall_e_3),
          n = Some(1)
        )
      )
      .map { response =>
        // TODO: handle error properly
        response.data.headOption.flatMap(_.get("url")).get
      }

    F.async[String] { cb =>
      future.onComplete {
        case scala.util.Success(response) =>
          logger.info("OpenAI createImage request succeeded")
          cb(Right(response))
        case scala.util.Failure(e) =>
          logger.warn("OpenAI createImage request failed", e)
          cb(Left(e))
      }
    }
  }

  def gpt4TextCompletion[F[_]](prompt: String)(
      implicit F: Concurrent[F]
  ): F[String] = {
    val future: Future[String] = openAIService
      .createChatCompletion(
        Seq(UserMessage(prompt)),
        CreateChatCompletionSettings(
          model = ModelId.gpt_4_1_mini,
          top_p = Some(0.5),
          temperature = Some(0.5)
        )
      )
      .map(response => response.choices.head.message.content)

    F.async[String] { cb =>
      future.onComplete {
        case scala.util.Success(response) =>
          logger.info("OpenAI gpt4 request succeeded")
          cb(Right(response))
        case scala.util.Failure(e) =>
          logger.warn("OpenAI gpt4 request failed", e)
          cb(Left(e))
      }
    }
  }
}

object OpenAIServiceImpl {

  private[this] val logger: Logger = Logger[this.type]

  /**
    * Provides the appropriate OpenAI service based on configuration and environment.
    * Priority order for enabling OpenAI:
    *   1. Environment variable: OPENAI_ENABLED = true/false
    *   2. Configuration: openai.enabled = true/false
    *   3. Default: false (disabled for safety)
    * - If enabled, returns OpenAIServiceImpl (will crash at startup if no API key)
    * - If disabled, returns DisabledOpenAIService
    */
  lazy val instance: OpenAIService = {
    val isEnabled = RhoRuntime.isOpenAIEnabled

    if (isEnabled) {
      logger.info("OpenAI service is enabled - initializing with API key validation")
      new OpenAIServiceImpl // This will crash at startup if no API key is provided
    } else {
      logger.info("OpenAI service is disabled")
      new DisabledOpenAIService
    }
  }

  /** @deprecated Use `instance` instead. Kept for backward compatibility. */
  lazy val realOpenAIService: OpenAIService = instance
}
