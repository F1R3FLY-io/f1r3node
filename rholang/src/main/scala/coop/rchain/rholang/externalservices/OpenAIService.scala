package coop.rchain.rholang.externalservices

import akka.actor.ActorSystem
import akka.stream.Materializer
import akka.stream.scaladsl.Sink
import cats.effect.Concurrent
import cats.implicits._
import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.Logger
import coop.rchain.rholang.interpreter.RhoRuntime
import coop.rchain.shared.{Log, LogSource}
import io.cequence.openaiscala.domain.settings.{
  CreateChatCompletionSettings,
  CreateImageSettings,
  CreateSpeechSettings,
  VoiceType
}
import io.cequence.openaiscala.domain.{ModelId, UserMessage}
import io.cequence.openaiscala.service.{OpenAIServiceFactory, OpenAIService => CeqOpenAIService}

import java.util.concurrent.{ExecutorService, Executors, ThreadFactory, TimeUnit}
import scala.concurrent.duration._
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.util.control.NonFatal

trait OpenAIService {

  /** @return raw mp3 bytes */
  def ttsCreateAudioSpeech[F[_]](prompt: String)(
      implicit F: Concurrent[F],
      L: Log[F]
  ): F[Array[Byte]]

  /** @return ULR to a generated image */
  def dalle3CreateImage[F[_]](prompt: String)(
      implicit F: Concurrent[F],
      L: Log[F]
  ): F[String]

  def gpt4TextCompletion[F[_]](prompt: String)(
      implicit F: Concurrent[F],
      L: Log[F]
  ): F[String]
}

class NoOpOpenAIService extends OpenAIService {

  implicit private val logSource: LogSource = LogSource(this.getClass)

  def ttsCreateAudioSpeech[F[_]](
      prompt: String
  )(implicit F: Concurrent[F], L: Log[F]): F[Array[Byte]] =
    for {
      _      <- L.debug("OpenAI service is disabled - ttsCreateAudioSpeech request ignored")
      result <- F.pure(Array.emptyByteArray)
    } yield result

  def dalle3CreateImage[F[_]](prompt: String)(implicit F: Concurrent[F], L: Log[F]): F[String] =
    for {
      _      <- L.debug("OpenAI service is disabled - dalle3CreateImage request ignored")
      result <- F.pure("")
    } yield result

  def gpt4TextCompletion[F[_]](prompt: String)(implicit F: Concurrent[F], L: Log[F]): F[String] =
    for {
      _      <- L.debug("OpenAI service is disabled - gpt4TextCompletion request ignored")
      result <- F.pure("")
    } yield result
}

class OpenAIServiceImpl extends OpenAIService {

  // Keep direct logger for initialization only
  private[this] val initLogger: Logger      = Logger[this.type]
  implicit private val logSource: LogSource = LogSource(this.getClass)

  // Create ExecutorService with daemon threads to prevent JVM hanging
  private val executorService: ExecutorService = Executors.newCachedThreadPool(new ThreadFactory {
    private val defaultFactory = Executors.defaultThreadFactory()
    def newThread(r: Runnable): Thread = {
      val t = defaultFactory.newThread(r)
      t.setDaemon(true) // Daemon threads won't prevent JVM shutdown
      t.setName(s"openai-service-${t.getName}")
      t
    }
  })
  implicit private val ec: ExecutionContext       = ExecutionContext.fromExecutor(executorService)
  private val system                              = ActorSystem()
  implicit private val materializer: Materializer = Materializer(system)

  // Build OpenAI client at startup.
  // The API key is resolved in the following order:
  //   1. From environment variable `OPENAI_SCALA_CLIENT_API_KEY`.
  //   2. From Typesafe configuration path `openai.api-key` if defined (e.g. in `rnode.conf` or `defaults.conf`).
  //   3. Throw an exception if the key is missing (crash the node at startup).
  private val openAIService: CeqOpenAIService = {
    val config = ConfigFactory.load()

    // Fallback to env variable
    val apiKeyFromEnv: Option[String] =
      Option(System.getenv("OPENAI_SCALA_CLIENT_API_KEY"))

    // Read from config if present.
    val apiKeyFromConfig: Option[String] =
      if (config.hasPath("openai.api-key")) Some(config.getString("openai.api-key")) else None

    (apiKeyFromEnv orElse apiKeyFromConfig) match {
      case Some(key) if key.nonEmpty =>
        initLogger.info("OpenAI service initialized successfully")
        val service: CeqOpenAIService = OpenAIServiceFactory(key)

        // Validate API key before first call (configurable)
        try {
          validateApiKeyOrFail(service, config)
        } catch {
          case e: Exception =>
            initLogger.info("Cleaning up resources due to validation failure")
            try {
              service.close()
            } catch {
              case NonFatal(closeError) =>
                initLogger.warn(s"Failed to close OpenAI service: ${closeError.getMessage}")
            }
            shutdownResources()
            throw e
        }

        service
      case _ =>
        val errorMessage = "OpenAI API key is not configured. Provide it via config path 'openai.api-key' " +
          "or env var OPENAI_SCALA_CLIENT_API_KEY."
        initLogger.error(errorMessage)
        shutdownResources()
        throw new RuntimeException(errorMessage)
    }
  }

  /** Clean shutdown of all resources */
  private def shutdownResources(): Unit =
    try {
      // Try graceful shutdown of ActorSystem with timeout
      val terminationFuture = system.terminate()
      Await.ready(terminationFuture, 10.seconds)
      initLogger.info("ActorSystem terminated successfully")
    } catch {
      case NonFatal(e) =>
        initLogger.warn(s"Failed to gracefully terminate ActorSystem: ${e.getMessage}")
    } finally {
      // Always shutdown ExecutorService
      try {
        executorService.shutdown()
        if (!executorService.awaitTermination(5, TimeUnit.SECONDS)) {
          executorService.shutdownNow()
          initLogger.warn("ExecutorService forced shutdown")
        } else {
          initLogger.info("ExecutorService terminated successfully")
        }
      } catch {
        case NonFatal(e) =>
          initLogger.warn(s"Failed to shutdown ExecutorService: ${e.getMessage}")
          executorService.shutdownNow()
      }
    }

  // shutdown system before jvm shutdown
  sys.addShutdownHook {
    shutdownResources()
  }

  def ttsCreateAudioSpeech[F[_]](prompt: String)(
      implicit F: Concurrent[F],
      L: Log[F]
  ): F[Array[Byte]] =
    for {
      _ <- L.info(s"Starting OpenAI text-to-speech request")
      result <- {
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
                cb(Right(response))
              case scala.util.Failure(e) =>
                cb(Left(e))
            }
          }
          .recoverWith {
            case error =>
              L.error(s"OpenAI text-to-speech request failed: ${error.getMessage}", error) >>
                F.raiseError(error)
          }
          .flatMap { result =>
            L.info("OpenAI createAudioSpeech request succeeded") >>
              F.pure(result)
          }
      }
    } yield result

  /** Performs a lightweight call to validate that the API key works. Fails fast on errors. */
  private def validateApiKeyOrFail(
      service: CeqOpenAIService,
      config: com.typesafe.config.Config
  ): Unit = {
    val doValidate: Boolean =
      if (config.hasPath("openai.validate-api-key")) config.getBoolean("openai.validate-api-key")
      else true

    if (!doValidate) {
      initLogger.info(
        "OpenAI API key validation is disabled by config 'openai.validate-api-key=false'"
      )
    } else {
      val timeoutSec: Int =
        if (config.hasPath("openai.validation-timeout-sec"))
          config.getInt("openai.validation-timeout-sec")
        else 15

      try {
        // Listing models is a free endpoint and suitable for key validation
        val models = Await.result(service.listModels, timeoutSec.seconds)
        initLogger.info(s"OpenAI API key validated (${models.size} models available)")
      } catch {
        case NonFatal(e) =>
          val errorMessage =
            "OpenAI API key validation failed. Check 'openai.api-key' or 'OPENAI_SCALA_CLIENT_API_KEY'."
          initLogger.error(errorMessage, e)
          throw e
      }
    }
  }

  def dalle3CreateImage[F[_]](prompt: String)(
      implicit F: Concurrent[F],
      L: Log[F]
  ): F[String] =
    for {
      _ <- L.info(s"Starting OpenAI DALL-E 3 image generation request")
      result <- {
        val future: Future[String] = openAIService
          .createImage(
            prompt,
            CreateImageSettings(
              model = Some(ModelId.dall_e_3),
              n = Some(1)
            )
          )
          .map { response =>
            response.data.headOption.flatMap(_.get("url")).get
          }

        F.async[String] { cb =>
            future.onComplete {
              case scala.util.Success(response) =>
                cb(Right(response))
              case scala.util.Failure(e) =>
                cb(Left(e))
            }
          }
          .recoverWith {
            case error =>
              L.warn(s"OpenAI DALL-E 3 image generation request failed: ${error.getMessage}") >>
                F.raiseError(error)
          }
          .flatMap { result =>
            L.info("OpenAI createImage request succeeded") >>
              F.pure(result)
          }
      }
    } yield result

  def gpt4TextCompletion[F[_]](prompt: String)(
      implicit F: Concurrent[F],
      L: Log[F]
  ): F[String] =
    for {
      _ <- L.info(s"Starting OpenAI GPT-4 text completion request")
      result <- {
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
                cb(Right(response))
              case scala.util.Failure(e) =>
                cb(Left(e))
            }
          }
          .recoverWith {
            case error =>
              L.warn(s"OpenAI GPT-4 text completion request failed: ${error.getMessage}") >>
                F.raiseError(error)
          }
          .flatMap { result =>
            L.info("OpenAI gpt4 request succeeded") >>
              F.pure {
                println(s"OpenAI GPT-4 text completion request succeeded: $result")

                result
              }
          }
      }
    } yield result
}

object OpenAIServiceImpl {

  lazy val noOpInstance: OpenAIService = new NoOpOpenAIService

  /**
    * Provides the appropriate OpenAI service based on configuration and environment.
    * Priority order for enabling OpenAI:
    *   1. Environment variable: OPENAI_ENABLED = true/false
    *   2. Configuration: openai.enabled = true/false
    *   3. Default: false (disabled for safety)
    * - If enabled, returns OpenAIServiceImpl (will crash at startup if no API key)
    * - If disabled, returns NoOpOpenAIService
    */
  lazy val instance: OpenAIService = {
    val isEnabled = coop.rchain.rholang.externalservices.isOpenAIEnabled

    if (isEnabled) {
      new OpenAIServiceImpl // This will crash at startup if no API key is provided
    } else {
      noOpInstance
    }
  }
}
