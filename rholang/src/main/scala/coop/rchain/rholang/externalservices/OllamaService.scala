package coop.rchain.rholang.externalservices

import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.model._
import akka.http.scaladsl.unmarshalling.Unmarshal
import akka.stream.Materializer
import cats.effect.{Concurrent, Sync}
import cats.syntax.all._
import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.Logger
import spray.json._
import spray.json.DefaultJsonProtocol._

import java.util.concurrent.Executors
import scala.concurrent.{ExecutionContext, Future}
import scala.concurrent.duration._
import scala.util.control.NonFatal

// JSON protocol for Ollama API
object OllamaJsonProtocol extends DefaultJsonProtocol {
  case class OllamaMessage(role: String, content: String)
  case class OllamaChatRequest(
      model: String,
      messages: List[OllamaMessage],
      stream: Boolean = false
  )
  case class OllamaGenerateRequest(model: String, prompt: String, stream: Boolean = false)
  case class OllamaChatResponse(message: OllamaMessage, done: Boolean)
  case class OllamaGenerateResponse(response: String, done: Boolean)
  case class OllamaModel(name: String, modified_at: String, size: Long)
  case class OllamaModelsResponse(models: List[OllamaModel])

  implicit val messageFormat          = jsonFormat2(OllamaMessage)
  implicit val chatRequestFormat      = jsonFormat3(OllamaChatRequest)
  implicit val generateRequestFormat  = jsonFormat3(OllamaGenerateRequest)
  implicit val chatResponseFormat     = jsonFormat2(OllamaChatResponse)
  implicit val generateResponseFormat = jsonFormat2(OllamaGenerateResponse)
  implicit val modelFormat            = jsonFormat3(OllamaModel)
  implicit val modelsResponseFormat   = jsonFormat1(OllamaModelsResponse)
}

trait OllamaService {

  /** Chat completion using Ollama's chat endpoint */
  def chatCompletion[F[_]](model: String, prompt: String)(
      implicit F: Concurrent[F]
  ): F[String]

  /** Text generation using Ollama's generate endpoint */
  def textGeneration[F[_]](model: String, prompt: String)(
      implicit F: Concurrent[F]
  ): F[String]

  /** List available models from Ollama */
  def listModels[F[_]]()(
      implicit F: Concurrent[F]
  ): F[List[String]]
}

class DisabledOllamaService extends OllamaService {

  private[this] val logger: Logger = Logger[this.type]

  def chatCompletion[F[_]](model: String, prompt: String)(implicit F: Concurrent[F]): F[String] = {
    logger.debug("Ollama service is disabled - chatCompletion request ignored")
    F.raiseError(new UnsupportedOperationException("Ollama service is disabled via configuration"))
  }

  def textGeneration[F[_]](model: String, prompt: String)(implicit F: Concurrent[F]): F[String] = {
    logger.debug("Ollama service is disabled - textGeneration request ignored")
    F.raiseError(new UnsupportedOperationException("Ollama service is disabled via configuration"))
  }

  def listModels[F[_]]()(implicit F: Concurrent[F]): F[List[String]] = {
    logger.debug("Ollama service is disabled - listModels request ignored")
    F.raiseError(new UnsupportedOperationException("Ollama service is disabled via configuration"))
  }
}

class OllamaServiceImpl extends OllamaService {
  import OllamaJsonProtocol._

  private[this] val logger: Logger = Logger[this.type]

  implicit private val ec: ExecutionContext =
    ExecutionContext.fromExecutor(Executors.newCachedThreadPool())
  private val system                              = ActorSystem()
  implicit private val materializer: Materializer = Materializer(system)

  // Build Ollama client configuration
  private val config = ConfigFactory.load()
  private val baseUrl: String =
    if (config.hasPath("ollama.base-url")) config.getString("ollama.base-url")
    else "http://localhost:11434"

  private val defaultModel: String =
    if (config.hasPath("ollama.default-model")) config.getString("ollama.default-model")
    else "llama4:latest"

  private val timeoutSec: Int =
    if (config.hasPath("ollama.timeout-sec")) config.getInt("ollama.timeout-sec")
    else 30

  // Validate connection on startup
  validateConnectionOrFail()

  // shutdown system before jvm shutdown
  sys.addShutdownHook {
    system.terminate()
  }

  private def validateConnectionOrFail(): Unit = {
    val doValidate: Boolean =
      if (config.hasPath("ollama.validate-connection"))
        config.getBoolean("ollama.validate-connection")
      else true

    if (!doValidate) {
      logger.info(
        "Ollama connection validation is disabled by config 'ollama.validate-connection=false'"
      )
      return
    }

    try {
      // Test connection by trying to list models
      val future = Http()(system).singleRequest(
        HttpRequest(
          method = HttpMethods.GET,
          uri = s"$baseUrl/api/tags"
        )
      )

      val response = scala.concurrent.Await.result(future, timeoutSec.seconds)
      if (response.status.isSuccess()) {
        logger.info(s"Ollama service connection validated successfully at $baseUrl")
      } else {
        throw new RuntimeException(s"Ollama service responded with status: ${response.status}")
      }
    } catch {
      case NonFatal(e) =>
        throw new IllegalStateException(
          s"Ollama service connection validation failed. Check that Ollama is running on $baseUrl",
          e
        )
    }
  }

  def chatCompletion[F[_]](model: String, prompt: String)(
      implicit F: Concurrent[F]
  ): F[String] = {
    val actualModel   = if (model.isEmpty) defaultModel else model
    val promptPreview = if (prompt.length <= 250) prompt else prompt.take(250) + "..."
    logger.info(
      s"Ollama chatCompletion - input model: '$model', actualModel: '$actualModel', defaultModel: '$defaultModel', prompt: '$promptPreview'"
    )
    val requestBody = OllamaChatRequest(
      model = actualModel,
      messages = List(OllamaMessage("user", prompt)),
      stream = false
    ).toJson.toString()

    val future: Future[String] = for {
      response <- Http()(system).singleRequest(
                   HttpRequest(
                     method = HttpMethods.POST,
                     uri = s"$baseUrl/api/chat",
                     entity = HttpEntity(ContentTypes.`application/json`, requestBody)
                   )
                 )
      responseBody <- Unmarshal(response.entity).to[String]
      chatResponse = responseBody.parseJson.convertTo[OllamaChatResponse]
    } yield chatResponse.message.content

    F.async[String] { cb =>
      future.onComplete {
        case scala.util.Success(response) =>
          logger.info(s"Ollama chat completion succeeded for model: $actualModel")
          cb(Right(response))
        case scala.util.Failure(e) =>
          logger.warn(s"Ollama chat completion failed for model: $actualModel", e)
          cb(Left(e))
      }
    }
  }

  def textGeneration[F[_]](model: String, prompt: String)(
      implicit F: Concurrent[F]
  ): F[String] = {
    val actualModel   = if (model.isEmpty) defaultModel else model
    val promptPreview = if (prompt.length <= 250) prompt else prompt.take(250) + "..."
    logger.info(
      s"Ollama textGeneration - input model: '$model', actualModel: '$actualModel', defaultModel: '$defaultModel', prompt: '$promptPreview'"
    )
    val requestBody = OllamaGenerateRequest(
      model = actualModel,
      prompt = prompt,
      stream = false
    ).toJson.toString()

    val future: Future[String] = for {
      response <- Http()(system).singleRequest(
                   HttpRequest(
                     method = HttpMethods.POST,
                     uri = s"$baseUrl/api/generate",
                     entity = HttpEntity(ContentTypes.`application/json`, requestBody)
                   )
                 )
      responseBody     <- Unmarshal(response.entity).to[String]
      generateResponse = responseBody.parseJson.convertTo[OllamaGenerateResponse]
    } yield generateResponse.response

    F.async[String] { cb =>
      future.onComplete {
        case scala.util.Success(response) =>
          logger.info(s"Ollama text generation succeeded for model: $actualModel")
          cb(Right(response))
        case scala.util.Failure(e) =>
          logger.warn(s"Ollama text generation failed for model: $actualModel", e)
          cb(Left(e))
      }
    }
  }

  def listModels[F[_]]()(
      implicit F: Concurrent[F]
  ): F[List[String]] = {
    val future: Future[List[String]] = for {
      response <- Http()(system).singleRequest(
                   HttpRequest(
                     method = HttpMethods.GET,
                     uri = s"$baseUrl/api/tags"
                   )
                 )
      responseBody   <- Unmarshal(response.entity).to[String]
      modelsResponse = responseBody.parseJson.convertTo[OllamaModelsResponse]
    } yield modelsResponse.models.map(_.name)

    F.async[List[String]] { cb =>
      future.onComplete {
        case scala.util.Success(models) =>
          logger.info(s"Ollama listModels succeeded (${models.size} models available)")
          cb(Right(models))
        case scala.util.Failure(e) =>
          logger.warn("Ollama listModels failed", e)
          cb(Left(e))
      }
    }
  }
}

object OllamaServiceImpl {

  private[this] val logger: Logger = Logger[this.type]

  /**
    * Provides the appropriate Ollama service based on configuration and environment.
    * Priority order for enabling Ollama:
    *   1. Environment variable: OLLAMA_ENABLED = true/false
    *   2. Configuration: ollama.enabled = true/false
    *   3. Default: false (disabled for safety)
    * - If enabled, returns OllamaServiceImpl (will crash at startup if connection fails)
    * - If disabled, returns DisabledOllamaService
    */
  lazy val instance: OllamaService = {
    val isEnabled = isOllamaEnabled

    if (isEnabled) {
      logger.info("Ollama service is enabled - initializing with connection validation")
      new OllamaServiceImpl // This will crash at startup if connection fails
    } else {
      logger.info("Ollama service is disabled")
      new DisabledOllamaService
    }
  }
}
