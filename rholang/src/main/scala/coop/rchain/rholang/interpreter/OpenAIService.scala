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

import java.util.concurrent.Executors
import scala.concurrent.{ExecutionContext, Future}

trait OpenAIService {

  /** @return raw mp3 bytes */
  def ttsCreateAudioSpeech[F[_]](prompt: String)(
      implicit F: Concurrent[F]
  ): F[Array[Byte]]

  /** @return ULR to a generated image */
  def dalle3CreateImage[F[_]](prompt: String)(
      implicit F: Concurrent[F]
  ): F[String]

  def gpt3TextCompletion[F[_]](prompt: String)(
      implicit F: Concurrent[F]
  ): F[String]

  def gpt4TextCompletion[F[_]](prompt: String)(
      implicit F: Concurrent[F]
  ): F[String]
}

class OpenAIServiceImpl extends OpenAIService {

  private[this] val logger: Logger = Logger[this.type]

  implicit private val ec: ExecutionContext =
    ExecutionContext.fromExecutor(Executors.newCachedThreadPool())
  private val system                              = ActorSystem()
  implicit private val materializer: Materializer = Materializer(system)

  // Initialize OpenAI client.
  // The API key is resolved in the following order:
  //   1. From Typesafe configuration path `openai.api-key` if defined (e.g. in `rnode.conf` or `defaults.conf`).
  //   2. Fallback to environment variable `OPENAI_SCALA_CLIENT_API_KEY` (to preserve backward compatibility).
  //   3. Throw an exception if the key is missing.
  private def buildService[F[_]: Sync]: F[CeqOpenAIService] =
    Sync[F].delay {
      val config = ConfigFactory.load()

      // Read from config if present.
      val apiKeyFromConfig: Option[String] =
        if (config.hasPath("openai.api-key")) Some(config.getString("openai.api-key")) else None

      // Fallback to env variable (legacy behaviour)
      val apiKey: Option[String] =
        apiKeyFromConfig.orElse(Option(System.getenv("OPENAI_SCALA_CLIENT_API_KEY")))

      apiKey match {
        case Some(key) if key.nonEmpty =>
          OpenAIServiceFactory(key)
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
    val futureF: F[Future[Array[Byte]]] = buildService[F].map { svc: CeqOpenAIService =>
      svc
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
    }

    futureF.flatMap { f =>
      F.async[Array[Byte]] { cb =>
        f.onComplete {
          case scala.util.Success(response) =>
            logger.info("OpenAI createAudioSpeech request succeeded")
            cb(Right(response))
          case scala.util.Failure(e) =>
            logger.warn("OpenAI createAudioSpeech request failed", e)
            cb(Left(e))
        }
      }
    }
  }

  def dalle3CreateImage[F[_]](prompt: String)(
      implicit F: Concurrent[F]
  ): F[String] = {
    val futureF: F[Future[String]] = buildService[F].map { svc: CeqOpenAIService =>
      svc
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
    }

    futureF.flatMap { f =>
      F.async[String] { cb =>
        f.onComplete {
          case scala.util.Success(response) =>
            logger.info("OpenAI createImage request succeeded")
            cb(Right(response))
          case scala.util.Failure(e) =>
            logger.warn("OpenAI createImage request failed", e)
            cb(Left(e))
        }
      }
    }
  }

  def gpt3TextCompletion[F[_]](prompt: String)(
      implicit F: Concurrent[F]
  ): F[String] = {
    val futureF: F[Future[String]] = buildService[F].map { svc: CeqOpenAIService =>
      svc
        .createCompletion(
          prompt,
          CreateCompletionSettings(
            model = ModelId.gpt_3_5_turbo_instruct,
            top_p = Some(0.5),
            temperature = Some(0.5)
          )
        )
        .map(response => response.choices.head.text)
    }

    futureF.flatMap { f =>
      F.async[String] { cb =>
        f.onComplete {
          case scala.util.Success(response) =>
            logger.info("OpenAI gpt3 request succeeded")
            cb(Right(response))
          case scala.util.Failure(e) =>
            logger.warn("OpenAI gpt3 request failed", e)
            cb(Left(e))
        }
      }
    }
  }

  def gpt4TextCompletion[F[_]](prompt: String)(
      implicit F: Concurrent[F]
  ): F[String] = {
    val futureF: F[Future[String]] = buildService[F].map { svc: CeqOpenAIService =>
      svc
        .createChatCompletion(
          Seq(UserMessage(prompt)),
          CreateChatCompletionSettings(
            model = ModelId.gpt_4_turbo_2024_04_09,
            top_p = Some(0.5),
            temperature = Some(0.5)
          )
        )
        .map(response => response.choices.head.message.content)
    }

    futureF.flatMap { f =>
      F.async[String] { cb =>
        f.onComplete {
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
}

object OpenAIServiceImpl {
  lazy val realOpenAIService = new OpenAIServiceImpl
}
