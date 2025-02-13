package coop.rchain.rholang.interpreter

import akka.actor.ActorSystem
import akka.stream.Materializer
import akka.stream.scaladsl.{FileIO, Sink}
import akka.util.ByteString
import cats.effect.Concurrent
import io.cequence.openaiscala.domain.{ModelId, UserMessage}
import io.cequence.openaiscala.domain.settings.{
  CreateChatCompletionSettings,
  CreateCompletionSettings,
  CreateImageSettings,
  CreateSpeechSettings,
  VoiceType
}
import io.cequence.openaiscala.service.OpenAIServiceFactory

import java.util.concurrent.Executors
import scala.concurrent.{ExecutionContext, Future}
import scala.util.Random

trait OpenAIService {

  /** @return mp3 data as a base64 decoded string */
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

  private val logger = org.slf4j.LoggerFactory.getLogger(getClass)

  implicit private val ec: ExecutionContext =
    ExecutionContext.fromExecutor(Executors.newCachedThreadPool())
  private val system                              = ActorSystem()
  implicit private val materializer: Materializer = Materializer(system)

  private lazy val service = OpenAIServiceFactory() // reads OPENAI_SCALA_CLIENT_API_KEY from env

  // shutdown system before jvm shutdown
  sys.addShutdownHook {
    system.terminate()
  }

  def ttsCreateAudioSpeech[F[_]](prompt: String)(
      implicit F: Concurrent[F]
  ): F[Array[Byte]] = {
    val f: Future[Array[Byte]] =
      service
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

    // future => F
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

  def dalle3CreateImage[F[_]](prompt: String)(
      implicit F: Concurrent[F]
  ): F[String] = {
    val f: Future[String] =
      service
        .createImage(
          prompt,
          CreateImageSettings(
            model = Some(ModelId.dall_e_3),
            n = Some(1)
          )
        )
        .map(response => response.data.headOption.flatMap(_.get("url")).get) // TODO: handle error

    // future => F
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

  def gpt3TextCompletion[F[_]](prompt: String)(
      implicit F: Concurrent[F]
  ): F[String] = {
    val f: Future[String] = service
      .createCompletion(
        prompt,
        CreateCompletionSettings(
          model = ModelId.gpt_3_5_turbo_instruct,
          top_p = Some(0.5),
          temperature = Some(0.5)
        )
      )
      .map(response => response.choices.head.text)

    // future => F
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

  def gpt4TextCompletion[F[_]](prompt: String)(
      implicit F: Concurrent[F]
  ): F[String] = {
    val f: Future[String] =
      service
        .createChatCompletion(
          Seq(UserMessage(prompt)),
          CreateChatCompletionSettings(
            model = ModelId.gpt_4_turbo_2024_04_09,
            top_p = Some(0.5),
            temperature = Some(0.5)
          )
        )
        .map(response => response.choices.head.message.content)

    // future => F
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

object OpenAIServiceImpl {
  lazy val realOpenAIService = new OpenAIServiceImpl
}
