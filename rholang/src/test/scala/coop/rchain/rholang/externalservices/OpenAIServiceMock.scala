package coop.rchain.rholang.externalservices

import cats.effect.Concurrent
import cats.syntax.all._
import coop.rchain.shared.Log

import scala.util.Random

// This is a mock of the OpenAIService that returns a random string for the text completion, image creation, and audio speech.
// It is used to test the non-deterministic processes.
class NonDeterministicOpenAIServiceMock extends OpenAIService {

  private val random = new Random()

  override def gpt4TextCompletion[F[_]](
      prompt: String
  )(implicit F: Concurrent[F], L: Log[F]): F[String] =
    F.pure(random.nextString(10))

  override def dalle3CreateImage[F[_]](
      prompt: String
  )(implicit F: Concurrent[F], L: Log[F]): F[String] =
    F.pure("https://example.com/image.png")

  override def ttsCreateAudioSpeech[F[_]](
      prompt: String
  )(implicit F: Concurrent[F], L: Log[F]): F[Array[Byte]] =
    F.pure(Array.empty[Byte])
}

// Base mock implementation that handles common OpenAI service interface
abstract class BaseOpenAIServiceMock extends OpenAIService {

  protected def throwUnsupported(operation: String): Nothing =
    throw new UnsupportedOperationException(s"$operation is not supported in this mock")

  override def dalle3CreateImage[F[_]](
      prompt: String
  )(implicit F: Concurrent[F], L: Log[F]): F[String] =
    F.raiseError(new UnsupportedOperationException("DALL-E 3 not implemented in mock"))

  override def ttsCreateAudioSpeech[F[_]](
      prompt: String
  )(implicit F: Concurrent[F], L: Log[F]): F[Array[Byte]] =
    F.raiseError(new UnsupportedOperationException("TTS not implemented in mock"))
}

// Mock that returns a single completion on first call
class SingleCompletionMock(completion: String) extends BaseOpenAIServiceMock {
  @volatile private var callCount = 0

  private def isFirstCall: Boolean = {
    callCount += 1
    callCount == 1
  }

  private def ensureFirstCallOnly[F[_]](implicit F: Concurrent[F]): F[Unit] =
    if (callCount > 1)
      F.raiseError(new RuntimeException("GPT4TextCompletion should be called only once"))
    else
      F.unit

  override def gpt4TextCompletion[F[_]](
      prompt: String
  )(implicit F: Concurrent[F], L: Log[F]): F[String] =
    if (isFirstCall)
      F.pure(completion)
    else
      ensureFirstCallOnly *> F.raiseError(new RuntimeException("Should not reach here"))
}

// Mock that returns a single DALL-E 3 image URL on first call
class SingleDalle3Mock(imageUrl: String) extends BaseOpenAIServiceMock {
  @volatile private var callCount = 0

  private def isFirstCall: Boolean = {
    callCount += 1
    callCount == 1
  }

  private def ensureFirstCallOnly[F[_]](implicit F: Concurrent[F]): F[Unit] =
    if (callCount > 1)
      F.raiseError(new RuntimeException("DALL-E 3 should be called only once"))
    else
      F.unit

  override def gpt4TextCompletion[F[_]](
      prompt: String
  )(implicit F: Concurrent[F], L: Log[F]): F[String] =
    throwUnsupported("GPT4 not implemented in DALL-E 3 mock")

  override def dalle3CreateImage[F[_]](
      prompt: String
  )(implicit F: Concurrent[F], L: Log[F]): F[String] =
    if (isFirstCall)
      F.pure(imageUrl)
    else
      ensureFirstCallOnly *> F.raiseError(new RuntimeException("Should not reach here"))
}

// Mock that returns audio bytes on first call
class SingleTtsAudioMock(audioBytes: Array[Byte]) extends BaseOpenAIServiceMock {
  @volatile private var callCount = 0

  private def isFirstCall: Boolean = {
    callCount += 1
    callCount == 1
  }

  private def ensureFirstCallOnly[F[_]](implicit F: Concurrent[F]): F[Unit] =
    if (callCount > 1)
      F.raiseError(new RuntimeException("TTS should be called only once"))
    else
      F.unit

  override def gpt4TextCompletion[F[_]](
      prompt: String
  )(implicit F: Concurrent[F], L: Log[F]): F[String] =
    throwUnsupported("GPT4 not implemented in TTS mock")

  override def ttsCreateAudioSpeech[F[_]](
      text: String
  )(implicit F: Concurrent[F], L: Log[F]): F[Array[Byte]] =
    if (isFirstCall)
      F.pure(audioBytes)
    else
      ensureFirstCallOnly *> F.raiseError(new RuntimeException("Should not reach here"))
}

// Mock that fails on first call for any service
class ErrorOnFirstCallMock(errorMessage: String = "HTTP 500") extends BaseOpenAIServiceMock {
  @volatile private var callCount = 0

  private def isFirstCall: Boolean = {
    callCount += 1
    callCount == 1
  }

  override def gpt4TextCompletion[F[_]](
      prompt: String
  )(implicit F: Concurrent[F], L: Log[F]): F[String] =
    if (isFirstCall)
      F.raiseError(new Exception(errorMessage))
    else
      throwUnsupported("Multiple GPT4 calls after error")

  override def dalle3CreateImage[F[_]](
      prompt: String
  )(implicit F: Concurrent[F], L: Log[F]): F[String] =
    if (isFirstCall)
      F.raiseError(new Exception(errorMessage))
    else
      throwUnsupported("Multiple DALL-E 3 calls after error")

  override def ttsCreateAudioSpeech[F[_]](
      text: String
  )(implicit F: Concurrent[F], L: Log[F]): F[Array[Byte]] =
    if (isFirstCall)
      F.raiseError(new Exception(errorMessage))
    else
      throwUnsupported("Multiple TTS calls after error")
}

// Echo mock that returns the prompt as the response
class EchoOpenAIServiceMock extends BaseOpenAIServiceMock {
  override def gpt4TextCompletion[F[_]](
      prompt: String
  )(implicit F: Concurrent[F], L: Log[F]): F[String] =
    F.pure(prompt)
}

object OpenAIServiceMock {
  val nonDeterministicService: OpenAIService = new NonDeterministicOpenAIServiceMock
  lazy val echoService: OpenAIService        = new EchoOpenAIServiceMock

  // Factory methods for creating mocks
  def createSingleCompletionMock(completion: String): OpenAIService =
    new SingleCompletionMock(completion)

  def createSingleDalle3Mock(imageUrl: String): OpenAIService =
    new SingleDalle3Mock(imageUrl)

  def createSingleTtsAudioMock(audioBytes: Array[Byte]): OpenAIService =
    new SingleTtsAudioMock(audioBytes)

  def createErrorOnFirstCallMock(): OpenAIService =
    new ErrorOnFirstCallMock()

}
