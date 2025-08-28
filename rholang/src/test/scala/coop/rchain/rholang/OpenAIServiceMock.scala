package coop.rchain.rholang

import cats.effect.{Concurrent, Sync}
import cats.syntax.all._
import coop.rchain.rholang.interpreter.{GrpcClientService, OpenAIService}
import scala.util.Random
import java.util.concurrent.atomic.AtomicInteger

// This is a mock of the OpenAIService that returns a random string for the text completion, image creation, and audio speech.
// It is used to test the non-deterministic processes.
class NonDeterministicOpenAIServiceMock extends OpenAIService {

  private val random = new Random()

  override def gpt4TextCompletion[F[_]](prompt: String)(implicit F: Concurrent[F]): F[String] =
    F.pure(random.nextString(10))

  override def dalle3CreateImage[F[_]](prompt: String)(implicit F: Concurrent[F]): F[String] =
    F.pure("https://example.com/image.png")

  override def ttsCreateAudioSpeech[F[_]](
      prompt: String
  )(implicit F: Concurrent[F]): F[Array[Byte]] =
    F.pure(Array.empty[Byte])
}

// Base mock implementation that handles common OpenAI service interface
abstract class BaseOpenAIServiceMock extends OpenAIService {

  protected def throwUnsupported(operation: String): Nothing =
    throw new UnsupportedOperationException(s"$operation is not supported in this mock")

  override def dalle3CreateImage[F[_]](prompt: String)(implicit F: Concurrent[F]): F[String] =
    F.raiseError(new UnsupportedOperationException("DALL-E 3 not implemented in mock"))

  override def ttsCreateAudioSpeech[F[_]](
      prompt: String
  )(implicit F: Concurrent[F]): F[Array[Byte]] =
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

  override def gpt4TextCompletion[F[_]](prompt: String)(implicit F: Concurrent[F]): F[String] =
    if (isFirstCall)
      F.pure(completion)
    else
      ensureFirstCallOnly *> F.never // Should never reach here after error
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

  override def gpt4TextCompletion[F[_]](prompt: String)(implicit F: Concurrent[F]): F[String] =
    throwUnsupported("GPT4 not implemented in DALL-E 3 mock")

  override def dalle3CreateImage[F[_]](prompt: String)(implicit F: Concurrent[F]): F[String] =
    if (isFirstCall)
      F.pure(imageUrl)
    else
      ensureFirstCallOnly *> F.never
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

  override def gpt4TextCompletion[F[_]](prompt: String)(implicit F: Concurrent[F]): F[String] =
    throwUnsupported("GPT4 not implemented in TTS mock")

  override def ttsCreateAudioSpeech[F[_]](text: String)(implicit F: Concurrent[F]): F[Array[Byte]] =
    if (isFirstCall)
      F.pure(audioBytes)
    else
      ensureFirstCallOnly *> F.never
}

// Mock that fails on first call for any service
class ErrorOnFirstCallMock(errorMessage: String = "HTTP 500") extends BaseOpenAIServiceMock {
  @volatile private var callCount = 0

  private def isFirstCall: Boolean = {
    callCount += 1
    callCount == 1
  }

  override def gpt4TextCompletion[F[_]](prompt: String)(implicit F: Concurrent[F]): F[String] =
    if (isFirstCall)
      F.raiseError(new Exception(errorMessage))
    else
      throwUnsupported("Multiple GPT4 calls after error")

  override def dalle3CreateImage[F[_]](prompt: String)(implicit F: Concurrent[F]): F[String] =
    if (isFirstCall)
      F.raiseError(new Exception(errorMessage))
    else
      throwUnsupported("Multiple DALL-E 3 calls after error")

  override def ttsCreateAudioSpeech[F[_]](text: String)(implicit F: Concurrent[F]): F[Array[Byte]] =
    if (isFirstCall)
      F.raiseError(new Exception(errorMessage))
    else
      throwUnsupported("Multiple TTS calls after error")
}

// Mock that succeeds on first grpcTell call only
class SingleGrpcClientMock(expectedHost: String, expectedPort: Int) extends GrpcClientService {
  @volatile private var callCount = 0

  private def isFirstCall: Boolean = {
    callCount += 1
    callCount == 1
  }

  private def ensureFirstCallOnly[F[_]](implicit F: Sync[F]): F[Unit] =
    F.raiseError(new RuntimeException("GrpcClient should be called only once"))

  override def initClientAndTell[F[_]: Sync](
      clientHost: String,
      clientPort: Long,
      payload: String
  ): F[Unit] =
    if (isFirstCall && clientHost == expectedHost && clientPort == expectedPort)
      Sync[F].unit
    else
      ensureFirstCallOnly // This will throw an error before reaching this point

  def wasCalled: Boolean = callCount > 0
}

object OpenAIServiceMock {
  val nonDeterministicService: OpenAIService = new NonDeterministicOpenAIServiceMock

  // Factory methods for creating mocks
  def createSingleCompletionMock(completion: String): OpenAIService =
    new SingleCompletionMock(completion)

  def createSingleDalle3Mock(imageUrl: String): OpenAIService =
    new SingleDalle3Mock(imageUrl)

  def createSingleTtsAudioMock(audioBytes: Array[Byte]): OpenAIService =
    new SingleTtsAudioMock(audioBytes)

  def createErrorOnFirstCallMock(): OpenAIService =
    new ErrorOnFirstCallMock()

  // Factory method for creating grpc mock
  def createSingleGrpcClientMock(expectedHost: String, expectedPort: Int): SingleGrpcClientMock =
    new SingleGrpcClientMock(expectedHost, expectedPort)
}
