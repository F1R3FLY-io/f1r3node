package coop.rchain.rholang

import cats.effect.Concurrent
import coop.rchain.rholang.interpreter.OpenAIService
import scala.util.Random

class OpenAIServiceMock extends OpenAIService {

  private val random = new Random()

  override def ttsCreateAudioSpeech[F[_]](
      prompt: String
  )(implicit F: Concurrent[F]): F[Array[Byte]] = {
    // Generate random audio data
    val randomBytes = new Array[Byte](random.nextInt(1000) + 100) // 100-1099 bytes
    random.nextBytes(randomBytes)
    F.pure(randomBytes)
  }

  override def dalle3CreateImage[F[_]](prompt: String)(implicit F: Concurrent[F]): F[String] = {
    // Generate random image URLs
    val imageId = random.nextInt(10000) + 1
    val domain  = List("example.com", "mock-ai.org", "test-dalle.net")(random.nextInt(3))
    F.pure(s"https://$domain/generated-image-$imageId.jpg")
  }

  override def gpt4TextCompletion[F[_]](prompt: String)(implicit F: Concurrent[F]): F[String] = {
    // Generate non-deterministic responses
    val responses = List(
      s"$prompt - Response A with random number ${random.nextInt(1000)}",
      s"$prompt - Response B with random number ${random.nextInt(1000)}",
      s"$prompt - Response C with random number ${random.nextInt(1000)}",
      s"Processed: $prompt [ID: ${random.nextLong().abs}]",
      s"AI generated response to '$prompt' (rand: ${random.nextDouble()})",
      s"$prompt -> Generated content ${random.alphanumeric.take(10).mkString}"
    )
    val selectedResponse = responses(random.nextInt(responses.length))
    F.pure(selectedResponse)
  }
}

object OpenAIServiceMock {
  val echoService: OpenAIService = new OpenAIServiceMock
}
