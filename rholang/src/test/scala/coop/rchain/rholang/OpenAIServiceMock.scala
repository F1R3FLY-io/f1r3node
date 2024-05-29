package coop.rchain.rholang

import cats.effect.Concurrent
import coop.rchain.rholang.interpreter.OpenAIService

class OpenAIServiceMock extends OpenAIService {

  override def ttsCreateAudioSpeech[F[_]](
      prompt: String
  )(implicit F: Concurrent[F]): F[Array[Byte]] =
    F.pure(Array.emptyByteArray)

  override def dalle3CreateImage[F[_]](prompt: String)(implicit F: Concurrent[F]): F[String] =
    F.pure("https://example.com/image.jpg")

  override def gpt3TextCompletion[F[_]](prompt: String)(
      implicit F: Concurrent[F]
  ): F[String] = F.pure(prompt)

  override def gpt4TextCompletion[F[_]](prompt: String)(implicit F: Concurrent[F]): F[String] =
    F.pure(prompt)
}

object OpenAIServiceMock {
  val echoService: OpenAIService = new OpenAIServiceMock
}
