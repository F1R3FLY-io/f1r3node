package coop.rchain.rholang.externalservices

import cats.effect.Concurrent
import cats.syntax.all._

class DisabledOllamaServiceMock extends OllamaService {
  def chatCompletion[F[_]](model: String, prompt: String)(implicit F: Concurrent[F]): F[String] =
    F.raiseError(new UnsupportedOperationException("OllamaService is disabled in tests"))

  def textGeneration[F[_]](model: String, prompt: String)(implicit F: Concurrent[F]): F[String] =
    F.raiseError(new UnsupportedOperationException("OllamaService is disabled in tests"))

  def listModels[F[_]]()(implicit F: Concurrent[F]): F[List[String]] =
    F.raiseError(new UnsupportedOperationException("OllamaService is disabled in tests"))
}

class EchoOllamaServiceMock extends OllamaService {
  def chatCompletion[F[_]](model: String, prompt: String)(implicit F: Concurrent[F]): F[String] =
    F.pure(s"Echo: $prompt")

  def textGeneration[F[_]](model: String, prompt: String)(implicit F: Concurrent[F]): F[String] =
    F.pure(s"Generate: $prompt")

  def listModels[F[_]]()(implicit F: Concurrent[F]): F[List[String]] =
    F.pure(List("mock-model-1", "mock-model-2"))
}

object OllamaServiceMock {
  lazy val disabledService: OllamaService = new DisabledOllamaServiceMock
  lazy val echoService: OllamaService     = new EchoOllamaServiceMock
}
