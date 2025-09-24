package coop.rchain.rholang.externalservices

import cats.effect.IO
import cats.effect.concurrent.Ref
import coop.rchain.metrics.{Metrics, NoopSpan, Span}
import coop.rchain.shared.Log
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import coop.rchain.crypto.hash.Blake2b512Random
import coop.rchain.models.Expr.ExprInstance.GString
import coop.rchain.models.rholang.implicits._
import coop.rchain.models._
import coop.rchain.rholang.externalservices.OllamaServiceMock
import coop.rchain.rholang.interpreter.RhoRuntime.RhoISpace
import coop.rchain.rholang.interpreter.InterpreterUtil._
import coop.rchain.rholang.syntax._
import coop.rchain.rspace.syntax._
import org.scalatest._
import org.scalatest.Matchers._

import scala.collection.immutable.BitSet

class OllamaServiceSpec extends FlatSpec with Matchers {

  implicit val logF: Log[Task]            = new Log.NOPLog[Task]
  implicit val noopMetrics: Metrics[Task] = new Metrics.MetricsNOP[Task]
  implicit val noopSpan: Span[Task]       = NoopSpan[Task]()

  val errorHandler                    = Ref.unsafe[IO, Vector[Throwable]](Vector.empty)
  implicit val rand: Blake2b512Random = Blake2b512Random(Array.empty[Byte])

  "Ollama chat process" should "work with model and prompt" in {
    val contract =
      """
        |new chat(`rho:ollama:chat`), return in {
        |  chat!("llama3.2", "What is 2+2?", *return)
        |}
      """.stripMargin

    TestOllamaFixture.testOllama(
      contract,
      List(Expr(GString("What is 2+2?")))
    )
  }

  it should "work with default model (prompt only)" in {
    val contract =
      """
        |new chat(`rho:ollama:chat`), return in {
        |  chat!("What is the meaning of life?", *return)
        |}
      """.stripMargin

    TestOllamaFixture.testOllama(
      contract,
      List(Expr(GString("What is the meaning of life?")))
    )
  }

  "Ollama generate process" should "work with model and prompt" in {
    val contract =
      """
        |new generate(`rho:ollama:generate`), return in {
        |  generate!("llama3.2", "Write a poem", *return)
        |}
      """.stripMargin

    TestOllamaFixture.testOllamaGenerate(
      contract,
      List(Expr(GString("Write a poem")))
    )
  }

  it should "work with default model (prompt only)" in {
    val contract =
      """
        |new generate(`rho:ollama:generate`), return in {
        |  generate!("Complete this sentence", *return)
        |}
      """.stripMargin

    TestOllamaFixture.testOllamaGenerate(
      contract,
      List(Expr(GString("Complete this sentence")))
    )
  }

  "Ollama models process" should "return list of available models" in {
    val contract =
      """
        |new models(`rho:ollama:models`), return in {
        |  models!(*return)
        |}
      """.stripMargin

    TestOllamaFixture.testOllamaModels(
      contract,
      List(
        EList(
          Seq(Expr(GString("mock-model-1")), Expr(GString("mock-model-2"))),
          locallyFree = BitSet(),
          connectiveUsed = false,
          remainder = None
        )
      )
    )
  }
}

object TestOllamaFixture {
  import coop.rchain.rholang.Resources._

  def testOllama(contract: String, expected: List[Expr]): Unit =
    testRuntime(contract, "rho:ollama:chat", expected)

  def testOllamaGenerate(contract: String, expected: List[Expr]): Unit =
    testRuntime(contract, "rho:ollama:generate", expected)

  def testOllamaModels(contract: String, expected: List[Expr]): Unit =
    testRuntime(contract, "rho:ollama:models", expected)

  private def testRuntime(contract: String, systemProcess: String, expected: List[Expr]): Unit = {
    implicit val logF: Log[Task]            = new Log.NOPLog[Task]
    implicit val noopMetrics: Metrics[Task] = new Metrics.MetricsNOP[Task]
    implicit val noopSpan: Span[Task]       = NoopSpan[Task]()
    implicit val rand: Blake2b512Random     = Blake2b512Random(Array.empty[Byte])

    val result = mkRuntime[Task]("ollama-system-processes-test")
      .use { rhoRuntime =>
        for {
          _ <- evaluate[Task](
                rhoRuntime,
                contract
              )
          data <- rhoRuntime.getData(Par())
        } yield data.head.a.pars.head.exprs
      }
      .runSyncUnsafe()

    result.toSeq should contain theSameElementsAs expected
  }
}
