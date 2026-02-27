package coop.rchain.rholang.interpreter

import coop.rchain.metrics
import coop.rchain.metrics.{Metrics, NoopSpan, Span}
import coop.rchain.rholang.Resources.mkRuntime
import coop.rchain.rholang.interpreter.errors.UserAbortError
import coop.rchain.rholang.syntax._
import coop.rchain.shared.Log
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Matchers}

import scala.concurrent.duration._

class AbortSpec extends FlatSpec with Matchers {
  private val tmpPrefix                   = "abort-spec-store-"
  private val maxDuration                 = 5.seconds
  implicit val logF: Log[Task]            = Log.log[Task]
  implicit val noopMetrics: Metrics[Task] = new metrics.Metrics.MetricsNOP[Task]
  implicit val noopSpan: Span[Task]       = NoopSpan[Task]()

  "rho:execution:abort" should "execute successfully when called with arguments" in {
    val rhoCode =
      """
        |new abort(`rho:execution:abort`) in {
        |  abort!("Test abort")
        |}
        |""".stripMargin

    val result = execute(rhoCode)
    // Abort should result in UserAbortError and mark execution as failed
    result.errors should contain(UserAbortError)
    result.cost.value should be(316L +- 100L)
    result.succeeded should be(false)
  }

  private def execute(source: String): EvaluateResult =
    mkRuntime[Task](tmpPrefix)
      .use { runtime =>
        runtime.evaluate(source)
      }
      .runSyncUnsafe(maxDuration)
}
