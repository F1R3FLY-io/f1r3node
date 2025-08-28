package coop.rchain.rholang.interpreter.accounting

import cats.Parallel
import cats.data.Chain
import cats.effect._
import cats.mtl.FunctorTell
import cats.syntax.all._
import coop.rchain.crypto.hash.Blake2b512Random
import coop.rchain.metrics
import coop.rchain.metrics.{Metrics, NoopSpan, Span}
import coop.rchain.models.{BindPattern, ListParWithRandom, Par, TaggedContinuation}
import coop.rchain.rholang.{OpenAIServiceMock, Resources}
import coop.rchain.rholang.interpreter.RhoRuntime.RhoHistoryRepository
import coop.rchain.rholang.interpreter.SystemProcesses.Definition
import coop.rchain.rholang.interpreter.accounting.utils._
import coop.rchain.rholang.interpreter._
import coop.rchain.rholang.syntax._
import coop.rchain.rspace.RSpace.RSpaceStore
import coop.rchain.rspace.syntax.rspaceSyntaxKeyValueStoreManager
import coop.rchain.rspace.{Checkpoint, Match, RSpace}
import coop.rchain.shared.Log
import coop.rchain.store.InMemoryStoreManager
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import org.scalatest.prop.PropertyChecks
import org.scalatest.{AppendedClues, FlatSpec, Matchers}

import scala.concurrent.duration._

class NonDeterministicProcessesSpec
    extends FlatSpec
    with Matchers
    with PropertyChecks
    with AppendedClues {

  // Enable OpenAI processes for testing
  System.setProperty("openai.enabled", "true")

  private def evaluateAndReplay(
      initialPhlo: Cost,
      term: String
  ): (EvaluateResult, EvaluateResult) = {

    implicit val logF: Log[Task]           = new Log.NOPLog[Task]
    implicit val metricsEff: Metrics[Task] = new metrics.Metrics.MetricsNOP[Task]
    implicit val noopSpan: Span[Task]      = NoopSpan[Task]()
    implicit val ms: Metrics.Source        = Metrics.BaseSource
    implicit val kvm                       = InMemoryStoreManager[Task]

    val evaluaResult = for {
      costLog                     <- costLog[Task]()
      cost                        <- CostAccounting.emptyCost[Task](implicitly, metricsEff, costLog, ms)
      store                       <- kvm.rSpaceStores
      spaces                      <- Resources.createRuntimes[Task](store)
      (runtime, replayRuntime, _) = spaces
      result <- {
        implicit def rand: Blake2b512Random = Blake2b512Random(Array.empty[Byte])
        runtime.evaluate(
          term,
          initialPhlo,
          Map.empty
        )(rand) >>= { playResult =>
          runtime.createCheckpoint >>= {
            case Checkpoint(root, log) =>
              replayRuntime.reset(root) >> replayRuntime.rig(log) >>
                replayRuntime.evaluate(
                  term,
                  initialPhlo,
                  Map.empty
                )(rand) >>= { replayResult =>
                replayRuntime.checkReplayData.as((playResult, replayResult))
              }
          }
        }
      }
    } yield result

    evaluaResult.runSyncUnsafe(7500.seconds)
  }

  private def assertReplayConsistency(
      initialPhlo: Cost,
      contract: String,
      testName: String,
      printCosts: Boolean = false
  ): Unit = {
    val (playResult, replayResult) = evaluateAndReplay(initialPhlo, contract)

    if (printCosts) {
      println(s"$testName - Initial Phlo: ${initialPhlo.value}")
      println(s"$testName - Play Cost: ${playResult.cost.value}")
      println(s"$testName - Replay Cost: ${replayResult.cost.value}")
      println(s"$testName - Cost Difference: ${playResult.cost.value - replayResult.cost.value}")
    }

    withClue(s"$testName - Play result should not have errors: ") {
      playResult.errors shouldBe empty
    }

    withClue(s"$testName - Replay result should not have errors: ") {
      replayResult.errors shouldBe empty
    }

    withClue(s"$testName - Replay cost should match play cost: ") {
      replayResult.cost shouldEqual playResult.cost
    }
  }

  "GPT4 process" should "produce deterministic costs during replay" in {
    assertReplayConsistency(
      Cost(Int.MaxValue),
      "new output, gpt4(`rho:ai:gpt4`) in { gpt4!(\"abc\", *output) | for (_ <- output) { Nil }}",
      "GPT4 process basic test",
      printCosts = true
    )
  }

  it should "produce consistent costs for multiple GPT4 queries" in {
    assertReplayConsistency(
      Cost(Int.MaxValue),
      """new output1, output2, gpt4(`rho:ai:gpt4`) in { 
        |  gpt4!("first query", *output1) |
        |  gpt4!("second query", *output2) |
        |  for (_ <- output1; _ <- output2) { Nil }
        |}""".stripMargin,
      "Multiple GPT4 queries"
    )
  }

  it should "produce consistent costs when GPT4 output is used in contract logic" in {
    assertReplayConsistency(
      Cost(Int.MaxValue),
      """new output, result, gpt4(`rho:ai:gpt4`) in { 
        |  gpt4!("test prompt", *output) |
        |  for (@response <- output) {
        |    match response {
        |      "" => @"empty"!("No response")
        |      _  => @"response"!(response)
        |    }
        |  }
        |}""".stripMargin,
      "GPT4 with response processing"
    )
  }

  "GPT4 non-deterministic process" should "handle nested operations consistently" in {
    assertReplayConsistency(
      Cost(Int.MaxValue),
      """new output1, output2, gpt4(`rho:ai:gpt4`) in { 
        |  gpt4!("initial prompt", *output1) |
        |  for (@firstResult <- output1) {
        |    gpt4!(firstResult, *output2) |
        |    for (@secondResult <- output2) {
        |      @"final"!(firstResult, secondResult)
        |    }
        |  }
        |}""".stripMargin,
      "Nested GPT4 operations"
    )
  }

  "Replay consistency" should "be maintained for individual executions" in {
    val contract = """new output, gpt4(`rho:ai:gpt4`) in { gpt4!("deterministic test", *output) }"""

    // Test single execution for replay consistency
    val (playResult, replayResult) = evaluateAndReplay(Cost(Int.MaxValue), contract)

    withClue("Play and replay should have the same cost: ") {
      replayResult.cost shouldEqual playResult.cost
    }
  }

  it should "handle complex contracts with GPT4 elements" in {
    assertReplayConsistency(
      Cost(Int.MaxValue),
      """new loop, gpt4(`rho:ai:gpt4`) in {
        |  contract loop(@n, @acc) = {
        |    match n {
        |      0 => @"result"!(acc)
        |      _ => {
        |        new gptOut in {
        |          gpt4!("test", *gptOut) |
        |          for (@g <- gptOut) {
        |            loop!(n - 1, acc + 1)
        |          }
        |        }
        |      }
        |    }
        |  } |
        |  loop!(2, 0)
        |}""".stripMargin,
      "Complex contract with loops and GPT4 processes"
    )
  }

  "Edge cases for Int.MaxValue" should "handle exact Int.MaxValue cost limits" in {
    assertReplayConsistency(
      Cost(Int.MaxValue),
      """new output, gpt4(`rho:ai:gpt4`) in { 
        |  gpt4!("test", *output) |
        |  for (@r <- output) { 
        |    @"result"!(r)
        |  }
        |}""".stripMargin,
      "Exact Int.MaxValue boundary test",
      printCosts = true
    )
  }

  it should "handle operations near Int.MaxValue without overflow" in {
    assertReplayConsistency(
      Cost(Int.MaxValue - 1000),
      """new output, gpt4(`rho:ai:gpt4`) in { 
        |  gpt4!("test with near max cost", *output) |
        |  for (@response <- output) {
        |    @"processed"!(response.length())
        |  }
        |}""".stripMargin,
      "Near Int.MaxValue cost test",
      printCosts = true
    )
  }

  it should "handle minimal cost scenarios" in {
    assertReplayConsistency(
      Cost(1000),
      "new output, gpt4(`rho:ai:gpt4`) in { gpt4!(\"short\", *output) }",
      "Minimal cost test",
      printCosts = true
    )
  }

  "Cost boundary tests" should "work with various cost limits" in {
    val testCases = List(
      (Cost(10000), "Low cost boundary"),
      (Cost(100000), "Medium cost boundary"),
      (Cost(1000000), "High cost boundary"),
      (Cost(Int.MaxValue / 2), "Half max cost boundary"),
      (Cost(Int.MaxValue - 100), "Near max cost boundary"),
      (Cost(Int.MaxValue), "Max cost boundary")
    )

    testCases.foreach {
      case (cost, testName) =>
        withClue(s"Testing $testName with cost ${cost.value}: ") {
          assertReplayConsistency(
            cost,
            """new output, gpt4(`rho:ai:gpt4`) in { 
            |  gpt4!("boundary test", *output) |
            |  for (@g <- output) { 
            |    @"result"!(g)
            |  }
            |}""".stripMargin,
            testName,
            printCosts = true
          )
        }
    }
  }

  it should "maintain consistency when cost consumption approaches limit" in {
    // Test with a more complex contract that might consume significant cost
    assertReplayConsistency(
      Cost(Int.MaxValue),
      """new gpt4(`rho:ai:gpt4`) in {
        |  new loopCh in {
        |    contract loopCh(@n) = {
        |      match n {
        |        0 => @"done"!("finished")
        |        _ => {
        |          new gptOut in {
        |            gpt4!("iteration", *gptOut) |
        |            for (@g <- gptOut) {
        |              @"counter"!(n, g) |
        |              loopCh!(n - 1)
        |            }
        |          }
        |        }
        |      }
        |    } |
        |    loopCh!(3)
        |  }
        |}""".stripMargin,
      "High cost consumption test",
      printCosts = true
    )
  }

  "Cost exhaustion scenarios" should "handle gracefully when approaching limits" in {
    // Test what happens when we might run out of cost
    val lowCost = Cost(50) // Intentionally low cost that might be exhausted

    val (playResult, replayResult) = evaluateAndReplay(
      lowCost,
      """new output, gpt4(`rho:ai:gpt4`) in { 
        |  gpt4!("test", *output) |
        |  for (@r <- output) { 
        |    @"result"!(r)
        |  }
        |}""".stripMargin
    )

    println(
      s"Low cost test - Initial: ${lowCost.value}, Play: ${playResult.cost.value}, Replay: ${replayResult.cost.value}"
    )

    // Both should have the same behavior (either both succeed or both fail the same way)
    withClue("Play and replay should have the same number of errors: ") {
      playResult.errors.size shouldEqual replayResult.errors.size
    }

    withClue("Cost consumption should be consistent: ") {
      playResult.cost shouldEqual replayResult.cost
    }
  }
}
