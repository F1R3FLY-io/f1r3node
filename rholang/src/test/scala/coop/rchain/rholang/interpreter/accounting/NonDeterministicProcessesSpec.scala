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
import coop.rchain.rholang.Resources
import coop.rchain.rholang.interpreter.RhoRuntime.RhoHistoryRepository
import coop.rchain.rholang.interpreter.SystemProcesses.Definition
import coop.rchain.rholang.interpreter.accounting.utils._
import coop.rchain.rholang.interpreter.{EvaluateResult, _}
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
import coop.rchain.rholang.externalservices.ExternalServices
import coop.rchain.rholang.externalservices.{
  GrpcClientMock,
  GrpcClientService,
  OllamaService,
  OllamaServiceMock,
  OpenAIService,
  OpenAIServiceImpl,
  OpenAIServiceMock
}
import coop.rchain.rholang.externalservices.TestExternalServices

class NonDeterministicProcessesSpec
    extends FlatSpec
    with Matchers
    with PropertyChecks
    with AppendedClues {

  // Enable OpenAI processes for testing
  System.setProperty("openai.enabled", "true")

  private def evaluateAndReplay(
      initialPhlo: Cost,
      term: String,
      externalServices: ExternalServices
  ): (EvaluateResult, EvaluateResult) = {

    implicit val logF: Log[Task]           = new Log.NOPLog[Task]
    implicit val metricsEff: Metrics[Task] = new metrics.Metrics.MetricsNOP[Task]
    implicit val noopSpan: Span[Task]      = NoopSpan[Task]()
    implicit val ms: Metrics.Source        = Metrics.BaseSource
    implicit val kvm                       = InMemoryStoreManager[Task]

    val evaluaResult = for {
      costLog <- costLog[Task]()
      cost    <- CostAccounting.emptyCost[Task](implicitly, metricsEff, costLog, ms)
      store   <- kvm.rSpaceStores
      spaces <- Resources.createRuntimes[Task](
                 store,
                 externalServices = externalServices
               )
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
      openAIService: OpenAIService = OpenAIServiceImpl.noOpInstance,
      grpcClient: GrpcClientService = GrpcClientService.noOpInstance,
      ollamaService: OllamaService = OllamaServiceMock.disabledService,
      printCosts: Boolean = false,
      expectError: Boolean = false
  ): Unit = {
    val (playResult, replayResult) =
      evaluateAndReplay(
        initialPhlo,
        contract,
        TestExternalServices(openAIService, grpcClient, ollamaService)
      )

    if (printCosts) {
      println(s"$testName - Initial Phlo: ${initialPhlo.value}")
      println(s"$testName - Play Cost: ${playResult.cost.value}")
      println(s"$testName - Replay Cost: ${replayResult.cost.value}")
      println(s"$testName - Cost Difference: ${playResult.cost.value - replayResult.cost.value}")
    }

    withClue(s"$testName - Play result should ${if (expectError) "have" else "not have"} errors: ") {
      if (expectError) {
        playResult.errors should not be empty
      } else {
        playResult.errors shouldBe empty
      }
    }

    withClue(
      s"$testName - Replay result should ${if (expectError) "have" else "not have"} errors: "
    ) {
      if (expectError) {
        replayResult.errors should not be empty
      } else {
        replayResult.errors shouldBe empty
      }
    }

    withClue(s"$testName - Replay cost should match play cost: ") {
      replayResult.cost shouldEqual playResult.cost
    }
  }

  "Replay" should "produce consistent costs without calling OpenAI service on successful execution" in {

    assertReplayConsistency(
      Cost(Int.MaxValue),
      "new output, gpt4(`rho:ai:gpt4`) in { gpt4!(\"abc\", *output) }",
      "GPT4 process basic test",
      openAIService = OpenAIServiceMock.createSingleCompletionMock("gpt4 completion")
    )
  }

  it should "handle OutOfPhlogistonsError during replay with consistent cost accounting" in {
    assertReplayConsistency(
      Cost(1000),
      "new output, gpt4(`rho:ai:gpt4`) in { gpt4!(\"abc\", *output) }",
      "GPT4 process basic test with OutOfPhlogistonsError",
      openAIService = OpenAIServiceMock.createSingleCompletionMock("a" * 1000000),
      expectError = true
    )
  }

  it should "handle OpenAI service errors during replay with consistent error propagation" in {
    assertReplayConsistency(
      Cost(Int.MaxValue),
      "new output, gpt4(`rho:ai:gpt4`) in { gpt4!(\"abc\", *output) }",
      "GPT4 process basic test with any other error",
      openAIService = OpenAIServiceMock.createErrorOnFirstCallMock(),
      expectError = true
    )
  }

  it should "produce consistent costs for DALL-E 3 without calling service on replay" in {
    assertReplayConsistency(
      Cost(Int.MaxValue),
      "new output, dalle3(`rho:ai:dalle3`) in { dalle3!(\"a cat painting\", *output) }",
      "DALL-E 3 process basic test",
      openAIService =
        OpenAIServiceMock.createSingleDalle3Mock("https://example.com/generated-image.png")
    )
  }

  it should "handle OutOfPhlogistonsError during DALL-E 3 replay with consistent cost accounting" in {
    assertReplayConsistency(
      Cost(1000),
      "new output, dalle3(`rho:ai:dalle3`) in { dalle3!(\"a cat painting\", *output) }",
      "DALL-E 3 process with OutOfPhlogistonsError",
      openAIService = OpenAIServiceMock.createSingleDalle3Mock("https://" + "a" * 1000000 + ".png"),
      expectError = true
    )
  }

  it should "handle DALL-E 3 service errors during replay with consistent error propagation" in {
    assertReplayConsistency(
      Cost(Int.MaxValue),
      "new output, dalle3(`rho:ai:dalle3`) in { dalle3!(\"a cat painting\", *output) }",
      "DALL-E 3 process with service error",
      openAIService = OpenAIServiceMock.createErrorOnFirstCallMock(),
      expectError = true
    )
  }

  it should "produce consistent costs for text-to-audio without calling service on replay" in {
    assertReplayConsistency(
      Cost(Int.MaxValue),
      "new output, tts(`rho:ai:textToAudio`) in { tts!(\"Hello world\", *output) }",
      "Text-to-audio process basic test",
      openAIService =
        OpenAIServiceMock.createSingleTtsAudioMock("fake audio bytes".getBytes("UTF-8"))
    )
  }

  it should "handle OutOfPhlogistonsError during text-to-audio replay with consistent cost accounting" in {
    assertReplayConsistency(
      Cost(1000),
      "new output, tts(`rho:ai:textToAudio`) in { tts!(\"Hello world\", *output) }",
      "Text-to-audio process with OutOfPhlogistonsError",
      openAIService = OpenAIServiceMock.createSingleTtsAudioMock(Array.fill(1000000)(0.toByte)),
      expectError = true
    )
  }

  it should "handle text-to-audio service errors during replay with consistent error propagation" in {
    assertReplayConsistency(
      Cost(Int.MaxValue),
      "new output, tts(`rho:ai:textToAudio`) in { tts!(\"Hello world\", *output) }",
      "Text-to-audio process with service error",
      openAIService = OpenAIServiceMock.createErrorOnFirstCallMock(),
      expectError = true
    )
  }

  it should "produce consistent costs for grpcTell on successful execution without calling service on replay" in {

    val grpcClientMock = GrpcClientMock.createSingleGrpcClientMock("localhost", 8080)

    assertReplayConsistency(
      Cost(Int.MaxValue),
      "new grpcTell(`rho:io:grpcTell`) in { grpcTell!(\"localhost\", 8080, \"payload\") }",
      "grpcTell process basic test",
      grpcClient = grpcClientMock
    )

    grpcClientMock.wasCalled shouldBe true
  }

  it should "handle grpcTell connection errors during replay with consistent error propagation" in {

    val grpcClientMock = GrpcClientMock.createSingleGrpcClientMock("abc", 8081)
    assertReplayConsistency(
      Cost(Int.MaxValue),
      "new grpcTell(`rho:io:grpcTell`) in { grpcTell!(\"localhost\", 8080, \"payload\") }",
      "grpcTell process with connection error",
      grpcClient = grpcClientMock,
      expectError = true
    )

    grpcClientMock.wasCalled shouldBe true
  }
}
