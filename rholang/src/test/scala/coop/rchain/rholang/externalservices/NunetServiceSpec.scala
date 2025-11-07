package coop.rchain.rholang.externalservices

import cats.effect.IO
import cats.effect.concurrent.Ref
import coop.rchain.crypto.hash.Blake2b512Random
import coop.rchain.metrics.{Metrics, NoopSpan, Span}
import coop.rchain.models.Expr.ExprInstance.{GInt, GString}
import coop.rchain.models.rholang.implicits._
import coop.rchain.models._
import coop.rchain.rholang.interpreter.InterpreterUtil._
import coop.rchain.rholang.syntax._
import coop.rchain.rspace.syntax._
import coop.rchain.shared.Log
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import org.scalatest._
import org.scalatest.Matchers._

import scala.collection.immutable.BitSet

class NunetServiceSpec extends FlatSpec with Matchers {

  implicit val logF: Log[Task]            = new Log.NOPLog[Task]
  implicit val noopMetrics: Metrics[Task] = new Metrics.MetricsNOP[Task]
  implicit val noopSpan: Span[Task]       = NoopSpan[Task]()

  val errorHandler                    = Ref.unsafe[IO, Vector[Throwable]](Vector.empty)
  implicit val rand: Blake2b512Random = Blake2b512Random(Array.empty[Byte])

  "Nunet deploy ensemble process" should "return ensemble ID" in {
    val contract =
      """
        |new deployEnsemble(`rho:nunet:deployment:new`) in {
        |  deployEnsemble!("version: 1.0\nallocations: []", 10, 0)
        |}
      """.stripMargin

    TestNunetFixture.testNunetDeploy(
      contract,
      List(Expr(GString("mock-ensemble-12345")))
    )
  }

  "Nunet deployment status process" should "return status information" in {
    val contract =
      """
        |new deploymentStatus(`rho:nunet:deployment:status`) in {
        |  deploymentStatus!("ensemble-123", 0)
        |}
      """.stripMargin

    TestNunetFixture.testNunetDeploymentStatus(
      contract,
      List(
        Expr(
          GString(
            "ensemble_id=ensemble-123,status=running,allocations=bootstrap:running,validator1:running,validator2:running"
          )
        )
      )
    )
  }

  "Nunet deployment list process" should "return list of deployments" in {
    val contract =
      """
        |new deploymentList(`rho:nunet:deployment:list`) in {
        |  deploymentList!(0)
        |}
      """.stripMargin

    TestNunetFixture.testNunetDeploymentList(
      contract,
      List(
        EList(
          Seq(
            Expr(GString("ensemble-1:running:3")),
            Expr(GString("ensemble-2:stopped:3"))
          ),
          locallyFree = BitSet(),
          connectiveUsed = false,
          remainder = None
        )
      )
    )
  }

  "Nunet deployment logs process" should "return logs as string" in {
    val contract =
      """
        |new deploymentLogs(`rho:nunet:deployment:logs`) in {
        |  deploymentLogs!("ensemble-123", 0)
        |}
      """.stripMargin

    TestNunetFixture.testNunetDeploymentLogs(
      contract,
      List(Expr(GString("Mock logs for ensemble ensemble-123\nLine 1\nLine 2")))
    )
  }

  "Nunet deployment manifest process" should "return manifest information" in {
    val contract =
      """
        |new deploymentManifest(`rho:nunet:deployment:manifest`) in {
        |  deploymentManifest!("ensemble-123", 0)
        |}
      """.stripMargin

    TestNunetFixture.testNunetDeploymentManifest(
      contract,
      List(
        Expr(
          GString(
            "ensemble_id=ensemble-123,peers=peer-bootstrap,peer-validator1,peer-validator2,ips=bootstrap=192.168.1.10;validator1=192.168.1.11;validator2=192.168.1.12,ports=bootstrap:[40400->40400@bootstrap,40401->40401@bootstrap];validator1:[40410->40400@validator1,40411->40401@validator1]"
          )
        )
      )
    )
  }

  "Nunet generate ensemble process" should "return YAML content" in {
    val contract =
      """
        |new generateEnsemble(`rho:nunet:ensemble:generate`) in {
        |  generateEnsemble!(Nil, 0)
        |}
      """.stripMargin

    val result = TestNunetFixture.testAndGetResult(contract)

    // Should return a string containing YAML
    result should not be empty
    result.head.getGString should include("version:")
    result.head.getGString should include("allocations:")
  }

  "Nunet validate ensemble process" should "validate correct YAML" in {
    val contract =
      """
        |new validateEnsemble(`rho:nunet:ensemble:validate`) in {
        |  validateEnsemble!("version: 1.0\nallocations:\n  - name: test", 0)
        |}
      """.stripMargin

    TestNunetFixture.testNunetValidateEnsemble(
      contract,
      List(Expr(GString("valid=true,errors=,warnings=")))
    )
  }

  it should "return errors for invalid YAML" in {
    val contract =
      """
        |new validateEnsemble(`rho:nunet:ensemble:validate`) in {
        |  validateEnsemble!("invalid yaml without required fields", 0)
        |}
      """.stripMargin

    val result = TestNunetFixture.testAndGetResult(contract)

    result should not be empty
    result.head.getGString should include("valid=false")
    result.head.getGString should include("errors=")
  }
}

object TestNunetFixture {
  import coop.rchain.rholang.Resources._

  def testNunetDeploy(contract: String, expected: List[Expr]): Unit =
    testRuntime(contract, "rho:nunet:deployment:new", expected)

  def testNunetDeploymentStatus(contract: String, expected: List[Expr]): Unit =
    testRuntime(contract, "rho:nunet:deployment:status", expected)

  def testNunetDeploymentList(contract: String, expected: List[Expr]): Unit =
    testRuntime(contract, "rho:nunet:deployment:list", expected)

  def testNunetDeploymentLogs(contract: String, expected: List[Expr]): Unit =
    testRuntime(contract, "rho:nunet:deployment:logs", expected)

  def testNunetDeploymentManifest(contract: String, expected: List[Expr]): Unit =
    testRuntime(contract, "rho:nunet:deployment:manifest", expected)

  def testNunetGenerateEnsemble(contract: String, expected: List[Expr]): Unit =
    testRuntime(contract, "rho:nunet:ensemble:generate", expected)

  def testNunetValidateEnsemble(contract: String, expected: List[Expr]): Unit =
    testRuntime(contract, "rho:nunet:ensemble:validate", expected)

  def testAndGetResult(contract: String): Seq[Expr] = {
    implicit val logF: Log[Task]            = new Log.NOPLog[Task]
    implicit val noopMetrics: Metrics[Task] = new Metrics.MetricsNOP[Task]
    implicit val noopSpan: Span[Task]       = NoopSpan[Task]()
    implicit val rand: Blake2b512Random     = Blake2b512Random(Array.empty[Byte])

    mkRuntime[Task]("nunet-system-processes-test")
      .use { rhoRuntime =>
        for {
          _ <- evaluate[Task](
                rhoRuntime,
                contract
              )
          data <- rhoRuntime.getData(GInt(0L))
        } yield {
          if (data.nonEmpty) data.head.a.pars.head.exprs
          else Nil
        }
      }
      .runSyncUnsafe()
  }

  private def testRuntime(contract: String, systemProcess: String, expected: List[Expr]): Unit = {
    implicit val logF: Log[Task]            = new Log.NOPLog[Task]
    implicit val noopMetrics: Metrics[Task] = new Metrics.MetricsNOP[Task]
    implicit val noopSpan: Span[Task]       = NoopSpan[Task]()
    implicit val rand: Blake2b512Random     = Blake2b512Random(Array.empty[Byte])

    val result = mkRuntime[Task]("nunet-system-processes-test")
      .use { rhoRuntime =>
        for {
          _ <- evaluate[Task](
                rhoRuntime,
                contract
              )
          data <- rhoRuntime.getData(GInt(0L))
        } yield {
          if (data.nonEmpty) data.head.a.pars.head.exprs
          else Nil
        }
      }
      .runSyncUnsafe()

    result.toSeq should contain theSameElementsAs expected
  }
}
