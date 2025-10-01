package coop.rchain.rholang.interpreter

import cats.effect.IO
import cats.effect.concurrent.Ref
import coop.rchain.metrics.{Metrics, NoopSpan, Span}
import coop.rchain.shared.{Base16, Log}
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import coop.rchain.crypto.PublicKey
import coop.rchain.crypto.hash.Blake2b512Random
import coop.rchain.models.Expr.ExprInstance.{GInt, GString}
import coop.rchain.models.rholang.implicits._
import coop.rchain.models._
import coop.rchain.rholang.interpreter.RhoRuntime.RhoISpace
import coop.rchain.rholang.interpreter.InterpreterUtil._
import coop.rchain.rholang.interpreter.SystemProcesses.DeployData
import coop.rchain.rholang.syntax._
import coop.rchain.rspace.syntax._
import coop.rchain.models.syntax._
import com.google.protobuf.ByteString
import org.scalatest._
import org.scalatest.Matchers._

class DeployDataSpec extends FlatSpec with Matchers {

  implicit val logF: Log[Task]            = new Log.NOPLog[Task]
  implicit val noopMetrics: Metrics[Task] = new Metrics.MetricsNOP[Task]
  implicit val noopSpan: Span[Task]       = NoopSpan[Task]()

  val errorHandler                    = Ref.unsafe[IO, Vector[Throwable]](Vector.empty)
  implicit val rand: Blake2b512Random = Blake2b512Random(Array.empty[Byte])

  "rho:deploy:data system channel" should "return timestamp, deployerId and deployId" in {
    val contract =
      """
        |new deployData(`rho:deploy:data`) in {
        |  deployData!(0)
        |}
      """.stripMargin

    val timestamp = 123L;
    val key       = PublicKey(Base16.unsafeDecode("abcd"));
    val sig       = Base16.unsafeDecode("1234").toByteString;

    TestDeployDataFixture.test(
      contract,
      DeployData(timestamp, key, sig),
      List(
        Par(exprs = Seq(Expr(GInt(timestamp)))),
        Par(
          unforgeables = Seq(
            GUnforgeable(
              GUnforgeable.UnfInstance.GDeployerIdBody(GDeployerId(key.bytes.toByteString))
            )
          )
        ),
        Par(
          unforgeables = Seq(
            GUnforgeable(
              GUnforgeable.UnfInstance.GDeployIdBody(GDeployId(sig))
            )
          )
        )
      )
    )
  }
}

object TestDeployDataFixture {
  import coop.rchain.rholang.Resources._

  def test(contract: String, deployData: DeployData, expected: List[Par]): Unit = {
    implicit val logF: Log[Task]            = new Log.NOPLog[Task]
    implicit val noopMetrics: Metrics[Task] = new Metrics.MetricsNOP[Task]
    implicit val noopSpan: Span[Task]       = NoopSpan[Task]()

    val result = mkRuntime[Task]("rho-deploy-data-test")
      .use { rhoRuntime =>
        for {
          _ <- rhoRuntime.setDeployData(deployData)
          _ <- evaluate[Task](
                rhoRuntime,
                contract
              )
          data <- rhoRuntime.getData(GInt(0L))
        } yield {
          if (data.nonEmpty) data.head.a.pars
          else Nil
        }
      }
      .runSyncUnsafe()

    result.toSeq shouldEqual expected
  }
}
