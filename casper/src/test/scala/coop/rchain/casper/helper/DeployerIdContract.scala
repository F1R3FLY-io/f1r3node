package coop.rchain.casper.helper
import cats.effect.Concurrent
import coop.rchain.metrics.Span
import coop.rchain.models.ListParWithRandom
import coop.rchain.rholang.interpreter.SystemProcesses.ProcessContext
import coop.rchain.rholang.interpreter.{ContractCall, RhoType}

/**
  * Warning: This should under no circumstances be available in production
  */
object DeployerIdContract {
  import cats.syntax.all._

  def get[F[_]: Concurrent: Span](
      ctx: ProcessContext[F]
  )(message: Seq[ListParWithRandom], isReplay: Boolean, previousOutput: Option[Any]): F[Any] = {

    val isContractCall = new ContractCall(ctx.space, ctx.dispatcher)
    (message, isReplay, previousOutput) match {
      case isContractCall(
          produce,
          _,
          _,
          Seq(RhoType.String("deployerId"), RhoType.ByteArray(pk), ackCh)
          ) =>
        for {
          _ <- produce(Seq(RhoType.DeployerId(pk)), ackCh)
        } yield ()
    }
  }
}
