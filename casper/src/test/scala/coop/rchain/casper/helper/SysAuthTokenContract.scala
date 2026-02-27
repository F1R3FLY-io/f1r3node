package coop.rchain.casper.helper

import cats.effect.Concurrent
import coop.rchain.metrics.Span
import coop.rchain.models.{GSysAuthToken, ListParWithRandom, Par}
import coop.rchain.rholang.interpreter.{ContractCall, RhoType}
import coop.rchain.rholang.interpreter.SystemProcesses.ProcessContext

/**
  * Warning: This should under no circumstances be available in production
  */
object SysAuthTokenContract {
  import cats.syntax.all._

  def get[F[_]: Concurrent: Span](
      ctx: ProcessContext[F]
  )(message: Seq[ListParWithRandom], isReplay: Boolean, previousOutput: Seq[Par]): F[Seq[Par]] = {

    val isContractCall = new ContractCall(ctx.space, ctx.dispatcher)
    (message, isReplay, previousOutput) match {
      case isContractCall(
          produce,
          _,
          _,
          Seq(ackCh)
          ) =>
        val output = Seq(RhoType.SysAuthToken(GSysAuthToken()))
        for {
          _ <- produce(output, ackCh)
        } yield output
    }
  }
}
