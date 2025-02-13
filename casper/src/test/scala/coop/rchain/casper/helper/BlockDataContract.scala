package coop.rchain.casper.helper

import cats.effect.Concurrent
import coop.rchain.crypto.PublicKey
import coop.rchain.metrics.Span
import coop.rchain.models.{ListParWithRandom, Par}
import coop.rchain.rholang.interpreter.{ContractCall, RhoType}
import coop.rchain.rholang.interpreter.SystemProcesses.ProcessContext

object BlockDataContract {
  import cats.syntax.all._

  def set[F[_]: Concurrent: Span](
      ctx: ProcessContext[F]
  )(message: Seq[ListParWithRandom], isReplay: Boolean, previousOutput: Seq[Par]): F[Seq[Par]] = {

    val isContractCall = new ContractCall(ctx.space, ctx.dispatcher)
    (message, isReplay, previousOutput) match {
      case isContractCall(
          produce,
          _,
          _,
          Seq(RhoType.String("sender"), RhoType.ByteArray(pk), ackCh)
          ) =>
        for {
          _      <- ctx.blockData.update(_.copy(sender = PublicKey(pk)))
          output = Seq(Par())
          _      <- produce(output, ackCh)
        } yield output

      case isContractCall(
          produce,
          _,
          _,
          Seq(RhoType.String("blockNumber"), RhoType.Number(n), ackCh)
          ) =>
        for {
          _      <- ctx.blockData.update(_.copy(blockNumber = n))
          output = Seq(Par())
          _      <- produce(output, ackCh)
        } yield output
    }
  }
}
