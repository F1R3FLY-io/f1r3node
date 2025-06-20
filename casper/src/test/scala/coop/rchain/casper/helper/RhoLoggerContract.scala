package coop.rchain.casper.helper
import cats.Functor
import cats.syntax.functor._
import cats.effect.Concurrent
import coop.rchain.metrics.Span
import coop.rchain.models.{ListParWithRandom, Par}
import coop.rchain.rholang.interpreter.SystemProcesses.ProcessContext
import coop.rchain.rholang.interpreter.{ContractCall, PrettyPrinter, RhoType}
import coop.rchain.shared.{Log, LogSource}

object RhoLoggerContract {
  val prettyPrinter = PrettyPrinter()

  //TODO extract a `RhoPatterns[F]` algebra that will move passing the Span, the Dispatcher, and the Space parameters closer to the edge of the world
  def handleMessage[F[_]: Log: Functor: Concurrent: Span](
      ctx: ProcessContext[F]
  )(message: Seq[ListParWithRandom], isReplay: Boolean, previousOutput: Seq[Par]): F[Seq[Par]] = {
    val isContractCall = new ContractCall(ctx.space, ctx.dispatcher)

    (message, isReplay, previousOutput) match {
      case isContractCall(_, _, _, Seq(RhoType.String(logLevel), par)) =>
        val msg         = prettyPrinter.buildString(par)
        implicit val ev = LogSource.matLogSource

        logLevel match {
          case "trace" => Log[F].trace(msg).map(_ => Seq.empty[Par])
          case "debug" => Log[F].debug(msg).map(_ => Seq.empty[Par])
          case "info"  => Log[F].info(msg).map(_ => Seq.empty[Par])
          case "warn"  => Log[F].warn(msg).map(_ => Seq.empty[Par])
          case "error" => Log[F].error(msg).map(_ => Seq.empty[Par])
        }
    }
  }
}
