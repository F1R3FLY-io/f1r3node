package coop.rchain.node.api

import cats.effect.Sync
import cats.syntax.all._
import coop.rchain.models.Par
import coop.rchain.monix.Monixable
import coop.rchain.node.model.lsp.{LspGrpcMonix, ValidateRequest, ValidateResponse}
import coop.rchain.rholang.interpreter.compiler.Compiler
import coop.rchain.rholang.interpreter.errors.InterpreterError
import coop.rchain.rholang.interpreter.RhoRuntime
import coop.rchain.shared.syntax._
import monix.eval.Task
import monix.execution.Scheduler

object LspService {

  def apply[F[_]: Monixable: Sync](worker: Scheduler): LspGrpcMonix.Lsp =
    new LspGrpcMonix.Lsp {
      def validate(source: String): F[ValidateResponse] =
        Sync[F]
          .attempt(
            Compiler[F]
              .sourceToADT(source, Map.empty[String, Par])
          )
          .flatMap {
            case Left(er) =>
              er match {
                case _: InterpreterError => Sync[F].delay(s"Error: ${er.toString}")
                case th: Throwable       => Sync[F].delay(s"Error: $th")
              }
            case Right(_) => Sync[F].delay("")
          }
          .map(ValidateResponse(_))

      private def defer[A](task: F[A]): Task[A] =
        task.toTask.executeOn(worker)

      def validate(request: ValidateRequest): Task[ValidateResponse] =
        defer(validate(request.text))
    }
}
