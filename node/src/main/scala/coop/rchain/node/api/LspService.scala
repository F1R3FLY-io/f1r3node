package coop.rchain.node.api

import java.io.{PrintWriter, StringWriter}
import scala.util.matching.Regex
import cats.effect.Sync
import cats.syntax.all._
import coop.rchain.models.Par
import coop.rchain.monix.Monixable
import coop.rchain.node.model.lsp.{
  Diagnostic,
  DiagnosticList,
  DiagnosticSeverity,
  LspGrpcMonix,
  Position,
  Range,
  ValidateRequest,
  ValidateResponse
}
import coop.rchain.rholang.interpreter.compiler.{Compiler, SourcePosition}
import coop.rchain.rholang.interpreter.errors.{
  InterpreterError,
  LexerError,
  ReceiveOnSameChannelsError,
  SyntaxError,
  UnboundVariableRef,
  UnexpectedNameContext,
  UnexpectedProcContext,
  UnexpectedReuseOfNameContextFree,
  UnexpectedReuseOfProcContextFree
}
import coop.rchain.rholang.interpreter.RhoRuntime
import coop.rchain.shared.syntax._
import monix.eval.Task
import monix.execution.Scheduler

object LspService {
  val SOURCE: String         = "rholang"
  val RE_SYNTAX_ERROR: Regex = """syntax error\([^)]*\): .* at (\d+):(\d+)-(\d+):(\d+)""".r
  val RE_LEXER_ERROR: Regex  = """.* at (\d+):(\d+)\(\d+\)""".r

  def apply[F[_]: Monixable: Sync](worker: Scheduler): LspGrpcMonix.Lsp =
    new LspGrpcMonix.Lsp {
      def validation(
          startLine: Int,
          startColumn: Int,
          endLine: Int,
          endColumn: Int,
          message: String
      ): Seq[Diagnostic] =
        Seq(
          Diagnostic(
            range = Some(
              Range(
                start = Some(
                  Position(
                    line = startLine.toLong - 1L,
                    column = startColumn.toLong - 1L
                  )
                ),
                end = Some(
                  Position(
                    line = endLine.toLong - 1L,
                    column = endColumn.toLong - 1L
                  )
                )
              )
            ),
            severity = DiagnosticSeverity.ERROR,
            source = SOURCE,
            message = message
          )
        )

      def default_validation(source: String, message: String): Seq[Diagnostic] = {
        val (lastLine, lastColumn) = source.foldLeft((0, 0)) {
          case ((line, column), char) =>
            char match {
              case '\n' => (line + 1, 0)
              case _    => (line, column + 1)
            }
        }
        validation(0, 0, lastLine, lastColumn, message)
      }

      def validate(source: String): F[ValidateResponse] =
        Sync[F]
          .attempt(
            Compiler[F]
              .sourceToADT(source, Map.empty[String, Par])
          )
          .flatMap {
            case Left(er) =>
              er match {
                case ie: InterpreterError =>
                  Sync[F].delay(
                    ValidateResponse(
                      result = ValidateResponse.Result.Success(
                        DiagnosticList(
                          diagnostics = ie match {
                            case UnboundVariableRef(varName, line, column) =>
                              validation(line, column, line, column + varName.length, er.getMessage)
                            case UnexpectedNameContext(
                                varName,
                                _,
                                SourcePosition(line, column)
                                ) =>
                              validation(line, column, line, column + varName.length, er.getMessage)
                            case UnexpectedReuseOfNameContextFree(
                                varName,
                                _,
                                SourcePosition(line, column)
                                ) =>
                              validation(line, column, line, column + varName.length, er.getMessage)
                            case UnexpectedProcContext(
                                varName,
                                _,
                                SourcePosition(line, column)
                                ) =>
                              validation(line, column, line, column + varName.length, er.getMessage)
                            case UnexpectedReuseOfProcContextFree(
                                varName,
                                _,
                                SourcePosition(line, column)
                                ) =>
                              validation(line, column, line, column + varName.length, er.getMessage)
                            case ReceiveOnSameChannelsError(line, column) =>
                              validation(line, column, line, column, er.getMessage)
                            case SyntaxError(message) =>
                              message match {
                                case RE_SYNTAX_ERROR(startLine, startColumn, endLine, endColumn) =>
                                  validation(
                                    Integer.parseInt(startLine),
                                    Integer.parseInt(startColumn),
                                    Integer.parseInt(endLine),
                                    Integer.parseInt(endColumn),
                                    er.getMessage
                                  )
                                case _ => default_validation(source, er.getMessage)
                              }
                            case LexerError(message) =>
                              message match {
                                case RE_LEXER_ERROR(lineStr, columnStr) => {
                                  val line   = Integer.parseInt(lineStr)
                                  val column = Integer.parseInt(columnStr)
                                  validation(line, column, line, column, er.getMessage)
                                }
                                case _ => default_validation(source, er.getMessage)
                              }
                            case _ => default_validation(source, er.getMessage)
                          }
                        )
                      )
                    )
                  )
                case th: Throwable =>
                  Sync[F].delay(
                    ValidateResponse(
                      result = ValidateResponse.Result.Error(
                        {
                          val sw = new StringWriter();
                          th.printStackTrace(new PrintWriter(sw));
                          s"${th.getMessage()}\n$sw"
                        }
                      )
                    )
                  )
              }
            case Right(_) =>
              Sync[F].delay(
                ValidateResponse(
                  result = ValidateResponse.Result.Success(
                    DiagnosticList(
                      diagnostics = Seq()
                    )
                  )
                )
              )
          }

      private def defer[A](task: F[A]): Task[A] =
        task.toTask.executeOn(worker)

      def validate(request: ValidateRequest): Task[ValidateResponse] =
        defer(validate(request.text))
    }
}
