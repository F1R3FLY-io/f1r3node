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
  AggregateError,
  InterpreterError,
  LexerError,
  ReceiveOnSameChannelsError,
  SyntaxError,
  TopLevelFreeVariablesNotAllowedError,
  TopLevelLogicalConnectivesNotAllowedError,
  TopLevelWildcardsNotAllowedError,
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
  val SOURCE: String                  = "rholang"
  val RE_SYNTAX_ERROR: Regex          = """syntax error\([^)]*\): .* at (\d+):(\d+)-(\d+):(\d+)""".r
  val RE_LEXER_ERROR: Regex           = """.* at (\d+):(\d+)""".r
  val RE_TOP_LEVEL_FREE_VARS: Regex   = """(\w+) at (\d+):(\d+)""".r
  val RE_TOP_LEVEL_WILDCARDS: Regex   = """_ \(wildcard\) at (\d+):(\d+)""".r
  val RE_TOP_LEVEL_CONNECTIVES: Regex = """([^ ]+) \(([^)]+)\) at (\d+):(\d+)""".r

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
        validation(1, 1, lastLine + 1, lastColumn + 1, message)
      }

      private def parseTopLevelFreeVars(message: String): Seq[(String, Int, Int)] = {
        val items = message.split(", ")
        items.flatMap { item =>
          item match {
            case RE_TOP_LEVEL_FREE_VARS(varName, lineStr, colStr) =>
              Some((varName, lineStr.toInt, colStr.toInt))
            case _ => None
          }
        }
      }

      private def parseTopLevelWildcards(message: String): Seq[(Int, Int)] = {
        val items = message.split(", ")
        items.flatMap { item =>
          item match {
            case RE_TOP_LEVEL_WILDCARDS(lineStr, colStr) =>
              Some((lineStr.toInt, colStr.toInt))
            case _ => None
          }
        }
      }

      private def parseTopLevelConnectives(message: String): Seq[(String, String, Int, Int)] = {
        val items = message.split(", ")
        items.flatMap { item =>
          item match {
            case RE_TOP_LEVEL_CONNECTIVES(connType, connDesc, lineStr, colStr) =>
              Some((connType, connDesc, lineStr.toInt, colStr.toInt))
            case _ => None
          }
        }
      }

      def errorToDiagnostics(error: InterpreterError, source: String): Seq[Diagnostic] =
        error match {
          case UnboundVariableRef(varName, line, column) =>
            validation(line, column, line, column + varName.length, error.getMessage)
          case UnexpectedNameContext(varName, _, SourcePosition(line, column)) =>
            validation(line, column, line, column + varName.length, error.getMessage)
          case UnexpectedReuseOfNameContextFree(varName, _, SourcePosition(line, column)) =>
            validation(line, column, line, column + varName.length, error.getMessage)
          case UnexpectedProcContext(varName, _, SourcePosition(line, column)) =>
            validation(line, column, line, column + varName.length, error.getMessage)
          case UnexpectedReuseOfProcContextFree(varName, _, SourcePosition(line, column)) =>
            validation(line, column, line, column + varName.length, error.getMessage)
          case ReceiveOnSameChannelsError(line, column) =>
            validation(line, column, line, column + 1, error.getMessage)
          case SyntaxError(message) =>
            message match {
              case RE_SYNTAX_ERROR(startLine, startColumn, endLine, endColumn) =>
                validation(
                  Integer.parseInt(startLine),
                  Integer.parseInt(startColumn),
                  Integer.parseInt(endLine),
                  Integer.parseInt(endColumn),
                  error.getMessage
                )
              case _ => default_validation(source, error.getMessage)
            }
          case LexerError(message) =>
            message match {
              case RE_LEXER_ERROR(lineStr, columnStr) =>
                val line   = Integer.parseInt(lineStr)
                val column = Integer.parseInt(columnStr)
                validation(line, column, line, column + 1, error.getMessage)
              case _ => default_validation(source, error.getMessage)
            }
          case TopLevelFreeVariablesNotAllowedError(message) =>
            val freeVars = parseTopLevelFreeVars(message)
            if (freeVars.nonEmpty) {
              freeVars.flatMap {
                case (varName, line, column) =>
                  val specificMessage =
                    s"Top level free variables are not allowed: $varName at $line:$column."
                  validation(line, column, line, column + varName.length, specificMessage)
              }
            } else {
              default_validation(source, message)
            }
          case TopLevelWildcardsNotAllowedError(message) =>
            val wildcards = parseTopLevelWildcards(message)
            if (wildcards.nonEmpty) {
              wildcards.flatMap {
                case (line, column) =>
                  val specificMessage =
                    s"Top level wildcards are not allowed: _ (wildcard) at $line:$column."
                  validation(line, column, line, column + 1, specificMessage)
              }
            } else {
              default_validation(source, error.getMessage)
            }
          case TopLevelLogicalConnectivesNotAllowedError(message) =>
            val connectives = parseTopLevelConnectives(message)
            if (connectives.nonEmpty) {
              connectives.flatMap {
                case (connType, connDesc, line, column) =>
                  val specificMessage =
                    s"Top level logical connectives are not allowed: $connType ($connDesc) at $line:$column."
                  validation(line, column, line, column + connType.length, specificMessage)
              }
            } else {
              default_validation(source, error.getMessage)
            }
          case AggregateError(interpreterErrors, errors) =>
            val interpreterDiagnostics = interpreterErrors.flatMap(errorToDiagnostics(_, source))
            val throwableDiagnostics = errors.map { th =>
              default_validation(source, s"Throwable error: ${th.getMessage}")
            }
            interpreterDiagnostics ++ throwableDiagnostics.flatten
          case _ =>
            default_validation(source, error.getMessage)
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
                          diagnostics = errorToDiagnostics(ie, source)
                        )
                      )
                    )
                  )
                case th: Throwable =>
                  Sync[F].delay(
                    ValidateResponse(
                      result = ValidateResponse.Result.Error(
                        {
                          val sw = new StringWriter()
                          th.printStackTrace(new PrintWriter(sw))
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
