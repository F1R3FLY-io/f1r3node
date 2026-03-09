package coop.rchain.rholang.interpreter.compiler.normalizer

import cats.MonadError
import cats.syntax.all._
import coop.rchain.models.Expr
import coop.rchain.models.Expr.ExprInstance.{GInt, GString, GUri}
import coop.rchain.rholang.ast.rholang_mercury.Absyn.{
  GroundBool,
  GroundInt,
  GroundString,
  GroundUri,
  Ground => AbsynGround
}
import coop.rchain.rholang.interpreter.errors.NormalizerError

import scala.util.Try

object GroundNormalizeMatcher {
  def normalizeMatch[F[_]](g: AbsynGround)(implicit M: MonadError[F, Throwable]): F[Expr] =
    g match {
      case gb: GroundBool => Expr(BoolNormalizeMatcher.normalizeMatch(gb.boolliteral_)).pure[F]
      case gi: GroundInt =>
        M.fromTry(
            Try(gi.longliteral_.toLong).adaptError {
              case e: NumberFormatException => NormalizerError(e.getMessage)
            }
          )
          .map { long =>
            Expr(GInt(long))
          }
      case gs: GroundString => Expr(GString(stripString(gs.stringliteral_))).pure[F]
      case gu: GroundUri    => Expr(GUri(stripUri(gu.uriliteral_))).pure[F]
    }
  // This is necessary to remove the backticks. We don't use a regular
  // expression because they're always there.
  def stripUri(raw: String): String = {
    require(raw.length >= 2)
    raw.substring(1, raw.length - 1)
  }
  // Similarly, we need to remove quotes from strings, since we are using
  // a custom string token
  def stripString(raw: String): String = {
    require(raw.length >= 2)
    val unquoted = raw.substring(1, raw.length - 1)
    unescapeString(unquoted)
  }

  // Process escape sequences in strings
  private def unescapeString(s: String): String = {
    val sb = new StringBuilder()
    var i  = 0
    while (i < s.length) {
      if (s(i) == '\\' && i + 1 <= s.length) {
        s(i + 1) match {
          case '"'  => sb.append('"'); i += 2
          case '\\' => sb.append('\\'); i += 2
          case 'n'  => sb.append('\n'); i += 2
          case 't'  => sb.append('\t'); i += 2
          case 'r'  => sb.append('\r'); i += 2
          case 'b'  => sb.append('\b'); i += 2
          case 'f'  => sb.append('\f'); i += 2
          case 'u' if i + 5 < s.length =>
            try {
              val hexCode   = s.substring(i + 2, i + 6)
              val codePoint = Integer.parseInt(hexCode, 16)
              sb.append(codePoint.toChar)
              i += 6
            } catch {
              case _: NumberFormatException =>
                sb.append(s(i))
                i += 1
            }
          case _ =>
            // For any unrecognized escape sequence, treat the backslash as literal
            // and append both characters to preserve original formatting
            sb.append(s(i))
            i += 1
        }
      } else {
        sb.append(s(i))
        i += 1
      }
    }
    sb.toString()
  }
}
