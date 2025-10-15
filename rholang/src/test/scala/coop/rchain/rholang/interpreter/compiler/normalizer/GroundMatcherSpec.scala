package coop.rchain.rholang.interpreter.compiler.normalizer

import coop.rchain.rholang.ast.rholang_mercury.Absyn._

import org.scalatest._

import coop.rchain.models.Expr.ExprInstance._
import coop.rchain.models._
import coop.rchain.models.rholang.implicits._
import monix.eval.Coeval

class GroundMatcherSpec extends FlatSpec with Matchers {
  "GroundInt" should "Compile as GInt" in {
    val gi                   = new GroundInt("7")
    val expectedResult: Expr = GInt(7)
    GroundNormalizeMatcher.normalizeMatch[Coeval](gi).value should be(expectedResult)
  }
  "GroundString" should "Compile as GString" in {
    val gs                   = new GroundString("\"String\"")
    val expectedResult: Expr = GString("String")
    GroundNormalizeMatcher.normalizeMatch[Coeval](gs).value should be(expectedResult)
  }

  "GroundString with escaped quote" should "properly unescape the quote" in {
    val gs                   = new GroundString("\"\\\"\"")
    val expectedResult: Expr = GString("\"")
    GroundNormalizeMatcher.normalizeMatch[Coeval](gs).value should be(expectedResult)
  }

  "GroundString with escaped backslash" should "properly unescape the backslash" in {
    val gs                   = new GroundString("\"\\\\\"")
    val expectedResult: Expr = GString("\\")
    GroundNormalizeMatcher.normalizeMatch[Coeval](gs).value should be(expectedResult)
  }

  "GroundString with escaped newline" should "properly unescape the newline" in {
    val gs                   = new GroundString("\"Hello\\nWorld\"")
    val expectedResult: Expr = GString("Hello\nWorld")
    GroundNormalizeMatcher.normalizeMatch[Coeval](gs).value should be(expectedResult)
  }

  "GroundString with escaped tab" should "properly unescape the tab" in {
    val gs                   = new GroundString("\"Hello\\tWorld\"")
    val expectedResult: Expr = GString("Hello\tWorld")
    GroundNormalizeMatcher.normalizeMatch[Coeval](gs).value should be(expectedResult)
  }

  "GroundString with multiple escape sequences" should "properly unescape all of them" in {
    val gs                   = new GroundString("\"Quote: \\\" Backslash: \\\\ Tab: \\t Newline: \\n\"")
    val expectedResult: Expr = GString("Quote: \" Backslash: \\ Tab: \t Newline: \n")
    GroundNormalizeMatcher.normalizeMatch[Coeval](gs).value should be(expectedResult)
  }

  "GroundUri" should "Compile as GUri" in {
    val gu                   = new GroundUri("`rho:uri`")
    val expectedResult: Expr = GUri("rho:uri")
    GroundNormalizeMatcher.normalizeMatch[Coeval](gu).value should be(expectedResult)
  }

  "GroundUri with escaped backtick" should "properly unescape the backtick" in {
    val gu                   = new GroundUri("`rho:\\`uri`")
    val expectedResult: Expr = GUri("rho:`uri")
    GroundNormalizeMatcher.normalizeMatch[Coeval](gu).value should be(expectedResult)
  }

  "GroundUri with escaped backslash" should "properly unescape the backslash" in {
    val gu                   = new GroundUri("`rho:\\\\uri`")
    val expectedResult: Expr = GUri("rho:\\uri")
    GroundNormalizeMatcher.normalizeMatch[Coeval](gu).value should be(expectedResult)
  }

  "GroundUri with multiple escape sequences" should "properly unescape all of them" in {
    val gu                   = new GroundUri("`rho:\\`io:\\\\std\\`out`")
    val expectedResult: Expr = GUri("rho:`io:\\std`out")
    GroundNormalizeMatcher.normalizeMatch[Coeval](gu).value should be(expectedResult)
  }
}
