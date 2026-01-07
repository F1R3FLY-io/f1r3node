package coop.rchain.models

import com.google.protobuf.ByteString
import coop.rchain.casper.protocol.{DeployData, DeployParameterData, RholangValueData}
import coop.rchain.crypto.PrivateKey
import coop.rchain.crypto.signatures.{Secp256k1, Signed}
import coop.rchain.models.Expr.ExprInstance.{GBool, GByteArray, GInt, GString}
import coop.rchain.models.rholang.implicits._
import coop.rchain.shared.Base16
import org.scalatest.{FlatSpec, Matchers}

/**
  * Tests for NormalizerEnv with DeployData that contains parameters.
  * DeployData now supports parameters natively (consolidated from ParametrizedDeployData).
  */
class NormalizerEnvParametrizedSpec extends FlatSpec with Matchers {

  val testPrivateKey: PrivateKey = PrivateKey(
    Base16.unsafeDecode("a68a6e6cca30f81bd24a719f3145d20e8424bd7b396309b0708a16c7d8000b76")
  )

  // Fixed timestamp for deterministic tests: 2024-01-01 00:00:00 UTC
  val fixedTimestamp: Long = 1704067200000L

  def createSignedDeployWithParams(
      term: String,
      parameters: Seq[DeployParameterData]
  ): Signed[DeployData] = {
    val deployData = DeployData(
      term = term,
      timestamp = fixedTimestamp,
      phloPrice = 1L,
      phloLimit = 100000L,
      validAfterBlockNumber = 0L,
      shardId = "root",
      parameters = parameters
    )
    Signed(deployData, Secp256k1, testPrivateKey)
  }

  "NormalizerEnv for DeployData with parameters" should "include standard deploy environment keys" in {
    val signedDeploy = createSignedDeployWithParams("Nil", Seq.empty)
    val env          = NormalizerEnv(signedDeploy)
    val envMap       = env.toEnv

    // Check that standard keys are present
    envMap.contains("rho:rchain:deployId") should be(true)
    envMap.contains("rho:rchain:deployerId") should be(true)
  }

  it should "include parameters with rho:deploy:param: prefix" in {
    val parameters = Seq(
      DeployParameterData("myInt", RholangValueData.IntValue(42L)),
      DeployParameterData("myString", RholangValueData.StringValue("hello"))
    )
    val signedDeploy = createSignedDeployWithParams("Nil", parameters)
    val env          = NormalizerEnv(signedDeploy)
    val envMap       = env.toEnv

    // Check that parameters are present with correct prefix
    envMap.contains("rho:deploy:param:myInt") should be(true)
    envMap.contains("rho:deploy:param:myString") should be(true)
  }

  it should "not include parameters without the prefix" in {
    val parameters = Seq(
      DeployParameterData("testParam", RholangValueData.BoolValue(true))
    )
    val signedDeploy = createSignedDeployWithParams("Nil", parameters)
    val env          = NormalizerEnv(signedDeploy)
    val envMap       = env.toEnv

    // The raw name should not be present, only the prefixed version
    envMap.contains("testParam") should be(false)
    envMap.contains("rho:deploy:param:testParam") should be(true)
  }

  it should "handle empty parameters list" in {
    val signedDeploy = createSignedDeployWithParams("Nil", Seq.empty)
    val env          = NormalizerEnv(signedDeploy)
    val envMap       = env.toEnv

    // Only standard keys should be present
    envMap.size should be(2)
    envMap.contains("rho:rchain:deployId") should be(true)
    envMap.contains("rho:rchain:deployerId") should be(true)
  }

  "rholangValueToPar" should "convert BoolValue to Par with GBool" in {
    val boolValue = RholangValueData.BoolValue(true)
    val par       = NormalizerEnv.rholangValueToPar(boolValue)

    par.exprs should have size 1
    par.exprs.head.exprInstance shouldBe a[GBool]
    par.exprs.head.exprInstance.asInstanceOf[GBool].value should be(true)
  }

  it should "convert IntValue to Par with GInt" in {
    val intValue = RholangValueData.IntValue(12345L)
    val par      = NormalizerEnv.rholangValueToPar(intValue)

    par.exprs should have size 1
    par.exprs.head.exprInstance shouldBe a[GInt]
    par.exprs.head.exprInstance.asInstanceOf[GInt].value should be(12345L)
  }

  it should "convert StringValue to Par with GString" in {
    val stringValue = RholangValueData.StringValue("test string")
    val par         = NormalizerEnv.rholangValueToPar(stringValue)

    par.exprs should have size 1
    par.exprs.head.exprInstance shouldBe a[GString]
    par.exprs.head.exprInstance.asInstanceOf[GString].value should be("test string")
  }

  it should "convert BytesValue to Par with GByteArray" in {
    val bytes      = ByteString.copyFromUtf8("binary data")
    val bytesValue = RholangValueData.BytesValue(bytes)
    val par        = NormalizerEnv.rholangValueToPar(bytesValue)

    par.exprs should have size 1
    par.exprs.head.exprInstance shouldBe a[GByteArray]
    par.exprs.head.exprInstance.asInstanceOf[GByteArray].value should be(bytes)
  }

  it should "handle negative integers" in {
    val intValue = RholangValueData.IntValue(-999L)
    val par      = NormalizerEnv.rholangValueToPar(intValue)

    par.exprs should have size 1
    par.exprs.head.exprInstance.asInstanceOf[GInt].value should be(-999L)
  }

  it should "handle empty string" in {
    val stringValue = RholangValueData.StringValue("")
    val par         = NormalizerEnv.rholangValueToPar(stringValue)

    par.exprs should have size 1
    par.exprs.head.exprInstance.asInstanceOf[GString].value should be("")
  }

  it should "handle empty bytes" in {
    val bytesValue = RholangValueData.BytesValue(ByteString.EMPTY)
    val par        = NormalizerEnv.rholangValueToPar(bytesValue)

    par.exprs should have size 1
    par.exprs.head.exprInstance.asInstanceOf[GByteArray].value should be(ByteString.EMPTY)
  }

  "DeployParameterPrefix" should "be rho:deploy:param:" in {
    NormalizerEnv.DeployParameterPrefix should be("rho:deploy:param:")
  }
}
