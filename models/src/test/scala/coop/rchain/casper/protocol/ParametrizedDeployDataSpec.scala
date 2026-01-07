package coop.rchain.casper.protocol

import com.google.protobuf.ByteString
import coop.rchain.crypto.PrivateKey
import coop.rchain.crypto.signatures.{Secp256k1, Signed}
import org.scalacheck.Arbitrary
import org.scalacheck.Arbitrary._
import org.scalacheck.Gen
import org.scalatest.prop.GeneratorDrivenPropertyChecks
import org.scalatest.{FlatSpec, Matchers, PropSpec}

/**
  * Tests for DeployData with parameters (formerly ParametrizedDeployData).
  * DeployData now supports parameters natively.
  */
class DeployDataWithParametersSpec
    extends PropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers {

  // Generator for RholangValueData
  val rholangValueGen: Gen[RholangValueData] = Gen.oneOf(
    arbBool.arbitrary.map(RholangValueData.BoolValue),
    arbLong.arbitrary.map(RholangValueData.IntValue),
    arbString.arbitrary.map(RholangValueData.StringValue),
    Gen
      .listOf(arbByte.arbitrary)
      .map(bytes => RholangValueData.BytesValue(ByteString.copyFrom(bytes.toArray)))
  )

  // Generator for DeployParameterData - names must start with letter/underscore
  val validParamNameGen: Gen[String] = for {
    firstChar <- Gen.alphaChar
    restChars <- Gen.alphaNumStr
  } yield s"$firstChar$restChars"

  val deployParameterGen: Gen[DeployParameterData] = for {
    name  <- validParamNameGen.suchThat(_.nonEmpty)
    value <- rholangValueGen
  } yield DeployParameterData(name, value)

  // Generator for DeployData with parameters
  implicit val deployDataWithParamsArb: Arbitrary[DeployData] = Arbitrary(
    for {
      term                  <- arbString.arbitrary
      timestamp             <- arbLong.arbitrary
      phloPrice             <- arbLong.arbitrary
      phloLimit             <- arbLong.arbitrary
      validAfterBlockNumber <- arbLong.arbitrary
      shardId               <- arbString.arbitrary
      parameters            <- Gen.listOfN(3, deployParameterGen)
    } yield DeployData(
      term,
      timestamp,
      phloPrice,
      phloLimit,
      validAfterBlockNumber,
      shardId,
      parameters
    )
  )

  property("DeployData with parameters serialization roundtrip works fine") {
    forAll { (dd: DeployData) =>
      val encoded = DeployData.serialize.encode(dd)
      val decoded = DeployData.serialize.decode(encoded)
      decoded should be(Right(dd))
    }
  }

  property("RholangValueData BoolValue serializes correctly") {
    forAll { (value: Boolean) =>
      val rv    = RholangValueData.BoolValue(value)
      val proto = RholangValueData.toProto(rv)
      proto.value.isBoolValue should be(true)
      proto.getBoolValue should be(value)
    }
  }

  property("RholangValueData IntValue serializes correctly") {
    forAll { (value: Long) =>
      val rv    = RholangValueData.IntValue(value)
      val proto = RholangValueData.toProto(rv)
      proto.value.isIntValue should be(true)
      proto.getIntValue should be(value)
    }
  }

  property("RholangValueData StringValue serializes correctly") {
    forAll { (value: String) =>
      val rv    = RholangValueData.StringValue(value)
      val proto = RholangValueData.toProto(rv)
      proto.value.isStringValue should be(true)
      proto.getStringValue should be(value)
    }
  }

  property("RholangValueData BytesValue serializes correctly") {
    forAll { (bytes: Array[Byte]) =>
      val rv    = RholangValueData.BytesValue(ByteString.copyFrom(bytes))
      val proto = RholangValueData.toProto(rv)
      proto.value.isBytesValue should be(true)
      proto.getBytesValue.toByteArray should be(bytes)
    }
  }

  property("RholangValueData roundtrip through proto works") {
    forAll(rholangValueGen) { (rv: RholangValueData) =>
      val proto   = RholangValueData.toProto(rv)
      val decoded = RholangValueData.fromProto(proto)
      decoded should be(Right(rv))
    }
  }

  property("DeployParameterData roundtrip through proto works") {
    forAll(deployParameterGen) { (dp: DeployParameterData) =>
      val proto   = DeployParameterData.toProto(dp)
      val decoded = DeployParameterData.fromProto(proto)
      decoded should be(Right(dp))
    }
  }
}

class DeployDataWithParametersSignatureSpec extends FlatSpec with Matchers {

  "DeployData.from" should "accept valid signatures for deploys with parameters" in {
    val privateKey = PrivateKey(
      coop.rchain.shared.Base16.unsafeDecode(
        "a68a6e6cca30f81bd24a719f3145d20e8424bd7b396309b0708a16c7d8000b76"
      )
    )

    val deployData = DeployData(
      term = """new x(`rho:deploy:param:myParam`) in { @"output"!(x) }""",
      timestamp = System.currentTimeMillis(),
      phloPrice = 1L,
      phloLimit = 100000L,
      validAfterBlockNumber = 0L,
      shardId = "root",
      parameters = Seq(
        DeployParameterData("myParam", RholangValueData.IntValue(42L)),
        DeployParameterData("myString", RholangValueData.StringValue("hello")),
        DeployParameterData(
          "myBytes",
          RholangValueData.BytesValue(ByteString.copyFromUtf8("binary data"))
        )
      )
    )

    // Sign the deploy
    val signed = Signed(deployData, Secp256k1, privateKey)

    // Convert to proto and back
    val proto  = DeployData.toProto(signed)
    val result = DeployData.from(proto)

    result.isRight should be(true)
    result.right.get.data should be(deployData)
    result.right.get.sig should be(signed.sig)
  }

  it should "reject invalid signatures" in {
    val privateKey = PrivateKey(
      coop.rchain.shared.Base16.unsafeDecode(
        "a68a6e6cca30f81bd24a719f3145d20e8424bd7b396309b0708a16c7d8000b76"
      )
    )

    val deployData = DeployData(
      term = "Nil",
      timestamp = 12345L,
      phloPrice = 1L,
      phloLimit = 100000L,
      validAfterBlockNumber = 0L,
      shardId = "root",
      parameters = Seq.empty
    )

    // Sign with one key
    val signed = Signed(deployData, Secp256k1, privateKey)
    val proto  = DeployData.toProto(signed)

    // Tamper with the term (invalidates signature)
    val tamperedProto = proto.withTerm("@0!(1)")

    val result = DeployData.from(tamperedProto)
    result.isLeft should be(true)
    result.left.get should include("signature")
  }

  it should "handle empty parameters list" in {
    val privateKey = PrivateKey(
      coop.rchain.shared.Base16.unsafeDecode(
        "a68a6e6cca30f81bd24a719f3145d20e8424bd7b396309b0708a16c7d8000b76"
      )
    )

    val deployData = DeployData(
      term = "Nil",
      timestamp = 12345L,
      phloPrice = 1L,
      phloLimit = 100000L,
      validAfterBlockNumber = 0L,
      shardId = "root",
      parameters = Seq.empty
    )

    val signed = Signed(deployData, Secp256k1, privateKey)
    val proto  = DeployData.toProto(signed)
    val result = DeployData.from(proto)

    result.isRight should be(true)
    result.right.get.data.parameters should be(empty)
  }

  it should "reject deploy with parameter missing a value" in {
    val privateKey = PrivateKey(
      coop.rchain.shared.Base16.unsafeDecode(
        "a68a6e6cca30f81bd24a719f3145d20e8424bd7b396309b0708a16c7d8000b76"
      )
    )

    val deployData = DeployData(
      term = "Nil",
      timestamp = 12345L,
      phloPrice = 1L,
      phloLimit = 100000L,
      validAfterBlockNumber = 0L,
      shardId = "root",
      parameters = Seq.empty
    )

    val signed = Signed(deployData, Secp256k1, privateKey)
    val proto  = DeployData.toProto(signed)

    // Manually add a parameter without a value
    val protoWithBadParam = proto.addParameters(DeployParameter("badParam", None))

    val result = DeployData.from(protoWithBadParam)
    result.isLeft should be(true)
    result.left.get should include("badParam")
    result.left.get should include("missing a value")
  }

  it should "reject deploy with duplicate parameter names" in {
    val privateKey = PrivateKey(
      coop.rchain.shared.Base16.unsafeDecode(
        "a68a6e6cca30f81bd24a719f3145d20e8424bd7b396309b0708a16c7d8000b76"
      )
    )

    val deployData = DeployData(
      term = "Nil",
      timestamp = 12345L,
      phloPrice = 1L,
      phloLimit = 100000L,
      validAfterBlockNumber = 0L,
      shardId = "root",
      parameters = Seq(
        DeployParameterData("duplicate", RholangValueData.IntValue(1L)),
        DeployParameterData("unique", RholangValueData.IntValue(2L)),
        DeployParameterData("duplicate", RholangValueData.IntValue(3L))
      )
    )

    val signed = Signed(deployData, Secp256k1, privateKey)
    val proto  = DeployData.toProto(signed)
    val result = DeployData.from(proto)

    result.isLeft should be(true)
    result.left.get should include("Duplicate parameter names")
    result.left.get should include("duplicate")
  }

  it should "reject deploy exceeding maximum parameter count" in {
    val privateKey = PrivateKey(
      coop.rchain.shared.Base16.unsafeDecode(
        "a68a6e6cca30f81bd24a719f3145d20e8424bd7b396309b0708a16c7d8000b76"
      )
    )

    // Create more than MaxParameters (50) parameters
    val tooManyParams = (1 to (DeployParameterData.MaxParameters + 1)).map { i =>
      DeployParameterData(s"param$i", RholangValueData.IntValue(i.toLong))
    }

    val deployData = DeployData(
      term = "Nil",
      timestamp = 12345L,
      phloPrice = 1L,
      phloLimit = 100000L,
      validAfterBlockNumber = 0L,
      shardId = "root",
      parameters = tooManyParams
    )

    val signed = Signed(deployData, Secp256k1, privateKey)
    val proto  = DeployData.toProto(signed)
    val result = DeployData.from(proto)

    result.isLeft should be(true)
    result.left.get should include("Too many parameters")
    result.left.get should include(s"${DeployParameterData.MaxParameters + 1}")
    result.left.get should include(s"${DeployParameterData.MaxParameters}")
  }

  it should "accept deploy at exactly the maximum parameter count" in {
    val privateKey = PrivateKey(
      coop.rchain.shared.Base16.unsafeDecode(
        "a68a6e6cca30f81bd24a719f3145d20e8424bd7b396309b0708a16c7d8000b76"
      )
    )

    // Create exactly MaxParameters (50) parameters
    val maxParams = (1 to DeployParameterData.MaxParameters).map { i =>
      DeployParameterData(s"param$i", RholangValueData.IntValue(i.toLong))
    }

    val deployData = DeployData(
      term = "Nil",
      timestamp = 12345L,
      phloPrice = 1L,
      phloLimit = 100000L,
      validAfterBlockNumber = 0L,
      shardId = "root",
      parameters = maxParams
    )

    val signed = Signed(deployData, Secp256k1, privateKey)
    val proto  = DeployData.toProto(signed)
    val result = DeployData.from(proto)

    result.isRight should be(true)
    result.right.get.data.parameters.size should be(DeployParameterData.MaxParameters)
  }
}
