import $ivy.`org.typelevel::cats-effect:2.5.4`
import $ivy.`org.bitcoin-s::bitcoin-s-crypto:1.9.3`
import $ivy.`org.scorexfoundation::scrypto:2.2.1`
import $ivy.`com.lihaoyi::os-lib:0.9.2`
import $ivy.`com.beachape::enumeratum:1.5.13`
import $ivy.`com.thesamet.scalapb::scalapb-json4s:0.11.0`
import $ivy.`io.circe::circe-literal:0.14.6`
import $ivy.`io.circe::circe-parser:0.14.6`
import $ivy.`org.typelevel::jawn-parser:1.5.1`
import $ivy.`org.scodec::scodec-core:1.11.7`

// Assumption: the .protos have generated Scala and been compiled
import $cp.^.shared.target.`scala-2.12`.classes
import $cp.^.crypto.target.`scala-2.12`.classes
import $cp.^.models.target.`scala-2.12`.classes

@

/* bitcoin-s-crypto for the Secp256k1 elliptic curve cipher
 * scrypto for Blake2b
 * os-lib for filesystem manipulation
 * scalapb-json4s hauls in json4s, which we use for to-ing and fro-ing JSON
 * circe-literal is for making it easy to use JSON in the Ammonite REPL
 */

import coop.rchain.shared.Serialize
import coop.rchain.casper.protocol._
import coop.rchain.crypto.util.CertificateHelper._
import org.bitcoins.crypto._
import scorex.crypto.hash.Blake2b256
import scodec.bits._
import com.google.protobuf.ByteString
import scalapb.json4s.JsonFormat
import io.circe._
import io.circe.literal._
import io.circe.parser.parse
import io.circe.syntax._
  
def signDeploy(privKey: ECPrivateKey, deploy: DeployDataProto): DeployDataProto = {
  // Take a projection of only the fields used to validate the signature
  val projection   = DeployDataProto(
    term                  = deploy.term,
    timestamp             = deploy.timestamp,
    phloPrice             = deploy.phloPrice,
    phloLimit             = deploy.phloLimit,
    validAfterBlockNumber = deploy.validAfterBlockNumber,
    shardId               = deploy.shardId 
  )
  val serialized   = ByteVector(projection.toByteArray)
  val deployer     = privKey.publicKey.decompressed
  val digest       = ByteVector(Blake2b256.hash(serialized.toArray))
  val signed       = privKey.sign(digest)
  projection.copy(
    sigAlgorithm   = "secp256k1",
    sig            = ByteString.copyFrom(signed.bytes.toArray),
    deployer       = ByteString.copyFrom(deployer.decompressedBytes.toArray)
  )
}

// For use with gprcurl, which uses Protobuf's JSON conversions
def signDeployJSON(privKey: String, deploy: DeployDataProto): Unit =
  println(JsonFormat.toJsonString(
    signDeploy(
      ECPrivateKey(privKey),
      deploy
    )
  ))

// For use with the "web" API
def signDeployWeb(privKey: String, deploy: DeployDataProto): Unit= {
  val pb = signDeploy(
    ECPrivateKey(privKey),
    deploy
  )

  val data = json"""
  {
    "term":                  ${pb.term},
    "timestamp":             ${pb.timestamp},
    "phloPrice":             ${pb.phloPrice},
    "phloLimit":             ${pb.phloLimit},
    "validAfterBlockNumber": ${pb.validAfterBlockNumber},
    "shardId":               ${pb.shardId}
  }
  """

  val deployerHex = ByteVector(pb.deployer.toByteArray).toHex.asJson
  val sigHex      = ByteVector(pb.sig.toByteArray).toHex.asJson

  println(json"""
  {
    "data":         $data,
    "sigAlgorithm": "secp256k1",
    "signature":    $sigHex,
    "deployer":     $deployerHex
  }
  """.noSpaces)
}

// Needed for creating keys for Helm Chart values
def generateKeys(n: Int) = {
  println(Seq.fill(n) {
    val key = ECPrivateKey.freshPrivateKey
    s"""|  - publicKey: ${key.publicKey.decompressedHex}
        |    privateKey: ${key.hex}
        |""".stripMargin
  }.mkString("\n"))
}
