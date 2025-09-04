package coop.rchain.rholang.interpreter.util
import coop.rchain.crypto.PublicKey
import coop.rchain.models.{GPrivate, Validator}
import coop.rchain.shared.Base16

final case class ASIAddress(address: Address) {

  def toBase58: String = address.toBase58
}

object ASIAddress {

  private val coinId  = "000000"
  private val version = "00"
  private val prefix  = Base16.unsafeDecode(coinId + version)

  private val tools = new AddressTools(prefix, keyLength = Validator.Length, checksumLength = 4)

  def fromDeployerId(deployerId: Array[Byte]): Option[ASIAddress] =
    fromPublicKey(PublicKey(deployerId))

  def fromPublicKey(pk: PublicKey): Option[ASIAddress] =
    tools.fromPublicKey(pk).map(ASIAddress(_))

  def fromEthAddress(ethAddress: String): Option[ASIAddress] =
    tools.fromEthAddress(ethAddress).map(ASIAddress(_))

  def fromUnforgeable(gprivate: GPrivate): ASIAddress =
    ASIAddress(tools.fromUnforgeable(gprivate))

  def parse(address: String): Either[String, ASIAddress] =
    tools.parse(address).map(ASIAddress(_))

  def isValid(address: String): Boolean = parse(address).isRight
}
