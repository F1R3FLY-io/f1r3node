package coop.rchain.rholang.interpreter.util
import coop.rchain.crypto.PublicKey
import coop.rchain.models.{GPrivate, Validator}
import coop.rchain.shared.Base16

final case class VaultAddress(address: Address) {

  def toBase58: String = address.toBase58
}

object VaultAddress {

  private val coinId  = "000000"
  private val version = "00"
  private val prefix  = Base16.unsafeDecode(coinId + version)

  private val tools = new AddressTools(prefix, keyLength = Validator.Length, checksumLength = 4)

  def fromDeployerId(deployerId: Array[Byte]): Option[VaultAddress] =
    fromPublicKey(PublicKey(deployerId))

  def fromPublicKey(pk: PublicKey): Option[VaultAddress] =
    tools.fromPublicKey(pk).map(VaultAddress(_))

  def fromEthAddress(ethAddress: String): Option[VaultAddress] =
    tools.fromEthAddress(ethAddress).map(VaultAddress(_))

  def fromUnforgeable(gprivate: GPrivate): VaultAddress =
    VaultAddress(tools.fromUnforgeable(gprivate))

  def parse(address: String): Either[String, VaultAddress] =
    tools.parse(address).map(VaultAddress(_))

  def isValid(address: String): Boolean = parse(address).isRight
}
