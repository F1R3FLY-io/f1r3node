package coop.rchain.rholang.interpreter.util

import coop.rchain.crypto.PublicKey
import coop.rchain.shared.Base16
import org.scalatest._

class VaultAddressSpec extends FlatSpec with Matchers {
  "fromPublicKey" should "work correctly" in {
    val pk =
      PublicKey(
        Base16.unsafeDecode(
          "00322ba649cebf90d8bd0eeb0658ea7957bcc59ecee0676c86f4fec517c062510031122ba649cebf90d8bd0eeb0658ea7957bcc59ecee0676c86f4fec517c06251"
        )
      )
    VaultAddress.fromPublicKey(pk).map(_.toBase58) should be(
      Some("1111eEt66obfqvEcKeD3ajeupzsfj6PdSQccNNaRmCoy6Dhv1wM9E")
    )
  }

  "fromEthAddress" should "work correctly without a prefix" in {
    val ethAddress = "06a441c277bf454c5d159b0e5bdafca69b296733"
    VaultAddress.fromEthAddress(ethAddress).map(_.toBase58) should be(
      Some("1111Gzo7ywxbcXVumSS9Lzd8JBAqnF1zniNszvMLHQ2APa3dzs2rG")
    )
  }

  "fromEthAddress" should "work correctly with a prefix" in {
    val ethAddress = "0x06a441c277bf454c5d159b0e5bdafca69b296733"
    VaultAddress.fromEthAddress(ethAddress).map(_.toBase58) should be(
      Some("1111Gzo7ywxbcXVumSS9Lzd8JBAqnF1zniNszvMLHQ2APa3dzs2rG")
    )
  }

  "fromEthAddress" should "fail when wrong prefix" in {
    val ethAddress = "1x06a441c277bf454c5d159b0e5bdafca69b296733"
    VaultAddress.fromEthAddress(ethAddress).map(_.toBase58) should be(None)
  }

  "fromEthAddress" should "fail when wrong length" in {
    val ethAddress = "0x06"
    VaultAddress.fromEthAddress(ethAddress).map(_.toBase58) should be(None)
  }

  // Use the known-good address derived from the 65-byte public key in the first test
  val knownValidAddress = "1111eEt66obfqvEcKeD3ajeupzsfj6PdSQccNNaRmCoy6Dhv1wM9E"
  // Same address with last char changed to break the checksum
  val knownInvalidAddress = "1111eEt66obfqvEcKeD3ajeupzsfj6PdSQccNNaRmCoy6Dhv1wM9F"

  "VaultAddress" should "be able to be parsed from a valid string" in {
    val result = VaultAddress.parse(knownValidAddress)
    result.isRight should be(true)
    result.right.get.toBase58 should be(knownValidAddress)
  }

  it should "not be able to be parsed from an invalid string" in {
    val result = VaultAddress.parse(knownInvalidAddress)
    result.isLeft should be(true)
  }

  it should "be valid for a valid address" in {
    VaultAddress.isValid(knownValidAddress) should be(true)
  }

  it should "be invalid for an invalid address" in {
    VaultAddress.isValid(knownInvalidAddress) should be(false)
  }

  it should "be able to be created from a public key" in {
    val pk = PublicKey(
      Base16.unsafeDecode(
        "00322ba649cebf90d8bd0eeb0658ea7957bcc59ecee0676c86f4fec517c062510031122ba649cebf90d8bd0eeb0658ea7957bcc59ecee0676c86f4fec517c06251"
      )
    )
    val result = VaultAddress.fromPublicKey(pk)
    result should be(defined)
    result.get.toBase58 should be(knownValidAddress)
  }
}
