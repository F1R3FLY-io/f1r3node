package coop.rchain.node.api

import com.google.protobuf.ByteString
import coop.rchain.casper.protocol.{BondInfo, JustificationInfo, LightBlockInfo}
import coop.rchain.node.api.WebApi.DeployLookupResponse
import org.scalatest.{FlatSpec, Matchers}

class DeployLookupResponseSpec extends FlatSpec with Matchers {

  val sampleLightBlockInfo: LightBlockInfo = LightBlockInfo(
    blockHash = "7bf8abc123",
    sender = "0487def456",
    seqNum = 17453,
    sig = "3044abcdef",
    sigAlgorithm = "secp256k1",
    shardId = "root",
    extraBytes = ByteString.EMPTY,
    version = 1,
    timestamp = 1770028092477L,
    headerExtraBytes = ByteString.EMPTY,
    parentsHashList = List("parent1hash", "parent2hash"),
    blockNumber = 52331,
    preStateHash = "preState123",
    postStateHash = "postState456",
    bodyExtraBytes = ByteString.EMPTY,
    bonds = List(BondInfo("validator1", 100), BondInfo("validator2", 200)),
    blockSize = "4096",
    deployCount = 5,
    faultTolerance = 0.5f,
    justifications = List(JustificationInfo("validator1", "latestBlockHash1")),
    rejectedDeploys = List.empty
  )

  "DeployLookupResponse.fromLightBlockInfo" should "extract only the minimal fields" in {
    val result = DeployLookupResponse.fromLightBlockInfo(sampleLightBlockInfo)

    result.blockHash should be("7bf8abc123")
    result.blockNumber should be(52331)
    result.timestamp should be(1770028092477L)
    result.sender should be("0487def456")
    result.seqNum should be(17453)
    result.sig should be("3044abcdef")
    result.sigAlgorithm should be("secp256k1")
    result.shardId should be("root")
    result.version should be(1)
  }

  it should "not contain block-level fields like bonds, justifications, parentsHashList" in {
    val result = DeployLookupResponse.fromLightBlockInfo(sampleLightBlockInfo)

    // Verify the case class only has the 9 expected fields
    // by checking the product arity (case classes extend Product)
    result.productArity should be(9)
  }

  it should "handle empty string fields" in {
    val emptyInfo = sampleLightBlockInfo.copy(
      blockHash = "",
      sender = "",
      sig = "",
      sigAlgorithm = "",
      shardId = ""
    )
    val result = DeployLookupResponse.fromLightBlockInfo(emptyInfo)

    result.blockHash should be("")
    result.sender should be("")
    result.sig should be("")
    result.sigAlgorithm should be("")
    result.shardId should be("")
  }

  it should "handle zero numeric fields" in {
    val zeroInfo = sampleLightBlockInfo.copy(
      blockNumber = 0,
      timestamp = 0,
      seqNum = 0,
      version = 0
    )
    val result = DeployLookupResponse.fromLightBlockInfo(zeroInfo)

    result.blockNumber should be(0)
    result.timestamp should be(0)
    result.seqNum should be(0)
    result.version should be(0)
  }

  it should "produce correct result regardless of bonds list size" in {
    val manyBonds = (1 to 1000).map(i => BondInfo(s"validator$i", i.toLong)).toList
    val largeInfo = sampleLightBlockInfo.copy(bonds = manyBonds)
    val result    = DeployLookupResponse.fromLightBlockInfo(largeInfo)

    // The response should be the same regardless of bonds size
    result.blockHash should be(sampleLightBlockInfo.blockHash)
    result.blockNumber should be(sampleLightBlockInfo.blockNumber)
  }
}
