package coop.rchain.models

import com.google.protobuf.ByteString
import coop.rchain.casper.protocol.deploy.v1._
import org.scalatest.{FlatSpec, Matchers}

class FileUploadProtoSpec extends FlatSpec with Matchers {

  "FileUploadMetadata" should "round-trip through protobuf serialization" in {
    val msg = FileUploadMetadata(
      deployer = ByteString.copyFrom(Array[Byte](1, 2, 3)),
      timestamp = 1700000000000L,
      sig = ByteString.copyFrom(Array[Byte](4, 5, 6)),
      sigAlgorithm = "ed25519",
      phloPrice = 1L,
      phloLimit = 10000000L,
      validAfterBlockNumber = 42L,
      shardId = "root",
      fileName = "ubuntu.iso",
      fileSize = 10737418240L,
      expectedFileHash = "abc123def456"
    )
    FileUploadMetadata.parseFrom(msg.toByteArray) shouldBe msg
  }

  "FileUploadChunk" should "correctly encode the oneof metadata variant" in {
    val meta = FileUploadMetadata(
      deployer = ByteString.copyFrom(Array[Byte](7, 8)),
      timestamp = 1L,
      shardId = "root",
      fileName = "test.bin",
      fileSize = 1024L
    )
    val chunk  = FileUploadChunk(FileUploadChunk.Chunk.Metadata(meta))
    val parsed = FileUploadChunk.parseFrom(chunk.toByteArray)
    parsed.chunk.isMetadata shouldBe true
    parsed.getMetadata.fileName shouldBe "test.bin"
  }

  it should "correctly encode the oneof data variant" in {
    val bytes  = ByteString.copyFrom(Array.fill(64)(0x42.toByte))
    val chunk  = FileUploadChunk(FileUploadChunk.Chunk.Data(bytes))
    val parsed = FileUploadChunk.parseFrom(chunk.toByteArray)
    parsed.chunk.isData shouldBe true
    parsed.getData shouldBe bytes
  }

  "FileUploadResult" should "preserve all cost fields" in {
    val result = FileUploadResult(
      fileHash = "deadbeef",
      deployId = "cafebabe",
      storagePhloCost = 1073741824L,
      totalPhloCharged = 1073742124L
    )
    FileUploadResult.parseFrom(result.toByteArray) shouldBe result
  }

  "FileUploadResponse" should "correctly encode the oneof error variant" in {
    import coop.rchain.casper.protocol.ServiceError
    val resp = FileUploadResponse(
      FileUploadResponse.Message.Error(ServiceError(Seq("something went wrong")))
    )
    val parsed = FileUploadResponse.parseFrom(resp.toByteArray)
    parsed.message.isError shouldBe true
  }

  it should "correctly encode the oneof result variant" in {
    val result = FileUploadResult(fileHash = "aabbccdd", deployId = "11223344")
    val resp   = FileUploadResponse(FileUploadResponse.Message.Result(result))
    val parsed = FileUploadResponse.parseFrom(resp.toByteArray)
    parsed.message.isResult shouldBe true
    parsed.getResult.fileHash shouldBe "aabbccdd"
  }

  "FileDownloadRequest" should "preserve the offset field" in {
    val req    = FileDownloadRequest(fileHash = "deadbeef", offset = 4194304L)
    val parsed = FileDownloadRequest.parseFrom(req.toByteArray)
    parsed.fileHash shouldBe "deadbeef"
    parsed.offset shouldBe 4194304L
  }

  "FileDownloadChunk" should "correctly encode the oneof metadata variant" in {
    val meta   = FileDownloadMetadata(fileHash = "deadbeef", fileSize = 1073741824L)
    val chunk  = FileDownloadChunk(FileDownloadChunk.Chunk.Metadata(meta))
    val parsed = FileDownloadChunk.parseFrom(chunk.toByteArray)
    parsed.chunk.isMetadata shouldBe true
    parsed.getMetadata.fileSize shouldBe 1073741824L
  }

  it should "correctly encode the oneof data variant" in {
    val bytes  = ByteString.copyFrom(Array.fill(32)(0xAB.toByte))
    val chunk  = FileDownloadChunk(FileDownloadChunk.Chunk.Data(bytes))
    val parsed = FileDownloadChunk.parseFrom(chunk.toByteArray)
    parsed.chunk.isData shouldBe true
    parsed.getData shouldBe bytes
  }
}
