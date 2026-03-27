package coop.rchain.node.api

import com.google.protobuf.ByteString
import coop.rchain.casper.protocol.deploy.v1._
import coop.rchain.crypto.hash.Blake2b256
import coop.rchain.crypto.signatures.Secp256k1
import coop.rchain.crypto.{PrivateKey, PublicKey}
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import monix.reactive.Observable
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

import java.nio.file.{Files, Path}
import scala.concurrent.duration._

class SyntheticDeploySpec extends FlatSpec with Matchers with BeforeAndAfterAll {

  private val uploadDir: Path = Files.createTempDirectory("synthetic-deploy-test-")

  override def afterAll(): Unit = {
    def deleteTree(p: Path): Unit =
      if (Files.isDirectory(p)) {
        val stream = Files.list(p)
        try {
          import scala.jdk.CollectionConverters._
          stream.iterator().asScala.foreach(deleteTree)
        } finally {
          stream.close()
        }
        Files.deleteIfExists(p)
        ()
      } else {
        Files.deleteIfExists(p)
        ()
      }
    deleteTree(uploadDir)
  }

  // ---- helpers -----------------------------------------------------------

  private val (privKey, pubKey): (PrivateKey, PublicKey) = Secp256k1.newKeyPair

  private val chunkSize  = 64 * 1024
  private val numChunks  = 4
  private val totalBytes = chunkSize * numChunks

  private def makeChunkData(i: Int): Array[Byte] = Array.fill(chunkSize)(i.toByte)

  private def allBytes: Array[Byte] = (0 until numChunks).flatMap(i => makeChunkData(i)).toArray

  private def referenceHash: String =
    coop.rchain.shared.Base16.encode(Blake2b256.hash(allBytes))

  /** Build a stub Rholang term the client would construct. */
  private def stubTerm(fileHash: String = referenceHash): String =
    s"""new return, file(`rho:io:file`) in { file!("register", "$fileHash", $totalBytes, "test.bin", *return) }"""

  /**
    * Compute the client's deploy signature over the DeployData fields.
    * This replicates what a real client SDK would do.
    */
  private def clientSignedMetadata(
      term: String = stubTerm(),
      phloPrice: Long = 2L,
      shardId: String = "root",
      fileHash: String = referenceHash
  ): FileUploadMetadata = {
    import coop.rchain.casper.protocol.{DeployData, DeployDataProto}
    import coop.rchain.crypto.signatures.Signed

    val timestamp = 1700000000L

    // Build DeployData (same fields the server will reconstruct)
    val deployData = DeployData(
      term = term,
      timestamp = timestamp,
      phloPrice = phloPrice,
      phloLimit = 10000000L,
      validAfterBlockNumber = 0L,
      shardId = shardId
    )

    // Client signs with its private key
    val signed = Signed(deployData, Secp256k1, privKey)

    FileUploadMetadata(
      deployer = ByteString.copyFrom(pubKey.bytes),
      timestamp = timestamp,
      sig = signed.sig,
      sigAlgorithm = signed.sigAlgorithm.name,
      phloPrice = phloPrice,
      phloLimit = 10000000L,
      validAfterBlockNumber = 0L,
      shardId = shardId,
      fileName = "test.bin",
      fileSize = totalBytes.toLong,
      fileHash = fileHash,
      term = term
    )
  }

  private def metadataChunk(
      term: String = stubTerm(),
      phloPrice: Long = 2L,
      shardId: String = "root",
      fileHash: String = referenceHash
  ): FileUploadChunk =
    FileUploadChunk(
      FileUploadChunk.Chunk.Metadata(
        clientSignedMetadata(term, phloPrice, shardId, fileHash)
      )
    )

  private def dataChunks: Observable[FileUploadChunk] =
    Observable.fromIterable(
      (0 until numChunks).map(
        i => FileUploadChunk(FileUploadChunk.Chunk.Data(ByteString.copyFrom(makeChunkData(i))))
      )
    )

  private def fullStream(
      term: String = stubTerm(),
      phloPrice: Long = 2L,
      shardId: String = "root",
      fileHash: String = referenceHash
  ): Observable[FileUploadChunk] =
    Observable.cons(metadataChunk(term, phloPrice, shardId, fileHash), dataChunks)

  private def run[A](task: Task[A]): A = task.runSyncUnsafe(30.seconds)

  private def newDir(): Path = {
    val d = uploadDir.resolve(java.util.UUID.randomUUID().toString)
    Files.createDirectories(d)
    d
  }

  // ---- SyntheticDeploy.metadataToDeployProto -----------------------------

  "SyntheticDeploy.metadataToDeployProto" should "map all metadata fields to DeployDataProto" in {
    val metadata = clientSignedMetadata()
    val proto    = SyntheticDeploy.metadataToDeployProto(metadata)

    proto.deployer shouldBe metadata.deployer
    proto.term shouldBe metadata.term
    proto.timestamp shouldBe metadata.timestamp
    proto.sig shouldBe metadata.sig
    proto.sigAlgorithm shouldBe metadata.sigAlgorithm
    proto.phloPrice shouldBe metadata.phloPrice
    proto.phloLimit shouldBe metadata.phloLimit
    proto.validAfterBlockNumber shouldBe metadata.validAfterBlockNumber
    proto.shardId shouldBe metadata.shardId
  }

  it should "produce a proto that passes DeployData.from() signature validation" in {
    import coop.rchain.casper.protocol.DeployData
    val metadata = clientSignedMetadata()
    val proto    = SyntheticDeploy.metadataToDeployProto(metadata)
    val result   = DeployData.from(proto)
    result shouldBe a[Right[_, _]]
  }

  it should "fail DeployData.from() when sig is tampered" in {
    import coop.rchain.casper.protocol.DeployData
    val metadata = clientSignedMetadata()
    val tampered = metadata.copy(sig = ByteString.copyFrom(Array[Byte](0, 0, 0)))
    val proto    = SyntheticDeploy.metadataToDeployProto(tampered)
    val result   = DeployData.from(proto)
    result shouldBe a[Left[_, _]]
  }

  // ---- SyntheticDeploy.computeStorageCost --------------------------------

  "SyntheticDeploy.computeStorageCost" should "set storagePhloCost = fileSize * phloPerStorageByte" in {
    val (storageCost, _) =
      SyntheticDeploy.computeStorageCost(fileSize = 5000L, phloPerStorageByte = 3L)
    storageCost shouldBe 15000L
  }

  it should "set totalPhloCharged = baseRegisterPhlo + storagePhloCost" in {
    val (_, total) = SyntheticDeploy.computeStorageCost(
      fileSize = 5000L,
      phloPerStorageByte = 3L,
      baseRegisterPhlo = 300L
    )
    total shouldBe 15300L
  }

  it should "handle zero fileSize" in {
    val (storage, total) = SyntheticDeploy.computeStorageCost(0L, 10L, 300L)
    storage shouldBe 0L
    total shouldBe 300L
  }

  it should "throw ArithmeticException on overflow" in {
    intercept[ArithmeticException] {
      SyntheticDeploy.computeStorageCost(Long.MaxValue, 2L)
    }
  }

  // ---- processFileUpload integration  ------------------------------------

  "FileUploadAPI.processFileUpload" should "return a deployProto with the client's term" in {
    val dir  = newDir()
    val term = stubTerm()
    val output = run(
      FileUploadAPI.processFileUpload(fullStream(term = term), "root", 1L, false, dir)
    )
    output.deployProto shouldBe defined
    output.deployProto.get.term shouldBe term
  }

  it should "return storagePhloCost equal to fileSize * phloPerStorageByte" in {
    val dir = newDir()
    val output = run(
      FileUploadAPI.processFileUpload(fullStream(), "root", 1L, false, dir)
    )
    output.result.storagePhloCost shouldBe totalBytes.toLong * FileUploadCosts.DEFAULT_PHLO_PER_STORAGE_BYTE
  }

  it should "return totalPhloCharged = baseRegisterPhlo + storagePhloCost" in {
    val dir = newDir()
    val output = run(
      FileUploadAPI.processFileUpload(fullStream(), "root", 1L, false, dir)
    )
    val expectedStorage = totalBytes.toLong * FileUploadCosts.DEFAULT_PHLO_PER_STORAGE_BYTE
    output.result.totalPhloCharged shouldBe FileUploadCosts.BASE_REGISTER_PHLO + expectedStorage
  }

  it should "leave deployId empty (filled by gRPC layer after sig validation)" in {
    val dir = newDir()
    val output = run(
      FileUploadAPI.processFileUpload(fullStream(), "root", 1L, false, dir)
    )
    output.result.deployId shouldBe ""
  }

  it should "return the correct fileHash in the result" in {
    val dir = newDir()
    val output = run(
      FileUploadAPI.processFileUpload(fullStream(), "root", 1L, false, dir)
    )
    output.result.fileHash shouldBe referenceHash
  }

  it should "reject when shardId mismatches" in {
    val dir = newDir()
    // metadata shardId = "root" but node shardId = "other-shard"
    val ex = intercept[Exception] {
      run(
        FileUploadAPI.processFileUpload(fullStream(shardId = "root"), "other-shard", 1L, false, dir)
      )
    }
    ex.getMessage should include("Invalid shardId")
  }

  it should "reject when term is empty" in {
    val dir = newDir()
    val metadata = FileUploadMetadata(
      deployer = ByteString.copyFrom(pubKey.bytes),
      timestamp = 1700000000L,
      shardId = "root",
      fileName = "test.bin",
      fileSize = totalBytes.toLong,
      fileHash = referenceHash,
      phloPrice = 2L,
      phloLimit = 10000000L,
      validAfterBlockNumber = 0L,
      sigAlgorithm = "secp256k1",
      sig = ByteString.copyFrom(Array[Byte](1, 2, 3)),
      term = "" // empty term
    )
    val stream = Observable.cons(
      FileUploadChunk(FileUploadChunk.Chunk.Metadata(metadata)),
      dataChunks
    )
    val ex = intercept[Exception] {
      run(FileUploadAPI.processFileUpload(stream, "root", 1L, false, dir))
    }
    ex.getMessage should include("term")
  }

  it should "still return deployProto on dedup (file already exists)" in {
    val dir = newDir()
    // First upload
    run(FileUploadAPI.processFileUpload(fullStream(), "root", 1L, false, dir))
    // Second upload — same hash, file already on disk
    val output2 = run(
      FileUploadAPI.processFileUpload(fullStream(), "root", 1L, false, dir)
    )
    output2.deployProto shouldBe defined
    output2.result.storagePhloCost shouldBe totalBytes.toLong * FileUploadCosts.DEFAULT_PHLO_PER_STORAGE_BYTE
    output2.result.fileHash shouldBe referenceHash
  }
}
