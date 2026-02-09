package coop.rchain.blockstorage.deploy

import cats.effect.Sync
import cats.syntax.all._
import com.google.protobuf.ByteString
import coop.rchain.blockstorage.dag.codecs.{codecDeploySignature, codecSignedDeployData}
import coop.rchain.casper.protocol.{DeployData, DeployDataProto}
import coop.rchain.crypto.signatures.{SignaturesAlg, Signed}
import coop.rchain.metrics.Metrics
import coop.rchain.shared.{Log, LogSource}
import coop.rchain.shared.syntax._
import coop.rchain.store.InMemoryStoreManager
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Matchers}
import scala.concurrent.duration._

class KeyValueDeployStorageSpec extends FlatSpec with Matchers {

  implicit val logSource: LogSource   = LogSource(this.getClass)
  implicit val log: Log[Task]         = Log.log[Task]
  implicit val metrics: Metrics[Task] = new Metrics.MetricsNOP[Task]

  object MockSignaturesAlg extends SignaturesAlg {
    def name: String                                                                 = "mock"
    val sigLength: Int                                                               = 0
    def sign(data: Array[Byte], sec: Array[Byte]): Array[Byte]                       = Array.empty[Byte]
    def verify(data: Array[Byte], signature: Array[Byte], pub: Array[Byte]): Boolean = true
    def toPublic(sec: coop.rchain.crypto.PrivateKey): coop.rchain.crypto.PublicKey =
      coop.rchain.crypto.PublicKey(Array.empty[Byte])
    def newKeyPair: (coop.rchain.crypto.PrivateKey, coop.rchain.crypto.PublicKey) =
      (
        coop.rchain.crypto.PrivateKey(Array.empty[Byte]),
        coop.rchain.crypto.PublicKey(Array.empty[Byte])
      )
  }

  def createSignedDeploy(): Signed[DeployData] = {
    val (privKey, _) = MockSignaturesAlg.newKeyPair
    val deployData = DeployData(
      term = "Nil",
      timestamp = System.currentTimeMillis(),
      phloLimit = 100000L,
      phloPrice = 1L,
      validAfterBlockNumber = 10L,
      shardId = "test-shard",
      parameters = Seq.empty
    )
    Signed(deployData, MockSignaturesAlg, privKey)
  }

  "KeyValueDeployStorage" should "reject deploys when storage space is low (< 1GB)" in {
    val kvm = InMemoryStoreManager[Task]()
    val test = for {
      store <- kvm.database("deploy_storage", codecDeploySignature, codecSignedDeployData)
      // Mock low space: 500MB < 1GB limit
      lowSpace = () => Task.now(500L * 1024 * 1024)
      ds       = KeyValueDeployStorage[Task](store, lowSpace)

      deploy = createSignedDeploy()
      result <- ds.add(List(deploy)).attempt
    } yield result

    val result = test.runSyncUnsafe(5.seconds)
    result.isLeft shouldBe true
    result.left.get.getMessage should include("Node is running low on storage space")
  }

  it should "accept deploys when storage space is sufficient (> 1GB)" in {
    val kvm = InMemoryStoreManager[Task]()
    val test = for {
      store <- kvm.database("deploy_storage", codecDeploySignature, codecSignedDeployData)
      // Mock sufficient space: 1.5GB > 1GB limit
      highSpace = () => Task.now(1500L * 1024 * 1024)
      ds        = KeyValueDeployStorage[Task](store, highSpace)

      deploy = createSignedDeploy()
      result <- ds.add(List(deploy)).attempt
    } yield result

    val result = test.runSyncUnsafe(5.seconds)
    result.isRight shouldBe true
  }
}
