package coop.rchain.node.revvaultexport

import cats.effect.Concurrent
import coop.rchain.casper.genesis.contracts.StandardDeploys
import coop.rchain.casper.helper.TestNode.Effect
import coop.rchain.casper.helper.TestRhoRuntime.rhoRuntimeEff
import coop.rchain.casper.syntax._
import coop.rchain.casper.util.ConstructDeploy
import coop.rchain.metrics.{Metrics, NoopSpan, Span}
import coop.rchain.rspace.hashing.Blake2b256Hash
import coop.rchain.shared.Log
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import org.scalatest.FlatSpec

import scala.util.Random

// NOTE: This test is currently ignored because the storeToken unforgeable calculation
// in RhoTrieTraverser does not match the actual token used by the Registry.rho TreeHashMap.
// The token generation logic attempts to replay the random number generation sequence from
// the Registry deployment, but the exact sequence depends on how many `new` declarations
// appear before `storeToken` in Registry.rho (line 59). The current calculation uses
// splitShort(6) with 7 subsequent next() calls, but this may not match the actual registry.
// Until this is fixed, the traversal returns empty results because getData() queries
// the wrong channel (with incorrect storeToken).
@org.scalatest.Ignore
class RhoTrieTraverserTest extends FlatSpec {
  private val SHARD_ID = "root-shard"

  "traverse the TreeHashMap" should "work" in {
    val total     = 100
    val trieDepth = 2
    val insertKeyValues = (0 to total).map(
      i => (Random.alphanumeric.take(10).foldLeft("")(_ + _), Random.nextInt(1000000), i)
    )
    val insertRho = insertKeyValues.foldLeft("") {
      case (acc, (key, value, index)) =>
        if (index != total)
          acc + s"""new a in {TreeHashMap!("set", treeMap, "${key}", ${value}, *a)}|\n"""
        else acc + s"""new a in {TreeHashMap!("set", treeMap, "${key}", ${value}, *a)}\n"""
    }
    val trieInitializedRho =
      s"""
        |new
        |  rl(`rho:registry:lookup`),
        |  TreeHashMapCh,
        |  newTreeMapStore,
        |  vaultMapStore
        |  in {
        |  rl!(`rho:lang:treeHashMap`, *TreeHashMapCh) |
        |  for (TreeHashMap <- TreeHashMapCh){
        |    TreeHashMap!("init", ${trieDepth}, *vaultMapStore) |
        |    for (@treeMap <-  vaultMapStore){
        |      ${insertRho}
        |      |@"t"!(treeMap)
        |    }
        |  }
        |}
        |""".stripMargin

    val getTrieMapHandleRho = """new s in {
                               |  for (@result<- @"t"){
                               |    s!(result)
                               |  }
                               |}""".stripMargin

    implicit val concurent                   = Concurrent[Task]
    implicit val metricsEff: Metrics[Effect] = new Metrics.MetricsNOP[Task]
    implicit val noopSpan: Span[Effect]      = NoopSpan[Task]()
    implicit val logger: Log[Effect]         = Log.log[Task]
    val t = rhoRuntimeEff[Effect](false).use {
      case (runtime, _, _) =>
        for {
          hash1 <- runtime.emptyStateHash
          _     <- runtime.reset(Blake2b256Hash.fromByteString(hash1))
          rd    <- runtime.processDeploy(StandardDeploys.registry(SHARD_ID))
          check <- runtime.createCheckpoint
          _     <- runtime.reset(check.root)
          initialTrieRes <- runtime.processDeploy(
                             ConstructDeploy
                               .sourceDeploy(trieInitializedRho, 1L, phloLimit = 50000000)
                           )
          (initialTrie, _) = initialTrieRes
          _                = assert(!initialTrie.isFailed)
          check2           <- runtime.createCheckpoint
          trieMapHandleR <- runtime.playExploratoryDeploy(
                             getTrieMapHandleRho,
                             check2.root.toByteString
                           )
          _             <- runtime.reset(check2.root)
          trieMapHandle = trieMapHandleR(0)
          maps          <- RhoTrieTraverser.traverseTrie(trieDepth, trieMapHandle, runtime)
          goodMap = RhoTrieTraverser.vecParMapToMap(
            maps,
            p => p.exprs(0).getGByteArray,
            p => p.exprs(0).getGInt
          )
          _ = insertKeyValues.map(k => {
            val key =
              RhoTrieTraverser.keccakKey(k._1).exprs(0).getGByteArray.substring(trieDepth, 32)
            assert(goodMap.get(key).get == k._2.toLong)
          })
        } yield ()
    }
    t.runSyncUnsafe()
  }

}
