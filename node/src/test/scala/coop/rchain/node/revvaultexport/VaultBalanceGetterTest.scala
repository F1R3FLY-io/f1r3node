package coop.rchain.node.revvaultexport

import cats.syntax.all._
import coop.rchain.casper.helper.TestNode
import coop.rchain.casper.util.GenesisBuilder.buildGenesis
import coop.rchain.rholang.interpreter.util.RevAddress
import coop.rchain.rspace.hashing.Blake2b256Hash
import monix.execution.Scheduler.Implicits.global
import org.scalatest.FlatSpec

class VaultBalanceGetterTest extends FlatSpec {
  val genesis               = buildGenesis()
  val genesisInitialBalance = 9000000
  "Get balance from VaultPar" should "return balance" in {
    val t = TestNode.standaloneEff(genesis).use { node =>
      val genesisPostStateHash =
        Blake2b256Hash.fromByteString(genesis.genesisBlock.body.state.postStateHash)
      val genesisVaultAddr = RevAddress.fromPublicKey(genesis.genesisVaults.toList(0)._2).get
      val getVault =
        s"""new return, rl(`rho:registry:lookup`), RevVaultCh, vaultCh, balanceCh in {
          |  rl!(`rho:rchain:revVault`, *RevVaultCh) |
          |  for (@(_, RevVault) <- RevVaultCh) {
          |    @RevVault!("findOrCreate", "${genesisVaultAddr.address.toBase58}", *vaultCh) |
          |    for (@(true, vault) <- vaultCh) {
          |      return!(vault)
          |    }
          |  }
          |}
          |""".stripMargin

      for {
        vaultPar <- node.runtimeManager
                     .playExploratoryDeploy(getVault, genesis.genesisBlock.body.state.postStateHash)
        runtime <- node.runtimeManager.spawnRuntime
        _       <- runtime.reset(genesisPostStateHash)
        balance <- VaultBalanceGetter.getBalanceFromVaultPar(vaultPar(0), runtime)
        // 9000000 is hard coded in genesis block generation
        _ = assert(balance.get == genesisInitialBalance)
      } yield ()
    }
    t.runSyncUnsafe()
  }

  "Get all vault" should "return all vault balance" in {
    val t = TestNode.standaloneEff(genesis).use { node =>
      val genesisPostStateHash =
        Blake2b256Hash.fromByteString(genesis.genesisBlock.body.state.postStateHash)

      // Get all genesis vault addresses
      val genesisVaultAddrs = genesis.genesisVaults
        .map { case (_, pub) => RevAddress.fromPublicKey(pub).get }

      // Query each vault balance using Rholang (similar to first test)
      for {
        runtime <- node.runtimeManager.spawnRuntime
        _       <- runtime.reset(genesisPostStateHash)
        // Query each genesis vault and verify its balance
        _ <- genesisVaultAddrs.toList.traverse { vaultAddr =>
              val getVaultBalance =
                s"""new return, rl(`rho:registry:lookup`), RevVaultCh, vaultCh, balanceCh in {
              |  rl!(`rho:rchain:revVault`, *RevVaultCh) |
              |  for (@(_, RevVault) <- RevVaultCh) {
              |    @RevVault!("findOrCreate", "${vaultAddr.address.toBase58}", *vaultCh) |
              |    for (@(true, vault) <- vaultCh) {
              |      @vault!("balance", *balanceCh) |
              |      for (@balance <- balanceCh) {
              |        return!(balance)
              |      }
              |    }
              |  }
              |}
              |""".stripMargin
              for {
                result <- node.runtimeManager
                           .playExploratoryDeploy(
                             getVaultBalance,
                             genesis.genesisBlock.body.state.postStateHash
                           )
                balance = result.head.exprs.head.getGInt
                // 9000000 is hard coded in genesis block generation
                _ = assert(
                  balance == genesisInitialBalance,
                  s"Expected $genesisInitialBalance but got $balance for ${vaultAddr.address.toBase58}"
                )
              } yield ()
            }
      } yield ()
    }
    t.runSyncUnsafe()
  }

}
