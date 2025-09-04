package coop.rchain.node.asivaultexport

import com.google.protobuf.ByteString
import coop.rchain.casper.helper.TestNode
import coop.rchain.casper.util.GenesisBuilder.buildGenesis
import coop.rchain.node.asivaultexport.mainnet1.StateBalanceMain
import coop.rchain.rholang.interpreter.util.ASIAddress
import coop.rchain.rspace.hashing.Blake2b256Hash
import monix.execution.Scheduler.Implicits.global
import cats.implicits._
import org.scalatest.FlatSpec

class VaultBalanceGetterTest extends FlatSpec {
  val genesis               = buildGenesis()
  val genesisInitialBalance = 9000000
  "Get balance from VaultPar" should "return balance" in {
    val t = TestNode.standaloneEff(genesis).use { node =>
      val genesisPostStateHash =
        Blake2b256Hash.fromByteString(genesis.genesisBlock.body.state.postStateHash)
      val genesisVaultAddr = ASIAddress.fromPublicKey(genesis.genesisVaults.toList(0)._2).get
      val getVault =
        s"""new return, rl(`rho:registry:lookup`), ASIVaultCh, vaultCh, balanceCh in {
          |  rl!(`rho:rchain:asiVault`, *ASIVaultCh) |
          |  for (@(_, ASIVault) <- ASIVaultCh) {
          |    @ASIVault!("findOrCreate", "${genesisVaultAddr.address.toBase58}", *vaultCh) |
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
      for {
        runtime <- node.runtimeManager.spawnRuntime
        _       <- runtime.reset(genesisPostStateHash)

        // Find vaultMap dynamically by checking all genesis vault addresses
        vaultPks = genesis.genesisVaults.toList.map(_._2)
        balances <- vaultPks.traverse { pub =>
                     val addr = ASIAddress.fromPublicKey(pub).get.address.toBase58
                     val getVault =
                       s"""new return, rl(`rho:registry:lookup`), ASIVaultCh, vaultCh in {
                         |  rl!(`rho:rchain:asiVault`, *ASIVaultCh) |
                         |  for (@(_, ASIVault) <- ASIVaultCh) {
                         |    @ASIVault!("findOrCreate", "${addr}", *vaultCh) |
                         |    for (@(true, vault) <- vaultCh) {
                         |      @vault!("balance", *return)
                         |    }
                         |  }
                         |}
                         |""".stripMargin
                     for {
                       balancePars <- node.runtimeManager
                                       .playExploratoryDeploy(
                                         getVault,
                                         genesis.genesisBlock.body.state.postStateHash
                                       )
                       balance = if (balancePars.nonEmpty) {
                         balancePars(0).exprs.headOption
                           .flatMap(_.exprInstance.gInt)
                           .map(_.toInt)
                           .getOrElse(0)
                       } else 0
                     } yield balance
                   }
        _ = assert(balances.forall(_ == genesisInitialBalance))
      } yield ()
    }
    t.runSyncUnsafe()
  }

}
