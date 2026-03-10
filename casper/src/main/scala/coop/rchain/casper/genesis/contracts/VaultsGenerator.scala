package coop.rchain.casper.genesis.contracts

import coop.rchain.models.NormalizerEnv
import coop.rchain.rholang.build.CompiledRholangSource

final class VaultsGenerator private (supply: Long, code: String)
    extends CompiledRholangSource(code, NormalizerEnv.Empty) {
  val path: String = "<synthetic in VaultsGenerator.scala>"
}

object VaultsGenerator {

  // System vault initialization in genesis is done in batches.
  // In the last batch `initContinue` channel will not receive
  // anything so further access to `SystemVault(@"init", _)` is impossible.

  def apply(userVaults: Seq[Vault], supply: Long, isLastBatch: Boolean): VaultsGenerator = {
    val vaultBalanceList =
      userVaults.map(v => s"""("${v.vaultAddress.toBase58}", ${v.initialBalance})""").mkString(", ")

    val code: String =
      s""" new rl(`rho:registry:lookup`), systemVaultCh in {
         #   rl!(`rho:vault:system`, *systemVaultCh) |
         #   for (@(_, SystemVault) <- systemVaultCh) {
         #     new systemVaultInitCh in {
         #       @SystemVault!("init", *systemVaultInitCh) |
         #       for (TreeHashMap, @vaultMap, initVault, initContinue <- systemVaultInitCh) {
         #         match [$vaultBalanceList] {
         #           vaults => {
         #             new iter in {
         #               contract iter(@[(addr, initialBalance) ... tail]) = {
         #                  iter!(tail) |
         #                  new vault, setDoneCh in {
         #                    initVault!(*vault, addr, initialBalance) |
         #                    TreeHashMap!("set", vaultMap, addr, *vault, *setDoneCh) |
         #                    for (_ <- setDoneCh) { Nil }
         #                  }
         #               } |
         #               iter!(vaults) ${if (!isLastBatch) "| initContinue!()" else ""}
         #             }
         #           }
         #         }
         #       }
         #     }
         #   }
         # }
     """.stripMargin('#')

    new VaultsGenerator(supply, code)
  }
}
