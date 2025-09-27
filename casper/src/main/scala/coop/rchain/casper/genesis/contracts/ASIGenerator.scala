package coop.rchain.casper.genesis.contracts

import coop.rchain.models.NormalizerEnv
import coop.rchain.rholang.build.CompiledRholangSource

final class ASIGenerator private (supply: Long, code: String)
    extends CompiledRholangSource(code, NormalizerEnv.Empty) {
  val path: String = "<synthetic in ASI.scala>"
}

object ASIGenerator {

  // ASI vault initialization in genesis is done in batches.
  // In the last batch `initContinue` channel will not receive
  // anything so further access to `ASIVault(@"init", _)` is impossible.

  def apply(userVaults: Seq[Vault], supply: Long, isLastBatch: Boolean): ASIGenerator = {
    val vaultBalanceList =
      userVaults.map(v => s"""("${v.asiAddress.toBase58}", ${v.initialBalance})""").mkString(", ")

    val code: String =
      s""" new rl(`rho:registry:lookup`), asiVaultCh in {
         |   rl!(`rho:rchain:asiVault`, *asiVaultCh) |
         |   for (@(_, ASIVault) <- asiVaultCh) {
         |     new asiVaultInitCh in {
         |       @ASIVault!("init", *asiVaultInitCh) |
         |       for (TreeHashMap, @vaultMap, initVault, initContinue <- asiVaultInitCh) {
         |         match [$vaultBalanceList] {
         |           vaults => {
         |             new iter in {
         |               contract iter(@[(addr, initialBalance) ... tail]) = {
         |                  iter!(tail) |
         |                  new vault, setDoneCh in {
         |                    initVault!(*vault, addr, initialBalance) |
         |                    TreeHashMap!("set", vaultMap, addr, *vault, *setDoneCh) |
         |                    for (_ <- setDoneCh) { Nil }
         |                  }
         |               } |
         |               iter!(vaults) ${if (!isLastBatch) "| initContinue!()" else ""}
         |             }
         |           }
         |         }
         |       }
         |     }
         |   }
         | }
     """.stripMargin('|')

    new ASIGenerator(supply, code)
  }
}
