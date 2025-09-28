package coop.rchain.casper.genesis.contracts

import coop.rchain.casper.helper.RhoSpec
import coop.rchain.casper.util.ConstructDeploy
import coop.rchain.models.NormalizerEnv
import coop.rchain.rholang.build.CompiledRholangSource
import coop.rchain.models.rholang.implicits._

class ASIVaultSpec
    extends RhoSpec(
      CompiledRholangSource("ASIVaultTest.rho", ASIVaultSpec.normalizerEnv),
      Seq.empty,
      GENESIS_TEST_TIMEOUT
    )

object ASIVaultSpec {
  val deployerPk    = ConstructDeploy.defaultPub
  val normalizerEnv = NormalizerEnv.withDeployerId(deployerPk)
}
