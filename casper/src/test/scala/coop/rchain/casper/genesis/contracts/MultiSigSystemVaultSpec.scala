package coop.rchain.casper.genesis.contracts

import coop.rchain.casper.helper.RhoSpec
import coop.rchain.casper.util.ConstructDeploy
import coop.rchain.models.NormalizerEnv
import coop.rchain.rholang.build.CompiledRholangSource

class MultiSigSystemVaultSpec
    extends RhoSpec(
      CompiledRholangSource("MultiSigSystemVaultTest.rho", NormalizerEnv.Empty),
      Seq.empty,
      GENESIS_TEST_TIMEOUT
    )
