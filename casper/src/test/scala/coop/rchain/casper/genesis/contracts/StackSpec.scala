package coop.rchain.casper.genesis.contracts
import coop.rchain.casper.helper.RhoSpec
import coop.rchain.models.NormalizerEnv
import coop.rchain.rholang.build.CompiledRholangSource

class StackSpec
    extends RhoSpec(
      CompiledRholangSource("StackTest.rho", NormalizerEnv.Empty),
      Seq.empty,
      GENESIS_TEST_TIMEOUT
    )
