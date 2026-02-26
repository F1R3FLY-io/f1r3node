package coop.rchain.casper.util.rholang.costacc

import coop.rchain.casper.util.rholang.{SystemDeploy, SystemDeployUserError}
import coop.rchain.crypto.hash.Blake2b512Random
import coop.rchain.models.NormalizerEnv.{Contains, ToEnvMap}
import coop.rchain.rholang.interpreter.RhoType._

/**
  * System deploy that initializes the FileRegistry contract with a SysAuthToken.
  *
  * Follows the CloseBlockDeploy pattern:
  *   1. NormalizerEnv provides a real GSysAuthToken (unforgeable by user Rholang)
  *   2. Rholang source does a registry lookup to find FileRegistry
  *   3. Calls FileRegistry!("init", sysAuthToken, *return) — one-shot init
  *
  * The FileRegistry stores the SysAuthToken on a private channel (_sysAuthTokenCh)
  * for later use by _deleteTemplate when the last deployer is removed from a file.
  *
  * Must be executed during computeGenesis AFTER the FileRegistry.rho contract
  * has been deployed as a blessed term.
  */
final case class FileRegistryInitDeploy(initialRand: Blake2b512Random)
    extends SystemDeploy(initialRand) {
  import coop.rchain.models._
  import rholang.{implicits => toPar}
  import shapeless._

  type Output = RhoBoolean
  type Result = Unit

  type Env =
    (`sys:casper:authToken` ->> GSysAuthToken) :: (`sys:casper:return` ->> GUnforgeable) :: HNil

  import toPar._
  protected override val envsReturnChannel = Contains[Env, `sys:casper:return`]
  protected override val toEnvMap          = ToEnvMap[Env]

  protected val normalizerEnv: NormalizerEnv[Env] = new NormalizerEnv(
    mkSysAuthToken :: mkReturnChannel :: HNil
  )

  // FileRegistry URI: rho:id:m6rqma7yas7o6ieos45ai4dskmc6zugs9rmsp6i3zan8qe5hsfqsdt
  override val source: String =
    """#new rl(`rho:registry:lookup`),
       #  fileRegistryCh,
       #  sysAuthToken(`sys:casper:authToken`),
       #  return(`sys:casper:return`)
       #in {
       #  rl!(`rho:id:m6rqma7yas7o6ieos45ai4dskmc6zugs9rmsp6i3zan8qe5hsfqsdt`, *fileRegistryCh) |
       #  for(@(_, FileRegistry) <- fileRegistryCh) {
       #    @FileRegistry!("init", *sysAuthToken, *return)
       #  }
       #}""".stripMargin('#')

  protected override val extractor = Extractor.derive

  protected override def processResult(
      value: Boolean
  ): Either[SystemDeployUserError, Unit] =
    if (value) Right(())
    else Left(SystemDeployUserError("FileRegistry init failed"))
}
