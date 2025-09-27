package coop.rchain.casper.util.rholang.costacc

import cats.syntax.all._
import coop.rchain.casper.util.rholang.{SystemDeploy, SystemDeployUserError}
import coop.rchain.crypto.PublicKey
import coop.rchain.crypto.hash.Blake2b512Random
import coop.rchain.models.NormalizerEnv.{Contains, ToEnvMap}
import coop.rchain.rholang.interpreter.RhoType.{Extractor, RhoNumber}

class CheckBalance(pk: PublicKey, rand: Blake2b512Random) extends SystemDeploy(rand) {

  import coop.rchain.models._
  import rholang.{implicits => toPar}
  import shapeless._

  type Output = RhoNumber
  type Result = Long
  type Env =
    (`sys:casper:deployerId` ->> GDeployerId) :: (`sys:casper:return` ->> GUnforgeable) :: HNil

  import toPar._
  protected def toEnvMap                   = ToEnvMap[Env]
  implicit protected val envsReturnChannel = Contains[Env, `sys:casper:return`]
  protected val normalizerEnv              = new NormalizerEnv(mkDeployerId(pk) :: mkReturnChannel :: HNil)
  protected val extractor                  = Extractor.derive

  val source: String =
    """
      # new deployerId(`sys:casper:deployerId`),
      #     return(`sys:casper:return`),
      #     rl(`rho:registry:lookup`),
      #     asiAddressOps(`rho:asi:address`),
      #     asiAddressCh,
      #     asiVaultCh in {
      #   rl!(`rho:rchain:asiVault`, *asiVaultCh) |
      #   asiAddressOps!("fromDeployerId", *deployerId, *asiAddressCh) |
      #   for(@userASIAddress <- asiAddressCh & @(_, asiVault) <- asiVaultCh){
      #     new userVaultCh in {
      #       @asiVault!("findOrCreate", userASIAddress, *userVaultCh) |
      #       for(@(true, userVault) <- userVaultCh){
      #         @userVault!("balance", *return)
      #       }
      #     }
      #   }
      # }
      #""".stripMargin('#')

  protected def processResult(value: Long): Either[SystemDeployUserError, Result] = value.asRight
}
