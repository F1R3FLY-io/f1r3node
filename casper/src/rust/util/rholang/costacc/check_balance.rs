// See casper/src/main/scala/coop/rchain/casper/util/rholang/costacc/CheckBalance.scala

use std::collections::HashMap;

use crypto::rust::{hash::blake2b512_random::Blake2b512Random, public_key::PublicKey};
use models::rhoapi::Par;
use rholang::rust::interpreter::rho_type::{Extractor, RhoNumber};
use rspace_plus_plus::rspace::history::Either;

use crate::rust::{
    errors::CasperError,
    util::rholang::{
        system_deploy::SystemDeployTrait, system_deploy_user_error::SystemDeployUserError,
    },
};

pub struct CheckBalance {
    pub pk: PublicKey,
    pub rand: Blake2b512Random,
}

impl SystemDeployTrait for CheckBalance {
    type Output = RhoNumber;
    type Result = i64;

    fn source() -> String {
        r#"
        new deployerId(`sys:casper:deployerId`),
        return(`sys:casper:return`),
        rl(`rho:registry:lookup`),
        revAddressOps(`rho:rev:address`),
        revAddressCh,
        revVaultCh in {
          rl!(`rho:rchain:revVault`, *revVaultCh) |
          revAddressOps!("fromDeployerId", *deployerId, *revAddressCh) |
          for(@userRevAddress <- revAddressCh & @(_, revVault) <- revVaultCh){
              new userVaultCh in {
                @revVault!("findOrCreate", userRevAddress, *userVaultCh) |
                for(@(true, userVault) <- userVaultCh){
                  @userVault!("balance", *return)
                }
              }
            }
          }
      "#
        .to_string()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn process_result(
        value: <Self::Output as Extractor<Self::Output>>::RustType,
    ) -> Either<SystemDeployUserError, Self::Result> {
        Either::Right(value)
    }

    fn rand(&self) -> Blake2b512Random {
        self.rand.clone()
    }

    fn env(&mut self) -> HashMap<String, Par> {
        let mut env = HashMap::new();

        let (d_key, d_value) = self.mk_deployer_id(&self.pk);
        env.insert(d_key, d_value);

        let (ret_key, ret_value) = self.mk_return_channel();
        env.insert(ret_key, ret_value);

        env
    }

    fn return_channel(&mut self) -> Result<Par, CasperError> {
        match self.env().get("sys:casper:return") {
            Some(par) => Ok(par.clone()),
            None => Err(CasperError::RuntimeError(
                "Return channel not found. This is a compile time error.".to_string(),
            )),
        }
    }
}
