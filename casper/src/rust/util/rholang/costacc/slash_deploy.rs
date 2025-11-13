// See casper/src/main/scala/coop/rchain/casper/util/rholang/costacc/SlashDeploy.scala

use std::collections::HashMap;

use crypto::rust::{hash::blake2b512_random::Blake2b512Random, public_key::PublicKey};
use models::{
    rhoapi::Par,
    rust::{block_hash::BlockHash, utils::new_gstring_par},
};
use rholang::rust::interpreter::rho_type::{Extractor, RhoBoolean, RhoNil, RhoString};
use rspace_plus_plus::rspace::history::Either;

use crate::rust::{
    errors::CasperError,
    util::rholang::{
        system_deploy::SystemDeployTrait, system_deploy_user_error::SystemDeployUserError,
    },
};

#[derive(Clone)]
pub struct SlashDeploy {
    pub invalid_block_hash: BlockHash,
    pub pk: PublicKey,
    pub initial_rand: Blake2b512Random,
}

impl SystemDeployTrait for SlashDeploy {
    type Output = (RhoBoolean, Either<RhoString, RhoNil>);
    type Result = ();

    fn source() -> String {
        r#"
          new rl(`rho:registry:lookup`),
          poSCh,
          deployerId(`sys:casper:deployerId`),
          invalidBlockHash(`sys:casper:invalidBlockHash`),
          sysAuthToken(`sys:casper:authToken`),
          return(`sys:casper:return`)
          in {
            rl!(`rho:rchain:pos`, *poSCh) |
            for(@(_, PoS) <- poSCh) {
              @PoS!("slash", *deployerId, *invalidBlockHash.hexToBytes(), *sysAuthToken, *return)
            }
        }"#
        .to_string()
    }

    fn process_result(
        value: <Self::Output as Extractor<Self::Output>>::RustType,
    ) -> Either<SystemDeployUserError, Self::Result> {
        match value {
            (true, _) => Either::Right(()),
            (false, Either::Left(error_msg)) => Either::Left(SystemDeployUserError::new(error_msg)),
            _ => Either::Left(SystemDeployUserError::new(
                "Slashing failed unexpectedly".to_string(),
            )),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn rand(&self) -> Blake2b512Random {
        self.initial_rand.clone()
    }

    fn env(&mut self) -> HashMap<String, Par> {
        let mut env = HashMap::new();

        let (d_key, d_value) = self.mk_deployer_id(&self.pk);
        env.insert(d_key, d_value);

        env.insert(
            "sys:casper:invalidBlockHash".to_string(),
            new_gstring_par(hex::encode(&self.invalid_block_hash), Vec::new(), false),
        );

        let (sys_key, sys_value) = self.mk_sys_auth_token();
        env.insert(sys_key, sys_value);

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
