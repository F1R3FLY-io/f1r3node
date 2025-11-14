// See casper/src/main/scala/coop/rchain/casper/util/rholang/costacc/CloseBlockDeploy.scala

use std::collections::HashMap;

use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::Par;
use rholang::rust::interpreter::rho_type::{RhoBoolean, RhoNil, RhoString};
use rspace_plus_plus::rspace::history::Either;

use crate::rust::{
    errors::CasperError,
    util::rholang::{
        system_deploy::SystemDeployTrait, system_deploy_user_error::SystemDeployUserError,
    },
};

// Currently we use parentHash as initial random seed
#[derive(Clone)]
pub struct CloseBlockDeploy {
    pub initial_rand: Blake2b512Random,
}

impl SystemDeployTrait for CloseBlockDeploy {
    type Output = (RhoBoolean, Either<RhoString, RhoNil>);
    type Result = ();

    fn source() -> String {
        r#"
        new rl(`rho:registry:lookup`),
        poSCh,
        sysAuthToken(`sys:casper:authToken`),
        return(`sys:casper:return`)
        in {
          rl!(`rho:rchain:pos`, *poSCh) |
          for(@(_, PoS) <- poSCh) {
             @PoS!("closeBlock", *sysAuthToken, *return)
          }
        }"#
        .to_string()
    }

    fn process_result(value: (bool, Either<String, ()>)) -> Either<SystemDeployUserError, ()> {
        match value {
            (true, _) => Either::Right(()),
            (false, Either::Left(error_msg)) => Either::Left(SystemDeployUserError::new(error_msg)),
            _ => Either::Left(SystemDeployUserError::new("<no cause>".to_string())),
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
