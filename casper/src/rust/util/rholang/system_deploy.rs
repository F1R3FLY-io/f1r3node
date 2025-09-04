// See casper/src/main/scala/coop/rchain/casper/util/rholang/SystemDeploy.scala

use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use crypto::rust::public_key::PublicKey;
use models::rhoapi::g_unforgeable::UnfInstance;
use models::rhoapi::GPrivate;
use models::rhoapi::{GDeployerId, GSysAuthToken, GUnforgeable, Par};
use rholang::rust::interpreter::rho_type::Extractor;
use rspace_plus_plus::rspace::history::Either;
use std::collections::HashMap;

use crate::rust::errors::CasperError;

use super::system_deploy_user_error::{SystemDeployPlatformFailure, SystemDeployUserError};

pub trait SystemDeployTrait {
    type Output: Extractor<Self::Output>;
    type Result;

    fn source() -> String;

    fn process_result(
        value: <Self::Output as Extractor<Self::Output>>::RustType,
    ) -> Either<SystemDeployUserError, Self::Result>;

    fn as_any(&self) -> &dyn std::any::Any;

    fn rand(&self) -> Blake2b512Random;

    fn env(&mut self) -> HashMap<String, Par>;

    fn return_channel(&mut self) -> Result<Par, CasperError>;

    fn mk_return_channel(&mut self) -> (String, Par) {
        (
            "sys:casper:return".to_string(),
            Par::default().with_unforgeables(vec![GUnforgeable {
                unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
                    id: self.rand().next().into_iter().map(|b| b as u8).collect(),
                })),
            }]),
        )
    }

    fn mk_deployer_id(&self, pk: &PublicKey) -> (String, Par) {
        (
            "sys:casper:deployerId".to_string(),
            Par::default().with_unforgeables(vec![GUnforgeable {
                unf_instance: Some(UnfInstance::GDeployerIdBody(GDeployerId {
                    public_key: pk.bytes.to_vec(),
                })),
            }]),
        )
    }

    fn mk_sys_auth_token(&self) -> (String, Par) {
        (
            "sys:casper:authToken".to_string(),
            Par::default().with_unforgeables(vec![GUnforgeable {
                unf_instance: Some(UnfInstance::GSysAuthTokenBody(GSysAuthToken {})),
            }]),
        )
    }

    fn extract_result(&self, output: &Par) -> Either<SystemDeployUserError, Self::Result> {
        match <Self::Output as Extractor<Self::Output>>::unapply(output) {
            Some(value) => Self::process_result(value),
            None => {
                let error = SystemDeployPlatformFailure::UnexpectedResult(vec![output.clone()]);
                Either::Left(SystemDeployUserError::from(error))
            }
        }
    }
}
