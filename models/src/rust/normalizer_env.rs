// See models/src/main/scala/coop/rchain/models/NormalizerEnv.scala

use std::collections::HashMap;

use crypto::rust::{public_key::PublicKey, signatures::signed::Signed};

use super::casper::protocol::casper_message::DeployData;

use crate::rhoapi::{g_unforgeable::UnfInstance, GDeployId, GDeployerId, GUnforgeable, Par};

pub fn with_deployer_id(deployer_pk: &PublicKey) -> HashMap<String, Par> {
    let mut env = HashMap::new();
    env.insert(
        "rho:rchain:deployerId".to_string(),
        Par::default().with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GDeployerIdBody(GDeployerId {
                public_key: deployer_pk.bytes.to_vec(),
            })),
        }]),
    );
    env
}

pub fn normalizer_env_from_deploy(deploy: &Signed<DeployData>) -> HashMap<String, Par> {
    let mut env = HashMap::new();
    env.insert(
        "rho:rchain:deployId".to_string(),
        Par::default().with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GDeployIdBody(GDeployId {
                sig: deploy.sig.to_vec(),
            })),
        }]),
    );
    env.insert(
        "rho:rchain:deployerId".to_string(),
        Par::default().with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GDeployerIdBody(GDeployerId {
                public_key: deploy.pk.bytes.to_vec(),
            })),
        }]),
    );
    env
}
