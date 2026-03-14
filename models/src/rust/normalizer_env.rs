// See models/src/main/scala/coop/rchain/models/NormalizerEnv.scala

use std::collections::HashMap;

use crypto::rust::{public_key::PublicKey, signatures::signed::Signed};

use super::casper::protocol::casper_message::DeployData;

use crate::rhoapi::{g_unforgeable::UnfInstance, GDeployId, GDeployerId, GUnforgeable, Par};

const SYSTEM_DEPLOY_ID_URI: &str = "rho:system:deployId";
const SYSTEM_DEPLOYER_ID_URI: &str = "rho:system:deployerId";
const LEGACY_DEPLOY_ID_URI: &str = "rho:rchain:deployId";
const LEGACY_DEPLOYER_ID_URI: &str = "rho:rchain:deployerId";

fn insert_legacy_alias(
    env: &mut HashMap<String, Par>,
    legacy_uri: &'static str,
    canonical_uri: &'static str,
    value: Par,
) {
    tracing::debug!(
        target: "f1r3fly.legacy-uri",
        "Resolved legacy URI alias `{}` via canonical `{}`",
        legacy_uri,
        canonical_uri
    );
    env.insert(legacy_uri.to_string(), value);
}

pub fn with_deployer_id(deployer_pk: &PublicKey) -> HashMap<String, Par> {
    let mut env = HashMap::new();
    let deployer_id_par = Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(UnfInstance::GDeployerIdBody(GDeployerId {
            public_key: deployer_pk.bytes.to_vec(),
        })),
    }]);

    env.insert(SYSTEM_DEPLOYER_ID_URI.to_string(), deployer_id_par.clone());
    // Backward-compatible alias used by external clients.
    insert_legacy_alias(
        &mut env,
        LEGACY_DEPLOYER_ID_URI,
        SYSTEM_DEPLOYER_ID_URI,
        deployer_id_par,
    );
    env
}

pub fn normalizer_env_from_deploy(deploy: &Signed<DeployData>) -> HashMap<String, Par> {
    let mut env = HashMap::new();

    let deploy_id_par = Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(UnfInstance::GDeployIdBody(GDeployId {
            sig: deploy.sig.to_vec(),
        })),
    }]);

    let deployer_id_par = Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(UnfInstance::GDeployerIdBody(GDeployerId {
            public_key: deploy.pk.bytes.to_vec(),
        })),
    }]);

    env.insert(SYSTEM_DEPLOY_ID_URI.to_string(), deploy_id_par.clone());
    // Backward-compatible alias used by external clients.
    insert_legacy_alias(
        &mut env,
        LEGACY_DEPLOY_ID_URI,
        SYSTEM_DEPLOY_ID_URI,
        deploy_id_par,
    );

    env.insert(SYSTEM_DEPLOYER_ID_URI.to_string(), deployer_id_par.clone());
    // Backward-compatible alias used by external clients.
    insert_legacy_alias(
        &mut env,
        LEGACY_DEPLOYER_ID_URI,
        SYSTEM_DEPLOYER_ID_URI,
        deployer_id_par,
    );

    env
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::rust::{public_key::PublicKey, signatures::secp256k1::Secp256k1};
    use prost::bytes::Bytes;

    fn signed_deploy_fixture() -> Signed<DeployData> {
        Signed {
            data: DeployData {
                term: "Nil".to_string(),
                time_stamp: 1,
                phlo_price: 1,
                phlo_limit: 1,
                valid_after_block_number: 0,
                shard_id: "root".to_string(),
                expiration_timestamp: None,
            },
            pk: PublicKey::from_bytes(&[1, 2, 3, 4]),
            sig: Bytes::from(vec![5, 6, 7, 8]),
            sig_algorithm: Box::new(Secp256k1),
        }
    }

    #[test]
    fn with_deployer_id_should_include_legacy_alias() {
        let deployer_pk = PublicKey::from_bytes(&[10, 11, 12, 13]);
        let env = with_deployer_id(&deployer_pk);

        let system = env
            .get(SYSTEM_DEPLOYER_ID_URI)
            .expect("Missing rho:system:deployerId");
        let legacy = env
            .get(LEGACY_DEPLOYER_ID_URI)
            .expect("Missing rho:rchain:deployerId");

        assert_eq!(system, legacy);
    }

    #[test]
    fn normalizer_env_from_deploy_should_include_legacy_aliases() {
        let deploy = signed_deploy_fixture();
        let env = normalizer_env_from_deploy(&deploy);

        let system_deploy_id = env
            .get(SYSTEM_DEPLOY_ID_URI)
            .expect("Missing rho:system:deployId");
        let legacy_deploy_id = env
            .get(LEGACY_DEPLOY_ID_URI)
            .expect("Missing rho:rchain:deployId");

        let system_deployer_id = env
            .get(SYSTEM_DEPLOYER_ID_URI)
            .expect("Missing rho:system:deployerId");
        let legacy_deployer_id = env
            .get(LEGACY_DEPLOYER_ID_URI)
            .expect("Missing rho:rchain:deployerId");

        assert_eq!(system_deploy_id, legacy_deploy_id);
        assert_eq!(system_deployer_id, legacy_deployer_id);
    }
}
