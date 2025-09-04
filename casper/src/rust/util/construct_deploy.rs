// See casper/src/main/scala/coop/rchain/casper/util/ConstructDeploy.scala

use lazy_static::lazy_static;
use std::time::{SystemTime, UNIX_EPOCH};

use crypto::rust::{
    private_key::PrivateKey,
    public_key::PublicKey,
    signatures::{secp256k1::Secp256k1, signatures_alg::SignaturesAlg, signed::Signed},
};
use models::{
    rhoapi::PCost,
    rust::casper::protocol::casper_message::{DeployData, ProcessedDeploy},
};

use crate::rust::errors::CasperError;

lazy_static! {
    pub static ref DEFAULT_SEC: PrivateKey = PrivateKey::from_bytes(
        &hex::decode("a68a6e6cca30f81bd24a719f3145d20e8424bd7b396309b0708a16c7d8000b76")
            .expect("ConstructDeploy: Failed to decode default private key")
    );
    pub static ref DEFAULT_PUB: PublicKey = {
        let secp = Secp256k1;
        secp.to_public(&DEFAULT_SEC)
    };
    pub static ref DEFAULT_KEY_PAIR: (&'static PrivateKey, &'static PublicKey) =
        (&DEFAULT_SEC, &DEFAULT_PUB);
    pub static ref DEFAULT_SEC2: PrivateKey = PrivateKey::from_bytes(
        &hex::decode("5a0bde2f5857124b1379c78535b07a278e3b9cefbcacc02e62ab3294c02765a1")
            .expect("ConstructDeploy: Failed to decode default private key")
    );
    pub static ref DEFAULT_PUB2: PublicKey = {
        let secp = Secp256k1;
        secp.to_public(&DEFAULT_SEC2)
    };
}

pub fn source_deploy(
    source: String,
    timestamp: i64,
    phlo_limit: Option<i64>,
    phlo_price: Option<i64>,
    sec: Option<PrivateKey>,
    valid_after_block_number: Option<i64>,
    shard_id: Option<String>,
) -> Result<Signed<DeployData>, CasperError> {
    let sec = sec.unwrap_or_else(|| DEFAULT_SEC.clone());
    let phlo_limit = phlo_limit.unwrap_or(90000);
    let phlo_price = phlo_price.unwrap_or(1);
    let valid_after_block_number = valid_after_block_number.unwrap_or(0);
    let shard_id = shard_id.unwrap_or_else(|| "".to_string());

    let data = DeployData {
        term: source,
        time_stamp: timestamp,
        phlo_price,
        phlo_limit,
        valid_after_block_number,
        shard_id,
        language: "rholang".to_string(),
    };

    Signed::create(data, Box::new(Secp256k1), sec).map_err(|e| CasperError::SigningError(e))
}

pub fn source_deploy_now(
    source: String,
    sec: Option<PrivateKey>,
    valid_after_block_number: Option<i64>,
    shard_id: Option<String>,
) -> Result<Signed<DeployData>, CasperError> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| CasperError::RuntimeError(e.to_string()))?
        .as_millis() as i64;

    source_deploy(
        source,
        timestamp,
        None,
        None,
        sec,
        valid_after_block_number,
        shard_id,
    )
}

pub fn source_deploy_now_full(
    source: String,
    phlo_limit: Option<i64>,
    phlo_price: Option<i64>,
    sec: Option<PrivateKey>,
    valid_after_block_number: Option<i64>,
    shard_id: Option<String>,
) -> Result<Signed<DeployData>, CasperError> {
    let phlo_limit = phlo_limit.unwrap_or(1000000);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| CasperError::RuntimeError(e.to_string()))?
        .as_millis() as i64;

    source_deploy(
        source,
        timestamp,
        Some(phlo_limit),
        phlo_price,
        sec,
        valid_after_block_number,
        shard_id,
    )
}

pub fn basic_deploy_data(
    id: i32,
    sec: Option<PrivateKey>,
    shard_id: Option<String>,
) -> Result<Signed<DeployData>, CasperError> {
    source_deploy_now(format!("@{}!({})", id, id), sec, None, shard_id)
}

pub fn basic_processed_deploy(
    id: i32,
    shard_id: Option<String>,
) -> Result<ProcessedDeploy, CasperError> {
    basic_deploy_data(id, None, shard_id).map(|deploy| ProcessedDeploy {
        deploy,
        cost: PCost { cost: 0 },
        deploy_log: Vec::new(),
        is_failed: false,
        system_deploy_error: None,
    })
}
