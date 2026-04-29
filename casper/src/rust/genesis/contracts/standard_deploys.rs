// See casper/src/main/scala/coop/rchain/casper/genesis/contracts/StandardDeploys.scala

use std::collections::HashMap;

use lazy_static::lazy_static;

use crypto::rust::{
    private_key::PrivateKey,
    public_key::PublicKey,
    signatures::{secp256k1::Secp256k1, signatures_alg::SignaturesAlg, signed::Signed},
};
use models::rust::casper::protocol::casper_message::DeployData;
use rholang::rust::build::compile_rholang_source::{
    CompiledRholangSource, CompiledRholangTemplate,
};

use super::{
    embedded_rho, proof_of_stake::ProofOfStake, vault::Vault,
    vaults_generator::VaultsGenerator,
};

/// Build a `CompiledRholangSource` from an embedded `.rho` constant. The
/// `name` is preserved on the resulting source as identification metadata
/// (used by the deploy/path field) — it is not a filesystem path.
fn embedded_source(name: &str, code: &str) -> CompiledRholangSource {
    CompiledRholangSource::new(code.to_string(), HashMap::new(), name.to_string())
        .unwrap_or_else(|e| panic!("Failed to compile embedded {}: {:?}", name, e))
}

// Private keys used to sign blessed (standard) contracts
pub const REGISTRY_PK: &str = "5a0bde2f5857124b1379c78535b07a278e3b9cefbcacc02e62ab3294c02765a1";
pub const LIST_OPS_PK: &str = "867c21c6a3245865444d80e49cac08a1c11e23b35965b566bbe9f49bb9897511";
pub const EITHER_PK: &str = "5248f8913f8572d8227a3c7787b54bd8263389f7209adc1422e36bb2beb160dc";
pub const NON_NEGATIVE_NUMBER_PK: &str =
    "e33c9f1e925819d04733db4ec8539a84507c9e9abd32822059349449fe03997d";
pub const MAKE_MINT_PK: &str = "de19d53f28d4cdee74bad062342d8486a90a652055f3de4b2efa5eb2fccc9d53";
pub const AUTH_KEY_PK: &str = "f450b26bac63e5dd9343cd46f5fae1986d367a893cd21eedd98a4cb3ac699abc";
pub const SYSTEM_VAULT_PK: &str =
    "27e5718bf55dd673cc09f13c2bcf12ed7949b178aef5dcb6cd492ad422d05e9d";
pub const MULTI_SIG_SYSTEM_VAULT_PK: &str =
    "2a2eaa76d6fea9f502629e32b0f8eea19b9de8e2188ec0d589fcafa98fb1f031";
pub const POS_GENERATOR_PK: &str =
    "a9585a0687761139ab3587a4938fb5ab9fcba675c79fefba889859674046d4a5";
pub const VAULTS_GENERATOR_PK: &str =
    "a06959868e39bb3a8502846686a23119716ecd001700baf9e2ecfa0dbf1a3247";
pub const STACK_PK: &str = "c94e647de6876c954ebb7b64c40a220227770f9be003635edfe3336a1a2c8605";
// Private key, timestamp, pubkey, signature, and URI for TokenMetadata were generated
// via RegistrySigGen. See casper/tests/util/rholang/token_metadata_sig_gen.rs for the
// one-off generator and the derivation table at the top of TokenMetadata.rhox.
pub const TOKEN_METADATA_PK: &str =
    "8f9a1c3b2d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a";

// Timestamps for each deploy
pub const REGISTRY_TIMESTAMP: i64 = 1559156071321;
pub const LIST_OPS_TIMESTAMP: i64 = 1559156082324;
pub const EITHER_TIMESTAMP: i64 = 1559156217509;
pub const NON_NEGATIVE_NUMBER_TIMESTAMP: i64 = 1559156251792;
pub const MAKE_MINT_TIMESTAMP: i64 = 1559156452968;
pub const AUTH_KEY_TIMESTAMP: i64 = 1559156356769;
pub const SYSTEM_VAULT_TIMESTAMP: i64 = 1559156183943;
pub const MULTI_SIG_SYSTEM_VAULT_TIMESTAMP: i64 = 1571408470880;
pub const POS_GENERATOR_TIMESTAMP: i64 = 1559156420651;
pub const STACK_TIMESTAMP: i64 = 1751539590099;
pub const TOKEN_METADATA_TIMESTAMP: i64 = 1737500000000;

lazy_static! {
    pub static ref REGISTRY_PUB_KEY: PublicKey = to_public(REGISTRY_PK);
    pub static ref LIST_OPS_PUB_KEY: PublicKey = to_public(LIST_OPS_PK);
    pub static ref EITHER_PUB_KEY: PublicKey = to_public(EITHER_PK);
    pub static ref NON_NEGATIVE_NUMBER_PUB_KEY: PublicKey = to_public(NON_NEGATIVE_NUMBER_PK);
    pub static ref MAKE_MINT_PUB_KEY: PublicKey = to_public(MAKE_MINT_PK);
    pub static ref AUTH_KEY_PUB_KEY: PublicKey = to_public(AUTH_KEY_PK);
    pub static ref SYSTEM_VAULT_PUB_KEY: PublicKey = to_public(SYSTEM_VAULT_PK);
    pub static ref MULTI_SIG_SYSTEM_VAULT_PUB_KEY: PublicKey = to_public(MULTI_SIG_SYSTEM_VAULT_PK);
    pub static ref POS_GENERATOR_PUB_KEY: PublicKey = to_public(POS_GENERATOR_PK);
    pub static ref VAULTS_GENERATOR_PUB_KEY: PublicKey = to_public(VAULTS_GENERATOR_PK);
    pub static ref STACK_PUB_KEY: PublicKey = to_public(STACK_PK);
    pub static ref TOKEN_METADATA_PUB_KEY: PublicKey = to_public(TOKEN_METADATA_PK);
}

pub fn system_public_keys() -> Vec<&'static PublicKey> {
    vec![
        &REGISTRY_PUB_KEY,
        &LIST_OPS_PUB_KEY,
        &EITHER_PUB_KEY,
        &NON_NEGATIVE_NUMBER_PUB_KEY,
        &MAKE_MINT_PUB_KEY,
        &AUTH_KEY_PUB_KEY,
        &SYSTEM_VAULT_PUB_KEY,
        &MULTI_SIG_SYSTEM_VAULT_PUB_KEY,
        &POS_GENERATOR_PUB_KEY,
        &VAULTS_GENERATOR_PUB_KEY,
        &STACK_PUB_KEY,
        &TOKEN_METADATA_PUB_KEY,
    ]
}

fn to_deploy(
    compiled_source: CompiledRholangSource,
    private_key_hex: &str,
    timestamp: i64,
    shard_id: &str,
) -> Signed<DeployData> {
    let sk = PrivateKey::from_bytes(
        &hex::decode(private_key_hex).expect("Invalid private key hex string"),
    );

    let deploy_data = DeployData {
        time_stamp: timestamp,
        term: compiled_source.code,
        phlo_limit: i64::MAX, // Equivalent to accounting.MAX_VALUE in Scala
        phlo_price: 0,
        valid_after_block_number: 0,
        shard_id: shard_id.to_string(),
        expiration_timestamp: None,
    };

    Signed::create(deploy_data, Box::new(Secp256k1), sk).expect("Failed to create signed deploy")
}

pub fn registry(shard_id: &str) -> Signed<DeployData> {
    to_deploy(
        embedded_source("Registry.rho", embedded_rho::REGISTRY),
        REGISTRY_PK,
        REGISTRY_TIMESTAMP,
        shard_id,
    )
}

pub fn list_ops(shard_id: &str) -> Signed<DeployData> {
    to_deploy(
        embedded_source("ListOps.rho", embedded_rho::LIST_OPS),
        LIST_OPS_PK,
        LIST_OPS_TIMESTAMP,
        shard_id,
    )
}

pub fn either(shard_id: &str) -> Signed<DeployData> {
    to_deploy(
        embedded_source("Either.rho", embedded_rho::EITHER),
        EITHER_PK,
        EITHER_TIMESTAMP,
        shard_id,
    )
}

pub fn non_negative_number(shard_id: &str) -> Signed<DeployData> {
    to_deploy(
        embedded_source("NonNegativeNumber.rho", embedded_rho::NON_NEGATIVE_NUMBER),
        NON_NEGATIVE_NUMBER_PK,
        NON_NEGATIVE_NUMBER_TIMESTAMP,
        shard_id,
    )
}

pub fn make_mint(shard_id: &str) -> Signed<DeployData> {
    to_deploy(
        embedded_source("MakeMint.rho", embedded_rho::MAKE_MINT),
        MAKE_MINT_PK,
        MAKE_MINT_TIMESTAMP,
        shard_id,
    )
}

pub fn auth_key(shard_id: &str) -> Signed<DeployData> {
    to_deploy(
        embedded_source("AuthKey.rho", embedded_rho::AUTH_KEY),
        AUTH_KEY_PK,
        AUTH_KEY_TIMESTAMP,
        shard_id,
    )
}

pub fn system_vault(shard_id: &str) -> Signed<DeployData> {
    to_deploy(
        embedded_source("SystemVault.rho", embedded_rho::SYSTEM_VAULT),
        SYSTEM_VAULT_PK,
        SYSTEM_VAULT_TIMESTAMP,
        shard_id,
    )
}

pub fn multi_sig_system_vault(shard_id: &str) -> Signed<DeployData> {
    to_deploy(
        embedded_source("MultiSigSystemVault.rho", embedded_rho::MULTI_SIG_SYSTEM_VAULT),
        MULTI_SIG_SYSTEM_VAULT_PK,
        MULTI_SIG_SYSTEM_VAULT_TIMESTAMP,
        shard_id,
    )
}

pub fn stack(shard_id: &str) -> Signed<DeployData> {
    to_deploy(
        embedded_source("Stack.rho", embedded_rho::STACK),
        STACK_PK,
        STACK_TIMESTAMP,
        shard_id,
    )
}

/// Deploys the `TokenMetadata` contract that stores the native token's
/// name, symbol, and decimals. Values are substituted into the Rholang
/// source at genesis time and registered at `rho:system:tokenMetadata`.
pub fn token_metadata(
    native_token_name: &str,
    native_token_symbol: &str,
    native_token_decimals: u32,
    shard_id: &str,
) -> Signed<DeployData> {
    let decimals_str = native_token_decimals.to_string();
    to_deploy(
        CompiledRholangTemplate::new(
            "TokenMetadata.rhox",
            embedded_rho::TOKEN_METADATA,
            HashMap::new(),
            &[
                ("nativeTokenName", native_token_name),
                ("nativeTokenSymbol", native_token_symbol),
                ("nativeTokenDecimals", &decimals_str),
            ],
        ),
        TOKEN_METADATA_PK,
        TOKEN_METADATA_TIMESTAMP,
        shard_id,
    )
}

pub fn pos_generator(pos: &ProofOfStake, shard_id: &str) -> Signed<DeployData> {
    assert!(pos.minimum_bond <= pos.maximum_bond);
    assert!(pos.validators.len() > 0);

    to_deploy(
        CompiledRholangTemplate::new(
            "PoS.rhox",
            embedded_rho::POS,
            HashMap::new(),
            &[
                ("minimumBond", &pos.minimum_bond.to_string()),
                ("maximumBond", &pos.maximum_bond.to_string()),
                (
                    "initialBonds",
                    &ProofOfStake::initial_bonds(&pos.validators),
                ),
                ("epochLength", &pos.epoch_length.to_string()),
                ("quarantineLength", &pos.quarantine_length.to_string()),
                (
                    "numberOfActiveValidators",
                    &pos.number_of_active_validators.to_string(),
                ),
                (
                    "posMultiSigPublicKeys",
                    &ProofOfStake::public_keys(&pos.pos_multi_sig_public_keys),
                ),
                ("posMultiSigQuorum", &pos.pos_multi_sig_quorum.to_string()),
            ],
        ),
        POS_GENERATOR_PK,
        POS_GENERATOR_TIMESTAMP,
        shard_id,
    )
}

pub fn vaults_generator(
    vaults: Vec<Vault>,
    supply: i64,
    timestamp: i64,
    is_last_batch: bool,
    shard_id: &str,
) -> Signed<DeployData> {
    let vaults_generator = VaultsGenerator::create_from_user_vaults(vaults, supply, is_last_batch);
    to_deploy(
        CompiledRholangSource::new(
            vaults_generator.code,
            HashMap::new(),
            "<synthetic in VaultsGenerator.scala>".to_string(),
        )
        .expect("Failed to compile VaultsGenerator.rho"),
        VAULTS_GENERATOR_PK,
        timestamp,
        shard_id,
    )
}

pub fn to_public(priv_key_hex: &str) -> PublicKey {
    let private_key =
        PrivateKey::from_bytes(&hex::decode(priv_key_hex).expect("Invalid private key hex string"));
    let secp256k1 = Secp256k1;
    secp256k1.to_public(&private_key)
}
