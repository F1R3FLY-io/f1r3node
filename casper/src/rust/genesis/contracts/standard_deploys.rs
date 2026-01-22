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

use super::{proof_of_stake::ProofOfStake, rev_generator::RevGenerator, vault::Vault};

// Private keys used to sign blessed (standard) contracts
pub const REGISTRY_PK: &str = "5a0bde2f5857124b1379c78535b07a278e3b9cefbcacc02e62ab3294c02765a1";
pub const LIST_OPS_PK: &str = "867c21c6a3245865444d80e49cac08a1c11e23b35965b566bbe9f49bb9897511";
pub const EITHER_PK: &str = "5248f8913f8572d8227a3c7787b54bd8263389f7209adc1422e36bb2beb160dc";
pub const NON_NEGATIVE_NUMBER_PK: &str =
    "e33c9f1e925819d04733db4ec8539a84507c9e9abd32822059349449fe03997d";
pub const MAKE_MINT_PK: &str = "de19d53f28d4cdee74bad062342d8486a90a652055f3de4b2efa5eb2fccc9d53";
pub const AUTH_KEY_PK: &str = "f450b26bac63e5dd9343cd46f5fae1986d367a893cd21eedd98a4cb3ac699abc";
pub const REV_VAULT_PK: &str = "27e5718bf55dd673cc09f13c2bcf12ed7949b178aef5dcb6cd492ad422d05e9d";
pub const MULTI_SIG_REV_VAULT_PK: &str =
    "2a2eaa76d6fea9f502629e32b0f8eea19b9de8e2188ec0d589fcafa98fb1f031";
pub const POS_GENERATOR_PK: &str =
    "a9585a0687761139ab3587a4938fb5ab9fcba675c79fefba889859674046d4a5";
pub const REV_GENERATOR_PK: &str =
    "a06959868e39bb3a8502846686a23119716ecd001700baf9e2ecfa0dbf1a3247";
pub const STACK_PK: &str = "c94e647de6876c954ebb7b64c40a220227770f9be003635edfe3336a1a2c8605";

// Timestamps for each deploy
pub const REGISTRY_TIMESTAMP: i64 = 1559156071321;
pub const LIST_OPS_TIMESTAMP: i64 = 1559156082324;
pub const EITHER_TIMESTAMP: i64 = 1559156217509;
pub const NON_NEGATIVE_NUMBER_TIMESTAMP: i64 = 1559156251792;
pub const MAKE_MINT_TIMESTAMP: i64 = 1559156452968;
pub const AUTH_KEY_TIMESTAMP: i64 = 1559156356769;
pub const REV_VAULT_TIMESTAMP: i64 = 1559156183943;
pub const MULTI_SIG_REV_VAULT_TIMESTAMP: i64 = 1571408470880;
pub const POS_GENERATOR_TIMESTAMP: i64 = 1559156420651;
pub const STACK_TIMESTAMP: i64 = 1751539590099;

lazy_static! {
    pub static ref REGISTRY_PUB_KEY: PublicKey = to_public(REGISTRY_PK);
    pub static ref LIST_OPS_PUB_KEY: PublicKey = to_public(LIST_OPS_PK);
    pub static ref EITHER_PUB_KEY: PublicKey = to_public(EITHER_PK);
    pub static ref NON_NEGATIVE_NUMBER_PUB_KEY: PublicKey = to_public(NON_NEGATIVE_NUMBER_PK);
    pub static ref MAKE_MINT_PUB_KEY: PublicKey = to_public(MAKE_MINT_PK);
    pub static ref AUTH_KEY_PUB_KEY: PublicKey = to_public(AUTH_KEY_PK);
    pub static ref REV_VAULT_PUB_KEY: PublicKey = to_public(REV_VAULT_PK);
    pub static ref MULTI_SIG_REV_VAULT_PUB_KEY: PublicKey = to_public(MULTI_SIG_REV_VAULT_PK);
    pub static ref POS_GENERATOR_PUB_KEY: PublicKey = to_public(POS_GENERATOR_PK);
    pub static ref REV_GENERATOR_PUB_KEY: PublicKey = to_public(REV_GENERATOR_PK);
    pub static ref STACK_PUB_KEY: PublicKey = to_public(STACK_PK);
}

pub fn system_public_keys() -> Vec<&'static PublicKey> {
    vec![
        &REGISTRY_PUB_KEY,
        &LIST_OPS_PUB_KEY,
        &EITHER_PUB_KEY,
        &NON_NEGATIVE_NUMBER_PUB_KEY,
        &MAKE_MINT_PUB_KEY,
        &AUTH_KEY_PUB_KEY,
        &REV_VAULT_PUB_KEY,
        &MULTI_SIG_REV_VAULT_PUB_KEY,
        &POS_GENERATOR_PUB_KEY,
        &REV_GENERATOR_PUB_KEY,
        &STACK_PUB_KEY,
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
        language: "rholang".to_string(),
    };

    Signed::create(deploy_data, Box::new(Secp256k1), sk).expect("Failed to create signed deploy")
}

pub fn registry(shard_id: &str) -> Signed<DeployData> {
    to_deploy(
        CompiledRholangSource::apply("Registry.rho").expect("Failed to compile Registry.rho"),
        REGISTRY_PK,
        REGISTRY_TIMESTAMP,
        shard_id,
    )
}

pub fn list_ops(shard_id: &str) -> Signed<DeployData> {
    to_deploy(
        CompiledRholangSource::apply("ListOps.rho").expect("Failed to compile ListOps.rho"),
        LIST_OPS_PK,
        LIST_OPS_TIMESTAMP,
        shard_id,
    )
}

pub fn either(shard_id: &str) -> Signed<DeployData> {
    to_deploy(
        CompiledRholangSource::apply("Either.rho").expect("Failed to compile Either.rho"),
        EITHER_PK,
        EITHER_TIMESTAMP,
        shard_id,
    )
}

pub fn non_negative_number(shard_id: &str) -> Signed<DeployData> {
    to_deploy(
        CompiledRholangSource::apply("NonNegativeNumber.rho")
            .expect("Failed to compile NonNegativeNumber.rho"),
        NON_NEGATIVE_NUMBER_PK,
        NON_NEGATIVE_NUMBER_TIMESTAMP,
        shard_id,
    )
}

pub fn make_mint(shard_id: &str) -> Signed<DeployData> {
    to_deploy(
        CompiledRholangSource::apply("MakeMint.rho").expect("Failed to compile MakeMint.rho"),
        MAKE_MINT_PK,
        MAKE_MINT_TIMESTAMP,
        shard_id,
    )
}

pub fn auth_key(shard_id: &str) -> Signed<DeployData> {
    to_deploy(
        CompiledRholangSource::apply("AuthKey.rho").expect("Failed to compile AuthKey.rho"),
        AUTH_KEY_PK,
        AUTH_KEY_TIMESTAMP,
        shard_id,
    )
}

pub fn rev_vault(shard_id: &str) -> Signed<DeployData> {
    to_deploy(
        CompiledRholangSource::apply("RevVault.rho").expect("Failed to compile RevVault.rho"),
        REV_VAULT_PK,
        REV_VAULT_TIMESTAMP,
        shard_id,
    )
}

pub fn multi_sig_rev_vault(shard_id: &str) -> Signed<DeployData> {
    to_deploy(
        CompiledRholangSource::apply("MultiSigRevVault.rho")
            .expect("Failed to compile MultiSigRevVault.rho"),
        MULTI_SIG_REV_VAULT_PK,
        MULTI_SIG_REV_VAULT_TIMESTAMP,
        shard_id,
    )
}

pub fn stack(shard_id: &str) -> Signed<DeployData> {
    to_deploy(
        CompiledRholangSource::apply("Stack.rho").expect("Failed to compile Stack.rho"),
        STACK_PK,
        STACK_TIMESTAMP,
        shard_id,
    )
}

pub fn pos_generator(pos: &ProofOfStake, shard_id: &str) -> Signed<DeployData> {
    assert!(pos.minimum_bond <= pos.maximum_bond);
    assert!(pos.validators.len() > 0);

    to_deploy(
        CompiledRholangTemplate::new(
            "PoS.rhox",
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

pub fn rev_generator(
    vaults: Vec<Vault>,
    supply: i64,
    timestamp: i64,
    is_last_batch: bool,
    shard_id: &str,
) -> Signed<DeployData> {
    let rev_generator = RevGenerator::create_from_user_vaults(vaults, supply, is_last_batch);
    to_deploy(
        CompiledRholangSource::new(
            rev_generator.code,
            HashMap::new(),
            "<synthetic in Rev.scala>".to_string(),
        )
        .expect("Failed to compile RevGenerator.rho"),
        REV_GENERATOR_PK,
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
