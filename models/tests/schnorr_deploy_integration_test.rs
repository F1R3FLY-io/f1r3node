#![cfg(feature = "schnorr_secp256k1_experimental")]

use crypto::rust::signatures::{
    frost_secp256k1::FrostSecp256k1, schnorr_secp256k1::SchnorrSecp256k1,
    signatures_alg::SignaturesAlg, signed::Signed,
};
use models::rust::casper::protocol::casper_message::DeployData;

fn sample_deploy() -> DeployData {
    DeployData {
        term: "@\"rho:io:stdout\"!(\"schnorr-deploy\")".to_string(),
        time_stamp: 1_773_865_000_000,
        phlo_price: 1,
        phlo_limit: 1_000_000,
        valid_after_block_number: 1,
        shard_id: "root".to_string(),
        expiration_timestamp: Some(1_773_865_060_000),
    }
}

#[test]
fn schnorr_signed_deploy_roundtrip_via_proto() {
    let alg: Box<dyn SignaturesAlg> = Box::new(SchnorrSecp256k1);
    let (sk, _pk) = alg.new_key_pair();
    let deploy = sample_deploy();

    let signed = Signed::create(deploy.clone(), alg, sk).expect("signed deploy");
    let proto = DeployData::to_proto(signed.clone());
    assert_eq!(proto.sig_algorithm, "schnorr-secp256k1");

    let decoded = DeployData::from_proto(proto).expect("decoded deploy");
    assert_eq!(decoded.data, deploy);
    assert_eq!(decoded.sig_algorithm.name(), "schnorr-secp256k1");
}

#[test]
fn frost_named_signed_deploy_roundtrip_via_proto() {
    let alg: Box<dyn SignaturesAlg> = Box::new(FrostSecp256k1);
    let (sk, _pk) = alg.new_key_pair();
    let deploy = sample_deploy();

    let signed = Signed::create(deploy.clone(), alg, sk).expect("signed deploy");
    let proto = DeployData::to_proto(signed.clone());
    assert_eq!(proto.sig_algorithm, "frost-secp256k1");

    let decoded = DeployData::from_proto(proto).expect("decoded deploy");
    assert_eq!(decoded.data, deploy);
    assert_eq!(decoded.sig_algorithm.name(), "frost-secp256k1");
}

#[test]
fn tampered_payload_fails_signature_verification() {
    let alg: Box<dyn SignaturesAlg> = Box::new(SchnorrSecp256k1);
    let (sk, _pk) = alg.new_key_pair();
    let deploy = sample_deploy();
    let signed = Signed::create(deploy, alg, sk).expect("signed deploy");
    let mut proto = DeployData::to_proto(signed);

    proto.term.push_str(" // tampered");
    assert!(DeployData::from_proto(proto).is_err());
}
