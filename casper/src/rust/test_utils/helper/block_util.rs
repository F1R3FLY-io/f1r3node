// See casper/src/test/scala/coop/rchain/casper/helper/BlockUtil.scala
// Moved from casper/tests/helper/block_util.rs to casper/src/rust/test_utils/helper/block_util.rs
// All imports fixed for library crate context

use crypto::rust::private_key::PrivateKey;
use crypto::rust::signatures::signatures_alg::SignaturesAlgFactory;
use models::rust::block_hash;
use models::rust::block_hash::BlockHash;
use models::rust::casper::protocol::casper_message::BlockMessage;
use models::rust::validator::{self, Validator};
use prost::bytes::Bytes;
use rand::Rng;

pub fn resign_block(b: &BlockMessage, sk: &PrivateKey) -> BlockMessage {
    let sign_function =
        SignaturesAlgFactory::apply(&b.sig_algorithm).expect("Failed to get signature algorithm");

    let block_hash = crate::rust::util::proto_util::hash_block(b);

    let sig = sign_function.sign(&block_hash, &sk.bytes);

    let mut new_block = b.clone();
    new_block.block_hash = block_hash;
    new_block.sig = prost::bytes::Bytes::from(sig);
    new_block
}

pub fn generate_validator(prefix: Option<&str>) -> Validator {
    let prefix_bytes = prefix.unwrap_or("").as_bytes();
    assert!(
        prefix_bytes.len() <= validator::LENGTH,
        "Prefix too long for validator length"
    );

    let mut array = [0u8; validator::LENGTH];
    array[..prefix_bytes.len()].copy_from_slice(prefix_bytes);
    rand::rng().fill(&mut array[prefix_bytes.len()..]);
    Bytes::copy_from_slice(&array)
}

pub fn generate_hash(prefix: Option<&str>) -> BlockHash {
    let _prefix = prefix.unwrap_or(""); // prefix is in signature but not used in Scala
    let mut array = [0u8; block_hash::LENGTH];
    rand::rng().fill(&mut array);
    Bytes::copy_from_slice(&array)
}


