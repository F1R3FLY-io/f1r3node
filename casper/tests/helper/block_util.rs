// See casper/src/test/scala/coop/rchain/casper/helper/BlockUtil.scala

use models::rust::validator::{self, Validator};
use rand::Rng;

pub fn generate_validator(prefix: Option<&str>) -> Validator {
    let prefix_bytes = prefix.unwrap_or("").as_bytes();
    assert!(
        prefix_bytes.len() <= validator::LENGTH,
        "Prefix too long for validator length"
    );

    let mut array = [0u8; validator::LENGTH];
    array[..prefix_bytes.len()].copy_from_slice(prefix_bytes);
    rand::rng().fill(&mut array[prefix_bytes.len()..]);
    prost::bytes::Bytes::copy_from_slice(&array)
}
