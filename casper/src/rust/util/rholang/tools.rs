// See casper/src/main/scala/coop/rchain/casper/util/rholang/Tools.scala

use crypto::rust::{hash::blake2b512_random::Blake2b512Random, public_key::PublicKey};
use models::casper::DeployDataProto;
use prost::Message;

pub struct Tools;

impl Tools {
    pub fn unforgeable_name_rng(deployer: &PublicKey, timestamp: i64) -> Blake2b512Random {
        let seed = DeployDataProto {
            deployer: deployer.bytes.clone(),
            timestamp,
            ..Default::default()
        };

        Blake2b512Random::create_from_bytes(&seed.encode_to_vec())
    }

    pub fn rng(signature: &[u8]) -> Blake2b512Random {
        Blake2b512Random::create_from_bytes(signature)
    }
}
