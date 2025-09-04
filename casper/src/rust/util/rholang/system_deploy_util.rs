// See casper/src/main/scala/coop/rchain/casper/util/rholang/SystemDeployUtil.scala

use byteorder::{LittleEndian, WriteBytesExt};
use crypto::rust::{
    hash::blake2b512_random::Blake2b512Random, public_key::PublicKey, signatures::signed::Signed,
};
use models::rust::{casper::protocol::casper_message::DeployData, validator::Validator};

use super::tools::Tools;

// Currently we have 4 system deploys -> refund, preCharge, closeBlock, Slashing
// In every user deploy, the rnode would do the preCharge first, then execute the
// user deploy and do the refund at last.
//
// The refund and preCharge system deploy
// would use user deploy signature to generate the system deploy. The random seed of
// the refund and preCharge has to be exactly the same to make sure replay the user
// deploy would come out the exact same result.
//
// As for closeBlock and slashing, the rnode would execute closeBlock system deploy in every block.
// So for a block seed it would be enough to have:
// PREFIX ++ PublicKey ++ seqNum serialized with protobuf.
// This way we can be completely sure that collision cannot happen.
// (Quote: https://github.com/rchain/rchain/pull/2879#discussion_r378948921)

const SYSTEM_DEPLOY_PREFIX: i32 = 1;

fn serialize_int32_fixed(value: i32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4);
    buf.write_i32::<LittleEndian>(value)
        .expect("Failed to write bytes");
    buf
}

pub fn generate_system_deploy_random_seed(sender: Validator, seq_num: i32) -> Blake2b512Random {
    let seed: Vec<u8> = serialize_int32_fixed(SYSTEM_DEPLOY_PREFIX)
        .into_iter()
        .chain(sender)
        .chain(serialize_int32_fixed(seq_num))
        .collect();

    Tools::rng(&seed)
}

pub fn generate_close_deploy_random_seed_from_validator(
    validator: Validator,
    seq_num: i32,
) -> Blake2b512Random {
    generate_system_deploy_random_seed(validator, seq_num).split_byte(0)
}

pub fn generate_close_deploy_random_seed_from_pk(pk: PublicKey, seq_num: i32) -> Blake2b512Random {
    let sender = pk.bytes;
    generate_close_deploy_random_seed_from_validator(sender, seq_num)
}

pub fn generate_slash_deploy_random_seed(validator: Validator, seq_num: i32) -> Blake2b512Random {
    generate_system_deploy_random_seed(validator, seq_num).split_byte(1)
}

pub fn generate_pre_charge_deploy_random_seed(deploy: &Signed<DeployData>) -> Blake2b512Random {
    Tools::rng(&deploy.sig).split_byte(0)
}

pub fn generate_refund_deploy_random_seed(deploy: &Signed<DeployData>) -> Blake2b512Random {
    Tools::rng(&deploy.sig).split_byte(1)
}
