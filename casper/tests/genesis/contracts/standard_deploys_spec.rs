// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/StandardDeploysSpec.scala

use casper::rust::genesis::contracts::standard_deploys;

#[test]
fn should_print_public_keys_used_for_signing_standard_blessed_contracts() {
    println!("Public keys used to sign standard (blessed) contracts");
    println!("=====================================================");

    for (idx, pub_key) in standard_deploys::system_public_keys()
        .iter()
        .enumerate()
    {
        println!("{}. {}", idx + 1, hex::encode(&pub_key.bytes));
    }
}
