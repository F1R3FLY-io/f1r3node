// See casper/src/main/scala/coop/rchain/casper/genesis/contracts/Vault.scala

use rholang::rust::interpreter::util::rev_address::RevAddress;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Vault {
    pub rev_address: RevAddress,
    pub initial_balance: u64,
}
