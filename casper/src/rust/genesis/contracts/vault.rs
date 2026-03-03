// See casper/src/main/scala/coop/rchain/casper/genesis/contracts/Vault.scala

use rholang::rust::interpreter::util::vault_address::VaultAddress as RevAddress;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Vault {
    pub rev_address: RevAddress,
    pub initial_balance: u64,
}
