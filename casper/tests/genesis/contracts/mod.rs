use std::time::Duration;

pub mod test_util;

pub mod auth_key_spec;
pub mod block_data_contract_spec;
pub mod either_spec;
pub mod failing_result_collector_spec;
pub mod list_ops_spec;
pub mod make_mint_spec;
pub mod multi_sig_rev_vault_spec;
pub mod non_negative_number_spec;
pub mod pos_spec;
pub mod registry_ops_spec;
pub mod registry_spec;
pub mod rev_address_spec;
pub mod rev_issuance_test;
pub mod rev_vault_spec;
pub mod rho_spec_contract_spec;
pub mod stack_spec;
pub mod standard_deploys_spec;
pub mod timeout_result_collector_spec;
pub mod tree_hash_map_spec;

// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/package.scala
pub const GENESIS_TEST_TIMEOUT: Duration = Duration::from_secs(60);
