// See casper/src/main/scala/coop/rchain/casper/CasperConf.scala

use std::path::PathBuf;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct CasperConf {
    pub fault_tolerance_threshold: f32,
    pub validator_public_key: Option<String>,
    pub validator_private_key: Option<String>,
    pub validator_private_key_path: Option<PathBuf>,
    pub shard_name: String,
    pub parent_shard_id: String,
    pub casper_loop_interval: Duration,
    pub requested_blocks_timeout: Duration,
    pub finalization_rate: i32,
    pub max_number_of_parents: i32,
    pub max_parent_depth: Option<i32>,
    pub fork_choice_stale_threshold: Duration,
    pub fork_choice_check_if_stale_interval: Duration,
    pub synchrony_constraint_threshold: f64,
    pub height_constraint_threshold: i64,
    pub round_robin_dispatcher: RoundRobinDispatcher,
    pub genesis_block_data: GenesisBlockData,
    pub genesis_ceremony: GenesisCeremonyConf,
    pub min_phlo_price: i64,
}

#[derive(Clone, Debug)]
pub struct GenesisBlockData {
    pub genesis_data_dir: PathBuf,
    pub bonds_file: String,
    pub wallets_file: String,
    pub bond_minimum: i64,
    pub bond_maximum: i64,
    pub epoch_length: i32,
    pub quarantine_length: i32,
    pub genesis_block_number: i64,
    pub number_of_active_validators: i32,
    pub deploy_timestamp: Option<i64>,
    pub pos_multi_sig_public_keys: Vec<String>,
    pub pos_multi_sig_quorum: i32,
}

#[derive(Clone, Debug)]
pub struct GenesisCeremonyConf {
    pub required_signatures: i32,
    pub approve_interval: Duration,
    pub approve_duration: Duration,
    pub autogen_shard_size: i32,
    pub genesis_validator_mode: bool,
    pub ceremony_master_mode: bool,
}

#[derive(Clone, Debug)]
pub struct RoundRobinDispatcher {
    pub max_peer_queue_size: i32,
    pub give_up_after_skipped: i32,
    pub drop_peer_after_retries: i32,
}
