use serde::{Deserialize, Serialize};
use std::{path::PathBuf, time::Duration};

/// Casper configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CasperConf {
    #[serde(rename = "fault-tolerance-threshold")]
    pub fault_tolerance_threshold: f32,

    #[serde(rename = "validator-public-key")]
    pub validator_public_key: Option<String>,
    #[serde(rename = "validator-private-key")]
    pub validator_private_key: Option<String>,
    #[serde(rename = "validator-private-key-path")]
    pub validator_private_key_path: Option<PathBuf>,

    #[serde(rename = "shard-name")]
    pub shard_name: String,
    #[serde(rename = "parent-shard-id")]
    pub parent_shard_id: String,

    #[serde(rename = "casper-loop-interval", deserialize_with = "de_duration")]
    pub casper_loop_interval: Duration,
    #[serde(rename = "requested-blocks-timeout", deserialize_with = "de_duration")]
    pub requested_blocks_timeout: Duration,
    #[serde(rename = "finalization-rate")]
    pub finalization_rate: u32,
    #[serde(rename = "max-number-of-parents")]
    pub max_number_of_parents: u32,
    #[serde(rename = "max-parent-depth")]
    pub max_parent_depth: u32,
    #[serde(
        rename = "fork-choice-stale-threshold",
        deserialize_with = "de_duration"
    )]
    pub fork_choice_stale_threshold: Duration,
    #[serde(
        rename = "fork-choice-check-if-stale-interval",
        deserialize_with = "de_duration"
    )]
    pub fork_choice_check_if_stale_interval: Duration,
    #[serde(rename = "synchrony-constraint-threshold")]
    pub synchrony_constraint_threshold: f32,
    #[serde(rename = "height-constraint-threshold")]
    pub height_constraint_threshold: u64,

    #[serde(rename = "round-robin-dispatcher")]
    pub round_robin_dispatcher: RoundRobinDispatcher,

    #[serde(rename = "genesis-block-data")]
    pub genesis_block_data: GenesisBlockData,

    #[serde(rename = "genesis-ceremony")]
    pub genesis_ceremony: GenesisCeremony,

    #[serde(rename = "min-phlo-price")]
    pub min_phlo_price: u64,
}

/// Round robin dispatcher configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundRobinDispatcher {
    #[serde(rename = "max-peer-queue-size")]
    pub max_peer_queue_size: u32,
    #[serde(rename = "give-up-after-skipped")]
    pub give_up_after_skipped: u32,
    #[serde(rename = "drop-peer-after-retries")]
    pub drop_peer_after_retries: u32,
}

/// Genesis block data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisBlockData {
    #[serde(rename = "genesis-data-dir")]
    pub genesis_data_dir: String,
    #[serde(rename = "bonds-file")]
    pub bonds_file: String,
    #[serde(rename = "wallets-file")]
    pub wallets_file: String,

    #[serde(rename = "bond-minimum")]
    pub bond_minimum: u64,
    #[serde(rename = "bond-maximum")]
    pub bond_maximum: u64,

    #[serde(rename = "epoch-length")]
    pub epoch_length: u32,
    #[serde(rename = "quarantine-length")]
    pub quarantine_length: u32,

    #[serde(rename = "number-of-active-validators")]
    pub number_of_active_validators: u32,

    #[serde(rename = "genesis-block-number")]
    pub genesis_block_number: u64,

    #[serde(rename = "pos-multi-sig-public-keys")]
    pub pos_multi_sig_public_keys: Vec<String>,

    #[serde(rename = "pos-multi-sig-quorum")]
    pub pos_multi_sig_quorum: u32,
}

/// Genesis ceremony configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisCeremony {
    #[serde(rename = "required-signatures")]
    pub required_signatures: u32,

    #[serde(rename = "approve-interval", deserialize_with = "de_duration")]
    pub approve_interval: Duration,

    #[serde(rename = "approve-duration", deserialize_with = "de_duration")]
    pub approve_duration: Duration,

    #[serde(rename = "autogen-shard-size")]
    pub autogen_shard_size: u32,

    #[serde(rename = "genesis-validator-mode")]
    pub genesis_validator_mode: bool,

    #[serde(rename = "ceremony-master-mode")]
    pub ceremony_master_mode: bool,
}

pub fn de_duration<'de, D>(deserializer: D) -> Result<Duration, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as _;
    use serde::Deserialize;
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum DurIn {
        Str(String),
        Secs(u64),
        FloatSecs(f64),
    }
    match DurIn::deserialize(deserializer)? {
        DurIn::Str(s) => humantime::parse_duration(&s)
            .map_err(|e| D::Error::custom(format!("invalid duration {s:?}: {e}"))),
        DurIn::Secs(n) => Ok(Duration::from_secs(n)),
        DurIn::FloatSecs(f) => {
            if f < 0.0 {
                return Err(D::Error::custom("negative duration"));
            }
            Ok(Duration::from_secs_f64(f))
        }
    }
}
