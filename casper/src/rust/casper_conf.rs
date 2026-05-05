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
    pub finalization_rate: i32,
    #[serde(rename = "max-number-of-parents")]
    pub max_number_of_parents: i32,
    #[serde(rename = "max-parent-depth")]
    pub max_parent_depth: i32,
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
    pub height_constraint_threshold: i64,

    #[serde(rename = "round-robin-dispatcher")]
    pub round_robin_dispatcher: RoundRobinDispatcher,

    #[serde(rename = "genesis-block-data")]
    pub genesis_block_data: GenesisBlockData,

    #[serde(rename = "genesis-ceremony")]
    pub genesis_ceremony: GenesisCeremony,

    #[serde(rename = "min-phlo-price")]
    pub min_phlo_price: i64,

    #[serde(rename = "heartbeat")]
    pub heartbeat_conf: HeartbeatConf,

    #[serde(rename = "finalizer", default)]
    pub finalizer: FinalizerConf,

    #[serde(
        rename = "synchrony-recovery-stall-window",
        deserialize_with = "de_duration",
        default = "default_synchrony_recovery_stall_window"
    )]
    pub synchrony_recovery_stall_window: Duration,
    #[serde(
        rename = "synchrony-recovery-cooldown",
        deserialize_with = "de_duration",
        default = "default_synchrony_recovery_cooldown"
    )]
    pub synchrony_recovery_cooldown: Duration,
    #[serde(
        rename = "synchrony-recovery-max-bypasses",
        default = "default_synchrony_recovery_max_bypasses"
    )]
    pub synchrony_recovery_max_bypasses: u32,
    #[serde(
        rename = "synchrony-finalized-baseline-enabled",
        default = "default_synchrony_finalized_baseline_enabled"
    )]
    pub synchrony_finalized_baseline_enabled: bool,
    #[serde(
        rename = "synchrony-finalized-baseline-max-distance",
        default = "default_synchrony_finalized_baseline_max_distance"
    )]
    pub synchrony_finalized_baseline_max_distance: u64,

    #[serde(
        rename = "max-user-deploys-per-block",
        default = "default_max_user_deploys_per_block"
    )]
    pub max_user_deploys_per_block: u32,

    /// Disable late block filtering in DagMerger.
    /// When true (default), all blocks are included in merged state regardless of when
    /// they were observed. This prevents deploy loss during network partitions.
    #[serde(
        rename = "disable-late-block-filtering",
        default = "default_disable_late_block_filtering"
    )]
    pub disable_late_block_filtering: bool,

    /// Enable background garbage collection for mergeable channels.
    /// When enabled, uses safe reachability-based GC (required for multi-parent mode).
    /// When disabled (default), mergeable data is retained.
    #[serde(
        rename = "enable-mergeable-channel-gc",
        default = "default_enable_mergeable_channel_gc"
    )]
    pub enable_mergeable_channel_gc: bool,

    /// Interval for garbage collecting mergeable channels (only when GC enabled).
    /// Background process that safely deletes mergeable data when provably unreachable.
    #[serde(
        rename = "mergeable-channels-gc-interval",
        deserialize_with = "de_duration",
        default = "default_mergeable_channels_gc_interval"
    )]
    pub mergeable_channels_gc_interval: Duration,

    /// Depth buffer for mergeable channels garbage collection (only when GC enabled).
    /// Additional safety margin beyond max-parent-depth before deleting data.
    #[serde(
        rename = "mergeable-channels-gc-depth-buffer",
        default = "default_mergeable_channels_gc_depth_buffer"
    )]
    pub mergeable_channels_gc_depth_buffer: i32,
}

fn default_synchrony_recovery_stall_window() -> Duration {
    Duration::from_secs(60)
}

fn default_synchrony_recovery_cooldown() -> Duration {
    Duration::from_secs(20)
}

fn default_synchrony_recovery_max_bypasses() -> u32 {
    2
}

fn default_synchrony_finalized_baseline_enabled() -> bool {
    true
}

fn default_synchrony_finalized_baseline_max_distance() -> u64 {
    2048
}

fn default_max_user_deploys_per_block() -> u32 {
    32
}

fn default_disable_late_block_filtering() -> bool {
    true
}

fn default_enable_mergeable_channel_gc() -> bool {
    false
}

fn default_mergeable_channels_gc_interval() -> Duration {
    Duration::from_secs(5 * 60) // 5 minutes
}

fn default_mergeable_channels_gc_depth_buffer() -> i32 {
    10
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
    pub bond_minimum: i64,
    #[serde(rename = "bond-maximum")]
    pub bond_maximum: i64,

    #[serde(rename = "epoch-length")]
    pub epoch_length: i32,
    #[serde(rename = "quarantine-length")]
    pub quarantine_length: i32,

    #[serde(rename = "number-of-active-validators")]
    pub number_of_active_validators: u32,

    #[serde(rename = "deploy-timestamp")]
    pub deploy_timestamp: Option<i64>,

    #[serde(rename = "genesis-block-number")]
    pub genesis_block_number: i64,

    #[serde(rename = "pos-multi-sig-public-keys")]
    pub pos_multi_sig_public_keys: Vec<String>,

    #[serde(rename = "pos-multi-sig-quorum")]
    pub pos_multi_sig_quorum: u32,

    /// Full display name of the native token. Substituted into the
    /// TokenMetadata Rholang contract at genesis and registered at
    /// `rho:system:tokenMetadata`. Immutable after genesis.
    #[serde(rename = "native-token-name")]
    pub native_token_name: String,

    /// Ticker symbol of the native token. Immutability rules are identical
    /// to `native-token-name`. Operators MUST set this in config before genesis.
    #[serde(rename = "native-token-symbol")]
    pub native_token_symbol: String,

    /// Number of decimal places used to display the native token
    /// (1 token = 10^decimals dust). Immutability rules are identical to
    /// `native-token-name`. Operators MUST set this in config before genesis.
    #[serde(rename = "native-token-decimals")]
    pub native_token_decimals: u32,
}

/// Maximum decimal places accepted for native token. Matches the de-facto
/// ERC-20 standard (ETH uses 18). Values above 18 exceed IEEE-754 double
/// safe-integer range (2^53), which breaks every JavaScript-based wallet
/// and block explorer. No production blockchain uses more than 18
/// (BTC=8, SOL=9, ATOM=6, DOT=10, KSM=12, ETH=18).
pub const MAX_NATIVE_TOKEN_DECIMALS: u32 = 18;

impl GenesisBlockData {
    /// Validates native-token-* fields. Called during config load so a
    /// misconfigured node fails startup loudly rather than baking bad
    /// values into genesis or serving misleading metadata via `/api/status`.
    pub fn validate_native_token(&self) -> Result<(), String> {
        if self.native_token_name.trim().is_empty() {
            return Err(format!(
                "native-token-name must be non-empty and non-whitespace; got {:?}",
                self.native_token_name
            ));
        }
        if self.native_token_symbol.trim().is_empty() {
            return Err(format!(
                "native-token-symbol must be non-empty and non-whitespace; got {:?}",
                self.native_token_symbol
            ));
        }
        if self.native_token_decimals > MAX_NATIVE_TOKEN_DECIMALS {
            return Err(format!(
                "native-token-decimals={} exceeds maximum of {} (industry standard; \
                 ETH=18, BTC=8, SOL=9, ATOM=6); values above {} exceed IEEE-754 \
                 double safe-integer range and break JavaScript clients",
                self.native_token_decimals, MAX_NATIVE_TOKEN_DECIMALS, MAX_NATIVE_TOKEN_DECIMALS
            ));
        }
        Ok(())
    }
}

/// Genesis ceremony configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisCeremony {
    #[serde(rename = "required-signatures")]
    pub required_signatures: i32,

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeartbeatConf {
    pub enabled: bool,
    #[serde(rename = "check-interval", deserialize_with = "de_duration")]
    pub check_interval: Duration,
    #[serde(rename = "max-lfb-age", deserialize_with = "de_duration")]
    pub max_lfb_age: Duration,
    #[serde(
        rename = "self-propose-cooldown",
        deserialize_with = "de_duration",
        default = "default_self_propose_cooldown"
    )]
    pub self_propose_cooldown: Duration,
    /// Minimum age of LFB/frontier before stale-recovery, leader-recovery,
    /// and pending-deploy backstop are allowed to fire. Debounces empty-block
    /// churn when the cluster is healthy.
    #[serde(
        rename = "stale-recovery-min-interval",
        deserialize_with = "de_duration",
        default = "default_stale_recovery_min_interval"
    )]
    pub stale_recovery_min_interval: Duration,
    /// When pending deploys land, opens a grace window during which lag caps
    /// relax to `advanced.deploy_recovery_max_lag` and self-propose-cooldown
    /// is bypassable. Burst-tolerance budget.
    #[serde(
        rename = "deploy-finalization-grace",
        deserialize_with = "de_duration",
        default = "default_deploy_finalization_grace"
    )]
    pub deploy_finalization_grace: Duration,
    /// EXPERIMENTAL tuning knobs. See [`HeartbeatAdvancedConf`].
    #[serde(default)]
    pub advanced: HeartbeatAdvancedConf,
}

impl Default for HeartbeatConf {
    fn default() -> Self {
        Self {
            enabled: false,
            check_interval: Duration::from_secs(5),
            max_lfb_age: Duration::from_secs(5),
            self_propose_cooldown: default_self_propose_cooldown(),
            stale_recovery_min_interval: default_stale_recovery_min_interval(),
            deploy_finalization_grace: default_deploy_finalization_grace(),
            advanced: HeartbeatAdvancedConf::default(),
        }
    }
}

fn default_self_propose_cooldown() -> Duration {
    Duration::from_secs(15)
}

fn default_stale_recovery_min_interval() -> Duration {
    Duration::from_secs(12)
}

fn default_deploy_finalization_grace() -> Duration {
    Duration::from_secs(25)
}

/// EXPERIMENTAL: tuning knobs for the heartbeat proposer's lag caps.
///
/// These thresholds bound DAG width relative to replay cost in lieu of
/// adaptive backpressure. Treat as unstable API; field names may change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeartbeatAdvancedConf {
    /// When this validator is already ahead of LFB, how many blocks of lag
    /// tolerate before "frontier-follow" proposing is throttled. `0` =
    /// never frontier-chase while ahead unless deploy recovery is active
    /// (which raises this dynamically).
    #[serde(
        rename = "frontier-chase-max-lag",
        default = "default_frontier_chase_max_lag"
    )]
    pub frontier_chase_max_lag: i64,
    /// If the validator has pending deploys but is already > N blocks
    /// ahead of LFB, suppress pending-deploy proposing. Prevents lag
    /// amplification: more deploys → more blocks → wider DAG → slower
    /// finalization → still "ahead" → keeps proposing forever. Lower →
    /// harder load-relief valve.
    #[serde(
        rename = "pending-deploy-max-lag",
        default = "default_pending_deploy_max_lag"
    )]
    pub pending_deploy_max_lag: i64,
    /// During an active deploy-finalization grace window, the lag cap
    /// widens to this value. The "absolute safe lag during recovery"
    /// ceiling.
    #[serde(
        rename = "deploy-recovery-max-lag",
        default = "default_deploy_recovery_max_lag"
    )]
    pub deploy_recovery_max_lag: i64,
}

impl Default for HeartbeatAdvancedConf {
    fn default() -> Self {
        Self {
            frontier_chase_max_lag: default_frontier_chase_max_lag(),
            pending_deploy_max_lag: default_pending_deploy_max_lag(),
            deploy_recovery_max_lag: default_deploy_recovery_max_lag(),
        }
    }
}

fn default_frontier_chase_max_lag() -> i64 {
    0
}

fn default_pending_deploy_max_lag() -> i64 {
    20
}

fn default_deploy_recovery_max_lag() -> i64 {
    64
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalizerConf {
    #[serde(
        rename = "work-budget",
        deserialize_with = "de_duration",
        default = "default_finalizer_work_budget"
    )]
    pub work_budget: Duration,
    #[serde(
        rename = "step-timeout",
        deserialize_with = "de_duration",
        default = "default_finalizer_step_timeout"
    )]
    pub step_timeout: Duration,
    #[serde(
        rename = "catchup-work-budget",
        deserialize_with = "de_duration",
        default = "default_finalizer_catchup_work_budget"
    )]
    pub catchup_work_budget: Duration,
    #[serde(
        rename = "catchup-step-timeout",
        deserialize_with = "de_duration",
        default = "default_finalizer_catchup_step_timeout"
    )]
    pub catchup_step_timeout: Duration,
}

impl Default for FinalizerConf {
    fn default() -> Self {
        Self {
            work_budget: default_finalizer_work_budget(),
            step_timeout: default_finalizer_step_timeout(),
            catchup_work_budget: default_finalizer_catchup_work_budget(),
            catchup_step_timeout: default_finalizer_catchup_step_timeout(),
        }
    }
}

fn default_finalizer_work_budget() -> Duration {
    Duration::from_secs(8)
}

fn default_finalizer_step_timeout() -> Duration {
    Duration::from_secs(1)
}

fn default_finalizer_catchup_work_budget() -> Duration {
    Duration::from_secs(8)
}

fn default_finalizer_catchup_step_timeout() -> Duration {
    Duration::from_secs(1)
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
