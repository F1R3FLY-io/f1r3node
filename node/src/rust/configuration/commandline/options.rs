//! Command-line options definition using clap
//!
//! This module defines all command-line arguments and subcommands for the F1r3fly node.

use casper::rust::util::comm::listen_at_name::Name;
use clap::builder::ValueParser;
use clap::{ArgAction, Args, Parser, Subcommand};
use crypto::rust::private_key::PrivateKey;
use crypto::rust::public_key::PublicKey;
use humantime::parse_duration;
use std::path::PathBuf;
use std::time::Duration;

use super::converters::{NameConverter, PrivateKeyConverter, PublicKeyConverter, VecNameConverter};

pub const GRPC_INTERNAL_PORT: u16 = 40402;
pub const GRPC_EXTERNAL_PORT: u16 = 40401;

/// F1r3fly node command-line interface
#[derive(Parser)]
#[command(
    name = "f1r3fly",
    version = env!("CARGO_PKG_VERSION"),
    about = "F1r3fly node | gRPC client",
    long_about = "F1r3fly node implementation with gRPC client capabilities"
)]
pub struct Options {
    /// Remote gRPC host for client calls
    #[arg(long = "grpc-host", default_value = "localhost")]
    pub grpc_host: String,

    /// Remote gRPC port for client calls
    #[arg(short = 'p', long = "grpc-port")]
    pub grpc_port: Option<u16>,

    /// Max inbound gRPC message size for client calls
    #[arg(
        short = 's',
        long = "grpc-max-recv-message-size",
        default_value = "16777216"
    )]
    pub grpc_max_recv_message_size: u32,

    /// Predefined set of defaults to use: default or docker
    #[arg(long = "profile")]
    pub profile: Option<String>,

    #[command(subcommand)]
    pub subcommand: Option<OptionsSubCommand>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum OptionsSubCommand {
    Run(RunOptions),
    Eval {
        files: Vec<String>,
        print_unmatched_sends_only: bool,
        language: String,
    },
    Repl,
    Deploy {
        phlo_limit: i64,
        phlo_price: i64,
        valid_after_block: i64,
        #[arg(value_parser = ValueParser::new(PrivateKeyConverter::parse))]
        private_key: Option<PrivateKey>,
        private_key_path: Option<PathBuf>,
        location: String,
        shard_id: String,
    },
    FindDeploy {
        id: Vec<u8>,
    },
    Propose {
        print_unmatched_sends: bool,
    },
    ShowBlock {
        hash: String,
    },
    ShowBlocks {
        depth: i32,
    },
    VisualizeDag {
        depth: i32,
        show_justification_lines: bool,
    },
    MachineVerifiableDag,
    Keygen {
        path: PathBuf,
    },
    LastFinalizedBlock,
    IsFinalized {
        hash: String,
    },
    BondStatus {
        #[arg(value_parser = ValueParser::new(PublicKeyConverter::parse))]
        public_key: PublicKey,
    },
    Help,
    DataAtName {
        #[arg(value_parser = ValueParser::new(|s: &str| NameConverter::parse_with_type("pub", s)))]
        name: Name,
    },
    ContAtName {
        #[arg(value_parser = ValueParser::new(VecNameConverter::parse))]
        names: Vec<Name>,
    },
    Status,
}

/// Run subcommand - Start RNode server
#[derive(Args, Debug, Clone)]
pub struct RunOptions {
    /// Path to the configuration file for RNode server
    #[arg(short = 'c', long = "config-file")]
    pub config_file: Option<PathBuf>,

    /// Number of threads allocated for main scheduler
    #[arg(long = "thread-pool-size", hide = true)]
    pub thread_pool_size: Option<u32>,

    /// Start a stand-alone node
    #[arg(short = 's', long = "standalone", action = ArgAction::SetTrue)]
    pub standalone: Option<bool>,

    /// Address of RNode to bootstrap from when connecting to a network
    #[arg(short = 'b', long = "bootstrap")]
    pub bootstrap: Option<String>,

    /// ID of the F1r3fly network to connect to
    #[arg(long = "network-id")]
    pub network_id: Option<String>,

    /// Make node automatically trying to propose block after new block added or new deploy received
    #[arg(long = "autopropose", action = ArgAction::SetTrue)]
    pub autopropose: Option<bool>,

    /// Use this flag to disable UPnP
    #[arg(long = "no-upnp", action = ArgAction::SetTrue)]
    pub no_upnp: Option<bool>,

    /// Host IP address changes dynamically
    #[arg(long = "dynamic-ip", action = ArgAction::SetTrue)]
    pub dynamic_ip: Option<bool>,

    /// If node has to create genesis block but no bonds file is provided, bonds file with a list of random public keys is generated
    #[arg(long = "autogen-shard-size")]
    pub autogen_shard_size: Option<u32>,

    /// Disable the node to start from Last Finalized State, instead it will start from genesis
    #[arg(long = "disable-lfs", action = ArgAction::SetTrue)]
    pub disable_lfs: Option<bool>,

    /// Address to bind F1r3fly Protocol server
    #[arg(long = "host")]
    pub host: Option<String>,

    /// Use random ports in case F1r3fly Protocol port and/or Kademlia port are not free
    #[arg(long = "use-random-ports", action = ArgAction::SetTrue)]
    pub use_random_ports: Option<bool>,

    /// Allow connections to peers with private network addresses
    #[arg(long = "allow-private-addresses", action = ArgAction::SetTrue)]
    pub allow_private_addresses: Option<bool>,

    /// Disable the node respond to export state requests
    #[arg(long = "disable-state-exporter", action = ArgAction::SetTrue)]
    pub disable_state_exporter: Option<bool>,

    /// Default timeout for network calls
    #[arg(long = "network-timeout", value_parser = ValueParser::new(parse_duration))]
    pub network_timeout: Option<Duration>,

    /// Port used for node discovery based on Kademlia algorithm
    #[arg(long = "discovery-port", default_value = "40400")]
    pub discovery_port: Option<u16>,

    /// Peer discovery interval
    #[arg(long = "discovery-lookup-interval", value_parser = ValueParser::new(parse_duration))]
    pub discovery_lookup_interval: Option<Duration>,

    /// Peer discovery cleanup interval
    #[arg(long = "discovery-cleanup-interval", value_parser = ValueParser::new(parse_duration))]
    pub discovery_cleanup_interval: Option<Duration>,

    /// Check for first connection loop interval
    #[arg(long = "discovery-init-wait-loop-interval", value_parser = ValueParser::new(parse_duration))]
    pub discovery_init_wait_loop_interval: Option<Duration>,

    /// Peer discovery heartbeat batch size
    #[arg(long = "discovery-heartbeat-batch-size")]
    pub discovery_heartbeat_batch_size: Option<u32>,

    /// gRPC port serving F1r3fly Protocol messages
    #[arg(short = 'p', long = "protocol-port", default_value = "40404")]
    pub protocol_port: Option<u16>,

    /// Maximum message size for gRPC transport server
    #[arg(long = "protocol-grpc-max-recv-message-size")]
    pub protocol_grpc_max_recv_message_size: Option<u32>,

    /// Maximum size of messages that can be received via transport layer streams
    #[arg(long = "protocol-grpc-max-recv-stream-message-size")]
    pub protocol_grpc_max_recv_stream_message_size: Option<u32>,

    /// Chunk size for streaming packets between nodes
    #[arg(long = "protocol-grpc-stream-chunk-size")]
    pub protocol_grpc_stream_chunk_size: Option<u32>,

    /// Number of connected peers picked randomly for broadcasting and streaming
    #[arg(long = "protocol-max-connections")]
    pub protocol_max_connections: Option<u32>,

    /// Number of incoming message consumers
    #[arg(long = "protocol-max-message-consumers")]
    pub protocol_max_message_consumers: Option<u32>,

    /// Path to private key for TLS
    #[arg(short = 'k', long = "tls-key-path")]
    pub tls_key_path: Option<PathBuf>,

    /// Path to X.509 certificate for TLS
    #[arg(long = "tls-certificate-path")]
    pub tls_certificate_path: Option<PathBuf>,

    /// Use a non blocking secure random instance
    #[arg(long = "tls-secure-random-non-blocking", action = ArgAction::SetTrue)]
    pub tls_secure_random_non_blocking: Option<bool>,

    /// Address to bind API servers
    #[arg(long = "api-host")]
    pub api_host: Option<String>,

    /// Port for external gRPC API
    #[arg(short = 'e', long = "api-port-grpc-external", default_value = "40401")]
    pub api_port_grpc_external: Option<u16>,

    /// Port for internal gRPC API
    #[arg(short = 'i', long = "api-port-grpc-internal", default_value = "40402")]
    pub api_port_grpc_internal: Option<u16>,

    /// Maximum message size for gRPC API server
    #[arg(long = "api-grpc-max-recv-message-size")]
    pub api_grpc_max_recv_message_size: Option<u32>,

    /// Port for HTTP services
    #[arg(long = "api-port-http", default_value = "40403")]
    pub api_port_http: Option<u16>,

    /// Port for admin HTTP services
    #[arg(short = 'a', long = "api-port-admin-http", default_value = "40405")]
    pub api_port_admin_http: Option<u16>,

    /// The max block numbers you can acquire from api
    #[arg(long = "api-max-blocks-limit")]
    pub api_max_blocks_limit: Option<u32>,

    /// Use this flag to enable reporting endpoints
    #[arg(long = "api-enable-reporting", action = ArgAction::SetTrue)]
    pub api_enable_reporting: Option<bool>,

    /// Sets a custom keepalive time
    #[arg(long = "api-keep-alive-time", value_parser = ValueParser::new(parse_duration))]
    pub api_keep_alive_time: Option<Duration>,

    /// Sets a custom keepalive timeout
    #[arg(long = "api-keep-alive-timeout", value_parser = ValueParser::new(parse_duration))]
    pub api_keep_alive_timeout: Option<Duration>,

    /// The most aggressive keep-alive time clients are permitted to configure
    #[arg(long = "api-permit-keep-alive-time", value_parser = ValueParser::new(parse_duration))]
    pub api_permit_keep_alive_time: Option<Duration>,

    /// Sets a custom max connection idle time
    #[arg(long = "api-max-connection-idle", value_parser = ValueParser::new(parse_duration))]
    pub api_max_connection_idle: Option<Duration>,

    /// Sets a custom max connection age
    #[arg(long = "api-max-connection-age", value_parser = ValueParser::new(parse_duration))]
    pub api_max_connection_age: Option<Duration>,

    /// Sets a custom grace time for the graceful connection termination
    #[arg(long = "api-max-connection-age-grace", value_parser = ValueParser::new(parse_duration))]
    pub api_max_connection_age_grace: Option<Duration>,

    /// Path to data directory
    #[arg(long = "data-dir")]
    pub data_dir: Option<PathBuf>,

    /// Name of the shard this node is connected to
    #[arg(long = "shard-name")]
    pub shard_name: Option<String>,

    /// Float value representing that the node tolerates up to fault-tolerance-threshold fraction of the total weight to equivocate
    #[arg(long = "fault-tolerance-threshold")]
    pub fault_tolerance_threshold: Option<f32>,

    /// Base16 encoding of the public key to use for signing a proposed blocks
    #[arg(long = "validator-public-key")]
    pub validator_public_key: Option<String>,

    /// Base16 encoding of the private key to use for signing a proposed blocks
    #[arg(long = "validator-private-key", hide = true)]
    pub validator_private_key: Option<String>,

    /// Path to the base16 encoded private key to use for signing a proposed blocks
    #[arg(long = "validator-private-key-path")]
    pub validator_private_key_path: Option<PathBuf>,

    /// Interval for the casper loop to maintain requested blocks and missing dependent blocks
    #[arg(long = "casper-loop-interval", value_parser = ValueParser::new(parse_duration))]
    pub casper_loop_interval: Option<Duration>,

    /// Timeout for blocks requests
    #[arg(long = "requested-blocks-timeout", value_parser = ValueParser::new(parse_duration))]
    pub requested_blocks_timeout: Option<Duration>,

    /// Finalization is called every `n` blocks
    #[arg(long = "finalization-rate")]
    pub finalization_rate: Option<u32>,

    /// Maximum number of block parents
    #[arg(long = "max-number-of-parents")]
    pub max_number_of_parents: Option<u32>,

    /// Maximum depth of block parents
    #[arg(long = "max-parent-depth")]
    pub max_parent_depth: Option<u32>,

    /// Node will request for fork choice tips if the latest FCT is more then forkChoiceStaleThreshold old
    #[arg(long = "fork-choice-stale-threshold", value_parser = ValueParser::new(parse_duration))]
    pub fork_choice_stale_threshold: Option<Duration>,

    /// Interval for check if fork choice tip is stale
    #[arg(long = "fork-choice-check-if-stale-interval", value_parser = ValueParser::new(parse_duration))]
    pub fork_choice_check_if_stale_interval: Option<Duration>,

    /// Float value representing that the node waits until at least synchrony-constraint-threshold fraction of the validators proposed at least one block
    #[arg(long = "synchrony-constraint-threshold")]
    pub synchrony_constraint_threshold: Option<f32>,

    /// Long value representing how far ahead of the last finalized block the node is allowed to propose
    #[arg(long = "height-constraint-threshold")]
    pub height_constraint_threshold: Option<u64>,

    /// Fair round robin dispatcher individual peer packet queue size
    #[arg(long = "frrd-max-peer-queue-size")]
    pub frrd_max_peer_queue_size: Option<u32>,

    /// Fair round robin dispatcher give up and try next peer after skipped packets
    #[arg(long = "frrd-give-up-after-skipped")]
    pub frrd_give_up_after_skipped: Option<u32>,

    /// Fair round robin dispatcher drop inactive peer after round robin rounds
    #[arg(long = "frrd-drop-peer-after-retries")]
    pub frrd_drop_peer_after_retries: Option<u32>,

    /// Plain text file consisting of lines of the form `<pk> <stake>`, which defines the bond amounts for each validator at genesis
    #[arg(long = "bonds-file")]
    pub bonds_file: Option<String>,

    /// Plain text file consisting of lines of the form `<algorithm> <pk> <revBalance>`, which defines the Rev wallets that exist at genesis
    #[arg(long = "wallets-file")]
    pub wallets_file: Option<String>,

    /// Minimum bond accepted by the PoS contract in the genesis block
    #[arg(long = "bond-minimum")]
    pub bond_minimum: Option<u64>,

    /// Configure genesis blockNumber for hard fork
    #[arg(long = "genesis-block-number")]
    pub genesis_block_number: Option<u64>,

    /// Maximum bond accepted by the PoS contract in the genesis block
    #[arg(long = "bond-maximum")]
    pub bond_maximum: Option<u64>,

    /// The length of the validation epoch in blocks
    #[arg(long = "epoch-length")]
    pub epoch_length: Option<u32>,

    /// The length of the quarantine time in blocks
    #[arg(long = "quarantine-length")]
    pub quarantine_length: Option<u32>,

    /// The number of the active validators
    #[arg(long = "number-of-active-validators")]
    pub number_of_active_validators: Option<u32>,

    /// Number of signatures from bonded validators required for Ceremony Master to approve the genesis block
    #[arg(long = "required-signatures")]
    pub required_signatures: Option<u32>,

    /// Each `approve-interval` Ceremony Master (CM) checks if it have gathered enough signatures to approve the genesis block
    #[arg(long = "approve-interval", value_parser = ValueParser::new(parse_duration))]
    pub approve_interval: Option<Duration>,

    /// Time window in which BlockApproval messages will be accumulated before checking conditions
    #[arg(long = "approve-duration", value_parser = ValueParser::new(parse_duration))]
    pub approve_duration: Option<Duration>,

    /// Start a node as a genesis validator
    #[arg(long = "genesis-validator", action = ArgAction::SetTrue)]
    pub genesis_validator: Option<bool>,

    /// Enable the Prometheus metrics reporter
    #[arg(long = "prometheus", action = ArgAction::SetTrue)]
    pub prometheus: Option<bool>,

    /// Enable the InfluxDB metrics reporter
    #[arg(long = "influxdb", action = ArgAction::SetTrue)]
    pub influxdb: Option<bool>,

    /// Enable the InfluxDB UDP metrics reporter
    #[arg(long = "influxdb-udp", action = ArgAction::SetTrue)]
    pub influxdb_udp: Option<bool>,

    /// Enable the Zipkin span reporter
    #[arg(long = "zipkin", action = ArgAction::SetTrue)]
    pub zipkin: Option<bool>,

    /// Enable Sigar host system metrics
    #[arg(long = "sigar", action = ArgAction::SetTrue)]
    pub sigar: Option<bool>,

    /// Timestamp for the deploys
    #[arg(long = "deploy-timestamp")]
    pub deploy_timestamp: Option<u64>,

    /// Enable all developer tools
    #[arg(long = "dev-mode", action = ArgAction::SetTrue)]
    pub dev_mode: Option<bool>,

    /// Private key for dummy deploys
    #[arg(long = "deployer-private-key")]
    pub deployer_private_key: Option<String>,

    /// MinPhloPrice
    #[arg(long = "min-phlo-price")]
    pub min_phlo_price: Option<u64>,
}

/// Keygen subcommand - Generates a public/private key pair
#[derive(Args, Debug, Clone)]
pub struct KeygenOptions {
    /// Folder to save keyfiles
    #[arg(default_value = "./")]
    pub location: PathBuf,
}

/// Eval subcommand - Starts a thin client that will evaluate rholang in file on a existing running node
#[derive(Args, Debug, Clone)]
pub struct EvalOptions {
    /// Rholang files to evaluate
    pub file_names: Vec<String>,

    /// Print only unmatched sends
    #[arg(long = "print-unmatched-sends-only")]
    pub print_unmatched_sends_only: bool,

    /// Language to use
    #[arg(long = "language", default_value = "rholang")]
    pub language: String,
}

/// Deploy subcommand - Deploy a Rholang source file to Casper on an existing running node
#[derive(Args, Debug, Clone)]
pub struct DeployOptions {
    /// The amount of phlo to use for the transaction
    #[arg(long = "phlo-limit")]
    pub phlo_limit: u64,

    /// The price of phlo for this transaction in units dust/phlo
    #[arg(long = "phlo-price")]
    pub phlo_price: u64,

    /// Set this value to one less than the current block height
    #[arg(long = "valid-after-block-number")]
    pub valid_after_block_number: Option<u64>,

    /// The deployer's secp256k1 private key encoded as Base16
    #[arg(long = "private-key")]
    pub private_key: Option<String>,

    /// The deployer's file with encrypted private key
    #[arg(long = "private-key-path")]
    pub private_key_path: Option<PathBuf>,

    /// The name of the shard
    #[arg(long = "shard-id", default_value = "")]
    pub shard_id: String,

    /// Location of the Rholang file to deploy
    pub location: String,
}

/// Show block subcommand - View properties of a block known by Casper
#[derive(Args, Debug, Clone)]
pub struct ShowBlockOptions {
    /// The hash value of the block
    pub hash: String,
}

/// Show blocks subcommand - View list of blocks in the current Casper view
#[derive(Args, Debug, Clone)]
pub struct ShowBlocksOptions {
    /// Lists blocks to the given depth in terms of block height
    #[arg(long = "depth")]
    pub depth: Option<u32>,
}

/// Visualize DAG subcommand - DAG in DOT format
#[derive(Args, Debug, Clone)]
pub struct VisualizeDagOptions {
    /// Depth in terms of block height
    #[arg(long = "depth")]
    pub depth: Option<u32>,

    /// If justification lines should be shown
    #[arg(long = "show-justification-lines")]
    pub show_justification_lines: bool,
}

/// Is finalized subcommand - Check if the given block has been finalized by Casper
#[derive(Args, Debug, Clone)]
pub struct IsFinalizedOptions {
    /// The hash value of the block to check
    pub hash: String,
}

/// Bond status subcommand - Check bond status for a validator
#[derive(Args, Debug, Clone)]
pub struct BondStatusOptions {
    /// Base16 encoding of the public key to check for bond status
    pub validator_public_key: String,
}

/// Data at name subcommand - Listen for data at the specified name
#[derive(Args, Debug, Clone)]
pub struct DataAtNameOptions {
    /// Type of the specified name
    #[arg(short = 't', long = "type")]
    pub type_of_name: String,

    /// Rholang name
    #[arg(short = 'c', long = "content")]
    pub content: String,
}

/// Cont at name subcommand - Listen for continuation at the specified name
#[derive(Args, Debug, Clone)]
pub struct ContAtNameOptions {
    /// Type of the specified name
    #[arg(short = 't', long = "type")]
    pub type_of_name: String,

    /// Rholang names
    #[arg(short = 'c', long = "content")]
    pub content: Vec<String>,
}

/// Find deploy subcommand - Searches for a block containing the deploy with provided id
#[derive(Args, Debug, Clone)]
pub struct FindDeployOptions {
    /// Id of the deploy
    #[arg(long = "deploy-id")]
    pub deploy_id: String,
}

/// Propose subcommand - Force Casper to propose a block based on its accumulated deploys
#[derive(Args, Debug, Clone)]
pub struct ProposeOptions {
    /// Print unmatched sends
    #[arg(long = "print-unmatched-sends")]
    pub print_unmatched_sends: bool,
}
