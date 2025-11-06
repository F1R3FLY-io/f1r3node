//! Configuration model definitions
//!
//! This module contains the data structures that represent the node configuration,
//! including all the nested configuration sections.

use casper::rust::casper_conf::de_duration;
use casper::rust::casper_conf::CasperConf;
use clap::Subcommand;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

use crate::rust::configuration::commandline::options::{
    BondStatusOptions, ContAtNameOptions, DataAtNameOptions, DeployOptions, EvalOptions,
    FindDeployOptions, IsFinalizedOptions, KeygenOptions, ProposeOptions, RunOptions,
    ShowBlockOptions, ShowBlocksOptions, VisualizeDagOptions,
};

/// Main node configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConf {
    pub standalone: bool,
    pub autopropose: bool,

    #[serde(rename = "protocol-server")]
    pub protocol_server: ProtocolServer,
    #[serde(rename = "protocol-client")]
    pub protocol_client: ProtocolClient,
    #[serde(rename = "peers-discovery")]
    pub peers_discovery: PeersDiscovery,
    #[serde(rename = "api-server")]
    pub api_server: ApiServer,
    pub storage: Storage,
    pub tls: TlsConf,
    pub casper: CasperConf,
    pub metrics: Metrics,

    #[serde(rename = "dev-mode")]
    pub dev_mode: bool,
    pub dev: DevConf,
}

/// Protocol server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolServer {
    #[serde(rename = "network-id")]
    pub network_id: String,

    pub host: Option<String>,

    #[serde(rename = "allow-private-addresses")]
    pub allow_private_addresses: bool,

    #[serde(rename = "use-random-ports")]
    pub use_random_ports: bool,

    #[serde(rename = "dynamic-ip")]
    pub dynamic_ip: bool,

    #[serde(rename = "no-upnp")]
    pub no_upnp: bool,

    pub port: u16,

    #[serde(rename = "grpc-max-recv-message-size", deserialize_with = "de_bytes")]
    pub grpc_max_recv_message_size: u32,

    #[serde(
        rename = "grpc-max-recv-stream-message-size",
        deserialize_with = "de_bytes"
    )]
    pub grpc_max_recv_stream_message_size: u32,

    #[serde(rename = "max-message-consumers")]
    pub max_message_consumers: u32,

    #[serde(rename = "disable-state-exporter")]
    pub disable_state_exporter: bool,
}

/// Protocol client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolClient {
    #[serde(rename = "network-id")]
    pub network_id: String,

    pub bootstrap: String,

    #[serde(rename = "disable-lfs")]
    pub disable_lfs: bool,

    #[serde(rename = "batch-max-connections")]
    pub batch_max_connections: u32,

    #[serde(rename = "network-timeout", deserialize_with = "de_duration")]
    pub network_timeout: Duration,

    #[serde(rename = "grpc-max-recv-message-size", deserialize_with = "de_bytes")]
    pub grpc_max_recv_message_size: u32,

    #[serde(rename = "grpc-stream-chunk-size", deserialize_with = "de_bytes")]
    pub grpc_stream_chunk_size: u32,
}

/// Peers discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeersDiscovery {
    pub host: Option<String>,
    pub port: u16,

    #[serde(rename = "lookup-interval", deserialize_with = "de_duration")]
    pub lookup_interval: Duration,

    #[serde(rename = "cleanup-interval", deserialize_with = "de_duration")]
    pub cleanup_interval: Duration,

    #[serde(rename = "heartbeat-batch-size")]
    pub heartbeat_batch_size: u32,

    #[serde(rename = "init-wait-loop-interval", deserialize_with = "de_duration")]
    pub init_wait_loop_interval: Duration,
}

/// API server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiServer {
    pub host: String,

    #[serde(rename = "port-grpc-external")]
    pub port_grpc_external: u16,
    #[serde(rename = "port-grpc-internal")]
    pub port_grpc_internal: u16,
    #[serde(rename = "grpc-max-recv-message-size", deserialize_with = "de_bytes")]
    pub grpc_max_recv_message_size: u32,

    #[serde(rename = "port-http")]
    pub port_http: u16,
    #[serde(rename = "port-admin-http")]
    pub port_admin_http: u16,

    #[serde(rename = "max-blocks-limit")]
    pub max_blocks_limit: u32,

    #[serde(rename = "enable-reporting")]
    pub enable_reporting: bool,

    #[serde(rename = "keep-alive-time", deserialize_with = "de_duration")]
    pub keep_alive_time: Duration,
    #[serde(rename = "keep-alive-timeout", deserialize_with = "de_duration")]
    pub keep_alive_timeout: Duration,
    #[serde(rename = "permit-keep-alive-time", deserialize_with = "de_duration")]
    pub permit_keep_alive_time: Duration,
    #[serde(rename = "max-connection-idle", deserialize_with = "de_duration")]
    pub max_connection_idle: Duration,
    #[serde(rename = "max-connection-age", deserialize_with = "de_duration")]
    pub max_connection_age: Duration,
    #[serde(rename = "max-connection-age-grace", deserialize_with = "de_duration")]
    pub max_connection_age_grace: Duration,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Storage {
    #[serde(rename = "data-dir")]
    pub data_dir: PathBuf,
}

/// TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConf {
    #[serde(rename = "certificate-path")]
    pub certificate_path: PathBuf,
    #[serde(rename = "key-path")]
    pub key_path: PathBuf,
    #[serde(rename = "secure-random-non-blocking")]
    pub secure_random_non_blocking: bool,
    #[serde(rename = "custom-certificate-location")]
    pub custom_certificate_location: bool,
    #[serde(rename = "custom-key-location")]
    pub custom_key_location: bool,
}

impl From<TlsConf> for comm::rust::transport::tls_conf::TlsConf {
    fn from(conf: TlsConf) -> Self {
        comm::rust::transport::tls_conf::TlsConf {
            certificate_path: conf.certificate_path,
            key_path: conf.key_path,
            secure_random_non_blocking: conf.secure_random_non_blocking,
            custom_certificate_location: conf.custom_certificate_location,
            custom_key_location: conf.custom_key_location,
        }
    }
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub prometheus: bool,
    pub influxdb: bool,
    #[serde(rename = "influxdb-udp")]
    pub influxdb_udp: bool,
    pub zipkin: bool,
    pub sigar: bool,
}

/// Development configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevConf {
    pub deployer_private_key: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Profile {
    pub name: &'static str,
    /// (path, description)
    pub data_dir: (PathBuf, &'static str),
}

/// Command enumeration for CLI operations
#[derive(Debug, Clone, Subcommand)]
pub enum Command {
    /// Start RNode server
    Run(RunOptions),

    /// Generates a public/private key pair
    Keygen(KeygenOptions),

    /// View properties of the last finalized block known by Casper
    LastFinalizedBlock,

    /// Check if the given block has been finalized by Casper
    IsFinalized(IsFinalizedOptions),

    /// Starts a thin client, that will connect to existing node
    Repl,

    /// Starts a thin client that will evaluate rholang in file on a existing running node
    Eval(EvalOptions),

    /// Deploy a Rholang source file to Casper on an existing running node
    Deploy(DeployOptions),

    /// View properties of a block known by Casper
    ShowBlock(ShowBlockOptions),

    /// View list of blocks in the current Casper view
    ShowBlocks(ShowBlocksOptions),

    /// DAG in DOT format
    Vdag(VisualizeDagOptions),

    /// Machine Verifiable Dag
    Mvdag,

    /// Listen for data at the specified name
    ListenDataAtName(DataAtNameOptions),

    /// Listen for continuation at the specified name
    ListenContAtName(ContAtNameOptions),

    /// Searches for a block containing the deploy with provided id
    FindDeploy(FindDeployOptions),

    /// Force Casper to propose a block based on its accumulated deploys
    Propose(ProposeOptions),

    /// Check bond status for a validator
    BondStatus(BondStatusOptions),

    /// Get RNode status information
    Status,
}

// Accept integers (bytes) or strings like "256K", "16M", "2G".
fn de_bytes<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as _;
    use serde::Deserialize;
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum BytesIn {
        Num(u32),
        Str(String),
    }
    fn parse_bytes(s: &str) -> Option<u32> {
        byte_unit::Byte::parse_str(s, true)
            .ok()
            .map(|num| num.as_u64() as u32)
    }
    match BytesIn::deserialize(deserializer)? {
        BytesIn::Num(n) => Ok(n),
        BytesIn::Str(s) => {
            parse_bytes(&s).ok_or_else(|| D::Error::custom(format!("invalid byte size {s:?}")))
        }
    }
}
