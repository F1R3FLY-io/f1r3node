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
    #[serde(default)]
    pub standalone: bool,
    #[serde(default)]
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

    /// OpenAI configuration - ported from Scala PR #123
    #[serde(default)]
    pub openai: OpenAIConf,
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

    #[serde(default)]
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
    #[serde(default)]
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

/// Metrics configuration. Combines reporter toggles with the (formerly
/// `kamon.conf`-resident) reporter endpoints — there is now one source of
/// truth for metrics config inside `defaults.conf`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub prometheus: bool,
    pub influxdb: bool,
    #[serde(rename = "influxdb-udp")]
    pub influxdb_udp: bool,
    pub zipkin: bool,
    pub sigar: bool,

    /// How often the metric reporters poll and emit. Drives the InfluxDB
    /// HTTP/UDP and Sigar (system-metrics) reporters.
    #[serde(
        rename = "tick-interval",
        default = "default_tick_interval",
        with = "duration_secs"
    )]
    pub tick_interval: Duration,

    /// Endpoint settings for the InfluxDB reporters (HTTP and UDP). Both
    /// reporters share the same hostname/port pair; protocol/auth/database
    /// only apply to the HTTP reporter.
    #[serde(rename = "influxdb-endpoint", default)]
    pub influxdb_endpoint: InfluxDbEndpoint,
}

fn default_tick_interval() -> Duration {
    Duration::from_secs(10)
}

mod duration_secs {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(d.as_secs())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        // HOCON "10 seconds" deserializes via serde as a string in some
        // setups and as a u64 in others; accept either.
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Secs(u64),
            Text(String),
        }
        match Repr::deserialize(d)? {
            Repr::Secs(s) => Ok(Duration::from_secs(s)),
            Repr::Text(t) => parse_duration_string(&t).map_err(serde::de::Error::custom),
        }
    }

    fn parse_duration_string(s: &str) -> Result<Duration, String> {
        let parts: Vec<&str> = s.trim().split_whitespace().collect();
        if parts.len() != 2 {
            return Err(format!("invalid duration: {}", s));
        }
        let n: u64 = parts[0]
            .parse()
            .map_err(|_| format!("invalid number: {}", parts[0]))?;
        let mult = match parts[1].to_lowercase().as_str() {
            "second" | "seconds" | "s" => 1,
            "minute" | "minutes" | "m" => 60,
            "hour" | "hours" | "h" => 3600,
            other => return Err(format!("unknown unit: {}", other)),
        };
        Ok(Duration::from_secs(n * mult))
    }
}

/// InfluxDB reporter endpoint. Mirrors the legacy `kamon.influxdb` schema
/// fields that the Rust reporter implementations actually consume.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InfluxDbEndpoint {
    #[serde(default)]
    pub hostname: String,
    #[serde(default)]
    pub port: u16,
    #[serde(default)]
    pub database: String,
    #[serde(default = "default_influxdb_protocol")]
    pub protocol: String,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub password: Option<String>,
}

fn default_influxdb_protocol() -> String {
    "http".to_string()
}

/// Development configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevConf {
    pub deployer_private_key: Option<String>,
}

/// OpenAI configuration
/// Ported from Scala PR #123 - Issue #127
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConf {
    /// Enable or disable OpenAI service functionality
    /// Priority: 1. Environment variable OPENAI_ENABLED, 2. Config, 3. Default (false)
    #[serde(default)]
    pub enabled: bool,

    /// API key used by OpenAIService
    /// Resolution: 1. OPENAI_API_KEY env, 2. OPENAI_SCALA_CLIENT_API_KEY env, 3. Config
    #[serde(rename = "api-key", default)]
    pub api_key: String,

    /// Validate API key at startup by calling a lightweight endpoint
    #[serde(rename = "validate-api-key", default = "default_validate_api_key")]
    pub validate_api_key: bool,

    /// Timeout for API key validation call in seconds
    #[serde(
        rename = "validation-timeout-sec",
        default = "default_validation_timeout_sec"
    )]
    pub validation_timeout_sec: u64,
}

fn default_validate_api_key() -> bool {
    true
}

fn default_validation_timeout_sec() -> u64 {
    15
}

impl Default for OpenAIConf {
    fn default() -> Self {
        Self {
            enabled: false,
            api_key: String::new(),
            validate_api_key: true,
            validation_timeout_sec: 15,
        }
    }
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
