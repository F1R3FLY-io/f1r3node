//! Configuration mapper for merging CLI options to with configuration
//!
//! This module provides functionality to map command-line options into a configuration

use super::options::Options;
use crate::rust::configuration::{commandline::options::OptionsSubCommand, NodeConf};

/// Configuration mapper for converting CLI options to configuration
pub trait ConfigMapper<T> {
    fn override_config_values(&mut self, options: Options);
    fn try_override_value<V>(config_value: &mut V, value: Option<V>);
    fn try_override_option<V>(config_value: &mut Option<V>, value: Option<V>);
}

impl ConfigMapper<Options> for NodeConf {
    /// Override config values with CLI provided options
    fn override_config_values(&mut self, options: Options) {
        if let Some(OptionsSubCommand::Run(run)) = options.subcommand {
            Self::try_override_value(&mut self.standalone, run.standalone);
            Self::try_override_value(&mut self.autopropose, run.autopropose);
            Self::try_override_value(&mut self.dev_mode, run.dev_mode);
            // Protocol server fields
            Self::try_override_value(&mut self.protocol_server.dynamic_ip, run.dynamic_ip);
            Self::try_override_value(&mut self.protocol_server.no_upnp, run.no_upnp);
            Self::try_override_value(
                &mut self.protocol_server.allow_private_addresses,
                run.allow_private_addresses,
            );
            Self::try_override_value(
                &mut self.protocol_server.disable_state_exporter,
                run.disable_state_exporter,
            );
            Self::try_override_value(&mut self.protocol_server.network_id, run.network_id);
            Self::try_override_option(&mut self.protocol_server.host, run.host);
            Self::try_override_value(
                &mut self.protocol_server.use_random_ports,
                run.use_random_ports,
            );
            Self::try_override_value(&mut self.protocol_server.port, run.protocol_port);
            Self::try_override_value(
                &mut self.protocol_server.grpc_max_recv_message_size,
                run.protocol_grpc_max_recv_message_size,
            );
            Self::try_override_value(
                &mut self.protocol_server.grpc_max_recv_stream_message_size,
                run.protocol_grpc_max_recv_stream_message_size,
            );
            Self::try_override_value(
                &mut self.protocol_server.max_message_consumers,
                run.protocol_max_message_consumers,
            );

            // Protocol client fields
            Self::try_override_value(&mut self.protocol_client.disable_lfs, run.disable_lfs);
            Self::try_override_value(&mut self.protocol_client.bootstrap, run.bootstrap);
            Self::try_override_value(
                &mut self.protocol_client.network_timeout,
                run.network_timeout,
            );
            Self::try_override_value(
                &mut self.protocol_client.batch_max_connections,
                run.protocol_max_connections,
            );
            Self::try_override_value(
                &mut self.protocol_client.grpc_max_recv_message_size,
                run.protocol_grpc_max_recv_message_size,
            );
            Self::try_override_value(
                &mut self.protocol_client.grpc_stream_chunk_size,
                run.protocol_grpc_stream_chunk_size,
            );

            // Peers discovery fields
            Self::try_override_value(&mut self.peers_discovery.port, run.discovery_port);
            Self::try_override_value(
                &mut self.peers_discovery.lookup_interval,
                run.discovery_lookup_interval,
            );
            Self::try_override_value(
                &mut self.peers_discovery.cleanup_interval,
                run.discovery_cleanup_interval,
            );
            Self::try_override_value(
                &mut self.peers_discovery.heartbeat_batch_size,
                run.discovery_heartbeat_batch_size,
            );
            Self::try_override_value(
                &mut self.peers_discovery.init_wait_loop_interval,
                run.discovery_init_wait_loop_interval,
            );

            // API server fields
            Self::try_override_value(
                &mut self.api_server.enable_reporting,
                run.api_enable_reporting,
            );
            Self::try_override_value(
                &mut self.api_server.port_grpc_external,
                run.api_port_grpc_external,
            );
            Self::try_override_value(
                &mut self.api_server.port_grpc_internal,
                run.api_port_grpc_internal,
            );
            Self::try_override_value(&mut self.api_server.port_http, run.api_port_http);
            Self::try_override_value(
                &mut self.api_server.port_admin_http,
                run.api_port_admin_http,
            );
            Self::try_override_value(&mut self.api_server.host, run.api_host);
            Self::try_override_value(
                &mut self.api_server.grpc_max_recv_message_size,
                run.api_grpc_max_recv_message_size,
            );
            Self::try_override_value(
                &mut self.api_server.max_blocks_limit,
                run.api_max_blocks_limit,
            );
            Self::try_override_value(
                &mut self.api_server.keep_alive_time,
                run.api_keep_alive_time,
            );
            Self::try_override_value(
                &mut self.api_server.keep_alive_timeout,
                run.api_keep_alive_timeout,
            );
            Self::try_override_value(
                &mut self.api_server.permit_keep_alive_time,
                run.api_permit_keep_alive_time,
            );
            Self::try_override_value(
                &mut self.api_server.max_connection_idle,
                run.api_max_connection_idle,
            );
            Self::try_override_value(
                &mut self.api_server.max_connection_age,
                run.api_max_connection_age,
            );
            Self::try_override_value(
                &mut self.api_server.max_connection_age_grace,
                run.api_max_connection_age_grace,
            );

            // Storage fields
            Self::try_override_value(&mut self.storage.data_dir, run.data_dir);

            // TLS fields
            Self::try_override_value(
                &mut self.tls.secure_random_non_blocking,
                run.tls_secure_random_non_blocking,
            );
            Self::try_override_value(&mut self.tls.key_path, run.tls_key_path);
            Self::try_override_value(&mut self.tls.certificate_path, run.tls_certificate_path);

            // Metrics fields
            Self::try_override_value(&mut self.metrics.prometheus, run.prometheus);
            Self::try_override_value(&mut self.metrics.influxdb, run.influxdb);
            Self::try_override_value(&mut self.metrics.influxdb_udp, run.influxdb_udp);
            Self::try_override_value(&mut self.metrics.zipkin, run.zipkin);
            Self::try_override_value(&mut self.metrics.sigar, run.sigar);

            // Dev fields
            Self::try_override_option(&mut self.dev.deployer_private_key, run.deployer_private_key);

            // Casper configuration fields
            Self::try_override_value(&mut self.casper.shard_name, run.shard_name);
            Self::try_override_value(
                &mut self.casper.fault_tolerance_threshold,
                run.fault_tolerance_threshold,
            );
            Self::try_override_value(&mut self.casper.finalization_rate, run.finalization_rate);
            Self::try_override_value(
                &mut self.casper.max_number_of_parents,
                run.max_number_of_parents,
            );
            Self::try_override_value(&mut self.casper.max_parent_depth, run.max_parent_depth);
            Self::try_override_value(
                &mut self.casper.synchrony_constraint_threshold,
                run.synchrony_constraint_threshold,
            );
            Self::try_override_value(
                &mut self.casper.height_constraint_threshold,
                run.height_constraint_threshold,
            );
            Self::try_override_option(
                &mut self.casper.validator_public_key,
                run.validator_public_key,
            );
            Self::try_override_option(
                &mut self.casper.validator_private_key,
                run.validator_private_key,
            );
            Self::try_override_option(
                &mut self.casper.validator_private_key_path,
                run.validator_private_key_path,
            );
            Self::try_override_value(
                &mut self.casper.casper_loop_interval,
                run.casper_loop_interval,
            );
            Self::try_override_value(
                &mut self.casper.requested_blocks_timeout,
                run.requested_blocks_timeout,
            );
            Self::try_override_value(
                &mut self.casper.fork_choice_stale_threshold,
                run.fork_choice_stale_threshold,
            );
            Self::try_override_value(
                &mut self.casper.fork_choice_check_if_stale_interval,
                run.fork_choice_check_if_stale_interval,
            );
            Self::try_override_value(
                &mut self.casper.round_robin_dispatcher.max_peer_queue_size,
                run.frrd_max_peer_queue_size,
            );
            Self::try_override_value(
                &mut self.casper.round_robin_dispatcher.give_up_after_skipped,
                run.frrd_give_up_after_skipped,
            );
            Self::try_override_value(
                &mut self.casper.round_robin_dispatcher.drop_peer_after_retries,
                run.frrd_drop_peer_after_retries,
            );
            Self::try_override_value(
                &mut self.casper.genesis_block_data.bonds_file,
                run.bonds_file,
            );
            Self::try_override_value(
                &mut self.casper.genesis_block_data.wallets_file,
                run.wallets_file,
            );
            Self::try_override_value(
                &mut self.casper.genesis_block_data.bond_minimum,
                run.bond_minimum,
            );
            Self::try_override_value(
                &mut self.casper.genesis_block_data.bond_maximum,
                run.bond_maximum,
            );
            Self::try_override_value(
                &mut self.casper.genesis_block_data.epoch_length,
                run.epoch_length,
            );
            Self::try_override_value(
                &mut self.casper.genesis_block_data.quarantine_length,
                run.quarantine_length,
            );
            Self::try_override_value(
                &mut self.casper.genesis_block_data.number_of_active_validators,
                run.number_of_active_validators,
            );
            Self::try_override_value(
                &mut self.casper.genesis_block_data.genesis_block_number,
                run.genesis_block_number,
            );

            Self::try_override_value(
                &mut self.casper.genesis_ceremony.required_signatures,
                run.required_signatures,
            );
            Self::try_override_value(
                &mut self.casper.genesis_ceremony.approve_interval,
                run.approve_interval,
            );
            Self::try_override_value(
                &mut self.casper.genesis_ceremony.approve_duration,
                run.approve_duration,
            );
            Self::try_override_value(
                &mut self.casper.genesis_ceremony.autogen_shard_size,
                run.autogen_shard_size,
            );
            Self::try_override_value(
                &mut self.casper.genesis_ceremony.genesis_validator_mode,
                run.genesis_validator,
            );
            Self::try_override_value(&mut self.casper.min_phlo_price, run.min_phlo_price);
        }
    }

    fn try_override_value<V>(config_value: &mut V, value: Option<V>) {
        if let Some(value) = value {
            *config_value = value;
        }
    }

    fn try_override_option<V>(config_value: &mut Option<V>, value: Option<V>) {
        if let Some(value) = value {
            *config_value = Some(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, time::Duration};

    use crate::rust::configuration::commandline::options::{OptionsSubCommand, RunOptions};

    use super::*;
    use clap::Parser;

    #[test]
    fn test_parse_args() {
        let argv = vec![
        "rnode",
        "run",
        "--standalone",
        "--dev-mode",
        "--host=localhost",
        "--bootstrap=rnode://de6eed5d00cf080fc587eeb412cb31a75fd10358@52.119.8.109?protocol=40400&discovery=40404",
        "--network-id=testnet",
        "--no-upnp",
        "--dynamic-ip",
        "--autogen-shard-size=111111",
        "--use-random-ports",
        "--allow-private-addresses",
        "--network-timeout=111111seconds",
        "--discovery-port=11111",
        "--discovery-lookup-interval=111111seconds",
        "--discovery-cleanup-interval=111111seconds",
        "--discovery-heartbeat-batch-size=111111",
        "--discovery-init-wait-loop-interval=111111seconds",
        "--protocol-port=11111",
        "--protocol-grpc-max-recv-message-size=111111",
        "--protocol-grpc-max-recv-stream-message-size=111111",
        "--protocol-grpc-stream-chunk-size=111111",
        "--protocol-max-connections=111111",
        "--protocol-max-message-consumers=111111",
        "--disable-state-exporter",
        "--tls-certificate-path=/var/lib/rnode/node.certificate.pem",
        "--tls-key-path=/var/lib/rnode/node.key.pem",
        "--tls-secure-random-non-blocking",
        "--api-host=localhost",
        "--api-port-grpc-external=11111",
        "--api-port-grpc-internal=11111",
        "--api-port-http=11111",
        "--api-port-admin-http=11111",
        "--api-grpc-max-recv-message-size=111111",
        "--api-max-blocks-limit=111111",
        "--api-enable-reporting",
        "--api-keep-alive-time=111111seconds",
        "--api-keep-alive-timeout=111111seconds",
        "--api-permit-keep-alive-time=111111seconds",
        "--api-max-connection-idle=111111seconds",
        "--api-max-connection-age=111111seconds",
        "--api-max-connection-age-grace=111111seconds",
        "--data-dir=/var/lib/rnode",
        "--shard-name=root",
        "--fault-tolerance-threshold=111111",
        "--validator-public-key=111111",
        "--validator-private-key=111111",
        "--validator-private-key-path=/var/lib/rnode/pem.key",
        "--casper-loop-interval=111111seconds",
        "--requested-blocks-timeout=111111seconds",
        "--finalization-rate=111111",
        "--max-number-of-parents=111111",
        "--max-parent-depth=111111",
        "--fork-choice-stale-threshold=111111seconds",
        "--fork-choice-check-if-stale-interval=111111seconds",
        "--synchrony-constraint-threshold=111111",
        "--height-constraint-threshold=111111",
        "--frrd-max-peer-queue-size=111111",
        "--frrd-give-up-after-skipped=111111",
        "--frrd-drop-peer-after-retries=111111",
        "--bonds-file=/var/lib/rnode/genesis/bonds1.txt",
        "--wallets-file=/var/lib/rnode/genesis/wallets1.txt",
        "--bond-minimum=111111",
        "--bond-maximum=111111",
        "--epoch-length=111111",
        "--quarantine-length=111111",
        "--genesis-block-number=222",
        "--number-of-active-validators=111111",
        "--deploy-timestamp=111111",
        "--required-signatures=111111",
        "--approve-interval=111111seconds",
        "--approve-duration=111111seconds",
        "--genesis-validator",
        "--disable-lfs",
        "--prometheus",
        "--influxdb",
        "--influxdb-udp",
        "--zipkin",
        "--sigar"
        ];

        let res = Options::try_parse_from(argv);
        assert!(res.is_ok());
    }

    #[test]
    fn test_cli_options_override_defaults() {
        // Create CLI options that mirror the Scala test
        let options = Options {
            grpc_host: "localhost".to_string(),
            grpc_port: 40401,
            grpc_max_recv_message_size: 16777216,
            profile: Some("docker".to_string()),
            subcommand: Some(OptionsSubCommand::Run(RunOptions {
                config_file: None,
                thread_pool_size: None,
                standalone: Some(true),
                bootstrap: Some("rnode://de6eed5d00cf080fc587eeb412cb31a75fd10358@52.119.8.109?protocol=40400&discovery=40404".to_string()),
                network_id: Some("testnet".to_string()),
                autopropose: Some(false),
                no_upnp: Some(true),
                dynamic_ip: Some(true),
                autogen_shard_size: Some(111111),
                disable_lfs: Some(true),
                host: Some("localhost".to_string()),
                use_random_ports: Some(true),
                allow_private_addresses: Some(true),
                disable_state_exporter: Some(true),
                network_timeout: Some(Duration::from_secs(111111)),
                discovery_port: Some(11111),
                discovery_lookup_interval: Some(Duration::from_secs(111111)),
                discovery_cleanup_interval: Some(Duration::from_secs(111111)),
                discovery_heartbeat_batch_size: Some(111111),
                discovery_init_wait_loop_interval: Some(Duration::from_secs(111111)),
                protocol_port: Some(11111),
                protocol_grpc_max_recv_message_size: Some(111111),
                protocol_grpc_max_recv_stream_message_size: Some(111111),
                protocol_grpc_stream_chunk_size: Some(111111),
                protocol_max_connections: Some(111111),
                protocol_max_message_consumers: Some(111111),
                tls_key_path: Some(PathBuf::from("/var/lib/rnode/node.key.pem")),
                tls_certificate_path: Some(PathBuf::from("/var/lib/rnode/node.certificate.pem")),
                tls_secure_random_non_blocking: Some(true),
                api_host: Some("localhost".to_string()),
                api_port_grpc_external: Some(11111),
                api_port_grpc_internal: Some(11111),
                api_grpc_max_recv_message_size: Some(111111),
                api_port_http: Some(11111),
                api_port_admin_http: Some(11111),
                api_max_blocks_limit: Some(111111),
                api_enable_reporting: Some(true),
                api_keep_alive_time: Some(Duration::from_secs(111111)),
                api_keep_alive_timeout: Some(Duration::from_secs(111111)),
                api_permit_keep_alive_time: Some(Duration::from_secs(111111)),
                api_max_connection_idle: Some(Duration::from_secs(111111)),
                api_max_connection_age: Some(Duration::from_secs(111111)),
                api_max_connection_age_grace: Some(Duration::from_secs(111111)),
                data_dir: Some(PathBuf::from("/var/lib/rnode")),
                shard_name: Some("root".to_string()),
                fault_tolerance_threshold: Some(111111.0),
                validator_public_key: Some("111111".to_string()),
                validator_private_key: Some("111111".to_string()),
                validator_private_key_path: Some(PathBuf::from("/var/lib/rnode/pem.key")),
                casper_loop_interval: Some(Duration::from_secs(111111)),
                requested_blocks_timeout: Some(Duration::from_secs(111111)),
                finalization_rate: Some(111111),
                max_number_of_parents: Some(111111),
                max_parent_depth: Some(111111),
                fork_choice_stale_threshold: Some(Duration::from_secs(111111)),
                fork_choice_check_if_stale_interval: Some(Duration::from_secs(111111)),
                synchrony_constraint_threshold: Some(111111.0),
                height_constraint_threshold: Some(111111),
                frrd_max_peer_queue_size: Some(111111),
                frrd_give_up_after_skipped: Some(111111),
                frrd_drop_peer_after_retries: Some(111111),
                bonds_file: Some("/var/lib/rnode/genesis/bonds1.txt".to_string()),
                wallets_file: Some("/var/lib/rnode/genesis/wallets1.txt".to_string()),
                bond_minimum: Some(111111),
                genesis_block_number: Some(222),
                bond_maximum: Some(111111),
                epoch_length: Some(111111),
                quarantine_length: Some(111111),
                number_of_active_validators: Some(111111),
                required_signatures: Some(111111),
                approve_interval: Some(Duration::from_secs(111111)),
                approve_duration: Some(Duration::from_secs(111111)),
                genesis_validator: Some(true),
                prometheus: Some(true),
                influxdb: Some(true),
                influxdb_udp: Some(true),
                zipkin: Some(true),
                sigar: Some(true),
                deploy_timestamp: Some(111111),
                dev_mode: Some(true),
                deployer_private_key: Some("test-key".to_string()),
                min_phlo_price: Some(1),
            })),
        };

        // Create a default configuration (similar to loading from defaults.conf)
        let mut default_config = NodeConf {
            standalone: false,
            autopropose: false,
            protocol_server: crate::rust::configuration::model::ProtocolServer {
                network_id: "default-network".to_string(),
                host: None,
                allow_private_addresses: false,
                use_random_ports: false,
                dynamic_ip: false,
                no_upnp: false,
                port: 40404,
                grpc_max_recv_message_size: 16777216,
                grpc_max_recv_stream_message_size: 16777216,
                max_message_consumers: 4,
                disable_state_exporter: false,
            },
            protocol_client: crate::rust::configuration::model::ProtocolClient {
                network_id: "default-network".to_string(),
                bootstrap: "".to_string(),
                disable_lfs: false,
                batch_max_connections: 4,
                network_timeout: Duration::from_secs(30),
                grpc_max_recv_message_size: 16777216,
                grpc_stream_chunk_size: 16777216,
            },
            peers_discovery: crate::rust::configuration::model::PeersDiscovery {
                host: None,
                port: 40400,
                lookup_interval: Duration::from_secs(30),
                cleanup_interval: Duration::from_secs(30),
                heartbeat_batch_size: 4,
                init_wait_loop_interval: Duration::from_secs(30),
            },
            api_server: crate::rust::configuration::model::ApiServer {
                host: "0.0.0.0".to_string(),
                port_grpc_external: 40401,
                port_grpc_internal: 40402,
                grpc_max_recv_message_size: 16777216,
                port_http: 40403,
                port_admin_http: 40405,
                max_blocks_limit: 50,
                enable_reporting: false,
                keep_alive_time: Duration::from_secs(2),
                keep_alive_timeout: Duration::from_secs(20),
                permit_keep_alive_time: Duration::from_secs(5),
                max_connection_idle: Duration::from_secs(300),
                max_connection_age: Duration::from_secs(3600),
                max_connection_age_grace: Duration::from_secs(5),
            },
            storage: crate::rust::configuration::model::Storage {
                data_dir: PathBuf::from("/var/lib/rnode"),
            },
            tls: crate::rust::configuration::model::TlsConf {
                certificate_path: PathBuf::from("/var/lib/rnode/node.certificate.pem"),
                key_path: PathBuf::from("/var/lib/rnode/node.key.pem"),
                secure_random_non_blocking: false,
                custom_certificate_location: false,
                custom_key_location: false,
            },
            casper: casper::rust::casper_conf::CasperConf {
                fault_tolerance_threshold: 0.1,
                validator_public_key: None,
                validator_private_key: None,
                validator_private_key_path: None,
                shard_name: "".to_string(),
                parent_shard_id: "/".to_string(),
                casper_loop_interval: Duration::from_secs(1),
                requested_blocks_timeout: Duration::from_secs(30),
                finalization_rate: 1,
                max_number_of_parents: 1,
                max_parent_depth: 1,
                fork_choice_stale_threshold: Duration::from_secs(30),
                fork_choice_check_if_stale_interval: Duration::from_secs(30),
                synchrony_constraint_threshold: 0.0,
                height_constraint_threshold: 1,
                round_robin_dispatcher: casper::rust::casper_conf::RoundRobinDispatcher {
                    max_peer_queue_size: 4,
                    give_up_after_skipped: 4,
                    drop_peer_after_retries: 4,
                },
                genesis_block_data: casper::rust::casper_conf::GenesisBlockData {
                    genesis_data_dir: "/var/lib/rnode/genesis".to_string(),
                    bonds_file: "".to_string(),
                    wallets_file: "".to_string(),
                    bond_minimum: 1,
                    bond_maximum: 1000000000,
                    epoch_length: 10000,
                    quarantine_length: 10000,
                    number_of_active_validators: 5,
                    genesis_block_number: 0,
                    pos_multi_sig_public_keys: vec![],
                    pos_multi_sig_quorum: 0,
                },
                genesis_ceremony: casper::rust::casper_conf::GenesisCeremony {
                    required_signatures: 0,
                    approve_interval: Duration::from_secs(30),
                    approve_duration: Duration::from_secs(30),
                    autogen_shard_size: 5,
                    genesis_validator_mode: false,
                    ceremony_master_mode: false,
                },
                min_phlo_price: 1,
            },
            metrics: crate::rust::configuration::model::Metrics {
                prometheus: false,
                influxdb: false,
                influxdb_udp: false,
                zipkin: false,
                sigar: false,
            },
            dev_mode: false,
            dev: crate::rust::configuration::model::DevConf {
                deployer_private_key: None,
            },
        };

        // Apply CLI options to the configuration
        default_config.override_config_values(options);

        // Verify that CLI options have overridden the default values
        assert_eq!(default_config.standalone, true);
        assert_eq!(default_config.autopropose, false);
        assert_eq!(default_config.dev_mode, true);

        // Protocol server fields
        assert_eq!(
            default_config.protocol_server.network_id,
            "testnet".to_string()
        );
        assert_eq!(
            default_config.protocol_server.host,
            Some("localhost".to_string())
        );
        assert_eq!(default_config.protocol_server.allow_private_addresses, true);
        assert_eq!(default_config.protocol_server.use_random_ports, true);
        assert_eq!(default_config.protocol_server.dynamic_ip, true);
        assert_eq!(default_config.protocol_server.no_upnp, true);
        assert_eq!(default_config.protocol_server.port, 11111);
        assert_eq!(
            default_config.protocol_server.grpc_max_recv_message_size,
            111111
        );
        assert_eq!(
            default_config
                .protocol_server
                .grpc_max_recv_stream_message_size,
            111111
        );
        assert_eq!(default_config.protocol_server.max_message_consumers, 111111);
        assert_eq!(default_config.protocol_server.disable_state_exporter, true);

        // Protocol client fields
        assert_eq!(default_config.protocol_client.disable_lfs, true);
        assert_eq!(default_config.protocol_client.bootstrap, "rnode://de6eed5d00cf080fc587eeb412cb31a75fd10358@52.119.8.109?protocol=40400&discovery=40404".to_string());
        assert_eq!(
            default_config.protocol_client.network_timeout,
            Duration::from_secs(111111)
        );
        assert_eq!(default_config.protocol_client.batch_max_connections, 111111);
        assert_eq!(
            default_config.protocol_client.grpc_max_recv_message_size,
            111111
        );
        assert_eq!(
            default_config.protocol_client.grpc_stream_chunk_size,
            111111
        );

        // Peers discovery fields
        assert_eq!(default_config.peers_discovery.port, 11111);
        assert_eq!(
            default_config.peers_discovery.lookup_interval,
            Duration::from_secs(111111)
        );
        assert_eq!(
            default_config.peers_discovery.cleanup_interval,
            Duration::from_secs(111111)
        );
        assert_eq!(default_config.peers_discovery.heartbeat_batch_size, 111111);
        assert_eq!(
            default_config.peers_discovery.init_wait_loop_interval,
            Duration::from_secs(111111)
        );

        // API server fields
        assert_eq!(default_config.api_server.enable_reporting, true);
        assert_eq!(default_config.api_server.port_grpc_external, 11111);
        assert_eq!(default_config.api_server.port_grpc_internal, 11111);
        assert_eq!(default_config.api_server.port_http, 11111);
        assert_eq!(default_config.api_server.port_admin_http, 11111);
        assert_eq!(default_config.api_server.host, "localhost".to_string());
        assert_eq!(default_config.api_server.grpc_max_recv_message_size, 111111);
        assert_eq!(default_config.api_server.max_blocks_limit, 111111);
        assert_eq!(
            default_config.api_server.keep_alive_time,
            Duration::from_secs(111111)
        );
        assert_eq!(
            default_config.api_server.keep_alive_timeout,
            Duration::from_secs(111111)
        );
        assert_eq!(
            default_config.api_server.permit_keep_alive_time,
            Duration::from_secs(111111)
        );
        assert_eq!(
            default_config.api_server.max_connection_idle,
            Duration::from_secs(111111)
        );
        assert_eq!(
            default_config.api_server.max_connection_age,
            Duration::from_secs(111111)
        );
        assert_eq!(
            default_config.api_server.max_connection_age_grace,
            Duration::from_secs(111111)
        );

        // Storage fields
        assert_eq!(
            default_config.storage.data_dir,
            PathBuf::from("/var/lib/rnode")
        );

        // TLS fields
        assert_eq!(default_config.tls.secure_random_non_blocking, true);
        assert_eq!(
            default_config.tls.key_path,
            PathBuf::from("/var/lib/rnode/node.key.pem")
        );
        assert_eq!(
            default_config.tls.certificate_path,
            PathBuf::from("/var/lib/rnode/node.certificate.pem")
        );

        // Metrics fields
        assert_eq!(default_config.metrics.prometheus, true);
        assert_eq!(default_config.metrics.influxdb, true);
        assert_eq!(default_config.metrics.influxdb_udp, true);
        assert_eq!(default_config.metrics.zipkin, true);
        assert_eq!(default_config.metrics.sigar, true);

        // Dev fields
        assert_eq!(
            default_config.dev.deployer_private_key,
            Some("test-key".to_string())
        );

        // Casper configuration fields
        assert_eq!(default_config.casper.shard_name, "root".to_string());
        assert_eq!(default_config.casper.fault_tolerance_threshold, 111111.0);
        assert_eq!(
            default_config.casper.validator_public_key,
            Some("111111".to_string())
        );
        assert_eq!(
            default_config.casper.validator_private_key,
            Some("111111".to_string())
        );
        assert_eq!(
            default_config.casper.validator_private_key_path,
            Some(PathBuf::from("/var/lib/rnode/pem.key"))
        );
        assert_eq!(
            default_config.casper.casper_loop_interval,
            Duration::from_secs(111111)
        );
        assert_eq!(
            default_config.casper.requested_blocks_timeout,
            Duration::from_secs(111111)
        );
        assert_eq!(default_config.casper.finalization_rate, 111111);
        assert_eq!(default_config.casper.max_number_of_parents, 111111);
        assert_eq!(default_config.casper.max_parent_depth, 111111);
        assert_eq!(
            default_config.casper.fork_choice_stale_threshold,
            Duration::from_secs(111111)
        );
        assert_eq!(
            default_config.casper.fork_choice_check_if_stale_interval,
            Duration::from_secs(111111)
        );
        assert_eq!(
            default_config.casper.synchrony_constraint_threshold,
            111111.0
        );
        assert_eq!(default_config.casper.height_constraint_threshold, 111111);
        assert_eq!(default_config.casper.min_phlo_price, 1);

        // Round robin dispatcher fields
        assert_eq!(
            default_config
                .casper
                .round_robin_dispatcher
                .max_peer_queue_size,
            111111
        );
        assert_eq!(
            default_config
                .casper
                .round_robin_dispatcher
                .give_up_after_skipped,
            111111
        );
        assert_eq!(
            default_config
                .casper
                .round_robin_dispatcher
                .drop_peer_after_retries,
            111111
        );

        // Genesis block data fields
        assert_eq!(
            default_config.casper.genesis_block_data.bonds_file,
            "/var/lib/rnode/genesis/bonds1.txt".to_string()
        );
        assert_eq!(
            default_config.casper.genesis_block_data.wallets_file,
            "/var/lib/rnode/genesis/wallets1.txt".to_string()
        );
        assert_eq!(
            default_config.casper.genesis_block_data.bond_minimum,
            111111
        );
        assert_eq!(
            default_config.casper.genesis_block_data.bond_maximum,
            111111
        );
        assert_eq!(
            default_config.casper.genesis_block_data.epoch_length,
            111111
        );
        assert_eq!(
            default_config.casper.genesis_block_data.quarantine_length,
            111111
        );
        assert_eq!(
            default_config
                .casper
                .genesis_block_data
                .number_of_active_validators,
            111111
        );
        assert_eq!(
            default_config
                .casper
                .genesis_block_data
                .genesis_block_number,
            222
        );

        // Genesis ceremony fields
        assert_eq!(
            default_config.casper.genesis_ceremony.required_signatures,
            111111
        );
        assert_eq!(
            default_config.casper.genesis_ceremony.approve_interval,
            Duration::from_secs(111111)
        );
        assert_eq!(
            default_config.casper.genesis_ceremony.approve_duration,
            Duration::from_secs(111111)
        );
        assert_eq!(
            default_config.casper.genesis_ceremony.autogen_shard_size,
            111111
        );
        assert_eq!(
            default_config
                .casper
                .genesis_ceremony
                .genesis_validator_mode,
            true
        );
    }
}
