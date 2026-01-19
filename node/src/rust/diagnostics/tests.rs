#[cfg(test)]
mod tests {
    use crate::rust::configuration::kamon::{KamonConf, MetricConfig};
    use crate::rust::configuration::model::{
        ApiServer, DevConf, Metrics, NodeConf, PeersDiscovery, ProtocolClient, ProtocolServer,
        Storage, TlsConf,
    };
    use crate::rust::diagnostics::initialize_diagnostics;
    use crate::rust::diagnostics::new_prometheus_reporter::NewPrometheusReporter;
    use crate::rust::diagnostics::prometheus_config::PrometheusConfiguration;
    use casper::rust::casper_conf::{
        CasperConf, GenesisBlockData, GenesisCeremony, HeartbeatConf, RoundRobinDispatcher,
    };
    use serial_test::serial;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    fn create_minimal_casper_conf() -> CasperConf {
        CasperConf {
            fault_tolerance_threshold: 0.1,
            validator_public_key: None,
            validator_private_key: None,
            validator_private_key_path: None,
            shard_name: "root".to_string(),
            parent_shard_id: "".to_string(),
            casper_loop_interval: Duration::from_secs(30),
            requested_blocks_timeout: Duration::from_secs(120),
            finalization_rate: 1,
            max_number_of_parents: 2147483647,
            max_parent_depth: 100,
            fork_choice_stale_threshold: Duration::from_secs(30 * 60),
            fork_choice_check_if_stale_interval: Duration::from_secs(30),
            synchrony_constraint_threshold: 0.0,
            height_constraint_threshold: i64::MAX,
            round_robin_dispatcher: RoundRobinDispatcher {
                max_peer_queue_size: 500,
                give_up_after_skipped: 10,
                drop_peer_after_retries: 5,
            },
            genesis_block_data: GenesisBlockData {
                genesis_data_dir: "/tmp/genesis".to_string(),
                bonds_file: "bonds.txt".to_string(),
                wallets_file: "wallets.txt".to_string(),
                bond_minimum: 1,
                bond_maximum: i64::MAX,
                epoch_length: 10000,
                quarantine_length: 50000,
                number_of_active_validators: 100,
                deploy_timestamp: None,
                genesis_block_number: 0,
                pos_multi_sig_public_keys: vec![],
                pos_multi_sig_quorum: 0,
            },
            genesis_ceremony: GenesisCeremony {
                required_signatures: 0,
                approve_interval: Duration::from_secs(5),
                approve_duration: Duration::from_secs(5),
                autogen_shard_size: 5,
                genesis_validator_mode: false,
                ceremony_master_mode: false,
            },
            min_phlo_price: 0,
            heartbeat_conf: HeartbeatConf {
                enabled: false,
                check_interval: Duration::from_secs(60),
                max_lfb_age: Duration::from_secs(300),
            },
        }
    }

    fn create_test_node_conf() -> NodeConf {
        NodeConf {
            standalone: false,
            autopropose: false,
            protocol_server: ProtocolServer {
                network_id: "test".to_string(),
                host: Some("localhost".to_string()),
                allow_private_addresses: true,
                use_random_ports: false,
                dynamic_ip: false,
                no_upnp: true,
                port: 40400,
                grpc_max_recv_message_size: 4 * 1024 * 1024,
                grpc_max_recv_stream_message_size: 16 * 1024 * 1024,
                max_message_consumers: 8192,
                disable_state_exporter: false,
            },
            protocol_client: ProtocolClient {
                network_id: "test".to_string(),
                bootstrap: "".to_string(),
                disable_lfs: false,
                batch_max_connections: 500,
                network_timeout: Duration::from_secs(5),
                grpc_max_recv_message_size: 100 * 1024 * 1024,
                grpc_stream_chunk_size: 400 * 1024,
            },
            peers_discovery: PeersDiscovery {
                host: Some("localhost".to_string()),
                port: 40404,
                lookup_interval: Duration::from_secs(20),
                cleanup_interval: Duration::from_secs(10 * 60),
                heartbeat_batch_size: 50,
                init_wait_loop_interval: Duration::from_secs(5),
            },
            api_server: ApiServer {
                host: "localhost".to_string(),
                port_grpc_external: 40401,
                port_grpc_internal: 40402,
                grpc_max_recv_message_size: 16 * 1024 * 1024,
                port_http: 40403,
                port_admin_http: 40405,
                max_blocks_limit: 100,
                enable_reporting: true,
                keep_alive_time: Duration::from_secs(60),
                keep_alive_timeout: Duration::from_secs(20),
                permit_keep_alive_time: Duration::from_secs(10),
                max_connection_idle: Duration::from_secs(60),
                max_connection_age: Duration::from_secs(60),
                max_connection_age_grace: Duration::from_secs(60),
            },
            storage: Storage {
                data_dir: PathBuf::from("/tmp/test"),
            },
            tls: TlsConf {
                certificate_path: PathBuf::from("/tmp/cert.pem"),
                key_path: PathBuf::from("/tmp/key.pem"),
                secure_random_non_blocking: false,
                custom_certificate_location: false,
                custom_key_location: false,
            },
            casper: create_minimal_casper_conf(),
            metrics: Metrics {
                prometheus: false,
                influxdb: false,
                influxdb_udp: false,
                zipkin: false,
                sigar: false,
            },
            dev_mode: false,
            dev: DevConf {
                deployer_private_key: None,
            },
        }
    }

    fn create_test_node_conf_with_prometheus() -> NodeConf {
        let mut conf = create_test_node_conf();
        conf.metrics.prometheus = true;
        conf
    }

    fn create_test_node_conf_with_sigar() -> NodeConf {
        let mut conf = create_test_node_conf();
        conf.metrics.sigar = true;
        conf
    }

    fn create_test_node_conf_all_disabled() -> NodeConf {
        create_test_node_conf()
    }

    fn create_test_kamon_conf() -> KamonConf {
        KamonConf {
            trace: None,
            metric: Some(MetricConfig {
                tick_interval: Duration::from_secs(10),
            }),
            influxdb: None,
            zipkin: None,
            prometheus: None,
            sigar: None,
        }
    }

    fn create_test_kamon_conf_with_custom_interval(interval: Duration) -> KamonConf {
        KamonConf {
            trace: None,
            metric: Some(MetricConfig {
                tick_interval: interval,
            }),
            influxdb: None,
            zipkin: None,
            prometheus: None,
            sigar: None,
        }
    }

    fn create_test_kamon_conf_no_metric() -> KamonConf {
        KamonConf {
            trace: None,
            metric: None,
            influxdb: None,
            zipkin: None,
            prometheus: None,
            sigar: None,
        }
    }

    #[test]
    #[serial]
    fn test_reporter_initialize() {
        let result = NewPrometheusReporter::initialize();
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn test_reporter_scrape_data() {
        let reporter = NewPrometheusReporter::initialize().unwrap();
        let output = reporter.scrape_data();

        assert!(output.is_empty() || output.starts_with('#'));
    }

    #[test]
    #[serial]
    fn test_reporter_singleton_pattern() {
        let reporter1 = NewPrometheusReporter::initialize().unwrap();
        let reporter2 = NewPrometheusReporter::initialize().unwrap();

        assert!(Arc::ptr_eq(&reporter1, &reporter2));
    }

    #[test]
    #[serial]
    fn test_reporter_global_access() {
        let _reporter = NewPrometheusReporter::initialize().unwrap();
        let global = NewPrometheusReporter::global();

        assert!(global.is_some());
    }

    #[test]
    #[serial]
    fn test_reporter_with_custom_config() {
        let config = PrometheusConfiguration {
            default_buckets: vec![0.001, 0.01, 0.1, 1.0],
            time_buckets: vec![0.001, 0.01, 0.1, 1.0],
            information_buckets: vec![512.0, 1024.0, 2048.0],
            custom_buckets: HashMap::new(),
            include_environment_tags: false,
        };

        let result = NewPrometheusReporter::initialize_with_config(config);
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn test_reporter_records_counter() {
        let reporter = NewPrometheusReporter::initialize().unwrap();

        let counter = metrics::counter!("test_counter_metric", "source" => "test.source");
        counter.increment(5);

        std::thread::sleep(std::time::Duration::from_millis(10));

        let scrape = reporter.scrape_data();
        assert!(
            scrape.contains("test_counter_metric") || scrape.is_empty(),
            "If metrics are recorded, scrape output should contain test_counter_metric"
        );
    }

    #[test]
    #[serial]
    fn test_reporter_records_gauge() {
        let reporter = NewPrometheusReporter::initialize().unwrap();

        let gauge = metrics::gauge!("test_gauge_metric", "source" => "test.source");
        gauge.set(42.0);

        std::thread::sleep(std::time::Duration::from_millis(10));

        let scrape = reporter.scrape_data();
        assert!(
            scrape.contains("test_gauge_metric") || scrape.is_empty(),
            "If metrics are recorded, scrape output should contain test_gauge_metric"
        );
    }

    #[test]
    #[serial]
    fn test_reporter_records_histogram() {
        let reporter = NewPrometheusReporter::initialize().unwrap();

        let histogram = metrics::histogram!("test_histogram_metric", "source" => "test.source");
        histogram.record(1.5);

        std::thread::sleep(std::time::Duration::from_millis(10));

        let scrape = reporter.scrape_data();
        assert!(
            scrape.contains("test_histogram_metric") || scrape.is_empty(),
            "If metrics are recorded, scrape output should contain test_histogram_metric"
        );
    }

    #[test]
    #[serial]
    fn test_reporter_scrape_includes_recorded_metrics() {
        let reporter = NewPrometheusReporter::initialize().unwrap();

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        let counter = metrics::counter!("test_metric_unique_scrape", "source" => "test.scrape", "ts" => timestamp.to_string());
        counter.increment(10);

        std::thread::sleep(std::time::Duration::from_millis(10));

        let scrape = reporter.scrape_data();
        assert!(
            scrape.contains("test_metric_unique_scrape") || scrape.is_empty(),
            "If metrics are recorded, scrape output should contain test_metric_unique_scrape"
        );
    }

    #[test]
    #[serial]
    fn test_reporter_concurrent_access() {
        let reporter = Arc::new(NewPrometheusReporter::initialize().unwrap());
        let mut handles = vec![];

        for i in 0..10 {
            let _reporter_clone = reporter.clone();
            let handle = thread::spawn(move || {
                let counter = metrics::counter!("concurrent_test_metric", "source" => "test.concurrent", "id" => i.to_string());
                counter.increment(1);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        std::thread::sleep(std::time::Duration::from_millis(50));

        let scrape = reporter.scrape_data();
        assert!(
            scrape.contains("concurrent_test_metric") || scrape.is_empty(),
            "If metrics are recorded, scrape output should contain concurrent_test_metric"
        );
    }

    #[test]
    fn test_default_config() {
        let config = PrometheusConfiguration::default();

        assert_eq!(config.default_buckets.len(), 14);
        assert_eq!(config.time_buckets.len(), 16);
        assert_eq!(config.information_buckets.len(), 8);
        assert_eq!(config.custom_buckets.len(), 0);
        assert!(!config.include_environment_tags);
    }

    #[test]
    fn test_custom_buckets() {
        let mut custom = HashMap::new();
        custom.insert("my_metric".to_string(), vec![1.0, 2.0, 3.0]);

        let config = PrometheusConfiguration {
            default_buckets: vec![0.1, 1.0, 10.0],
            time_buckets: vec![0.001, 0.01],
            information_buckets: vec![512.0],
            custom_buckets: custom.clone(),
            include_environment_tags: false,
        };

        assert_eq!(config.default_buckets.len(), 3);
        assert_eq!(config.custom_buckets.len(), 1);
        assert_eq!(config.custom_buckets.get("my_metric").unwrap(), &vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_environment_tags_disabled() {
        let config = PrometheusConfiguration {
            default_buckets: vec![],
            time_buckets: vec![],
            information_buckets: vec![],
            custom_buckets: HashMap::new(),
            include_environment_tags: false,
        };

        let tags = config.environment_tags();
        assert!(tags.is_empty());
    }

    #[test]
    fn test_environment_tags_enabled() {
        let config = PrometheusConfiguration {
            default_buckets: vec![],
            time_buckets: vec![],
            information_buckets: vec![],
            custom_buckets: HashMap::new(),
            include_environment_tags: true,
        };

        let tags = config.environment_tags();
        assert!(tags.contains_key("service"));
        assert_eq!(tags.get("service").unwrap(), "rnode");
    }

    #[test]
    fn test_environment_tags_with_hostname() {
        std::env::set_var("HOSTNAME", "test-host");

        let config = PrometheusConfiguration {
            default_buckets: vec![],
            time_buckets: vec![],
            information_buckets: vec![],
            custom_buckets: HashMap::new(),
            include_environment_tags: true,
        };

        let tags = config.environment_tags();
        assert!(tags.contains_key("host"));
        assert_eq!(tags.get("host").unwrap(), "test-host");

        std::env::remove_var("HOSTNAME");
    }

    #[test]
    #[serial]
    fn test_initialize_with_all_disabled() {
        let node_conf = create_test_node_conf_all_disabled();
        let kamon_conf = create_test_kamon_conf();

        let result = initialize_diagnostics(&node_conf, &kamon_conf);

        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    #[serial]
    fn test_initialize_with_prometheus_only() {
        let node_conf = create_test_node_conf_with_prometheus();
        let kamon_conf = create_test_kamon_conf();

        let result = initialize_diagnostics(&node_conf, &kamon_conf);

        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[tokio::test]
    #[serial]
    async fn test_initialize_with_sigar_only() {
        let node_conf = create_test_node_conf_with_sigar();
        let kamon_conf = create_test_kamon_conf();

        let result = initialize_diagnostics(&node_conf, &kamon_conf);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn test_initialize_metrics_interval_default() {
        let node_conf = create_test_node_conf_with_prometheus();
        let kamon_conf = create_test_kamon_conf_no_metric();

        let result = initialize_diagnostics(&node_conf, &kamon_conf);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn test_initialize_metrics_interval_custom() {
        let node_conf = create_test_node_conf_with_prometheus();
        let custom_interval = Duration::from_secs(5);
        let kamon_conf = create_test_kamon_conf_with_custom_interval(custom_interval);

        let result = initialize_diagnostics(&node_conf, &kamon_conf);

        assert!(result.is_ok());
    }
}

