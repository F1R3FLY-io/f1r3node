//! Configuration module for F1r3fly node.
//!
//! This module provides configuration management for the F1r3fly node,
//! including command-line argument parsing, configuration file loading,
//! and configuration merging with proper precedence.

pub mod commandline;
pub mod config_check;
pub mod model;

pub use commandline::Options;
pub use model::{NodeConf, Profile};

/// Embedded HOCON defaults — what every node starts from before applying
/// the optional `<data-dir>/rnode.conf` override and CLI flags. Baked in
/// at compile time so the binary is self-contained (no `DEFAULT_DIR` env
/// var, no on-disk `node/src/main/resources/defaults.conf` lookup).
const EMBEDDED_DEFAULTS: &str = include_str!("../../main/resources/defaults.conf");

/// Configuration building and parsing functionality
pub mod builder {
    use super::*;
    use crate::rust::configuration::commandline::ConfigMapper;
    use std::{collections::HashMap, env, path::PathBuf};

    /// Builds Configuration instance from CLI options.
    /// If config file is provided as part of CLI options, it shall be parsed and merged
    /// with CLI options having higher priority.
    ///
    /// # Arguments
    /// * `options` - CLI options
    ///
    /// # Returns
    /// * `Result<(NodeConf, Profile, Option<PathBuf>)>` - Configuration tuple
    pub fn build(options: Options) -> eyre::Result<(NodeConf, Profile, Option<PathBuf>)> {
        let profile = options
            .profile
            .as_ref()
            .and_then(|p| profiles().get(p).cloned())
            .unwrap_or_else(|| default_profile());

        let (data_dir, config_file_path) = options
            .subcommand
            .as_ref()
            .and_then(|subcommand| match &subcommand {
                &commandline::options::OptionsSubCommand::Run(run_options) => Some((
                    run_options
                        .data_dir
                        .clone()
                        .unwrap_or_else(|| profile.data_dir.0.clone()),
                    run_options
                        .config_file
                        .clone()
                        .unwrap_or_else(|| profile.data_dir.0.join("rnode.conf")),
                )),
                _ => None,
            })
            .unwrap_or_else(|| {
                (
                    profile.data_dir.0.clone(),
                    profile.data_dir.0.join("rnode.conf"),
                )
            });

        let config_file: Option<PathBuf> = if config_file_path.exists() {
            Some(config_file_path)
        } else {
            None
        };

        // Build configuration from multiple sources with proper precedence:
        // 1. CLI options (highest priority)
        // 2. Config file (`<data-dir>/rnode.conf` or `--config-file <path>`)
        // 3. Embedded defaults baked into the binary (lowest priority)
        let default_config = hocon::HoconLoader::new().load_str(super::EMBEDDED_DEFAULTS)?;

        // Merging the embedded defaults with the optional override
        let merged_config = config_file
            .as_ref()
            .map(|config_file| default_config.load_file(config_file))
            .unwrap_or(Ok(default_config))?;

        let mut node_conf: NodeConf = merged_config.resolve()?;

        // Set data_dir if it's empty (HOCON couldn't resolve ${default-data-dir})
        if node_conf.storage.data_dir.as_os_str().is_empty() {
            node_conf.storage.data_dir = data_dir.clone();
            // Also fix TLS paths which depend on data_dir
            node_conf.tls.certificate_path = data_dir.join("node.certificate.pem");
            node_conf.tls.key_path = data_dir.join("node.key.pem");
            // Fix genesis data dir which also depends on data_dir
            node_conf.casper.genesis_block_data.genesis_data_dir =
                data_dir.join("genesis").to_string_lossy().to_string();
            node_conf.casper.genesis_block_data.bonds_file = data_dir
                .join("genesis")
                .join("bonds.txt")
                .to_string_lossy()
                .to_string();
            node_conf.casper.genesis_block_data.wallets_file = data_dir
                .join("genesis")
                .join("wallets.txt")
                .to_string_lossy()
                .to_string();
        }

        // override config values with CLI options
        node_conf.override_config_values(options);

        // Validate configuration
        validate_config(&node_conf)?;

        let node_conf = check_dev_mode(node_conf);

        Ok((node_conf, profile, config_file))
    }

    /// Validate configuration parameters
    fn validate_config(node_conf: &NodeConf) -> eyre::Result<()> {
        let pos_multi_sig_quorum = node_conf.casper.genesis_block_data.pos_multi_sig_quorum;
        let pos_multi_sig_public_keys_length = node_conf
            .casper
            .genesis_block_data
            .pos_multi_sig_public_keys
            .len();

        if pos_multi_sig_quorum > pos_multi_sig_public_keys_length as u32 {
            eyre::bail!(
                "defaults.conf: The value 'pos-multi-sig-quorum' should be less or equal the length of 'pos-multi-sig-public-keys' \
                (the actual values are '{}' and '{}' respectively)",
                pos_multi_sig_quorum,
                pos_multi_sig_public_keys_length
            );
        }

        // Reject empty/whitespace native token name/symbol and out-of-range
        // decimals before the node starts. Catches misconfigured shell variable
        // expansion, typos, and values outside the industry-standard range.
        node_conf
            .casper
            .genesis_block_data
            .validate_native_token()
            .map_err(|e| eyre::eyre!("native token config invalid: {}", e))?;

        // The proposer computes its recovery cap as
        // `max(pending_deploy_max_lag, deploy_recovery_max_lag)`. When
        // deploy_recovery is set below pending_deploy, the recovery knob
        // collapses to the pending floor and has no effect — warn the
        // operator instead of letting the misconfiguration sit silently.
        let pending_deploy_max_lag = node_conf
            .casper
            .heartbeat_conf
            .advanced
            .pending_deploy_max_lag;
        let deploy_recovery_max_lag = node_conf
            .casper
            .heartbeat_conf
            .advanced
            .deploy_recovery_max_lag;
        if deploy_recovery_max_lag < pending_deploy_max_lag {
            tracing::warn!(
                "casper.heartbeat.advanced.deploy-recovery-max-lag ({}) is less than \
                pending-deploy-max-lag ({}); the recovery knob has no effect under this \
                configuration. Set deploy-recovery-max-lag >= pending-deploy-max-lag.",
                deploy_recovery_max_lag,
                pending_deploy_max_lag,
            );
        }

        Ok(())
    }

    /// Check dev mode and adjust configuration accordingly
    fn check_dev_mode(node_conf: NodeConf) -> NodeConf {
        if node_conf.dev_mode {
            node_conf
        } else {
            if node_conf.dev.deployer_private_key.is_some() {
                println!("Node is not in dev mode, ignoring --deployer-private-key");
            }
            NodeConf {
                dev: model::DevConf {
                    deployer_private_key: None,
                },
                ..node_conf
            }
        }
    }

    fn docker_profile() -> Profile {
        Profile {
            name: "docker",
            data_dir: (
                PathBuf::from("/var/lib/rnode"),
                "Defaults to /var/lib/rnode",
            ),
        }
    }

    fn default_profile() -> Profile {
        // Resolve $HOME (fallback to current dir if not set)
        let home = env::var_os("HOME")
            .map(PathBuf::from)
            .or_else(|| env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."));

        let path = home.join(".rnode");

        Profile {
            name: "default",
            data_dir: (path, "Defaults to $HOME/.rnode"),
        }
    }

    pub fn profiles() -> HashMap<String, Profile> {
        let mut map = HashMap::new();
        let def = default_profile();
        let dock = docker_profile();
        map.insert(def.name.to_string(), def);
        map.insert(dock.name.to_string(), dock);
        map
    }
}

// Re-export commonly used types
pub use builder::build;

#[cfg(test)]
mod heartbeat_conf_hocon_tests {
    //! Targeted HOCON deserialization tests for the heartbeat tuning fields.
    //!
    //! Lives here (rather than in `casper::casper_conf`) because the `hocon`
    //! crate is a `node` dependency, not a `casper` dependency. Exercises
    //! the same `serde::Deserialize` path the production binary uses.

    use casper::rust::casper_conf::{HeartbeatAdvancedConf, HeartbeatConf};
    use std::time::Duration;

    fn parse_heartbeat(hocon_text: &str) -> HeartbeatConf {
        try_parse_heartbeat(hocon_text).expect("HOCON should deserialize into HeartbeatConf")
    }

    fn try_parse_heartbeat(hocon_text: &str) -> Result<HeartbeatConf, String> {
        let loader = hocon::HoconLoader::new()
            .load_str(hocon_text)
            .map_err(|e| format!("hocon load: {e}"))?;
        loader.resolve().map_err(|e| format!("hocon resolve: {e}"))
    }

    #[test]
    fn full_block_with_advanced_round_trips() {
        let cfg = parse_heartbeat(
            r#"
            enabled = true
            check-interval = 7 seconds
            max-lfb-age = 8 seconds
            self-propose-cooldown = 9 seconds
            stale-recovery-min-interval = 11 seconds
            deploy-finalization-grace = 22 seconds
            advanced {
              frontier-chase-max-lag = 1
              pending-deploy-max-lag = 33
              deploy-recovery-max-lag = 99
            }
            "#,
        );

        assert!(cfg.enabled);
        assert_eq!(cfg.check_interval, Duration::from_secs(7));
        assert_eq!(cfg.max_lfb_age, Duration::from_secs(8));
        assert_eq!(cfg.self_propose_cooldown, Duration::from_secs(9));
        assert_eq!(cfg.stale_recovery_min_interval, Duration::from_secs(11));
        assert_eq!(cfg.deploy_finalization_grace, Duration::from_secs(22));
        assert_eq!(cfg.advanced.frontier_chase_max_lag, 1);
        assert_eq!(cfg.advanced.pending_deploy_max_lag, 33);
        assert_eq!(cfg.advanced.deploy_recovery_max_lag, 99);
    }

    #[test]
    fn missing_new_fields_fall_back_to_defaults() {
        // A HOCON config that omits the new keys must still parse and use
        // the defaults declared on HeartbeatConf / HeartbeatAdvancedConf.
        let cfg = parse_heartbeat(
            r#"
            enabled = false
            check-interval = 5 seconds
            max-lfb-age = 5 seconds
            "#,
        );

        assert_eq!(cfg.self_propose_cooldown, Duration::from_secs(15));
        assert_eq!(cfg.stale_recovery_min_interval, Duration::from_secs(12));
        assert_eq!(cfg.deploy_finalization_grace, Duration::from_secs(25));
        // Advanced block absent → all three fields default.
        assert_eq!(cfg.advanced, HeartbeatAdvancedConf::default());
    }

    #[test]
    fn partial_advanced_block_defaults_remaining_fields() {
        // A partial advanced block fills missing fields with defaults
        // rather than failing to parse.
        let cfg = parse_heartbeat(
            r#"
            enabled = false
            check-interval = 5 seconds
            max-lfb-age = 5 seconds
            advanced {
              pending-deploy-max-lag = 7
            }
            "#,
        );

        assert_eq!(cfg.advanced.frontier_chase_max_lag, 0);
        assert_eq!(cfg.advanced.pending_deploy_max_lag, 7);
        assert_eq!(cfg.advanced.deploy_recovery_max_lag, 64);
    }

    #[test]
    fn negative_advanced_lag_values_are_rejected() {
        // Negative caps would silently disable the corresponding code
        // path in the proposer (e.g. `lag <= cap` where cap < 0 is
        // never true). Each of the three advanced fields rejects at
        // deserialization time.
        for field in &[
            "frontier-chase-max-lag",
            "pending-deploy-max-lag",
            "deploy-recovery-max-lag",
        ] {
            let hocon = format!(
                r#"
                enabled = false
                check-interval = 5 seconds
                max-lfb-age = 5 seconds
                advanced {{
                  {field} = -1
                }}
                "#
            );
            let result = try_parse_heartbeat(&hocon);
            assert!(
                result.is_err(),
                "negative {field} should fail HOCON deserialization, got Ok({:?})",
                result.ok()
            );
            let err = result.unwrap_err();
            assert!(
                err.contains("value must be >= 0"),
                "error for {field} should mention non-negative requirement, got: {err}"
            );
        }
    }
}
