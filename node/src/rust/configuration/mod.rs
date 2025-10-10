//! Configuration module for F1r3fly node.
//!
//! This module provides configuration management for the F1r3fly node,
//! including command-line argument parsing, configuration file loading,
//! and configuration merging with proper precedence.

pub mod commandline;
pub mod kamon;
pub mod model;

pub use commandline::Options;
pub use kamon::KamonConf;
pub use model::{NodeConf, Profile};

/// Configuration building and parsing functionality
pub mod builder {
    use super::*;
    use crate::rust::configuration::commandline::ConfigMapper;
    use std::{
        collections::HashMap,
        env,
        path::{Path, PathBuf},
    };

    /// Builds Configuration instance from CLI options.
    /// If config file is provided as part of CLI options, it shall be parsed and merged
    /// with CLI options having higher priority.
    ///
    /// # Arguments
    /// * `options` - CLI options
    ///
    /// # Returns
    /// * `Result<(NodeConf, Profile, Option<PathBuf>, KamonConf)>` - Configuration tuple
    pub fn build(
        default_dir: &Path,
        options: Options,
    ) -> eyre::Result<(NodeConf, Profile, Option<PathBuf>, KamonConf)> {
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

        let default_config_file = default_dir.join("defaults.conf");

        let config_file: Option<PathBuf> = if config_file_path.exists() {
            Some(config_file_path)
        } else {
            None
        };

        // Build configuration from multiple sources with proper precedence:
        // 1. CLI options (highest priority)
        // 2. Config file
        // 3. Default configuration (lowest priority)

        // Loading the default configuration
        let default_config = hocon::HoconLoader::new().load_file(default_config_file)?;

        // Merging the default configuration with the data directory config
        let merged_config = config_file
            .as_ref()
            .map(|config_file| default_config.load_file(config_file))
            .unwrap_or(Ok(default_config))?;

        let mut node_conf: NodeConf = merged_config.resolve()?;

        // override config values with CLI options
        node_conf.override_config_values(options);

        // Validate configuration
        validate_config(&node_conf)?;

        let kamon_config = load_kamon_config(default_dir, &data_dir)?;
        let node_conf = check_dev_mode(node_conf);

        Ok((node_conf, profile, config_file, kamon_config))
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

        Ok(())
    }

    /// Load Kamon configuration
    fn load_kamon_config(default_dir: &Path, data_dir: &Path) -> eyre::Result<KamonConf> {
        let default_kamon_config_file = default_dir.join("kamon.conf");

        let default_kamon_config =
            hocon::HoconLoader::new().load_file(default_kamon_config_file)?;

        let kamon_config_file = data_dir.join("kamon.conf");

        let kamon_config_file: Option<PathBuf> = if kamon_config_file.exists() {
            Some(kamon_config_file)
        } else {
            None
        };

        // Merging the default configuration with the data directory config
        let merged_config = kamon_config_file
            .map(|config_file| default_kamon_config.load_file(config_file))
            .unwrap_or(Ok(default_kamon_config))?;

        let kamon_conf: KamonConf = merged_config.resolve()?;

        Ok(kamon_conf)
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
