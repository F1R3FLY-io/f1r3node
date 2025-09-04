use crate::args::*;
use crate::commands::*;
use crate::error::{NodeCliError, Result};
use crate::utils::print_error;

/// Central command dispatcher that routes and executes all CLI commands
pub struct Dispatcher;

impl Dispatcher {
    /// Dispatch a command to its appropriate handler
    pub async fn dispatch(cli: &Cli) -> Result<()> {
        let result = match &cli.command {
            Commands::Deploy(args) => deploy_command(args).await.map_err(NodeCliError::from),
            Commands::Propose(args) => propose_command(args).await.map_err(NodeCliError::from),
            Commands::FullDeploy(args) => {
                full_deploy_command(args).await.map_err(NodeCliError::from)
            }
            Commands::DeployAndWait(args) => deploy_and_wait_command(args)
                .await
                .map_err(NodeCliError::from),
            Commands::IsFinalized(args) => {
                is_finalized_command(args).await.map_err(NodeCliError::from)
            }
            Commands::ExploratoryDeploy(args) => exploratory_deploy_command(args)
                .await
                .map_err(NodeCliError::from),
            Commands::GeneratePublicKey(args) => {
                generate_public_key_command(args).map_err(NodeCliError::from)
            }
            Commands::GenerateKeyPair(args) => {
                generate_key_pair_command(args).map_err(NodeCliError::from)
            }
            Commands::GenerateRevAddress(args) => {
                generate_rev_address_command(args).map_err(NodeCliError::from)
            }
            Commands::Status(args) => status_command(args).await.map_err(NodeCliError::from),
            Commands::Blocks(args) => blocks_command(args).await.map_err(NodeCliError::from),
            Commands::Bonds(args) => bonds_command(args).await.map_err(NodeCliError::from),
            Commands::ActiveValidators(args) => active_validators_command(args)
                .await
                .map_err(NodeCliError::from),
            Commands::WalletBalance(args) => wallet_balance_command(args)
                .await
                .map_err(NodeCliError::from),
            Commands::BondStatus(args) => {
                bond_status_command(args).await.map_err(NodeCliError::from)
            }
            Commands::Metrics(args) => metrics_command(args).await.map_err(NodeCliError::from),
            Commands::BondValidator(args) => bond_validator_command(args)
                .await
                .map_err(NodeCliError::from),
            Commands::NetworkHealth(args) => network_health_command(args)
                .await
                .map_err(NodeCliError::from),
            Commands::LastFinalizedBlock(args) => last_finalized_block_command(args)
                .await
                .map_err(NodeCliError::from),
            Commands::ShowMainChain(args) => show_main_chain_command(args)
                .await
                .map_err(NodeCliError::from),
            Commands::Transfer(args) => transfer_command(args).await.map_err(NodeCliError::from),
            Commands::GetDeploy(args) => get_deploy_command(args).await.map_err(NodeCliError::from),
            Commands::EpochInfo(args) => epoch_info_command(args).await.map_err(NodeCliError::from),
            Commands::ValidatorStatus(args) => validator_status_command(args)
                .await
                .map_err(NodeCliError::from),
            Commands::EpochRewards(args) => epoch_rewards_command(args)
                .await
                .map_err(NodeCliError::from),
            Commands::NetworkConsensus(args) => network_consensus_command(args)
                .await
                .map_err(NodeCliError::from),
        };

        // Handle errors with better formatting
        if let Err(e) = result {
            Self::handle_error(&e);
            return Err(e);
        }

        Ok(())
    }

    /// Handle errors with appropriate formatting and user-friendly messages
    fn handle_error(error: &NodeCliError) {
        match error {
            NodeCliError::Network(net_err) => {
                print_error(&format!("Network issue: {}", net_err));
                eprintln!("💡 Suggestion: Check your internet connection and node availability");
            }
            NodeCliError::Crypto(crypto_err) => {
                print_error(&format!("Cryptographic issue: {}", crypto_err));
                eprintln!("💡 Suggestion: Verify your private/public key format and validity");
            }
            NodeCliError::File(file_err) => {
                print_error(&format!("File operation failed: {}", file_err));
                eprintln!("💡 Suggestion: Check file permissions and paths");
            }
            NodeCliError::Api(api_err) => {
                print_error(&format!("API communication failed: {}", api_err));
                eprintln!(
                    "💡 Suggestion: Verify the node is running and API endpoints are accessible"
                );
            }
            NodeCliError::Config(config_err) => {
                print_error(&format!("Configuration issue: {}", config_err));
                eprintln!("💡 Suggestion: Check your command arguments and configuration");
            }
            NodeCliError::General(msg) => {
                print_error(msg);
            }
        }
    }

    /// Get the command name for logging purposes
    pub fn get_command_name(cli: &Cli) -> &'static str {
        match &cli.command {
            Commands::Deploy(_) => "deploy",
            Commands::Propose(_) => "propose",
            Commands::FullDeploy(_) => "full-deploy",
            Commands::DeployAndWait(_) => "deploy-and-wait",
            Commands::IsFinalized(_) => "is-finalized",
            Commands::ExploratoryDeploy(_) => "exploratory-deploy",
            Commands::GeneratePublicKey(_) => "generate-public-key",
            Commands::GenerateKeyPair(_) => "generate-key-pair",
            Commands::GenerateRevAddress(_) => "generate-rev-address",
            Commands::Status(_) => "status",
            Commands::Blocks(_) => "blocks",
            Commands::Bonds(_) => "bonds",
            Commands::ActiveValidators(_) => "active-validators",
            Commands::WalletBalance(_) => "wallet-balance",
            Commands::BondStatus(_) => "bond-status",
            Commands::Metrics(_) => "metrics",
            Commands::BondValidator(_) => "bond-validator",
            Commands::NetworkHealth(_) => "network-health",
            Commands::LastFinalizedBlock(_) => "last-finalized-block",
            Commands::ShowMainChain(_) => "show-main-chain",
            Commands::Transfer(_) => "transfer",
            Commands::GetDeploy(_) => "get-deploy",
            Commands::EpochInfo(_) => "epoch-info",
            Commands::ValidatorStatus(_) => "validator-status",
            Commands::EpochRewards(_) => "epoch-rewards",
            Commands::NetworkConsensus(_) => "network-consensus",
        }
    }
}
