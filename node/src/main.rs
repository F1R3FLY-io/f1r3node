use casper::rust::util::comm::deploy_runtime::DeployRuntime;
use casper::rust::util::comm::grpc_deploy_service::GrpcDeployService;
use casper::rust::util::comm::grpc_propose_service::GrpcProposeService;
use clap::{CommandFactory, Parser};
use crypto::rust::{
    private_key::PrivateKey, signatures::secp256k1::Secp256k1,
    signatures::signatures_alg::SignaturesAlg, util::key_util::KeyUtil,
};
use eyre::Result;
use node::rust::configuration::commandline::options::{GRPC_EXTERNAL_PORT, GRPC_INTERNAL_PORT};
use node::rust::configuration::config_check::{
    check_host, check_ports, load_private_key_from_file,
};
use node::rust::configuration::{
    commandline::options::OptionsSubCommand, KamonConf, NodeConf, Options, Profile,
};
use node::rust::effects::console_io::{console_io, decrypt_key_from_file};
use node::rust::effects::repl_client::GrpcReplClient;
use node::rust::repl::ReplRuntime;
use node::rust::web::version_info::get_version_info_str;
use std::path::PathBuf;
use tokio::runtime::{Builder, Runtime};
use tracing::{info, warn};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    init_json_logging()?;

    // Parse CLI arguments
    let options = Options::try_parse()?;

    // Determine if we should start the node or run CLI commands
    if options
        .subcommand
        .as_ref()
        .is_some_and(|subcommand| matches!(subcommand, OptionsSubCommand::Run(_)))
    {
        // Start the node
        let rt = Builder::new_multi_thread().enable_all().build()?;
        rt.block_on(async {
            // Execute CLI command
            start_node(options).await?;
            Ok::<_, eyre::Error>(())
        })?;
    } else {
        // we should not bother about blocking calls in this case since we are expecting consecutive execution
        let rt = Builder::new_current_thread().enable_all().build()?;
        run_cli(options, &rt)?;
    }

    Ok(())
}

/// Starts the F1r3fly node instance
async fn start_node(options: Options) -> Result<()> {
    // Create merged configuration from CLI options and config file
    let default_dir = std::env::var("DEFAULT_DIR")
        .and_then(|path| Ok(PathBuf::from(path)))
        .or_else(|_| std::env::current_dir().map(|path| path.join("node/src/main/resources")))?;

    let (node_conf, profile, config_file, kamon_conf) =
        node::rust::configuration::builder::build(&default_dir, options)?;

    // Set system property for data directory (equivalent to Scala's System.setProperty)
    std::env::set_var(
        "RNODE_DATA_DIR",
        node_conf.storage.data_dir.to_string_lossy().to_string(),
    );

    // Start the node with configuration validation and setup
    check_host(&node_conf).await?;
    let conf_with_ports = check_ports(&node_conf).await?;
    let conf_with_decrypt = load_private_key_from_file(conf_with_ports).await?;
    info!("{}", get_version_info_str());
    log_configuration(&conf_with_decrypt, &profile, config_file.as_ref()).await?;

    // Create and start node runtime
    start_node_runtime(conf_with_decrypt, kamon_conf).await?;

    Ok(())
}

/// Executes CLI commands
fn run_cli(options: Options, rt: &Runtime) -> Result<()> {
    let (grpc_port, grpc_deploy_port) = if let Some(port) = options.grpc_port {
        (port, port)
    } else {
        (GRPC_INTERNAL_PORT, GRPC_EXTERNAL_PORT)
    };

    let (repl_client, mut deploy_client, propose_client) = rt.block_on(async {
        let repl_client = GrpcReplClient::new(
            options.grpc_host.clone(),
            grpc_port,
            options.grpc_max_recv_message_size as usize,
        )
        .await
        .map_err(|e| eyre::eyre!("Failed to create REPL client: {}", e))?;

        let deploy_client = GrpcDeployService::new(
            &options.grpc_host,
            grpc_deploy_port,
            options.grpc_max_recv_message_size as usize,
        )
        .await?;

        let propose_client = GrpcProposeService::new(
            &options.grpc_host,
            grpc_port,
            options.grpc_max_recv_message_size as usize,
        )
        .await?;

        eyre::Ok((repl_client, deploy_client, propose_client))
    })?;

    match options.subcommand {
        Some(command) => match command {
            OptionsSubCommand::Eval {
                files,
                print_unmatched_sends_only,
                language,
            } => {
                ReplRuntime::new().eval_program(
                    &rt,
                    &mut console_io()?,
                    &repl_client,
                    files,
                    print_unmatched_sends_only,
                    language,
                )?;

                Ok::<(), eyre::Error>(())
            }
            OptionsSubCommand::Repl => {
                ReplRuntime::new().repl_program(&rt, &mut console_io()?, &repl_client)?;

                Ok(())
            }
            OptionsSubCommand::Deploy {
                phlo_limit,
                phlo_price,
                valid_after_block,
                private_key,
                private_key_path,
                location,
                shard_id,
            } => {
                let private_key =
                    get_private_key(private_key, private_key_path, &mut console_io()?)?;
                rt.block_on(DeployRuntime::deploy_file_program(
                    &mut deploy_client,
                    phlo_limit,
                    phlo_price,
                    valid_after_block,
                    &private_key,
                    &location,
                    &shard_id,
                ));
                Ok(())
            }
            OptionsSubCommand::FindDeploy { id } => {
                rt.block_on(DeployRuntime::find_deploy(&mut deploy_client, &id));
                Ok(())
            }
            OptionsSubCommand::Propose {
                print_unmatched_sends,
            } => {
                rt.block_on(DeployRuntime::propose(
                    propose_client,
                    print_unmatched_sends,
                ));
                Ok(())
            }
            OptionsSubCommand::ShowBlock { hash } => {
                rt.block_on(DeployRuntime::get_block(&mut deploy_client, hash));
                Ok(())
            }
            OptionsSubCommand::ShowBlocks { depth } => {
                rt.block_on(DeployRuntime::get_blocks(&mut deploy_client, depth));
                Ok(())
            }
            OptionsSubCommand::VisualizeDag {
                depth,
                show_justification_lines,
            } => {
                rt.block_on(DeployRuntime::visualize_dag(
                    &mut deploy_client,
                    depth,
                    show_justification_lines,
                ));
                Ok(())
            }
            OptionsSubCommand::MachineVerifiableDag => {
                rt.block_on(DeployRuntime::machine_verifiable_dag(&mut deploy_client));
                Ok(())
            }
            OptionsSubCommand::Keygen { path } => {
                generate_key(&path, &mut console_io()?)?;
                Ok(())
            }
            OptionsSubCommand::LastFinalizedBlock => {
                rt.block_on(DeployRuntime::last_finalized_block(&mut deploy_client));
                Ok(())
            }
            OptionsSubCommand::IsFinalized { hash } => {
                rt.block_on(DeployRuntime::is_finalized(&mut deploy_client, hash));
                Ok(())
            }
            OptionsSubCommand::BondStatus { public_key } => {
                rt.block_on(DeployRuntime::bond_status(&mut deploy_client, &public_key));
                Ok(())
            }
            OptionsSubCommand::DataAtName { name } => {
                rt.block_on(DeployRuntime::listen_for_data_at_name(
                    &mut deploy_client,
                    name,
                ));
                Ok(())
            }
            OptionsSubCommand::ContAtName { names } => {
                rt.block_on(DeployRuntime::listen_for_continuation_at_name(
                    &mut deploy_client,
                    names,
                ));
                Ok(())
            }
            OptionsSubCommand::Status => {
                rt.block_on(DeployRuntime::status(&mut deploy_client));
                Ok(())
            }
            _ => Ok(()),
        },
        None => {
            Options::command().print_help()?;
            Ok(())
        }
    }?;

    Ok(())
}

pub fn init_json_logging() -> eyre::Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .json()
                .with_target(true)
                .with_file(true)
                .with_line_number(true)
                .with_current_span(false) // logs only
                .with_span_list(false) // logs only
                .flatten_event(true), // put event fields at top level
        )
        .try_init()?;
    Ok(())
}

/// Generate a new key pair and save to file (equivalent to generateKey)
fn generate_key(
    path: &PathBuf,
    console_io: &mut impl node::rust::effects::console_io::ConsoleIO,
) -> Result<()> {
    let password = console_io.read_password("Enter password for keyfile: ")?;
    let password_repeat = console_io.read_password("Repeat password: ")?;

    if password != password_repeat {
        console_io.println_str("Passwords do not match. Try again:")?;
        return generate_key(path, console_io);
    }

    if password.is_empty() {
        console_io.println_str("Password is empty. Try again:")?;
        return generate_key(path, console_io);
    }

    let secp256k1 = Secp256k1;
    let (private_key, public_key) = <Secp256k1 as SignaturesAlg>::new_key_pair(&secp256k1);

    let private_pem_key_path = path.join("rnode.key");
    let public_pem_key_path = path.join("rnode.pub.pem");
    let public_key_hex_file = path.join("rnode.pub.hex");

    KeyUtil::write_keys(
        &private_key,
        &public_key,
        Box::new(Secp256k1),
        &password,
        &private_pem_key_path,
        &public_pem_key_path,
        &public_key_hex_file,
    )?;

    console_io.println_str(&format!(
        "\nSuccess!\n\
         Private key file (encrypted PEM format):  {}\n\
         Public  key file (PEM format):            {}\n\
         Public  key file (HEX format):            {}",
        private_pem_key_path.canonicalize()?.display(),
        public_pem_key_path.canonicalize()?.display(),
        public_key_hex_file.canonicalize()?.display()
    ))?;

    Ok(())
}

/// Get private key from either direct key or file path (equivalent to Scala's getPrivateKey)
fn get_private_key(
    maybe_private_key: Option<PrivateKey>,
    maybe_private_key_path: Option<PathBuf>,
    console_io: &mut impl node::rust::effects::console_io::ConsoleIO,
) -> Result<PrivateKey> {
    match maybe_private_key {
        Some(key) => Ok(key),
        None => match maybe_private_key_path {
            Some(path) => decrypt_key_from_file(&path, console_io),
            None => Err(eyre::eyre!("Private key is missing")),
        },
    }
}

/// Start node runtime (equivalent to Scala's NodeRuntime.start)
async fn start_node_runtime(conf: NodeConf, kamon_conf: KamonConf) -> Result<()> {
    // --- Observability Setup ---
    #[allow(unused_variables)]
    let prometheus_reporter = node::rust::diagnostics::initialize_diagnostics(&conf, &kamon_conf)?;

    // Initialize the logging and tracing subscriber. This will collect trace spans
    // and events and log them to the console.
    tracing_subscriber::fmt::init();

    // TODO: Create AppState with prometheus_reporter and other dependencies
    // TODO: Initialize HTTP servers with Routes::create_main_routes() and Routes::create_admin_routes()
    // See Scala implementation in ServersInstances.scala for reference
    // Pass prometheus_reporter (Option<Arc<NewPrometheusReporter>>) to AppState::new()
    
    info!("Node runtime started successfully");
    Ok(())
}

/// Log configuration (equivalent to Scala's logConfiguration)
async fn log_configuration(
    conf: &NodeConf,
    profile: &Profile,
    config_file: Option<&PathBuf>,
) -> Result<()> {
    info!("Starting with profile {}", profile.name);

    if let Some(config_file) = config_file {
        info!(
            "Using configuration file: {}",
            config_file.canonicalize()?.display()
        );
    } else {
        warn!("No configuration file found, using defaults");
    }

    info!("Running on network: {}", conf.protocol_server.network_id);

    Ok(())
}
