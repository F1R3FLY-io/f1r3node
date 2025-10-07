use crate::rust::effects::console_io::console_io;
use crate::rust::{configuration::NodeConf, effects::console_io::decrypt_key_from_file};
use eyre::Result;
use tokio::net::lookup_host;
use tracing::{error, info};

/// Check host configuration (equivalent to Scala's checkHost)
pub async fn check_host(conf: &NodeConf) -> Result<()> {
    if let Some(host) = &conf.protocol_server.host {
        let is_valid = if conf.protocol_server.allow_private_addresses {
            is_valid_inet_address(host)
        } else {
            is_valid_public_inet_address(host).await
        };

        if !is_valid {
            // TODO: change error text to more relevant, Rust version will not use a Kademlia
            error!(
                "Kademlia hostname '{}' is not valid or it does not resolve to a public IP address",
                host
            );
            error!("Hint: Run me with --allow-private-addresses in private networks");
            return Err(eyre::eyre!("Invalid Kademlia hostname"));
        }
    }
    Ok(())
}

/// Check ports configuration (equivalent to Scala's checkPorts)
pub async fn check_ports(conf: &NodeConf) -> Result<NodeConf> {
    let mut updated_conf = conf.clone();

    // Check if ports are available
    let ports_to_check = vec![
        ("http", conf.api_server.port_http),
        ("admin http", conf.api_server.port_admin_http),
        ("grpc server external", conf.api_server.port_grpc_external),
        ("grpc server internal", conf.api_server.port_grpc_internal),
    ];

    let mut unavailable_ports = Vec::new();

    for (name, port) in ports_to_check {
        if !is_local_port_available(port) {
            unavailable_ports.push(name);
        }
    }

    if !unavailable_ports.is_empty() {
        return Err(eyre::eyre!(
            "Required ports are already in use: {}",
            unavailable_ports.join(", ")
        ));
    }

    // Check F1r3fly Protocol port
    if !is_local_port_available(conf.protocol_server.port) {
        if conf.protocol_server.use_random_ports {
            let free_port = get_free_port().await?;
            info!("Using random port {} as F1r3fly Protocol port", free_port);
            updated_conf.protocol_server.port = free_port;
        } else {
            error!("Hint: Run me with --use-random-ports to use a random port for F1r3fly Protocol port");
            return Err(eyre::eyre!("Invalid F1r3fly Protocol port"));
        }
    }

    // TODO: mostly this logic would stay the same, but renaming from Kademlia would be required as it is not used in Rust
    // Currently left the naming for compatibility with Scala version
    if !is_local_port_available(conf.peers_discovery.port) {
        if conf.protocol_server.use_random_ports {
            let free_port = get_free_port().await?;
            info!("Using random port {} as Kademlia port", free_port);
            updated_conf.peers_discovery.port = free_port;
        } else {
            error!("Hint: Run me with --use-random-ports to use a random port for Kademlia port");
            return Err(eyre::eyre!("Invalid Kademlia port"));
        }
    }

    Ok(updated_conf)
}

/// Load private key from file
pub async fn load_private_key_from_file(conf: NodeConf) -> Result<NodeConf> {
    if let Some(private_key_path) = &conf.casper.validator_private_key_path {
        let private_key_path = private_key_path.clone();

        let private_key = tokio::task::spawn_blocking(move || {
            let mut console = console_io()?;
            decrypt_key_from_file(&private_key_path, &mut console)
        })
        .await??;

        let private_key_base16 = hex::encode_upper(&private_key.bytes);

        let mut updated_conf = conf;
        updated_conf.casper.validator_private_key = Some(private_key_base16);
        Ok(updated_conf)
    } else {
        Ok(conf)
    }
}

/// Check if an address is a valid inet address
fn is_valid_inet_address(host: &str) -> bool {
    host.parse::<std::net::IpAddr>()
        .map(|addr| addr.is_unspecified())
        .unwrap_or(false)
}

/// Check if an address is a valid public inet address
async fn is_valid_public_inet_address(host: &str) -> bool {
    let is_valid_addr_check = |ip| match ip {
        std::net::IpAddr::V4(ipv4) => {
            !ipv4.is_private()
                && !ipv4.is_loopback()
                && !ipv4.is_link_local()
                && !ipv4.is_multicast()
        }
        std::net::IpAddr::V6(ipv6) => !ipv6.is_loopback() && !ipv6.is_unspecified(),
    };

    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        is_valid_addr_check(ip)
    } else {
        if let Ok(mut addrs) = lookup_host(host).await {
            addrs
                .next()
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::NotFound, "No addresses found")
                })
                .map(|socket_addr| is_valid_addr_check(socket_addr.ip()))
                .unwrap_or(false)
        } else {
            false
        }
    }
}

/// Check if a local port is available
fn is_local_port_available(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_ok()
}

/// Get a free port
async fn get_free_port() -> Result<u16> {
    use std::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    Ok(addr.port())
}
