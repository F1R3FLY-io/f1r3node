// See comm/src/test/scala/coop/rchain/comm/WhoAmISpec.scala

use std::sync::Once;

use comm::rust::peer_node::NodeIdentifier;
use comm::rust::who_am_i::fetch_local_peer_node;

static INIT: Once = Once::new();

fn init_logger() {
    INIT.call_once(|| {
        env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Info)
            .try_init()
            .unwrap();
    });
}

/// Test fetching a peer node with default values from node/src/main/resources/defaults.conf
#[tokio::test]
#[ignore]
async fn test_fetch_local_peer_node_with_defaults() {
    init_logger();

    // Default values from defaults.conf
    let protocol_port: u16 = 40400;
    let discovery_port: u16 = 40404;
    let no_upnp = false;
    let host: Option<String> = None;

    // Create a test node identifier (using a sample hex string)
    // In real usage, this would be derived from a TLS certificate
    let test_node_id = "de6eed5d00cf080fc587eeb412cb31a75fd10358";
    let node_identifier = NodeIdentifier::new(test_node_id.to_string());

    // Fetch the peer node
    let result = fetch_local_peer_node(
        host,
        protocol_port,
        discovery_port,
        no_upnp,
        node_identifier,
    )
    .await;

    match result {
        Ok(peer_node) => {
            assert_eq!(
                peer_node.endpoint.tcp_port, protocol_port as u32,
                "TCP port should match protocol_port"
            );
            assert_eq!(
                peer_node.endpoint.udp_port, discovery_port as u32,
                "UDP port should match discovery_port"
            );

            assert_eq!(
                peer_node.id.to_string(),
                test_node_id,
                "Node identifier should match"
            );

            assert!(
                !peer_node.endpoint.host.is_empty(),
                "Host should not be empty"
            );

            log::info!(
                "Successfully fetched peer node: id={}, host={}, tcp_port={}, udp_port={}",
                peer_node.id.to_string(),
                peer_node.endpoint.host,
                peer_node.endpoint.tcp_port,
                peer_node.endpoint.udp_port
            );
        }
        Err(e) => {
            // If UPnP fails or network is unavailable, that's acceptable for a test
            // But we should log it for debugging
            log::warn!(
                "Failed to fetch peer node (this may be expected if UPnP is unavailable or network is down): {}",
                e
            );
            panic!("Failed to fetch peer node: {}", e);
        }
    }
}

/// Test fetching a peer node with explicit host
#[tokio::test]
#[ignore]
async fn test_fetch_local_peer_node_with_host() {
    init_logger();

    let protocol_port: u16 = 40400;
    let discovery_port: u16 = 40404;
    let no_upnp = true;
    let host: Option<String> = Some("localhost".to_string());

    let test_node_id = "de6eed5d00cf080fc587eeb412cb31a75fd10358";
    let node_identifier = NodeIdentifier::new(test_node_id.to_string());

    let result = fetch_local_peer_node(
        host.clone(),
        protocol_port,
        discovery_port,
        no_upnp,
        node_identifier,
    )
    .await;

    match result {
        Ok(peer_node) => {
            assert_eq!(
                peer_node.endpoint.host,
                host.unwrap(),
                "Host should match the provided host"
            );
            assert_eq!(
                peer_node.endpoint.tcp_port, protocol_port as u32,
                "TCP port should match"
            );
            assert_eq!(
                peer_node.endpoint.udp_port, discovery_port as u32,
                "UDP port should match"
            );

            log::info!(
                "Successfully fetched peer node with explicit host: {}",
                peer_node.endpoint.host
            );
        }
        Err(e) => {
            panic!("Failed to fetch peer node with explicit host: {}", e);
        }
    }
}

/// Test fetching a peer node with UPnP disabled

#[tokio::test]
#[ignore]
async fn test_fetch_local_peer_node_no_upnp() {
    init_logger();

    let protocol_port: u16 = 40400;
    let discovery_port: u16 = 40404;
    let no_upnp = true; // Disable UPnP
    let host: Option<String> = None;

    let test_node_id = "de6eed5d00cf080fc587eeb412cb31a75fd10358";
    let node_identifier = NodeIdentifier::new(test_node_id.to_string());

    let result = fetch_local_peer_node(
        host,
        protocol_port,
        discovery_port,
        no_upnp,
        node_identifier,
    )
    .await;

    match result {
        Ok(peer_node) => {
            assert_eq!(
                peer_node.endpoint.tcp_port, protocol_port as u32,
                "TCP port should match"
            );
            assert_eq!(
                peer_node.endpoint.udp_port, discovery_port as u32,
                "UDP port should match"
            );

            assert!(
                !peer_node.endpoint.host.is_empty(),
                "Host should be determined from external IP services"
            );

            log::info!(
                "Successfully fetched peer node without UPnP: host={}",
                peer_node.endpoint.host
            );
        }
        Err(e) => {
            log::warn!(
                "Failed to fetch peer node without UPnP (may be expected if external IP services are down): {}",
                e
            );
            panic!("Failed to fetch peer node  without UPnP : {}", e);
        }
    }
}
