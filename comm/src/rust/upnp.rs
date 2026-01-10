// See comm/src/main/scala/coop/rchain/comm/UPnP.scala
//
// NOTE: This implementation needs to be refined based on the actual igd 0.12.1 API.
// The methods used here may need adjustment once the exact API is confirmed.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use futures::future::try_join_all;
use igd_next::aio::tokio::{search_gateway, Tokio};
use igd_next::aio::Gateway;
use igd_next::{GetGenericPortMappingEntryError, PortMappingEntry, PortMappingProtocol};
use local_ip_address::local_ip;
use regex::Regex;

use crate::rust::errors::CommError;

static IPV4_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$")
        .expect("Failed to compile IPV4_PATTERN regex")
});

fn is_private_ip_address(ip: &str) -> Option<bool> {
    if !IPV4_PATTERN.is_match(ip) {
        return None;
    }

    let parts: Vec<&str> = ip.split('.').collect();
    if parts.len() != 4 {
        return None;
    }

    let nums: Option<Vec<u8>> = parts.iter().map(|s| s.parse::<u8>().ok()).collect();

    let nums = nums?;
    let (p1, p2, p3, p4) = (nums[0], nums[1], nums[2], nums[3]);

    Some(match (p1, p2, p3, p4) {
        (10, _, _, _) => true,
        (127, _, _, _) => true,
        (192, 168, _, _) => true,
        (172, p, _, _) if p > 15 && p < 32 => true,
        (169, 254, _, _) => true,
        (0, 0, 0, 0) => true,
        _ => false,
    })
}

fn find_local_ip() -> Option<std::net::Ipv4Addr> {
    use std::net::UdpSocket;

    // Connect to a remote address to determine local interface
    // Using Google DNS as it's reliably reachable
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;

    match socket.local_addr().ok()?.ip() {
        std::net::IpAddr::V4(ipv4) if !ipv4.is_loopback() => Some(ipv4),
        _ => None,
    }
}

/// Get the local address to connect to the gateway through
///
/// Ports the Java `getLocalAddress()` method.
pub fn get_local_address() -> Result<std::net::Ipv4Addr, CommError> {
    find_local_ip().ok_or_else(|| {
        CommError::UnknownCommError(
            "Failed to determine local IP address. Cannot determine which network interface to use for gateway communication.".to_string(),
        )
    })
}

/// UPnP devices discovered on the network
#[derive(Debug, Clone)]
pub struct UPnPDevices {
    /// All devices mapped by their IP address
    pub all: HashMap<IpAddr, Arc<Gateway<Tokio>>>,
    /// Sequence of gateway devices found
    pub gateways: Vec<Arc<Gateway<Tokio>>>,
    /// Optional valid gateway (preferred gateway to use)
    pub valid_gateway: Option<Arc<Gateway<Tokio>>>,
}

impl UPnPDevices {
    /// Create a new `UPnPDevices` instance
    pub fn new(
        all: HashMap<IpAddr, Arc<Gateway<Tokio>>>,
        gateways: Vec<Arc<Gateway<Tokio>>>,
        valid_gateway: Option<Arc<Gateway<Tokio>>>,
    ) -> Self {
        Self {
            all,
            gateways,
            valid_gateway,
        }
    }

    /// Create an empty `UPnPDevices` instance (no devices found)
    pub fn empty() -> Self {
        Self {
            all: HashMap::new(),
            gateways: Vec::new(),
            valid_gateway: None,
        }
    }
}

/// Show device information in a formatted string
/// Ports the Scala `showDevice` function
///
/// NOTE: The igd-next crate has a simpler API than the Java GatewayDevice.
/// We format what information is available. Additional fields may require
/// accessing the gateway's internal state or parsing the UPnP response.
async fn show_device(ip: &IpAddr, gateway: &Gateway<Tokio>) -> Result<String, CommError> {
    // Get external IP (may fail, so we handle it)
    // Note: get_external_ip() returns a Future, so we need to handle it in async context
    let external_ip = gateway
        .get_external_ip()
        .await
        .map(|ip| ip.to_string())
        .map_err(|e| CommError::UnknownCommError(format!("Failed to get external IP: {}", e)))?;

    // The igd-next crate has limited device info compared to Java GatewayDevice
    // In the Java version, we had: Interface, Name, Model, Manufacturer, Description,
    // Type, Search type, Service type, Location, External IP, Connected status
    // Here we only show what's readily available from the igd API
    Ok(format!(
        "\n\
        |Interface:    {}\n\
        |External IP:  {}\n\
        |",
        ip, external_ip
    ))
}

/// Print all discovered devices
/// Ports the Scala `printDevices` function
async fn print_devices(devices: &UPnPDevices) -> Result<(), CommError> {
    let devices_str = try_join_all(
        devices
            .all
            .iter()
            .map(|(ip, gateway)| show_device(ip, gateway)),
    )
    .await?
    .join("\n");

    tracing::info!("Other devices:{}", devices_str);

    Ok(())
}

/// Log when no gateway devices are found
pub async fn log_gateway_empty(devices: &UPnPDevices) -> Result<(), CommError> {
    tracing::info!("INFO - No gateway devices found");

    if devices.all.is_empty() {
        tracing::info!("No need to open any port");
    } else {
        print_devices(devices).await?;
    }

    Ok(())
}

async fn remove_ports(
    device: &Gateway<Tokio>,
    ports: &[PortMappingEntry],
) -> Result<(), CommError> {
    for port in ports {
        let res = remove_port(device, port).await;

        let res_msg = if res.is_ok() { "[success]" } else { "[failed]" };

        tracing::info!(
            "Removing an existing port mapping for port {}/{} {}",
            port.protocol,
            port.external_port,
            res_msg
        );
    }

    Ok(())
}

async fn remove_port(device: &Gateway<Tokio>, port: &PortMappingEntry) -> Result<(), CommError> {
    device
        .remove_port(port.protocol, port.external_port)
        .await
        .map_err(|e| CommError::UnknownCommError(format!("Failed to remove port: {}", e)))?;

    Ok(())
}

async fn get_port_mappings(device: &Gateway<Tokio>) -> Vec<PortMappingEntry> {
    let mut mappings = Vec::new();
    let mut index = 0u32;

    loop {
        match device.get_generic_port_mapping_entry(index).await {
            Ok(entry) => {
                mappings.push(entry);
                index += 1;
            }
            Err(e) => {
                // Stop on index out of bounds (no more entries)
                // This matches the Scala behavior where getGenericPortMappingEntry
                // returns false when index is out of bounds
                match e {
                    GetGenericPortMappingEntryError::SpecifiedArrayIndexInvalid => {
                        // Index out of bounds means we've reached the end
                        break;
                    }
                    _ => {
                        // For other errors, also stop (similar to Scala's behavior where
                        // getGenericPortMappingEntry returns false for any failure)
                        break;
                    }
                }
            }
        }
    }

    mappings
}

// TODO: Allow different external and internal ports. Left this comment for reference as same as in the Scala implementation
async fn add_port(
    device: &Gateway<Tokio>,
    port: u16,
    protocol: PortMappingProtocol,
    description: &str,
) -> Result<(), CommError> {
    let local_ip = local_ip()
        .map_err(|e| CommError::UnknownCommError(format!("Failed to get local address: {}", e)))?;

    // Convert IpAddr to SocketAddr by adding the port
    // (external_port == internal_port, matching the Scala implementation)
    let local_addr = SocketAddr::new(local_ip, port);

    device
        .add_port(protocol, port, local_addr, 0, description)
        .await
        .map_err(|e| CommError::UnknownCommError(e.to_string()))
}

async fn add_ports(
    device: &Gateway<Tokio>,
    ports: &[u16],
    protocol: PortMappingProtocol,
    description: &str,
) -> Vec<bool> {
    let mut results = Vec::new();
    for port in ports {
        let res = add_port(device, *port, protocol, description).await;

        let res_msg = if res.is_ok() { "[success]" } else { "[failed]" };

        tracing::info!(
            "Adding a port mapping for port {}/{} {}",
            protocol,
            port,
            res_msg
        );

        results.push(res.is_ok());
    }

    results
}

async fn try_open_ports(ports: &[u16], devices: &UPnPDevices) -> Result<Option<String>, CommError> {
    let res = devices
        .gateways
        .iter()
        .map(|gateway| format!("{}", *gateway))
        .collect::<Vec<String>>()
        .join(", ");

    tracing::info!("Available gateway devices: {}", res);

    let gateway = devices
        .valid_gateway
        .clone()
        .or_else(|| devices.gateways.first().cloned())
        .ok_or_else(|| CommError::UnknownCommError("No gateway available".to_string()))?;

    tracing::info!(
        "Picking gateway for port forwarding: {} as gateway",
        gateway
    );

    let external_ip = gateway
        .get_external_ip()
        .await
        .map_err(|e| CommError::UnknownCommError(format!("Failed to get external IP: {}", e)))?
        .to_string();

    match is_private_ip_address(&external_ip) {
        Some(true) => tracing::warn!("Gateway's external IP address {} is from a private address block. This machine is behind more than one NAT.", external_ip.to_string()),
        Some(_) => tracing::info!("Gateway's external IP address is from a public address block."),
        None => tracing::warn!("Can't parse gateway's external IP address. It's maybe IPv6."),
    }

    let mappings = get_port_mappings(&gateway).await;

    let relevant_mappings: Vec<_> = mappings
        .into_iter()
        .filter(|m| ports.contains(&m.external_port))
        .collect();

    remove_ports(&gateway, &relevant_mappings).await?;

    let res = add_ports(&gateway, ports, PortMappingProtocol::TCP, "F1r3fly").await;

    if res.iter().any(|&success| !success) {
        tracing::error!("Could not open the ports via UPnP. Please open it manually on your router!");
    } else {
        tracing::info!("UPnP port forwarding was most likely successful!");
    }

    tracing::info!("{}", show_port_mapping_header());

    let mappings = get_port_mappings(&gateway).await;

    for mapping in mappings {
        tracing::info!("{}", show_port_mapping(&mapping));
    }

    Ok(Some(external_ip))
}

pub async fn assure_port_forwarding(ports: &[u16]) -> Result<Option<String>, CommError> {
    tracing::info!("trying to open ports using UPnP....");

    let devices = discover().await?;

    if devices.gateways.is_empty() {
        log_gateway_empty(&devices).await?;
        Ok(None)
    } else {
        try_open_ports(ports, &devices).await
    }
}

/// Discover UPnP gateways on the network
///
/// Ports the Scala `discover` function.
///
/// # Limitations and Differences from Java GatewayDiscover
///
/// The Java `org.bitlet.weupnp.GatewayDiscover` class works as follows:
/// 1. Sends a single UPnP SSDP M-SEARCH broadcast request
/// 2. Listens for ALL responses during a timeout period (typically several seconds)
/// 3. Collects all unique gateway devices that respond (can be multiple)
/// 4. Provides methods to access:
///    - `discover()`: Returns a Map<InetAddress, GatewayDevice> of all discovered devices
///    - `getAllGateways()`: Returns all gateway devices as a collection
///    - `getValidGateway()`: Returns the first/primary gateway device
///
/// However, the Rust `igd-next` crate's `search_gateway()` function works differently:
/// - It sends a UPnP broadcast and listens for responses
/// - BUT it returns the FIRST valid gateway found and then immediately exits
/// - The internal implementation has a loop that collects responses, but it's designed
///   to return early upon finding the first valid gateway
/// - The discovery logic is not exposed as a public API for collecting multiple gateways
///
/// # Current Implementation Approach
///
/// Since `igd-next` doesn't expose the ability to collect multiple gateways from a
/// single broadcast (its internal loop exits after finding the first valid gateway),
/// we work around this limitation by:
///
/// 1. Performing multiple sequential searches (up to 3 attempts)
/// 2. Each search may discover the same or different gateways
/// 3. We deduplicate by IP address to avoid adding the same gateway multiple times
/// 4. The first gateway found becomes the "valid gateway"
///
/// # Why This Works (But Not Perfectly)
///
/// - In most home networks, there's only ONE gateway/router, so multiple searches
///   will find the same device
/// - In complex networks with multiple gateways, this approach might still only find
///   the first one if they respond quickly
/// - The timing of UPnP responses is not deterministic, so we may miss some gateways
///
/// # Alternative: Full Reimplementation (Not Implemented)
///
/// To truly match Java's GatewayDiscover behavior, we would need to:
/// 1. Manually send UPnP SSDP M-SEARCH broadcast packets
/// 2. Implement our own response collection loop that doesn't exit early
/// 3. Parse UPnP XML responses ourselves
/// 4. Build Gateway objects from the responses
/// 5. Continue listening for the full timeout period
///
/// This would require reimplementing significant parts of the UPnP discovery protocol
/// that `igd-next` already handles, but doesn't expose in the way we need.
///
/// For now, the sequential search approach is a pragmatic compromise that works
/// for the common case (single gateway) while being compatible with the `igd-next` API.
async fn discover() -> Result<UPnPDevices, CommError> {
    use igd_next::SearchOptions;

    let mut all: HashMap<IpAddr, Arc<Gateway<Tokio>>> = HashMap::new();
    let mut gateways: Vec<Arc<Gateway<Tokio>>> = Vec::new();
    let mut valid_gateway: Option<Arc<Gateway<Tokio>>> = None;

    let timeout = Duration::from_secs(3);

    // Workaround: Perform multiple sequential searches since igd-next's search_gateway
    // returns after finding the first gateway, not all gateways like Java's GatewayDiscover
    for attempt in 0..3 {
        let options = SearchOptions {
            timeout: Some(timeout),
            ..SearchOptions::default()
        };

        match search_gateway(options).await {
            Ok(gateway) => {
                let gateway_arc = Arc::new(gateway);

                // Use the gateway's IP address as the key for deduplication
                let gateway_ip = gateway_arc.addr.ip();

                // Only add if we haven't seen this gateway before
                if !all.contains_key(&gateway_ip) {
                    all.insert(gateway_ip, gateway_arc.clone());
                    gateways.push(gateway_arc.clone());

                    // First gateway found becomes the valid gateway
                    // (matches Java's getValidGateway() behavior)
                    if valid_gateway.is_none() {
                        valid_gateway = Some(gateway_arc.clone());
                    }
                }
            }
            Err(e) => {
                tracing::debug!("Gateway search attempt {} failed: {:?}", attempt + 1, e);
                // Continue searching even if one attempt fails
                // Only return empty if all attempts failed
                if attempt == 2 && gateways.is_empty() {
                    // No gateways found after all attempts
                    return Ok(UPnPDevices::empty());
                }
            }
        }

        // Small delay between searches to avoid overwhelming the network
        // This also gives time for different gateways to respond in subsequent searches
        if attempt < 2 {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    }

    Ok(UPnPDevices {
        all,
        gateways,
        valid_gateway,
    })
}

// Ports the Scala `showPortMappingHeader` function
fn show_port_mapping_header() -> String {
    let protocol = format!("{:<10}", "Protocol");
    let external_port = format!("{:<8}", "Extern");
    let internal_client = format!("{:<15}", "Host");
    let internal_port = format!("{:<8}", "Intern");
    let description = "Description";

    format!(
        "{} {} {} {} {}",
        protocol, external_port, internal_client, internal_port, description
    )
}

// Ports the Scala `showPortMapping` function
fn show_port_mapping(m: &PortMappingEntry) -> String {
    let protocol_str = match m.protocol {
        PortMappingProtocol::TCP => "TCP",
        PortMappingProtocol::UDP => "UDP",
    };

    let protocol = format!("{:<10}", protocol_str);
    let external_port = format!("{:<8}", m.external_port);

    let internal_client_str = if m.internal_client.is_empty() {
        "*"
    } else {
        &m.internal_client
    };

    let internal_client = format!("{:<15}", internal_client_str);

    let internal_port = format!("{:<8}", m.internal_port);

    // Use port_mapping_description field from PortMappingEntry
    let description = &m.port_mapping_description;

    format!(
        "{} {} {} {} {}",
        protocol, external_port, internal_client, internal_port, description
    )
}
