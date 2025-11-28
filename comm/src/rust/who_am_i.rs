// See comm/src/main/scala/coop/rchain/comm/WhoAmI.scala

use crate::rust::errors::CommError;
use crate::rust::peer_node::{NodeIdentifier, PeerNode};
use crate::rust::upnp;

/// Fetch local peer node with optional UPnP port forwarding
///
/// Ports the Scala `fetchLocalPeerNode` function.
pub async fn fetch_local_peer_node(
    host: Option<String>,
    protocol_port: u16,
    discovery_port: u16,
    no_upnp: bool,
    id: NodeIdentifier,
) -> Result<PeerNode, CommError> {
    let external_address =
        retrieve_external_address(no_upnp, &[protocol_port, discovery_port]).await?;
    let host_str = fetch_host(host, external_address).await?;

    Ok(PeerNode::new(id, host_str, protocol_port, discovery_port))
}

/// Check if local peer node's external IP has changed
///
/// Ports the Scala `checkLocalPeerNode` function.
pub async fn check_local_peer_node(
    protocol_port: u16,
    discovery_port: u16,
    peer_node: &PeerNode,
) -> Result<Option<PeerNode>, CommError> {
    let (_, current_ip) = check_all(None).await?;

    if current_ip == peer_node.endpoint.host {
        Ok(None)
    } else {
        tracing::info!("external IP address has changed to {}", current_ip);
        Ok(Some(PeerNode::new(
            peer_node.id.clone(),
            current_ip,
            protocol_port,
            discovery_port,
        )))
    }
}

async fn fetch_host(
    host: Option<String>,
    external_address: Option<String>,
) -> Result<String, CommError> {
    match host {
        Some(h) => Ok(h),
        None => who_am_i(external_address).await,
    }
}

async fn retrieve_external_address(
    no_upnp: bool,
    ports: &[u16],
) -> Result<Option<String>, CommError> {
    if no_upnp {
        Ok(None)
    } else {
        upnp::assure_port_forwarding(ports).await
    }
}

pub async fn check_from(url: &str) -> Result<Option<String>, CommError> {
    let response = reqwest::get(url)
        .await
        .map_err(|e| CommError::UnknownCommError(e.to_string()))?;

    let text = response
        .text()
        .await
        .map_err(|e| CommError::UnknownCommError(e.to_string()))?;

    let hostname = text.trim();
    // Resolve hostname to IP
    let addrs = tokio::net::lookup_host(hostname)
        .await
        .map_err(|e| CommError::UnknownCommError(format!("Failed to resolve hostname: {}", e)))?;

    let ip_addr = addrs.into_iter().next().map(|addr| addr.ip().to_string());

    Ok(ip_addr)
}

async fn check_next(
    prev: (String, Option<String>),
    next: impl std::future::Future<Output = (String, Option<String>)>,
) -> (String, Option<String>) {
    match prev.1 {
        Some(_) => prev,
        None => next.await,
    }
}

async fn upnp_ip_check(external_address: Option<String>) -> (String, Option<String>) {
    let ip = match external_address {
        Some(addr) => {
            // Resolve hostname to IP address (ports InetAddress.getByName().getHostAddress)
            tokio::net::lookup_host(&addr)
                .await
                .ok()
                .and_then(|addrs| addrs.into_iter().next())
                .map(|addr| addr.ip().to_string())
        }
        None => None,
    };
    ("UPnP".to_string(), ip)
}

async fn check_all(external_address: Option<String>) -> Result<(String, String), CommError> {
    let r1 = (
        "AmazonAWS service".to_string(),
        check_from("http://checkip.amazonaws.com")
            .await
            .ok()
            .flatten(),
    );
    let r2 = check_next(r1, async {
        (
            "WhatIsMyIP service".to_string(),
            check_from("http://bot.whatismyipaddress.com")
                .await
                .ok()
                .flatten(),
        )
    })
    .await;
    let r3 = check_next(r2, async { upnp_ip_check(external_address).await }).await;
    let r4 = check_next(r3, async {
        ("failed to guess".to_string(), Some("localhost".to_string()))
    })
    .await;

    match r4.1 {
        Some(ip) => Ok((r4.0, ip)),
        None => Err(CommError::UnknownCommError(
            "Failed to determine external IP address from any source".to_string(),
        )),
    }
}

async fn who_am_i(external_address: Option<String>) -> Result<String, CommError> {
    tracing::info!("flag --host was not provided, guessing your external IP address");

    let (source, ip) = check_all(external_address).await?;

    tracing::info!("guessed {} from source: {}", ip, source);

    Ok(ip)
}
