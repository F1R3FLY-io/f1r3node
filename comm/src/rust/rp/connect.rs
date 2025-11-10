// See comm/src/main/scala/coop/rchain/comm/rp/Connect.scala

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use log::{info, warn};
use rand::seq::SliceRandom;

use crate::rust::discovery::node_discovery::NodeDiscovery;
use crate::rust::transport::transport_layer::TransportLayer;
use crate::rust::{
    errors::CommError,
    metrics_constants::{CONNECT_METRIC, RP_CONNECT_METRICS_SOURCE},
    peer_node::PeerNode,
    rp::{protocol_helper, rp_conf::RPConf},
};

pub type Connection = PeerNode;

#[derive(Debug, Clone)]
pub struct Connections(pub Vec<Connection>);

impl Connections {
    pub fn empty() -> Self {
        Self(Vec::new())
    }

    pub fn from_vec(connections: Vec<Connection>) -> Self {
        Self(connections)
    }

    pub fn into_vec(self) -> Vec<Connection> {
        self.0
    }

    pub fn as_slice(&self) -> &[Connection] {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Connection> {
        self.0.iter()
    }

    pub fn take(&self, n: usize) -> Connections {
        Connections(self.0.iter().take(n).cloned().collect())
    }

    pub fn to_set(&self) -> HashSet<PeerNode> {
        self.0.iter().cloned().collect()
    }

    pub fn add_conn_and_report(&self, connection: Connection) -> Result<Connections, CommError> {
        let new_connections = self.add_conn(connection)?;
        new_connections.report_conn()
    }

    pub fn add_conn(&self, connection: Connection) -> Result<Connections, CommError> {
        self.add_conns(vec![connection])
    }

    pub fn add_conns(&self, to_be_added: Vec<Connection>) -> Result<Connections, CommError> {
        let ids_to_add: Vec<_> = to_be_added.iter().map(|peer| &peer.id.key).collect();

        // Remove any existing connections with the same IDs
        let existing_without_duplicates: Vec<Connection> = self
            .0
            .iter()
            .filter(|peer| !ids_to_add.contains(&&peer.id.key))
            .cloned()
            .collect();

        // Add the new connections
        let mut new_connections = existing_without_duplicates;
        new_connections.extend(to_be_added);

        Ok(Connections(new_connections))
    }

    pub fn remove_conn_and_report(&self, connection: Connection) -> Result<Connections, CommError> {
        let new_connections = self.remove_conn(connection)?;
        new_connections.report_conn()
    }

    pub fn remove_conn(&self, connection: Connection) -> Result<Connections, CommError> {
        self.remove_conns(vec![connection])
    }

    pub fn remove_conns(&self, to_be_removed: Vec<Connection>) -> Result<Connections, CommError> {
        let ids_to_remove: Vec<_> = to_be_removed.iter().map(|peer| &peer.id.key).collect();

        // Keep only connections whose IDs are not in the removal list
        let remaining_connections: Vec<Connection> = self
            .0
            .iter()
            .filter(|peer| !ids_to_remove.contains(&&peer.id.key))
            .cloned()
            .collect();

        Ok(Connections(remaining_connections))
    }

    pub fn refresh_conn(&self, connection: Connection) -> Result<Connections, CommError> {
        let mut new_connections: Vec<Connection> = self
            .0
            .iter()
            .filter(|peer| peer.id.key != connection.id.key)
            .cloned()
            .collect();

        // If the connection existed in the original list, add it to the end
        if self.0.iter().any(|peer| peer.id.key == connection.id.key) {
            new_connections.push(connection);
        }

        Ok(Connections(new_connections))
    }

    pub fn report_conn(&self) -> Result<Connections, CommError> {
        let size = self.0.len();
        info!("Peers: {}", size);
        metrics::gauge!("peers", "source" => RP_CONNECT_METRICS_SOURCE).set(size as f64);
        metrics::counter!(CONNECT_METRIC, "source" => RP_CONNECT_METRICS_SOURCE).increment(1);
        Ok(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct ConnectionsCell {
    pub peers: Arc<Mutex<Connections>>,
}

impl ConnectionsCell {
    pub fn new() -> Self {
        Self {
            peers: Arc::new(Mutex::new(Connections::empty())),
        }
    }

    pub fn random(&self, max: usize) -> Result<Connections, CommError> {
        let peers = self.peers.lock().map_err(|_| {
            CommError::InternalCommunicationError("ConnectionsCell lock poisoned".to_string())
        })?;

        let mut rng = rand::rng();
        let mut shuffled = peers.0.clone();
        shuffled.shuffle(&mut rng);

        Ok(Connections(shuffled.into_iter().take(max).collect()))
    }

    pub fn read(&self) -> Result<Connections, CommError> {
        let peers = self.peers.lock().map_err(|_| {
            CommError::InternalCommunicationError("ConnectionsCell lock poisoned".to_string())
        })?;
        Ok(peers.clone())
    }

    pub fn flat_modify<F>(&self, f: F) -> Result<Connections, CommError>
    where
        F: FnOnce(Connections) -> Result<Connections, CommError>,
    {
        let mut peers = self.peers.lock().map_err(|_| {
            CommError::InternalCommunicationError("ConnectionsCell lock poisoned".to_string())
        })?;

        let current_peers = peers.clone();
        let new_peers = f(current_peers)?;
        *peers = new_peers.clone();

        Ok(new_peers)
    }
}

/// Clear connections by sending heartbeats and removing failed peers
pub async fn clear_connections<T: TransportLayer>(
    connections_cell: &ConnectionsCell,
    conf: &RPConf,
    transport: &T,
) -> Result<usize, CommError> {
    let connections = connections_cell.read()?;
    let num_to_ping = conf.clear_connections.num_of_connections_pinged;
    let to_ping = connections.take(num_to_ping);

    let mut results = Vec::new();

    // Send heartbeats to each peer
    for peer in to_ping.iter() {
        let heartbeat_msg = protocol_helper::heartbeat(&conf.local, &conf.network_id);
        let result = transport.send(peer, &heartbeat_msg).await;
        results.push((peer.clone(), result));
    }

    // Separate successful and failed peers
    let successful_peers: Vec<PeerNode> = results
        .iter()
        .filter_map(|(peer, result)| {
            if result.is_ok() {
                Some(peer.clone())
            } else {
                None
            }
        })
        .collect();

    let failed_peers: Vec<PeerNode> = results
        .iter()
        .filter_map(|(peer, result)| {
            if result.is_err() {
                Some(peer.clone())
            } else {
                None
            }
        })
        .collect();

    // Log removal of failed peers
    for peer in &failed_peers {
        info!("Removing peer {} from connections", peer);
    }

    // Update connections: remove all pinged peers, then add back successful ones
    let failed_count = failed_peers.len();
    connections_cell.flat_modify(|conns| {
        let updated = conns.remove_conns(to_ping.into_vec())?;
        updated.add_conns(successful_peers)
    })?;

    // Report connections if any were cleared
    if failed_count > 0 {
        let updated_connections = connections_cell.read()?;
        updated_connections.report_conn()?;
    }

    Ok(failed_count)
}

/// Reset connections by removing all current connections
pub fn reset_connections(connections_cell: &ConnectionsCell) -> Result<(), CommError> {
    connections_cell.flat_modify(|conns| conns.remove_conns(conns.clone().into_vec()))?;
    Ok(())
}

/// Find new peers and attempt to connect to them
pub async fn find_and_connect<N: NodeDiscovery + ?Sized, F, Fut>(
    connections_cell: &ConnectionsCell,
    node_discovery: &N,
    connect_fn: F,
) -> Result<Vec<PeerNode>, CommError>
where
    F: Fn(&PeerNode) -> Fut,
    Fut: std::future::Future<Output = Result<(), CommError>>,
{
    let current_connections = connections_cell.read()?.to_set();
    let all_peers = node_discovery.peers()?;

    // Filter out peers we're already connected to
    let new_peers: Vec<PeerNode> = all_peers
        .into_iter()
        .filter(|peer| !current_connections.contains(peer))
        .collect();

    let mut successful_connections = Vec::new();

    // Attempt to connect to each new peer
    for peer in new_peers {
        match connect_fn(&peer).await {
            Ok(()) => {
                successful_connections.push(peer);
            }
            Err(CommError::WrongNetwork(peer_addr, msg)) => {
                warn!("Can't connect to peer {}. {}", peer_addr, msg);
            }
            Err(_) => {
                warn!(
                    "An error occurred while trying to connect to peer: {:?}",
                    peer
                );
            }
        }
    }

    Ok(successful_connections)
}

/// Connect to a peer by sending a protocol handshake
pub async fn connect<T: TransportLayer>(
    peer: &PeerNode,
    conf: &RPConf,
    transport: &T,
) -> Result<(), CommError> {
    let handshake_msg = protocol_helper::protocol_handshake(&conf.local, &conf.network_id);
    transport.send(peer, &handshake_msg).await
}
