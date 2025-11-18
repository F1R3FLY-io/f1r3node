// See comm/src/main/scala/coop/rchain/comm/rp/Connect.scala

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::rust::errors::CommError;
use crate::rust::peer_node::PeerNode;

#[derive(Debug, Clone)]
pub struct RPConf {
    pub local: PeerNode,
    pub network_id: String,
    pub bootstrap: Option<PeerNode>,
    pub default_timeout: Duration,
    pub max_num_of_connections: usize,
    pub clear_connections: ClearConnectionsConf,
}

impl RPConf {
    pub fn new(
        local: PeerNode,
        network_id: String,
        bootstrap: Option<PeerNode>,
        default_timeout: Duration,
        max_num_of_connections: usize,
        num_of_connections_pinged: usize,
    ) -> Self {
        Self {
            local,
            network_id,
            bootstrap,
            default_timeout,
            max_num_of_connections,
            clear_connections: ClearConnectionsConf::new(num_of_connections_pinged),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClearConnectionsConf {
    pub num_of_connections_pinged: usize,
}

impl ClearConnectionsConf {
    pub fn new(num_of_connections_pinged: usize) -> Self {
        Self {
            num_of_connections_pinged,
        }
    }
}

/// Cell wrapper for RPConf to allow shared mutable access
/// Follows the same pattern as ConnectionsCell
#[derive(Clone)]
pub struct RPConfCell {
    conf: Arc<Mutex<RPConf>>,
}

impl RPConfCell {
    /// Create a new RPConfCell wrapping the given configuration
    pub fn new(conf: RPConf) -> Self {
        Self {
            conf: Arc::new(Mutex::new(conf)),
        }
    }

    /// Read the current RPConf
    pub fn read(&self) -> Result<RPConf, CommError> {
        self.conf
            .lock()
            .map(|conf| conf.clone())
            .map_err(|_| {
                CommError::InternalCommunicationError("RPConfCell lock poisoned".to_string())
            })
    }

    /// Update the local peer node
    pub fn update_local(&self, new_local: PeerNode) -> Result<(), CommError> {
        self.conf
            .lock()
            .map(|mut conf| {
                conf.local = new_local;
            })
            .map_err(|_| {
                CommError::InternalCommunicationError("RPConfCell lock poisoned".to_string())
            })
    }

    /// Modify the entire RPConf using a transformation function
    pub fn modify<F>(&self, f: F) -> Result<RPConf, CommError>
    where
        F: FnOnce(RPConf) -> Result<RPConf, CommError>,
    {
        let mut conf = self.conf.lock().map_err(|_| {
            CommError::InternalCommunicationError("RPConfCell lock poisoned".to_string())
        })?;

        let current = conf.clone();
        let new_conf = f(current)?;
        *conf = new_conf.clone();

        Ok(new_conf)
    }
}
