// See comm/src/main/scala/coop/rchain/comm/rp/Connect.scala

use std::time::Duration;

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
