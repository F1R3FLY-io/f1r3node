// See comm/src/test/scala/coop/rchain/p2p/EffectsTestInstances.scala

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use models::routing::Protocol;

use crate::rust::discovery::node_discovery::NodeDiscovery;
use crate::rust::errors::CommError;
use crate::rust::peer_node::PeerNode;
use crate::rust::rp::protocol_helper;
use crate::rust::rp::rp_conf::{ClearConnectionsConf, RPConf};
use crate::rust::transport::transport_layer::{Blob, TransportLayer};

pub const NETWORK_ID: &str = "test";

pub struct NodeDiscoveryStub {
    pub nodes: Vec<PeerNode>,
}

impl NodeDiscoveryStub {
    pub fn new() -> Self {
        Self { nodes: vec![] }
    }

    pub fn reset(&mut self) {
        self.nodes = vec![];
    }

    pub fn peers(&self) -> Vec<PeerNode> {
        self.nodes.clone()
    }
}

impl NodeDiscovery for NodeDiscoveryStub {
    fn discover(&self) -> Result<(), CommError> {
        todo!()
    }

    fn peers(&self) -> Result<Vec<PeerNode>, CommError> {
        Ok(self.nodes.clone())
    }
}

pub fn create_rp_conf_ask(
    local: PeerNode,
    default_timeout: Option<Duration>,
    clear_connections: Option<ClearConnectionsConf>,
) -> RPConf {
    RPConf {
        local: local.clone(),
        network_id: NETWORK_ID.to_string(),
        bootstrap: Some(local),
        default_timeout: default_timeout.unwrap_or(Duration::from_millis(1)),
        max_num_of_connections: 20,
        clear_connections: clear_connections.unwrap_or(ClearConnectionsConf::new(1)),
    }
}

#[derive(Debug, Clone)]
pub struct Request {
    pub peer: PeerNode,
    pub msg: Protocol,
}

pub type Responses = Box<dyn Fn(&PeerNode, &Protocol) -> Result<(), CommError> + Send + Sync>;

pub struct TransportLayerStub {
    reqresp: Arc<Mutex<Option<Arc<Responses>>>>,
    requests: Arc<Mutex<Vec<Request>>>,
}

impl TransportLayerStub {
    pub fn new() -> Self {
        Self {
            reqresp: Arc::new(Mutex::new(None)),
            requests: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn set_responses<F>(&self, responses: F)
    where
        F: Fn(&PeerNode, &Protocol) -> Result<(), CommError> + Send + Sync + 'static,
    {
        let mut reqresp = self.reqresp.lock().unwrap();
        *reqresp = Some(Arc::new(Box::new(responses)));
    }

    pub fn reset(&self) {
        let mut reqresp = self.reqresp.lock().unwrap();
        let mut requests = self.requests.lock().unwrap();
        *reqresp = None;
        requests.clear();
    }

    pub fn get_request(&self, i: usize) -> Option<(PeerNode, Protocol)> {
        let requests = self.requests.lock().unwrap();
        requests
            .get(i)
            .map(|req| (req.peer.clone(), req.msg.clone()))
    }

    pub fn request_count(&self) -> usize {
        let requests = self.requests.lock().unwrap();
        requests.len()
    }

    pub fn get_all_requests(&self) -> Vec<Request> {
        let requests = self.requests.lock().unwrap();
        requests.clone()
    }

    pub fn pop_request(&self) -> Option<Request> {
        let mut requests = self.requests.lock().unwrap();
        requests.pop()
    }
}

#[async_trait]
impl TransportLayer for TransportLayerStub {
    async fn send(&self, peer: &PeerNode, msg: &Protocol) -> Result<(), CommError> {
        // Add request to the list
        {
            let mut requests = self.requests.lock().unwrap();
            requests.push(Request {
                peer: peer.clone(),
                msg: msg.clone(),
            });
        }

        // Execute response function if available
        let reqresp = self.reqresp.lock().unwrap();
        if let Some(ref response_fn) = *reqresp {
            response_fn(peer, msg)
        } else {
            // Default to success if no response function is set
            Ok(())
        }
    }

    async fn broadcast(&self, peers: &[PeerNode], msg: &Protocol) -> Result<(), CommError> {
        // Add all requests to the list
        {
            let mut requests = self.requests.lock().unwrap();
            for peer in peers {
                requests.push(Request {
                    peer: peer.clone(),
                    msg: msg.clone(),
                });
            }
        }

        // For broadcast, we return success for all peers in the stub
        Ok(())
    }

    async fn stream(&self, peer: &PeerNode, blob: &Blob) -> Result<(), CommError> {
        self.stream_mult(&[peer.clone()], blob).await
    }

    async fn stream_mult(&self, peers: &[PeerNode], blob: &Blob) -> Result<(), CommError> {
        let protocol_msg = protocol_helper::packet(&blob.sender, NETWORK_ID, blob.packet.clone());
        self.broadcast(peers, &protocol_msg).await
    }
}
