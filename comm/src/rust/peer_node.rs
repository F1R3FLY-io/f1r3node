// See comm/src/main/scala/coop/rchain/comm/PeerNode.scala

use crate::rust::errors::{parse_error, CommError};
use models::routing::Node;
use prost::bytes::Bytes;
use std::fmt;
use std::hash::{Hash, Hasher};
use url::Url;

#[derive(Debug, Clone)]
pub struct NodeIdentifier {
    pub key: Bytes,
}

impl PartialEq for NodeIdentifier {
    fn eq(&self, other: &Self) -> bool {
        // Compare the content of Bytes, not the Arc pointer
        self.key.as_ref() == other.key.as_ref()
    }
}

impl Eq for NodeIdentifier {}

impl Hash for NodeIdentifier {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the content of Bytes, not the Arc pointer
        self.key.as_ref().hash(state);
    }
}

impl NodeIdentifier {
    pub fn new(name: String) -> Self {
        let mut bytes = Vec::new();
        let chars: Vec<char> = name.chars().collect();

        for i in (0..chars.len()).step_by(2) {
            if i + 1 < chars.len() {
                let pair: String = chars[i..=i + 1].iter().collect();

                if let Ok(value) = u8::from_str_radix(&pair, 16) {
                    bytes.push(value);
                }
            }
        }

        Self {
            key: Bytes::from(bytes),
        }
    }

    pub fn to_string(&self) -> String {
        hex::encode(self.key.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Endpoint {
    pub host: String,
    pub tcp_port: u32,
    pub udp_port: u32,
}

impl Endpoint {
    pub fn new(host: String, tcp_port: u32, udp_port: u32) -> Self {
        Self {
            host,
            tcp_port,
            udp_port,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PeerNode {
    pub id: NodeIdentifier,
    pub endpoint: Endpoint,
}

impl PartialEq for PeerNode {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for PeerNode {}

impl Hash for PeerNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PeerNode {
    pub fn new(id: NodeIdentifier, host: String, tcp_port: u16, udp_port: u16) -> Self {
        Self {
            id,
            endpoint: Endpoint::new(host, tcp_port as u32, udp_port as u32),
        }
    }

    pub fn from_node(node: Node) -> Result<Self, CommError> {
        let host = String::from_utf8(node.host.to_vec())
            .map_err(|_| parse_error("Failed to convert host to string".to_string()))?;

        Ok(Self {
            id: NodeIdentifier { key: node.id },
            endpoint: Endpoint::new(host, node.tcp_port, node.udp_port),
        })
    }

    pub fn from_address(address: &str) -> Result<Self, CommError> {
        // Parse URL with proper scheme checking
        let url = match Url::parse(address) {
            Ok(url) => url,
            Err(_) => return Err(parse_error(format!("bad address: {}", address))),
        };

        // Check scheme is 'rnode'
        if url.scheme() != "rnode" {
            return Err(parse_error(format!("invalid scheme: {}", url.scheme())));
        }

        let id = match url.username() {
            "" => return Err(parse_error("missing node ID".to_string())),
            id => id,
        };

        let host = match url.host_str() {
            Some(host) => host.to_string(),
            None => return Err(parse_error("missing host".to_string())),
        };

        let discovery_port = url
            .query_pairs()
            .find(|(name, _)| name == "discovery")
            .and_then(|(_, value)| value.parse::<u16>().ok())
            .ok_or_else(|| parse_error("missing or invalid discovery port".to_string()))?;

        let protocol_port = url
            .query_pairs()
            .find(|(name, _)| name == "protocol")
            .and_then(|(_, value)| value.parse::<u16>().ok())
            .ok_or_else(|| parse_error("missing or invalid protocol port".to_string()))?;

        // Create PeerNode
        Ok(Self::new(
            NodeIdentifier::new(id.to_string()),
            host,
            protocol_port,
            discovery_port,
        ))
    }

    pub fn key(&self) -> &Bytes {
        &self.id.key
    }

    pub fn s_key(&self) -> String {
        self.id.to_string()
    }

    pub fn to_address(&self) -> String {
        format!(
            "rnode://{}@{}?protocol={}&discovery={}",
            self.s_key(),
            self.endpoint.host,
            self.endpoint.tcp_port,
            self.endpoint.udp_port
        )
    }
}

impl fmt::Display for PeerNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_address())
    }
}
