// F1r3fly custom TLS connector for establishing connections with custom certificate verification
//
// This module provides connection builders that integrate F1r3fly's HostnameTrustManager
// with tokio-rustls to establish TLS connections compatible with tonic transport.

use std::fmt;
use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::TcpStream;
use tokio_rustls::{TlsAcceptor, TlsConnector};

use crate::rust::transport::{
    f1r3fly_tls_transport::{
        F1r3flyClientTlsTransport, F1r3flyServerTlsTransport, F1r3flyTlsTransportError,
    },
    hostname_trust_manager_factory::HostnameTrustManagerFactory,
};

/// F1r3fly TLS connector for client connections
///
/// Provides methods to establish TLS client connections using F1r3fly's custom
/// certificate verification while maintaining compatibility with tonic's transport layer.
#[derive(Clone)]
pub struct F1r3flyTlsConnector {
    connector: TlsConnector,
    network_id: String,
}

impl fmt::Debug for F1r3flyTlsConnector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("F1r3flyTlsConnector")
            .field("network_id", &self.network_id)
            .field("connector", &"TlsConnector { ... }")
            .finish()
    }
}

impl F1r3flyTlsConnector {
    /// Create a new F1r3fly TLS connector for client connections
    ///
    /// # Arguments
    /// * `network_id` - The network ID for peer validation
    /// * `cert_pem` - Client certificate in PEM format
    /// * `key_pem` - Client private key in PEM format
    ///
    /// # Returns
    /// * `Ok(F1r3flyTlsConnector)` if successful
    /// * `Err(F1r3flyTlsTransportError)` if configuration fails
    pub fn new(
        network_id: String,
        cert_pem: &str,
        key_pem: &str,
    ) -> Result<Self, F1r3flyTlsTransportError> {
        // Create custom client configuration using HostnameTrustManagerFactory
        let client_config =
            HostnameTrustManagerFactory::client_config(cert_pem, key_pem).map_err(|e| {
                F1r3flyTlsTransportError::ConfigError(format!(
                    "Failed to create client TLS config: {}",
                    e
                ))
            })?;

        let connector = TlsConnector::from(Arc::new(client_config));

        Ok(Self {
            connector,
            network_id,
        })
    }

    /// Connect to a remote peer with F1r3fly TLS verification
    ///
    /// # Arguments
    /// * `tcp_stream` - Established TCP connection to the peer
    /// * `server_name` - Expected server name (usually the F1r3fly address)
    ///
    /// # Returns
    /// * `Ok(F1r3flyClientTlsTransport)` if TLS handshake succeeds
    /// * `Err(F1r3flyTlsTransportError)` if handshake or verification fails
    pub async fn connect(
        &self,
        tcp_stream: TcpStream,
        server_name: &str,
    ) -> Result<F1r3flyClientTlsTransport<TcpStream>, F1r3flyTlsTransportError> {
        let remote_addr = tcp_stream.peer_addr().ok();

        // Convert server name to rustls ServerName using String to avoid lifetime issues
        let server_name = rustls::pki_types::ServerName::try_from(server_name.to_string())
            .map_err(|e| {
                F1r3flyTlsTransportError::ConfigError(format!("Invalid server name: {}", e))
            })?;

        // Perform TLS handshake
        let tls_stream = self
            .connector
            .connect(server_name, tcp_stream)
            .await
            .map_err(F1r3flyTlsTransportError::from)?;

        // Wrap in F1r3fly transport
        let transport = if let Some(addr) = remote_addr {
            F1r3flyClientTlsTransport::new_with_addr(tls_stream, addr)
        } else {
            F1r3flyClientTlsTransport::new(tls_stream)
        };

        Ok(transport)
    }

    /// Connect to a peer by address
    ///
    /// Convenience method that establishes both TCP and TLS connections.
    ///
    /// # Arguments
    /// * `peer_addr` - Socket address of the peer
    /// * `server_name` - Expected server name for TLS verification
    ///
    /// # Returns
    /// * `Ok(F1r3flyClientTlsTransport)` if connection succeeds
    /// * `Err(F1r3flyTlsTransportError)` if connection or verification fails
    pub async fn connect_to_peer(
        &self,
        peer_addr: SocketAddr,
        server_name: &str,
    ) -> Result<F1r3flyClientTlsTransport<TcpStream>, F1r3flyTlsTransportError> {
        // Establish TCP connection
        let tcp_stream = TcpStream::connect(peer_addr)
            .await
            .map_err(F1r3flyTlsTransportError::Io)?;

        // Perform TLS handshake
        self.connect(tcp_stream, server_name).await
    }

    /// Get the network ID for this connector
    pub fn network_id(&self) -> &str {
        &self.network_id
    }
}

/// F1r3fly TLS acceptor for server connections
///
/// Provides methods to accept TLS server connections using F1r3fly's custom
/// client certificate verification while maintaining compatibility with tonic's server.
#[derive(Clone)]
pub struct F1r3flyTlsAcceptor {
    acceptor: TlsAcceptor,
    network_id: String,
}

impl fmt::Debug for F1r3flyTlsAcceptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("F1r3flyTlsAcceptor")
            .field("network_id", &self.network_id)
            .field("acceptor", &"TlsAcceptor { ... }")
            .finish()
    }
}

impl F1r3flyTlsAcceptor {
    /// Create a new F1r3fly TLS acceptor for server connections
    ///
    /// # Arguments
    /// * `network_id` - The network ID for client validation
    /// * `cert_pem` - Server certificate in PEM format
    /// * `key_pem` - Server private key in PEM format
    ///
    /// # Returns
    /// * `Ok(F1r3flyTlsAcceptor)` if successful
    /// * `Err(F1r3flyTlsTransportError)` if configuration fails
    pub fn new(
        network_id: String,
        cert_pem: &str,
        key_pem: &str,
    ) -> Result<Self, F1r3flyTlsTransportError> {
        // Create custom server configuration using HostnameTrustManagerFactory
        let server_config =
            HostnameTrustManagerFactory::server_config(cert_pem, key_pem).map_err(|e| {
                F1r3flyTlsTransportError::ConfigError(format!(
                    "Failed to create server TLS config: {}",
                    e
                ))
            })?;

        let acceptor = TlsAcceptor::from(Arc::new(server_config));

        Ok(Self {
            acceptor,
            network_id,
        })
    }

    /// Accept a TLS connection from a client
    ///
    /// # Arguments
    /// * `tcp_stream` - Incoming TCP connection from client
    ///
    /// # Returns
    /// * `Ok(F1r3flyServerTlsTransport)` if TLS handshake succeeds
    /// * `Err(F1r3flyTlsTransportError)` if handshake or verification fails
    pub async fn accept(
        &self,
        tcp_stream: TcpStream,
    ) -> Result<F1r3flyServerTlsTransport<TcpStream>, F1r3flyTlsTransportError> {
        let remote_addr = tcp_stream.peer_addr().ok();

        // Perform TLS handshake with client certificate verification
        let tls_stream = self
            .acceptor
            .accept(tcp_stream)
            .await
            .map_err(F1r3flyTlsTransportError::from)?;

        // Wrap in F1r3fly transport
        let transport = if let Some(addr) = remote_addr {
            F1r3flyServerTlsTransport::new_with_addr(tls_stream, addr)
        } else {
            F1r3flyServerTlsTransport::new(tls_stream)
        };

        Ok(transport)
    }

    /// Get the network ID for this acceptor
    pub fn network_id(&self) -> &str {
        &self.network_id
    }
}

/// Builder for F1r3fly TLS configurations
///
/// Provides a convenient builder pattern for creating TLS connectors and acceptors
/// with F1r3fly's custom certificate verification.
#[derive(Debug)]
pub struct F1r3flyTlsBuilder {
    network_id: String,
    cert_pem: String,
    key_pem: String,
}

impl F1r3flyTlsBuilder {
    /// Create a new TLS configuration builder
    ///
    /// # Arguments
    /// * `network_id` - The network ID for peer validation
    /// * `cert_pem` - Certificate in PEM format
    /// * `key_pem` - Private key in PEM format
    pub fn new(network_id: String, cert_pem: String, key_pem: String) -> Self {
        Self {
            network_id,
            cert_pem,
            key_pem,
        }
    }

    /// Build a TLS connector for client connections
    pub fn build_connector(&self) -> Result<F1r3flyTlsConnector, F1r3flyTlsTransportError> {
        F1r3flyTlsConnector::new(self.network_id.clone(), &self.cert_pem, &self.key_pem)
    }

    /// Build a TLS acceptor for server connections
    pub fn build_acceptor(&self) -> Result<F1r3flyTlsAcceptor, F1r3flyTlsTransportError> {
        F1r3flyTlsAcceptor::new(self.network_id.clone(), &self.cert_pem, &self.key_pem)
    }

    /// Get the network ID
    pub fn network_id(&self) -> &str {
        &self.network_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = F1r3flyTlsBuilder::new(
            "test_network".to_string(),
            "test_cert".to_string(),
            "test_key".to_string(),
        );

        assert_eq!(builder.network_id(), "test_network");
    }
}
