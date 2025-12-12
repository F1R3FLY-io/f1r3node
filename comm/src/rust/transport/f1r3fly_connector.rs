// F1r3fly Connector for tonic's connect_with_connector API
// This implements Service<Uri> to provide custom TLS connections to tonic

use std::error::Error as StdError;
use std::future::Future;
use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use http::Uri;
use tokio::net::TcpStream;
use tower::Service;

use crate::rust::transport::{
    f1r3fly_tls_connector::F1r3flyTlsConnector,
    f1r3fly_tls_transport::{F1r3flyClientTlsTransport, F1r3flyTlsTransportError},
};
use crate::rust::utils::resolve_hostname_to_ip;

/// Default connection timeout for F1r3fly connections (30 seconds)
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// F1r3fly Connector for tonic's connect_with_connector API
///
/// This connector implements Service<Uri> and uses F1r3flyTlsConnector
/// to establish TLS connections with F1r3fly certificate verification.
/// It provides the bridge between tonic's channel creation and our custom TLS.
#[derive(Clone, Debug)]
pub struct F1r3flyConnector {
    tls_connector: F1r3flyTlsConnector,
    connect_timeout: Duration,
    /// F1r3fly address to use for TLS hostname verification (instead of IP address)
    peer_f1r3fly_address: String,
}

/// Error type for F1r3flyConnector
#[derive(Debug, thiserror::Error)]
pub enum F1r3flyConnectorError {
    #[error("Failed to parse URI: {0}")]
    UriParseError(String),
    #[error("TLS connection failed: {0}")]
    TlsConnectionError(#[from] F1r3flyTlsTransportError),
    #[error("TCP connection failed: {0}")]
    TcpConnectionError(#[from] std::io::Error),
    #[error("DNS resolution failed: {0}")]
    DnsResolutionError(String),
    #[error("Connection timeout after {timeout:?}")]
    ConnectionTimeout { timeout: Duration },
}

impl F1r3flyConnector {
    /// Create a new F1r3flyConnector
    ///
    /// # Arguments
    /// * `network_id` - The network identifier
    /// * `cert` - PEM-encoded client certificate
    /// * `key` - PEM-encoded private key
    /// * `peer_f1r3fly_address` - The peer's F1r3fly address (hex-encoded) to use for TLS hostname verification
    pub fn new(
        network_id: String,
        cert: &str,
        key: &str,
        peer_f1r3fly_address: String,
    ) -> Result<Self, F1r3flyTlsTransportError> {
        let tls_connector = F1r3flyTlsConnector::new(network_id, cert, key)?;

        Ok(Self {
            tls_connector,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            peer_f1r3fly_address,
        })
    }

    /// Create a new F1r3flyConnector with custom timeout
    ///
    /// # Arguments
    /// * `network_id` - The network identifier
    /// * `cert` - PEM-encoded client certificate
    /// * `key` - PEM-encoded private key
    /// * `peer_f1r3fly_address` - The peer's F1r3fly address (hex-encoded) to use for TLS hostname verification
    /// * `connect_timeout` - Maximum time to wait for connection establishment
    pub fn new_with_timeout(
        network_id: String,
        cert: &str,
        key: &str,
        peer_f1r3fly_address: String,
        connect_timeout: Duration,
    ) -> Result<Self, F1r3flyTlsTransportError> {
        let tls_connector = F1r3flyTlsConnector::new(network_id, cert, key)?;

        Ok(Self {
            tls_connector,
            connect_timeout,
            peer_f1r3fly_address,
        })
    }

    /// Extract host and port from URI and resolve to socket address
    /// Uses async DNS resolution to avoid blocking the runtime
    async fn extract_address(&self, uri: &Uri) -> Result<SocketAddr, F1r3flyConnectorError> {
        let host = uri
            .host()
            .ok_or_else(|| F1r3flyConnectorError::UriParseError("Missing host".to_string()))?;

        let port = uri
            .port_u16()
            .ok_or_else(|| F1r3flyConnectorError::UriParseError("Missing port".to_string()))?;

        // For hostnames, we need to resolve to an IP address
        // Try parsing as IP address first (fast path), if that fails, use async DNS resolution
        let addr_str = format!("{}:{}", host, port);

        match addr_str.parse::<SocketAddr>() {
            Ok(addr) => Ok(addr),
            Err(_) => {
                let resolved_addr = resolve_hostname_to_ip(&host, port as u32)
                    .await
                    .map_err(|e| F1r3flyConnectorError::DnsResolutionError(e.to_string()))?;

                Ok(resolved_addr)
            }
        }
    }

    /// Get the configured connection timeout
    pub fn connect_timeout(&self) -> Duration {
        self.connect_timeout
    }

    /// Get the network ID for this connector
    pub fn network_id(&self) -> &str {
        self.tls_connector.network_id()
    }

    /// Get the peer F1r3fly address used for TLS hostname verification
    pub fn peer_f1r3fly_address(&self) -> &str {
        &self.peer_f1r3fly_address
    }
}

impl Service<Uri> for F1r3flyConnector {
    type Response = F1r3flyClientTlsTransport<TcpStream>;
    type Error = Box<dyn StdError + Send + Sync>;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        // F1r3flyConnector is always ready to accept connections
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, uri: Uri) -> Self::Future {
        let tls_connector = self.tls_connector.clone();
        let connect_timeout = self.connect_timeout;
        let peer_f1r3fly_address = self.peer_f1r3fly_address.clone();

        Box::pin(async move {
            // Step 1: Extract and resolve address (with async DNS)
            let connector = F1r3flyConnector {
                tls_connector: tls_connector.clone(),
                connect_timeout,
                peer_f1r3fly_address: peer_f1r3fly_address.clone(),
            };

            let addr = connector.extract_address(&uri).await.map_err(|e| {
                tracing::error!(
                    "F1r3flyConnector: Address resolution failed for {}: {}",
                    uri,
                    e
                );
                Box::new(e) as Box<dyn StdError + Send + Sync>
            })?;

            // Step 2: Establish TCP connection with timeout
            let tcp_stream = tokio::time::timeout(connect_timeout, TcpStream::connect(addr))
                .await
                .map_err(|_| {
                    tracing::error!(
                        "F1r3flyConnector: TCP connection timeout after {:?} to {}",
                        connect_timeout,
                        addr
                    );
                    Box::new(F1r3flyConnectorError::ConnectionTimeout {
                        timeout: connect_timeout,
                    }) as Box<dyn StdError + Send + Sync>
                })?
                .map_err(|e| {
                    tracing::error!("F1r3flyConnector: TCP connection failed to {}: {}", addr, e);
                    Box::new(F1r3flyConnectorError::TcpConnectionError(e))
                        as Box<dyn StdError + Send + Sync>
                })?;

            // Step 3: Establish F1r3fly TLS connection with F1r3fly address as hostname
            let tls_transport = tokio::time::timeout(
                connect_timeout,
                tls_connector.connect(tcp_stream, &peer_f1r3fly_address),
            )
            .await
            .map_err(|_| {
                tracing::error!(
                    "F1r3flyConnector: TLS handshake timeout after {:?} to {}",
                    connect_timeout,
                    addr
                );
                Box::new(F1r3flyConnectorError::ConnectionTimeout {
                    timeout: connect_timeout,
                }) as Box<dyn StdError + Send + Sync>
            })?
            .map_err(|e| {
                tracing::error!("F1r3flyConnector: TLS handshake failed to {}: {}", addr, e);
                Box::new(F1r3flyConnectorError::TlsConnectionError(e))
                    as Box<dyn StdError + Send + Sync>
            })?;

            tracing::info!(
                "F1r3flyConnector: TLS connection established to {} with hostname verification: {}",
                addr,
                peer_f1r3fly_address
            );
            Ok(tls_transport)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Create a test F1r3flyConnector with valid certificates for error testing
    fn create_test_connector_with_invalid_certs(
    ) -> Result<F1r3flyConnector, F1r3flyTlsTransportError> {
        let network_id = "test".to_string();
        // Use properly formatted but invalid certificate data
        let cert = "-----BEGIN CERTIFICATE-----\nMIIBkTCB+wIJAMlyFqk69v+9MA0GCSqGSIb3DQEBCwUAMBQxEjAQBgNVBAMMCWxvY2FsaG9zdDAe\nFw0yMzAxMDEwMDAwMDBaFw0yNDAxMDEwMDAwMDBaMA0GCSqGSIb3DQEBCwUAA4GBQAA=\n-----END CERTIFICATE-----";
        let key = "-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgABCDEFGHIJKLMNOP\nQRSTUVWXYZ0123456789+/=\n-----END PRIVATE KEY-----";

        F1r3flyConnector::new(network_id, cert, key, "test_f1r3fly_address".to_string())
    }

    // Helper to create a valid connector for testing
    fn create_valid_test_connector() -> F1r3flyConnector {
        use crypto::rust::util::certificate_helper::{CertificateHelper, CertificatePrinter};

        let (secret_key, public_key) = CertificateHelper::generate_key_pair(true);
        let cert_der = CertificateHelper::generate_certificate(&secret_key, &public_key)
            .expect("Failed to generate test certificate");
        let cert_pem = CertificatePrinter::print_certificate(&cert_der);
        let key_pem = CertificatePrinter::print_private_key_from_secret(&secret_key)
            .expect("Failed to print private key");

        // Extract F1r3fly address from the generated public key
        let f1r3fly_address = CertificateHelper::public_address(&public_key)
            .map(|addr| hex::encode(&addr))
            .unwrap_or_else(|| "test_f1r3fly_address".to_string());

        F1r3flyConnector::new("test".to_string(), &cert_pem, &key_pem, f1r3fly_address)
            .expect("Failed to create test connector")
    }

    #[tokio::test]
    async fn test_extract_address_valid_uri() {
        let connector = create_valid_test_connector();

        // Use IP address instead of hostname to avoid DNS resolution in tests
        let uri: Uri = "https://127.0.0.1:443".parse().unwrap();

        let result = connector.extract_address(&uri).await;
        if let Err(ref e) = result {
            println!("Extract address error: {:?}", e);
        }
        assert!(result.is_ok());

        let addr = result.unwrap();
        assert_eq!(addr.port(), 443);
        assert_eq!(addr.ip().to_string(), "127.0.0.1");
    }

    #[tokio::test]
    async fn test_extract_address_missing_port() {
        let connector = create_valid_test_connector();

        // Use IP address without port
        let uri: Uri = "https://127.0.0.1".parse().unwrap();

        let result = connector.extract_address(&uri).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Missing port"));
    }

    #[tokio::test]
    async fn test_extract_address_with_hostname_resolution() {
        let connector = create_valid_test_connector();

        // Test with localhost hostname which should resolve to 127.0.0.1
        let uri: Uri = "https://localhost:8080".parse().unwrap();

        let result = connector.extract_address(&uri).await;

        // This should either succeed (if localhost resolves) or fail with a resolution error
        // Both are acceptable outcomes depending on the test environment
        match result {
            Ok(addr) => {
                assert_eq!(addr.port(), 8080);
                // Should be either 127.0.0.1 or ::1 depending on the resolution
                assert!(addr.ip().is_loopback());
            }
            Err(e) => {
                // If resolution fails, that's also acceptable in test environments
                assert!(
                    e.to_string().contains("Failed to resolve")
                        || e.to_string().contains("DNS resolution failed")
                );
            }
        }
    }

    #[test]
    fn test_connector_poll_ready() {
        let mut connector = create_valid_test_connector();

        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);

        let result = connector.poll_ready(&mut cx);
        assert!(matches!(result, Poll::Ready(Ok(()))));
    }

    #[test]
    fn test_connector_creation_with_invalid_certs() {
        // Test that connector creation properly handles invalid certificates
        let result = create_test_connector_with_invalid_certs();
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(
            error.to_string().contains("Certificate parsing error")
                || error
                    .to_string()
                    .contains("Failed to create client TLS config")
        );
    }

    #[test]
    fn test_connector_with_custom_timeout() {
        use crypto::rust::util::certificate_helper::{CertificateHelper, CertificatePrinter};

        let (secret_key, public_key) = CertificateHelper::generate_key_pair(true);
        let cert_der = CertificateHelper::generate_certificate(&secret_key, &public_key)
            .expect("Failed to generate test certificate");
        let cert_pem = CertificatePrinter::print_certificate(&cert_der);
        let key_pem = CertificatePrinter::print_private_key_from_secret(&secret_key)
            .expect("Failed to print private key");

        let custom_timeout = Duration::from_secs(10);
        let connector = F1r3flyConnector::new_with_timeout(
            "test".to_string(),
            &cert_pem,
            &key_pem,
            "test_f1r3fly_address".to_string(),
            custom_timeout,
        )
        .expect("Failed to create connector with timeout");

        assert_eq!(connector.connect_timeout(), custom_timeout);
        assert_eq!(connector.network_id(), "test");
    }

    #[test]
    fn test_default_timeout() {
        let connector = create_valid_test_connector();
        assert_eq!(connector.connect_timeout(), DEFAULT_CONNECT_TIMEOUT);
    }

    #[tokio::test]
    async fn test_dns_resolution_error() {
        let connector = create_valid_test_connector();

        // Use a hostname that should not resolve
        let uri: Uri = "https://this-hostname-should-not-exist-12345.invalid:443"
            .parse()
            .unwrap();

        let result = connector.extract_address(&uri).await;
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(matches!(
            error,
            F1r3flyConnectorError::DnsResolutionError(_)
        ));
    }

    #[test]
    fn test_error_display() {
        let timeout = Duration::from_secs(5);
        let timeout_error = F1r3flyConnectorError::ConnectionTimeout { timeout };
        assert!(timeout_error.to_string().contains("5s"));

        let dns_error = F1r3flyConnectorError::DnsResolutionError("test error".to_string());
        assert!(dns_error.to_string().contains("DNS resolution failed"));

        let uri_error = F1r3flyConnectorError::UriParseError("invalid URI".to_string());
        assert!(uri_error.to_string().contains("Failed to parse URI"));
    }
}
