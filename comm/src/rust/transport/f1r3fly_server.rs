// F1r3fly Server for tonic's gRPC server with custom TLS acceptor
//
// This module provides F1r3flyServer::builder() to create tonic servers
// that use F1r3fly's custom TLS acceptor with HostnameTrustManager integration.

use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::transport::server::{Connected, TcpConnectInfo};

use crate::rust::transport::{
    f1r3fly_tls_connector::F1r3flyTlsAcceptor,
    f1r3fly_tls_transport::{F1r3flyServerTlsTransport, F1r3flyTlsTransportError},
};

/// F1r3fly Server Builder for creating tonic servers with custom TLS
///
/// This builder allows creating tonic gRPC servers that use F1r3fly's custom
/// TLS acceptor with HostnameTrustManager integration instead of standard TLS.
///
/// # Architecture
///
/// The builder creates a custom incoming stream that:
/// 1. Accepts TCP connections via TcpListener
/// 2. Performs TLS handshake using F1r3flyTlsAcceptor
/// 3. Wraps connected streams in F1r3flyServerTlsTransport
/// 4. Provides these to tonic as pre-established TLS connections
///
/// This approach bypasses tonic's built-in TLS configuration and gives us
/// full control over the TLS handshake and certificate verification process.
#[derive(Clone, Debug)]
pub struct F1r3flyServer {
    acceptor: F1r3flyTlsAcceptor,
    bind_addr: SocketAddr,
    tcp_keepalive: Option<Duration>,
    tcp_nodelay: bool,
    http2_keepalive_interval: Option<Duration>,
    http2_keepalive_timeout: Option<Duration>,
}

/// Error types for F1r3flyServer
#[derive(Debug, thiserror::Error)]
pub enum F1r3flyServerError {
    #[error("TLS transport error: {0}")]
    TlsTransport(#[from] F1r3flyTlsTransportError),
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Address binding error: {0}")]
    Bind(String),
    #[error("Server configuration error: {0}")]
    Config(String),
}

impl F1r3flyServer {
    /// Create a new F1r3flyServer builder
    ///
    /// # Arguments
    /// * `network_id` - Network identifier for F1r3fly
    /// * `cert_pem` - Server certificate in PEM format
    /// * `key_pem` - Server private key in PEM format
    /// * `bind_addr` - Address to bind the server to
    pub fn builder(
        network_id: String,
        cert_pem: &str,
        key_pem: &str,
        bind_addr: SocketAddr,
    ) -> Result<Self, F1r3flyServerError> {
        // Create F1r3fly TLS acceptor with custom certificate verification
        let acceptor = F1r3flyTlsAcceptor::new(network_id, cert_pem, key_pem)
            .map_err(F1r3flyServerError::TlsTransport)?;

        Ok(Self {
            acceptor,
            bind_addr,
            tcp_keepalive: Some(Duration::from_secs(600)), // Default 10 minutes
            tcp_nodelay: true,
            http2_keepalive_interval: Some(Duration::from_secs(30)),
            http2_keepalive_timeout: Some(Duration::from_secs(5)),
        })
    }

    /// Configure TCP keepalive settings
    pub fn tcp_keepalive(mut self, keepalive: Option<Duration>) -> Self {
        self.tcp_keepalive = keepalive;
        self
    }

    /// Configure TCP nodelay (Nagle's algorithm)
    pub fn tcp_nodelay(mut self, nodelay: bool) -> Self {
        self.tcp_nodelay = nodelay;
        self
    }

    /// Configure HTTP/2 keepalive interval
    pub fn http2_keepalive_interval(mut self, interval: Option<Duration>) -> Self {
        self.http2_keepalive_interval = interval;
        self
    }

    /// Configure HTTP/2 keepalive timeout
    pub fn http2_keepalive_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.http2_keepalive_timeout = timeout;
        self
    }

    /// Create the custom incoming stream for tonic server
    ///
    /// This creates a stream of TLS-authenticated connections that tonic can use
    /// to serve gRPC requests. Each connection is pre-authenticated using F1r3fly's
    /// custom certificate verification.
    pub async fn incoming(self) -> Result<F1r3flyIncoming, F1r3flyServerError> {
        // Bind TCP listener
        let listener = TcpListener::bind(&self.bind_addr).await.map_err(|e| {
            F1r3flyServerError::Bind(format!("Failed to bind to {}: {}", self.bind_addr, e))
        })?;

        tracing::info!("F1r3fly server listening on {}", self.bind_addr);

        // Create channel for connection results
        let (tx, rx) = mpsc::channel(10); // Buffer up to 10 connections

        let acceptor = self.acceptor;
        let tcp_keepalive = self.tcp_keepalive;
        let tcp_nodelay = self.tcp_nodelay;

        // Spawn background task to handle incoming connections
        let listener_task = tokio::spawn(async move {
            let mut tcp_listener_stream = TcpListenerStream::new(listener);

            loop {
                use tokio_stream::StreamExt;

                match tcp_listener_stream.next().await {
                    Some(Ok(tcp_stream)) => {
                        // Configure TCP socket options
                        if let Err(e) = tcp_stream.set_nodelay(tcp_nodelay) {
                            tracing::warn!("Failed to set TCP nodelay: {}", e);
                        }

                        if tcp_keepalive.is_some() {
                            tracing::debug!("TCP keepalive configuration requested but not implemented in tokio TcpStream");
                        }

                        // Get peer address
                        let peer_addr = tcp_stream
                            .peer_addr()
                            .unwrap_or_else(|_| std::net::SocketAddr::from(([0, 0, 0, 0], 0)));

                        // Spawn TLS handshake in background
                        let acceptor_clone = acceptor.clone();
                        let tx_clone = tx.clone();

                        tokio::spawn(async move {
                            let result = match acceptor_clone.accept(tcp_stream).await {
                                Ok(f1r3fly_transport) => {
                                    tracing::debug!(
                                        "F1r3fly TLS handshake successful for {}",
                                        peer_addr
                                    );
                                    Ok(F1r3flyServerConnection::new(f1r3fly_transport, peer_addr))
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "F1r3fly TLS handshake failed for {}: {}",
                                        peer_addr,
                                        e
                                    );
                                    Err(F1r3flyServerError::TlsTransport(e))
                                }
                            };

                            // Send result through channel
                            if let Err(_) = tx_clone.send(result).await {
                                tracing::debug!(
                                    "Connection channel closed, stopping TLS handshake task"
                                );
                            }
                        });
                    }
                    Some(Err(e)) => {
                        tracing::error!("TCP listener error: {}", e);
                        let _ = tx.send(Err(F1r3flyServerError::Io(e))).await;
                    }
                    None => {
                        tracing::info!("TCP listener closed");
                        break;
                    }
                }
            }
        });

        Ok(F1r3flyIncoming {
            receiver: ReceiverStream::new(rx),
            _listener_task: listener_task,
        })
    }
}

/// F1r3fly incoming connection stream
///
/// This stream accepts TCP connections and performs F1r3fly TLS handshakes
/// to create authenticated connections for tonic. Uses a channel-based approach
/// to handle async TLS handshakes without blocking the stream.
pub struct F1r3flyIncoming {
    receiver: ReceiverStream<Result<F1r3flyServerConnection, F1r3flyServerError>>,
    _listener_task: tokio::task::JoinHandle<()>,
}

impl tokio_stream::Stream for F1r3flyIncoming {
    type Item = Result<F1r3flyServerConnection, F1r3flyServerError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Simply poll the receiver stream
        Pin::new(&mut self.receiver).poll_next(cx)
    }
}

/// F1r3fly server connection wrapper
///
/// This wraps the F1r3flyServerTlsTransport to provide the necessary traits
/// for tonic server integration, including access to peer certificates.
#[derive(Debug)]
pub struct F1r3flyServerConnection {
    inner: F1r3flyServerTlsTransport<tokio::net::TcpStream>,
    peer_addr: SocketAddr,
    peer_certificates: Option<Vec<Vec<u8>>>,
}

/// Custom connection info that includes F1r3fly certificate data
#[derive(Clone, Debug)]
pub struct F1r3flyConnectInfo {
    pub tcp_info: TcpConnectInfo,
    pub peer_certificates: Option<Vec<Vec<u8>>>,
}

impl F1r3flyServerConnection {
    /// Create a new F1r3flyServerConnection and extract certificates
    pub fn new(
        transport: F1r3flyServerTlsTransport<tokio::net::TcpStream>,
        peer_addr: SocketAddr,
    ) -> Self {
        // Extract peer certificates from TLS session
        let peer_certificates = transport.peer_certificates();

        tracing::debug!(
            "F1r3flyServerConnection: Extracted {} peer certificates for {}",
            peer_certificates.as_ref().map_or(0, |certs| certs.len()),
            peer_addr
        );

        Self {
            inner: transport,
            peer_addr,
            peer_certificates,
        }
    }
}

impl AsyncRead for F1r3flyServerConnection {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for F1r3flyServerConnection {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, io::Error>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), io::Error>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), io::Error>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

impl Connected for F1r3flyServerConnection {
    type ConnectInfo = F1r3flyConnectInfo;

    fn connect_info(&self) -> Self::ConnectInfo {
        F1r3flyConnectInfo {
            tcp_info: TcpConnectInfo {
                local_addr: None, // Would need to store this if required
                remote_addr: Some(self.peer_addr),
            },
            peer_certificates: self.peer_certificates.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::rust::util::certificate_helper::{CertificateHelper, CertificatePrinter};

    #[test]
    fn test_f1r3fly_server_builder_creation() {
        // Generate test certificates
        let (secret_key, public_key) = CertificateHelper::generate_key_pair(true);
        let cert_der = CertificateHelper::generate_certificate(&secret_key, &public_key)
            .expect("Failed to generate test certificate");
        let cert_pem = CertificatePrinter::print_certificate(&cert_der);
        let key_pem = CertificatePrinter::print_private_key_from_secret(&secret_key)
            .expect("Failed to print private key");

        let bind_addr = "127.0.0.1:0".parse().unwrap();

        let server =
            F1r3flyServer::builder("test_network".to_string(), &cert_pem, &key_pem, bind_addr);

        assert!(server.is_ok());
        let server = server.unwrap();
        assert_eq!(server.bind_addr, bind_addr);
        assert_eq!(server.tcp_nodelay, true);
        assert_eq!(server.tcp_keepalive, Some(Duration::from_secs(600)));
    }

    #[test]
    fn test_f1r3fly_server_configuration() {
        // Generate test certificates
        let (secret_key, public_key) = CertificateHelper::generate_key_pair(true);
        let cert_der = CertificateHelper::generate_certificate(&secret_key, &public_key)
            .expect("Failed to generate test certificate");
        let cert_pem = CertificatePrinter::print_certificate(&cert_der);
        let key_pem = CertificatePrinter::print_private_key_from_secret(&secret_key)
            .expect("Failed to print private key");

        let bind_addr = "127.0.0.1:0".parse().unwrap();

        let server =
            F1r3flyServer::builder("test_network".to_string(), &cert_pem, &key_pem, bind_addr)
                .unwrap()
                .tcp_keepalive(Some(Duration::from_secs(300)))
                .tcp_nodelay(false)
                .http2_keepalive_interval(Some(Duration::from_secs(60)))
                .http2_keepalive_timeout(Some(Duration::from_secs(10)));

        assert_eq!(server.tcp_keepalive, Some(Duration::from_secs(300)));
        assert_eq!(server.tcp_nodelay, false);
        assert_eq!(
            server.http2_keepalive_interval,
            Some(Duration::from_secs(60))
        );
        assert_eq!(
            server.http2_keepalive_timeout,
            Some(Duration::from_secs(10))
        );
    }

    #[test]
    fn test_f1r3fly_server_builder_invalid_certs() {
        let bind_addr = "127.0.0.1:0".parse().unwrap();

        let result = F1r3flyServer::builder(
            "test_network".to_string(),
            "invalid cert",
            "invalid key",
            bind_addr,
        );

        assert!(result.is_err());
    }
}
