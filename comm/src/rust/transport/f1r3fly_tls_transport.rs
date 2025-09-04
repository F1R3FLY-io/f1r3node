// F1r3fly custom TLS transport wrapper for tonic compatibility
//
// This module provides wrappers around tokio-rustls TLS streams that implement
// the traits required by tonic for transport layer integration. This enables our
// complete F1r3fly TLS architecture that uses custom certificate verification
// (HostnameTrustManager) while maintaining full compatibility with tonic's gRPC functionality.
//
// ## Architecture Overview
//
// F1r3fly TLS integration consists of:
// - **Client Side**: F1r3flyConnector + F1r3flyClientTlsTransport (this module)
// - **Server Side**: F1r3flyServer + F1r3flyServerTlsTransport (this module)
// - **Certificate Verification**: HostnameTrustManager for secp256r1 and F1r3fly addresses
// - **SSL Interceptors**: Application-level validation on top of TLS layer

use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::task::{Context, Poll};

use hyper::rt::{Read, ReadBufCursor, Write};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio_rustls::{client::TlsStream as ClientTlsStream, server::TlsStream as ServerTlsStream};
use tonic::transport::server::Connected;

/// Custom TLS transport wrapper for F1r3fly client connections
///
/// Wraps a tokio-rustls client TLS stream to provide tonic-compatible transport.
/// This enables F1r3fly's custom certificate verification while maintaining
/// compatibility with tonic's Channel and gRPC client functionality.
#[derive(Debug)]
pub struct F1r3flyClientTlsTransport<S> {
    inner: ClientTlsStream<S>,
    remote_addr: Option<SocketAddr>,
}

impl<S> F1r3flyClientTlsTransport<S> {
    /// Create a new F1r3fly client TLS transport wrapper
    pub fn new(tls_stream: ClientTlsStream<S>) -> Self {
        Self {
            inner: tls_stream,
            remote_addr: None,
        }
    }

    /// Create a new F1r3fly client TLS transport wrapper with remote address
    pub fn new_with_addr(tls_stream: ClientTlsStream<S>, remote_addr: SocketAddr) -> Self {
        Self {
            inner: tls_stream,
            remote_addr: Some(remote_addr),
        }
    }

    /// Get the underlying TLS stream
    pub fn get_ref(&self) -> &ClientTlsStream<S> {
        &self.inner
    }

    /// Get a mutable reference to the underlying TLS stream
    pub fn get_mut(&mut self) -> &mut ClientTlsStream<S> {
        &mut self.inner
    }

    /// Get the remote socket address if available
    pub fn remote_addr(&self) -> Option<SocketAddr> {
        self.remote_addr
    }
}

/// Custom TLS transport wrapper for F1r3fly server connections
///
/// Wraps a tokio-rustls server TLS stream to provide tonic-compatible transport.
/// This enables F1r3fly's custom client certificate verification while maintaining
/// compatibility with tonic's Server and gRPC service functionality.
#[derive(Debug)]
pub struct F1r3flyServerTlsTransport<S> {
    inner: ServerTlsStream<S>,
    remote_addr: Option<SocketAddr>,
}

impl<S> F1r3flyServerTlsTransport<S> {
    /// Create a new F1r3fly server TLS transport wrapper
    pub fn new(tls_stream: ServerTlsStream<S>) -> Self {
        Self {
            inner: tls_stream,
            remote_addr: None,
        }
    }

    /// Create a new F1r3fly server TLS transport wrapper with remote address
    pub fn new_with_addr(tls_stream: ServerTlsStream<S>, remote_addr: SocketAddr) -> Self {
        Self {
            inner: tls_stream,
            remote_addr: Some(remote_addr),
        }
    }

    /// Get the underlying TLS stream
    pub fn get_ref(&self) -> &ServerTlsStream<S> {
        &self.inner
    }

    /// Get a mutable reference to the underlying TLS stream
    pub fn get_mut(&mut self) -> &mut ServerTlsStream<S> {
        &mut self.inner
    }

    /// Get the remote socket address if available
    pub fn remote_addr(&self) -> Option<SocketAddr> {
        self.remote_addr
    }

    /// Extract peer certificates from the TLS session
    ///
    /// This method returns the peer certificates as raw DER bytes that can be used
    /// for F1r3fly certificate verification in the application layer.
    pub fn peer_certificates(&self) -> Option<Vec<Vec<u8>>> {
        // Access the rustls connection state to get peer certificates
        let (_, connection) = self.inner.get_ref();

        // Get peer certificates from the rustls ServerConnection
        if let Some(cert_chain) = connection.peer_certificates() {
            if !cert_chain.is_empty() {
                let cert_bytes: Vec<Vec<u8>> = cert_chain
                    .iter()
                    .map(|cert| cert.as_ref().to_vec())
                    .collect();
                return Some(cert_bytes);
            }
        }

        None
    }
}

// Implement AsyncRead for client transport
impl<S> AsyncRead for F1r3flyClientTlsTransport<S>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

// Implement AsyncWrite for client transport
impl<S> AsyncWrite for F1r3flyClientTlsTransport<S>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
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

// Implement AsyncRead for server transport
impl<S> AsyncRead for F1r3flyServerTlsTransport<S>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

// Implement AsyncWrite for server transport
impl<S> AsyncWrite for F1r3flyServerTlsTransport<S>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
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

// Implement Connected trait for server transport (required by tonic server)
impl<S> Connected for F1r3flyServerTlsTransport<S>
where
    S: Connected,
{
    type ConnectInfo = S::ConnectInfo;

    fn connect_info(&self) -> Self::ConnectInfo {
        self.inner.get_ref().0.connect_info()
    }
}

// Add Unpin implementations for both transports to make them easier to work with
impl<S> Unpin for F1r3flyClientTlsTransport<S> where S: Unpin {}
impl<S> Unpin for F1r3flyServerTlsTransport<S> where S: Unpin {}

// Implement hyper::rt::io::Read for client transport (required by tonic)
impl<S> Read for F1r3flyClientTlsTransport<S>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        mut buf: ReadBufCursor<'_>,
    ) -> Poll<Result<(), io::Error>> {
        let n = unsafe {
            let mut tokio_buf = ReadBuf::uninit(buf.as_mut());
            match Pin::new(&mut self.inner).poll_read(cx, &mut tokio_buf) {
                Poll::Ready(Ok(())) => tokio_buf.filled().len(),
                Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
                Poll::Pending => return Poll::Pending,
            }
        };

        unsafe {
            buf.advance(n);
        }
        Poll::Ready(Ok(()))
    }
}

// Implement hyper::rt::io::Write for client transport (required by tonic)
impl<S> Write for F1r3flyClientTlsTransport<S>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
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

    fn is_write_vectored(&self) -> bool {
        // Delegate to the inner TLS stream
        false // Conservative default - tokio_rustls may not support vectored writes
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<Result<usize, io::Error>> {
        Pin::new(&mut self.inner).poll_write_vectored(cx, bufs)
    }
}

// Implement hyper::rt::io::Read for server transport (for completeness)
impl<S> Read for F1r3flyServerTlsTransport<S>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        mut buf: ReadBufCursor<'_>,
    ) -> Poll<Result<(), io::Error>> {
        let n = unsafe {
            let mut tokio_buf = ReadBuf::uninit(buf.as_mut());
            match Pin::new(&mut self.inner).poll_read(cx, &mut tokio_buf) {
                Poll::Ready(Ok(())) => tokio_buf.filled().len(),
                Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
                Poll::Pending => return Poll::Pending,
            }
        };

        unsafe {
            buf.advance(n);
        }
        Poll::Ready(Ok(()))
    }
}

// Implement hyper::rt::io::Write for server transport (for completeness)
impl<S> Write for F1r3flyServerTlsTransport<S>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
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

    fn is_write_vectored(&self) -> bool {
        false // Conservative default
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<Result<usize, io::Error>> {
        Pin::new(&mut self.inner).poll_write_vectored(cx, bufs)
    }
}

/// Type aliases for common use cases
pub type F1r3flyClientTlsConnection = F1r3flyClientTlsTransport<tokio::net::TcpStream>;
pub type F1r3flyServerTlsConnection = F1r3flyServerTlsTransport<tokio::net::TcpStream>;

/// Error type for F1r3fly TLS transport operations
#[derive(Debug, thiserror::Error)]
pub enum F1r3flyTlsTransportError {
    #[error("TLS handshake failed: {0}")]
    HandshakeFailed(#[from] rustls::Error),
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Invalid peer address: {0}")]
    InvalidPeerAddress(String),
    #[error("Certificate verification failed: {0}")]
    CertificateVerificationFailed(String),
    #[error("Transport configuration error: {0}")]
    ConfigError(String),
}

impl From<F1r3flyTlsTransportError> for io::Error {
    fn from(err: F1r3flyTlsTransportError) -> Self {
        match err {
            F1r3flyTlsTransportError::Io(io_err) => io_err,
            _ => io::Error::new(io::ErrorKind::Other, err.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_transport_creation() {
        // Test that we can create a client transport wrapper
        // Note: This is a basic structure test since we can't easily create
        // a real TLS stream in a unit test without a full TLS handshake

        // For now, just test that the error types work correctly
        let err = F1r3flyTlsTransportError::ConfigError("test".to_string());
        let io_err: io::Error = err.into();
        assert_eq!(io_err.kind(), io::ErrorKind::Other);
    }

    #[test]
    fn test_server_transport_creation() {
        // Similar basic test for server transport
        let err = F1r3flyTlsTransportError::CertificateVerificationFailed("test".to_string());
        assert!(err.to_string().contains("Certificate verification failed"));
    }

    #[test]
    fn test_transport_error_conversion() {
        // Test error type conversions
        let io_error = io::Error::new(io::ErrorKind::ConnectionRefused, "test");
        let transport_error = F1r3flyTlsTransportError::Io(io_error);

        let converted_back: io::Error = transport_error.into();
        assert_eq!(converted_back.kind(), io::ErrorKind::ConnectionRefused);
    }

    #[test]
    fn test_tonic_trait_requirements() {
        // Verify that our transport types implement the required traits for tonic compatibility
        // This test ensures that the types can be used where tonic expects them

        use std::marker::PhantomData;

        // Test that we can use the types in contexts that require AsyncRead + AsyncWrite + Unpin
        fn requires_async_io<T>(_: PhantomData<T>)
        where
            T: AsyncRead + AsyncWrite + Unpin + Send + 'static,
        {
        }

        // Test that server transport implements Connected trait
        fn requires_connected<T>(_: PhantomData<T>)
        where
            T: Connected + AsyncRead + AsyncWrite + Unpin + Send + 'static,
        {
        }

        // These calls will only compile if our types implement the required traits
        requires_async_io::<F1r3flyClientTlsConnection>(PhantomData);
        requires_async_io::<F1r3flyServerTlsConnection>(PhantomData);
        requires_connected::<F1r3flyServerTlsConnection>(PhantomData);
    }
}
