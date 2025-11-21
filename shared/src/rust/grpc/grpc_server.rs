// See shared/src/main/scala/coop/rchain/grpc/GrpcServer.scala

use std::time::Duration;
use tokio::time::timeout;
use tonic::transport::Server as TonicServer;

/// A gRPC server wrapper that provides lifecycle management
pub struct GrpcServer {
    server_future: Option<tokio::task::JoinHandle<Result<(), tonic::transport::Error>>>,
    port: u16,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl GrpcServer {
    /// Create a new GrpcServer with the given port
    pub fn new(port: u16) -> Self {
        Self {
            server_future: None,
            port,
            shutdown_tx: None,
        }
    }

    /// Start the gRPC server with a service
    pub async fn start_with_service<S>(
        &mut self,
        service: S,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        S: tonic::server::NamedService
            + Clone
            + Send
            + Sync
            + 'static
            + tower::Service<
                http::Request<tonic::body::Body>,
                Response = http::Response<tonic::body::Body>,
                Error = std::convert::Infallible,
            >,
        S::Future: Send + 'static,
    {
        if self.server_future.is_some() {
            return Err("Server is already running".into());
        }

        let addr = ([127, 0, 0, 1], self.port).into();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

        let server_future = tokio::spawn(async move {
            TonicServer::builder()
                .add_service(service)
                .serve_with_shutdown(addr, async {
                    shutdown_rx.await.ok();
                })
                .await
        });

        self.server_future = Some(server_future);
        self.shutdown_tx = Some(shutdown_tx);

        Ok(())
    }

    /// Start the gRPC server with a router
    pub async fn start_with_router(
        &mut self,
        router: tonic::transport::server::Router,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.server_future.is_some() {
            return Err("Server is already running".into());
        }

        let addr = ([127, 0, 0, 1], self.port).into();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

        let server_future = tokio::spawn(async move {
            router
                .serve_with_shutdown(addr, async {
                    shutdown_rx.await.ok();
                })
                .await
        });

        self.server_future = Some(server_future);
        self.shutdown_tx = Some(shutdown_tx);

        Ok(())
    }

    /// Stop the gRPC server
    pub async fn stop(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            // Send shutdown signal
            let _ = shutdown_tx.send(());

            if let Some(server_future) = self.server_future.take() {
                // Attempt graceful shutdown with timeout
                match timeout(Duration::from_millis(1000), server_future).await {
                    Ok(result) => {
                        // Server shut down within timeout
                        result??;
                    }
                    Err(_) => {
                        // Timeout occurred
                        // The server task will be dropped, effectively forcing shutdown
                        tracing::warn!("Server shutdown timed out, forcing termination");
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the port the server is configured to run on
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Check if the server is currently running
    pub fn is_running(&self) -> bool {
        self.server_future.is_some()
    }

    /// Take the server future handle for external lifecycle management
    /// 
    /// This allows the caller to await the server task for monitoring.
    /// After calling this, the server will no longer manage its own lifecycle.
    /// 
    /// Returns None if the server is not running or handle was already taken.
    pub fn take_handle(&mut self) -> Option<tokio::task::JoinHandle<Result<(), tonic::transport::Error>>> {
        self.server_future.take()
    }
}

impl Drop for GrpcServer {
    fn drop(&mut self) {
        // Ensure cleanup on drop
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_server_lifecycle() {
        let mut server = GrpcServer::new(0); // Use port 0 for testing

        // Initially not running
        assert!(!server.is_running());

        // Port should be accessible
        assert_eq!(server.port(), 0);

        // Stop should work even if not started
        assert!(server.stop().await.is_ok());
    }

    #[tokio::test]
    async fn test_server_port() {
        let server = GrpcServer::new(8080);
        assert_eq!(server.port(), 8080);
    }
}
