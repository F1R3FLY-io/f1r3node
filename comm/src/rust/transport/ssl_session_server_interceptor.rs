// See comm/src/main/scala/coop/rchain/comm/transport/SslSessionServerInterceptor.scala

use p256::PublicKey as P256PublicKey;
use tonic::service::Interceptor;
use tonic::{Request, Status};

use crypto::rust::util::certificate_helper::CertificateHelper;
use hex;
use models::routing::{Header, Protocol, TlRequest};

/// SSL Session Server Interceptor for validating incoming gRPC requests
///
/// This interceptor validates incoming TLS connections and gRPC messages to ensure they:
/// 1. Come from the correct network (network ID validation)
/// 2. Have valid TLS certificates that match the sender identity
/// 3. Are properly formatted gRPC messages
///
/// # Architecture Notes
///
/// Unlike Scala's ServerInterceptor which can intercept the actual message content,
/// tonic's Interceptor trait only operates on metadata. Our F1r3fly architecture
/// solves this with a two-phase validation approach:
///
/// 1. **Interceptor Phase**: F1r3flyServer extracts TLS certificates and stores validation context
/// 2. **Service Phase**: This interceptor validates message content using the TLS context
///
/// This design enables full F1r3fly certificate verification while maintaining compatibility
/// with tonic's interceptor system and providing the same security guarantees as the Scala implementation.
///
#[derive(Clone, Debug)]
pub struct SslSessionServerInterceptor {
    network_id: String,
}

/// Certificate validation context stored in request extensions
#[derive(Clone, Debug)]
pub struct CertificateValidationContext {
    /// Peer certificates from TLS session (stored as raw DER bytes)
    pub peer_certificates: Option<Vec<Vec<u8>>>,
    /// Whether TLS validation passed
    pub tls_validation_passed: bool,
    /// Network ID for this interceptor
    pub network_id: String,
}

impl SslSessionServerInterceptor {
    /// Create a new SSL session server interceptor
    ///
    /// # Arguments
    /// * `network_id` - The expected network ID for validation
    pub fn new(network_id: String) -> Self {
        Self { network_id }
    }

    /// Validate a TLRequest message against TLS session context
    ///
    /// This method should be called from the gRPC service implementation to validate
    /// the actual message content against the TLS certificate validation performed
    /// by the interceptor.
    ///
    /// # Arguments
    /// * `request` - The gRPC request containing TLRequest and extensions
    ///
    /// # Returns
    /// * `Ok(())` if validation passes
    /// * `Err(Status)` with appropriate error code if validation fails
    pub fn validate_tl_request(request: &Request<TlRequest>) -> Result<(), Status> {
        // Extract validation context from request extensions
        let validation_context = request
            .extensions()
            .get::<CertificateValidationContext>()
            .ok_or_else(|| {
                tracing::warn!("No certificate validation context found in request");
                Status::internal("Missing certificate validation context")
            })?;

        // Check if TLS validation passed in interceptor phase
        if !validation_context.tls_validation_passed {
            tracing::warn!("TLS validation failed in interceptor phase");
            return Err(Status::unauthenticated("TLS validation failed"));
        }

        // Extract the TLRequest message
        let tl_request = request.get_ref();

        // Validate the protocol and header structure
        let protocol = tl_request
            .protocol
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("Malformed message - missing protocol"))?;

        Self::validate_protocol_with_certificates(
            protocol,
            &validation_context.network_id,
            &validation_context.peer_certificates,
        )
    }

    /// Validate protocol message against network ID and certificates
    fn validate_protocol_with_certificates(
        protocol: &Protocol,
        expected_network_id: &str,
        peer_certificates: &Option<Vec<Vec<u8>>>,
    ) -> Result<(), Status> {
        // Extract header from protocol
        let header = protocol
            .header
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("Malformed message - missing header"))?;

        // Validate network ID
        Self::validate_network_id(&header.network_id, expected_network_id)?;

        // Log request details if trace enabled
        if tracing::enabled!(tracing::Level::TRACE) {
            if let Some(sender) = &header.sender {
                tracing::trace!(
                    "Request from peer {}:{}",
                    String::from_utf8_lossy(&sender.host),
                    sender.tcp_port
                );
            }
        }

        // Validate certificate chain if certificates are present
        if let Some(certificates) = peer_certificates {
            Self::validate_certificate_chain(header, certificates)?;
        } else {
            tracing::warn!("No TLS Session. Closing connection");
            return Err(Status::unauthenticated("No TLS Session"));
        }

        Ok(())
    }

    /// Validate network ID matches expected value
    fn validate_network_id(
        received_network_id: &str,
        expected_network_id: &str,
    ) -> Result<(), Status> {
        if received_network_id == expected_network_id {
            Ok(())
        } else {
            let nid_display = if received_network_id.is_empty() {
                "<empty>"
            } else {
                received_network_id
            };
            tracing::warn!("Wrong network id '{}'. Closing connection", nid_display);
            Err(Status::permission_denied(format!(
                "Wrong network id '{}'. This node runs on network '{}'",
                nid_display, expected_network_id
            )))
        }
    }

    /// Validate certificate chain using CertificateHelper
    fn validate_certificate_chain(
        header: &Header,
        peer_certificates: &[Vec<u8>],
    ) -> Result<(), Status> {
        if peer_certificates.is_empty() {
            tracing::warn!("No TLS Session. Closing connection");
            return Err(Status::unauthenticated("No TLS Session"));
        }

        let sender = header
            .sender
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("Header missing sender"))?;

        // Get the head certificate (first in chain)
        let peer_cert = &peer_certificates[0];
        let cert_bytes = peer_cert.as_slice(); // Get the DER bytes

        // Extract public key and calculate F1r3fly address
        let public_key = Self::extract_public_key_from_der(cert_bytes)?;
        let calculated_address =
            CertificateHelper::public_address(&public_key).ok_or_else(|| {
                tracing::warn!("Failed to calculate public address from certificate");
                Status::unauthenticated("Certificate verification failed")
            })?;

        // Compare with sender ID - ensure both are in bytes format
        let sender_id_bytes = sender.id.as_ref();

        if calculated_address == sender_id_bytes {
            tracing::debug!("Certificate verification successful for sender");
            Ok(())
        } else {
            tracing::warn!(
                "Certificate verification failed. Expected: {}, Got: {}",
                hex::encode(sender_id_bytes),
                hex::encode(&calculated_address)
            );
            Err(Status::unauthenticated("Certificate verification failed"))
        }
    }

    /// Extract secp256r1 public key from DER-encoded X.509 certificate
    fn extract_public_key_from_der(cert_bytes: &[u8]) -> Result<P256PublicKey, Status> {
        use p256::PublicKey;
        use x509_parser::prelude::*;

        // Parse certificate
        let (_, cert) = X509Certificate::from_der(cert_bytes).map_err(|e| {
            tracing::warn!("Failed to parse certificate: {}", e);
            Status::unauthenticated("Invalid certificate format")
        })?;

        // Extract public key data
        let public_key_info = cert.public_key();
        let public_key_bytes = &public_key_info.subject_public_key.data;

        // Validate secp256r1 uncompressed format (0x04 + 32-byte x + 32-byte y = 65 bytes)
        if public_key_bytes.len() == 65 && public_key_bytes[0] == 0x04 {
            PublicKey::from_sec1_bytes(public_key_bytes).map_err(|e| {
                tracing::warn!("Failed to parse secp256r1 public key: {}", e);
                Status::unauthenticated("Invalid public key format")
            })
        } else {
            tracing::warn!(
                "Unexpected public key format: {} bytes, prefix: {:02x}",
                public_key_bytes.len(),
                public_key_bytes.first().unwrap_or(&0)
            );
            Err(Status::unauthenticated("Invalid public key format"))
        }
    }

    /// Get the network ID for this interceptor
    pub fn network_id(&self) -> &str {
        &self.network_id
    }

    /// Validate a streaming request against TLS session context
    ///
    /// For streaming requests, we can only validate the TLS session itself since
    /// the request doesn't contain a TlRequest with protocol headers. The actual
    /// message validation happens when processing the stream chunks.
    ///
    /// # Arguments
    /// * `request` - The gRPC streaming request
    ///
    /// # Returns
    /// * `Ok(())` if TLS validation passes
    /// * `Err(Status)` with appropriate error code if validation fails
    pub fn validate_stream_request<T>(request: &Request<T>) -> Result<(), Status> {
        // Extract validation context from request extensions
        let validation_context = request
            .extensions()
            .get::<CertificateValidationContext>()
            .ok_or_else(|| {
                tracing::warn!("No certificate validation context found in streaming request");
                Status::internal("Missing certificate validation context")
            })?;

        // Check if TLS validation passed in interceptor phase
        if !validation_context.tls_validation_passed {
            tracing::warn!("TLS validation failed for streaming request");
            return Err(Status::unauthenticated("TLS validation failed"));
        }

        // For streaming requests, we only validate the TLS session
        // Message-level validation (network ID, sender verification) happens
        // when processing individual chunks in the stream
        tracing::debug!("TLS validation passed for streaming request");
        Ok(())
    }
}

impl Interceptor for SslSessionServerInterceptor {
    /// Implement tonic Interceptor trait
    ///
    /// Validates TLS certificates and stores validation context in request extensions
    /// for later use by the service implementation.
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        // Extract peer certificates from F1r3fly connection info
        let peer_certificates = if let Some(connect_info) =
            request
                .extensions()
                .get::<crate::rust::transport::f1r3fly_server::F1r3flyConnectInfo>()
        {
            connect_info.peer_certificates.clone()
        } else {
            // Fallback to standard tonic peer_certs() method for compatibility
            if let Some(certificates) = request.peer_certs() {
                if !certificates.is_empty() {
                    // Convert CertificateDer to Vec<u8>
                    let cert_bytes: Vec<Vec<u8>> = certificates
                        .iter()
                        .map(|cert| cert.as_ref().to_vec())
                        .collect();
                    Some(cert_bytes)
                } else {
                    None
                }
            } else {
                None
            }
        };

        // Determine if TLS validation passed
        let tls_validation_passed = if let Some(ref certs) = peer_certificates {
            if !certs.is_empty() {
                true
            } else {
                tracing::warn!("No TLS certificates found in session");
                false
            }
        } else {
            tracing::warn!("No TLS session found");
            false
        };

        // Create validation context
        let validation_context = CertificateValidationContext {
            peer_certificates,
            tls_validation_passed,
            network_id: self.network_id.clone(),
        };

        // Store validation context in request extensions for service to use
        request.extensions_mut().insert(validation_context);

        // Store network ID in extensions as well (for compatibility)
        request.extensions_mut().insert(self.network_id.clone());

        Ok(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use models::routing::{Node, Protocol};
    use prost::bytes::Bytes;
    use tonic::Code;

    fn create_test_header(network_id: &str, sender_id: Vec<u8>) -> Header {
        Header {
            sender: Some(Node {
                id: Bytes::from(sender_id),
                host: Bytes::from("127.0.0.1"),
                tcp_port: 8080,
                udp_port: 8080,
            }),
            network_id: network_id.to_string(),
        }
    }

    fn create_test_protocol(header: Header) -> Protocol {
        Protocol {
            header: Some(header),
            message: None, // Message would be set in real usage
        }
    }

    fn create_test_tl_request(protocol: Protocol) -> TlRequest {
        TlRequest {
            protocol: Some(protocol),
        }
    }

    #[test]
    fn test_interceptor_creation() {
        let network_id = "test_network".to_string();
        let interceptor = SslSessionServerInterceptor::new(network_id.clone());
        assert_eq!(interceptor.network_id, network_id);
    }

    #[test]
    fn test_validate_correct_network_id() {
        let result =
            SslSessionServerInterceptor::validate_network_id("test_network", "test_network");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_wrong_network_id() {
        let result =
            SslSessionServerInterceptor::validate_network_id("wrong_network", "correct_network");
        assert!(result.is_err());

        if let Err(status) = result {
            assert_eq!(status.code(), Code::PermissionDenied);
            assert!(status.message().contains("Wrong network id"));
        }
    }

    #[test]
    fn test_validate_empty_network_id() {
        let result = SslSessionServerInterceptor::validate_network_id("", "correct_network");
        assert!(result.is_err());

        if let Err(status) = result {
            assert_eq!(status.code(), Code::PermissionDenied);
            assert!(status.message().contains("<empty>"));
        }
    }

    #[test]
    fn test_validate_protocol_missing_header() {
        let protocol = Protocol {
            header: None,
            message: None,
        };

        let result = SslSessionServerInterceptor::validate_protocol_with_certificates(
            &protocol,
            "test_network",
            &None,
        );

        assert!(result.is_err());
        if let Err(status) = result {
            assert_eq!(status.code(), Code::InvalidArgument);
            assert!(status.message().contains("missing header"));
        }
    }

    #[test]
    fn test_validate_protocol_no_certificates() {
        let header = create_test_header("test_network", b"test_sender".to_vec());
        let protocol = create_test_protocol(header);

        let result = SslSessionServerInterceptor::validate_protocol_with_certificates(
            &protocol,
            "test_network",
            &None,
        );

        assert!(result.is_err());
        if let Err(status) = result {
            assert_eq!(status.code(), Code::Unauthenticated);
            assert!(status.message().contains("No TLS Session"));
        }
    }

    #[test]
    fn test_validate_tl_request_missing_protocol() {
        let tl_request = TlRequest { protocol: None };
        let mut request = Request::new(tl_request);

        // Add validation context
        let validation_context = CertificateValidationContext {
            peer_certificates: None,
            tls_validation_passed: true,
            network_id: "test_network".to_string(),
        };
        request.extensions_mut().insert(validation_context);

        let result = SslSessionServerInterceptor::validate_tl_request(&request);
        assert!(result.is_err());

        if let Err(status) = result {
            assert_eq!(status.code(), Code::InvalidArgument);
            assert!(status.message().contains("missing protocol"));
        }
    }

    #[test]
    fn test_validate_tl_request_missing_context() {
        let header = create_test_header("test_network", b"test_sender".to_vec());
        let protocol = create_test_protocol(header);
        let tl_request = create_test_tl_request(protocol);
        let request = Request::new(tl_request);

        let result = SslSessionServerInterceptor::validate_tl_request(&request);
        assert!(result.is_err());

        if let Err(status) = result {
            assert_eq!(status.code(), Code::Internal);
            assert!(status
                .message()
                .contains("Missing certificate validation context"));
        }
    }

    #[test]
    fn test_validate_tl_request_tls_validation_failed() {
        let header = create_test_header("test_network", b"test_sender".to_vec());
        let protocol = create_test_protocol(header);
        let tl_request = create_test_tl_request(protocol);
        let mut request = Request::new(tl_request);

        // Add validation context with failed TLS validation
        let validation_context = CertificateValidationContext {
            peer_certificates: None,
            tls_validation_passed: false,
            network_id: "test_network".to_string(),
        };
        request.extensions_mut().insert(validation_context);

        let result = SslSessionServerInterceptor::validate_tl_request(&request);
        assert!(result.is_err());

        if let Err(status) = result {
            assert_eq!(status.code(), Code::Unauthenticated);
            assert!(status.message().contains("TLS validation failed"));
        }
    }

    #[test]
    fn test_interceptor_call() {
        let mut interceptor = SslSessionServerInterceptor::new("test_network".to_string());
        let request = Request::new(());

        let result = interceptor.call(request);
        assert!(result.is_ok());

        if let Ok(req) = result {
            // Check that validation context was stored in extensions
            let validation_context = req.extensions().get::<CertificateValidationContext>();
            assert!(validation_context.is_some());

            if let Some(context) = validation_context {
                assert_eq!(context.network_id, "test_network");
                // TLS validation should fail since there are no certificates in test
                assert!(!context.tls_validation_passed);
            }

            // Check that network_id was also stored in extensions
            let network_id = req.extensions().get::<String>();
            assert!(network_id.is_some());
            assert_eq!(network_id.unwrap(), "test_network");
        }
    }

    #[test]
    fn test_network_id_accessor() {
        let network_id = "test_network";
        let interceptor = SslSessionServerInterceptor::new(network_id.to_string());
        assert_eq!(interceptor.network_id(), network_id);
    }
}
