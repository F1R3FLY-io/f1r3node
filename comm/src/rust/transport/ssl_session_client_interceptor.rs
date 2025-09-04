// See comm/src/main/scala/coop/rchain/comm/transport/SslSessionClientInterceptor.scala

use p256::PublicKey as P256PublicKey;
use tonic::service::Interceptor;
use tonic::{Request, Response, Status};

use crypto::rust::util::certificate_helper::CertificateHelper;
use hex;
use models::routing::{tl_response::Payload, Ack, Header, TlResponse};

/// SSL Session Client Interceptor for validating TLS sessions and certificates
///
/// This interceptor validates gRPC responses to ensure they come from trusted peers
/// in the correct network. It performs two main validations:
/// 1. Network ID validation - ensures responses are from the expected network
/// 2. Certificate validation - verifies the sender's identity using TLS certificates
///
#[derive(Clone, Debug)]
pub struct SslSessionClientInterceptor {
    network_id: String,
}

impl SslSessionClientInterceptor {
    /// Create a new SSL session client interceptor
    ///
    /// # Arguments
    /// * `network_id` - The expected network ID for validation
    #[inline]
    pub fn new(network_id: String) -> Self {
        Self { network_id }
    }

    /// Validate TLResponse message
    ///
    /// Validates the response
    /// - ACK responses: validate network ID and header structure
    /// - InternalServerError: pass through without validation
    /// - Other/None: reject as malformed
    pub fn validate_tl_response(&self, response: &TlResponse) -> Result<(), Status> {
        match &response.payload {
            Some(Payload::Ack(ack)) => self.validate_ack_response(ack),
            Some(Payload::InternalServerError(_)) => Ok(()), // Pass through
            _ => {
                log::warn!("Malformed response with no payload");
                Err(Status::invalid_argument("Malformed message"))
            }
        }
    }

    /// Validate TLResponse with TLS certificate verification
    ///
    /// Performs complete validation including certificate verification:
    /// - Network ID validation
    /// - TLS certificate extraction and verification
    /// - F1r3fly address calculation and comparison
    ///
    /// # Arguments
    /// * `response` - The TLResponse to validate
    /// * `peer_certificates` - DER-encoded peer certificates from TLS session
    pub fn validate_tl_response_with_tls(
        &self,
        response: &TlResponse,
        peer_certificates: &[Vec<u8>],
    ) -> Result<(), Status> {
        match &response.payload {
            Some(Payload::Ack(ack)) => self.validate_ack_with_certificates(ack, peer_certificates),
            Some(Payload::InternalServerError(_)) => Ok(()), // Pass through
            _ => {
                log::warn!("Malformed response with no payload");
                Err(Status::invalid_argument("Malformed message"))
            }
        }
    }

    /// Validate ACK response with basic checks (network ID and structure)
    #[inline]
    fn validate_ack_response(&self, ack: &Ack) -> Result<(), Status> {
        let header = ack
            .header
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("ACK missing header"))?;

        self.validate_network_id(&header.network_id)?;
        self.validate_header_structure(header)
    }

    /// Validate ACK response with full TLS certificate verification
    fn validate_ack_with_certificates(
        &self,
        ack: &Ack,
        peer_certificates: &[Vec<u8>],
    ) -> Result<(), Status> {
        let header = ack
            .header
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("ACK missing header"))?;

        self.validate_network_id(&header.network_id)?;
        self.validate_certificate_chain(header, peer_certificates)
    }

    /// Validate network ID matches expected value
    #[inline]
    fn validate_network_id(&self, received_network_id: &str) -> Result<(), Status> {
        if received_network_id == self.network_id {
            Ok(())
        } else {
            let nid_display = if received_network_id.is_empty() {
                "<empty>"
            } else {
                received_network_id
            };
            log::warn!("Wrong network id '{}'. Closing connection", nid_display);
            Err(Status::permission_denied(format!(
                "Wrong network id '{}'",
                nid_display
            )))
        }
    }

    /// Validate header has required sender field
    #[inline]
    fn validate_header_structure(&self, header: &Header) -> Result<(), Status> {
        match header.sender {
            Some(_) => Ok(()),
            None => Err(Status::invalid_argument("Header missing sender")),
        }
    }

    /// Validate certificate chain using CertificateHelper
    fn validate_certificate_chain(
        &self,
        header: &Header,
        peer_certificates: &[Vec<u8>],
    ) -> Result<(), Status> {
        // Check TLS session exists
        if peer_certificates.is_empty() {
            log::warn!("No TLS Session. Closing connection");
            return Err(Status::unauthenticated("No TLS Session"));
        }

        let sender = header
            .sender
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("Header missing sender"))?;

        // Get the head certificate
        let peer_cert_bytes = &peer_certificates[0];

        // Extract public key and calculate F1r3fly address
        let public_key = self.extract_public_key_from_der(peer_cert_bytes)?;
        let calculated_address =
            CertificateHelper::public_address(&public_key).ok_or_else(|| {
                log::warn!("Failed to calculate public address from certificate");
                Status::unauthenticated("Certificate verification failed")
            })?;

        // Compare with sender ID - ensure both are in bytes format
        let sender_id_bytes = sender.id.as_ref();

        log::debug!(
            "Certificate verification: calculated_address={}, sender_id_bytes={}, match={}",
            hex::encode(&calculated_address),
            hex::encode(sender_id_bytes),
            calculated_address == sender_id_bytes
        );

        if calculated_address == sender_id_bytes {
            log::debug!("Certificate verification successful for sender");
            Ok(())
        } else {
            log::warn!(
                "Certificate verification failed. Expected: {}, Got: {}",
                hex::encode(sender_id_bytes),
                hex::encode(&calculated_address)
            );
            Err(Status::unauthenticated("Certificate verification failed"))
        }
    }

    /// Extract secp256r1 public key from DER-encoded X.509 certificate
    fn extract_public_key_from_der(&self, cert_bytes: &[u8]) -> Result<P256PublicKey, Status> {
        use p256::PublicKey;
        use x509_parser::prelude::*;

        // Parse certificate
        let (_, cert) = X509Certificate::from_der(cert_bytes).map_err(|e| {
            log::warn!("Failed to parse certificate: {}", e);
            Status::unauthenticated("Invalid certificate format")
        })?;

        // Extract public key data
        let public_key_info = cert.public_key();
        let public_key_bytes = &public_key_info.subject_public_key.data;

        // Validate secp256r1 uncompressed format (0x04 + 32-byte x + 32-byte y = 65 bytes)
        if public_key_bytes.len() == 65 && public_key_bytes[0] == 0x04 {
            PublicKey::from_sec1_bytes(public_key_bytes).map_err(|e| {
                log::warn!("Failed to parse secp256r1 public key: {}", e);
                Status::unauthenticated("Invalid public key format")
            })
        } else {
            log::warn!(
                "Unexpected public key format: {} bytes, prefix: {:02x}",
                public_key_bytes.len(),
                public_key_bytes.first().unwrap_or(&0)
            );
            Err(Status::unauthenticated("Invalid public key format"))
        }
    }

    /// Process intercepted response (pass-through for tonic integration)
    #[inline]
    pub fn process_response<T>(
        &self,
        response: Result<Response<T>, Status>,
    ) -> Result<Response<T>, Status> {
        response
    }

    /// Get the network ID for this interceptor
    #[inline]
    pub fn network_id(&self) -> &str {
        &self.network_id
    }
}

impl Interceptor for SslSessionClientInterceptor {
    /// Implement tonic Interceptor trait
    ///
    /// Stores the network ID in request extensions for potential future use
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        request.extensions_mut().insert(self.network_id.clone());
        Ok(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use models::routing::{InternalServerError, Node};
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

    fn create_test_ack(header: Header) -> Ack {
        Ack {
            header: Some(header),
        }
    }

    fn create_test_tl_response(payload: Payload) -> TlResponse {
        TlResponse {
            payload: Some(payload),
        }
    }

    #[test]
    fn test_interceptor_creation() {
        let network_id = "test_network".to_string();
        let interceptor = SslSessionClientInterceptor::new(network_id.clone());
        assert_eq!(interceptor.network_id, network_id);
    }

    #[test]
    fn test_validate_correct_network_id() {
        let network_id = "test_network";
        let interceptor = SslSessionClientInterceptor::new(network_id.to_string());

        let header = create_test_header(network_id, b"test_sender".to_vec());
        let ack = create_test_ack(header);
        let response = create_test_tl_response(Payload::Ack(ack));

        let result = interceptor.validate_tl_response(&response);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_wrong_network_id() {
        let network_id = "correct_network";
        let interceptor = SslSessionClientInterceptor::new(network_id.to_string());

        let header = create_test_header("wrong_network", b"test_sender".to_vec());
        let ack = create_test_ack(header);
        let response = create_test_tl_response(Payload::Ack(ack));

        let result = interceptor.validate_tl_response(&response);
        assert!(result.is_err());

        if let Err(status) = result {
            assert_eq!(status.code(), Code::PermissionDenied);
            assert!(status.message().contains("Wrong network id"));
        }
    }

    #[test]
    fn test_validate_empty_network_id() {
        let network_id = "correct_network";
        let interceptor = SslSessionClientInterceptor::new(network_id.to_string());

        let header = create_test_header("", b"test_sender".to_vec());
        let ack = create_test_ack(header);
        let response = create_test_tl_response(Payload::Ack(ack));

        let result = interceptor.validate_tl_response(&response);
        assert!(result.is_err());

        if let Err(status) = result {
            assert_eq!(status.code(), Code::PermissionDenied);
            assert!(status.message().contains("<empty>"));
        }
    }

    #[test]
    fn test_validate_internal_server_error_passes_through() {
        let network_id = "test_network";
        let interceptor = SslSessionClientInterceptor::new(network_id.to_string());

        let internal_error = InternalServerError {
            error: Bytes::from("test error"),
        };
        let response = create_test_tl_response(Payload::InternalServerError(internal_error));

        let result = interceptor.validate_tl_response(&response);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_malformed_response() {
        let network_id = "test_network";
        let interceptor = SslSessionClientInterceptor::new(network_id.to_string());

        // Response with no payload
        let response = TlResponse { payload: None };

        let result = interceptor.validate_tl_response(&response);
        assert!(result.is_err());

        if let Err(status) = result {
            assert_eq!(status.code(), Code::InvalidArgument);
            assert!(status.message().contains("Malformed message"));
        }
    }

    #[test]
    fn test_validate_ack_missing_header() {
        let network_id = "test_network";
        let interceptor = SslSessionClientInterceptor::new(network_id.to_string());

        let ack = Ack { header: None };
        let response = create_test_tl_response(Payload::Ack(ack));

        let result = interceptor.validate_tl_response(&response);
        assert!(result.is_err());

        if let Err(status) = result {
            assert_eq!(status.code(), Code::InvalidArgument);
            assert!(status.message().contains("ACK missing header"));
        }
    }

    #[test]
    fn test_validate_with_tls_no_certificates() {
        let network_id = "test_network";
        let interceptor = SslSessionClientInterceptor::new(network_id.to_string());

        let header = create_test_header(network_id, b"test_sender".to_vec());
        let ack = create_test_ack(header);
        let response = create_test_tl_response(Payload::Ack(ack));

        // No certificates provided
        let peer_certificates: Vec<Vec<u8>> = vec![];
        let result = interceptor.validate_tl_response_with_tls(&response, &peer_certificates);

        assert!(result.is_err());
        if let Err(status) = result {
            assert_eq!(status.code(), Code::Unauthenticated);
            assert!(status.message().contains("No TLS Session"));
        }
    }

    #[test]
    fn test_validate_with_tls_invalid_certificate() {
        let network_id = "test_network";
        let interceptor = SslSessionClientInterceptor::new(network_id.to_string());

        let header = create_test_header(network_id, b"test_sender".to_vec());
        let ack = create_test_ack(header);
        let response = create_test_tl_response(Payload::Ack(ack));

        // Invalid certificate data
        let peer_certificates = vec![vec![0x00, 0x01, 0x02]];
        let result = interceptor.validate_tl_response_with_tls(&response, &peer_certificates);

        assert!(result.is_err());
        if let Err(status) = result {
            assert_eq!(status.code(), Code::Unauthenticated);
        }
    }

    #[test]
    fn test_process_response_passthrough() {
        let interceptor = SslSessionClientInterceptor::new("test".to_string());

        // Test successful response
        let ok_response: Result<Response<()>, Status> = Ok(Response::new(()));
        let result = interceptor.process_response(ok_response);
        assert!(result.is_ok());

        // Test error response
        let err_response: Result<Response<()>, Status> = Err(Status::internal("test error"));
        let result = interceptor.process_response(err_response);
        assert!(result.is_err());
    }

    #[test]
    fn test_interceptor_call() {
        let mut interceptor = SslSessionClientInterceptor::new("test_network".to_string());
        let request = Request::new(());

        let result = interceptor.call(request);
        assert!(result.is_ok());

        if let Ok(req) = result {
            // Check that network_id was stored in extensions
            let network_id = req.extensions().get::<String>();
            assert!(network_id.is_some());
            assert_eq!(network_id.unwrap(), "test_network");
        }
    }

    #[test]
    fn test_network_id_validation() {
        let interceptor = SslSessionClientInterceptor::new("expected_network".to_string());

        // Test correct network ID
        assert!(interceptor.validate_network_id("expected_network").is_ok());

        // Test wrong network ID
        let result = interceptor.validate_network_id("wrong_network");
        assert!(result.is_err());
        if let Err(status) = result {
            assert_eq!(status.code(), Code::PermissionDenied);
        }

        // Test empty network ID
        let result = interceptor.validate_network_id("");
        assert!(result.is_err());
        if let Err(status) = result {
            assert_eq!(status.code(), Code::PermissionDenied);
            assert!(status.message().contains("<empty>"));
        }
    }

    #[test]
    fn test_header_structure_validation() {
        let interceptor = SslSessionClientInterceptor::new("test".to_string());

        // Test valid header with sender
        let valid_header = create_test_header("test", b"sender".to_vec());
        assert!(interceptor.validate_header_structure(&valid_header).is_ok());

        // Test header without sender
        let invalid_header = Header {
            sender: None,
            network_id: "test".to_string(),
        };
        let result = interceptor.validate_header_structure(&invalid_header);
        assert!(result.is_err());
        if let Err(status) = result {
            assert_eq!(status.code(), Code::InvalidArgument);
            assert!(status.message().contains("Header missing sender"));
        }
    }

    #[test]
    fn test_validate_response_with_request_context_no_tls() {
        let network_id = "test_network";
        let interceptor = SslSessionClientInterceptor::new(network_id.to_string());

        let header = create_test_header(network_id, b"test_sender".to_vec());
        let ack = create_test_ack(header);
        let response = create_test_tl_response(Payload::Ack(ack));

        // Test basic validation without TLS certificates
        let result = interceptor.validate_tl_response(&response);

        // Should pass basic validation (network ID and header structure)
        assert!(result.is_ok());
    }

    #[test]
    fn test_network_id_accessor() {
        let network_id = "test_network";
        let interceptor = SslSessionClientInterceptor::new(network_id.to_string());

        assert_eq!(interceptor.network_id(), network_id);
    }
}
