// comm/src/main/scala/coop/rchain/comm/transport/HostnameTrustManagerFactory.scala

use crypto::rust::util::certificate_helper::CertificateHelper;
use p256::elliptic_curve::sec1::FromEncodedPoint;
use p256::pkcs8::{DecodePrivateKey, EncodePrivateKey};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::PrivatePkcs8KeyDer;
use rustls::pki_types::{CertificateDer, ServerName};
use rustls::{DigitallySignedStruct, DistinguishedName, Error as RustlsError, SignatureScheme};
use std::sync::Arc;

/// Custom error type for certificate validation
#[derive(Debug, Clone)]
pub enum CertificateValidationError {
    ValidationFailed(String),
    NotAllowedMethod,
    NoHandshakeSession,
    WrongAlgorithm,
    AddressHostnameMismatch,
    NoEndpointIdentification,
    UnknownIdentificationAlgorithm(String),
    ParsingError(String),
}

impl std::fmt::Display for CertificateValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ValidationFailed(msg) => write!(f, "Certificate validation failed: {}", msg),
            Self::NotAllowedMethod => write!(f, "Not allowed validation method"),
            Self::NoHandshakeSession => write!(f, "No handshake session"),
            Self::WrongAlgorithm => write!(f, "Certificate's public key has the wrong algorithm"),
            Self::AddressHostnameMismatch => {
                write!(f, "Certificate's public address doesn't match the hostname")
            }
            Self::NoEndpointIdentification => write!(f, "No endpoint identification algorithm"),
            Self::UnknownIdentificationAlgorithm(alg) => {
                write!(f, "Unknown identification algorithm: {}", alg)
            }
            Self::ParsingError(msg) => write!(f, "Certificate parsing error: {}", msg),
        }
    }
}

impl std::error::Error for CertificateValidationError {}

/// HostnameTrustManagerFactory - creates custom certificate verifiers for F1r3fly
pub struct HostnameTrustManagerFactory;

impl HostnameTrustManagerFactory {
    /// Get the singleton instance of HostnameTrustManagerFactory
    pub fn instance() -> &'static Self {
        static INSTANCE: std::sync::OnceLock<HostnameTrustManagerFactory> =
            std::sync::OnceLock::new();
        INSTANCE.get_or_init(|| {
            // Initialize rustls crypto provider once
            let _ = rustls::crypto::ring::default_provider().install_default();

            HostnameTrustManagerFactory {
                // Empty for now - singleton instance  
            }
        })
    }

    /// Create a hostname trust manager
    pub fn create_trust_manager(&self) -> Arc<HostnameTrustManager> {
        Arc::new(HostnameTrustManager::new())
    }

    /// Create a client certificate verifier
    ///
    /// This creates an F1r3flyClientCertVerifier that uses the same HostnameTrustManager
    /// logic for client certificate validation on the server side.
    pub fn create_client_cert_verifier(&self) -> Arc<F1r3flyClientCertVerifier> {
        let trust_manager = self.create_trust_manager();
        Arc::new(F1r3flyClientCertVerifier::with_trust_manager(trust_manager))
    }

    /// Create a rustls ClientConfig with F1r3fly's custom certificate verification
    ///
    /// # Arguments
    /// * `cert_pem` - Client certificate in PEM format
    /// * `key_pem` - Client private key in PEM format
    ///
    /// # Returns
    /// * `Ok(ClientConfig)` with F1r3fly certificate verification
    /// * `Err(CertificateValidationError)` if configuration fails
    pub fn client_config(
        cert_pem: &str,
        key_pem: &str,
    ) -> Result<rustls::ClientConfig, CertificateValidationError> {
        // Parse certificate and key
        let cert_der = Self::parse_pem_certificate(cert_pem)?;
        let key_der = Self::parse_pem_private_key(key_pem)?;

        // Create custom certificate verifier
        let cert_verifier = HostnameTrustManagerFactory::instance().create_trust_manager();

        // Build client configuration
        let config = rustls::ClientConfig::builder_with_provider(
            rustls::crypto::ring::default_provider().into(),
        )
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|e| {
            CertificateValidationError::ValidationFailed(format!(
                "Failed to create client config builder: {}",
                e
            ))
        })?
        .dangerous() // Enter dangerous mode to use custom verifier
        .with_custom_certificate_verifier(cert_verifier)
        .with_client_auth_cert(vec![cert_der], key_der)
        .map_err(|e| {
            CertificateValidationError::ValidationFailed(format!(
                "Failed to configure client certificate: {}",
                e
            ))
        })?;

        Ok(config)
    }

    /// Create a rustls ServerConfig with F1r3fly's custom client certificate verification
    ///
    /// # Arguments
    /// * `cert_pem` - Server certificate in PEM format
    /// * `key_pem` - Server private key in PEM format
    ///
    /// # Returns
    /// * `Ok(ServerConfig)` with F1r3fly client certificate verification
    /// * `Err(CertificateValidationError)` if configuration fails
    pub fn server_config(
        cert_pem: &str,
        key_pem: &str,
    ) -> Result<rustls::ServerConfig, CertificateValidationError> {
        // Parse certificate and key
        let cert_der = Self::parse_pem_certificate(cert_pem)?;
        let key_der = Self::parse_pem_private_key(key_pem)?;

        // Create custom client certificate verifier
        let client_cert_verifier =
            HostnameTrustManagerFactory::instance().create_client_cert_verifier();

        // Build server configuration
        let config = rustls::ServerConfig::builder_with_provider(
            rustls::crypto::ring::default_provider().into(),
        )
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|e| {
            CertificateValidationError::ValidationFailed(format!(
                "Failed to create server config builder: {}",
                e
            ))
        })?
        .with_client_cert_verifier(client_cert_verifier)
        .with_single_cert(vec![cert_der], key_der)
        .map_err(|e| {
            CertificateValidationError::ValidationFailed(format!(
                "Failed to configure server certificate: {}",
                e
            ))
        })?;

        Ok(config)
    }

    /// Parse a PEM certificate into DER format
    fn parse_pem_certificate(
        pem_data: &str,
    ) -> Result<CertificateDer<'static>, CertificateValidationError> {
        use rustls_pemfile::Item;

        let mut cursor = std::io::Cursor::new(pem_data.as_bytes());

        // rustls_pemfile::read_all returns an iterator, so we need to collect and handle errors
        for item_result in rustls_pemfile::read_all(&mut cursor) {
            let item = item_result.map_err(|e| {
                CertificateValidationError::ParsingError(format!("Failed to parse PEM: {}", e))
            })?;

            if let Item::X509Certificate(cert_der) = item {
                return Ok(cert_der);
            }
        }

        Err(CertificateValidationError::ParsingError(
            "No certificate found in PEM data".to_string(),
        ))
    }

    /// Parse a PEM private key into DER format
    fn parse_pem_private_key(
        pem_data: &str,
    ) -> Result<rustls::pki_types::PrivateKeyDer<'static>, CertificateValidationError> {
        use rustls_pemfile::Item;

        let mut cursor = std::io::Cursor::new(pem_data);

        if let Some(item) = rustls_pemfile::read_one(&mut cursor).unwrap() {
            match item {
                Item::Pkcs1Key(key_der) => {
                    return Ok(rustls::pki_types::PrivateKeyDer::Pkcs1(key_der));
                }
                Item::Pkcs8Key(key_der) => {
                    let pkcs8_bytes = key_der.secret_pkcs8_der();

                    let secret_key: p256::SecretKey =
                        p256::SecretKey::from_pkcs8_der(&pkcs8_bytes).unwrap();

                    // Re-encode using p256's PKCS#8 encoding to get rustls-compatible format
                    let rustls_compatible_pkcs8 = secret_key.to_pkcs8_der().unwrap();

                    // Convert to PrivatePkcs8KeyDer
                    let private_key_der: PrivatePkcs8KeyDer =
                        rustls_compatible_pkcs8.as_bytes().to_vec().into();

                    return Ok(rustls::pki_types::PrivateKeyDer::Pkcs8(private_key_der));
                }
                Item::Sec1Key(key_der) => {
                    tracing::info!("Parsing item 3: {:?}", key_der);
                    return Ok(rustls::pki_types::PrivateKeyDer::Sec1(key_der));
                }
                _ => {
                    return Err(CertificateValidationError::ParsingError(format!(
                        "Unknown key format: {:?}",
                        item
                    )))
                }
            }
        }

        Err(CertificateValidationError::ParsingError(
            "No private key found in PEM data".to_string(),
        ))
    }
}

/// HostnameTrustManager - implements F1r3fly's custom certificate validation logic
/// Directly implements rustls ServerCertVerifier for seamless integration
#[derive(Debug)]
pub struct HostnameTrustManager;

impl HostnameTrustManager {
    pub fn new() -> Self {
        Self
    }

    /// Check client certificate trust
    pub fn check_client_trusted(
        &self,
        cert_der: &[u8],
        _auth_type: &str,
    ) -> Result<String, CertificateValidationError> {
        // Extract public key from certificate
        let public_key = self.extract_public_key_from_cert(cert_der)?;

        // Extract F1r3fly address
        let f1r3fly_address = CertificateHelper::public_address(&public_key)
            .map(|addr| hex::encode(&addr))
            .ok_or(CertificateValidationError::WrongAlgorithm)?;

        // Perform identity check with F1r3fly address as hostname
        self.check_identity(Some(&f1r3fly_address), cert_der, "https")?;

        Ok(f1r3fly_address)
    }

    /// Check server certificate trust  
    pub fn check_server_trusted(
        &self,
        cert_der: &[u8],
        _auth_type: &str,
        peer_hostname: Option<&str>,
    ) -> Result<(), CertificateValidationError> {
        // Standard hostname identity verification
        self.check_identity(peer_hostname, cert_der, "https")?;

        // Extract F1r3fly address from certificate
        let public_key = self.extract_public_key_from_cert(cert_der)?;
        let f1r3fly_address = CertificateHelper::public_address(&public_key)
            .map(|addr| hex::encode(&addr))
            .ok_or(CertificateValidationError::WrongAlgorithm)?;

        // Verify F1r3fly address matches hostname
        let peer_host = peer_hostname.unwrap_or("");
        if f1r3fly_address != peer_host {
            return Err(CertificateValidationError::AddressHostnameMismatch);
        }

        Ok(())
    }

    /// Extract secp256r1 public key from DER-encoded X.509 certificate
    fn extract_public_key_from_cert(
        &self,
        cert_der: &[u8],
    ) -> Result<p256::PublicKey, CertificateValidationError> {
        // Parse certificate
        let (_, cert) = x509_parser::parse_x509_certificate(cert_der)
            .map_err(|e| CertificateValidationError::ParsingError(format!("Invalid DER: {}", e)))?;

        // Extract public key bytes
        let public_key_info = cert.public_key();
        let public_key_bytes = public_key_info.subject_public_key.data.as_ref();

        // Convert to p256::PublicKey
        if public_key_bytes.len() == 65 && public_key_bytes[0] == 0x04 {
            // Uncompressed secp256r1 point format
            let encoded_point = p256::EncodedPoint::from_bytes(public_key_bytes)
                .map_err(|_| CertificateValidationError::WrongAlgorithm)?;

            let public_key = p256::PublicKey::from_encoded_point(&encoded_point);
            if public_key.is_some().into() {
                Ok(public_key.unwrap())
            } else {
                Err(CertificateValidationError::WrongAlgorithm)
            }
        } else {
            Err(CertificateValidationError::WrongAlgorithm)
        }
    }

    /// Perform hostname verification
    fn check_identity(
        &self,
        hostname: Option<&str>,
        cert_der: &[u8],
        algorithm: &str,
    ) -> Result<(), CertificateValidationError> {
        match algorithm.to_lowercase().as_str() {
            "https" => {
                if let Some(host) = hostname {
                    // Parse certificate to verify hostname against CN and SAN
                    let (_, cert) = x509_parser::parse_x509_certificate(cert_der).map_err(|e| {
                        tracing::warn!("Certificate parsing failed: {}", e);
                        CertificateValidationError::ParsingError(format!(
                            "Invalid certificate: {}",
                            e
                        ))
                    })?;

                    // Check CN field for hostname match
                    for attribute in cert.subject().iter() {
                        for name in attribute.iter() {
                            if name.attr_type() == &x509_parser::oid_registry::OID_X509_COMMON_NAME
                            {
                                if let Ok(cn_str) = name.attr_value().as_str() {
                                    if cn_str == host {
                                        return Ok(());
                                    }
                                }
                            }
                        }
                    }

                    // Check SAN (Subject Alternative Names) for hostname match
                    for extension in cert.extensions() {
                        if extension.oid == x509_parser::oid_registry::OID_X509_EXT_SUBJECT_ALT_NAME
                        {
                            if let x509_parser::extensions::ParsedExtension::SubjectAlternativeName(san) = &extension.parsed_extension() {
                                for name in &san.general_names {
                                    match name {
                                        x509_parser::extensions::GeneralName::DNSName(dns_name) => {
                                            if dns_name == &host {
                                                return Ok(());
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }

                    tracing::warn!(
                        "Hostname verification failed: '{}' not found in certificate CN or SAN",
                        host
                    );
                    Err(CertificateValidationError::ValidationFailed(format!(
                        "Hostname '{}' not found in certificate CN or SAN",
                        host
                    )))
                } else {
                    tracing::warn!("No hostname provided for verification");
                    Err(CertificateValidationError::ValidationFailed(
                        "No hostname provided for verification".to_string(),
                    ))
                }
            }
            _ => {
                tracing::warn!("Unsupported algorithm: {}", algorithm);
                Err(CertificateValidationError::UnknownIdentificationAlgorithm(
                    algorithm.to_string(),
                ))
            }
        }
    }

    /// Get accepted issuers
    pub fn get_accepted_issuers(&self) -> Vec<Vec<u8>> {
        // Return empty list - we use custom validation, not CA-based
        Vec::new()
    }
}

/// Implement rustls ServerCertVerifier directly
impl ServerCertVerifier for HostnameTrustManager {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<ServerCertVerified, RustlsError> {
        // Convert server_name to hostname string for our existing validation logic
        let hostname = match server_name {
            ServerName::DnsName(dns_name) => Some(dns_name.as_ref()),
            ServerName::IpAddress(ip) => {
                // Convert IP to string representation
                let ip_str = format!("{:?}", ip);
                Some(ip_str.leak() as &str) // Leak string to get 'static lifetime
            }
            _ => None,
        };

        // Use our existing server trust validation logic
        self.check_server_trusted(end_entity.as_ref(), "RSA", hostname)
            .map_err(|_| {
                RustlsError::InvalidCertificate(
                    rustls::CertificateError::ApplicationVerificationFailure,
                )
            })?;

        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, RustlsError> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, RustlsError> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![
            SignatureScheme::ECDSA_NISTP256_SHA256,
            SignatureScheme::RSA_PSS_SHA256,
            SignatureScheme::RSA_PKCS1_SHA256,
        ]
    }
}

/// Custom ClientCertVerifier that uses HostnameTrustManager for F1r3fly client certificate validation
///
/// This implements the server-side client certificate verification using the same HostnameTrustManager
/// logic that's used for server certificate verification on the client side.
#[derive(Debug)]
pub struct F1r3flyClientCertVerifier {
    trust_manager: Arc<HostnameTrustManager>,
}

impl F1r3flyClientCertVerifier {
    pub fn new() -> Self {
        let trust_manager = HostnameTrustManagerFactory::instance().create_trust_manager();
        Self { trust_manager }
    }

    /// Create with existing trust manager
    ///
    /// This allows creating the verifier with a pre-existing trust manager,
    /// following the same pattern as the client side SSL context creation.
    pub fn with_trust_manager(trust_manager: Arc<HostnameTrustManager>) -> Self {
        Self { trust_manager }
    }
}

impl rustls::server::danger::ClientCertVerifier for F1r3flyClientCertVerifier {
    fn verify_client_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::server::danger::ClientCertVerified, RustlsError> {
        // Use HostnameTrustManager's client certificate validation
        match self
            .trust_manager
            .check_client_trusted(end_entity.as_ref(), "RSA")
        {
            Ok(_f1r3fly_address) => {
                // Client certificate is valid
                Ok(rustls::server::danger::ClientCertVerified::assertion())
            }
            Err(_validation_error) => {
                // Client certificate validation failed
                Err(RustlsError::InvalidCertificate(
                    rustls::CertificateError::ApplicationVerificationFailure,
                ))
            }
        }
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, RustlsError> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, RustlsError> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![
            SignatureScheme::ECDSA_NISTP256_SHA256,
            SignatureScheme::RSA_PSS_SHA256,
            SignatureScheme::RSA_PKCS1_SHA256,
        ]
    }

    fn client_auth_mandatory(&self) -> bool {
        // Require client certificates (equivalent to ClientAuth.REQUIRE)
        true
    }

    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        // Return empty list - we use custom F1r3fly validation, not CA-based
        &[]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hostname_trust_manager_factory_singleton() {
        let factory1 = HostnameTrustManagerFactory::instance();
        let factory2 = HostnameTrustManagerFactory::instance();

        // Should be the same instance (singleton pattern)
        assert!(std::ptr::eq(factory1, factory2));
    }

    #[test]
    fn test_create_trust_manager() {
        let factory = HostnameTrustManagerFactory::instance();
        let _trust_manager = factory.create_trust_manager();

        // Should create successfully without panicking
    }

    #[test]
    fn test_client_trusted_validation() {
        let manager = HostnameTrustManager::new();

        // Generate test certificate
        let (secret_key, public_key) = CertificateHelper::generate_key_pair(false);

        if let Ok(cert_der) = CertificateHelper::generate_certificate(&secret_key, &public_key) {
            // Test client certificate validation
            let result = manager.check_client_trusted(&cert_der, "RSA");
            assert!(
                result.is_ok(),
                "Should validate client certificate successfully"
            );
        }
    }

    #[test]
    fn test_server_trusted_validation() {
        let manager = HostnameTrustManager::new();

        // Generate test certificate
        let (secret_key, public_key) = CertificateHelper::generate_key_pair(false);

        if let (Ok(cert_der), Some(expected_address)) = (
            CertificateHelper::generate_certificate(&secret_key, &public_key),
            CertificateHelper::public_address(&public_key),
        ) {
            let expected_hex = hex::encode(&expected_address);

            // Test server certificate validation with F1r3fly address as hostname (should pass)
            let result = manager.check_server_trusted(&cert_der, "RSA", Some(&expected_hex));
            assert!(
                result.is_ok(),
                "Should validate server certificate when hostname matches F1r3fly address"
            );

            // Test server certificate validation with wrong hostname (should fail)
            let result = manager.check_server_trusted(&cert_der, "RSA", Some("wrong_hostname"));
            assert!(
                result.is_err(),
                "Should reject server certificate when hostname doesn't match certificate CN/SAN"
            );

            // Verify the error is ValidationFailed (from hostname verification, not address mismatch)
            if let Err(error) = result {
                match error {
                    CertificateValidationError::AddressHostnameMismatch => {
                        // Expected - hostname doesn't match F1r3fly address
                    }
                    CertificateValidationError::ValidationFailed(msg) => {
                        assert!(msg.contains("not found in certificate"));
                    }
                    _ => panic!(
                        "Expected AddressHostnameMismatch or ValidationFailed error, got: {:?}",
                        error
                    ),
                }
            }
        }
    }
}
