// See comm/src/main/scala/coop/rchain/comm/transport/GenerateCertificateIfAbsent.scala

use crypto::rust::util::certificate_helper::{
    CertificateError, CertificateHelper, CertificatePrinter,
};
use p256::{PublicKey as P256PublicKey, SecretKey as P256SecretKey};
use tokio::fs;

use crate::rust::transport::tls_conf::TlsConf;

use log::info;

/// Generate certificate if absent
///
/// This function checks if a certificate exists at the configured path.
/// If not, and customCertificateLocation is false, it generates a new certificate.
/// If a key already exists, it uses that; otherwise, it generates a new key pair.
pub async fn create(tls: &TlsConf) -> Result<(), CertificateError> {
    // Generate certificate if not provided as option or in the data dir
    if !tls.custom_certificate_location && !tls.certificate_path.exists() {
        generate_certificate(tls).await?;
    }
    Ok(())
}

async fn generate_certificate(tls: &TlsConf) -> Result<(), CertificateError> {
    info!(
        "No certificate found at path {}",
        tls.certificate_path.display()
    );
    info!("Generating a X.509 certificate for the node");

    // If there is a private key, use it for the certificate
    if tls.key_path.exists() {
        read_key_pair(tls).await?;
    } else {
        generate_secret_key(tls).await?;
    }

    Ok(())
}

async fn read_key_pair(tls: &TlsConf) -> Result<(), CertificateError> {
    info!("Using secret key {}", tls.key_path.display());

    let key_path_str = tls.key_path.to_string_lossy();
    let (secret_key, public_key) = CertificateHelper::read_key_pair(&key_path_str)?;

    write_cert(tls, &secret_key, &public_key).await?;

    Ok(())
}

async fn generate_secret_key(tls: &TlsConf) -> Result<(), CertificateError> {
    info!("Generating a PEM secret key for the node");

    let (secret_key, public_key) =
        CertificateHelper::generate_key_pair(tls.secure_random_non_blocking);

    write_cert(tls, &secret_key, &public_key).await?;
    write_key(tls, &secret_key).await?;

    Ok(())
}

async fn write_cert(
    tls: &TlsConf,
    secret_key: &P256SecretKey,
    public_key: &P256PublicKey,
) -> Result<(), CertificateError> {
    let cert_der = CertificateHelper::generate_certificate(secret_key, public_key)?;

    let cert_pem = CertificatePrinter::print_certificate(&cert_der);

    if let Some(parent) = tls.certificate_path.parent() {
        fs::create_dir_all(parent).await.map_err(|e| {
            CertificateError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create certificate directory: {}", e),
            ))
        })?;
    }

    fs::write(&tls.certificate_path, cert_pem)
        .await
        .map_err(|e| {
            CertificateError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to write certificate: {}", e),
            ))
        })?;

    Ok(())
}

async fn write_key(tls: &TlsConf, secret_key: &P256SecretKey) -> Result<(), CertificateError> {
    let key_pem = CertificatePrinter::print_private_key_from_secret(secret_key)?;

    if let Some(parent) = tls.key_path.parent() {
        fs::create_dir_all(parent).await.map_err(|e| {
            CertificateError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create key directory: {}", e),
            ))
        })?;
    }

    fs::write(&tls.key_path, key_pem).await.map_err(|e| {
        CertificateError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to write key: {}", e),
        ))
    })?;

    Ok(())
}
