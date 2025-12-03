// See node/src/main/scala/coop/rchain/node/NodeEnvironment.scala

use std::path::Path;

use crate::rust::configuration::model::{NodeConf, TlsConf};
use comm::rust::{peer_node::NodeIdentifier, transport::generate_certificate_if_absent};
use crypto::rust::util::certificate_helper::CertificateHelper;

use tokio::{fs, io};
use tracing::{error, info};

/// Create a NodeIdentifier from the given configuration
///
/// This function performs the following operations:
/// 1. Validates the data directory exists and is accessible
/// 2. Generates a certificate if absent
/// 3. Validates certificate and key files exist
/// 4. Extracts the node name from the certificate
pub async fn create(conf: &NodeConf) -> eyre::Result<NodeIdentifier> {
    let data_dir = &conf.storage.data_dir;

    can_create_data_dir(data_dir)?;
    have_access_to_data_dir(data_dir).await?;
    info!("Using data dir: {}", data_dir.display());

    generate_certificate_if_absent::create(&conf.tls.clone().into()).await?;

    has_certificate(&conf.tls)?;
    has_key(&conf.tls)?;

    let name = name(conf)?;
    Ok(NodeIdentifier::new(name))
}

fn is_valid(pred: bool, msg: &str) -> eyre::Result<()> {
    if pred {
        Err(eyre::eyre!(msg.to_string()))
    } else {
        Ok(())
    }
}

fn name(conf: &NodeConf) -> eyre::Result<String> {
    let certificate = CertificateHelper::from_file(&conf.tls.certificate_path.to_string_lossy())
        .map_err(|e| eyre::eyre!(format!("Failed to read the X.509 certificate: {}", e)))?;

    let name = certificate_public_key_to_node_name(certificate.public_key_data().to_vec())?;

    Ok(name)
}

fn certificate_public_key_to_node_name(public_key: Vec<u8>) -> eyre::Result<String> {
    let normalized = CertificateHelper::normalize_public_key_coordinates(public_key)?;
    let res = CertificateHelper::public_address_from_bytes(&normalized);
    Ok(hex::encode(res))
}

fn can_create_data_dir(data_dir: &Path) -> eyre::Result<()> {
    if !data_dir.exists() {
        std::fs::create_dir_all(data_dir)
            .map_err(|e| eyre::eyre!(format!(
                "The data dir must be a directory and have read and write permissions:\n{}\nError: {}",
                data_dir.display(), e
            )))?;
    }
    Ok(())
}

async fn have_access_to_data_dir(data_dir: &Path) -> eyre::Result<()> {
    is_valid(
        !dir_is_readable_and_writable(data_dir).await,
        &format!(
            "The data dir must be a directory and have read and write permissions:\n{}",
            data_dir.display()
        ),
    )
}

pub async fn dir_is_readable_and_writable(path: &Path) -> bool {
    // Check if path exists and is a directory
    match fs::metadata(path).await {
        Ok(meta) if meta.is_dir() => {}
        _ => return false,
    }

    if fs::read_dir(path).await.is_err() {
        return false;
    }

    let test_file = path.join(".perm_test");
    match fs::File::create(&test_file).await {
        Ok(_) => {
            let _ = fs::remove_file(&test_file).await;
            true
        }
        Err(err) => {
            if err.kind() == io::ErrorKind::PermissionDenied {
                error!("Write permission denied: {}", err);
            } else {
                error!("Failed to create test file: {}", err);
            }
            false
        }
    }
}

fn has_certificate(tls: &TlsConf) -> eyre::Result<()> {
    is_valid(
        !tls.certificate_path.exists(),
        &format!(
            "Certificate file {} not found",
            tls.certificate_path.display()
        ),
    )
}

fn has_key(tls: &TlsConf) -> eyre::Result<()> {
    is_valid(
        !tls.key_path.exists(),
        &format!("Secret key file {} not found", tls.key_path.display()),
    )
}
