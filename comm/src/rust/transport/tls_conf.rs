// See comm/src/main/scala/coop/rchain/comm/transport/TlsConf.scala

use std::path::PathBuf;

/// TLS configuration for the transport layer
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlsConf {
    pub certificate_path: PathBuf,
    pub key_path: PathBuf,
    pub secure_random_non_blocking: bool,
    pub custom_certificate_location: bool,
    pub custom_key_location: bool,
}

impl TlsConf {
    pub fn new(
        certificate_path: PathBuf,
        key_path: PathBuf,
        secure_random_non_blocking: bool,
        custom_certificate_location: bool,
        custom_key_location: bool,
    ) -> Self {
        Self {
            certificate_path,
            key_path,
            secure_random_non_blocking,
            custom_certificate_location,
            custom_key_location,
        }
    }
}
