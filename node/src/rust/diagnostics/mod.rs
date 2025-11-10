pub mod batch_influx_db_reporter;
pub mod new_prometheus_reporter;
pub mod prometheus_config;
pub mod sigar_reporter;
pub mod udp_influx_db_reporter;
pub mod zipkin_reporter;

#[cfg(test)]
mod tests;

use crate::rust::configuration::{KamonConf, NodeConf};
use eyre::Result;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

pub const SYSTEM_METRICS_SOURCE: &str = "f1r3fly.system";

pub fn initialize_diagnostics(conf: &NodeConf, kamon_conf: &KamonConf) -> Result<Option<Arc<new_prometheus_reporter::NewPrometheusReporter>>> {
    let prometheus_reporter = if conf.metrics.prometheus {
        let reporter = new_prometheus_reporter::NewPrometheusReporter::initialize()?;
        Some(reporter)
    } else {
        None
    };

    if conf.metrics.influxdb {
        if let Some(influx_conf) = &kamon_conf.influxdb {
            let host = influx_conf.hostname.clone().unwrap_or_default();
            let port = influx_conf.port.unwrap_or(8089);
            let database = influx_conf.database.clone().unwrap_or_default();
            let protocol = influx_conf
                .protocol
                .clone()
                .unwrap_or_else(|| "http".to_string());
            let (username, password) = influx_conf
                .authentication
                .as_ref()
                .map_or((None, None), |auth| (auth.user.clone(), auth.password.clone()));
            if let Err(e) =
                batch_influx_db_reporter::create_batch_influx_db_reporter(
                    host,
                    port,
                    database,
                    protocol,
                    username,
                    password,
                    Duration::from_secs(10), // Default interval
                )
            {
                warn!("Failed to initialize InfluxDB HTTP reporter: {}", e);
            }
        }
    }

    if conf.metrics.influxdb_udp {
        if let Some(influx_conf) = &kamon_conf.influxdb {
            let host = influx_conf.hostname.clone().unwrap_or_default();
            let port = influx_conf.port.unwrap_or(8089);
            if let Err(e) = udp_influx_db_reporter::create_udp_influx_db_reporter(
                host,
                port,
                Duration::from_secs(10), // Default interval
            ) {
                warn!("Failed to initialize InfluxDB UDP reporter: {}", e);
            }
        }
    }

    if conf.metrics.zipkin {
        match zipkin_reporter::create_zipkin_reporter() {
            Ok(_) => info!("Zipkin reporter initialized successfully."),
            Err(e) => warn!("Failed to initialize Zipkin reporter: {}", e),
        }
    }

    if conf.metrics.sigar {
        sigar_reporter::start_sigar_reporter();
        info!("Sigar (system metrics) reporter started.");
    }

    Ok(prometheus_reporter)
}
