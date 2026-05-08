pub mod batch_influx_db_reporter;
pub mod new_prometheus_reporter;
pub mod prometheus_config;
pub mod sigar_reporter;
pub mod udp_influx_db_reporter;
pub mod zipkin_reporter;

#[cfg(test)]
mod tests;

use crate::rust::configuration::NodeConf;
use eyre::Result;
use std::sync::Arc;
use tracing::{info, warn};

pub const SYSTEM_METRICS_SOURCE: &str = "f1r3fly.system";

pub fn initialize_diagnostics(
    conf: &NodeConf,
) -> Result<Option<Arc<new_prometheus_reporter::NewPrometheusReporter>>> {
    let metrics_interval = conf.metrics.tick_interval;

    let prometheus_reporter = if conf.metrics.prometheus {
        let reporter = new_prometheus_reporter::NewPrometheusReporter::initialize()?;
        Some(reporter)
    } else {
        None
    };

    if conf.metrics.influxdb {
        let influx = &conf.metrics.influxdb_endpoint;
        if let Err(e) = batch_influx_db_reporter::create_batch_influx_db_reporter(
            influx.hostname.clone(),
            influx.port,
            influx.database.clone(),
            influx.protocol.clone(),
            influx.user.clone(),
            influx.password.clone(),
            metrics_interval,
        ) {
            warn!("Failed to initialize InfluxDB HTTP reporter: {}", e);
        }
    }

    if conf.metrics.influxdb_udp {
        let influx = &conf.metrics.influxdb_endpoint;
        if let Err(e) = udp_influx_db_reporter::create_udp_influx_db_reporter(
            influx.hostname.clone(),
            influx.port,
            metrics_interval,
        ) {
            warn!("Failed to initialize InfluxDB UDP reporter: {}", e);
        }
    }

    if conf.metrics.zipkin {
        match zipkin_reporter::create_zipkin_reporter() {
            Ok(_) => info!("Zipkin reporter initialized successfully."),
            Err(e) => warn!("Failed to initialize Zipkin reporter: {}", e),
        }
    }

    if conf.metrics.sigar {
        sigar_reporter::start_sigar_reporter(metrics_interval);
        info!(
            "Sigar (system metrics) reporter started with interval {:?}.",
            metrics_interval
        );
    }

    Ok(prometheus_reporter)
}
