use crate::rust::diagnostics::prometheus_config::PrometheusConfiguration;
use eyre::Result;
use metrics_exporter_prometheus::PrometheusHandle;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use tracing::{info, warn};

static GLOBAL_REPORTER: OnceLock<Arc<NewPrometheusReporter>> = OnceLock::new();

pub struct NewPrometheusReporter {
    #[allow(dead_code)]
    config: PrometheusConfiguration,
    #[allow(dead_code)]
    environment_tags: HashMap<String, String>,
    prometheus_handle: PrometheusHandle,
}

impl NewPrometheusReporter {
    pub fn initialize() -> Result<Arc<Self>> {
        let config = PrometheusConfiguration::default();
        Self::initialize_with_config(config)
    }

    pub fn initialize_with_config(config: PrometheusConfiguration) -> Result<Arc<Self>> {
        if let Some(reporter) = GLOBAL_REPORTER.get() {
            return Ok(Arc::clone(reporter));
        }

        let prometheus_builder = metrics_exporter_prometheus::PrometheusBuilder::new();
        let recorder = prometheus_builder
            .set_buckets(&config.default_buckets)?
            .build_recorder();

        let handle = recorder.handle();

        if let Err(e) = metrics::set_global_recorder(recorder) {
            warn!("Failed to set global metrics recorder: {}", e);
        }

        info!("Prometheus metrics exporter initialized");

        let environment_tags = config.environment_tags();

        let reporter = Arc::new(Self {
            config,
            environment_tags,
            prometheus_handle: handle,
        });

        if let Err(_) = GLOBAL_REPORTER.set(Arc::clone(&reporter)) {
            warn!("Failed to set global Prometheus reporter (already set)");
        }

        Ok(reporter)
    }

    pub fn global() -> Option<Arc<Self>> {
        GLOBAL_REPORTER.get().map(Arc::clone)
    }

    pub fn scrape_data(&self) -> String {
        self.prometheus_handle.render()
    }
}
