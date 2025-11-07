use crate::rust::diagnostics::prometheus_config::PrometheusConfiguration;
use eyre::Result;
use metrics_exporter_prometheus::PrometheusHandle;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

pub struct NewPrometheusReporter {
    #[allow(dead_code)]
    config: PrometheusConfiguration,
    #[allow(dead_code)]
    environment_tags: HashMap<String, String>,
    prometheus_handle: PrometheusHandle,
}

impl NewPrometheusReporter {
    pub fn initialize() -> Result<(Arc<Self>, PrometheusHandle)> {
        let config = PrometheusConfiguration::default();
        Self::initialize_with_config(config)
    }

    pub fn initialize_with_config(
        config: PrometheusConfiguration,
    ) -> Result<(Arc<Self>, PrometheusHandle)> {
        let prometheus_builder = metrics_exporter_prometheus::PrometheusBuilder::new();
        let recorder = prometheus_builder
            .set_buckets(&config.default_buckets)?
            .build_recorder();

        let handle = recorder.handle();

        metrics::set_global_recorder(recorder)
            .map_err(|e| eyre::eyre!("Failed to install Prometheus recorder: {}", e))?;

        info!("Prometheus metrics exporter initialized");

        let environment_tags = config.environment_tags();

        let reporter = Arc::new(Self {
            config,
            environment_tags,
            prometheus_handle: handle.clone(),
        });

        Ok((reporter, handle))
    }

    pub fn scrape_data(&self) -> String {
        self.prometheus_handle.render()
    }
}
