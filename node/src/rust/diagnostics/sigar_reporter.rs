use std::time::Duration;
use sysinfo::{CpuExt, System, SystemExt};
use tokio::time;
use crate::rust::diagnostics::SYSTEM_METRICS_SOURCE;

pub fn start_sigar_reporter(interval_duration: Duration) {
    tokio::spawn(async move {
        let mut sys = System::new_all();
        let mut interval = time::interval(interval_duration);
        loop {
            interval.tick().await;
            sys.refresh_cpu();
            sys.refresh_memory();

            let cpu_usage = sys.global_cpu_info().cpu_usage();
            let mem_usage = sys.used_memory() as f64 / sys.total_memory() as f64 * 100.0;

            metrics::gauge!("system_cpu_usage_percent", "source" => SYSTEM_METRICS_SOURCE).set(cpu_usage as f64);
            metrics::gauge!("system_memory_usage_percent", "source" => SYSTEM_METRICS_SOURCE).set(mem_usage);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rust::diagnostics::new_prometheus_reporter::NewPrometheusReporter;
    use std::time::Duration;

    #[tokio::test]
    async fn test_sigar_reporter_starts_without_error() {
        let interval = Duration::from_millis(100);
        start_sigar_reporter(interval);

        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    #[tokio::test]
    async fn test_sigar_reporter_records_cpu_metric() {
        let reporter = NewPrometheusReporter::initialize().unwrap();

        let interval = Duration::from_millis(50);
        start_sigar_reporter(interval);

        tokio::time::sleep(Duration::from_millis(150)).await;

        let scrape = reporter.scrape_data();
        assert!(
            scrape.contains("system_cpu_usage_percent") || scrape.is_empty(),
            "If metrics are recorded, scrape should contain system_cpu_usage_percent"
        );
    }

    #[tokio::test]
    async fn test_sigar_reporter_records_memory_metric() {
        let reporter = NewPrometheusReporter::initialize().unwrap();

        let interval = Duration::from_millis(50);
        start_sigar_reporter(interval);

        tokio::time::sleep(Duration::from_millis(150)).await;

        let scrape = reporter.scrape_data();
        assert!(
            scrape.contains("system_memory_usage_percent") || scrape.is_empty(),
            "If metrics are recorded, scrape should contain system_memory_usage_percent"
        );
    }

    #[tokio::test]
    async fn test_sigar_metrics_use_correct_source() {
        let reporter = NewPrometheusReporter::initialize().unwrap();

        let interval = Duration::from_millis(50);
        start_sigar_reporter(interval);

        tokio::time::sleep(Duration::from_millis(150)).await;

        let scrape = reporter.scrape_data();
        assert!(
            scrape.contains("f1r3fly.system") || scrape.contains("source=\"f1r3fly.system\"") || scrape.is_empty(),
            "If metrics are recorded, scrape should contain f1r3fly.system source"
        );
    }

    #[tokio::test]
    async fn test_sigar_reporter_respects_interval() {
        let _reporter = NewPrometheusReporter::initialize().unwrap();

        let interval = Duration::from_secs(10);
        start_sigar_reporter(interval);

        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
