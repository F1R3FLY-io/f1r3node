#[cfg(test)]
mod tests {
    use crate::rust::diagnostics::new_prometheus_reporter::NewPrometheusReporter;

    #[test]
    fn test_reporter_initialize() {
        // Test that we can initialize the reporter
        let result = NewPrometheusReporter::initialize();
        assert!(result.is_ok());
    }

    #[test]
    fn test_reporter_scrape_data() {
        // Test that scrape_data returns Prometheus format
        let (reporter, _handle) = NewPrometheusReporter::initialize().unwrap();
        let output = reporter.scrape_data();
        
        // Should return valid Prometheus format (even if empty)
        // Prometheus format starts with # or is empty
        assert!(output.is_empty() || output.starts_with('#'));
    }
}

