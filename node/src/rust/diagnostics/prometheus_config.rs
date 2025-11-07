use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PrometheusConfiguration {
    pub default_buckets: Vec<f64>,
    pub time_buckets: Vec<f64>,
    pub information_buckets: Vec<f64>,
    pub custom_buckets: HashMap<String, Vec<f64>>,
    pub include_environment_tags: bool,
}

impl PrometheusConfiguration {
    pub fn default() -> Self {
        Self {
            default_buckets: vec![
                0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
            ],
            time_buckets: vec![
                0.001, 0.003, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5,
                5.0, 7.5, 10.0,
            ],
            information_buckets: vec![
                512.0, 1024.0, 2048.0, 4096.0, 16384.0, 65536.0, 524288.0, 1048576.0,
            ],
            custom_buckets: HashMap::new(),
            include_environment_tags: false,
        }
    }

    pub fn environment_tags(&self) -> HashMap<String, String> {
        if self.include_environment_tags {
            let mut tags = HashMap::new();
            tags.insert("service".to_string(), "rnode".to_string());
            if let Ok(hostname) = std::env::var("HOSTNAME") {
                tags.insert("host".to_string(), hostname);
            }
            tags
        } else {
            HashMap::new()
        }
    }
}

