use std::collections::HashMap;

// Prometheus histogram bucket definitions
// These values are taken from Kamon's default Prometheus configuration to maintain
// compatibility with the Scala implementation in NewPrometheusReporter.scala
// Source: kamon-prometheus library defaults (buckets.default-buckets, buckets.time-buckets, buckets.information-buckets)

// Time-based buckets (milliseconds to seconds)
const BUCKET_1MS: f64 = 0.001;
const BUCKET_3MS: f64 = 0.003;
const BUCKET_5MS: f64 = 0.005;
const BUCKET_10MS: f64 = 0.01;
const BUCKET_25MS: f64 = 0.025;
const BUCKET_50MS: f64 = 0.05;
const BUCKET_75MS: f64 = 0.075;
const BUCKET_100MS: f64 = 0.1;
const BUCKET_250MS: f64 = 0.25;
const BUCKET_500MS: f64 = 0.5;
const BUCKET_750MS: f64 = 0.75;
const BUCKET_1S: f64 = 1.0;
const BUCKET_2_5S: f64 = 2.5;
const BUCKET_5S: f64 = 5.0;
const BUCKET_7_5S: f64 = 7.5;
const BUCKET_10S: f64 = 10.0;

// Information-based buckets (bytes)
const BUCKET_512B: f64 = 512.0;
const BUCKET_1KB: f64 = 1024.0;
const BUCKET_2KB: f64 = 2048.0;
const BUCKET_4KB: f64 = 4096.0;
const BUCKET_16KB: f64 = 16384.0;
const BUCKET_64KB: f64 = 65536.0;
const BUCKET_512KB: f64 = 524288.0;
const BUCKET_1MB: f64 = 1048576.0;

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
                BUCKET_5MS,
                BUCKET_10MS,
                BUCKET_25MS,
                BUCKET_50MS,
                BUCKET_75MS,
                BUCKET_100MS,
                BUCKET_250MS,
                BUCKET_500MS,
                BUCKET_750MS,
                BUCKET_1S,
                BUCKET_2_5S,
                BUCKET_5S,
                BUCKET_7_5S,
                BUCKET_10S,
            ],
            time_buckets: vec![
                BUCKET_1MS,
                BUCKET_3MS,
                BUCKET_5MS,
                BUCKET_10MS,
                BUCKET_25MS,
                BUCKET_50MS,
                BUCKET_75MS,
                BUCKET_100MS,
                BUCKET_250MS,
                BUCKET_500MS,
                BUCKET_750MS,
                BUCKET_1S,
                BUCKET_2_5S,
                BUCKET_5S,
                BUCKET_7_5S,
                BUCKET_10S,
            ],
            information_buckets: vec![
                BUCKET_512B,
                BUCKET_1KB,
                BUCKET_2KB,
                BUCKET_4KB,
                BUCKET_16KB,
                BUCKET_64KB,
                BUCKET_512KB,
                BUCKET_1MB,
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

