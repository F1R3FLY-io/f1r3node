/// DEPRECATED: Kamon configuration for JVM compatibility
/// This configuration is maintained for migration purposes only.
/// Kamon is JVM-specific and not available in Rust.
///
/// For Rust monitoring, use:
/// - tracing-subscriber for structured logging
/// - prometheus for metrics
/// - opentelemetry for distributed tracing
use serde::Deserialize;
use std::fmt;
use std::time::Duration;

use byte_unit::{Byte, Unit};

#[derive(Debug, Deserialize)]
pub struct KamonConf {
    #[serde(default)]
    pub trace: Option<TraceConfig>,
    #[serde(default)]
    pub metric: Option<MetricConfig>,
    #[serde(default)]
    pub influxdb: Option<InfluxDbConfig>,
    #[serde(default)]
    pub zipkin: Option<ZipkinConfig>,
    #[serde(default)]
    pub prometheus: Option<ToggleSection>,
    #[serde(default)]
    pub sigar: Option<ToggleSection>,
}

#[derive(Debug, Deserialize)]
pub struct TraceConfig {
    pub sampler: Option<String>,
    #[serde(rename = "join-remote-parents-with-same-span-id")]
    pub join_remote_parents_with_same_span_id: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct MetricConfig {
    #[serde(
        rename = "tick-interval",
        deserialize_with = "de_duration_from_string",
        default = "default_tick_interval"
    )]
    pub tick_interval: Duration,
}

fn default_tick_interval() -> Duration {
    Duration::from_secs(10)
}

#[derive(Debug, Deserialize)]
pub struct InfluxDbConfig {
    pub hostname: Option<String>,
    pub port: Option<u16>,
    pub database: Option<String>,
    pub protocol: Option<String>,
    pub authentication: Option<Auth>,

    /// HOCON: `max-packet-size = 1024 bytes` або `"1 MiB"`
    #[serde(
        rename = "max-packet-size",
        deserialize_with = "de_byte_allow_number_or_string",
        default
    )]
    pub max_packet_size: Option<Byte>,

    pub percentiles: Option<Vec<f64>>,

    #[serde(rename = "additional-tags")]
    pub additional_tags: Option<AdditionalTags>,
}

#[derive(Debug, Deserialize)]
pub struct Auth {
    pub user: Option<String>,
    pub password: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AdditionalTags {
    pub service: Option<bool>,
    pub host: Option<bool>,
    pub instance: Option<bool>,

    #[serde(rename = "blacklisted-tags", default)]
    pub blacklisted_tags: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ZipkinConfig {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub protocol: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ToggleSection {
    pub enabled: Option<bool>,
}

fn de_duration_from_string<'de, D>(de: D) -> Result<Duration, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct DurationVisitor;

    impl<'de> serde::de::Visitor<'de> for DurationVisitor {
        type Value = Duration;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("duration as string (e.g. \"10 seconds\", \"5 minutes\") or number of seconds")
        }

        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
            Ok(Duration::from_secs(v))
        }

        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if v < 0 {
                return Err(E::custom("negative duration not allowed"));
            }
            Ok(Duration::from_secs(v as u64))
        }

        fn visit_str<E>(self, s: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            // Parse strings like "10 seconds", "5 minutes", etc.
            let parts: Vec<&str> = s.trim().split_whitespace().collect();
            if parts.len() != 2 {
                return Err(E::custom(format!("invalid duration format: {}", s)));
            }

            let value: u64 = parts[0]
                .parse()
                .map_err(|_| E::custom(format!("invalid number: {}", parts[0])))?;
            
            let multiplier = match parts[1].to_lowercase().as_str() {
                "second" | "seconds" | "s" => 1,
                "minute" | "minutes" | "m" => 60,
                "hour" | "hours" | "h" => 3600,
                "day" | "days" | "d" => 86400,
                unit => return Err(E::custom(format!("unknown time unit: {}", unit))),
            };

            Ok(Duration::from_secs(value * multiplier))
        }
    }

    de.deserialize_any(DurationVisitor)
}

fn de_byte_allow_number_or_string<'de, D>(de: D) -> Result<Option<Byte>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct V;

    impl<'de> serde::de::Visitor<'de> for V {
        type Value = Option<Byte>;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str(
                "byte size as number of bytes or string with unit (e.g. \"1024 bytes\", \"1MiB\")",
            )
        }

        fn visit_none<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_unit<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
            Ok(Some(Byte::from_u64(v)))
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if v < 0 {
                return Err(E::custom("negative size not allowed"));
            }
            Ok(Some(Byte::from_u64(v as u64)))
        }
        fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if v.is_sign_negative() {
                return Err(E::custom("negative size not allowed"));
            }
            Byte::from_f64_with_unit(v, Unit::B)
                .ok_or_else(|| E::custom("value too large"))
                .map(Some)
        }
        fn visit_str<E>(self, s: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Byte::parse_str(s, true).map(Some).map_err(E::custom)
        }
        fn visit_string<E>(self, s: String) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            self.visit_str(&s)
        }
    }

    de.deserialize_any(V)
}
