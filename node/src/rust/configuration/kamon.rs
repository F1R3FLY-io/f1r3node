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

use byte_unit::{Byte, Unit};

#[derive(Debug, Clone, Deserialize)]
pub struct KamonConf {
    #[serde(default)]
    pub trace: Option<TraceConfig>,
    #[serde(default)]
    pub influxdb: Option<InfluxDbConfig>,
    #[serde(default)]
    pub zipkin: Option<ZipkinConfig>,
    #[serde(default)]
    pub prometheus: Option<ToggleSection>,
    #[serde(default)]
    pub sigar: Option<ToggleSection>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TraceConfig {
    pub sampler: Option<String>,
    #[serde(rename = "join-remote-parents-with-same-span-id")]
    pub join_remote_parents_with_same_span_id: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InfluxDbConfig {
    pub hostname: Option<String>,
    pub port: Option<u16>,

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

#[derive(Debug, Clone, Deserialize)]
pub struct AdditionalTags {
    pub service: Option<bool>,
    pub host: Option<bool>,
    pub instance: Option<bool>,

    #[serde(rename = "blacklisted-tags", default)]
    pub blacklisted_tags: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ZipkinConfig {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub protocol: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToggleSection {
    pub enabled: Option<bool>,
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
