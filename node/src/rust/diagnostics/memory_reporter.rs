/// Process-level memory reporter and structural size gauge runner.
///
/// Runs a background OS thread (like sigar_reporter) that:
/// 1. On Linux: reads `/proc/self/status` every `interval` and emits
///    `process_memory_rss_bytes` and `process_memory_peak_bytes` gauges.
/// 2. Calls each closure in `extra_gauges` and records the returned count
///    under the given metric name.  Callers use this to expose the live size
///    of in-process collections (channels_map, requested_blocks, etc.).
///
/// Usage:
/// ```
/// memory_reporter::start_memory_reporter(
///     Duration::from_secs(10),
///     vec![
///         ("transport_channels_count",
///          Box::new(move || channels_arc.try_lock().map(|g| g.len()).unwrap_or(0))),
///     ],
/// );
/// ```
use std::time::Duration;
#[cfg(target_os = "linux")]
use tracing::warn;

use crate::rust::diagnostics::SYSTEM_METRICS_SOURCE;

/// Start the memory metrics background thread.
///
/// `extra_gauges`: list of `(metric_name, count_fn)` pairs.  Each `count_fn`
/// must be `Send` (it will be moved into the spawned thread) and returns the
/// current size of the collection it watches.  The gauge is recorded with the
/// `source = SYSTEM_METRICS_SOURCE` label so it appears alongside CPU / memory
/// system metrics in Prometheus.
pub fn start_memory_reporter(
    interval: Duration,
    extra_gauges: Vec<(&'static str, Box<dyn Fn() -> usize + Send>)>,
) {
    std::thread::spawn(move || {
        loop {
            // ── Process-level memory (Linux only via /proc/self/status) ──────
            #[cfg(target_os = "linux")]
            report_process_memory();

            // ── Structural collection sizes ───────────────────────────────────
            for (name, size_fn) in &extra_gauges {
                metrics::gauge!(*name, "source" => SYSTEM_METRICS_SOURCE)
                    .set(size_fn() as f64);
            }

            std::thread::sleep(interval);
        }
    });
}

/// Parse `/proc/self/status` and emit RSS / peak-virtual gauges (bytes).
#[cfg(target_os = "linux")]
fn report_process_memory() {
    match std::fs::read_to_string("/proc/self/status") {
        Err(e) => warn!("memory_reporter: failed to read /proc/self/status: {}", e),
        Ok(contents) => {
            for line in contents.lines() {
                let mut parts = line.split_whitespace();
                match parts.next() {
                    Some("VmRSS:") => {
                        if let Some(kb_str) = parts.next() {
                            if let Ok(kb) = kb_str.parse::<f64>() {
                                metrics::gauge!(
                                    "process_memory_rss_bytes",
                                    "source" => SYSTEM_METRICS_SOURCE
                                )
                                .set(kb * 1024.0);
                            }
                        }
                    }
                    Some("VmPeak:") => {
                        if let Some(kb_str) = parts.next() {
                            if let Ok(kb) = kb_str.parse::<f64>() {
                                metrics::gauge!(
                                    "process_memory_peak_bytes",
                                    "source" => SYSTEM_METRICS_SOURCE
                                )
                                .set(kb * 1024.0);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}
