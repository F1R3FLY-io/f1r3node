/// jemalloc heap stats reporter.
///
/// Advances the jemalloc epoch (forces stat refresh) then reads five
/// allocator-internal counters and emits them as Prometheus gauges.
///
/// Metrics emitted (source = "f1r3fly.system"):
///   jemalloc_allocated_bytes  – bytes in live allocations
///   jemalloc_active_bytes     – bytes in active pages (allocated + overhead)
///   jemalloc_mapped_bytes     – bytes in mapped chunks
///   jemalloc_resident_bytes   – bytes in resident pages (≈ RSS contribution)
///   jemalloc_retained_bytes   – bytes freed but not returned to the OS
///
/// `resident - allocated` reveals fragmentation + retained overhead.
/// `retained` growing per-block indicates jemalloc is not purging freed arenas.
use std::time::Duration;

/// Start the jemalloc stats background thread.
///
/// Only active in non-test builds because jemalloc itself is only used as
/// the global allocator outside of tests (`#[cfg(not(test))]` in main.rs).
pub fn start_jemalloc_reporter(_interval: Duration) {
    #[cfg(not(test))]
    {
        use crate::rust::diagnostics::SYSTEM_METRICS_SOURCE;

        std::thread::spawn(move || {
            use tikv_jemalloc_ctl::{epoch, stats};

            // Pre-build epoch MIB handle once; reuse each iteration.
            let epoch_mib = match epoch::mib() {
                Ok(m) => m,
                Err(_) => return,
            };

            loop {
                // Advance epoch so jemalloc refreshes its cached stat values.
                let _ = epoch_mib.advance();

                macro_rules! emit_stat {
                    ($read:expr, $name:literal) => {
                        if let Ok(v) = $read {
                            let bytes: usize = v;
                            metrics::gauge!($name, "source" => SYSTEM_METRICS_SOURCE)
                                .set(bytes as f64);
                        }
                    };
                }

                emit_stat!(stats::allocated::read(), "jemalloc_allocated_bytes");
                emit_stat!(stats::active::read(),    "jemalloc_active_bytes");
                emit_stat!(stats::mapped::read(),    "jemalloc_mapped_bytes");
                emit_stat!(stats::resident::read(),  "jemalloc_resident_bytes");
                emit_stat!(stats::retained::read(),  "jemalloc_retained_bytes");

                std::thread::sleep(_interval);
            }
        });
    }
}
