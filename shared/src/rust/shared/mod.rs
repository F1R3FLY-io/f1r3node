pub mod compression;
pub mod f1r3fly_event;
pub mod f1r3fly_events;
pub mod list_ops;
pub mod printer;
pub mod string_ops;

// TODO: Port MetricsSemaphore to Rust for contention tracking.
//
// The Scala `MetricsSemaphore` is a wrapper around a semaphore that collects metrics
// on contention. This is critical for diagnosing performance bottlenecks related to
// concurrent resource access.
//
// Required Metrics (from scala-metrics-and-spans-table.md):
// - Gauge: "lock.queue" - The number of tasks waiting to acquire the semaphore.
//   - Source: Dynamic (e.g., "rchain.casper.propose-lock")
//   - Metrics[F].incrementGauge("lock.queue") - when task enters wait queue
//   - Metrics[F].decrementGauge("lock.queue") - when task leaves wait queue
// - Gauge: "lock.permit" - The number of available permits.
//   - Metrics[F].incrementGauge("lock.permit") - when permit is released
//   - Metrics[F].decrementGauge("lock.permit") - when permit is acquired
// - Timer: "lock.acquire" - The time it takes to acquire a permit.
//
// Implementation Details:
// - Create a `MetricsSemaphore` struct that wraps `tokio::sync::Semaphore`.
// - The `acquire` method should be instrumented to update the gauges and record
//   the acquisition time in a histogram.
// - The source for the metrics should be configurable on creation, allowing different
//   parts of the system to have their own semaphore metrics.
// - Use metrics crate: metrics::gauge!("lock.queue", "source" => source).increment(1.0) / .decrement(1.0)
// - Use metrics crate: metrics::gauge!("lock.permit", "source" => source).increment(1.0) / .decrement(1.0)
