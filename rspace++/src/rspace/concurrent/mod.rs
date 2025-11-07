// TODO: Port MultiLock from Scala for deadlock prevention and contention tracking.
//
// The Scala `MultiLock` provides a mechanism to acquire locks on multiple keys in a
// canonical order, which is essential for preventing deadlocks in concurrent systems
// like RSpace. It also collects metrics on lock contention.
//
// Required Metrics (from scala-metrics-and-spans-table.md):
// - Gauge: "lock.queue" - The number of tasks waiting to acquire a lock for a specific key.
//   - Source: Dynamic (based on the lock key)
//   - Metrics[F].incrementGauge("lock.queue") - when entering queue
//   - Metrics[F].decrementGauge("lock.queue") - when leaving queue
// - Timer: "lock.acquire" - The time it takes to acquire a lock.
//
// Implementation Details:
// - Create a `MultiLock` struct that can manage a collection of locks (e.g., using `tokio::sync::Mutex`).
// - The `acquire` method must take a set of keys, sort them to ensure a canonical locking order,
//   and then acquire the locks sequentially.
// - The `acquire` method should be instrumented to update the queue gauge and record the
//   acquisition time in a histogram for each lock.
// - Use metrics crate: metrics::gauge!("lock.queue", "source" => source).increment(1.0) / .decrement(1.0)

