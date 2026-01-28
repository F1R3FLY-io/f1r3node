// Metrics constants for the shared package
// Matching Scala: shared/src/main/scala/coop/rchain/metrics/MetricsSemaphore.scala

pub const SHARED_METRICS_SOURCE: &str = "f1r3fly.shared";

// MetricsSemaphore gauge metrics
pub const LOCK_QUEUE_METRIC: &str = "lock.queue";
pub const LOCK_PERMIT_METRIC: &str = "lock.permit";
pub const LOCK_ACQUIRE_TIME_METRIC: &str = "lock.acquire.time";

pub const CASPER_PACKET_HANDLER_METRICS_SOURCE: &str = "f1r3fly.casper.packet-handler";
pub const COST_ACCOUNTING_METRICS_SOURCE: &str = "f1r3fly.rholang.cost";
