// MetricsSemaphore - Semaphore wrapper with metrics instrumentation
// See shared/src/main/scala/coop/rchain/metrics/MetricsSemaphore.scala

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

use super::metrics_constants::{LOCK_ACQUIRE_TIME_METRIC, LOCK_PERMIT_METRIC, LOCK_QUEUE_METRIC};

/// A semaphore wrapper that records metrics for lock operations.
///
/// Matches the Scala MetricsSemaphore which records:
/// - `lock.queue` gauge: Number of tasks waiting to acquire the semaphore
/// - `lock.permit` gauge: Number of permits currently held
/// - `lock.acquire` timer: Time spent waiting to acquire the semaphore
pub struct MetricsSemaphore {
    semaphore: Arc<Semaphore>,
    source: &'static str,
    queue_count: Arc<AtomicI64>,
    permit_count: Arc<AtomicI64>,
}

impl MetricsSemaphore {
    /// Creates a new MetricsSemaphore with the given number of permits.
    ///
    /// # Arguments
    /// * `permits` - The number of permits available
    /// * `source` - The metrics source label for this semaphore
    pub fn new(permits: usize, source: &'static str) -> Self {
        MetricsSemaphore {
            semaphore: Arc::new(Semaphore::new(permits)),
            source,
            queue_count: Arc::new(AtomicI64::new(0)),
            permit_count: Arc::new(AtomicI64::new(0)),
        }
    }

    /// Creates a new MetricsSemaphore with a single permit (mutex-like behavior).
    /// This matches Scala's `MetricsSemaphore.single`.
    pub fn single(source: &'static str) -> Self {
        Self::new(1, source)
    }

    /// Returns the current number of available permits.
    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }

    /// Acquires a permit from the semaphore, recording metrics.
    ///
    /// This method increments `lock.queue` while waiting, records `lock.acquire` time,
    /// and decrements `lock.queue` after acquisition.
    pub async fn acquire(&self) -> MetricsSemaphorePermit {
        // Increment queue gauge
        let queue_val = self.queue_count.fetch_add(1, Ordering::SeqCst) + 1;
        metrics::gauge!(LOCK_QUEUE_METRIC, "source" => self.source).set(queue_val as f64);

        let start = Instant::now();

        // Acquire the permit
        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("Semaphore closed unexpectedly");

        // Record acquisition time
        metrics::histogram!(LOCK_ACQUIRE_TIME_METRIC, "source" => self.source)
            .record(start.elapsed().as_secs_f64());

        // Decrement queue gauge
        let queue_val = self.queue_count.fetch_sub(1, Ordering::SeqCst) - 1;
        metrics::gauge!(LOCK_QUEUE_METRIC, "source" => self.source).set(queue_val as f64);

        // Increment permit gauge
        let permit_val = self.permit_count.fetch_add(1, Ordering::SeqCst) + 1;
        metrics::gauge!(LOCK_PERMIT_METRIC, "source" => self.source).set(permit_val as f64);

        MetricsSemaphorePermit {
            _permit: permit,
            permit_count: self.permit_count.clone(),
            source: self.source,
        }
    }

    /// Tries to acquire a permit without waiting.
    /// Returns None if no permits are available.
    pub fn try_acquire(&self) -> Option<MetricsSemaphorePermit> {
        match self.semaphore.clone().try_acquire_owned() {
            Ok(permit) => {
                // Increment permit gauge
                let permit_val = self.permit_count.fetch_add(1, Ordering::SeqCst) + 1;
                metrics::gauge!(LOCK_PERMIT_METRIC, "source" => self.source).set(permit_val as f64);

                Some(MetricsSemaphorePermit {
                    _permit: permit,
                    permit_count: self.permit_count.clone(),
                    source: self.source,
                })
            }
            Err(TryAcquireError::NoPermits) => None,
            Err(TryAcquireError::Closed) => panic!("Semaphore closed unexpectedly"),
        }
    }

    /// Executes the given async closure while holding a permit.
    /// This is the equivalent of Scala's `withPermit`.
    pub async fn with_permit<F, T>(&self, f: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        let _permit = self.acquire().await;
        f.await
    }
}

/// A permit acquired from a MetricsSemaphore.
/// When dropped, it releases the permit and updates metrics.
pub struct MetricsSemaphorePermit {
    _permit: OwnedSemaphorePermit,
    permit_count: Arc<AtomicI64>,
    source: &'static str,
}

impl Drop for MetricsSemaphorePermit {
    fn drop(&mut self) {
        // Decrement permit gauge when permit is released
        let permit_val = self.permit_count.fetch_sub(1, Ordering::SeqCst) - 1;
        metrics::gauge!(LOCK_PERMIT_METRIC, "source" => self.source).set(permit_val as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::metrics_constants::SHARED_METRICS_SOURCE;

    #[tokio::test]
    async fn test_metrics_semaphore_single() {
        let sem = MetricsSemaphore::single(SHARED_METRICS_SOURCE);
        assert_eq!(sem.available_permits(), 1);

        let permit = sem.acquire().await;
        assert_eq!(sem.available_permits(), 0);

        drop(permit);
        assert_eq!(sem.available_permits(), 1);
    }

    #[tokio::test]
    async fn test_metrics_semaphore_with_permit() {
        let sem = MetricsSemaphore::single(SHARED_METRICS_SOURCE);

        let result = sem.with_permit(async { 42 }).await;
        assert_eq!(result, 42);
        assert_eq!(sem.available_permits(), 1);
    }

    #[tokio::test]
    async fn test_metrics_semaphore_try_acquire() {
        let sem = MetricsSemaphore::single(SHARED_METRICS_SOURCE);

        let permit1 = sem.try_acquire();
        assert!(permit1.is_some());
        assert_eq!(sem.available_permits(), 0);

        let permit2 = sem.try_acquire();
        assert!(permit2.is_none());

        drop(permit1);
        assert_eq!(sem.available_permits(), 1);
    }
}
