/// Metrics trait for collecting application metrics
pub trait Metrics: Send + Sync {
    fn increment_counter(&self, name: &str)
        -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}
