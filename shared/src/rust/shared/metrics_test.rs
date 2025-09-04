use crate::rust::shared::metrics::Metrics;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Test implementation of Metrics for isolated testing environment
#[derive(Clone)]
pub struct MetricsTestImpl {
    counters: Arc<Mutex<HashMap<String, i32>>>,
}

impl MetricsTestImpl {
    pub fn new() -> Self {
        Self {
            counters: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn get_counter(&self, name: &str) -> i32 {
        let counters = self.counters.lock().unwrap();
        counters
            .get(&format!("approve-block.{}", name))
            .copied()
            .unwrap_or(0)
    }

    pub fn has_counter(&self, name: &str) -> bool {
        let counters = self.counters.lock().unwrap();
        counters.contains_key(&format!("approve-block.{}", name))
    }
}

impl Metrics for MetricsTestImpl {
    fn increment_counter(
        &self,
        name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut counters = self.counters.lock().unwrap();
        let full_name = format!("approve-block.{}", name);
        let current = counters.get(&full_name).copied().unwrap_or(0);
        counters.insert(full_name, current + 1);
        Ok(())
    }
}
