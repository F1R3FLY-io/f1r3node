use std::sync::{Arc, Mutex};

use costs::Cost;
use tokio::sync::Semaphore;

use super::errors::InterpreterError;

pub mod cost_accounting;
pub mod costs;
pub mod has_cost;

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/accounting/package.scala
#[allow(non_camel_case_types)]
pub type _cost = CostManager;

#[derive(Clone)]
pub struct CostManager {
    state: Arc<Mutex<Cost>>,
    semaphore: Arc<Semaphore>,
    log: Arc<Mutex<Vec<Cost>>>,
}

impl CostManager {
    pub fn new(initial_value: Cost, semaphore_count: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(initial_value)),
            semaphore: Arc::new(Semaphore::new(semaphore_count)),
            log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn charge(&self, amount: Cost) -> Result<(), InterpreterError> {
        let permit = self
            .semaphore
            .try_acquire()
            .map_err(|_| InterpreterError::SetupError("Failed to acquire semaphore".to_string()))?;

        let mut current_cost = self.state.try_lock().unwrap();

        if current_cost.value < 0 {
            return Err(InterpreterError::OutOfPhlogistonsError);
        }

        current_cost.value -= amount.value;
        self.log.lock().unwrap().push(amount.clone());
        drop(permit);

        Ok(())
    }

    pub fn get(&self) -> Cost {
        let current_cost = self.state.try_lock().unwrap();
        current_cost.clone()
    }

    pub fn set(&self, new_value: Cost) {
        let mut current_cost = self.state.try_lock().unwrap();
        *current_cost = new_value;
    }

    pub fn get_log(&self) -> Vec<Cost> {
        self.log.lock().unwrap().clone()
    }

    pub fn clear_log(&self) {
        self.log.lock().unwrap().clear();
    }
}
