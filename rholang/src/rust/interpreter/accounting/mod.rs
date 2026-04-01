use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicI64, Ordering},
        Arc, Mutex,
    },
};

use costs::Cost;

use super::errors::InterpreterError;

pub mod cost_accounting;
pub mod costs;
pub mod has_cost;

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/accounting/package.scala
#[allow(non_camel_case_types)]
pub type _cost = CostManager;

#[derive(Clone)]
pub struct CostManager {
    value: Arc<AtomicI64>,
    log: Arc<Mutex<VecDeque<Cost>>>,
    max_log_entries: usize,
}

impl CostManager {
    fn resolve_max_log_entries() -> usize {
        if cfg!(test) {
            return usize::MAX;
        }

        std::env::var("F1R3_COST_LOG_MAX_ENTRIES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    }

    pub fn new(initial_value: Cost, _semaphore_count: usize) -> Self {
        let max_log_entries = Self::resolve_max_log_entries();
        let initial_capacity = if max_log_entries == 0 {
            0
        } else if max_log_entries == usize::MAX {
            1024
        } else {
            max_log_entries.min(1024)
        };

        Self {
            value: Arc::new(AtomicI64::new(initial_value.value)),
            log: Arc::new(Mutex::new(VecDeque::with_capacity(initial_capacity))),
            max_log_entries,
        }
    }

    pub fn charge(&self, amount: Cost) -> Result<(), InterpreterError> {
        loop {
            let current = self.value.load(Ordering::Acquire);
            if current < 0 {
                return Err(InterpreterError::OutOfPhlogistonsError);
            }
            let new_value = current - amount.value;
            match self.value.compare_exchange_weak(
                current,
                new_value,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if self.max_log_entries > 0 {
                        let mut log = self.log.lock().expect("cost log lock poisoned");
                        if log.len() >= self.max_log_entries {
                            let _ = log.pop_front();
                        }
                        log.push_back(amount);
                    }
                    if new_value < 0 {
                        return Err(InterpreterError::OutOfPhlogistonsError);
                    }
                    return Ok(());
                }
                Err(_) => continue,
            }
        }
    }

    pub fn get(&self) -> Cost {
        Cost {
            value: self.value.load(Ordering::Acquire),
            operation: "current".into(),
        }
    }

    pub fn set(&self, new_value: Cost) {
        self.value.store(new_value.value, Ordering::Release);
    }

    pub fn get_log(&self) -> Vec<Cost> {
        self.log.lock().expect("cost log lock poisoned").iter().cloned().collect()
    }

    pub fn clear_log(&self) {
        self.log.lock().expect("cost log lock poisoned").clear();
    }
}
