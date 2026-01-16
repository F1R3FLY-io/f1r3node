//! Phlogiston (Gas) Accounting for Reified RSpaces
//!
//! This module implements the gas/phlogiston accounting system that measures and limits
//! computational resource usage in RSpace operations. It ensures that processes cannot
//! consume unbounded resources by requiring payment for each operation.
//!
//! # Formal Correspondence
//! - `Phlogiston.v`: Core phlogiston invariants (non-negativity, charge preservation)
//! - `GenericRSpace.v`: Integration with space operations
//! - `Safety/Properties.v`: Gas accounting safety properties
//!
//! # Design Overview
//! The phlogiston system consists of:
//! 1. **Cost Functions**: Define the gas cost of each operation type
//! 2. **PhlogistonMeter**: Tracks gas consumption and enforces limits
//! 3. **ChargingSpaceAgent**: Wrapper that charges for operations
//!
//! # Invariants
//! - Phlogiston balance is always non-negative (enforced by type system with u64)
//! - Charge operations preserve non-negativity (checked before deduction)
//! - All space operations have defined costs

use std::sync::atomic::{AtomicU64, Ordering};
use super::errors::SpaceError;

// =============================================================================
// Cost Constants
// =============================================================================

/// Base cost for sending a message to a channel.
/// This covers the fundamental overhead of routing and storing the message.
pub const SEND_BASE_COST: u64 = 100;

/// Cost per byte of data being sent.
pub const SEND_PER_BYTE_COST: u64 = 1;

/// Base cost for receiving (consuming) from a channel.
pub const RECEIVE_BASE_COST: u64 = 100;

/// Base cost for pattern matching operations.
pub const MATCH_BASE_COST: u64 = 50;

/// Cost per pattern element matched.
pub const MATCH_PER_ELEMENT_COST: u64 = 10;

/// Base cost for creating a new channel.
pub const CHANNEL_CREATE_COST: u64 = 200;

/// Base cost for freeing a channel.
pub const CHANNEL_FREE_COST: u64 = 50;

/// Base cost for creating a checkpoint.
pub const CHECKPOINT_COST: u64 = 500;

/// Base cost for replaying from a checkpoint.
pub const REPLAY_BASE_COST: u64 = 100;

/// Cost per operation replayed.
pub const REPLAY_PER_OP_COST: u64 = 10;

/// Base cost for space creation.
pub const SPACE_CREATE_COST: u64 = 1000;

/// Cost for looking up a channel.
pub const LOOKUP_COST: u64 = 20;

/// Cost for VectorDB similarity search (base).
pub const VECTORDB_SEARCH_BASE_COST: u64 = 200;

/// Cost per dimension in VectorDB similarity search.
pub const VECTORDB_SEARCH_PER_DIM_COST: u64 = 5;

/// Cost for priority queue insertion (includes heapify).
pub const PRIORITY_QUEUE_INSERT_COST: u64 = 50;

/// Cost for priority queue pop.
pub const PRIORITY_QUEUE_POP_COST: u64 = 30;

// =============================================================================
// Operation Types for Cost Calculation
// =============================================================================

/// Types of operations that consume phlogiston.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation {
    /// Sending data to a channel.
    Send {
        /// Size of the data being sent in bytes.
        data_size: usize,
    },
    /// Receiving data from a channel.
    Receive,
    /// Pattern matching operation.
    Match {
        /// Number of elements in the pattern.
        pattern_size: usize,
    },
    /// Creating a new channel.
    CreateChannel,
    /// Freeing a channel.
    FreeChannel,
    /// Creating a checkpoint.
    Checkpoint,
    /// Replaying from a checkpoint.
    Replay {
        /// Number of operations to replay.
        operation_count: usize,
    },
    /// Creating a new space.
    CreateSpace,
    /// Looking up a channel.
    Lookup,
    /// VectorDB similarity search.
    VectorDbSearch {
        /// Dimensionality of vectors.
        dimensions: usize,
    },
    /// Priority queue insert.
    PriorityQueueInsert,
    /// Priority queue pop.
    PriorityQueuePop,
    /// Custom operation with explicit cost.
    Custom {
        /// Name of the operation.
        name: String,
        /// Cost of the operation.
        cost: u64,
    },
}

impl Operation {
    /// Calculate the gas cost for this operation.
    ///
    /// # Returns
    /// The gas cost in phlogiston units.
    pub fn cost(&self) -> u64 {
        match self {
            Operation::Send { data_size } => {
                SEND_BASE_COST + (*data_size as u64) * SEND_PER_BYTE_COST
            }
            Operation::Receive => RECEIVE_BASE_COST,
            Operation::Match { pattern_size } => {
                MATCH_BASE_COST + (*pattern_size as u64) * MATCH_PER_ELEMENT_COST
            }
            Operation::CreateChannel => CHANNEL_CREATE_COST,
            Operation::FreeChannel => CHANNEL_FREE_COST,
            Operation::Checkpoint => CHECKPOINT_COST,
            Operation::Replay { operation_count } => {
                REPLAY_BASE_COST + (*operation_count as u64) * REPLAY_PER_OP_COST
            }
            Operation::CreateSpace => SPACE_CREATE_COST,
            Operation::Lookup => LOOKUP_COST,
            Operation::VectorDbSearch { dimensions } => {
                VECTORDB_SEARCH_BASE_COST + (*dimensions as u64) * VECTORDB_SEARCH_PER_DIM_COST
            }
            Operation::PriorityQueueInsert => PRIORITY_QUEUE_INSERT_COST,
            Operation::PriorityQueuePop => PRIORITY_QUEUE_POP_COST,
            Operation::Custom { cost, .. } => *cost,
        }
    }

    /// Get a human-readable description of the operation.
    pub fn description(&self) -> String {
        match self {
            Operation::Send { data_size } => format!("send({} bytes)", data_size),
            Operation::Receive => "receive".to_string(),
            Operation::Match { pattern_size } => format!("match({} elements)", pattern_size),
            Operation::CreateChannel => "create_channel".to_string(),
            Operation::FreeChannel => "free_channel".to_string(),
            Operation::Checkpoint => "checkpoint".to_string(),
            Operation::Replay { operation_count } => format!("replay({} ops)", operation_count),
            Operation::CreateSpace => "create_space".to_string(),
            Operation::Lookup => "lookup".to_string(),
            Operation::VectorDbSearch { dimensions } => format!("vector_search({} dims)", dimensions),
            Operation::PriorityQueueInsert => "priority_queue_insert".to_string(),
            Operation::PriorityQueuePop => "priority_queue_pop".to_string(),
            Operation::Custom { name, .. } => name.clone(),
        }
    }
}

// =============================================================================
// Phlogiston Meter
// =============================================================================

/// A meter that tracks phlogiston consumption and enforces limits.
///
/// The meter is thread-safe and uses atomic operations for concurrent access.
/// It maintains a balance that starts at the initial limit and decreases as
/// operations are charged.
///
/// # Invariants (from Phlogiston.v)
/// - `balance >= 0` (enforced by u64 type)
/// - `charge(amount)` only succeeds if `balance >= amount`
/// - Total consumed = initial_limit - current_balance
#[derive(Debug)]
pub struct PhlogistonMeter {
    /// Current phlogiston balance (remaining gas).
    balance: AtomicU64,
    /// Initial limit for tracking total consumption.
    initial_limit: u64,
    /// Total amount consumed (for reporting).
    total_consumed: AtomicU64,
}

impl PhlogistonMeter {
    /// Create a new phlogiston meter with the given initial balance.
    ///
    /// # Arguments
    /// * `initial_limit` - The starting phlogiston balance.
    pub fn new(initial_limit: u64) -> Self {
        PhlogistonMeter {
            balance: AtomicU64::new(initial_limit),
            initial_limit,
            total_consumed: AtomicU64::new(0),
        }
    }

    /// Create a meter with unlimited phlogiston (for testing or privileged operations).
    pub fn unlimited() -> Self {
        PhlogistonMeter {
            balance: AtomicU64::new(u64::MAX),
            initial_limit: u64::MAX,
            total_consumed: AtomicU64::new(0),
        }
    }

    /// Get the current phlogiston balance.
    pub fn balance(&self) -> u64 {
        self.balance.load(Ordering::Relaxed)
    }

    /// Get the initial limit.
    pub fn initial_limit(&self) -> u64 {
        self.initial_limit
    }

    /// Get the total amount consumed.
    pub fn total_consumed(&self) -> u64 {
        self.total_consumed.load(Ordering::Relaxed)
    }

    /// Check if a charge of the given amount is possible.
    ///
    /// # Arguments
    /// * `amount` - The amount to check.
    ///
    /// # Returns
    /// `true` if the balance is sufficient.
    pub fn can_charge(&self, amount: u64) -> bool {
        self.balance.load(Ordering::Relaxed) >= amount
    }

    /// Attempt to charge phlogiston for an operation.
    ///
    /// This atomically deducts the amount from the balance if sufficient
    /// funds are available.
    ///
    /// # Arguments
    /// * `operation` - The operation being charged.
    ///
    /// # Returns
    /// - `Ok(())` if the charge succeeded
    /// - `Err(SpaceError::OutOfPhlogiston)` if insufficient balance
    ///
    /// # Formal Correspondence
    /// Implements `charge_preserves_non_negative` from Phlogiston.v
    pub fn charge(&self, operation: &Operation) -> Result<(), SpaceError> {
        let amount = operation.cost();
        self.charge_amount(amount, &operation.description())
    }

    /// Charge a specific amount with a description.
    ///
    /// # Arguments
    /// * `amount` - The amount to charge.
    /// * `description` - Description of the operation for error messages.
    ///
    /// # Returns
    /// - `Ok(())` if the charge succeeded
    /// - `Err(SpaceError::OutOfPhlogiston)` if insufficient balance
    pub fn charge_amount(&self, amount: u64, description: &str) -> Result<(), SpaceError> {
        // Use compare-exchange loop for atomic deduction
        loop {
            let current = self.balance.load(Ordering::Relaxed);

            if current < amount {
                return Err(SpaceError::OutOfPhlogiston {
                    required: amount,
                    available: current,
                    operation: description.to_string(),
                });
            }

            let new_balance = current - amount;
            match self.balance.compare_exchange_weak(
                current,
                new_balance,
                Ordering::Release,  // Release is sufficient - CAS retry loop handles races
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.total_consumed.fetch_add(amount, Ordering::Relaxed);
                    return Ok(());
                }
                Err(_) => continue, // Retry if another thread modified
            }
        }
    }

    /// Refund phlogiston (e.g., for operations that were rolled back).
    ///
    /// # Arguments
    /// * `amount` - The amount to refund.
    ///
    /// # Note
    /// This can increase balance beyond initial_limit if called multiple times.
    /// In practice, refunds should only restore previously charged amounts.
    pub fn refund(&self, amount: u64) {
        self.balance.fetch_add(amount, Ordering::Relaxed);
        // Note: We don't decrease total_consumed for refunds
        // This keeps accurate accounting of gross consumption
    }

    /// Reset the meter to its initial state.
    pub fn reset(&self) {
        self.balance.store(self.initial_limit, Ordering::Relaxed);
        self.total_consumed.store(0, Ordering::Relaxed);
    }
}

impl Clone for PhlogistonMeter {
    fn clone(&self) -> Self {
        PhlogistonMeter {
            balance: AtomicU64::new(self.balance.load(Ordering::Relaxed)),
            initial_limit: self.initial_limit,
            total_consumed: AtomicU64::new(self.total_consumed.load(Ordering::Relaxed)),
        }
    }
}

impl Default for PhlogistonMeter {
    fn default() -> Self {
        // Default to 1 million units
        Self::new(1_000_000)
    }
}

// =============================================================================
// Gas Configuration
// =============================================================================

/// Configuration for phlogiston/gas accounting.
#[derive(Debug, Clone)]
pub struct GasConfig {
    /// Initial gas limit for new transactions.
    pub initial_limit: u64,
    /// Whether to enforce gas limits (can be disabled for testing).
    pub enabled: bool,
    /// Cost multiplier (for chain economics).
    pub cost_multiplier: f64,
}

impl Default for GasConfig {
    fn default() -> Self {
        GasConfig {
            initial_limit: 10_000_000,
            enabled: true,
            cost_multiplier: 1.0,
        }
    }
}

impl GasConfig {
    /// Create a configuration with unlimited gas (for testing).
    pub fn unlimited() -> Self {
        GasConfig {
            initial_limit: u64::MAX,
            enabled: false,
            cost_multiplier: 1.0,
        }
    }

    /// Create a meter from this configuration.
    pub fn create_meter(&self) -> PhlogistonMeter {
        if self.enabled {
            PhlogistonMeter::new(self.initial_limit)
        } else {
            PhlogistonMeter::unlimited()
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_costs() {
        assert_eq!(Operation::Send { data_size: 0 }.cost(), SEND_BASE_COST);
        assert_eq!(
            Operation::Send { data_size: 100 }.cost(),
            SEND_BASE_COST + 100 * SEND_PER_BYTE_COST
        );
        assert_eq!(Operation::Receive.cost(), RECEIVE_BASE_COST);
        assert_eq!(
            Operation::Match { pattern_size: 5 }.cost(),
            MATCH_BASE_COST + 5 * MATCH_PER_ELEMENT_COST
        );
    }

    #[test]
    fn test_meter_charge_success() {
        let meter = PhlogistonMeter::new(1000);

        assert!(meter.charge(&Operation::Receive).is_ok());
        assert_eq!(meter.balance(), 1000 - RECEIVE_BASE_COST);
        assert_eq!(meter.total_consumed(), RECEIVE_BASE_COST);
    }

    #[test]
    fn test_meter_charge_failure() {
        let meter = PhlogistonMeter::new(50);

        let result = meter.charge(&Operation::Receive);
        assert!(result.is_err());

        match result {
            Err(SpaceError::OutOfPhlogiston { required, available, .. }) => {
                assert_eq!(required, RECEIVE_BASE_COST);
                assert_eq!(available, 50);
            }
            _ => panic!("Expected OutOfPhlogiston error"),
        }
    }

    #[test]
    fn test_meter_refund() {
        let meter = PhlogistonMeter::new(1000);

        meter.charge(&Operation::Receive).unwrap();
        assert_eq!(meter.balance(), 1000 - RECEIVE_BASE_COST);

        meter.refund(50);
        assert_eq!(meter.balance(), 1000 - RECEIVE_BASE_COST + 50);
    }

    #[test]
    fn test_unlimited_meter() {
        let meter = PhlogistonMeter::unlimited();

        // Should be able to charge a huge amount
        assert!(meter.charge_amount(1_000_000_000, "test").is_ok());
        assert!(meter.balance() > 1_000_000_000);
    }

    #[test]
    fn test_gas_config() {
        let config = GasConfig::default();
        let meter = config.create_meter();

        assert_eq!(meter.initial_limit(), config.initial_limit);

        let unlimited = GasConfig::unlimited().create_meter();
        assert_eq!(unlimited.initial_limit(), u64::MAX);
    }
}
