//! Multi-Space RSpace Integration Tests
//!
//! This module contains comprehensive tests for the Reified RSpaces implementation,
//! including property-based tests derived from Rocq formal proofs.
//!
//! # Test Structure
//!
//! - `proptest_invariants`: Core invariants (NoPendingMatch, Produce/Consume exclusivity)
//! - `proptest_collections`: Collection semantics (Queue FIFO, Stack LIFO, Set idempotent)
//! - `proptest_phlogiston`: Gas accounting properties
//! - `integration_multi_space`: Multi-space coordination tests
//! - `integration_checkpoint`: Checkpoint and replay tests
//! - `e2e_workflows`: End-to-end workflow tests
//!
//! # Rholang Syntax Examples
//!
//! These tests verify the Rust implementation that supports the new Rholang syntax:
//!
//! ```rholang
//! // Creating a Queue space via factory URN
//! new QueueSpace(`rho:space:queue:hashmap:default`), myQueue in {
//!   QueueSpace!({}, *myQueue) |
//!   use myQueue {
//!     // Inside use block, operations target this space
//!     new producer, consumer in {
//!       producer!(1) | producer!(2) | producer!(3) |
//!       for (@x <- producer) { stdout!(x) }  // Prints 1 first (FIFO)
//!     }
//!   }
//! }
//!
//! // Creating a Cell space for one-shot reply channels
//! new CellSpace(`rho:space:cell:hashmap:default`), replyCell in {
//!   CellSpace!({}, *replyCell) |
//!   use replyCell {
//!     new reply in {
//!       reply!(42) |          // First send succeeds
//!       // reply!(43)         // Would fail: CellAlreadyFull error
//!       for (@result <- reply) { stdout!(result) }
//!     }
//!   }
//! }
//! ```

pub mod e2e_workflows;
pub mod integration_checkpoint;
pub mod integration_multi_space;
pub mod proptest_adapter;
pub mod proptest_charging_agent;
pub mod proptest_collections;
pub mod proptest_factory;
pub mod proptest_get_space_agent;
pub mod proptest_history;
pub mod proptest_invariants;
pub mod proptest_matcher;
pub mod proptest_pathmap;
pub mod proptest_phlogiston;
pub mod proptest_registry;
pub mod proptest_similarity;
pub mod proptest_theory_enforcement;
pub mod proptest_types;
pub mod test_utils;
pub mod vectordb;
