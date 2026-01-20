//! Ranking functions for VectorDB query results.
//!
//! This module provides ranking function handlers that select and order
//! results from similarity scores based on ranking criteria.
//!
//! # Handlers
//!
//! - [`TopKRankingHandler`]: Return up to K highest-scoring results
//! - [`AllRankingHandler`]: Return all results above threshold
//! - [`ThresholdRankingHandler`]: Filter by custom threshold
//!
//! See the [`handlers`](super::handlers) module for the full handler system.

// Re-export from handlers module
pub use super::handlers::ranking::{AllRankingHandler, ThresholdRankingHandler, TopKRankingHandler};
pub use super::handlers::traits::RankingFunctionHandler;
pub use super::handlers::types::{RankingResult, ResolvedArg};
