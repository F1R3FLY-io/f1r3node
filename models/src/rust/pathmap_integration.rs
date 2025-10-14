//! Integration layer between Rholang Par types and the PathMap crate.

use pathmap::PathMap;
use crate::rhoapi::{Par, Var};

/// Type alias for our standard use case: PathMap from bytes to Rholang Par.
pub type RholangPathMap = PathMap<Par>;
// Additional functions and types for conversion will be added in follow-up steps.
