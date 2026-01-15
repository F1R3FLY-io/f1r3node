//! Layer 5-6: Space Factories
//!
//! This module provides factory functions for creating spaces with various
//! configurations. Each factory creates a `SpaceAgent` implementation with
//! the specified storage and collection types.
//!
//! # URN Formats
//!
//! ## Short Format (Legacy)
//!
//! | URN | Description |
//! |-----|-------------|
//! | `rho:space:HashMapBagSpace` | HashMap + Bag (original default) |
//! | `rho:space:PathMapSpace` | PathMap + Bag (recommended) |
//! | `rho:space:QueueSpace` | HashMap + Queue (FIFO) |
//! | `rho:space:StackSpace` | HashMap + Stack (LIFO) |
//! | `rho:space:SetSpace` | HashMap + Set (idempotent) |
//! | `rho:space:CellSpace` | HashMap + Cell (exactly-once) |
//! | `rho:space:ArraySpace(n,cyclic)` | Array + Bag (fixed size) |
//! | `rho:space:VectorSpace` | Vector + Bag (unbounded) |
//! | `rho:space:SeqSpace` | HashSet + Set (sequential) |
//! | `rho:space:TempSpace` | HashMap + Bag (non-persistent) |
//! | `rho:space:PriorityQueueSpace(n)` | HashMap + PriorityQueue |
//! | `rho:space:VectorDBSpace(dims)` | HashMap + VectorDB (similarity) |
//!
//! ## Extended Format: `rho:space:{inner}:{outer}:{qualifier}`
//!
//! All valid outer/inner combinations are supported:
//!
//! ### HashMap Outer
//! | URN | Description |
//! |-----|-------------|
//! | `rho:space:bag:hashmap:default` | HashMap + Bag |
//! | `rho:space:queue:hashmap:default` | HashMap + Queue (FIFO) |
//! | `rho:space:stack:hashmap:default` | HashMap + Stack (LIFO) |
//! | `rho:space:set:hashmap:default` | HashMap + Set (idempotent) |
//! | `rho:space:cell:hashmap:default` | HashMap + Cell (exactly-once) |
//! | `rho:space:priorityqueue:hashmap:default` | HashMap + PriorityQueue |
//! | `rho:space:vectordb:hashmap:default` | HashMap + VectorDB |
//!
//! ### PathMap Outer (Hierarchical with Prefix Aggregation)
//! | URN | Description |
//! |-----|-------------|
//! | `rho:space:bag:pathmap:default` | PathMap + Bag |
//! | `rho:space:queue:pathmap:default` | PathMap + Queue (FIFO) |
//! | `rho:space:stack:pathmap:default` | PathMap + Stack (LIFO) |
//! | `rho:space:set:pathmap:default` | PathMap + Set (idempotent) |
//! | `rho:space:cell:pathmap:default` | PathMap + Cell (exactly-once) |
//! | `rho:space:priorityqueue:pathmap:default` | PathMap + PriorityQueue |
//!
//! ### Array Outer (Fixed Size)
//! | URN | Description |
//! |-----|-------------|
//! | `rho:space:bag:array:default` | Array + Bag |
//! | `rho:space:queue:array:default` | Array + Queue |
//! | `rho:space:stack:array:default` | Array + Stack |
//! | `rho:space:set:array:default` | Array + Set |
//! | `rho:space:cell:array:default` | Array + Cell |
//! | `rho:space:priorityqueue:array:default` | Array + PriorityQueue |
//! | `rho:space:{inner}:array(n,cyclic):default` | With custom size/cyclic params |
//!
//! ### Vector Outer (Unbounded)
//! | URN | Description |
//! |-----|-------------|
//! | `rho:space:bag:vector:default` | Vector + Bag |
//! | `rho:space:queue:vector:default` | Vector + Queue |
//! | `rho:space:stack:vector:default` | Vector + Stack |
//! | `rho:space:set:vector:default` | Vector + Set |
//! | `rho:space:cell:vector:default` | Vector + Cell |
//! | `rho:space:priorityqueue:vector:default` | Vector + PriorityQueue |
//! | `rho:space:vectordb:vector:default` | Vector + VectorDB |
//!
//! ## Invalid Combinations
//!
//! The following combinations are rejected:
//! - VectorDB + PathMap (incompatible - VectorDB needs O(1) lookup)
//! - VectorDB + Array (incompatible)
//! - VectorDB + HashSet (incompatible)
//! - HashSet + anything except Set (Seq qualifier requires Set)
//!
//! # Theory Extensions
//!
//! URNs can include theory annotations to create typed tuple spaces:
//!
//! | URN Extension | Description |
//! |--------------|-------------|
//! | `[theory=Nat]` | Validate data against the Nat type |
//! | `[theory=mettail:file.metta]` | Load theory from a MeTTaIL file |
//! | `[theory=inline:(: Nat Type)]` | Parse inline MeTTaIL theory |
//!
//! # Example
//!
//! ```ignore
//! // Create a typed space that only accepts natural numbers
//! let urn = "rho:space:HashMapBagSpace[theory=Nat]";
//! let (config, _theory_spec) = parse_urn_with_theory(urn);
//!
//! // Create a PathMap + Queue space
//! let config = config_from_urn("rho:space:queue:pathmap:default").unwrap();
//!
//! // Create an Array + Stack space with custom size
//! let config = config_from_urn("rho:space:stack:array(500,true):default").unwrap();
//! ```

pub mod urn;
pub mod config;
pub mod registry;
pub mod theory;

// =============================================================================
// Re-exports for backward compatibility
// =============================================================================

// From urn module
pub use urn::{
    InnerType,
    OuterType,
    Qualifier,
    is_valid_combination,
    all_valid_urns,
    valid_urn_count,
    urn_to_byte_name,
    byte_name_to_urn,
    urn_from_config,
    inner_collection_to_str,
};

// From config module
pub use config::{
    InnerParams,
    OuterParams,
    parse_inner_with_params,
    parse_outer_with_params,
    compute_config,
    config_from_urn,
    config_from_urn_computed,
    parse_inner_collection_type,
    parse_outer_storage_type,
};

// From registry module
pub use registry::{
    SpaceFactory,
    FactoryRegistry,
};

// From theory module
pub use theory::{
    TheorySpec,
    TheoryLoader,
    BuiltinTheoryLoader,
    SharedTheoryLoader,
    parse_urn_with_theory,
    config_from_full_urn,
};
