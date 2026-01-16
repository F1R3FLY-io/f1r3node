//! Type definitions for spaces module.

pub mod bounds;
pub mod id;
pub mod collections;

pub use bounds::*;
pub use id::SpaceId;
pub use collections::{InnerCollectionType, OuterStorageType};
