//! Space Factory Registry
//!
//! This module defines the SpaceFactory trait and FactoryRegistry for creating
//! space instances from configuration.

use super::super::errors::SpaceError;
use super::super::types::{ChannelBound, ContinuationBound, DataBound, PatternBound, SpaceConfig, SpaceId};

/// Trait for space factories.
///
/// Each factory knows how to create a specific type of space from configuration.
pub trait SpaceFactory<C, P, A, K>: Send + Sync
where
    C: ChannelBound,
    P: PatternBound,
    A: DataBound,
    K: ContinuationBound,
{
    /// The type of space agent created by this factory.
    type Agent: super::super::agent::SpaceAgent<C, P, A, K>;

    /// Create a new space with the given configuration.
    fn create(&self, space_id: SpaceId, config: &SpaceConfig) -> Result<Self::Agent, SpaceError>;

    /// Get the URN for spaces created by this factory.
    fn urn(&self) -> &'static str;

    /// Get a description of this factory.
    fn description(&self) -> &'static str;
}

/// Registry of all available space factories.
///
/// Maps URNs to factory implementations.
pub struct FactoryRegistry<C, P, A, K>
where
    C: ChannelBound,
    P: PatternBound,
    A: DataBound,
    K: ContinuationBound,
{
    factories: Vec<Box<dyn SpaceFactory<C, P, A, K, Agent = Box<dyn super::super::agent::SpaceAgent<C, P, A, K>>>>>,
}

impl<C, P, A, K> Default for FactoryRegistry<C, P, A, K>
where
    C: ChannelBound,
    P: PatternBound,
    A: DataBound,
    K: ContinuationBound,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<C, P, A, K> FactoryRegistry<C, P, A, K>
where
    C: ChannelBound,
    P: PatternBound,
    A: DataBound,
    K: ContinuationBound,
{
    /// Create a new empty factory registry.
    pub fn new() -> Self {
        FactoryRegistry {
            factories: Vec::new(),
        }
    }

    /// Get the list of all registered URNs.
    pub fn registered_urns(&self) -> Vec<&'static str> {
        self.factories.iter().map(|f| f.urn()).collect()
    }
}
