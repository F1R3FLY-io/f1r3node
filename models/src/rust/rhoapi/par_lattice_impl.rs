use pathmap::ring::{Lattice, DistributiveLattice, AlgebraicResult, SELF_IDENT};
use crate::rhoapi::Par;

/// Left-biased Lattice implementation for Par.
/// Uses Identity to avoid cloning - signals to PathMap to keep the existing value unchanged.
impl Lattice for Par {
    fn pjoin(&self, _other: &Self) -> AlgebraicResult<Self> {
        // Left-bias: keep self unchanged, avoiding clone
        AlgebraicResult::Identity(SELF_IDENT)
    }
    fn pmeet(&self, _other: &Self) -> AlgebraicResult<Self> {
        // Left-bias: keep self unchanged, avoiding clone
        AlgebraicResult::Identity(SELF_IDENT)
    }
    fn bottom() -> Self {
        Par::default()
    }
}

impl DistributiveLattice for Par {
    fn psubtract(&self, _other: &Self) -> AlgebraicResult<Self> {
        // Left-bias: keep self unchanged, avoiding clone
        AlgebraicResult::Identity(SELF_IDENT)
    }
}
