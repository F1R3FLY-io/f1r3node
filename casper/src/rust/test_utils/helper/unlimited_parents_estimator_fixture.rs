// See casper/src/test/scala/coop/rchain/casper/helper/UnlimitedParentsEstimatorFixture.scala
// Moved from casper/tests/helper/unlimited_parents_estimator_fixture.rs to casper/src/rust/test_utils/helper/unlimited_parents_estimator_fixture.rs
// All imports fixed for library crate context

use crate::rust::estimator::Estimator;

pub struct UnlimitedParentsEstimatorFixture;

impl UnlimitedParentsEstimatorFixture {
    /// Create estimator like in Scala: Estimator[Task](Estimator.UnlimitedParents, None)
    /// where Estimator.UnlimitedParents = Int.MaxValue
    pub fn create_estimator() -> Estimator {
        Estimator::apply(Estimator::UNLIMITED_PARENTS, None)
    }
}


