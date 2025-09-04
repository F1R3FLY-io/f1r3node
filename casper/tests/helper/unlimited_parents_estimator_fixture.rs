// See casper/src/test/scala/coop/rchain/casper/helper/UnlimitedParentsEstimatorFixture.scala

use casper::rust::estimator::Estimator;

pub struct UnlimitedParentsEstimatorFixture;

impl UnlimitedParentsEstimatorFixture {
    /// Create estimator like in Scala: Estimator[Task](Estimator.UnlimitedParents, None)
    /// where Estimator.UnlimitedParents = Int.MaxValue
    pub fn create_estimator() -> Estimator {
        Estimator::apply(Estimator::UNLIMITED_PARENTS, None)
    }
}
