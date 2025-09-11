/**
 * See rspace/src/main/scala/coop/rchain/rspace/Match.scala
 *
 * Type trait for matching patterns with data.
 *
 * @tparam P A type representing patterns
 * @tparam A A type representing data and match result
 */
pub trait Match<P, A>: Send + Sync {
    fn get(&self, p: P, a: A) -> Option<A>;
}
