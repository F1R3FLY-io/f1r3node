// See rspace/src/main/scala/coop/rchain/rspace/Tuplespace.scala

use std::collections::BTreeSet;

use super::errors::RSpaceError;
use super::rspace_interface::{MaybeConsumeResult, MaybeProduceResult};

pub trait Tuplespace<C, P, A, K> {
    /** Searches the store for data matching all the given patterns at the
     * given channels.
     *
     * If no match is found, then the continuation and patterns are put in
     * the store at the given channels.
     *
     * If a match is found, then the continuation is returned along with the
     * matching data.
     *
     * Matching data stored with the `persist` flag set to `true` will not
     * be removed when it is retrieved. See below for more information
     * about using the `persist` flag.
     *
     * '''NOTE''':
     *
     * A call to [[consume]] that is made with the persist flag set to
     * `true` only persists when there is no matching data.
     *
     * This means that in order to make a continuation "stick" in the store,
     * the user will have to continue to call [[consume]] until a `None`
     * is received.
     *
     * @param channels A Seq of channels on which to search for matching
     * data @param patterns A Seq of patterns with which to search for
     * matching data @param continuation A continuation
     * @param persist Whether or not to attempt to persist the data
     */
    fn consume(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<MaybeConsumeResult<C, P, A, K>, RSpaceError>;

    /** Searches the store for a continuation that has patterns that match
     * the given data at the given channel.
     *
     * If no match is found, then the data is put in the store at the given
     * channel.
     *
     * If a match is found, then the continuation is returned along with the
     * matching data.
     *
     * Matching data or continuations stored with the `persist` flag set to
     * `true` will not be removed when they are retrieved. See below for
     * more information about using the `persist` flag.
     *
     * '''NOTE''':
     *
     * A call to [[produce]] that is made with the persist flag set to
     * `true` only persists when there are no matching continuations.
     *
     * This means that in order to make a piece of data "stick" in the
     * store, the user will have to continue to call [[produce]] until a
     * `None` is received.
     *
     * @param channel A channel on which to search for matching
     * continuations and/or store data @param data A piece of data
     * @param persist Whether or not to attempt to persist the data
     */
    fn produce(
        &mut self,
        channel: C,
        data: A,
        persist: bool,
    ) -> Result<MaybeProduceResult<C, P, A, K>, RSpaceError>;

    fn install(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
    ) -> Result<Option<(K, Vec<A>)>, RSpaceError>;
}
