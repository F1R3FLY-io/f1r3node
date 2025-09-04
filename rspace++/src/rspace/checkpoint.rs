use super::trace::Log;
use super::trace::event::Produce;
use super::{hashing::blake2b256_hash::Blake2b256Hash, hot_store::HotStoreState};
use std::collections::BTreeMap;
use std::hash::Hash;

// See rspace/src/main/scala/coop/rchain/rspace/Checkpoint.scala
#[derive(Clone)]
pub struct SoftCheckpoint<C: Eq + Hash, P: Clone, A: Clone, K: Clone> {
    pub cache_snapshot: HotStoreState<C, P, A, K>,
    pub log: Log,
    pub produce_counter: BTreeMap<Produce, i32>,
}

pub struct Checkpoint {
    pub root: Blake2b256Hash,
    pub log: Log,
}
