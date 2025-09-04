use super::hashing::blake2b256_hash::Blake2b256Hash;
use crate::rspace::internal::{Datum, WaitingContinuation};

// See rspace/src/main/scala/coop/rchain/rspace/HotStoreTrieAction.scala
#[derive(Debug, Clone)]
pub enum HotStoreTrieAction<C: Clone, P: Clone, A: Clone, K: Clone> {
    TrieInsertAction(TrieInsertAction<C, P, A, K>),
    TrieDeleteAction(TrieDeleteAction),
}

#[derive(Debug, Clone)]
pub enum TrieInsertAction<C: Clone, P: Clone, A: Clone, K: Clone> {
    TrieInsertProduce(TrieInsertProduce<A>),
    TrieInsertJoins(TrieInsertJoins<C>),
    TrieInsertConsume(TrieInsertConsume<P, K>),
    TrieInsertBinaryProduce(TrieInsertBinaryProduce),
    TrieInsertBinaryJoins(TrieInsertBinaryJoins),
    TrieInsertBinaryConsume(TrieInsertBinaryConsume),
}

#[derive(Debug, Clone)]
pub struct TrieInsertProduce<A: Clone> {
    pub hash: Blake2b256Hash,
    pub data: Vec<Datum<A>>,
}

impl<A: Clone> TrieInsertProduce<A> {
    pub fn new(hash: Blake2b256Hash, data: Vec<Datum<A>>) -> Self {
        TrieInsertProduce { hash, data }
    }
}

#[derive(Debug, Clone)]
pub struct TrieInsertJoins<C: Clone> {
    pub hash: Blake2b256Hash,
    pub joins: Vec<Vec<C>>,
}

impl<C: Clone> TrieInsertJoins<C> {
    pub fn new(hash: Blake2b256Hash, joins: Vec<Vec<C>>) -> Self {
        TrieInsertJoins { hash, joins }
    }
}

#[derive(Debug, Clone)]
pub struct TrieInsertConsume<P: Clone, K: Clone> {
    pub hash: Blake2b256Hash,
    pub continuations: Vec<WaitingContinuation<P, K>>,
}

impl<P: Clone, K: Clone> TrieInsertConsume<P, K> {
    pub fn new(hash: Blake2b256Hash, continuations: Vec<WaitingContinuation<P, K>>) -> Self {
        TrieInsertConsume {
            hash,
            continuations,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrieInsertBinaryProduce {
    pub hash: Blake2b256Hash,
    pub data: Vec<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct TrieInsertBinaryJoins {
    pub hash: Blake2b256Hash,
    pub joins: Vec<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct TrieInsertBinaryConsume {
    pub hash: Blake2b256Hash,
    pub continuations: Vec<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub enum TrieDeleteAction {
    TrieDeleteProduce(TrieDeleteProduce),
    TrieDeleteJoins(TrieDeleteJoins),
    TrieDeleteConsume(TrieDeleteConsume),
}

#[derive(Debug, Clone)]
pub struct TrieDeleteProduce {
    pub hash: Blake2b256Hash,
}

impl TrieDeleteProduce {
    pub fn new(hash: Blake2b256Hash) -> Self {
        TrieDeleteProduce { hash }
    }
}

#[derive(Debug, Clone)]
pub struct TrieDeleteJoins {
    pub hash: Blake2b256Hash,
}

impl TrieDeleteJoins {
    pub fn new(hash: Blake2b256Hash) -> Self {
        TrieDeleteJoins { hash }
    }
}

#[derive(Debug, Clone)]
pub struct TrieDeleteConsume {
    pub hash: Blake2b256Hash,
}

impl TrieDeleteConsume {
    pub fn new(hash: Blake2b256Hash) -> Self {
        TrieDeleteConsume { hash }
    }
}
