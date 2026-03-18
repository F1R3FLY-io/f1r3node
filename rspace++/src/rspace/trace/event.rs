use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};

use proptest_derive::Arbitrary;
use serde::{Deserialize, Serialize};

use crate::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use crate::rspace::hashing::stable_hash_provider::{hash, hash_consume, hash_produce, hash_vec};
use crate::rspace::internal::ConsumeCandidate;

// See rspace/src/main/scala/coop/rchain/rspace/trace/Event.scala
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum Event {
    Comm(COMM),
    IoEvent(IOEvent),
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub enum IOEvent {
    Produce(Produce),
    Consume(Consume),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct COMM {
    pub consume: Consume,
    pub produces: Vec<Produce>,
    pub peeks: BTreeSet<i32>,
    pub times_repeated: BTreeMap<Produce, i32>,
}

impl COMM {
    pub fn new<C, A: Clone>(
        data_candidates: &[ConsumeCandidate<C, A>],
        consume_ref: Consume,
        peeks: BTreeSet<i32>,
        produce_counters: impl Fn(&[Produce]) -> BTreeMap<Produce, i32>,
    ) -> Self {
        let mut produce_refs: Vec<Produce> = data_candidates
            .iter()
            .map(|candidate| candidate.datum.source.clone())
            .collect();

        // produce_refs.sort_by(|a, b| {
        //     let a_cloned = a.clone();
        //     let b_cloned = b.clone();
        //     (a_cloned.channel_hash, a_cloned.hash, a.persistent).cmp(&(
        //         b_cloned.channel_hash,
        //         b_cloned.hash,
        //         b.persistent,
        //     ))
        // });
        produce_refs.sort_by(|a, b| {
            a.channel_hash
                .cmp(&b.channel_hash)
                .then_with(|| a.hash.cmp(&b.hash))
                .then_with(|| a.persistent.cmp(&b.persistent))
        });
        // produce_refs.sort_by_key(|p| {
        //     let p_cloned = p.clone();
        //     (p_cloned.channel_hash, p_cloned.hash, p.persistent)
        // });

        COMM {
            consume: consume_ref,
            produces: produce_refs.clone(),
            peeks,
            times_repeated: produce_counters(&produce_refs),
        }
    }
}

// Needed for 'counter' crate
impl Hash for COMM {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.consume.hash(state);
        self.produces.hash(state);
        self.peeks.hash(state);

        for (key, value) in &self.times_repeated {
            key.hash(state);
            value.hash(state);
        }
    }
}

// The 'Arbitrary' macro is needed here for proptest in hot_store_spec.rs
// The 'Default' macro is needed here for hot_store_spec.rs
//
// Custom PartialEq/Eq/Hash/Ord: identity is determined solely by the `hash`
// field (a cryptographic hash of channel + data + persist). Metadata fields
// like `is_deterministic`, `output_value`, and `failed` are set after creation
// (e.g. via mark_as_non_deterministic) and must NOT affect identity.
#[derive(
    Serialize,
    Deserialize,
    Clone,
    Debug,
    Arbitrary,
    Default,
)]
pub struct Produce {
    pub channel_hash: Blake2b256Hash,
    pub hash: Blake2b256Hash,
    pub persistent: bool,
    pub is_deterministic: bool,
    pub output_value: Vec<Vec<u8>>,
    /// Indicates whether this produce event represents a failed
    /// non-deterministic process. Used for replay safety of external
    /// service calls (OpenAI, Ollama, gRPC).
    pub failed: bool,
}

impl PartialEq for Produce {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
    }
}

impl Eq for Produce {}

impl std::hash::Hash for Produce {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

impl Ord for Produce {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.hash.cmp(&other.hash)
    }
}

impl PartialOrd for Produce {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Produce {
    pub fn create<C: Serialize, A: Serialize>(channel: &C, datum: &A, persistent: bool) -> Produce {
        let channel_hash = hash(channel);
        let hash = hash_produce(channel_hash.bytes(), datum, persistent);
        Produce {
            channel_hash,
            hash,
            persistent,
            is_deterministic: true,
            output_value: vec![],
            failed: false,
        }
    }

    pub fn new(channel_hash: Blake2b256Hash, hash: Blake2b256Hash, persistent: bool) -> Produce {
        Produce {
            channel_hash,
            hash,
            persistent,
            is_deterministic: true,
            output_value: vec![],
            failed: false,
        }
    }

    pub fn mark_as_non_deterministic(self, previous: Vec<Vec<u8>>) -> Self {
        Produce {
            is_deterministic: false,
            output_value: previous,
            ..self
        }
    }

    /// Mark this produce event as failed, indicating a non-deterministic
    /// process failure. Used to record failures from external service calls
    /// (OpenAI, Ollama, gRPC) so replay can correctly handle them without
    /// re-executing the external call.
    pub fn with_error(&self) -> Self {
        Produce {
            failed: true,
            ..self.clone()
        }
    }
}

// The 'Arbitrary' macro is needed here for proptest in hot_store_spec.rs
// The 'Default' macro is needed here for hot_store_spec.rs
#[derive(
    Serialize,
    Deserialize,
    Clone,
    Debug,
    PartialEq,
    Eq,
    Arbitrary,
    Hash,
    Default,
    Ord,
    PartialOrd
)]
pub struct Consume {
    pub channel_hashes: Vec<Blake2b256Hash>,
    pub hash: Blake2b256Hash,
    pub persistent: bool,
}

impl Consume {
    pub fn create<C: Serialize, P: Serialize, K: Serialize>(
        channels: &Vec<C>,
        patterns: &Vec<P>,
        continuation: &K,
        persistent: bool,
    ) -> Consume {
        let channel_hashes = hash_vec(channels);
        let channels_encoded_sorted: Vec<Vec<u8>> =
            channel_hashes.iter().map(|hash| hash.bytes()).collect();
        let hash = hash_consume(channels_encoded_sorted, &patterns, &continuation, persistent);
        Consume {
            channel_hashes,
            hash,
            persistent,
        }
    }
}
