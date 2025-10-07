// See rspace/src/main/scala/coop/rchain/rspace/ReportingTransformer.scala

use super::reporting_rspace::{ReportingComm, ReportingConsume, ReportingEvent, ReportingProduce};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// The purpose of the reportingTransformer is to create transformer to transform reporting events like
/// `ReportingProduce`, `ReportingConsume` and `ReportingComm`(see coop.rchain.rspace.ReportingRspace) into
/// something else which is more readable or easier to interact.
pub trait ReportingTransformer<C, P, A, K, E>
where
    C: Clone + Debug,
    P: Clone + Debug,
    A: Clone + Debug,
    K: Clone + Debug,
{
    fn serialize_consume(&self, rc: &ReportingConsume<C, P, K>) -> E;

    fn serialize_produce(&self, rp: &ReportingProduce<C, A>) -> E;

    fn serialize_comm(&self, rcm: &ReportingComm<C, P, A, K>) -> E;

    fn transform_event(&self, re: &ReportingEvent<C, P, A, K>) -> E {
        match re {
            ReportingEvent::ReportingComm(comm) => self.serialize_comm(comm),
            ReportingEvent::ReportingConsume(cons) => self.serialize_consume(cons),
            ReportingEvent::ReportingProduce(prod) => self.serialize_produce(prod),
        }
    }
}

/// Specialized trait for string transformers that return concrete types instead of RhoEvent
/// This matches the Scala pattern where ReportingEventStringTransformer methods return
/// concrete types (RhoConsume, RhoProduce, RhoComm) instead of the generic RhoEvent
pub trait ReportingStringTransformer<C, P, A, K>
where
    C: Clone + Debug,
    P: Clone + Debug,
    A: Clone + Debug,
    K: Clone + Debug,
{
    fn serialize_consume(&self, rc: &ReportingConsume<C, P, K>) -> RhoConsume;

    fn serialize_produce(&self, rp: &ReportingProduce<C, A>) -> RhoProduce;

    fn serialize_comm(&self, rcm: &ReportingComm<C, P, A, K>) -> RhoComm;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RhoEvent {
    RhoComm(RhoComm),
    RhoProduce(RhoProduce),
    RhoConsume(RhoConsume),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RhoComm {
    pub consume: RhoConsume,
    pub produces: Vec<RhoProduce>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RhoProduce {
    pub channel: String,
    pub data: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RhoConsume {
    pub channels: String,
    pub patterns: String,
    pub continuation: String,
}

/// String transformer for ReportingEvents
pub struct ReportingEventStringTransformer<C, P, A, K, F1, F2, F3, F4>
where
    C: Clone + Debug,
    P: Clone + Debug,
    A: Clone + Debug,
    K: Clone + Debug,
    F1: Fn(&C) -> String,
    F2: Fn(&P) -> String,
    F3: Fn(&A) -> String,
    F4: Fn(&K) -> String,
{
    serialize_c: F1,
    serialize_p: F2,
    serialize_a: F3,
    serialize_k: F4,
    // PhantomData is REQUIRED - tells Rust compiler that this struct "conceptually owns" C,P,A,K types
    _phantom: std::marker::PhantomData<(C, P, A, K)>,
}

impl<C, P, A, K, F1, F2, F3, F4> ReportingEventStringTransformer<C, P, A, K, F1, F2, F3, F4>
where
    C: Clone + Debug,
    P: Clone + Debug,
    A: Clone + Debug,
    K: Clone + Debug,
    F1: Fn(&C) -> String,
    F2: Fn(&P) -> String,
    F3: Fn(&A) -> String,
    F4: Fn(&K) -> String,
{
    pub fn new(serialize_c: F1, serialize_p: F2, serialize_a: F3, serialize_k: F4) -> Self {
        Self {
            serialize_c,
            serialize_p,
            serialize_a,
            serialize_k,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Implementation of ReportingStringTransformer that returns concrete types
impl<C, P, A, K, F1, F2, F3, F4> ReportingStringTransformer<C, P, A, K>
    for ReportingEventStringTransformer<C, P, A, K, F1, F2, F3, F4>
where
    C: Clone + Debug,
    P: Clone + Debug,
    A: Clone + Debug,
    K: Clone + Debug,
    F1: Fn(&C) -> String,
    F2: Fn(&P) -> String,
    F3: Fn(&A) -> String,
    F4: Fn(&K) -> String,
{
    fn serialize_consume(&self, rc: &ReportingConsume<C, P, K>) -> RhoConsume {
        let k = (self.serialize_k)(&rc.continuation);
        let chs = format!(
            "[{}]",
            rc.channels
                .iter()
                .map(|c| (self.serialize_c)(c))
                .collect::<Vec<_>>()
                .join(";")
        );
        let ps = format!(
            "[{}]",
            rc.patterns
                .iter()
                .map(|p| (self.serialize_p)(p))
                .collect::<Vec<_>>()
                .join(";")
        );

        RhoConsume {
            channels: chs,
            patterns: ps,
            continuation: k,
        }
    }

    fn serialize_produce(&self, rp: &ReportingProduce<C, A>) -> RhoProduce {
        let d = (self.serialize_a)(&rp.data);
        let ch = (self.serialize_c)(&rp.channel);

        RhoProduce {
            channel: ch,
            data: d,
        }
    }

    fn serialize_comm(&self, rcm: &ReportingComm<C, P, A, K>) -> RhoComm {
        let consume = ReportingStringTransformer::serialize_consume(self, &rcm.consume);
        let produces = rcm
            .produces
            .iter()
            .map(|p| ReportingStringTransformer::serialize_produce(self, p))
            .collect();

        RhoComm { consume, produces }
    }
}

/// Implementation of the main ReportingTransformer trait
impl<C, P, A, K, F1, F2, F3, F4> ReportingTransformer<C, P, A, K, RhoEvent>
    for ReportingEventStringTransformer<C, P, A, K, F1, F2, F3, F4>
where
    C: Clone + Debug,
    P: Clone + Debug,
    A: Clone + Debug,
    K: Clone + Debug,
    F1: Fn(&C) -> String,
    F2: Fn(&P) -> String,
    F3: Fn(&A) -> String,
    F4: Fn(&K) -> String,
{
    fn serialize_consume(&self, rc: &ReportingConsume<C, P, K>) -> RhoEvent {
        RhoEvent::RhoConsume(ReportingStringTransformer::serialize_consume(self, rc))
    }

    fn serialize_produce(&self, rp: &ReportingProduce<C, A>) -> RhoEvent {
        RhoEvent::RhoProduce(ReportingStringTransformer::serialize_produce(self, rp))
    }

    fn serialize_comm(&self, rcm: &ReportingComm<C, P, A, K>) -> RhoEvent {
        RhoEvent::RhoComm(ReportingStringTransformer::serialize_comm(self, rcm))
    }
}
