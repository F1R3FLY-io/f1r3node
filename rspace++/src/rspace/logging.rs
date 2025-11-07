
use super::internal::{ConsumeCandidate, WaitingContinuation};
use super::trace::event::{COMM, Consume, Produce};
use std::collections::BTreeSet;

/// Core logging operations that can be overridden by different RSpace loggers
pub trait RSpaceLogger<C, P: Clone, A: Clone, K: Clone>: Send + Sync {
    fn log_comm(
        &mut self,
        _data_candidates: &Vec<ConsumeCandidate<C, A>>,
        _channels: &Vec<C>,
        _wk: WaitingContinuation<P, K>,
        comm: COMM,
        _label: &str,
    ) -> COMM {
        comm
    }

    fn log_consume(
        &mut self,
        consume_ref: Consume,
        _channels: &Vec<C>,
        _patterns: &Vec<P>,
        _continuation: &K,
        _persist: bool,
        _peeks: &BTreeSet<i32>,
    ) -> Consume {
        consume_ref
    }

    fn log_produce(
        &mut self,
        produce_ref: Produce,
        _channel: &C,
        _data: &A,
        _persist: bool,
    ) -> Produce {
        produce_ref
    }
}

/// Default logger that mirrors current ReplayRSpace behavior (no-op passthrough)
pub struct BasicLogger;

impl BasicLogger {
    pub fn new() -> Self {
        BasicLogger
    }
}

impl<C, P: Clone, A: Clone, K: Clone> RSpaceLogger<C, P, A, K> for BasicLogger {}


