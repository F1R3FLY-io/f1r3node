// See rspace/src/main/scala/coop/rchain/rspace/util/package.scala

use super::rspace_interface::{ContResult, RSpaceResult};
use super::trace::event::Produce;

pub fn unpack_option<C, P, K: Clone, R: Clone>(
    v: &Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, R>>)>,
) -> Option<(K, Vec<R>)> {
    match v {
        Some(tuple) => Some(unpack_tuple(tuple)),
        None => None,
    }
}

pub fn unpack_produce_option<C, P, K: Clone, R: Clone>(
    v: &Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, R>>, Produce)>,
) -> Option<(K, Vec<R>, Produce)> {
    match v {
        Some(tuple) => Some(unpack_produce_tuple(tuple)),
        None => None,
    }
}

pub fn unpack_tuple<C, P, K: Clone, R: Clone>(
    tuple: &(ContResult<C, P, K>, Vec<RSpaceResult<C, R>>),
) -> (K, Vec<R>) {
    match tuple {
        (ContResult { continuation, .. }, data) => (
            continuation.clone(),
            data.into_iter()
                .map(|result| result.matched_datum.clone())
                .collect(),
        ),
    }
}

pub fn unpack_produce_tuple<C, P, K: Clone, R: Clone>(
    tuple: &(ContResult<C, P, K>, Vec<RSpaceResult<C, R>>, Produce),
) -> (K, Vec<R>, Produce) {
    match tuple {
        (ContResult { continuation, .. }, data, previous) => (
            continuation.clone(),
            data.into_iter()
                .map(|result| result.matched_datum.clone())
                .collect(),
            previous.clone(),
        ),
    }
}

pub fn unpack_option_with_peek<C, P, K, R>(
    v: Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, R>>)>,
) -> Option<(K, Vec<(C, R, R, bool)>, bool)> {
    v.map(unpack_tuple_with_peek)
}

pub fn unpack_tuple_with_peek<C, P, K, R>(
    v: (ContResult<C, P, K>, Vec<RSpaceResult<C, R>>),
) -> (K, Vec<(C, R, R, bool)>, bool) {
    let (cont_result, data) = v;

    let ContResult { continuation, .. } = cont_result;

    let mapped_data: Vec<(C, R, R, bool)> = data
        .into_iter()
        .map(|d| (d.channel, d.matched_datum, d.removed_datum, d.persistent))
        .collect();

    (continuation, mapped_data, cont_result.peek)
}
