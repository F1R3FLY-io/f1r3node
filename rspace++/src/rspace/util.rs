// See rspace/src/main/scala/coop/rchain/rspace/util/package.scala

use super::{
    rspace_interface::{ContResult, RSpaceResult},
    trace::event::Produce,
};

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

/// Result tuple with suffix key for PathMap prefix semantics.
///
/// When data is consumed at a prefix path (e.g., `@[0,1]`) and the actual data
/// is at a descendant path (e.g., `@[0,1,2]`), the suffix key contains the
/// path elements between them (e.g., `[2]`).
///
/// Per the "Reifying RSpaces" spec (lines 163-184), the suffix key should be
/// prepended to the data: data at `@[0,1,2]` becomes `[2, data]` at `@[0,1]`.
pub type RSpaceResultWithSuffix<C, R> = (C, R, R, bool, Option<Vec<u8>>);

/// Unpack consume result with peek flag and suffix keys.
///
/// Returns data tuples with suffix keys for PathMap prefix matching.
/// For exact matches (non-PathMap or same path), suffix_key is `None`.
pub fn unpack_option_with_peek_and_suffix<C, P, K, R>(
    v: Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, R>>)>,
) -> Option<(K, Vec<RSpaceResultWithSuffix<C, R>>, bool)> {
    v.map(unpack_tuple_with_peek_and_suffix)
}

/// Unpack consume result tuple with peek flag and suffix keys.
///
/// # Returns
/// - Continuation
/// - Vec of (channel, matched_datum, removed_datum, persistent, suffix_key)
/// - Peek flag
pub fn unpack_tuple_with_peek_and_suffix<C, P, K, R>(
    v: (ContResult<C, P, K>, Vec<RSpaceResult<C, R>>),
) -> (K, Vec<RSpaceResultWithSuffix<C, R>>, bool) {
    let (cont_result, data) = v;

    let ContResult { continuation, .. } = cont_result;

    let mapped_data: Vec<RSpaceResultWithSuffix<C, R>> = data
        .into_iter()
        .map(|d| (d.channel, d.matched_datum, d.removed_datum, d.persistent, d.suffix_key))
        .collect();

    (continuation, mapped_data, cont_result.peek)
}
