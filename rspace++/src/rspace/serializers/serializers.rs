// See rspace/src/main/scala/coop/rchain/rspace/serializers/ScodecSerialize.scala

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

use crate::rspace::internal::{Datum, WaitingContinuation};

pub fn encode_datum<A: Clone + Serialize>(datum: &Datum<A>) -> Vec<u8> {
    bincode::serialize(datum).expect("Serializers: Unable to serialize datum")
}

pub fn decode_datum<A: Clone + for<'a> Deserialize<'a>>(vec_bytes: &Vec<u8>) -> Datum<A> {
    bincode::deserialize(vec_bytes).expect("Serializers: Unable to deserialize datum")
}

pub fn encode_datums<A: Clone + Serialize>(datums: &Vec<Datum<A>>) -> Vec<u8> {
    let mut serialized_datums: Vec<Vec<u8>> = datums
        .iter()
        .map(|datum| bincode::serialize(datum).expect("Serializers: Unable to serialize datum"))
        .collect();

    serialized_datums.sort_by(compare_byte_vectors);
    bincode::serialize(&serialized_datums).expect("Serializers: Unable to serialize datums")
}

pub fn decode_datums<A: Clone + for<'a> Deserialize<'a>>(vec_bytes: &Vec<u8>) -> Vec<Datum<A>> {
    let encoded_datums: Vec<Vec<u8>> =
        bincode::deserialize(vec_bytes).expect("Serializers: Unable to deserialize datums");

    let datums = encoded_datums
        .iter()
        .map(|bytes| bincode::deserialize(bytes).expect("Serializers: Unable to deserialize datum"))
        .collect();
    datums
}

pub fn encode_continuations<P: Clone + Serialize, K: Clone + Serialize>(
    conts: &Vec<WaitingContinuation<P, K>>,
) -> Vec<u8> {
    let mut serialized_continuations: Vec<Vec<u8>> = conts
        .iter()
        .map(|wk| bincode::serialize(wk).expect("Serializers: Unable to serialize continuation"))
        .collect();

    serialized_continuations.sort_by(compare_byte_vectors);
    bincode::serialize(&serialized_continuations)
        .expect("Serializers: Unable to serialize continuations")
}

pub fn decode_continuations<P, K>(vec_bytes: &Vec<u8>) -> Vec<WaitingContinuation<P, K>>
where
    P: Clone + for<'a> Deserialize<'a>,
    K: Clone + for<'a> Deserialize<'a>,
{
    let encoded_continuations: Vec<Vec<u8>> =
        bincode::deserialize(vec_bytes).expect("Serializers: Unable to deserialize continuations");

    let continuations = encoded_continuations
        .iter()
        .map(|bytes| {
            bincode::deserialize(bytes).expect("Serializers: Unable to deserialize continuation")
        })
        .collect();
    continuations
}

pub fn encode_joins<C: Clone + Serialize>(joins: &Vec<Vec<C>>) -> Vec<u8> {
    let mut serialized_joins: Vec<Vec<u8>> = joins
        .iter()
        .map(|datum| bincode::serialize(datum).expect("Serializers: Unable to serialize join"))
        .collect();

    serialized_joins.sort_by(compare_byte_vectors);
    bincode::serialize(&serialized_joins).expect("Serializers: Unable to serialize joins")
}

pub fn decode_joins<C: Clone + for<'a> Deserialize<'a>>(vec_bytes: &Vec<u8>) -> Vec<Vec<C>> {
    let encoded_joins: Vec<Vec<u8>> =
        bincode::deserialize(vec_bytes).expect("Serializers: Unable to deserialize joins");

    let joins = encoded_joins
        .iter()
        .map(|bytes| bincode::deserialize(bytes).expect("Serializers: Unable to deserialize join"))
        .collect();
    joins
}

pub fn encode_binary(binary: &Vec<Vec<u8>>) -> Vec<u8> {
    let mut serialized_binary: Vec<Vec<u8>> = binary
        .iter()
        .map(|datum| bincode::serialize(datum).expect("Serializers: Unable to serialize binary"))
        .collect();

    serialized_binary.sort_by(compare_byte_vectors);
    bincode::serialize(&serialized_binary).expect("Serializers: Unable to serialize binaries")
}

// See rspace/src/main/scala/coop/rchain/rspace/util/package.scala
fn compare_byte_vectors(a: &Vec<u8>, b: &Vec<u8>) -> Ordering {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.cmp(y))
        .find(|&ord| ord != Ordering::Equal)
        .unwrap_or(a.len().cmp(&b.len()))
}
