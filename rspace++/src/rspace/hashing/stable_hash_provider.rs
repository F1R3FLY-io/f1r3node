use super::blake2b256_hash::Blake2b256Hash;
use bincode;
use serde::Serialize;

// See rspace/src/main/scala/coop/rchain/rspace/hashing/StableHashProvider.scala
pub fn hash<C: Serialize>(channel: &C) -> Blake2b256Hash {
    let bytes = bincode::serialize(channel).unwrap();
    Blake2b256Hash::new(&bytes)
}

// TODO: Double check the sorting here against scala side
pub fn hash_vec<C: Serialize>(channels: &Vec<C>) -> Vec<Blake2b256Hash> {
    let mut hashes = channels
        .into_iter()
        .map(|channel| {
            let bytes = bincode::serialize(&channel).unwrap();
            Blake2b256Hash::new(&bytes)
        })
        .collect::<Vec<_>>();
    hashes.sort();
    hashes
}

pub fn hash_from_vec<C: Serialize>(channels: &Vec<C>) -> Blake2b256Hash {
    hash_from_hashes(&hash_vec(channels))
}

// TODO: Double check the sorting here against scala side
pub fn hash_from_hashes(channels_hashes: &Vec<Blake2b256Hash>) -> Blake2b256Hash {
    let mut ord_refs: Vec<&Blake2b256Hash> = channels_hashes.iter().collect();
    ord_refs.sort();

    let concatenated: Vec<u8> = ord_refs.iter().flat_map(|h| h.0.iter().copied()).collect();
    Blake2b256Hash::new(&concatenated)
}

// See rspace/src/main/scala/coop/rchain/rspace/hashing/StableHashProvider.scala
pub fn hash_consume<P: Serialize, K: Serialize>(
    mut encoded_channels: Vec<Vec<u8>>,
    patterns: &[P],
    continuation: &K,
    persist: bool,
) -> Blake2b256Hash {
    let mut encoded_patterns = patterns
        .iter()
        .map(|pattern| bincode::serialize(pattern).unwrap())
        .collect::<Vec<_>>();
    encoded_patterns.sort();

    let encoded_continuation = bincode::serialize(continuation).unwrap();
    let encoded_persist = bincode::serialize(&persist).unwrap();

    encoded_channels.extend(encoded_patterns);
    encoded_channels.push(encoded_continuation);
    encoded_channels.push(encoded_persist);

    let encoded = bincode::serialize(&encoded_channels).unwrap();
    Blake2b256Hash::new(&encoded)
}

pub fn hash_produce<A: Serialize>(encoded_channel: Vec<u8>, datum: &A, persist: bool) -> Blake2b256Hash {
    let encoded_datum = bincode::serialize(datum).unwrap();
    let encoded_persist = bincode::serialize(&persist).unwrap();

    let encoded_vec = vec![encoded_channel, encoded_datum, encoded_persist];

    let encoded = bincode::serialize(&encoded_vec).unwrap();
    Blake2b256Hash::new(&encoded)
}
