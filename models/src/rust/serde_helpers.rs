use serde::Serializer;

/// Always serialize as an empty byte vec, regardless of actual content.
///
/// Used for `locally_free` fields which are transient analysis data
/// (free-variable bitvectors) that must NOT affect Blake2b256 channel
/// hashes in RSpace. This matches Scala's `AlwaysEqual` semantics
/// where `locally_free` is ignored for equality, hashing, and
/// effectively for serialization-based hashing.
pub fn serialize_as_empty_bytes<S: Serializer>(
    _value: &Vec<u8>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_bytes(&[])
}
