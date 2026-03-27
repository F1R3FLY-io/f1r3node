use serde::Serializer;

/// Always serialize as an empty byte vec, regardless of actual content.
///
/// Used for `locally_free` fields which are transient analysis data (free-variable
/// bit-vectors) that must NOT affect Blake2b256 channel hashes in RSpace.
/// The field position is preserved in the bincode format (unlike `skip_serializing`),
/// but the content is always empty, ensuring consistent hashing between validator
/// and observer nodes.
pub fn serialize_as_empty_bytes<S: Serializer>(
    _value: &Vec<u8>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_bytes(&[])
}
