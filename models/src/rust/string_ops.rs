// See models/src/main/scala/coop/rchain/models/StringSyntax.scala
// See shared/src/main/scala/coop/rchain/shared/Base16.scala
pub struct StringOps;

impl StringOps {
    pub fn decode_hex(s: String) -> Option<Vec<u8>> {
        // Match Scala Base16.decode: pad odd-length hex strings with leading zero
        let digits_only: String = s.chars().filter(|c| c.is_ascii_hexdigit()).collect();
        let padded = if digits_only.len() % 2 == 0 {
            digits_only
        } else {
            format!("0{}", digits_only)
        };
        hex::decode(&padded).ok()
    }

    pub fn unsafe_decode_hex(s: String) -> Vec<u8> {
        // Match Scala Base16.unsafeDecode: pad odd-length hex strings with leading zero
        // This matches the behavior in Base16.scala lines 26-28:
        // val padded = if (digitsOnly.length % 2 == 0) digitsOnly else "0" + digitsOnly
        let digits_only: String = s.chars().filter(|c| c.is_ascii_hexdigit()).collect();
        let padded = if digits_only.len() % 2 == 0 {
            digits_only
        } else {
            format!("0{}", digits_only)
        };
        hex::decode(&padded).expect(&format!("Failed to decode hex string: {}", s))
    }
}
