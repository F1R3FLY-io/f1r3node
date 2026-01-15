//! PathMap Prefix Aggregation and Path Encoding
//!
//! This module provides the path encoding scheme for PathMap prefix semantics
//! with Rholang channels. It supports converting between Rholang Par expressions
//! and byte paths for hierarchical channel indexing.

use std::fmt;

use models::rhoapi::{expr::ExprInstance, EList, Expr, Par};

// ==========================================================================
// PathMap Prefix Aggregation Types
// ==========================================================================

/// Suffix path stripped during prefix aggregation.
///
/// When data at path `@[0, 1, 2]` is viewed from prefix `@[0, 1]`, the suffix
/// is `[2]`. This suffix becomes part of the aggregated data as a key.
///
/// # Spec Reference
/// From "Reifying RSpaces" lines 159-192:
/// ```text
/// @[0, 1, 2]!({|"hi"|}) | @[0, 1, 2]!({|"hello"|}) | @[0, 1, 3]!({|"there"|})
/// = @[0, 1]!({|[2, "hi"], [2, "hello"], [3, "there"]|})
/// ```
///
/// The suffix `[2]` is prepended to the data when viewed at `@[0, 1]`.
pub type SuffixKey = Vec<u8>;

/// Aggregated data item with its suffix key for PathMap prefix semantics.
///
/// When consuming at a prefix path, data from descendant paths is aggregated
/// with their relative suffix keys attached. This allows pattern matching
/// to distinguish data from different child paths.
///
/// # Example
/// ```ignore
/// // Data at @[0, 1, 2] viewed at @[0, 1]:
/// AggregatedDatum {
///     suffix_key: vec![2],  // The suffix [2]
///     data: "hi",
///     persist: false,
/// }
/// ```
///
/// # Formal Correspondence
/// - `PathMapStore.v`: `send_visible_from_prefix` theorem
/// - `PathMapQuantale.v`: Path concatenation properties
#[derive(Clone, Debug, PartialEq)]
pub struct AggregatedDatum<A> {
    /// The suffix path relative to the consuming prefix.
    /// Empty for exact-path matches (data at the same path as the consumer).
    pub suffix_key: SuffixKey,

    /// The actual data being aggregated.
    pub data: A,

    /// Whether this data persists after consumption.
    pub persist: bool,
}

impl<A> AggregatedDatum<A> {
    /// Create a new aggregated datum with the given suffix key.
    pub fn new(suffix_key: SuffixKey, data: A, persist: bool) -> Self {
        AggregatedDatum {
            suffix_key,
            data,
            persist,
        }
    }

    /// Create an aggregated datum with empty suffix (exact path match).
    pub fn exact(data: A, persist: bool) -> Self {
        AggregatedDatum {
            suffix_key: Vec::new(),
            data,
            persist,
        }
    }

    /// Check if this datum is from an exact path match (no suffix).
    pub fn is_exact_match(&self) -> bool {
        self.suffix_key.is_empty()
    }

    /// Get the suffix key length (depth of path difference).
    pub fn suffix_depth(&self) -> usize {
        self.suffix_key.len()
    }

    /// Map the data through a function, preserving suffix and persist.
    pub fn map<B, F: FnOnce(A) -> B>(self, f: F) -> AggregatedDatum<B> {
        AggregatedDatum {
            suffix_key: self.suffix_key,
            data: f(self.data),
            persist: self.persist,
        }
    }
}

impl<A: Clone> AggregatedDatum<A> {
    /// Get the data, cloning it.
    pub fn data_cloned(&self) -> A {
        self.data.clone()
    }
}

impl<A: fmt::Display> fmt::Display for AggregatedDatum<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.suffix_key.is_empty() {
            write!(f, "{}", self.data)
        } else {
            write!(f, "[{:?}, {}]", self.suffix_key, self.data)
        }
    }
}

// ==========================================================================
// Path Prefix Utilities
// ==========================================================================

/// Compute the suffix of a child path relative to a prefix path.
///
/// Returns `None` if `prefix` is not a prefix of `child`.
///
/// # Example
/// ```ignore
/// assert_eq!(get_path_suffix(&[0, 1], &[0, 1, 2, 3]), Some(vec![2, 3]));
/// assert_eq!(get_path_suffix(&[0, 1], &[0, 2]), None);
/// assert_eq!(get_path_suffix(&[0, 1], &[0, 1]), Some(vec![]));
/// ```
pub fn get_path_suffix(prefix: &[u8], child: &[u8]) -> Option<SuffixKey> {
    if child.len() >= prefix.len() && child.starts_with(prefix) {
        Some(child[prefix.len()..].to_vec())
    } else {
        None
    }
}

/// Generate all prefixes of a path, from shortest to longest.
///
/// # Example
/// ```ignore
/// assert_eq!(
///     path_prefixes(&[0, 1, 2]),
///     vec![vec![0], vec![0, 1], vec![0, 1, 2]]
/// );
/// ```
pub fn path_prefixes(path: &[u8]) -> Vec<Vec<u8>> {
    (1..=path.len())
        .map(|len| path[..len].to_vec())
        .collect()
}

/// Check if `prefix` is a prefix of `path`.
pub fn is_path_prefix(prefix: &[u8], path: &[u8]) -> bool {
    path.len() >= prefix.len() && path.starts_with(prefix)
}

/// Get byte offsets at element boundaries in a tagged path.
///
/// For tagged paths, each element has variable length:
/// - INTEGER (0x01): 9 bytes (tag + 8-byte i64)
/// - STRING (0x02): 1 + varint_len + data_len bytes
/// - BYTE_ARRAY (0x03): 1 + varint_len + data_len bytes
///
/// Returns byte offsets at element boundaries (after each complete element).
/// Only complete elements are included; partial elements at the end are ignored.
///
/// # Example
/// ```ignore
/// // Integer path [0, 1, 2] = 27 bytes (3 integers × 9 bytes each)
/// let path = par_to_path(&create_int_elist_par(vec![0, 1, 2])).unwrap();
/// let boundaries = path_element_boundaries(&path);
/// assert_eq!(boundaries, vec![9, 18, 27]);
///
/// // Generates prefixes at element boundaries:
/// // - path[..9]  = @[0]
/// // - path[..18] = @[0, 1]
/// // - path[..27] = @[0, 1, 2]
/// ```
pub fn path_element_boundaries(path: &[u8]) -> Vec<usize> {
    let mut boundaries = Vec::new();
    let mut offset = 0;

    while offset < path.len() {
        if let Some(element_size) = get_element_size(&path[offset..]) {
            offset += element_size;
            boundaries.push(offset);
        } else {
            break; // Invalid or incomplete element, stop parsing
        }
    }

    boundaries
}

/// Get the size of a path element without fully decoding it.
///
/// Returns `Some(size)` if a valid element starts at the given bytes,
/// `None` if the element is invalid or incomplete.
fn get_element_size(bytes: &[u8]) -> Option<usize> {
    if bytes.is_empty() {
        return None;
    }

    let tag = bytes[0];
    let rest = &bytes[1..];

    match tag {
        path_tags::INTEGER => {
            // INTEGER: tag (1) + i64 LE (8) = 9 bytes
            if rest.len() >= 8 {
                Some(9)
            } else {
                None // Incomplete integer
            }
        }
        path_tags::STRING | path_tags::BYTE_ARRAY => {
            // STRING or BYTE_ARRAY: tag (1) + varint length + data
            let (len, varint_size) = decode_varint_size(rest)?;
            let total = 1 + varint_size + len;
            if bytes.len() >= total {
                Some(total)
            } else {
                None // Incomplete string/byte array
            }
        }
        _ => None, // Unknown tag
    }
}

/// Decode a varint and return just the value and bytes consumed.
/// Helper for `get_element_size` to avoid duplicating varint logic.
fn decode_varint_size(bytes: &[u8]) -> Option<(usize, usize)> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    let mut consumed = 0;

    for &byte in bytes {
        consumed += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some((result as usize, consumed));
        }
        shift += 7;
        if shift > 63 {
            return None; // Overflow protection
        }
    }
    None // Incomplete varint
}

// ==========================================================================
// Par-to-Path Conversion (for PathMap prefix semantics with Rholang)
// ==========================================================================
//
// Path Encoding Scheme (Varint + MORK Style)
// ------------------------------------------
// Each path element is encoded with a type tag followed by data:
//
//   Tag bytes:
//     0x01 = Integer (i64, 8 bytes little-endian)
//     0x02 = String (varint length + UTF-8 bytes)
//     0x03 = ByteArray (varint length + raw bytes)
//
//   Varint encoding: 7 bits per byte, high bit (0x80) = continuation flag
//
//   Examples:
//     Integer 42     → [0x01, 42, 0, 0, 0, 0, 0, 0, 0]  (tag + i64 LE)
//     String "auth"  → [0x02, 4, 'a', 'u', 't', 'h']     (tag + varint_len + utf8)
//     String (300 chars) → [0x02, 0xAC, 0x02, ...]       (varint 300 = 0xAC 0x02)
//
// Prefix semantics are preserved because string lengths are encoded:
//   @["auth"] is NOT a prefix of @["author"] since their lengths differ.
//
// Backward compatibility: Legacy paths (single-byte integers 0-255) are detected
// by checking if the first byte is NOT a valid tag (0x01, 0x02, 0x03).
// ==========================================================================

/// Tag bytes for path element encoding.
pub mod path_tags {
    /// Integer (i64, 8 bytes little-endian)
    pub const INTEGER: u8 = 0x01;
    /// String (varint length + UTF-8 bytes)
    pub const STRING: u8 = 0x02;
    /// ByteArray (varint length + raw bytes)
    pub const BYTE_ARRAY: u8 = 0x03;
}

/// Encode a varint (variable-length integer) into a buffer.
///
/// Uses 7 bits per byte with the high bit (0x80) as a continuation flag.
/// This matches the encoding used by MeTTaTron for compatibility.
///
/// # Example
/// ```ignore
/// let mut buf = Vec::new();
/// encode_varint(&mut buf, 127);   // [127]
/// encode_varint(&mut buf, 128);   // [0x80, 0x01]
/// encode_varint(&mut buf, 300);   // [0xAC, 0x02]
/// ```
pub fn encode_varint(buf: &mut Vec<u8>, mut n: u64) {
    while n >= 0x80 {
        buf.push((n as u8) | 0x80);
        n >>= 7;
    }
    buf.push(n as u8);
}

/// Decode a varint from a byte slice.
///
/// Returns `Some((value, bytes_consumed))` on success, `None` on error.
///
/// # Example
/// ```ignore
/// assert_eq!(decode_varint(&[127]), Some((127, 1)));
/// assert_eq!(decode_varint(&[0x80, 0x01]), Some((128, 2)));
/// assert_eq!(decode_varint(&[0xAC, 0x02]), Some((300, 2)));
/// ```
pub fn decode_varint(bytes: &[u8]) -> Option<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    let mut consumed = 0;

    for &byte in bytes {
        consumed += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some((result, consumed));
        }
        shift += 7;
        if shift > 63 {
            return None; // Overflow protection
        }
    }
    None // Incomplete varint
}

/// Encode a single path element (Par) into the buffer.
///
/// Supports GInt, GString, and GByteArray types.
/// Returns `true` if encoding succeeded, `false` if the element type is unsupported.
fn encode_path_element(buf: &mut Vec<u8>, par: &Par) -> bool {
    for expr in &par.exprs {
        match &expr.expr_instance {
            Some(ExprInstance::GInt(n)) => {
                buf.push(path_tags::INTEGER);
                buf.extend_from_slice(&n.to_le_bytes());
                return true;
            }
            Some(ExprInstance::GString(s)) => {
                buf.push(path_tags::STRING);
                encode_varint(buf, s.len() as u64);
                buf.extend_from_slice(s.as_bytes());
                return true;
            }
            Some(ExprInstance::GByteArray(bytes)) => {
                buf.push(path_tags::BYTE_ARRAY);
                encode_varint(buf, bytes.len() as u64);
                buf.extend_from_slice(bytes);
                return true;
            }
            _ => continue,
        }
    }
    false
}

/// Decode a single path element from bytes.
///
/// Returns `Some((par, bytes_consumed))` on success, `None` on error.
fn decode_path_element(bytes: &[u8]) -> Option<(Par, usize)> {
    if bytes.is_empty() {
        return None;
    }

    let tag = bytes[0];
    let rest = &bytes[1..];

    match tag {
        path_tags::INTEGER => {
            if rest.len() < 8 {
                return None;
            }
            let n = i64::from_le_bytes(rest[..8].try_into().ok()?);
            let par = Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GInt(n)),
            }]);
            Some((par, 9)) // tag + 8 bytes
        }
        path_tags::STRING => {
            let (len, varint_size) = decode_varint(rest)?;
            let len = len as usize;
            let data_start = varint_size;
            if rest.len() < data_start + len {
                return None;
            }
            let s = std::str::from_utf8(&rest[data_start..data_start + len]).ok()?;
            let par = Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GString(s.to_string())),
            }]);
            Some((par, 1 + data_start + len)) // tag + varint + data
        }
        path_tags::BYTE_ARRAY => {
            let (len, varint_size) = decode_varint(rest)?;
            let len = len as usize;
            let data_start = varint_size;
            if rest.len() < data_start + len {
                return None;
            }
            let bytes_data = rest[data_start..data_start + len].to_vec();
            let par = Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GByteArray(bytes_data)),
            }]);
            Some((par, 1 + data_start + len)) // tag + varint + data
        }
        _ => None, // Unknown tag
    }
}

/// Check if a byte path uses the new tagged encoding format.
///
/// New format starts with a valid tag (0x01, 0x02, 0x03).
/// Legacy format uses raw bytes (integers 0-255).
fn is_tagged_path(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    matches!(bytes[0], path_tags::INTEGER | path_tags::STRING | path_tags::BYTE_ARRAY)
}

/// Convert a Rholang Par representing a path to a byte path.
///
/// For PathMap prefix semantics to work with Rholang channels, we need to convert
/// Par channels like `@[0, 1, 2]` or `@["sys", "auth"]` to byte paths.
///
/// # Supported Path Formats
/// - `@[0, 1, 2]` → Integer path (tagged encoding)
/// - `@["sys", "auth"]` → String path (tagged encoding with length prefixes)
/// - `@[0, "auth", 2]` → Mixed path (tagged encoding)
///
/// # Invalid Formats (returns None)
/// - `@"string"` - Not a list
/// - Any EList element that isn't GInt, GString, or GByteArray
///
/// # Prefix Semantics
/// String paths preserve prefix semantics because lengths are encoded:
/// - `@["auth"]` encodes as `[0x02, 4, 'a', 'u', 't', 'h']`
/// - `@["author"]` encodes as `[0x02, 6, 'a', 'u', 't', 'h', 'o', 'r']`
/// These are NOT prefix-related because the length bytes differ.
///
/// # Example
/// ```ignore
/// let par = create_string_elist_par(vec!["sys", "auth"]);
/// let path = par_to_path(&par);
/// assert!(path.is_some());
/// ```
pub fn par_to_path(par: &Par) -> Option<Vec<u8>> {
    // Look for an EList expression in the Par
    for expr in &par.exprs {
        if let Some(ExprInstance::EListBody(elist)) = &expr.expr_instance {
            return elist_to_path(elist);
        }
    }

    // Also check if it's a single element (path of length 1)
    // Try to encode it as a single-element path
    let mut buf = Vec::new();
    if encode_path_element(&mut buf, par) {
        return Some(buf);
    }

    None
}

/// Convert an EList to a byte path with tagged encoding.
fn elist_to_path(elist: &EList) -> Option<Vec<u8>> {
    // Pre-allocate with estimate: 9 bytes per integer, variable for strings
    let mut path = Vec::with_capacity(elist.ps.len() * 9);

    for p in &elist.ps {
        if !encode_path_element(&mut path, p) {
            return None; // Unsupported element type
        }
    }

    Some(path)
}

/// Convert a byte path to a Rholang Par (EList).
///
/// This is the inverse of `par_to_path`. It decodes tagged elements and
/// creates a Par with an EList containing the appropriate expression types.
///
/// # Backward Compatibility
/// Detects legacy format (raw bytes) vs new tagged format:
/// - If first byte is a valid tag (0x01, 0x02, 0x03), decode as tagged
/// - Otherwise, treat as legacy integer-only format (each byte = one GInt 0-255)
///
/// # Example
/// ```ignore
/// // New format
/// let path = vec![0x02, 4, b'a', b'u', b't', b'h']; // String "auth"
/// let par = path_to_par(&path);
/// // par is equivalent to @["auth"]
///
/// // Legacy format (backward compatible)
/// let legacy = vec![0, 1, 2];
/// let par_legacy = path_to_par(&legacy);
/// // par_legacy is equivalent to @[0, 1, 2]
/// ```
pub fn path_to_par(path: &[u8]) -> Par {
    if path.is_empty() {
        return Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EListBody(EList {
                ps: vec![],
                locally_free: vec![],
                connective_used: false,
                remainder: None,
            })),
        }]);
    }

    // Check if this is the new tagged format
    if is_tagged_path(path) {
        if let Some(elements) = decode_tagged_path(path) {
            return Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::EListBody(EList {
                    ps: elements,
                    locally_free: vec![],
                    connective_used: false,
                    remainder: None,
                })),
            }]);
        }
    }

    // Legacy format: each byte is a GInt 0-255
    let elements: Vec<Par> = path
        .iter()
        .map(|&byte| {
            Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GInt(byte as i64)),
            }])
        })
        .collect();

    Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EListBody(EList {
            ps: elements,
            locally_free: vec![],
            connective_used: false,
            remainder: None,
        })),
    }])
}

/// Decode a tagged path into a list of Par elements.
fn decode_tagged_path(path: &[u8]) -> Option<Vec<Par>> {
    let mut elements = Vec::new();
    let mut offset = 0;

    while offset < path.len() {
        let (par, consumed) = decode_path_element(&path[offset..])?;
        elements.push(par);
        offset += consumed;
    }

    Some(elements)
}

/// Check if a Par represents a valid path (EList of integers 0-255).
///
/// This is a convenience function for checking if prefix semantics can apply.
pub fn is_par_path(par: &Par) -> bool {
    par_to_path(par).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_path_suffix_valid() {
        assert_eq!(get_path_suffix(&[0, 1], &[0, 1, 2]), Some(vec![2]));
        assert_eq!(get_path_suffix(&[0, 1], &[0, 1, 2, 3]), Some(vec![2, 3]));
        assert_eq!(get_path_suffix(&[0, 1], &[0, 1]), Some(vec![]));
        assert_eq!(get_path_suffix(&[], &[0, 1, 2]), Some(vec![0, 1, 2]));
    }

    #[test]
    fn test_get_path_suffix_invalid() {
        assert_eq!(get_path_suffix(&[0, 1], &[0, 2]), None);
        assert_eq!(get_path_suffix(&[0, 1, 2], &[0, 1]), None);
        assert_eq!(get_path_suffix(&[1, 2, 3], &[4, 5, 6]), None);
    }

    #[test]
    fn test_path_prefixes() {
        let prefixes = path_prefixes(&[0, 1, 2]);
        assert_eq!(prefixes, vec![vec![0], vec![0, 1], vec![0, 1, 2]]);

        let prefixes = path_prefixes(&[5]);
        assert_eq!(prefixes, vec![vec![5]]);

        let prefixes = path_prefixes(&[]);
        assert!(prefixes.is_empty());
    }

    #[test]
    fn test_is_path_prefix() {
        assert!(is_path_prefix(&[0, 1], &[0, 1, 2]));
        assert!(is_path_prefix(&[0, 1], &[0, 1]));
        assert!(is_path_prefix(&[], &[0, 1, 2]));

        assert!(!is_path_prefix(&[0, 1], &[0, 2]));
        assert!(!is_path_prefix(&[0, 1, 2], &[0, 1]));
    }

    #[test]
    fn test_aggregated_datum_new() {
        let datum: AggregatedDatum<String> = AggregatedDatum::new(
            vec![2],
            "hello".to_string(),
            false,
        );

        assert_eq!(datum.suffix_key, vec![2]);
        assert_eq!(datum.data, "hello");
        assert!(!datum.persist);
        assert!(!datum.is_exact_match());
        assert_eq!(datum.suffix_depth(), 1);
    }

    #[test]
    fn test_aggregated_datum_exact() {
        let datum: AggregatedDatum<i32> = AggregatedDatum::exact(42, true);

        assert!(datum.suffix_key.is_empty());
        assert_eq!(datum.data, 42);
        assert!(datum.persist);
        assert!(datum.is_exact_match());
        assert_eq!(datum.suffix_depth(), 0);
    }

    #[test]
    fn test_aggregated_datum_map() {
        let datum: AggregatedDatum<i32> = AggregatedDatum::new(vec![1, 2], 10, true);
        let mapped = datum.map(|x| x * 2);

        assert_eq!(mapped.suffix_key, vec![1, 2]);
        assert_eq!(mapped.data, 20);
        assert!(mapped.persist);
    }

    #[test]
    fn test_varint_encode_small() {
        let mut buf = Vec::new();
        encode_varint(&mut buf, 0);
        assert_eq!(buf, vec![0]);

        let mut buf = Vec::new();
        encode_varint(&mut buf, 127);
        assert_eq!(buf, vec![127]);
    }

    #[test]
    fn test_varint_encode_large() {
        let mut buf = Vec::new();
        encode_varint(&mut buf, 128);
        assert_eq!(buf, vec![0x80, 0x01]);

        let mut buf = Vec::new();
        encode_varint(&mut buf, 300);
        assert_eq!(buf, vec![0xAC, 0x02]);
    }

    #[test]
    fn test_varint_roundtrip() {
        for n in [0, 1, 127, 128, 255, 256, 300, 1000, 16383, 16384, 100000] {
            let mut buf = Vec::new();
            encode_varint(&mut buf, n);
            let (decoded, _) = decode_varint(&buf).expect("varint decode failed");
            assert_eq!(decoded, n, "Roundtrip failed for {}", n);
        }
    }

    #[test]
    fn test_par_to_path_integer_list() {
        let elements: Vec<Par> = vec![0i64, 1, 2]
            .into_iter()
            .map(|n| Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GInt(n)),
            }]))
            .collect();

        let par = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EListBody(EList {
                ps: elements,
                locally_free: vec![],
                connective_used: false,
                remainder: None,
            })),
        }]);

        let path = par_to_path(&par).expect("par_to_path failed");

        assert_eq!(path.len(), 27);
        assert_eq!(path[0], path_tags::INTEGER);
        assert_eq!(path[9], path_tags::INTEGER);
        assert_eq!(path[18], path_tags::INTEGER);
    }

    #[test]
    fn test_par_to_path_string_list() {
        let elements: Vec<Par> = vec!["sys", "auth"]
            .into_iter()
            .map(|s| Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GString(s.to_string())),
            }]))
            .collect();

        let par = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EListBody(EList {
                ps: elements,
                locally_free: vec![],
                connective_used: false,
                remainder: None,
            })),
        }]);

        let path = par_to_path(&par).expect("par_to_path failed");

        assert_eq!(path.len(), 11);
        assert_eq!(path[0], path_tags::STRING);
        assert_eq!(path[1], 3); // length of "sys"
        assert_eq!(path[5], path_tags::STRING);
        assert_eq!(path[6], 4); // length of "auth"
    }

    #[test]
    fn test_path_to_par_legacy_format() {
        let legacy_path = vec![0u8, 4, 5];
        let par = path_to_par(&legacy_path);

        if let Some(ExprInstance::EListBody(elist)) = &par.exprs[0].expr_instance {
            assert_eq!(elist.ps.len(), 3);
            for (i, expected) in [0i64, 4, 5].iter().enumerate() {
                if let Some(ExprInstance::GInt(n)) = &elist.ps[i].exprs[0].expr_instance {
                    assert_eq!(*n, *expected);
                } else {
                    panic!("Expected GInt at position {}", i);
                }
            }
        } else {
            panic!("Expected EListBody");
        }
    }
}
