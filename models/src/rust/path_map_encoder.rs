// PathMap S-expression to byte path encoder
//
// Tag Format (1 byte):
//   ┌─────────┬──────────┬─────────────────────────┐
//   │ Bits    │ Pattern  │ Meaning                 │
//   ├─────────┼──────────┼─────────────────────────┤
//   │ 11000000│ 0xC0     │ NewVar ($)              │
//   │ 11xxxxxx│ 0xC0+n   │ SymbolSize(n) 1-63 bytes│
//   │ 10xxxxxx│ 0x80+n   │ VarRef(_n) reference    │
//   │ 00xxxxxx│ 0x00+n   │ Arity(n) 0-63 children  │
//   └─────────┴──────────┴─────────────────────────┘

use std::fmt;

/// Tag types for S-expression encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tag {
    /// NewVar marker (0xC0)
    NewVar,
    /// Symbol with size n (0xC0 + n, where 1 <= n <= 63)
    Symbol(u8),
    /// Variable reference (0x80 + n, where 0 <= n <= 63)
    VarRef(u8),
    /// Arity with n children (0x00 + n, where 0 <= n <= 63)
    Arity(u8),
}

impl Tag {
    /// Encode a tag to its byte representation
    pub fn to_byte(self) -> u8 {
        match self {
            Tag::NewVar => 0xC0,
            Tag::Symbol(n) if n >= 1 && n <= 63 => 0xC0 + n,
            Tag::VarRef(n) if n <= 63 => 0x80 + n,
            Tag::Arity(n) if n <= 63 => 0x00 + n,
            _ => panic!("Invalid tag value"),
        }
    }

    /// Decode a byte to a Tag
    pub fn from_byte(byte: u8) -> Result<Self, String> {
        match byte {
            0xC0 => Ok(Tag::NewVar),
            b if b >= 0xC1 => Ok(Tag::Symbol(b - 0xC0)),
            b if b >= 0x80 && b <= 0xBF => Ok(Tag::VarRef(b - 0x80)),
            b if b <= 0x3F => Ok(Tag::Arity(b)),
            _ => Err(format!("Invalid tag byte: 0x{:02X}", byte)),
        }
    }
}

/// S-expression node
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SExpr {
    /// A symbol (e.g., "new", "x", "!")
    Symbol(String),
    /// A list of sub-expressions
    List(Vec<SExpr>),
}

impl SExpr {
    /// Encode an S-expression to a byte path
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.encode_into(&mut bytes);
        bytes
    }

    fn encode_into(&self, bytes: &mut Vec<u8>) {
        match self {
            SExpr::Symbol(s) => {
                let s_bytes = s.as_bytes();
                let len = s_bytes.len();

                if len == 0 || len > 63 {
                    panic!("Symbol length must be between 1 and 63 bytes");
                }

                // Emit symbol tag with length
                bytes.push(Tag::Symbol(len as u8).to_byte());
                // Emit symbol bytes
                bytes.extend_from_slice(s_bytes);
            }
            SExpr::List(children) => {
                let arity = children.len();

                if arity > 63 {
                    panic!("Arity must not exceed 63");
                }

                // Emit arity tag
                bytes.push(Tag::Arity(arity as u8).to_byte());

                // Recursively encode children
                for child in children {
                    child.encode_into(bytes);
                }
            }
        }
    }

    /// Decode a byte path back to an S-expression
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let mut cursor = 0;
        Self::decode_from(bytes, &mut cursor)
    }

    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self, String> {
        if *cursor >= bytes.len() {
            return Err("Unexpected end of input".to_string());
        }

        let tag = Tag::from_byte(bytes[*cursor])?;
        *cursor += 1;

        match tag {
            Tag::Symbol(len) => {
                let end = *cursor + len as usize;
                if end > bytes.len() {
                    return Err(format!("Symbol length {} exceeds remaining bytes", len));
                }

                let symbol = String::from_utf8(bytes[*cursor..end].to_vec())
                    .map_err(|e| format!("Invalid UTF-8 in symbol: {}", e))?;
                *cursor = end;

                Ok(SExpr::Symbol(symbol))
            }
            Tag::Arity(n) => {
                let mut children = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    children.push(Self::decode_from(bytes, cursor)?);
                }
                Ok(SExpr::List(children))
            }
            Tag::NewVar => {
                // NewVar is a special marker, treat as symbol "$"
                Ok(SExpr::Symbol("$".to_string()))
            }
            Tag::VarRef(n) => {
                // Variable reference, represent as "_n"
                Ok(SExpr::Symbol(format!("_{}", n)))
            }
        }
    }
}

impl fmt::Display for SExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SExpr::Symbol(s) => write!(f, "{}", s),
            SExpr::List(children) => {
                write!(f, "(")?;
                for (i, child) in children.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", child)?;
                }
                write!(f, ")")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tag_encoding() {
        assert_eq!(Tag::NewVar.to_byte(), 0xC0);
        assert_eq!(Tag::Symbol(1).to_byte(), 0xC1);
        assert_eq!(Tag::Symbol(3).to_byte(), 0xC3);
        assert_eq!(Tag::VarRef(0).to_byte(), 0x80);
        assert_eq!(Tag::Arity(3).to_byte(), 0x03);
    }

    #[test]
    fn test_tag_decoding() {
        assert_eq!(Tag::from_byte(0xC0).unwrap(), Tag::NewVar);
        assert_eq!(Tag::from_byte(0xC1).unwrap(), Tag::Symbol(1));
        assert_eq!(Tag::from_byte(0xC3).unwrap(), Tag::Symbol(3));
        assert_eq!(Tag::from_byte(0x80).unwrap(), Tag::VarRef(0));
        assert_eq!(Tag::from_byte(0x03).unwrap(), Tag::Arity(3));
    }

    #[test]
    fn test_simple_symbol() {
        let sexpr = SExpr::Symbol("x".to_string());
        let encoded = sexpr.encode();
        assert_eq!(encoded, vec![0xC1, b'x']);

        let decoded = SExpr::decode(&encoded).unwrap();
        assert_eq!(decoded, sexpr);
    }

    #[test]
    fn test_simple_list() {
        // (new x (! x z))
        let sexpr = SExpr::List(vec![
            SExpr::Symbol("new".to_string()),
            SExpr::Symbol("x".to_string()),
            SExpr::List(vec![
                SExpr::Symbol("!".to_string()),
                SExpr::Symbol("x".to_string()),
                SExpr::Symbol("z".to_string()),
            ]),
        ]);

        let encoded = sexpr.encode();
        let decoded = SExpr::decode(&encoded).unwrap();
        assert_eq!(decoded, sexpr);
    }

    #[test]
    fn test_example_encoding() {
        // Example from spec: (new x (! x (! y z)))
        // Should encode to: 03 C3 n e w C1 x 03 C1 ! C1 x 03 C1 ! C1 y C1 z
        let sexpr = SExpr::List(vec![
            SExpr::Symbol("new".to_string()),
            SExpr::Symbol("x".to_string()),
            SExpr::List(vec![
                SExpr::Symbol("!".to_string()),
                SExpr::Symbol("x".to_string()),
                SExpr::List(vec![
                    SExpr::Symbol("!".to_string()),
                    SExpr::Symbol("y".to_string()),
                    SExpr::Symbol("z".to_string()),
                ]),
            ]),
        ]);

        let encoded = sexpr.encode();
        let expected = vec![
            0x03, 0xC3, b'n', b'e', b'w', 0xC1, b'x', 0x03, 0xC1, b'!', 0xC1, b'x', 0x03, 0xC1,
            b'!', 0xC1, b'y', 0xC1, b'z',
        ];

        assert_eq!(encoded, expected);

        // Verify round-trip
        let decoded = SExpr::decode(&encoded).unwrap();
        assert_eq!(decoded, sexpr);
    }

    #[test]
    fn test_empty_list() {
        let sexpr = SExpr::List(vec![]);
        let encoded = sexpr.encode();
        assert_eq!(encoded, vec![0x00]); // Arity 0

        let decoded = SExpr::decode(&encoded).unwrap();
        assert_eq!(decoded, sexpr);
    }

    #[test]
    fn test_nested_lists() {
        let sexpr = SExpr::List(vec![
            SExpr::List(vec![
                SExpr::Symbol("a".to_string()),
                SExpr::Symbol("b".to_string()),
            ]),
            SExpr::Symbol("c".to_string()),
        ]);

        let encoded = sexpr.encode();
        let decoded = SExpr::decode(&encoded).unwrap();
        assert_eq!(decoded, sexpr);
    }
}
