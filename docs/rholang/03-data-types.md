# Data Types

Rholang's type system as implemented in the Rust shard. All types map to protobuf `ExprInstance` variants in `RhoTypes.proto`.

## Primitive Types

### Integer (GInt)

64-bit signed integer. Arithmetic uses wrapping semantics (no overflow panic).

```rho
42
-7
0
9223372036854775807   // i64::MAX
```

Proto: `GInt(i64)`

### Boolean (GBool)

```rho
true
false
```

Used by comparison operators, logical operators, and `if`/`match`.

Proto: `GBool(bool)`

### String (GString)

UTF-8 strings. Double-quoted.

```rho
"hello"
"line\nbreak"
"tab\there"
""                    // empty string
```

**Methods:**
| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `length()` | 0 | Int | Character count |
| `slice(from, until)` | 2 | String | Substring (0-indexed, exclusive end) |
| `hexToBytes()` | 0 | ByteArray | Parse hex string to bytes |
| `toUtf8Bytes()` | 0 | ByteArray | Encode as UTF-8 bytes |
| `toByteArray()` | 0 | ByteArray | Protobuf-encoded bytes |

**Operators:**
- `++` concatenation: `"hello" ++ " world"` -> `"hello world"`
- `%%` interpolation: `"Hi ${name}" %% {"name": "World"}` -> `"Hi World"`

Proto: `GString(String)`

### ByteArray (GByteArray)

Raw binary data. Created via method calls, not literal syntax.

```rho
"deadbeef".hexToBytes()       // from hex string
"hello".toUtf8Bytes()         // from UTF-8 string
```

**Methods:**
| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `nth(index)` | 1 | Int | Byte value at index (0-255) |
| `length()` | 0 | Int | Byte count |
| `slice(from, until)` | 2 | ByteArray | Sub-range |
| `bytesToHex()` | 0 | String | Hex-encoded string |
| `toByteArray()` | 0 | ByteArray | Identity (protobuf encoding) |

Proto: `GByteArray(Vec<u8>)`

### URI (GUri)

Uniform resource identifier. Backtick-quoted. Primarily used for system channel binding.

```rho
`rho:io:stdout`
`rho:crypto:sha256Hash`
`rho:registry:lookup`
```

Always used inside `new` to bind a name to a system channel:

```rho
new stdout(`rho:io:stdout`) in { ... }
```

Proto: `GUri(String)`

## Numeric Types

See [Numeric Types](15-numeric-types.md) for full details. Summary:

| Type | Suffix | Example | Proto |
|------|--------|---------|-------|
| Float | f32/f64 | `3.14f64` | `GDouble(fixed64)` |
| BigInt | n | `100n` | `GBigInt(bytes)` |
| BigRat | r | `1r / 3r` | `GBigRat { numerator, denominator }` |
| FixedPoint | p + scale | `1.50p2` | `GFixedPoint { unscaled, scale }` |

No implicit coercion between any numeric types. All binary operations require matching types.

## Collection Types

See [Collections](06-collections.md) for full method reference.

| Type | Syntax | Proto |
|------|--------|-------|
| List | `[1, 2, 3]` | `EList { ps }` |
| Tuple | `(1, 2, 3)` | `ETuple { ps }` |
| Set | `Set(1, 2, 3)` | `ESet { ps }` (stored as `SortedParHashSet`) |
| Map | `{"k": v}` | `EMap { kvs }` (stored as `SortedParMap`) |
| PathMap | `{\| path1, path2 \|}` | `EPathMap` |
| Zipper | via methods | `EZipper` |

## Special Values

### Nil

The empty process. Equivalent to "nothing" or "unit".

```rho
Nil
```

Used as:
- Default return value
- Acknowledgment signal: `ack!(Nil)`
- Empty collection element
- "No value" in map lookups

### Unforgeable Names (GUnforgeable)

Created by `new`. Cannot be constructed from source code. Four variants:

| Variant | Created By | Purpose |
|---------|-----------|---------|
| GPrivate | `new x in { ... }` | Fresh unique channel |
| GDeployId | system | Deploy signature identifier |
| GDeployerId | system | Deployer public key hash |
| GSysAuthToken | system | System authorization token |

Unforgeable names support one method:
- `toString()` -- returns a string representation

## Type Matching in Patterns

You can match on types in `for` and `match` patterns:

```rho
match value {
  x: Int     => { stdout!("integer") }
  x: Bool    => { stdout!("boolean") }
  x: String  => { stdout!("string") }
  x: Uri     => { stdout!("URI") }
  x: ByteArray => { stdout!("bytes") }
  _ => { stdout!("other") }
}
```

## Type Coercion Rules

There is no implicit type coercion. These are all errors:

```rho
42 + "hello"        // error: type mismatch in +
42 + 3.14f64        // error: type mismatch (Int vs Float)
100n + 42           // error: type mismatch (BigInt vs Int)
1.5p1 + 1.50p2      // error: FixedPoint scale mismatch
```

To work with different types, use explicit conversions or ensure matching types at the source.
