# Numeric Types

Rholang supports six numeric types. No implicit coercion between any of them.

## Integer (GInt)

64-bit signed integer. The default numeric type.

```rho
42
-7
0
9223372036854775807    // i64::MAX
-9223372036854775808   // i64::MIN
```

Arithmetic uses wrapping semantics -- overflow wraps around silently (no panic, no error).

```rho
9223372036854775807 + 1    // wraps to -9223372036854775808
```

Division by zero returns an error. Special case: `i64::MIN / -1` also returns an error (would overflow).

## Float (GDouble)

IEEE 754 double-precision (f64). Stored as raw bits (`fixed64` in protobuf) to preserve `-0.0`, NaN payloads, and exact bit representations.

### Literal Syntax

```rho
3.14f64
2.5f32       // f32 suffix (stored as f64 internally)
-0.0f64
```

### IEEE 754 Semantics

```rho
1.0f64 / 0.0f64      // Inf (not an error)
-1.0f64 / 0.0f64     // -Inf
0.0f64 / 0.0f64      // NaN

// NaN comparisons
NaN == NaN            // false
NaN != NaN            // true
NaN < 1.0f64          // false
NaN > 1.0f64          // false
```

NaN detection is recursive: `[NaN] == [NaN]` is also `false`.

### Float Modulo

**Not supported.** Returns error: `"modulus not defined on floating point"`.

## BigInt (GBigInt)

Arbitrary-precision signed integer. Suffix: `n`.

```rho
100n
999999999999999999999999999999n
-42n
0n
```

No size cap. Gas scales with operand byte length (see [Cost Model](13-cost-model.md)):
- Add/sub: `O(max(a_len, b_len))`
- Mul/div/mod: `O(a_len * b_len)` -- quadratic

### Unsigned Int Literals

Unsigned literals (`u8`, `u16`, `u32`, `u64`) compile down to `Int` (if they fit in i64) or `BigInt`. There is no separate unsigned runtime type.

```rho
255u8              // becomes Int(255)
4294967295u32      // becomes Int(4294967295)
18446744073709551615u64   // becomes BigInt (doesn't fit in i64)
```

### Signed Int Literals with Bit Width

```rho
42i8               // becomes Int(42)
42i16              // becomes Int(42)
42i32              // becomes Int(42)
42i64              // becomes Int(42)
```

These are compile-time annotations. All compile to `Int` at runtime.

## BigRat (GBigRational)

Exact rational numbers (fractions). Always stored in lowest terms via GCD normalization.

### Literal Syntax

BigRat values are constructed using the `r` suffix and division:

```rho
1r / 3r            // 1/3
5r / 2r            // 5/2
-3r / 7r           // -3/7
```

### Auto-Reduction

All operations automatically reduce to lowest terms:

```rho
2r / 4r            // stored as 1/2
(2r / 3r) * (3r / 4r)   // 1/2 (not 6/12)
```

### Operations

All standard arithmetic and comparison operators work:

```rho
1r / 3r + 1r / 6r       // 1/2
5r / 3r - 1r / 3r       // 4/3
2r / 3r * 3r / 5r       // 2/5
1r / 2r / 1r / 4r       // 2/1 (= 2)
```

BigRat modulo always returns zero (exact division property):

```rho
5r / 3r % 1r / 3r       // 0/1 (zero)
```

Division by zero returns an error.

## FixedPoint (GFixedPoint)

Fixed-point decimal numbers. Stored as `(unscaled: BigInt, scale: u32)` where `value = unscaled / 10^scale`.

### Literal Syntax

```rho
1.50p2             // scale=2: unscaled=150, represents 1.50
10p0               // scale=0: unscaled=10, represents 10
0.001p3            // scale=3: unscaled=1, represents 0.001
3.3p1              // scale=1: unscaled=33, represents 3.3
```

The number of decimal places in the literal must not exceed the scale:

```rho
1.234p2            // COMPILE ERROR: 3 decimal places > scale 2
1.23p2             // OK
1.2p2              // OK (unscaled=120)
```

### Scale Matching

All binary operations (add, sub, mul, div, mod, compare) require matching scales:

```rho
1.5p1 + 2.0p1           // OK: 3.5p1
1.50p2 + 0.25p2         // OK: 1.75p2
1.5p1 + 1.50p2          // ERROR: scale mismatch
```

### Multiplication

Scale-preserving with floor division:

```rho
unscaled_result = floor((ua * ub) / 10^scale)
```

Examples:

```rho
1.5p1 * 2.0p1           // 3.0p1: floor((15 * 20) / 10) = 30
0.1p1 * 0.1p1           // 0.0p1: floor((1 * 1) / 10) = 0 (precision loss)
1.50p2 * 2.00p2         // 3.00p2: floor((150 * 200) / 100) = 300
```

Floor division means negative results round toward negative infinity:

```rho
-1.5p1 * 0.3p1          // -0.5p1: -((-(-15 * 3) - 1) / 10 + 1) = -5
```

### Division

Scale-preserving:

```rho
3.0p1 / 2.0p1           // result preserves scale p1
```

### Modulo

Direct modulo on unscaled values:

```rho
1.50p2 % 1.00p2         // 0.50p2: 150 % 100 = 50
10.0p1 % 3.0p1          // 1.0p1: 100 % 30 = 10
```

## Type Compatibility Matrix

All binary operations require matching types. None of these work:

```rho
42 + 3.14f64             // ERROR: Int + Float
100n + 42                // ERROR: BigInt + Int
1r / 2r + 0.5f64         // ERROR: BigRat + Float
1.5p1 + 1.50p2           // ERROR: scale mismatch (even within FixedPoint)
```

## Proto Representation

| Type | Proto Field | Encoding |
|------|-------------|----------|
| Int | `g_int` (int64) | Direct |
| Float | `g_double` (fixed64) | Raw IEEE 754 bits |
| BigInt | `g_big_int` (bytes) | Big-endian two's complement |
| BigRat | `g_big_rat` { numerator, denominator } | Both big-endian two's complement bytes |
| FixedPoint | `g_fixed_point` { unscaled, scale } | unscaled: big-endian two's complement bytes; scale: uint32 |
