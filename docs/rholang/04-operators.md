# Operators

All operators as implemented in the Rust shard reducer (`reduce.rs`).

## Arithmetic Operators

All arithmetic operators require operands of the same type. No implicit coercion.

### Addition (`+`)

| Left Type | Right Type | Result | Notes |
|-----------|-----------|--------|-------|
| Int | Int | Int | Wrapping addition (no overflow panic) |
| Float | Float | Float | IEEE 754 |
| BigInt | BigInt | BigInt | Arbitrary precision |
| BigRat | BigRat | BigRat | Auto-reduced via GCD |
| FixedPoint | FixedPoint | FixedPoint | Requires matching scale |
| String | String | String | Concatenation (also via `++`) |

```rho
3 + 4            // 7
3.0f64 + 1.5f64  // 4.5f64
100n + 200n      // 300n
1r / 3r + 1r / 6r  // 1r / 2r (auto-reduced)
1.50p2 + 0.25p2  // 1.75p2
```

### Subtraction (`-`)

Same type requirements as addition. Int subtraction uses wrapping semantics.

```rho
10 - 3           // 7
5.0f64 - 2.5f64  // 2.5f64
```

Map subtraction is also supported:

```rho
{"a": 1, "b": 2} - {"b": 2}  // {"a": 1}
```

### Multiplication (`*`)

| Left Type | Right Type | Result | Notes |
|-----------|-----------|--------|-------|
| Int | Int | Int | Wrapping multiplication |
| Float | Float | Float | IEEE 754 |
| BigInt | BigInt | BigInt | Arbitrary precision |
| BigRat | BigRat | BigRat | Auto-reduced |
| FixedPoint | FixedPoint | FixedPoint | Scale-preserving, floor division |

FixedPoint multiplication preserves scale:
```rho
1.5p1 * 2.0p1    // 3.0p1 (floor((15 * 20) / 10) = 30, scale 1)
0.1p1 * 0.1p1    // 0.0p1 (precision loss: floor((1 * 1) / 10) = 0)
```

### Division (`/`)

| Left Type | Right Type | Result | Notes |
|-----------|-----------|--------|-------|
| Int | Int | Int | Truncating division |
| Float | Float | Float | IEEE 754 (div by 0 -> Inf/NaN) |
| BigInt | BigInt | BigInt | Truncating |
| BigRat | BigRat | BigRat | Exact |
| FixedPoint | FixedPoint | FixedPoint | Scale-preserving |

Division by zero:
- **Int, BigInt**: error `"Division by zero"`
- **Float**: produces `Inf`, `-Inf`, or `NaN` per IEEE 754
- **BigRat**: error `"Division by zero"`
- **FixedPoint**: error `"Division by zero"`

### Modulo (`%`)

| Left Type | Right Type | Result | Notes |
|-----------|-----------|--------|-------|
| Int | Int | Int | C99 identity: `(a/b)*b + a%b == a` |
| BigInt | BigInt | BigInt | Same identity |
| BigRat | BigRat | BigRat | Always zero (exact division) |
| FixedPoint | FixedPoint | FixedPoint | Direct `ua % ub` on unscaled |
| **Float** | **Float** | **error** | `"modulus not defined on floating point"` |

Special case: `i64::MIN % -1` returns `0` (avoids overflow).

```rho
10 % 3            // 1
-10 % 3           // -1
1.50p2 % 1.00p2   // 0.50p2
```

### Unary Negation (`-`)

```rho
-42               // Int
-3.14f64          // Float
-100n             // BigInt
-1r / 3r          // BigRat (negates numerator)
```

## Comparison Operators

Return `GBool`. Work on matching types.

```rho
x < y     // less than
x <= y    // less than or equal
x > y     // greater than
x >= y    // greater than or equal
```

Supported types: Int, Float, BigInt, BigRat, FixedPoint (matching scale required).

Float comparisons follow IEEE 754: any comparison involving NaN returns `false`.

## Equality Operators

```rho
x == y    // structural equality
x != y    // structural inequality
```

Work on all types including Bool, String, ByteArray, collections.

**NaN semantics**: `NaN == NaN` is `false`, `NaN != NaN` is `true`. This applies recursively -- `[NaN] == [NaN]` is also `false`.

## Logical Operators

Require boolean operands.

```rho
true and false    // false
true or false     // true
not true          // false
```

## String and Collection Operators

### Concatenation (`++`)

```rho
"hello" ++ " world"       // "hello world"
[1, 2] ++ [3, 4]          // [1, 2, 3, 4]
```

Also works on ByteArrays (appends bytes).

### Set Difference (`--`)

```rho
Set(1, 2, 3) -- Set(2)    // Set(1, 3)
```

### String Interpolation (`%%`)

```rho
"Hello ${name}, age ${age}" %% {"name": "Alice", "age": "30"}
// "Hello Alice, age 30"
```

Right operand must be a Map with string keys. Values are converted to strings.

### Pattern Test (`=~`)

```rho
42 =~ Int                 // true (type match)
[1, 2, 3] =~ [_, 2, _]   // true (structural match)
```

## Operator Precedence

From lowest to highest:

1. `|` (parallel composition)
2. `or`
3. `and`
4. `==`, `!=`
5. `<`, `<=`, `>`, `>=`
6. `++`, `--`
7. `+`, `-`
8. `*`, `/`, `%`
9. `%%`
10. `not`, unary `-`
11. `.method()`

Use parentheses to override: `(a + b) * c`
