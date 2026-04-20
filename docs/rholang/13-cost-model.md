# Cost Model

Every Rholang operation consumes phlogiston (gas). Deploys specify a `phlo_limit` and `phlo_price`. The runtime charges costs as it executes. If the limit is exhausted, execution halts with `OutOfPhlogistonsError`.

## Charging Mechanism

```
Total Phlo Charge = phlo_limit * phlo_price
Refund = (phlo_limit - cost_used) * phlo_price
```

The deployer pays for the full limit upfront. Unused phlogiston is refunded after execution.

Default limits (from `construct_deploy.rs`):
- Standard deploy: 90,000
- Full deploy: 1,000,000

## Cost Table: Arithmetic

| Operation | Cost | Notes |
|-----------|------|-------|
| Addition (`+`) | 3 | Int, String |
| Subtraction (`-`) | 3 | Int |
| Multiplication (`*`) | 9 | Int |
| Division (`/`) | 9 | Int |
| Modulo (`%`) | 9 | Int |
| Boolean AND | 2 | |
| Boolean OR | 2 | |
| Comparison (`<`, `>`, etc.) | 3 | |

## Cost Table: BigInt (Size-Proportional)

Costs scale with operand byte length. `a_len` and `b_len` are the byte lengths of the two operands.

| Operation | Formula | Minimum |
|-----------|---------|---------|
| Addition | `max(a_len, b_len) + 1` | 3 |
| Subtraction | `max(a_len, b_len) + 1` | 3 |
| Multiplication | `a_len * b_len` | 9 |
| Division | `a_len * b_len` | 9 |
| Modulo | `a_len * b_len` | 9 |
| Negation | `len` | 1 |
| Comparison | `max(a_len, b_len)` | 3 |

This means multiplying two 100-byte BigInts costs 10,000 phlogiston. Gas is the rate limiter for large operands.

## Cost Table: BigRat (Rational)

Costs account for cross-multiplication and GCD normalization. `na`, `da`, `nb`, `db` are byte lengths of numerator/denominator pairs.

| Operation | Formula | Minimum |
|-----------|---------|---------|
| Addition | `4 * max_len^2 + max_len` | 3 |
| Subtraction | `4 * max_len^2 + max_len` | 3 |
| Multiplication | `na*nb + da*db + max_len` | 9 |
| Division | `na*db + da*nb + max_len` | 9 |
| Negation | `num_len` | 1 |
| Comparison | `max(na*db, nb*da)` | 3 |

Where `max_len = max(na, da, nb, db)`.

## Cost Table: Collections

| Operation | Cost |
|-----------|------|
| Map/Set lookup (`get`, `contains`) | 3 |
| Map/Set add | 3 |
| Map/Set remove | 3 |
| Set/Map diff | 3 * num_elements |
| Set/Map union | 3 * num_elements |

## Cost Table: String and Bytes

| Operation | Cost |
|-----------|------|
| String concatenation (`++`) | `len(a) + len(b)` |
| ByteArray append | `log10(len(left))` |
| List append (`++`) | `len(right)` |
| String interpolation (`%%`) | `str_len * map_size` |
| Slice | `to` index value |
| Take | `to` count value |
| hexToBytes | `len(hex_string)` |
| bytesToHex | `len(byte_array)` |
| toList | collection size |
| Parsing | `len(source)` in bytes |
| toByteArray | protobuf encoded size |

## Cost Table: Evaluation

| Operation | Cost |
|-----------|------|
| Method call | 10 |
| Operator call | 10 |
| Variable evaluation | 10 |
| `nth` | 10 |
| `keys` | 10 |
| `length` | 10 |
| `size` | collection size |
| Send evaluation | 11 |
| Receive evaluation | 11 |
| Channel evaluation | 11 |
| `new` binding | 2 per binding + 10 base |
| Match evaluation | 12 |

## Cost Table: Storage (RSpace)

Storage costs are proportional to protobuf-encoded sizes.

| Operation | Formula |
|-----------|---------|
| Produce (send) | `encoded_size(channel) + encoded_size(data)` |
| Consume (receive) | `sum(encoded_size(channels)) + sum(encoded_size(patterns)) + encoded_size(continuation)` |
| Event storage | `32 + (channels * 32)` bytes |
| Comm event | `event_storage(channels) + event_storage(1) * channels` |

## Equality Cost

```
equality_check_cost(x, y) = min(encoded_len(x), encoded_len(y))
```

Equality is bounded by the smaller operand since the comparison can short-circuit.

## Cost Optimization Tips

1. **Avoid large BigInt multiplication** -- cost is quadratic: `a_len * b_len`
2. **Use `fastUnsafeGet` for TreeHashMap** when you know the key exists -- avoids the existence check
3. **Minimize string interpolation on large maps** -- cost is `str_len * map_size`
4. **Prefer peek (`<<-`) over consume+resend** for read-only access -- saves storage costs
5. **Keep deploys focused** -- unused phlogiston is refunded, but overly broad deploys waste resources
6. **Use smaller phlo limits** when testing -- start with 90,000 and increase as needed

## Phlogiston Errors

When phlogiston is exhausted:
- Execution halts immediately
- All state changes from the current deploy are rolled back
- The `OutOfPhlogistonsError` is recorded in the deploy result
- The deployer is charged for the full `phlo_limit * phlo_price`

To diagnose: check the deploy result's `cost` field to see how much was consumed before the error.
