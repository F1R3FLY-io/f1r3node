# Syntax Reference

Complete syntax for all Rholang language constructs as implemented in the Rust shard.

## Processes

### Nil

The empty process. Does nothing.

```rho
Nil
```

### Send

Send a message on a channel. The message is one or more processes.

```rho
chan!(msg)                    // single send (consumed once)
chan!!(msg)                   // persistent send (can be consumed multiple times)
chan!(msg1, msg2, msg3)       // send multiple values (arity 3)
```

### Receive

Wait for a message on one or more channels.

```rho
for (@x <- chan) { body }                 // single receive (consumed once)
for (@x <= chan) { body }                 // persistent receive (reusable)
for (@x <<- chan) { body }                // peek (read without consuming)
for (@x <- chan1; @y <- chan2) { body }   // join: wait for messages on both
for (@x <- chan1 & @y <- chan2) { body }  // alternative join syntax
```

The `@` prefix in `@x` means "bind the process arriving on this channel to the name `x`". Without `@`, you bind a name directly.

### Contract

Syntactic sugar for a persistent receive. A contract keeps listening for messages.

```rho
contract name(@arg1, @arg2, ret) = {
  ret!(arg1 + arg2)
}
```

is equivalent to:

```rho
for (@arg1, @arg2, ret <= name) {
  ret!(arg1 + arg2)
}
```

### New

Create fresh unforgeable names.

```rho
new x in { body }
new x, y, z in { body }
new stdout(`rho:io:stdout`) in { body }    // bind to system channel
new x, stdout(`rho:io:stdout`), ack in { body }
```

### Match

Pattern matching with multiple cases.

```rho
match expr {
  pattern1 => { body1 }
  pattern2 => { body2 }
  _ => { default_body }
}
```

### Select

Non-deterministic choice. Executes the first case where a message is available.

```rho
select {
  case @x <- chan1 => { body1 }
  case @y <- chan2 => { body2 }
}
```

### If / Else

Conditional execution. Sugar for `match`.

```rho
if (condition) { trueBody }
if (condition) { trueBody } else { falseBody }
```

is equivalent to:

```rho
match condition {
  true => { trueBody }
  false => { falseBody }
}
```

### Parallel Composition

Run processes concurrently.

```rho
process1 | process2 | process3
```

### Bundle

Restrict capabilities on a name.

```rho
bundle+{name}   // write-only: can send, cannot receive
bundle-{name}   // read-only: can receive, cannot send
bundle0{name}   // no capability (sealed)
bundle+{*name}  // write-only on the name's process
```

## Names

### Quoted Process

Turn a process into a name.

```rho
@42              // the name whose identity is the process 42
@"hello"         // the name whose identity is the string "hello"
@{x + y}         // the name whose identity is the expression x + y
```

### Dereference

Turn a name back into a process.

```rho
*name            // the process that name refers to
```

### System URIs

Bind to built-in system channels.

```rho
`rho:io:stdout`
`rho:crypto:sha256Hash`
`rho:registry:lookup`
```

Always used inside `new`:

```rho
new stdout(`rho:io:stdout`), sha256(`rho:crypto:sha256Hash`) in {
  ...
}
```

## Expressions

### Arithmetic

```rho
x + y    // addition (Int, Float, BigInt, BigRat, FixedPoint)
x - y    // subtraction
x * y    // multiplication
x / y    // division
x % y    // modulo (not defined on Float)
```

### Comparison

```rho
x < y    x <= y    x > y    x >= y
x == y   x != y
```

### Logical

```rho
x and y    // logical AND (requires booleans)
x or y     // logical OR
not x      // logical NOT
```

### String and Collection

```rho
"hello" ++ " world"           // string concatenation: "hello world"
[1, 2] ++ [3, 4]              // list concatenation: [1, 2, 3, 4]
Set(1, 2, 3) -- Set(2)        // set difference: Set(1, 3)
"Hello ${name}" %% {"name": "World"}  // string interpolation
```

### Pattern Test

```rho
expr =~ pattern              // returns true if expr matches pattern
```

### Method Calls

```rho
target.method(args)
[1, 2, 3].nth(0)            // 1
{"a": 1}.get("a")           // 1
"hello".length()             // 5
```

## Literals

| Type | Syntax | Examples |
|------|--------|----------|
| Integer | digits | `0`, `42`, `-7` |
| Float | digits + f32/f64 | `3.14f64`, `2.5f32` |
| BigInt | digits + n | `100n`, `999999999999999n` |
| BigRat | expr with r suffix | `1r`, `3r` (use `1r / 3r` for rationals) |
| FixedPoint | decimal + p + scale | `1.50p2`, `10p0`, `0.001p3` |
| String | double quotes | `"hello"`, `"line\nbreak"` |
| Boolean | keywords | `true`, `false` |
| URI | backtick quotes | `` `rho:io:stdout` `` |
| ByteArray | via method | `"deadbeef".hexToBytes()` |
| Nil | keyword | `Nil` |

## Collections

```rho
[1, 2, 3]                     // List
(1, 2, 3)                     // Tuple
Set(1, 2, 3)                  // Set
{"key1": val1, "key2": val2}  // Map
```

See [Collections](06-collections.md) for methods.

## Comments

```rho
// single line comment
/* multi-line
   comment */
```
