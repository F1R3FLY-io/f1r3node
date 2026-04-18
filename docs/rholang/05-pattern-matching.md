# Pattern Matching

Rholang's pattern matching is structural -- it matches on the shape and content of processes and names.

## Where Patterns Appear

### In `for` (receive)

```rho
for (@x <- chan) { body }                   // bind process to x
for (@{x, y} <- chan) { body }              // destructure tuple
for (@[head ...tail] <- chan) { body }      // destructure list
```

### In `match`

```rho
match value {
  42 => { stdout!("the answer") }
  x => { stdout!(x) }          // variable binds anything
  _ => { stdout!("wildcard") } // discard
}
```

### In `=~` (inline test)

```rho
value =~ pattern    // returns boolean
```

## Pattern Types

### Literal Patterns

Match exact values:

```rho
match x {
  42 => { ... }
  "hello" => { ... }
  true => { ... }
  Nil => { ... }
}
```

### Variable Patterns

Bind the matched value to a name:

```rho
match x {
  y => { stdout!(y) }    // y binds to whatever x is
}
```

In `for`, process variables use `@`:

```rho
for (@value <- chan) { ... }    // value is a process variable
for (name <- chan) { ... }      // name is a name variable
```

### Wildcard

Match anything, discard the value:

```rho
match x {
  _ => { stdout!("matched anything") }
}

for (_ <- chan) { ... }    // consume but ignore
```

### Tuple Destructuring

```rho
match (1, 2, 3) {
  (a, b, c) => { stdout!(a + b + c) }
}
```

### List Destructuring

```rho
match [1, 2, 3] {
  [] => { stdout!("empty") }
  [head ...tail] => { stdout!(head) }    // head=1, tail=[2,3]
  [a, b, c] => { stdout!(b) }            // exact length match
}
```

### Map Destructuring

```rho
match {"name": "Alice", "age": 30} {
  {"name": name} => { stdout!(name) }    // partial match, binds name
}
```

### Process Patterns

Match on process structure:

```rho
match value {
  @{chan!(msg)} => { ... }     // matches a send
  @{for (@x <- ch) { body }} => { ... }  // matches a receive
}
```

## Type Guards

Match on the type of a value:

```rho
match value {
  x: Int => { stdout!("integer: " ++ x.toString()) }
  x: Bool => { stdout!("boolean") }
  x: String => { stdout!("string: " ++ x) }
  x: Uri => { stdout!("URI") }
  x: ByteArray => { stdout!("bytes") }
  _ => { stdout!("other type") }
}
```

## Logical Connectives

### AND (`/\`)

Both patterns must match:

```rho
for (@{x /\ y} <- chan) { ... }    // x and y both bind to the same value

match value {
  x /\ 42 => { ... }    // binds x AND requires value == 42
}
```

### OR (`\/`)

Either pattern matches:

```rho
match value {
  42 \/ 43 => { stdout!("42 or 43") }
}
```

### NOT (`~`)

Negation -- match succeeds if the inner pattern does NOT match:

```rho
match value {
  ~42 => { stdout!("not 42") }
  ~Nil => { stdout!("not nil") }
}
```

### Combining Connectives

```rho
// Match a non-nil, non-empty-string value and bind it
match value {
  x /\ ~Nil /\ ~"" => { stdout!(x) }
  _ => { stdout!("nil or empty") }
}
```

**Precedence**: `~` (NOT) binds tightest, then `/\` (AND), then `\/` (OR).

## Pattern Restrictions

Patterns can only appear in designated positions:

- To the left of `<-` in `for`
- After `case` in `match`
- Right side of `=~`

These are NOT valid standalone processes:

```rho
// INVALID - connectives outside pattern context:
@Nil!(Nil) /\ @Nil!(Nil)
for (x <- @"ch") { _ }       // wildcard not in pattern position
[1, 2 ... x]                 // remainder not in pattern position
```

## Name Equivalence

Two names are equal if and only if the processes they quote are structurally equivalent:

```rho
@{P | Q} == @{Q | P}    // true: parallel composition is commutative
@{P | Nil} == @{P}      // true: Nil is identity for |
```

This matters because `for` matching on a channel compares names structurally.

## Examples

### Recursive list processing

```rho
contract forEach(@list, proc, ret) = {
  match list {
    [] => { ret!(Nil) }
    [head ...tail] => {
      new ack in {
        proc!(head, *ack) |
        for (_ <- ack) {
          forEach!(tail, *proc, *ret)
        }
      }
    }
  }
}
```

### Multi-case dispatch

```rho
contract handler(@request, ret) = {
  match request {
    {"action": "get", "key": key} => {
      lookup!(key, *ret)
    }
    {"action": "set", "key": key, "value": value} => {
      store!(key, value, *ret)
    }
    _ => {
      ret!("unknown action")
    }
  }
}
```

### Guard-like patterns

```rho
for (@x /\ ~0 <- divisorChan) {
  // x is guaranteed non-zero
  result!(100 / x)
}
```
