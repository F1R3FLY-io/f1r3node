# Language Overview

Rholang is a concurrent programming language based on the rho-calculus (reflective, higher-order pi-calculus). All computation happens through message passing on channels. There are no variables in the traditional sense -- only names (channels) and processes.

## Two Kinds of Values

Rholang has exactly two kinds of values:

- **Processes** -- everything that computes: sends, receives, arithmetic, data structures, even literals like `42` or `"hello"`
- **Names** -- communication channels, created with `new` or by quoting a process with `@`

The relationship between them:
- `@P` quotes process `P` into a name (a channel whose identity is `P`)
- `*x` dereferences name `x` back into a process

```rho
new chan in {
  chan!(42) |              // send the process 42 on channel chan
  for (@value <- chan) {   // receive, binding the process to value
    stdout!(value)         // value is 42
  }
}
```

## Core Computational Model

### Message Passing

All communication is asynchronous. Sending a message does not block. There is no way to send a message and then do something "once it is received" without explicitly waiting for an acknowledgment.

```rho
// This does NOT guarantee ordering:
chan!("first") | chan!("second")

// For ordering, use acknowledgment channels:
new ack in {
  chan!("first", *ack) |
  for (_ <- ack) {
    chan!("second")
  }
}
```

### Channels Are Bags, Not Queues

Messages on a channel form a multiset (bag), not a queue. There is no guaranteed ordering of messages. If you send `1`, `2`, `3` on the same channel, a receiver might get any of them first.

### Parallel Composition

The `|` operator runs processes concurrently. This is fundamental -- not syntactic sugar.

```rho
// Three independent processes running concurrently
process1 | process2 | process3
```

### Unforgeable Names

Names created with `new` are cryptographically unique. No other process can guess or reconstruct them. Even if the bits of a private name are visible on the blockchain, there is no language construct to turn bits back into that name.

```rho
new privateChan in {
  // Only code inside this block can use privateChan
  privateChan!("secret") |
  for (@msg <- privateChan) {
    stdout!(msg)
  }
}
```

This is the foundation of Rholang's security model: capability-based security through unforgeable names.

## Program Structure

A Rholang program is a process. The simplest process is `Nil` (does nothing). Processes compose with `|`:

```rho
// A complete program:
new stdout(`rho:io:stdout`) in {
  stdout!("Hello, World!")
}
```

The `new ... in { ... }` block:
1. Creates fresh unforgeable names
2. Optionally binds system channels via URI (e.g., `` `rho:io:stdout` ``)
3. Executes the body process

## Execution Model

Rholang executes on a tuple space called RSpace. The runtime:
1. **Parses** source code into an AST
2. **Normalizes** the AST into a canonical form (De Bruijn indices, sorted terms)
3. **Reduces** by matching sends against receives in the tuple space
4. Charges **phlogiston** (gas) for each operation

Reduction is non-deterministic when multiple matches are possible. The runtime picks one. Programs that depend on a specific reduction order are not portable.

## Key Differences from Conventional Languages

| Concept | Conventional | Rholang |
|---------|-------------|---------|
| Variables | Mutable storage locations | Names (channels) |
| Assignment | `x = 5` | `chan!(5)` (send) |
| Read | `x` | `for (@v <- chan) { ... }` (receive) |
| Function call | `f(x)` | `f!(x, resultChan)` (send args + return channel) |
| Sequential | `a; b` | `a | for (_ <- ack) { b }` (ack pattern) |
| Concurrency | Threads, async/await | `a \| b` (parallel composition) |
| Encapsulation | Private fields | Unforgeable names via `new` |
| No implicit coercion | Varies | All binary ops require matching types |
