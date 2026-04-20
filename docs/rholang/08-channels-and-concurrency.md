# Channels and Concurrency

All communication in Rholang happens through channels. This document covers send/receive semantics, persistence, joins, and concurrency patterns.

## Send

Send a message on a channel. Non-blocking -- execution continues immediately.

### Single Send (`!`)

The message is consumed by at most one receiver. After consumption, it is gone.

```rho
chan!("hello")                // send one value
chan!(1, 2, 3)                // send three values (arity 3)
chan!(*otherChan)              // send a name
```

### Persistent Send (`!!`)

The message persists and can be consumed by multiple receivers.

```rho
config!!("production")        // every receiver on config gets "production"
```

## Receive

Wait for a message on a channel. Blocks until a matching message arrives.

### Single Receive (`<-`)

Consumes one message, then the body executes once.

```rho
for (@msg <- chan) {
  stdout!(msg)
}
```

### Persistent Receive (`<=`)

Keeps listening. Each incoming message triggers a new body execution.

```rho
for (@msg <= chan) {
  stdout!(msg)        // executes for every message sent on chan
}
```

This is equivalent to `contract`:

```rho
contract chan(@msg) = {
  stdout!(msg)
}
```

### Peek (`<<-`)

Read a message without consuming it. The message remains available.

```rho
for (@msg <<- chan) {
  stdout!(msg)        // msg is read but NOT removed from chan
}
```

Useful for reading shared state without disrupting other readers.

## Arity

Send and receive must agree on arity (number of values):

```rho
// Arity 2 send
chan!("Alice", 30)

// Must match arity 2 receive
for (@name, @age <- chan) {
  stdout!(name ++ " is " ++ age)
}
```

Mismatched arity means the send/receive will never match.

## Joins

Wait for messages on multiple channels simultaneously. The body executes only when ALL channels have a message.

```rho
for (@x <- chan1; @y <- chan2) {
  stdout!(x + y)
}
```

This is a synchronization primitive. The body does not execute until both `chan1` and `chan2` have pending messages.

### Join vs Parallel Receives

```rho
// JOIN: waits for BOTH channels
for (@x <- chan1; @y <- chan2) { ... }

// PARALLEL: two independent receives
for (@x <- chan1) { ... } | for (@y <- chan2) { ... }
```

## Parallel Composition

The `|` operator runs processes concurrently. This is Rholang's primary composition mechanism.

```rho
process1 | process2 | process3
```

Each process runs independently. They only interact through shared channels.

### Ordering

There is no guaranteed ordering. In:

```rho
chan!(1) | chan!(2) | chan!(3) |
for (@x <- chan) { stdout!(x) }
```

The receiver might get 1, 2, or 3 -- it's non-deterministic.

### Acknowledgment Pattern

Use acknowledgment channels for sequencing:

```rho
new ack in {
  step1!(*ack) |
  for (_ <- ack) {
    step2!(*ack2) |
    for (_ <- ack2) {
      step3!(Nil)
    }
  }
}
```

This is the idiomatic way to express sequential operations in Rholang.

## Channels as State

A channel with one message acts like a mutable variable:

```rho
new counter in {
  counter!(0) |          // initial state

  // Increment: read, add 1, write back
  for (@n <- counter) {
    counter!(n + 1)
  }
}
```

This is safe because `for` consumes the message atomically. No two readers can see the same value.

### Read-Modify-Write

```rho
new state in {
  state!({"count": 0, "name": "default"}) |

  // Update
  for (@current <- state) {
    state!(current.set("count", current.get("count") + 1))
  }
}
```

## Name Creation with `new`

Every `new` creates fresh, unforgeable channels.

```rho
new x in { x!(42) }
new x in { x!(42) }    // different x! These are separate channels.
```

Binding system channels:

```rho
new stdout(`rho:io:stdout`),
    sha256(`rho:crypto:sha256Hash`),
    registry(`rho:registry:lookup`) in {
  ...
}
```

## Bundles

Restrict what can be done with a name:

```rho
new privateChan in {
  // Give someone write-only access
  publicApi!(bundle+{*privateChan}) |

  // We keep full access
  for (@msg <- privateChan) {
    process!(msg)
  }
}
```

| Bundle | Can Send | Can Receive |
|--------|----------|-------------|
| `bundle+{name}` | Yes | No |
| `bundle-{name}` | No | Yes |
| `bundle0{name}` | No | No |

## Common Patterns

### Request-Response

```rho
contract service(@request, ret) = {
  // Process request, send result back on ret
  ret!(request + 1)
}

// Client
new ret in {
  service!(42, *ret) |
  for (@result <- ret) {
    stdout!(result)     // 43
  }
}
```

### Fan-Out

Send to multiple listeners:

```rho
new topic in {
  // Publisher
  topic!("event1") | topic!("event2") |

  // Subscribers (persistent receive)
  for (@event <= topic) { handler1!(event) } |
  for (@event <= topic) { handler2!(event) }
}
```

### Barrier

Wait for N processes to complete:

```rho
new done in {
  // Launch 3 tasks
  task1!(*done) | task2!(*done) | task3!(*done) |

  // Wait for all 3
  for (_ <- done; _ <- done; _ <- done) {
    stdout!("all tasks complete")
  }
}
```

### Mutex

```rho
new mutex in {
  mutex!(Nil) |         // initially unlocked

  // Critical section
  for (_ <- mutex) {    // acquire (consume token)
    doWork!(Nil) |
    for (_ <- workDone) {
      mutex!(Nil)       // release (put token back)
    }
  }
}
```
