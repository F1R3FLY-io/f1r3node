# Contracts and State

Patterns for defining reusable services and managing mutable state in Rholang.

## Contracts

A contract is a persistent receiver -- it handles every message sent to its channel.

```rho
contract add(@x, @y, ret) = {
  ret!(x + y)
}
```

This is syntactic sugar for:

```rho
for (@x, @y, ret <= add) {
  ret!(x + y)
}
```

### Calling a Contract

```rho
new ret in {
  add!(3, 4, *ret) |
  for (@result <- ret) {
    stdout!(result)       // 7
  }
}
```

### Contract Lifecycle

- Created once
- Lives as long as the channel exists
- Handles messages concurrently (each invocation is independent)
- No explicit shutdown mechanism

## Mutable State: The Cell Pattern

The fundamental pattern for mutable state in Rholang. A channel holds a single value; operations consume it and put back a new value.

### Basic Cell

```rho
new state in {
  state!(0) |                    // initial value

  contract get(ret) = {
    for (@value <<- state) {     // peek (non-destructive read)
      ret!(value)
    }
  } |

  contract set(@newValue, ack) = {
    for (_ <- state) {           // consume old value
      state!(newValue) |         // store new value
      ack!(Nil)
    }
  }
}
```

### Cell with Get/Set/Update

```rho
contract MakeCell(@init, get, set) = {
  new state in {
    state!(init) |

    contract get(ret) = {
      for (@value <<- state) {
        ret!(value)
      }
    } |

    contract set(@newValue, ack) = {
      for (_ <- state) {
        state!(newValue) |
        ack!(Nil)
      }
    }
  }
}
```

Usage:

```rho
new get, set in {
  MakeCell!(0, *get, *set) |
  new ack in {
    set!(42, *ack) |
    for (_ <- ack) {
      new ret in {
        get!(*ret) |
        for (@value <- ret) {
          stdout!(value)   // 42
        }
      }
    }
  }
}
```

## CRUD Pattern

A complete create/read/update/delete service using a Map as state.

```rho
contract CrudService(create, read, update, delete, list) = {
  new state in {
    state!({}) |

    contract create(@key, @value, ack) = {
      for (@store <- state) {
        state!(store.set(key, value)) |
        ack!(true)
      }
    } |

    contract read(@key, ret) = {
      for (@store <<- state) {
        ret!(store.getOrElse(key, Nil))
      }
    } |

    contract update(@key, @value, ack) = {
      for (@store <- state) {
        if (store.contains(key)) {
          state!(store.set(key, value)) |
          ack!(true)
        } else {
          state!(store) |
          ack!(false)
        }
      }
    } |

    contract delete(@key, ack) = {
      for (@store <- state) {
        state!(store.delete(key)) |
        ack!(true)
      }
    } |

    contract list(ret) = {
      for (@store <<- state) {
        ret!(store)
      }
    }
  }
}
```

## Factory Pattern

Create instances with private state:

```rho
contract MakeCounter(inc, dec, get) = {
  new state in {
    state!(0) |

    contract inc(ack) = {
      for (@n <- state) {
        state!(n + 1) | ack!(Nil)
      }
    } |

    contract dec(ack) = {
      for (@n <- state) {
        state!(n - 1) | ack!(Nil)
      }
    } |

    contract get(ret) = {
      for (@n <<- state) {
        ret!(n)
      }
    }
  }
}

// Create two independent counters
new inc1, dec1, get1, inc2, dec2, get2 in {
  MakeCounter!(*inc1, *dec1, *get1) |
  MakeCounter!(*inc2, *dec2, *get2)
}
```

## Select (Non-Deterministic Choice)

Handle the first available message from multiple channels:

```rho
contract Cell(get, set, state) = {
  select {
    case ret <- get; @value <<- state => {
      ret!(value) | Cell!(*get, *set, *state)
    }
    case @newValue <- set; _ <- state => {
      state!(newValue) | Cell!(*get, *set, *state)
    }
  }
}
```

`select` picks whichever case has messages available first. If both are ready, the choice is non-deterministic.

## Recursive Contracts

Contracts cannot call themselves directly by name. Use explicit recursion via a channel:

```rho
new loop in {
  contract loop(@n) = {
    if (n > 0) {
      stdout!(n) |
      loop!(n - 1)
    }
  } |
  loop!(5)
}
```

Or using `match` for recursive list processing:

```rho
contract sum(@list, ret) = {
  match list {
    [] => { ret!(0) }
    [head ...tail] => {
      new tailSum in {
        sum!(tail, *tailSum) |
        for (@s <- tailSum) {
          ret!(head + s)
        }
      }
    }
  }
}
```

## Capability-Based API Design

Expose limited interfaces using bundles:

```rho
new internal in {
  // Full implementation with internal access
  contract internal(@"admin", @cmd, ret) = { ... } |
  contract internal(@"query", @key, ret) = { ... } |

  // Public read-only API
  new publicQuery in {
    contract publicQuery(@key, ret) = {
      internal!("query", key, *ret)
    } |
    // Export only the query capability
    registry!("insert", bundle+{*publicQuery})
  }
}
```

## Common Pitfalls

### Deadlock

```rho
// DEADLOCK: both channels wait for each other
for (@x <- chan1; @y <- chan2) { ... } |
for (@a <- chan2; @b <- chan1) { ... }
```

Both joins need messages on both channels, but neither can proceed.

### Lost Updates

```rho
// BUG: two concurrent reads see the same value
for (@n <- counter) { counter!(n + 1) } |
for (@n <- counter) { counter!(n + 1) }
// Expected: +2. Actual: +1 (second read sees original value)
```

Fix: serialize access through a single contract:

```rho
contract increment(ack) = {
  for (@n <- counter) {
    counter!(n + 1) | ack!(Nil)
  }
}
```

### Forgetting to Put State Back

```rho
// BUG: state consumed but never restored on error path
for (@store <- state) {
  if (store.contains(key)) {
    state!(store.delete(key))     // OK: state restored
  }
  // MISSING: else { state!(store) } -- state is lost!
}
```

Always ensure the state channel gets a value back on every code path.
