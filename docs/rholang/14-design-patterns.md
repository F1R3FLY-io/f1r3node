# Design Patterns

Common patterns for writing production Rholang contracts.

## Capability Security

Rholang's security model is capability-based: if you have a name, you can use it. If you don't, you can't construct it. All access control flows from this.

### Principle of Least Authority

Only give out the minimum capability needed.

```rho
new fullAccess in {
  // Internal: full access to read and write
  contract fullAccess(@"read", ret) = { ... } |
  contract fullAccess(@"write", @data, ack) = { ... } |

  // External: only expose read capability
  new readOnly in {
    contract readOnly(ret) = {
      fullAccess!("read", *ret)
    } |
    registry!("insert", bundle+{*readOnly})
  }
}
```

### Attenuating Forwarder

Create a restricted proxy for a channel:

```rho
new makeReadOnly in {
  contract makeReadOnly(source, readOnlyCh) = {
    new ro in {
      contract ro(ret) = {
        for (@value <<- source) {
          ret!(value)
        }
      } |
      readOnlyCh!(bundle+{*ro})
    }
  }
}
```

### Revocation

Create capabilities that can be revoked:

```rho
new makeRevocable in {
  contract makeRevocable(target, revocableCh, revokeCh) = {
    new active in {
      active!(true) |

      contract revokeCh(_) = {
        for (_ <- active) {
          active!(false)
        }
      } |

      new proxy in {
        contract proxy(@msg, ret) = {
          for (@isActive <<- active) {
            if (isActive) {
              target!(msg, *ret)
            } else {
              ret!("revoked")
            }
          }
        } |
        revocableCh!(bundle+{*proxy})
      }
    }
  }
}
```

## State Management

### Atomic Read-Modify-Write

Always consume, modify, and restore in one operation:

```rho
contract atomicUpdate(state, @f, ack) = {
  for (@current <- state) {
    match f(current) {
      newValue => {
        state!(newValue) |
        ack!(newValue)
      }
    }
  }
}
```

### State with Multiple Fields

Use a Map for structured state:

```rho
new state in {
  state!({"users": {}, "count": 0, "version": 1}) |

  contract addUser(@name, @data, ack) = {
    for (@s <- state) {
      state!(
        s.set("users", s.get("users").set(name, data))
         .set("count", s.get("count") + 1)
      ) |
      ack!(true)
    }
  }
}
```

### Event Sourcing

Store events rather than current state:

```rho
new events, state in {
  events!([]) |
  state!({"balance": 0}) |

  contract deposit(@amount, ack) = {
    for (@evts <- events; @s <- state) {
      events!(evts ++ [{"type": "deposit", "amount": amount}]) |
      state!(s.set("balance", s.get("balance") + amount)) |
      ack!(true)
    }
  }
}
```

### Method Dispatch Object

A single contract name that dispatches on a method string. This is the standard pattern for building service objects in production Rholang (used throughout Embers).

```rho
new lookup(`rho:registry:lookup`), envCh in {
  lookup!(`rho:id:my_service_uri`, *envCh) |
  for (@(_, agents) <- envCh) {

    contract agents(@"create", @id, @name, @metadata, ack) = {
      // ... create logic ...
      ack!((true, id))
    } |

    contract agents(@"get", @id, ret) = {
      // ... lookup logic ...
      ret!(data)
    } |

    contract agents(@"list", @address, ret) = {
      // ... list logic ...
      ret!(results)
    } |

    contract agents(@"delete", @id, ack) = {
      // ... delete logic ...
      ack!(true)
    }
  }
}
```

Callers use it like method calls:

```rho
@agents!("create", id, "My Agent", {"version": 1}, *ack) |
@agents!("list", myAddress, *ret)
```

### Persistent Library Reference

When you look up a library contract from the registry, `for` consumes the message. If you need the library in multiple places, re-send the reference to keep it available.

```rho
new lookup(`rho:registry:lookup`), treeHashMapCh in {
  lookup!(`rho:lang:treeHashMap`, *treeHashMapCh) |
  for (@(_, TreeHashMap) <- treeHashMapCh) {
    // Re-send so it stays available for later use
    treeHashMapCh!(TreeHashMap) |

    // First use
    @TreeHashMap!("init", 3, *mapCh) |
    for (@map <- mapCh) {

      // Second use (would block without the re-send above)
      for (@treeHashMap <- treeHashMapCh) {
        @treeHashMap!("set", map, "key", "value", *ack)
      }
    }
  }
}
```

Alternative: use peek (`<<-`) when the library is stored on a persistent channel:

```rho
for (@TreeHashMap <<- libCh) {
  // TreeHashMap is read without consuming -- channel still has the value
  @TreeHashMap!("get", map, "key", *ret)
}
```

### Latest Pointer (Versioned Data)

When storing versioned entries, maintain a separate "latest" key alongside version-specific entries. This avoids scanning all versions to find the most recent one.

```rho
// Store version-specific entry
@TreeHashMap!("set", versionsMap, versionId, data, *ack1) |

// Also update the "latest" pointer
@TreeHashMap!("set", versionsMap, "latest", data, *ack2) |

for (_ <- ack1; _ <- ack2) {
  ret!(true)
}
```

Reading the latest version is O(1):

```rho
@TreeHashMap!("get", versionsMap, "latest", *ret)
```

### Multi-Level TreeHashMap

For complex data models, nest TreeHashMaps to create hierarchical storage. Each level scopes by a different key (e.g., address -> resource ID -> version).

```rho
new lookup(`rho:registry:lookup`), treeHashMapCh,
    devNull(`rho:io:devNull`) in {
  lookup!(`rho:lang:treeHashMap`, *treeHashMapCh) |
  for (@(_, TreeHashMap) <- treeHashMapCh) {

    // Level 1: address -> resources
    @TreeHashMap!("init", 3, *level1Ch) |
    for (@addressMap <- level1Ch) {

      // Level 2: resource ID -> versions
      @TreeHashMap!("init", 3, *level2Ch) |
      for (@resourceMap <- level2Ch) {

        // Level 3: version -> data
        @TreeHashMap!("init", 3, *level3Ch) |
        for (@versionMap <- level3Ch) {

          // Store data at the deepest level
          @TreeHashMap!("set", versionMap, "v1", {"name": "doc", "size": 1024}, *devNull) |
          @TreeHashMap!("set", versionMap, "latest", {"name": "doc", "size": 1024}, *devNull) |

          // Link levels together
          @TreeHashMap!("set", resourceMap, "doc-001", versionMap, *devNull) |
          @TreeHashMap!("set", addressMap, myAddress, resourceMap, *devNull)
        }
      }
    }
  }
}
```

This pattern is used throughout Embers for address-scoped, versioned resource management (agents, agent teams, OSLFs).

## Communication Patterns

### Request-Response

The standard contract calling convention:

```rho
contract service(@request, ret) = {
  ret!(process(request))
}

// Client:
new ret in {
  service!(request, *ret) |
  for (@response <- ret) { ... }
}
```

### Pipeline

Chain multiple services:

```rho
new ret1 in {
  step1!(input, *ret1) |
  for (@r1 <- ret1) {
    new ret2 in {
      step2!(r1, *ret2) |
      for (@r2 <- ret2) {
        new ret3 in {
          step3!(r2, *ret3) |
          for (@result <- ret3) {
            stdout!(result)
          }
        }
      }
    }
  }
}
```

### Fan-Out / Fan-In

Process in parallel, collect results:

```rho
new r1, r2, r3 in {
  worker!(task1, *r1) |
  worker!(task2, *r2) |
  worker!(task3, *r3) |

  // Fan-in: wait for all results
  for (@res1 <- r1; @res2 <- r2; @res3 <- r3) {
    stdout!([res1, res2, res3])
  }
}
```

### Observer Pattern

Publish/subscribe using persistent channels:

```rho
new subscribe, publish, subscribers in {
  subscribers!(Set()) |

  contract subscribe(listener) = {
    for (@subs <- subscribers) {
      subscribers!(subs.add(*listener))
    }
  } |

  contract publish(@event) = {
    for (@subs <<- subscribers) {
      // Notify all subscribers
      new iter in {
        contract iter(@remaining) = {
          match remaining.toList() {
            [] => { Nil }
            [sub ...rest] => {
              @sub!(event) |
              iter!(rest.toSet())
            }
          }
        } |
        iter!(subs)
      }
    }
  }
}
```

## Error Handling

### Either Pattern

Use `(true, value)` / `(false, error)` tuples:

```rho
contract safeDivide(@a, @b, ret) = {
  if (b == 0) {
    ret!((false, "division by zero"))
  } else {
    ret!((true, a / b))
  }
}

// Usage:
new ret in {
  safeDivide!(10, 0, *ret) |
  for (@result <- ret) {
    match result {
      (true, value) => { stdout!(value) }
      (false, error) => { stderr!(error) }
    }
  }
}
```

The `Either.rho` system contract provides a more formal implementation with `map`, `flatMap`, and `fold`.

### Timeout Pattern

No built-in timeouts. Use a parallel process to provide a default:

```rho
new result in {
  // Try the real operation
  slowService!(request, *result) |

  // Provide a fallback (both race; first one consumed wins)
  result!((false, "timeout"))
}
```

## Cryptographic Authentication

### ECDSA Auth Pattern

The standard pattern for verifying that a message was signed by a specific key: hash the data with blake2b256, then verify the signature with secp256k1.

```rho
new blake2b256(`rho:crypto:blake2b256Hash`),
    secp256k1(`rho:crypto:secp256k1Verify`),
    stdout(`rho:io:stdout`) in {

  contract verify(@data, @signature, @publicKey, ret) = {
    new hashCh in {
      // Step 1: Hash the data
      blake2b256!(data.toByteArray(), *hashCh) |
      for (@hash <- hashCh) {
        // Step 2: Verify signature against hash
        new verifyCh in {
          secp256k1!(hash, signature, publicKey, *verifyCh) |
          for (@valid <- verifyCh) {
            ret!(valid)
          }
        }
      }
    }
  }
}
```

This is the pattern used by `rho:registry:insertSigned:secp256k1` and throughout the vault system for authenticating operations.

### Nonce-Based Replay Protection

Prevent replay attacks by requiring a monotonically increasing nonce with each signed operation. The contract stores the last-seen nonce and rejects any nonce that is not strictly greater.

```rho
contract SecureService(execute, @publicKey) = {
  new nonceCh, blake2b256(`rho:crypto:blake2b256Hash`),
      secp256k1(`rho:crypto:secp256k1Verify`) in {

    // Initial nonce
    nonceCh!(0) |

    contract execute(@nonce, @payload, @signature, ret) = {
      for (@lastNonce <- nonceCh) {
        if (nonce <= lastNonce) {
          // Reject: nonce must be strictly increasing
          nonceCh!(lastNonce) |
          ret!((false, "stale nonce"))
        } else {
          // Verify signature over (nonce, payload)
          new hashCh in {
            blake2b256!((nonce, payload).toByteArray(), *hashCh) |
            for (@hash <- hashCh) {
              new verifyCh in {
                secp256k1!(hash, signature, publicKey, *verifyCh) |
                for (@valid <- verifyCh) {
                  if (valid) {
                    // Accept: update nonce and process
                    nonceCh!(nonce) |
                    // ... process payload ...
                    ret!((true, "ok"))
                  } else {
                    nonceCh!(lastNonce) |
                    ret!((false, "invalid signature"))
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
```

Key points:
- The nonce channel acts as atomic state (consume + restore pattern)
- Each operation must include a nonce greater than the last accepted one
- The signature covers both the nonce and the payload, so an attacker cannot reuse a signature with a different nonce
- If verification fails, the old nonce is restored (state is not corrupted)

### Persistent Contract via Signed Registry Insert

Register an updatable contract using `insertSigned:secp256k1`. The contract can be updated by the key holder with a new nonce.

```rho
new insertSigned(`rho:registry:insertSigned:secp256k1`),
    lookup(`rho:registry:lookup`),
    stdout(`rho:io:stdout`) in {

  // Initial registration: (nonce, value) + signature
  new uriCh in {
    insertSigned!(publicKey, (nonce, contractProcess), signature, *uriCh) |
    for (@uri <- uriCh) {
      stdout!(["Registered at:", uri])
      // URI is deterministic based on the public key
      // Future updates use the same URI with an incremented nonce
    }
  }
}
```

The signed insert pattern ensures:
- Only the holder of the private key can register or update at the URI
- The nonce prevents replay of old registrations
- The URI is derived from the public key, so it is stable across updates

## Naming Conventions

- Contract names: `camelCase` (e.g., `findOrCreate`, `deployerAuthKey`)
- State channels: descriptive nouns (e.g., `state`, `counter`, `subscribers`)
- Ack channels: `ack`, `ret`, `done`, `result`
- System channel bindings: match the last segment (e.g., `stdout` for `rho:io:stdout`)
- Boolean results: prefix with `is` or use `(success, value)` tuples
