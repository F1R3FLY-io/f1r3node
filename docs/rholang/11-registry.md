# Registry

The registry provides persistent name resolution -- register a process under a URI, look it up later (even from a different deploy).

## Overview

The registry is implemented in Rholang itself (`casper/src/main/resources/Registry.rho`). It uses a TreeHashMap internally for O(log n) operations and is bootstrapped during genesis.

Three registry channels are pre-bound as system channels:
- `rho:registry:lookup` -- find a process by URI
- `rho:registry:insertArbitrary` -- register with auto-generated URI
- `rho:registry:insertSigned:secp256k1` -- register with cryptographic verification

## Lookup

Retrieve a registered process by URI.

```rho
new lookup(`rho:registry:lookup`) in {
  new ret in {
    lookup!(`rho:id:someuri`, *ret) |
    for (value <- ret) {
      // *value is the registered process
      stdout!(*value)
    }
  }
}
```

Note: `value` is a name (not `@value`). The registered process is received as a name and dereferenced with `*value`.

If the URI is not found, the receive blocks indefinitely (no error, no Nil return).

## Insert Arbitrary

Register any process and receive a generated URI.

```rho
new insert(`rho:registry:insertArbitrary`) in {
  new ret, myService in {
    contract myService(@msg, ack) = {
      ack!(msg ++ " processed")
    } |

    // Register a bundled (write-only) reference
    insert!(bundle+{*myService}, *ret) |
    for (@uri <- ret) {
      stdout!(uri)    // e.g., rho:id:abc123...
    }
  }
}
```

The generated URI is a `rho:id:*` URI derived from the deployer and unforgeable name, encoded as base32 with a CRC14 checksum.

Best practice: register `bundle+{*name}` (write-only) so consumers can call the service but not intercept messages meant for it.

## Insert Signed

Register with secp256k1 signature verification. This is used for contracts that need authenticated updates.

```rho
new insertSigned(`rho:registry:insertSigned:secp256k1`) in {
  new ret in {
    insertSigned!(publicKey, (nonce, value), signature, *ret) |
    for (@uri <- ret) {
      stdout!(uri)
    }
  }
}
```

## Registry Ops

The `rho:registry:ops` channel provides utility operations.

```rho
new regOps(`rho:registry:ops`) in {
  new ret in {
    regOps!("buildUri", someData, *ret) |
    for (@uri <- ret) {
      stdout!(uri)
    }
  }
}
```

## TreeHashMap

The registry includes a TreeHashMap implementation available to any Rholang contract. It provides O(log n) persistent hash maps.

### Initialize

```rho
// depth controls parallelization: 3 means 3*8 = 24 bits
TreeHashMap!("init", 3, *mapCh) |
for (@map <- mapCh) {
  // map is the handle for subsequent operations
}
```

### Operations

```rho
// Set a key
TreeHashMap!("set", map, "key", "value", *ack)

// Get a key (returns Nil if not found)
TreeHashMap!("get", map, "key", *ret)

// Fast unsafe get (assumes key exists -- undefined behavior if not)
TreeHashMap!("fastUnsafeGet", map, "key", *ret)

// Check if key exists
TreeHashMap!("contains", map, "key", *ret)    // returns boolean

// Delete a key
TreeHashMap!("delete", map, "key", *ack)
```

### How It Works

TreeHashMap uses keccak256 hashing to distribute keys across a nybble-tree structure. Each key is hashed, and the hash is decomposed into nybbles (4-bit values) to navigate the tree. The `depth` parameter controls how many nybbles of parallelization are used.

Key properties:
- Lookups use only peeks (no conflicts between concurrent reads)
- Inserts conflict only when keys share a common prefix that hasn't been populated
- O(log n) for both insert and lookup
- O(1) lookup via `fastUnsafeGet` when you know the key exists

### Example: Key-Value Store

```rho
new stdout(`rho:io:stdout`), mapCh in {
  TreeHashMap!("init", 3, *mapCh) |
  for (@map <- mapCh) {
    new ack1, ack2, ret in {
      TreeHashMap!("set", map, "alice", {"balance": 100}, *ack1) |
      TreeHashMap!("set", map, "bob", {"balance": 50}, *ack2) |
      for (_ <- ack1; _ <- ack2) {
        TreeHashMap!("get", map, "alice", *ret) |
        for (@value <- ret) {
          stdout!(value)   // {"balance": 100}
        }
      }
    }
  }
}
```

## Standard Library URIs

Several library contracts are pre-registered during genesis and available via lookup:

| URI | Contract | Source File |
|-----|----------|-------------|
| `rho:lang:treeHashMap` | TreeHashMap | `Registry.rho` (embedded) |
| `rho:lang:listOps` | ListOps | `ListOps.rho` |
| `rho:lang:either` | Either | `Either.rho` |
| `rho:lang:stack` | Stack | `Stack.rho` |
| `rho:system:makeMint` | MakeMint | `MakeMint.rho` |
| `rho:vault:system` | SystemVault | `SystemVault.rho` |

Access them the same way as any registry entry:

```rho
new lookup(`rho:registry:lookup`), treeHashMapCh, listOpsCh in {
  lookup!(`rho:lang:treeHashMap`, *treeHashMapCh) |
  lookup!(`rho:lang:listOps`, *listOpsCh) |
  for (@(_, TreeHashMap) <- treeHashMapCh;
       @(_, ListOps) <- listOpsCh) {
    // TreeHashMap and ListOps are now available
  }
}
```

The lookup returns a `(nonce, contract)` tuple. Destructure with `@(_, ContractName)` to discard the nonce.

## Common Patterns

### Service Registration

```rho
new insert(`rho:registry:insertArbitrary`),
    lookup(`rho:registry:lookup`),
    stdout(`rho:io:stdout`) in {

  // Producer: register a service
  new myService in {
    contract myService(@request, ret) = {
      match request {
        {"action": "echo", "data": data} => { ret!(data) }
        _ => { ret!("unknown request") }
      }
    } |
    new uriCh in {
      insert!(bundle+{*myService}, *uriCh) |
      for (@uri <- uriCh) {
        stdout!(["Service registered at:", uri])
      }
    }
  }
}
```

### Looking Up a Known URI

```rho
new lookup(`rho:registry:lookup`) in {
  new ret in {
    // Use a known URI (from a previous registration or published)
    lookup!(`rho:id:known_service_uri`, *ret) |
    for (service <- ret) {
      new result in {
        service!({"action": "echo", "data": "hello"}, *result) |
        for (@response <- result) {
          stdout!(response)
        }
      }
    }
  }
}
```
