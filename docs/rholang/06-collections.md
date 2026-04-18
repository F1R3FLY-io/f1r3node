# Collections

All collection types and their methods, verified against `reduce.rs`.

## List

Ordered, indexed sequence. Allows duplicates.

```rho
[]                    // empty list
[1, 2, 3]
["a", "b", "c"]
[1, "mixed", true]   // heterogeneous
```

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `nth(i)` | 1 (Int) | Par | Element at index (0-based). Error if out of bounds. |
| `length()` | 0 | Int | Number of elements |
| `slice(from, until)` | 2 (Int, Int) | List | Sub-list, 0-based, exclusive end. Returns empty if bounds invalid. |
| `take(n)` | 1 (Int) | List | First n elements |
| `toList()` | 0 | List | Identity |
| `toSet()` | 0 | Set | Convert to set (removes duplicates) |
| `toByteArray()` | 0 | ByteArray | Protobuf encoding |

### Operators

```rho
[1, 2] ++ [3, 4]     // [1, 2, 3, 4] (concatenation)
```

### Pattern Destructuring

```rho
match list {
  [] => { ... }                // empty
  [head ...tail] => { ... }   // head + rest
  [a, b, c] => { ... }        // exact 3 elements
}
```

## Tuple

Fixed-size, ordered collection. Immutable. Cannot be constructed dynamically.

```rho
(1, 2)
(1, "hello", true)
(1, (2, 3))          // nested
```

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `nth(i)` | 1 (Int) | Par | Element at index (0-based) |
| `toList()` | 0 | List | Convert to list |
| `toByteArray()` | 0 | ByteArray | Protobuf encoding |

Tuples have no `length()` method. Use pattern matching to decompose.

## Set

Unordered collection of unique elements. Backed by `SortedParHashSet`.

```rho
Set()                 // empty set
Set(1, 2, 3)
Set("a", "b")
```

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `contains(elem)` | 1 | Bool | Membership test |
| `add(elem)` | 1 | Set | New set with element added |
| `delete(elem)` | 1 | Set | New set with element removed |
| `union(other)` | 1 (Set) | Set | Set union |
| `diff(other)` | 1 (Set) | Set | Set difference (elements in this but not other) |
| `size()` | 0 | Int | Cardinality |
| `toList()` | 0 | List | Convert to list |
| `toSet()` | 0 | Set | Identity |
| `toByteArray()` | 0 | ByteArray | Protobuf encoding |

### Operators

```rho
Set(1, 2, 3) -- Set(2)    // Set(1, 3) (difference)
```

Sets are immutable. All mutation methods return new sets.

## Map

Key-value mapping. Backed by `SortedParMap`. Keys can be any Par (not just strings).

```rho
{}                           // empty map
{"name": "Alice", "age": 30}
{1: "one", 2: "two"}        // integer keys
{@"chan": value}             // name keys
```

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `get(key)` | 1 | Par | Value for key, or `Nil` if absent |
| `getOrElse(key, default)` | 2 | Par | Value for key, or default if absent |
| `set(key, value)` | 2 | Map | New map with key set |
| `contains(key)` | 1 | Bool | Key existence test |
| `delete(key)` | 1 | Map | New map with key removed |
| `keys()` | 0 | Set | Set of all keys |
| `size()` | 0 | Int | Number of entries |
| `union(other)` | 1 (Map) | Map | Merge maps (other's values win on conflict) |
| `diff(other)` | 1 (Map) | Map | Remove keys present in other |
| `toList()` | 0 | List | List of `(key, value)` tuples |
| `toSet()` | 0 | Set | Set of `(key, value)` tuples |
| `toMap()` | 0 | Map | Identity |
| `toByteArray()` | 0 | ByteArray | Protobuf encoding |

### Operators

```rho
map1 - map2    // subtraction (same as diff)
```

Maps are immutable. All mutation methods return new maps.

### Pattern Matching

Maps support partial matching -- you don't need to match all keys:

```rho
match {"name": "Alice", "age": 30} {
  {"name": name} => { stdout!(name) }    // matches, binds name="Alice"
}
```

## Type Conversion Summary

| From | `.toList()` | `.toSet()` | `.toMap()` |
|------|-------------|------------|------------|
| List | identity | removes dupes | error unless `[(k,v)]` pairs |
| Tuple | to list | - | - |
| Set | to list | identity | error unless `{(k,v)}` pairs |
| Map | `[(k,v)]` pairs | `{(k,v)}` pairs | identity |

## Common Patterns

### Building a map from a list

```rho
// Start with empty map, add entries
new state in {
  state!({}) |
  for (@map <- state) {
    state!(map.set("key1", "value1"))
  }
}
```

### Checking membership before access

```rho
if (map.contains("key")) {
  stdout!(map.get("key"))
} else {
  stdout!("not found")
}
```

### Iterating a list

```rho
contract forEach(@list, body, done) = {
  match list {
    [] => { done!(Nil) }
    [head ...tail] => {
      new ack in {
        body!(head, *ack) |
        for (_ <- ack) {
          forEach!(tail, *body, *done)
        }
      }
    }
  }
}
```
