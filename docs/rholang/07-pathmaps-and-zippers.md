# PathMaps and Zippers

PathMaps are hierarchical trie structures for organizing data by path segments. Zippers provide cursor-based navigation and mutation over PathMaps.

## PathMap

A PathMap stores values at paths (lists of keys). Think of it as a filesystem-like tree.

### Literal Syntax

```rho
{| ["books", "fiction", "gatsby"],
   ["books", "fiction", "moby"],
   ["books", "nonfiction", "history"],
   ["music", "jazz"] |}
```

This creates a trie:
```
books/
  fiction/
    gatsby
    moby
  nonfiction/
    history
music/
  jazz
```

### PathMap Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `union(other)` | 1 (PathMap) | PathMap | Merge two PathMaps |
| `diff(other)` | 1 (PathMap) | PathMap | Remove paths present in other |
| `intersection(other)` | 1 (PathMap) | PathMap | Keep only common paths |
| `restriction(other)` | 1 (PathMap) | PathMap | Keep only paths under prefixes in other |
| `dropHead(n)` | 1 (Int) | PathMap | Remove first n segments from all paths |
| `run(env)` | 1 (PathMap) | PathMap | Execute PathMap in environment |
| `atPath(path)` | 1 (List) | Par | Value at path, or Nil |
| `pathExists(path)` | 1 (List) | Bool | Check if path exists |
| `createPath(path)` | 1 (List) | PathMap | Ensure path exists |
| `prunePath(path)` | 1 (List) | PathMap | Remove a path |
| `getLeaf()` | 0 | Par | Value at root |
| `getSubtrie()` | 0 | PathMap | Entire trie |
| `readZipper()` | 0 | Zipper | Read-only cursor at root |
| `readZipperAt(path)` | 1 (List) | Zipper | Read-only cursor at path |
| `writeZipper()` | 0 | Zipper | Write cursor at root |
| `writeZipperAt(path)` | 1 (List) | Zipper | Write cursor at path |

### Examples

```rho
new stdout(`rho:io:stdout`) in {
  // Union
  {| ["books", "gatsby"] |}.union({| ["music", "jazz"] |})
  // -> {| ["books", "gatsby"], ["music", "jazz"] |}

  // Intersection
  {| ["a", "b"], ["a", "c"] |}.intersection({| ["a", "b"], ["x", "y"] |})
  // -> {| ["a", "b"] |}

  // Restriction: keep subtree under prefix
  {| ["books", "fiction", "gatsby"],
     ["books", "nonfiction", "history"] |}.restriction({| ["books", "fiction"] |})
  // -> {| ["books", "fiction", "gatsby"] |}

  // Drop first path segment
  {| ["books", "gatsby"], ["books", "moby"] |}.dropHead(1)
  // -> {| ["gatsby"], ["moby"] |}
}
```

## Zippers

A Zipper is a cursor into a PathMap that tracks your position. There are two modes:

- **Read Zipper** -- navigate and read, cannot modify
- **Write Zipper** -- navigate, read, and modify

### Creating Zippers

```rho
// From a PathMap
pathmap.readZipper()           // read cursor at root
pathmap.readZipperAt(["a"])    // read cursor at path ["a"]
pathmap.writeZipper()          // write cursor at root
pathmap.writeZipperAt(["a"])   // write cursor at path ["a"]
```

### Navigation Methods

All navigation methods return a new Zipper at the new position.

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `descendTo(path)` | 1 (List) | Zipper | Navigate to a child path |
| `descendFirst()` | 0 | Zipper | Navigate to first child |
| `descendIndexedBranch(i)` | 1 (Int) | Zipper | Navigate to i-th child |
| `toNextSibling()` | 0 | Zipper | Move to next sibling |
| `toPrevSibling()` | 0 | Zipper | Move to previous sibling |
| `ascendOne()` | 0 | Zipper | Move up one level |
| `ascend()` | 0 | Zipper | Move to root |
| `reset()` | 0 | Zipper | Return path to root |
| `childCount()` | 0 | Int | Number of children at current position |

### Read Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `getLeaf()` | 0 | Par | Value at current position (Nil if none) |
| `getSubtrie()` | 0 | PathMap | Subtrie rooted at current position |

### Write Methods (write-mode only)

These error if called on a read-mode Zipper.

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `setLeaf(value)` | 1 | Zipper | Set value at current position |
| `setSubtrie(pathmap)` | 1 (PathMap) | Zipper | Replace subtrie at current position |
| `removeLeaf()` | 0 | Zipper | Remove value at current position |
| `removeBranches()` | 0 | Zipper | Remove all children |
| `graft(dest, src)` | 2 (Zipper, Zipper) | Zipper | Copy subtrie from src to dest |
| `joinInto(dest, src)` | 2 (Zipper, Zipper) | Zipper | Merge src into dest |

### Zipper Example

```rho
new stdout(`rho:io:stdout`), stdoutAck(`rho:io:stdoutAck`) in {
  // Create a PathMap
  new pm in {
    pm!({| ["users", "alice", "age"],
           ["users", "alice", "email"],
           ["users", "bob", "age"] |}) |

    for (@pathmap <- pm) {
      // Create a read zipper and navigate
      new ack in {
        stdoutAck!(
          pathmap
            .readZipper()
            .descendTo(["users", "alice"])
            .childCount(),
          *ack
        ) |
        for (_ <- ack) {
          // Create a write zipper and modify
          stdout!(
            pathmap
              .writeZipper()
              .descendTo(["users", "charlie"])
              .setLeaf("new user")
              .ascend()
              .getSubtrie()
          )
        }
      }
    }
  }
}
```

### Read vs Write Mode

```rho
// This works:
pathmap.writeZipper().setLeaf("value")

// This errors at runtime:
pathmap.readZipper().setLeaf("value")  // error: write operation on read-mode zipper
```

The mode distinction prevents accidental mutation when you only intend to read.

## When to Use PathMaps vs Maps

| Use Case | PathMap | Map |
|----------|---------|-----|
| Flat key-value | No | Yes |
| Hierarchical/nested data | Yes | Cumbersome |
| Tree navigation | Yes (zippers) | No |
| Set operations on structure | Yes (union, diff, etc.) | Limited |
| Pattern matching keys | No | Yes |
| Dynamic key access | Limited | Yes (`.get()`) |

PathMaps excel at representing tree-structured data (filesystems, org charts, nested configs) where you need structural operations like union, intersection, and subtree extraction.
