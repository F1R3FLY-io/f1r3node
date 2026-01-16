# Extending Reified RSpaces

This document describes how to extend the reified RSpaces system with custom implementations.

## Architecture Overview

The reified RSpaces system uses a 6-layer architecture:

1. **Inner Collections** - Data/continuation storage at channels
2. **Outer Storage (ChannelStore)** - Channel indexing structures
3. **Space Agent** - Core space operations (produce, consume, gensym)
4. **Checkpointing** - State management
5. **Generic RSpace** - Parameterized implementation combining storage + matching
6. **Space Factories** - Space construction from configuration

## Implementing Custom Collection Types

### DataCollection Trait

To implement a custom data collection, implement the `DataCollection<A>` trait:

```rust
use rholang::spaces::collections::{DataCollection, SemanticEq};
use rholang::spaces::errors::SpaceError;

pub struct MyCustomCollection<A> {
    items: Vec<A>,
}

impl<A: Clone + SemanticEq + Send + Sync> DataCollection<A> for MyCustomCollection<A> {
    fn add(&mut self, data: A) {
        self.items.push(data);
    }

    fn remove(&mut self, data: &A) -> Option<A> {
        if let Some(pos) = self.items.iter().position(|x| x.semantic_eq(data)) {
            Some(self.items.remove(pos))
        } else {
            None
        }
    }

    fn find_match<P, M>(&self, pattern: &P, matcher: &M) -> Option<&A>
    where
        M: Match<P, A>,
    {
        self.items.iter().find(|data| matcher.matches(pattern, data))
    }

    fn iter(&self) -> impl Iterator<Item = &A> {
        self.items.iter()
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
```

### ContinuationCollection Trait

For custom continuation storage:

```rust
use rholang::spaces::collections::ContinuationCollection;

impl<P, K> ContinuationCollection<P, K> for MyCustomCollection<(Vec<P>, K)>
where
    P: Clone + Send + Sync,
    K: Clone + Send + Sync,
{
    fn add(&mut self, patterns: Vec<P>, continuation: K) {
        self.items.push((patterns, continuation));
    }

    fn find_fireable<A, M>(
        &self,
        data_per_channel: &[Vec<&A>],
        matcher: &M,
    ) -> Option<(Vec<P>, K, Vec<usize>)>
    where
        M: Match<P, A>,
    {
        // Find a continuation whose patterns all match available data
        for (patterns, continuation) in &self.items {
            if let Some(indices) = find_matching_data(patterns, data_per_channel, matcher) {
                return Some((patterns.clone(), continuation.clone(), indices));
            }
        }
        None
    }

    // ... other required methods
}
```

## Implementing Custom Channel Stores

### ChannelStore Trait

To create a custom channel indexing strategy:

```rust
use rholang::spaces::channel_store::{ChannelStore, DataCollection, ContinuationCollection};
use rholang::spaces::types::SpaceId;
use rholang::spaces::errors::SpaceError;

pub struct MyCustomStore<C, P, A, K, DC, CC> {
    // Your storage structure
}

impl<C, P, A, K, DC, CC> ChannelStore for MyCustomStore<C, P, A, K, DC, CC>
where
    C: Clone + Eq + Hash + Send + Sync,
    P: Clone + Send + Sync,
    A: Clone + Send + Sync + Debug + 'static,
    K: Clone + Send + Sync,
    DC: DataCollection<A> + Default + Clone + Send + Sync + 'static,
    CC: ContinuationCollection<P, K> + Default + Clone + Send + Sync,
{
    type Channel = C;
    type Pattern = P;
    type Data = A;
    type Continuation = K;
    type DataColl = DC;
    type ContColl = CC;

    fn get_or_create_data_collection(&mut self, channel: &C) -> &mut DC {
        // Implementation
    }

    fn gensym(&mut self, space_id: &SpaceId) -> Result<C, SpaceError> {
        // Generate unique channel name
    }

    // ... other required methods
}
```

### Prefix Semantics

If your store supports hierarchical paths (like PathMap), implement:

```rust
impl<...> ChannelStore for MyPathStore<...> {
    fn supports_prefix_semantics(&self) -> bool {
        true  // Enable prefix matching
    }

    fn get_channels_with_prefix(&self, prefix: &Self::Channel) -> Vec<&Self::Channel> {
        // Return all channels that start with prefix
    }

    fn get_data_with_prefix(&self, prefix: &Self::Channel)
        -> Vec<(SuffixKey, &Self::DataColl)> {
        // Return data at prefix and all child paths with suffix keys
    }
}
```

## Implementing Custom Vector Backends

### VectorBackendFactory Trait

For custom vector similarity backends:

```rust
use rholang::spaces::vectordb::registry::{
    BackendConfig, VectorBackendDyn, VectorBackendFactory, ResolvedArg,
};

pub struct MyVectorBackendFactory;

impl VectorBackendFactory for MyVectorBackendFactory {
    fn name(&self) -> &str {
        "my-backend"
    }

    fn aliases(&self) -> Vec<&str> {
        vec!["mb", "my"]
    }

    fn create(&self, config: &BackendConfig) -> Result<Box<dyn VectorBackendDyn>, SpaceError> {
        let dimension = config.get_int("dimension")
            .unwrap_or(128) as usize;

        Ok(Box::new(MyVectorBackend::new(dimension)))
    }

    fn expected_args(&self) -> Vec<(&str, &str)> {
        vec![
            ("dimension", "int"),
            ("metric", "string"),
        ]
    }
}
```

### Registering Your Backend

```rust
use rholang::spaces::vectordb::registry::BackendRegistry;

// Register at startup
let mut registry = BackendRegistry::with_defaults();
registry.register(Box::new(MyVectorBackendFactory));
```

## Space Factory Registration

### Custom SpaceFactory

To create spaces from custom configurations:

```rust
use rholang::spaces::factory::{SpaceFactory, FactoryRegistry};
use rholang::spaces::types::SpaceConfig;

pub struct MySpaceFactory;

impl SpaceFactory for MySpaceFactory {
    fn create_space(&self, config: &SpaceConfig)
        -> Result<Box<dyn SpaceAgent<...>>, SpaceError> {
        // Create and return your space implementation
    }

    fn supported_urns(&self) -> Vec<&str> {
        vec!["rho:space:custom:myspace"]
    }
}

// Register with the factory registry
let mut factory_registry = FactoryRegistry::new();
factory_registry.register(Box::new(MySpaceFactory));
```

## URN Conventions

Space URNs follow the pattern: `rho:space:<outer>:<inner>`

### Standard URNs

| URN | Outer Storage | Inner Collection |
|-----|---------------|------------------|
| `rho:space:hashmap:bag` | HashMap | Bag |
| `rho:space:pathmap:queue` | PathMap | Queue |
| `rho:space:array:stack` | Array | Stack |
| `rho:space:vector:set` | Vector | Set |

### Theory-Annotated URNs

For typed tuple spaces: `rho:space:<outer>:<inner>/<theory>`

Example: `rho:space:hashmap:bag/NatTheory`

### Custom URNs

Use the `custom` prefix for your implementations:

- `rho:space:custom:mystore:mybag` - Custom store with custom collection
- `rho:space:vectordb:myvector` - Custom vector backend

## Testing Requirements

### Unit Tests

Test your implementation against the trait contracts:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_collection_add_remove() {
        let mut coll = MyCustomCollection::new();
        coll.add("test".to_string());
        assert_eq!(coll.len(), 1);
        assert_eq!(coll.remove(&"test".to_string()), Some("test".to_string()));
        assert!(coll.is_empty());
    }

    #[test]
    fn test_channel_store_gensym_uniqueness() {
        let mut store = MyCustomStore::new();
        let space_id = SpaceId::default_space();
        let ch1 = store.gensym(&space_id).unwrap();
        let ch2 = store.gensym(&space_id).unwrap();
        assert_ne!(ch1, ch2);
    }
}
```

### Integration Tests

Test with `GenericRSpace`:

```rust
#[test]
fn test_produce_consume_integration() {
    let store = MyCustomStore::new();
    let matcher = ExactMatch::new();
    let mut space = GenericRSpace::new(
        store,
        matcher,
        SpaceId::default_space(),
        SpaceQualifier::Default,
    );

    // Test produce/consume cycle
    let channel = space.gensym().unwrap();
    space.produce(&channel, data, false).unwrap();
    let result = space.consume(&[channel], &[pattern], continuation, false);
    assert!(result.is_some());
}
```

### Compatibility Checklist

Before deploying a custom implementation:

- [ ] All trait methods implemented correctly
- [ ] Unit tests pass for all operations
- [ ] Integration test with GenericRSpace passes
- [ ] Checkpoint/restore works correctly (if applicable)
- [ ] Thread safety verified (Send + Sync bounds satisfied)
- [ ] Memory usage within acceptable bounds
- [ ] Performance benchmarks meet requirements

## Example: Complete Custom Space

```rust
// 1. Define your collection
pub struct LRUDataCollection<A> { /* ... */ }
impl<A> DataCollection<A> for LRUDataCollection<A> { /* ... */ }

// 2. Define your store
pub struct LRUChannelStore<C, P, A, K> { /* ... */ }
impl<...> ChannelStore for LRUChannelStore<...> { /* ... */ }

// 3. Create type alias for convenience
pub type LRURSpace<M> = GenericRSpace<LRUChannelStore<Par, Par, ListParWithRandom, Par>, M>;

// 4. Create factory
pub struct LRUSpaceFactory;
impl SpaceFactory for LRUSpaceFactory {
    fn create_space(&self, config: &SpaceConfig) -> Result<Box<dyn SpaceAgent<...>>, SpaceError> {
        let store = LRUChannelStore::new(config.max_size.unwrap_or(1000));
        let matcher = ExactMatch::new();
        Ok(Box::new(GenericRSpace::new(store, matcher, config.space_id.clone(), config.qualifier)))
    }

    fn supported_urns(&self) -> Vec<&str> {
        vec!["rho:space:lru:bag", "rho:space:lru"]
    }
}

// 5. Register
registry.register(Box::new(LRUSpaceFactory));
```

## See Also

- `rholang/src/rust/interpreter/spaces/collections.rs` - Collection trait definitions
- `rholang/src/rust/interpreter/spaces/channel_store/` - Channel store implementations
- `rholang/src/rust/interpreter/spaces/vectordb/` - Vector backend examples
- `rholang/src/rust/interpreter/spaces/factory.rs` - Factory registration
