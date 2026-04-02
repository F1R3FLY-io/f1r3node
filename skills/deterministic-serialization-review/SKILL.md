# Deterministic Serialization Review

Use this skill when adding/changing persisted execution state, IDs, or hashes.

## Objective

Guarantee serialization is canonical, versioned, and replay-stable.

## Checklist

1. Confirm serialization format is explicit and stable.
2. Introduce version fields for persisted state.
3. Verify ordering is canonical before encode/hash operations.
4. Add round-trip tests for encode/decode.
5. Add change-detection tests for state-root/hash transitions.
6. Ensure no host-dependent values (timestamps, locale, random order) leak into serialization.

## Required verification

```bash
cargo test -p models --test models_tests sorted_par_hash_set_test
cargo test -p models --test models_tests sorted_par_map_test
```

## Completion report format

- Serialization contract changes
- Backward/forward compatibility notes
- Determinism risks
