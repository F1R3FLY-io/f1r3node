#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "==> [1/4] Formatting check"
# Use scoped formatting checks for runtime-critical serialization crates.
cargo fmt -p shared -p models -- --check

echo "==> [2/4] Lint check"
# Keep lint scope stable on baseline branch health.
cargo clippy -p shared --lib

echo "==> [3/4] Unit tests"
cargo test -p models --lib
cargo test -p shared --lib

echo "==> [4/4] Deterministic serialization tests"
# Deterministic ordering/serialization focused checks.
cargo test -p models --test models_tests sorted_par_hash_set_test
cargo test -p models --test models_tests sorted_par_map_test
cargo test -p models --test models_tests par_sort_matcher_test

echo "Runtime verification completed successfully."
