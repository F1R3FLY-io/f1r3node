#!/bin/bash

set -e

AMD64_TARGET="x86_64-unknown-linux-gnu"
AARCH64_TARGET="aarch64-unknown-linux-gnu"

RSPACE_PLUS_PLUS_AMD64_BUILD_ARTIFACTS_PATH="target/$AMD64_TARGET/release"
RSPACE_PLUS_PLUS_AARCH64_BUILD_ARTIFACTS_PATH="target/$AARCH64_TARGET/release"

RHOLANG_AMD64_BUILD_ARTIFACTS_PATH="target/$AMD64_TARGET/release"
RHOLANG_AARCH64_BUILD_ARTIFACTS_PATH="target/$AARCH64_TARGET/release"

RUST_LIBRARIES_AMD64_RELEASE_PATH="rust_libraries/docker/release/amd64"
RUST_LIBRARIES_AARCH64_RELEASE_PATH="rust_libraries/docker/release/aarch64"

set -e

mkdir -p "$RUST_LIBRARIES_AMD64_RELEASE_PATH"
mkdir -p "$RUST_LIBRARIES_AARCH64_RELEASE_PATH"

cross build --release --target $AMD64_TARGET -p rspace_plus_plus_rhotypes
cross build --release --target $AARCH64_TARGET -p rspace_plus_plus_rhotypes

cp -r "$RSPACE_PLUS_PLUS_AMD64_BUILD_ARTIFACTS_PATH"/librspace_plus_plus_rhotypes.* "./$RUST_LIBRARIES_AMD64_RELEASE_PATH"/
cp -r "$RSPACE_PLUS_PLUS_AARCH64_BUILD_ARTIFACTS_PATH"/librspace_plus_plus_rhotypes.* "./$RUST_LIBRARIES_AARCH64_RELEASE_PATH"/

cross build --release --target $AMD64_TARGET -p rholang
cross build --release --target $AARCH64_TARGET -p rholang

cp -r "$RHOLANG_AMD64_BUILD_ARTIFACTS_PATH"/librholang.* "./$RUST_LIBRARIES_AMD64_RELEASE_PATH"/
cp -r "$RHOLANG_AARCH64_BUILD_ARTIFACTS_PATH"/librholang.* "./$RUST_LIBRARIES_AARCH64_RELEASE_PATH"/
