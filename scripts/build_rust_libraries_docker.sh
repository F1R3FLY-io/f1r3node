#!/bin/bash

set -e

echo "=== Firefly Rust Libraries Docker Build Script ==="
echo "Starting build process for cross-platform Rust libraries..."

AMD64_TARGET="x86_64-unknown-linux-gnu"
AARCH64_TARGET="aarch64-unknown-linux-gnu"

echo "Target architectures:"
echo "  - AMD64: $AMD64_TARGET"
echo "  - AARCH64: $AARCH64_TARGET"

RSPACE_PLUS_PLUS_AMD64_BUILD_ARTIFACTS_PATH="target/$AMD64_TARGET/release"
RSPACE_PLUS_PLUS_AARCH64_BUILD_ARTIFACTS_PATH="target/$AARCH64_TARGET/release"

RHOLANG_AMD64_BUILD_ARTIFACTS_PATH="target/$AMD64_TARGET/release"
RHOLANG_AARCH64_BUILD_ARTIFACTS_PATH="target/$AARCH64_TARGET/release"

RUST_LIBRARIES_AMD64_RELEASE_PATH="rust_libraries/docker/release/amd64"
RUST_LIBRARIES_AARCH64_RELEASE_PATH="rust_libraries/docker/release/aarch64"

echo ""
echo "Creating output directories..."

mkdir -p "$RUST_LIBRARIES_AMD64_RELEASE_PATH"
echo "  ✓ Created: $RUST_LIBRARIES_AMD64_RELEASE_PATH"

mkdir -p "$RUST_LIBRARIES_AARCH64_RELEASE_PATH"
echo "  ✓ Created: $RUST_LIBRARIES_AARCH64_RELEASE_PATH"

echo ""
echo "=== Building rspace_plus_plus_rhotypes ==="
echo "Building for AMD64 ($AMD64_TARGET)..."
cross build --release --target $AMD64_TARGET -p rspace_plus_plus_rhotypes
echo "  ✓ AMD64 build completed"

echo "Building for AARCH64 ($AARCH64_TARGET)..."
cross build --release --target $AARCH64_TARGET -p rspace_plus_plus_rhotypes
echo "  ✓ AARCH64 build completed"

echo ""
echo "=== Copying rspace_plus_plus_rhotypes artifacts ==="
echo "Copying AMD64 artifacts from $RSPACE_PLUS_PLUS_AMD64_BUILD_ARTIFACTS_PATH to $RUST_LIBRARIES_AMD64_RELEASE_PATH"
cp -r "$RSPACE_PLUS_PLUS_AMD64_BUILD_ARTIFACTS_PATH"/librspace_plus_plus_rhotypes.* "./$RUST_LIBRARIES_AMD64_RELEASE_PATH"/
echo "  ✓ AMD64 artifacts copied"

echo "Copying AARCH64 artifacts from $RSPACE_PLUS_PLUS_AARCH64_BUILD_ARTIFACTS_PATH to $RUST_LIBRARIES_AARCH64_RELEASE_PATH"
cp -r "$RSPACE_PLUS_PLUS_AARCH64_BUILD_ARTIFACTS_PATH"/librspace_plus_plus_rhotypes.* "./$RUST_LIBRARIES_AARCH64_RELEASE_PATH"/
echo "  ✓ AARCH64 artifacts copied"

echo ""
echo "=== Building rholang ==="
echo "Building for AMD64 ($AMD64_TARGET)..."
cross build --release --target $AMD64_TARGET -p rholang
echo "  ✓ AMD64 build completed"

echo "Building for AARCH64 ($AARCH64_TARGET)..."
cross build --release --target $AARCH64_TARGET -p rholang
echo "  ✓ AARCH64 build completed"

echo ""
echo "=== Copying rholang artifacts ==="
echo "Copying AMD64 artifacts from $RHOLANG_AMD64_BUILD_ARTIFACTS_PATH to $RUST_LIBRARIES_AMD64_RELEASE_PATH"
cp -r "$RHOLANG_AMD64_BUILD_ARTIFACTS_PATH"/librholang.* "./$RUST_LIBRARIES_AMD64_RELEASE_PATH"/
echo "  ✓ AMD64 artifacts copied"

echo "Copying AARCH64 artifacts from $RHOLANG_AARCH64_BUILD_ARTIFACTS_PATH to $RUST_LIBRARIES_AARCH64_RELEASE_PATH"
cp -r "$RHOLANG_AARCH64_BUILD_ARTIFACTS_PATH"/librholang.* "./$RUST_LIBRARIES_AARCH64_RELEASE_PATH"/
echo "  ✓ AARCH64 artifacts copied"

echo ""
echo "=== Build completed successfully! ==="
echo "All Rust libraries have been built and copied to their respective release directories."
