#!/bin/bash

set -e

# Detect host architecture and OS
HOST_ARCH=$(uname -m)
HOST_OS=$(uname -s)

echo "Building Rust libraries for Docker (Linux) on $HOST_OS $HOST_ARCH..."

# Determine Linux target and output directory based on host architecture
case "$HOST_ARCH" in
    x86_64|amd64)
        LINUX_TARGET="x86_64-unknown-linux-gnu"
        OUTPUT_DIR="rust_libraries/docker/release/amd64"
        echo "🖥️  Host: x86_64 → Target: x86_64-unknown-linux-gnu → $OUTPUT_DIR"
        ;;
    aarch64|arm64)
        LINUX_TARGET="aarch64-unknown-linux-gnu"
        OUTPUT_DIR="rust_libraries/docker/release/aarch64"
        echo "🍎 Host: arm64 → Target: aarch64-unknown-linux-gnu → $OUTPUT_DIR"
        ;;
    *)
        echo "❌ Unsupported architecture: $HOST_ARCH"
        exit 1
        ;;
esac

# Create release directory
mkdir -p "./$OUTPUT_DIR"

# Determine build command based on host OS
if [ "$HOST_OS" = "Linux" ]; then
    echo "🐧 Building natively on Linux"
    BUILD_CMD="cargo build --release"
    SOURCE_PATH="./target/release"
else
    echo "📱 Cross-compiling to Linux from $HOST_OS using cross"
    # Check if cross is available
    if ! command -v cross &> /dev/null; then
        echo "❌ 'cross' tool not found. Installing..."
        cargo install cross --git https://github.com/cross-rs/cross 2>/dev/null || {
            echo "❌ Failed to install cross. Please install it manually:"
            echo "   cargo install cross --git https://github.com/cross-rs/cross"
            exit 1
        }
    fi
    # Install target if not already available
    rustup target add $LINUX_TARGET 2>/dev/null || true
    BUILD_CMD="cross build --release --target $LINUX_TARGET"
    SOURCE_PATH="./target/$LINUX_TARGET/release"
fi

# Build rspace++ library
echo "Building rspace++ library..."
cd rspace++/
$BUILD_CMD -p rspace_plus_plus_rhotypes
cp "$SOURCE_PATH/librspace_plus_plus_rhotypes.so" "../$OUTPUT_DIR/"

# Build rholang library  
echo "Building rholang library..."
cd ../rholang/
$BUILD_CMD -p rholang
cp "$SOURCE_PATH/librholang.so" "../$OUTPUT_DIR/"

cd ..
echo "✅ Native Rust libraries built successfully for Docker ($LINUX_TARGET)"
echo "📁 Libraries available in: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR"/*.so
