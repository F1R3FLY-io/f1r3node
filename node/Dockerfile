# Multi-stage Dockerfile for Rust F1r3fly Node
# Equivalent to build.sbt Docker configuration

# ============================================================================
# Stage 1: Builder - Build node binary (Cargo handles all dependencies)
# ============================================================================
FROM rust:bookworm AS builder

# Declare build arguments at the top to ensure consistent caching
ARG TARGETARCH
ARG BUILDPLATFORM

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libclang-dev \
    clang \
    build-essential \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy workspace files first for better caching
COPY Cargo.toml Cargo.lock ./

# Create directory structure and copy only Cargo.toml files from all workspace members
# This allows us to cache dependencies separately from source code
RUN mkdir -p models crypto rholang rholang/src/main/tree_sitter shared casper comm rspace++ rspace++/libs/rspace_rhotypes block-storage graphz node node-cli

# Copy all Cargo.toml files - keeping them as separate commands for clarity
# Docker will cache each layer independently, but all must be unchanged for cache to work
COPY models/Cargo.toml ./models/
COPY crypto/Cargo.toml ./crypto/
COPY rholang/Cargo.toml ./rholang/
COPY rholang/src/main/tree_sitter/Cargo.toml ./rholang/src/main/tree_sitter/
COPY shared/Cargo.toml ./shared/
COPY casper/Cargo.toml ./casper/
COPY comm/Cargo.toml ./comm/
COPY rspace++/Cargo.toml ./rspace++/
COPY block-storage/Cargo.toml ./block-storage/
COPY graphz/Cargo.toml ./graphz/
COPY node/Cargo.toml ./node/
COPY node-cli/Cargo.toml ./node-cli/
COPY rspace++/libs/rspace_rhotypes/Cargo.toml ./rspace++/libs/rspace_rhotypes/

# Create dummy source files so cargo can parse the workspace and fetch dependencies
# Library crates need src/lib.rs, binary crates need src/main.rs
RUN mkdir -p models/src crypto/src rholang/src rholang/src/main/tree_sitter/src shared/src casper/src comm/src \
    rspace++/src block-storage/src graphz/src node/src node-cli/src \
    rspace++/libs/rspace_rhotypes/src && \
    echo "// Dummy file for dependency caching" > models/src/lib.rs && \
    echo "// Dummy file for dependency caching" > crypto/src/lib.rs && \
    echo "// Dummy file for dependency caching" > rholang/src/lib.rs && \
    echo "// Dummy file for dependency caching" > rholang/src/rholang_cli.rs && \
    echo "// Dummy file for dependency caching" > rholang/src/main/tree_sitter/src/lib.rs && \
    echo "// Dummy file for dependency caching" > shared/src/lib.rs && \
    echo "// Dummy file for dependency caching" > casper/src/lib.rs && \
    echo "// Dummy file for dependency caching" > comm/src/lib.rs && \
    echo "// Dummy file for dependency caching" > rspace++/src/lib.rs && \
    echo "// Dummy file for dependency caching" > block-storage/src/lib.rs && \
    echo "// Dummy file for dependency caching" > graphz/src/lib.rs && \
    echo "// Dummy file for dependency caching" > node/src/lib.rs && \
    echo "fn main() {}" > node/src/main.rs && \
    echo "fn main() {}" > node-cli/src/main.rs && \
    echo "// Dummy file for dependency caching" > rspace++/libs/rspace_rhotypes/src/lib.rs

# Build dependencies for target architecture to cache compiled dependencies
# This layer will be cached unless Cargo.toml or Cargo.lock files change
# TARGETARCH is already declared at the top of the stage
# First fetch all dependencies (this always succeeds and caches them)
# Then try to build libraries to compile dependencies
RUN case "$TARGETARCH" in \
        amd64) LINUX_TARGET="x86_64-unknown-linux-gnu" ;; \
        arm64) LINUX_TARGET="aarch64-unknown-linux-gnu" ;; \
        *) LINUX_TARGET="x86_64-unknown-linux-gnu" ;; \
    esac && \
    rustup target add $LINUX_TARGET && \
    cargo fetch --target $LINUX_TARGET && \
    cargo build --release --target $LINUX_TARGET --lib || \
    (cargo build --release --target $LINUX_TARGET -p models || true)

# Now copy all source code (this layer invalidates when source changes)
# Copy in dependency order to maximize cache hits - base dependencies first
COPY shared/ ./shared/
COPY crypto/ ./crypto/
COPY models/ ./models/
COPY graphz/ ./graphz/
COPY block-storage/ ./block-storage/
COPY rspace++/ ./rspace++/
COPY casper/ ./casper/
COPY comm/ ./comm/
COPY rholang/ ./rholang/
COPY node-cli/ ./node-cli/
COPY node/ ./node/

# Build the node binary (Cargo will automatically build all dependencies)
RUN case "$TARGETARCH" in \
        amd64) LINUX_TARGET="x86_64-unknown-linux-gnu" ;; \
        arm64) LINUX_TARGET="aarch64-unknown-linux-gnu" ;; \
        *) echo "Unsupported architecture: $TARGETARCH" >&2 && exit 1 ;; \
    esac && \
    rustup target add $LINUX_TARGET && \
    cargo build --release --target $LINUX_TARGET -p node && \
    if [ ! -f "target/$LINUX_TARGET/release/node" ]; then \
        echo "Error: Binary not found at target/$LINUX_TARGET/release/node" >&2 && \
        exit 1; \
    fi && \
    rm -f /build/node_binary && \
    cp target/$LINUX_TARGET/release/node /build/node_binary && \
    chmod +x /build/node_binary && \
    test -f /build/node_binary || (echo "ERROR: node_binary is not a file!" >&2 && exit 1)

# ============================================================================
# Stage 2: Runtime - Minimal image with only runtime dependencies
# ============================================================================
FROM debian:12-slim

# Install runtime dependencies
# Note: grpcurl and jq for healthcheck (matching build.sbt healthcheck)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libc6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install grpcurl and jq for healthcheck (matching build.sbt)
# Note: These are installed in the runtime stage, architecture will be detected at runtime
RUN GRPCURL_VERSION="1.8.9" && \
    ARCH=$(dpkg --print-architecture) && \
    case "$ARCH" in \
        amd64) GRPCURL_ARCH="x86_64" ;; \
        arm64) GRPCURL_ARCH="arm64" ;; \
        *) GRPCURL_ARCH="x86_64" ;; \
    esac && \
    curl -L "https://github.com/fullstorydev/grpcurl/releases/download/v${GRPCURL_VERSION}/grpcurl_${GRPCURL_VERSION}_linux_${GRPCURL_ARCH}.tar.gz" -o /tmp/grpcurl.tar.gz && \
    tar -xzf /tmp/grpcurl.tar.gz -C /tmp && \
    find /tmp -name grpcurl -type f -executable -exec mv {} /usr/local/bin/ \; && \
    chmod +x /usr/local/bin/grpcurl && \
    rm -rf /tmp/grpcurl* || true && \
    ARCH=$(dpkg --print-architecture) && \
    case "$ARCH" in \
        amd64) JQ_ARCH="linux64" ;; \
        arm64) JQ_ARCH="linuxarm64" ;; \
        *) JQ_ARCH="linux64" ;; \
    esac && \
    curl -L "https://github.com/jqlang/jq/releases/download/jq-1.7.1/jq-${JQ_ARCH}" -o /usr/local/bin/jq && \
    chmod +x /usr/local/bin/jq || true

# Use existing daemon user (matching build.sbt: daemonUser in Docker := "daemon")
# Note: daemon user already exists in debian:12-slim, so we just ensure it has the right UID
# RUN id -u daemon >/dev/null 2>&1 || useradd -r -s /bin/false -u 1 daemon

# Set up directory structure (matching build.sbt defaultLinuxInstallLocation)
WORKDIR /opt/docker

# Create bin directory first
RUN mkdir -p /opt/docker/bin

# Copy node binary (using node_binary from builder to avoid conflict with node/ directory)
# Verify it's a file before copying
COPY --from=builder --chmod=755 /build/node_binary /opt/docker/bin/node

# Copy node resources (defaults.conf, kamon.conf, logback.xml, etc.)
# The node executable expects these at node/src/main/resources relative to current directory
COPY --from=builder /build/node/src/main/resources ./node/src/main/resources

# Copy casper resources (Registry.rho, ListOps.rho, etc.)
# The node executable needs these at runtime to create genesis blocks
COPY --from=builder /build/casper/src/main/resources ./casper/src/main/resources

# Copy rholang examples (matching build.sbt: directory((baseDirectory in rholang).value / "examples"))
COPY --from=builder /build/rholang/examples ./examples

# Create entrypoint script for Rust (adapted from docker-entrypoint.sh)
# Copy the Rust-specific entrypoint script from node directory
# Note: This Dockerfile should be built from workspace root
COPY node/docker-entrypoint-rust.sh /opt/docker/bin/docker-entrypoint.sh

# Set permissions (matching build.sbt: withUser "daemon" withGroup "daemon")
RUN if [ ! -f /opt/docker/bin/node ]; then \
        echo "ERROR: /opt/docker/bin/node is not a file!" >&2 && \
        exit 1; \
    fi && \
    chown -R daemon:daemon /opt/docker && \
    chmod +x /opt/docker/bin/node && \
    chmod +x /opt/docker/bin/docker-entrypoint.sh

# Expose ports (matching build.sbt: dockerExposedPorts)
EXPOSE 40400 40401 40402 40403 40404

# Healthcheck (matching build.sbt healthcheck, adapted for Rust)
# Note: The original uses grpcurl and curl with jq
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD grpcurl -plaintext 127.0.0.1:40401 casper.v1.DeployService.status | jq -e && \
        curl -s 127.0.0.1:40403/status | jq -e || exit 1

# Labels (matching build.sbt)
LABEL maintainer="F1r3fly.io LCA https://f1r3fly.io/"
LABEL version="0.1.0"

# Switch to daemon user (matching build.sbt: daemonUser in Docker := "daemon")
USER daemon

# Entrypoint (matching build.sbt: dockerEntrypoint)
ENTRYPOINT ["/opt/docker/bin/docker-entrypoint.sh"]

# Default command (matching build.sbt: ExecCmd("CMD", "run"))
CMD ["run"]

