#!/usr/bin/env sh

# Detect architecture
arch=$(uname -m)

# Set JNA library path based on architecture
echo "Using Rust libraries for $arch"
case "$arch" in
  x86_64|amd64)
    JNA_PATH="-Djna.library.path=/opt/docker/rust_libraries/release/amd64/"
    ;;
  aarch64|arm64)
    JNA_PATH="-Djna.library.path=/opt/docker/rust_libraries/release/aarch64/"
    ;;
  *)
    echo "Unsupported architecture: $arch" >&2
    exit 1
    ;;
esac

# Export JAVA_OPTS with the determined path and any existing opts
export JAVA_OPTS="$JNA_PATH $JAVA_OPTS"

# Execute the original application script, passing along the original default arguments
# plus any additional arguments passed to the container ("$@")
exec /opt/docker/bin/rnode --profile=docker -XX:ErrorFile=/var/lib/rnode/hs_err_pid%p.log "$@" 