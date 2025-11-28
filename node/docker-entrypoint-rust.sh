#!/usr/bin/env sh
set -e

# Execute the node binary with docker profile and any arguments passed to the container
# All dependencies are built into the binary by Cargo, so no library path setup needed
# This matches the Scala entrypoint behavior which automatically sets --profile=docker
exec /opt/docker/bin/node --profile=docker "$@"
