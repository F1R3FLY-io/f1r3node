#!/bin/bash
# Docker commands for Rust node (equivalent to build.sbt Docker operations)
# Usage: source node/docker-commands.sh or run individual commands

set -e

# Configuration (matching build.sbt)
DOCKER_REPOSITORY="f1r3flyindustries"
IMAGE_NAME="f1r3fly-rust-node"
FULL_IMAGE_NAME="${DOCKER_REPOSITORY}/${IMAGE_NAME}"
VERSION="${VERSION:-latest}"

# Build the Docker image
# Equivalent to: sbt "node/Docker/publishLocal"
docker_build() {
    local is_cross_build="${MULTI_ARCH:-}"
    
    if [ -n "$is_cross_build" ]; then
        echo "Building multi-architecture image (amd64, arm64)..."
        docker buildx build \
            --platform linux/amd64,linux/arm64 \
            --file node/Dockerfile \
            --tag "${FULL_IMAGE_NAME}:${VERSION}" \
            --tag "${FULL_IMAGE_NAME}:latest" \
            --push \
            .
    else
        echo "Building single-architecture image..."
        docker build \
            --file node/Dockerfile \
            --tag "${FULL_IMAGE_NAME}:${VERSION}" \
            --tag "${FULL_IMAGE_NAME}:latest" \
            .
    fi
}

# Build for local use (publishLocal equivalent)
docker_build_local() {
    echo "Building Docker image for local use..."
    docker build \
        --file node/Dockerfile \
        --tag "${IMAGE_NAME}:latest" \
        --tag "${FULL_IMAGE_NAME}:latest" \
        .
}

# Run the container (equivalent to docker run)
docker_run() {
    docker run -it --rm \
        -p 40400-40404:40400-40404 \
        "${FULL_IMAGE_NAME}:latest" \
        "$@"
}

# Run with volumes (matching docker-compose usage)
docker_run_with_volumes() {
    local data_dir="${1:-./data}"
    local conf_dir="${2:-./conf}"
    shift 2 2>/dev/null || true  # Remove first 2 args if they exist
    
    docker run -it --rm \
        -p 40400-40404:40400-40404 \
        -v "${data_dir}:/var/lib/rnode" \
        -v "${conf_dir}:/var/lib/rnode/conf" \
        "${FULL_IMAGE_NAME}:latest" \
        "$@"
}

# Publish to registry (equivalent to sbt "node/Docker/publish")
docker_publish() {
    local tag="${1:-latest}"
    
    echo "Publishing ${FULL_IMAGE_NAME}:${tag}..."
    docker push "${FULL_IMAGE_NAME}:${tag}"
}

# Publish with DRONE build number tag (matching build.sbt behavior)
docker_publish_drone() {
    local drone_build_num="${DRONE_BUILD_NUMBER:-}"
    
    if [ -z "$drone_build_num" ]; then
        echo "Error: DRONE_BUILD_NUMBER environment variable not set"
        exit 1
    fi
    
    local drone_tag="DRONE-${drone_build_num}"
    echo "Publishing with DRONE tag: ${drone_tag}..."
    
    docker tag "${FULL_IMAGE_NAME}:latest" "${FULL_IMAGE_NAME}:${drone_tag}"
    docker push "${FULL_IMAGE_NAME}:${drone_tag}"
    docker push "${FULL_IMAGE_NAME}:latest"  # Also update latest if not in DRONE
}

usage() {
    cat << EOF
Docker commands for Rust F1r3fly Node

Usage (direct execution):
    ./node/docker-commands.sh <command> [args...]
    
    # Build locally
    ./node/docker-commands.sh build-local
    
    # Build for production (single arch)
    ./node/docker-commands.sh build
    
    # Build multi-architecture (requires buildx)
    MULTI_ARCH=1 ./node/docker-commands.sh build
    
    # Run container
    ./node/docker-commands.sh run run --host=localhost
    
    # Run with volumes
    ./node/docker-commands.sh run-with-volumes ./data ./conf run
    
    # Publish to registry
    ./node/docker-commands.sh publish
    
    # Publish with DRONE tag
    DRONE_BUILD_NUMBER=123 ./node/docker-commands.sh publish-drone

Usage (source for interactive use):
    source node/docker-commands.sh
    
    # Then use functions directly:
    docker_build_local
    docker_build
    docker_run run
    docker_run_with_volumes ./data ./conf run
    docker_publish
    DRONE_BUILD_NUMBER=123 docker_publish_drone

Environment variables:
    MULTI_ARCH          - Set to enable multi-architecture build
    DRONE_BUILD_NUMBER  - Build number for DRONE tag
    VERSION             - Image version tag (default: latest)

Examples:
    # Build and run locally (direct execution)
    ./node/docker-commands.sh build-local
    ./node/docker-commands.sh run run
    
    # Build and publish multi-arch
    MULTI_ARCH=1 ./node/docker-commands.sh build
    
    # Run with custom command
    ./node/docker-commands.sh run run --host=0.0.0.0 --allow-private-addresses
EOF
}

# If script is executed directly, parse command and execute
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    if [ $# -eq 0 ]; then
        usage
        exit 0
    fi
    
    COMMAND="$1"
    shift  # Remove command from arguments
    
    case "$COMMAND" in
        build-local)
            docker_build_local
            ;;
        build)
            docker_build
            ;;
        run)
            docker_run "$@"
            ;;
        run-with-volumes)
            docker_run_with_volumes "$@"
            ;;
        publish)
            docker_publish "$@"
            ;;
        publish-drone)
            docker_publish_drone
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            echo "Error: Unknown command '$COMMAND'" >&2
            echo "" >&2
            usage >&2
            exit 1
            ;;
    esac
fi

