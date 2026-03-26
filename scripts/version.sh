#!/usr/bin/env bash
# Shared version discovery and bumping logic for f1r3node (Scala)
#
# Usage (sourced by other scripts):
#   source scripts/version.sh
#   get_current_version    # sets CURRENT_VERSION
#   bump_version minor     # sets NEXT_VERSION, TAG_NAME
#
# Usage (standalone):
#   ./scripts/version.sh [current|next [major|minor|patch]]

set -euo pipefail

TAG_PREFIX="scala-v"
TAG_PATTERN="scala-v*"

# Discover the current version from the latest matching tag
get_current_version() {
    local latest_tag
    latest_tag=$(git tag -l "$TAG_PATTERN" --sort=-v:refname | head -1)
    if [ -z "$latest_tag" ]; then
        CURRENT_VERSION="0.1.0"
        return
    fi
    CURRENT_VERSION="${latest_tag#$TAG_PREFIX}"

    # Validate semver format
    if ! [[ "$CURRENT_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "ERROR: invalid version format in tag '$latest_tag': $CURRENT_VERSION" >&2
        exit 1
    fi
}

# Bump version based on type (major|minor|patch)
# Sets NEXT_VERSION and TAG_NAME
bump_version() {
    local bump_type="${1:-minor}"

    get_current_version

    local major minor patch
    major=$(echo "$CURRENT_VERSION" | cut -d. -f1)
    minor=$(echo "$CURRENT_VERSION" | cut -d. -f2)
    patch=$(echo "$CURRENT_VERSION" | cut -d. -f3)

    case "$bump_type" in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            ;;
        patch)
            patch=$((patch + 1))
            ;;
        *)
            echo "ERROR: invalid bump type '$bump_type' (use major|minor|patch)" >&2
            exit 1
            ;;
    esac

    NEXT_VERSION="${major}.${minor}.${patch}"
    TAG_NAME="${TAG_PREFIX}${NEXT_VERSION}"
}

# Standalone usage
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    CMD="${1:-current}"
    case "$CMD" in
        current)
            get_current_version
            echo "$CURRENT_VERSION"
            ;;
        next)
            bump_version "${2:-minor}"
            echo "$NEXT_VERSION (tag: $TAG_NAME)"
            ;;
        *)
            echo "Usage: $0 [current|next [major|minor|patch]]"
            exit 1
            ;;
    esac
fi
