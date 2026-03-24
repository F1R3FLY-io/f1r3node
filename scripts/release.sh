#!/usr/bin/env bash
# Manual release script for f1r3node (Rust)
#
# Usage: ./scripts/release.sh [major|minor|patch]
#   Default bump type: minor
#
# This is an escape hatch for when you need a non-minor bump (e.g., major
# or patch). For normal releases, merging to rust/main triggers the CI
# workflow which auto-bumps minor.

set -euo pipefail

BUMP_TYPE="${1:-minor}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_DIR"

# Find the latest rust-v* tag
LATEST_TAG=$(git tag -l 'rust-v*' --sort=-v:refname | head -1)
if [ -z "$LATEST_TAG" ]; then
    echo "No existing rust-v* tag found, starting at 0.2.0"
    CURRENT="0.1.0"
else
    CURRENT="${LATEST_TAG#rust-v}"
fi

MAJOR=$(echo "$CURRENT" | cut -d. -f1)
MINOR=$(echo "$CURRENT" | cut -d. -f2)
PATCH=$(echo "$CURRENT" | cut -d. -f3)

case "$BUMP_TYPE" in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + 1))
        ;;
    *)
        echo "Usage: $0 [major|minor|patch]"
        exit 1
        ;;
esac

NEXT_VERSION="${MAJOR}.${MINOR}.${PATCH}"
TAG_NAME="rust-v${NEXT_VERSION}"

echo "Current: ${CURRENT} -> Next: ${NEXT_VERSION} (${TAG_NAME})"
echo ""

# Update node/Cargo.toml version
sed -i "0,/^version = \".*\"/s//version = \"${NEXT_VERSION}\"/" node/Cargo.toml
echo "Updated node/Cargo.toml to ${NEXT_VERSION}"

# Update Dockerfile LABEL
sed -i "s/^LABEL version=\".*\"/LABEL version=\"${NEXT_VERSION}\"/" node/Dockerfile
echo "Updated node/Dockerfile LABEL to ${NEXT_VERSION}"

# Update Cargo.lock
cargo generate-lockfile 2>/dev/null || true
echo "Updated Cargo.lock"

# Generate CHANGELOG if git-cliff is available
if command -v git-cliff &>/dev/null; then
    git-cliff --config cliff.toml --tag "$TAG_NAME" -o CHANGELOG.md
    echo "Generated CHANGELOG.md"
else
    echo "WARNING: git-cliff not found, skipping CHANGELOG generation"
    echo "Install: cargo install git-cliff"
fi

# Commit and tag
git add node/Cargo.toml node/Dockerfile Cargo.lock CHANGELOG.md
git commit -m "chore(release): rust v${NEXT_VERSION}"
git tag -a "$TAG_NAME" -m "Release rust v${NEXT_VERSION}"

echo ""
echo "Release ${TAG_NAME} created."
echo ""
echo "To publish:"
echo "  git push origin $(git branch --show-current) --follow-tags"
