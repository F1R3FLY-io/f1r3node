#!/usr/bin/env bash
# Manual release script for f1r3node (Scala)
#
# Usage: ./scripts/release.sh [major|minor|patch]
#   Default bump type: minor
#
# This is an escape hatch for when you need a non-minor bump (e.g., major
# or patch). For normal releases, merging to main triggers the CI workflow
# which auto-bumps minor.

set -euo pipefail

BUMP_TYPE="${1:-minor}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_DIR"

# Find the latest scala-v* tag
LATEST_TAG=$(git tag -l 'scala-v*' --sort=-v:refname | head -1)
if [ -z "$LATEST_TAG" ]; then
    echo "No existing scala-v* tag found, starting at 0.2.0"
    CURRENT="0.1.0"
else
    CURRENT="${LATEST_TAG#scala-v}"
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
TAG_NAME="scala-v${NEXT_VERSION}"

echo "Current: ${CURRENT} -> Next: ${NEXT_VERSION} (${TAG_NAME})"
echo ""

# Update version.sbt
echo "ThisBuild / version := \"${NEXT_VERSION}\"" > version.sbt
echo "Updated version.sbt to ${NEXT_VERSION}"

# Generate CHANGELOG if git-cliff is available
if command -v git-cliff &>/dev/null; then
    git-cliff --config cliff.toml --tag "$TAG_NAME" -o CHANGELOG.md
    echo "Generated CHANGELOG.md"
else
    echo "WARNING: git-cliff not found, skipping CHANGELOG generation"
    echo "Install: cargo install git-cliff"
fi

# Commit and tag
git add version.sbt CHANGELOG.md
git commit -m "chore(release): scala v${NEXT_VERSION}"
git tag -a "$TAG_NAME" -m "Release scala v${NEXT_VERSION}"

# Prepare next SNAPSHOT
DEV_MINOR=$((MINOR + 1))
DEV_VERSION="${MAJOR}.${DEV_MINOR}.0-SNAPSHOT"
echo "ThisBuild / version := \"${DEV_VERSION}\"" > version.sbt
git add version.sbt
git commit -m "chore: prepare next development cycle (${DEV_VERSION})"

echo ""
echo "Release ${TAG_NAME} created."
echo "Next development version: ${DEV_VERSION}"
echo ""
echo "To publish:"
echo "  git push origin $(git branch --show-current) --follow-tags"
