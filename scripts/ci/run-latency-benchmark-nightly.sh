#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${1:-docker/shard-with-autopropose.yml}"
DURATION_SECONDS="${2:-120}"
OUT_BASE="${3:-/tmp/casper-latency-benchmark-nightly-$(date -u +%Y%m%dT%H%M%SZ)}"

STRICT_OUT="${OUT_BASE}-strict"
SOAK_OUT="${OUT_BASE}-soak-autoheal"

echo "Nightly latency sequence"
echo "  compose_file: $COMPOSE_FILE"
echo "  duration_seconds: $DURATION_SECONDS"
echo "  strict_output: $STRICT_OUT"
echo "  fallback_output: $SOAK_OUT"

if ./scripts/ci/run-latency-benchmark-mode.sh strict-ci "$COMPOSE_FILE" "$DURATION_SECONDS" "$STRICT_OUT"; then
  echo "Nightly result: strict-ci passed (fallback not needed)."
  echo "Artifacts: $STRICT_OUT"
  exit 0
fi

echo "Nightly result: strict-ci failed; running soak-autoheal fallback."
./scripts/ci/run-latency-benchmark-mode.sh soak-autoheal "$COMPOSE_FILE" "$DURATION_SECONDS" "$SOAK_OUT"
echo "Artifacts:"
echo "  strict: $STRICT_OUT"
echo "  fallback: $SOAK_OUT"
