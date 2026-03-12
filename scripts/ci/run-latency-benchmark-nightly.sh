#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${1:-docker/shard-with-autopropose.yml}"
DURATION_SECONDS="${2:-120}"
OUT_BASE="${3:-/tmp/casper-latency-benchmark-nightly-$(date -u +%Y%m%dT%H%M%SZ)}"
SUMMARY_OUT="${SUMMARY_OUT:-}"

STRICT_OUT="${OUT_BASE}-strict"
SOAK_OUT="${OUT_BASE}-soak-autoheal"
SUMMARY_JSON="${OUT_BASE}-summary.json"

extract_metric() {
  local summary_file="$1"
  local key="$2"
  if [[ ! -f "$summary_file" ]]; then
    echo "null"
    return 0
  fi
  local value
  value="$(awk -F': *' -v key="$key" '$1 == key { print $2; exit }' "$summary_file")"
  if [[ -z "$value" ]]; then
    echo "null"
  else
    printf '"%s"\n' "$value"
  fi
}

write_summary_json() {
  local strict_status="$1"
  local fallback_status="$2"
  local path_taken="$3"
  local completed_at
  completed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  local strict_summary="$STRICT_OUT/profile/summary.txt"
  local fallback_summary="$SOAK_OUT/profile/summary.txt"
  local strict_retry_ratio strict_broadcast_only strict_propose strict_p95
  local fallback_retry_ratio fallback_broadcast_only fallback_propose fallback_p95

  strict_retry_ratio="$(extract_metric "$strict_summary" "block_requests_retry_ratio")"
  strict_broadcast_only="$(extract_metric "$strict_summary" "block_requests_retry_action_broadcast_only")"
  strict_propose="$(extract_metric "$strict_summary" "propose_total")"
  strict_p95="$(extract_metric "$strict_summary" "finalizer_total")"

  fallback_retry_ratio="$(extract_metric "$fallback_summary" "block_requests_retry_ratio")"
  fallback_broadcast_only="$(extract_metric "$fallback_summary" "block_requests_retry_action_broadcast_only")"
  fallback_propose="$(extract_metric "$fallback_summary" "propose_total")"
  fallback_p95="$(extract_metric "$fallback_summary" "finalizer_total")"

  cat >"$SUMMARY_JSON" <<EOF
{
  "completed_at_utc": "$completed_at",
  "compose_file": "$COMPOSE_FILE",
  "duration_seconds": $DURATION_SECONDS,
  "path_taken": "$path_taken",
  "strict": {
    "status": "$strict_status",
    "artifact_dir": "$STRICT_OUT",
    "summary_file": "$strict_summary",
    "block_requests_retry_ratio": $strict_retry_ratio,
    "block_requests_retry_action_broadcast_only": $strict_broadcast_only,
    "propose_total": $strict_propose,
    "finalizer_total": $strict_p95
  },
  "fallback": {
    "status": "$fallback_status",
    "artifact_dir": "$SOAK_OUT",
    "summary_file": "$fallback_summary",
    "block_requests_retry_ratio": $fallback_retry_ratio,
    "block_requests_retry_action_broadcast_only": $fallback_broadcast_only,
    "propose_total": $fallback_propose,
    "finalizer_total": $fallback_p95
  }
}
EOF
  echo "Summary JSON: $SUMMARY_JSON"
  if [[ -n "$SUMMARY_OUT" ]]; then
    mkdir -p "$(dirname "$SUMMARY_OUT")"
    cp "$SUMMARY_JSON" "$SUMMARY_OUT"
    echo "Summary JSON copy: $SUMMARY_OUT"
  fi
}

echo "Nightly latency sequence"
echo "  compose_file: $COMPOSE_FILE"
echo "  duration_seconds: $DURATION_SECONDS"
echo "  strict_output: $STRICT_OUT"
echo "  fallback_output: $SOAK_OUT"
if [[ -n "$SUMMARY_OUT" ]]; then
  echo "  summary_out: $SUMMARY_OUT"
fi

if ./scripts/ci/run-latency-benchmark-mode.sh strict-ci "$COMPOSE_FILE" "$DURATION_SECONDS" "$STRICT_OUT"; then
  write_summary_json "passed" "not_run" "strict"
  echo "Nightly result: strict-ci passed (fallback not needed)."
  echo "Artifacts: $STRICT_OUT"
  exit 0
fi

echo "Nightly result: strict-ci failed; running soak-autoheal fallback."
if ./scripts/ci/run-latency-benchmark-mode.sh soak-autoheal "$COMPOSE_FILE" "$DURATION_SECONDS" "$SOAK_OUT"; then
  write_summary_json "failed" "passed" "fallback"
else
  write_summary_json "failed" "failed" "fallback"
  exit 1
fi
echo "Artifacts:"
echo "  strict: $STRICT_OUT"
echo "  fallback: $SOAK_OUT"
