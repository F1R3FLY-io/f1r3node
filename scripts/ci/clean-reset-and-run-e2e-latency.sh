#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${1:-docker/shard-with-autopropose.local.yml}"
E2E_SUITE_DIR="${2:-/home/purplezky/work/asi/tests/firefly-rholang-tests-finality-suite-v2}"
LATENCY_DURATION_SECONDS="${3:-120}"
LATENCY_OUT_DIR="${4:-/tmp/validator-latency-clean-run-$(date -u +%Y%m%dT%H%M%SZ)}"

DEPLOY_HOST="${DEPLOY_HOST:-localhost}"
DEPLOY_HTTP_PORT="${DEPLOY_HTTP_PORT:-40413}"
GENESIS_READY_TIMEOUT_SECONDS="${GENESIS_READY_TIMEOUT_SECONDS:-300}"
GENESIS_READY_SLEEP_SECONDS="${GENESIS_READY_SLEEP_SECONDS:-2}"

# Security-sensitive defaults.
export F1R3_SYNCHRONY_CONSTRAINT_THRESHOLD="${F1R3_SYNCHRONY_CONSTRAINT_THRESHOLD:-0.67}"
export F1R3_SYNCHRONY_RECOVERY_MAX_BYPASSES="${F1R3_SYNCHRONY_RECOVERY_MAX_BYPASSES:-0}"

# Use strict latency benchmark mode by default (no gate overrides).
BENCHMARK_MODE="${BENCHMARK_MODE:-strict-ci}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 2
fi

if docker compose -f "$COMPOSE_FILE" ps >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose -f "$COMPOSE_FILE")
elif docker-compose -f "$COMPOSE_FILE" ps >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose -f "$COMPOSE_FILE")
else
  echo "Unable to access compose services using $COMPOSE_FILE" >&2
  exit 2
fi

if [[ ! -d "$E2E_SUITE_DIR" ]]; then
  echo "E2E suite dir not found: $E2E_SUITE_DIR" >&2
  exit 2
fi

COMPOSE_FILE_ABS="$(cd "$(dirname "$COMPOSE_FILE")" && pwd)/$(basename "$COMPOSE_FILE")"
DATA_DIR="$(cd "$(dirname "$COMPOSE_FILE_ABS")" && pwd)/data"

extract_json_int() {
  local key="$1"
  local payload="$2"
  awk -v key="$key" '
    match($0, "\"" key "\"[[:space:]]*:[[:space:]]*(-?[0-9]+)", m) {
      print m[1]
      exit
    }
  ' <<<"$payload"
}

wait_for_genesis_readiness() {
  local deadline=$(( $(date +%s) + GENESIS_READY_TIMEOUT_SECONDS ))
  local peers seq_num finalized
  local status_payload prepare_payload finalized_payload

  while true; do
    status_payload="$(curl -fsS "http://${DEPLOY_HOST}:${DEPLOY_HTTP_PORT}/api/status" 2>/dev/null || true)"
    prepare_payload="$(curl -fsS "http://${DEPLOY_HOST}:${DEPLOY_HTTP_PORT}/api/prepare-deploy" 2>/dev/null || true)"
    finalized_payload="$(curl -fsS "http://${DEPLOY_HOST}:${DEPLOY_HTTP_PORT}/api/last-finalized-block" 2>/dev/null || true)"

    peers="$(extract_json_int "peers" "$status_payload")"
    seq_num="$(extract_json_int "seqNumber" "$prepare_payload")"
    finalized="$(extract_json_int "seqNum" "$finalized_payload")"

    echo "readiness: peers=${peers:-?} seqNumber=${seq_num:-?} finalized=${finalized:-?}"

    if [[ -n "${peers:-}" && -n "${seq_num:-}" && -n "${finalized:-}" ]] \
      && (( peers >= 3 )) && (( seq_num >= 0 )) && (( finalized >= 0 )); then
      return 0
    fi

    if (( $(date +%s) >= deadline )); then
      echo "Timed out waiting for genesis readiness (${GENESIS_READY_TIMEOUT_SECONDS}s)" >&2
      return 1
    fi

    sleep "$GENESIS_READY_SLEEP_SECONDS"
  done
}

echo "Clean reset + e2e + latency run"
echo "  compose_file: $COMPOSE_FILE"
echo "  e2e_suite_dir: $E2E_SUITE_DIR"
echo "  latency_duration_seconds: $LATENCY_DURATION_SECONDS"
echo "  latency_output_dir: $LATENCY_OUT_DIR"
echo "  synchrony_threshold: $F1R3_SYNCHRONY_CONSTRAINT_THRESHOLD"
echo "  synchrony_recovery_max_bypasses: $F1R3_SYNCHRONY_RECOVERY_MAX_BYPASSES"
echo "  benchmark_mode: $BENCHMARK_MODE"

echo "Step 1/6: stopping compose stack"
"${COMPOSE_CMD[@]}" down -v

echo "Step 2/6: wiping bind-mounted node data under $DATA_DIR"
if [[ -d "$DATA_DIR" ]]; then
  docker run --rm -v "$DATA_DIR:/data" alpine sh -lc 'rm -rf /data/rnode.*'
fi

echo "Step 3/6: starting compose stack"
"${COMPOSE_CMD[@]}" up -d

echo "Step 4/6: waiting for genesis readiness"
wait_for_genesis_readiness

echo "Step 5/6: running e2e suite"
(
  cd "$E2E_SUITE_DIR"
  ./test.sh
)

echo "Step 6/6: running latency benchmark"
./scripts/ci/run-latency-benchmark-mode.sh \
  "$BENCHMARK_MODE" \
  "$COMPOSE_FILE" \
  "$LATENCY_DURATION_SECONDS" \
  "$LATENCY_OUT_DIR"

echo "Completed successfully."
echo "Latency artifacts: $LATENCY_OUT_DIR"
