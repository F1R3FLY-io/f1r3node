#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${1:-docker/shard-with-autopropose.yml}"
DURATION_SECONDS="${2:-90}"
OUT_DIR="${3:-/tmp/casper-latency-benchmark-$(date +%Y%m%d-%H%M%S)}"

RUST_CLIENT_DIR="${RUST_CLIENT_DIR:-/home/purplezky/work/asi/tests/rust-client}"
DEPLOY_FILE="${DEPLOY_FILE:-/home/purplezky/work/asi/tests/firefly-rholang-tests-finality-suite-v2/corpus/core/hello_world/contract.rho}"
DEPLOY_HOST="${DEPLOY_HOST:-localhost}"
DEPLOY_GRPC_PORT="${DEPLOY_GRPC_PORT:-40412}"
DEPLOY_HTTP_PORT="${DEPLOY_HTTP_PORT:-40413}"
DEPLOY_INTERVAL_SECONDS="${DEPLOY_INTERVAL_SECONDS:-1}"
PROPOSE_EVERY="${PROPOSE_EVERY:-15}"
CASPER_INIT_SLA_SECONDS="${CASPER_INIT_SLA_SECONDS:-240}"
GRPC_READY_TIMEOUT_SECONDS="${GRPC_READY_TIMEOUT_SECONDS:-90}"
GRPC_READY_CONSECUTIVE_SUCCESSES="${GRPC_READY_CONSECUTIVE_SUCCESSES:-2}"
GRPC_READY_SLEEP_SECONDS="${GRPC_READY_SLEEP_SECONDS:-2}"
PRELOAD_REQUIRE_PEERS_MIN="${PRELOAD_REQUIRE_PEERS_MIN:-3}"
PRELOAD_PEER_READY_TIMEOUT_SECONDS="${PRELOAD_PEER_READY_TIMEOUT_SECONDS:-45}"
PRELOAD_PEER_READY_SLEEP_SECONDS="${PRELOAD_PEER_READY_SLEEP_SECONDS:-2}"
PRELOAD_RETRY_RATIO_MAX="${PRELOAD_RETRY_RATIO_MAX:-2.50}"
PRELOAD_RETRY_RATIO_MIN_REQUESTS="${PRELOAD_RETRY_RATIO_MIN_REQUESTS:-100}"
POSTLOAD_RETRY_RATIO_MAX="${POSTLOAD_RETRY_RATIO_MAX:-}"
POSTLOAD_RETRY_RATIO_MIN_REQUESTS="${POSTLOAD_RETRY_RATIO_MIN_REQUESTS:-100}"
AUTO_RECREATE_ON_PRELOAD_FAIL="${AUTO_RECREATE_ON_PRELOAD_FAIL:-0}"
AUTO_RECREATE_MAX_ATTEMPTS="${AUTO_RECREATE_MAX_ATTEMPTS:-1}"
SERVICES=(validator1 validator2 validator3)

mkdir -p "$OUT_DIR"

wait_for_endpoint_stability() {
  local deadline
  local now
  local attempt=0
  local consecutive=0
  local probe_log="$OUT_DIR/grpc-readiness.log"

  deadline=$(( $(date +%s) + GRPC_READY_TIMEOUT_SECONDS ))
  : >"$probe_log"

  while true; do
    now="$(date +%s)"
    if (( now >= deadline )); then
      echo "endpoint readiness FAILED after ${GRPC_READY_TIMEOUT_SECONDS}s. See $probe_log" >&2
      return 1
    fi

    attempt=$((attempt + 1))
    if (
      cd "$RUST_CLIENT_DIR" &&
      timeout 30 cargo run --quiet -- status -H "$DEPLOY_HOST" -p "$DEPLOY_HTTP_PORT"
    ) >>"$probe_log" 2>&1; then
      consecutive=$((consecutive + 1))
      echo "endpoint readiness probe succeeded (${consecutive}/${GRPC_READY_CONSECUTIVE_SUCCESSES})"
      if (( consecutive >= GRPC_READY_CONSECUTIVE_SUCCESSES )); then
        return 0
      fi
    else
      consecutive=0
      echo "endpoint readiness probe failed (attempt $attempt), retrying..."
    fi

    sleep "$GRPC_READY_SLEEP_SECONDS"
  done
}

detect_compose_cmd() {
  if docker compose -f "$COMPOSE_FILE" ps >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose -f $COMPOSE_FILE"
  elif docker-compose -f "$COMPOSE_FILE" ps >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose -f $COMPOSE_FILE"
  else
    echo "Unable to access compose services using $COMPOSE_FILE" >&2
    exit 2
  fi
}

collect_metrics_snapshot() {
  local target_dir="$1"
  mkdir -p "$target_dir"
  for service in "${SERVICES[@]}"; do
    local cid
    cid="$($COMPOSE_CMD ps -q "$service" || true)"
    if [[ -z "$cid" ]]; then
      continue
    fi
    local host_port
    host_port="$(docker port "$cid" 40403/tcp 2>/dev/null | awk -F: 'NR==1 {print $NF}')"
    if [[ -z "$host_port" ]]; then
      continue
    fi
    if command -v curl >/dev/null 2>&1; then
      curl -fsS --max-time 3 "http://127.0.0.1:${host_port}/metrics" >"$target_dir/${service}.metrics.prom" 2>/dev/null || true
    elif command -v wget >/dev/null 2>&1; then
      wget -q -T 3 -O "$target_dir/${service}.metrics.prom" "http://127.0.0.1:${host_port}/metrics" 2>/dev/null || true
    fi
  done
}

sum_metric_from_file() {
  local metrics_file="$1"
  local metric_name="$2"
  awk -v metric_name="$metric_name" '
    index($0, "#") == 1 { next }
    $1 ~ ("^" metric_name "({.*})?$") {
      sum += $2
      found = 1
    }
    END {
      if (found) {
        printf "%.10f\n", sum
      } else {
        print "0"
      }
    }
  ' "$metrics_file"
}

write_preload_diagnostics() {
  local diag_dir="$OUT_DIR/preload-diag"
  mkdir -p "$diag_dir"
  $COMPOSE_CMD ps >"$diag_dir/compose-ps.txt" 2>&1 || true
  collect_metrics_snapshot "$diag_dir/metrics"
  for service in "${SERVICES[@]}"; do
    local cid
    cid="$($COMPOSE_CMD ps -q "$service" || true)"
    if [[ -n "$cid" ]]; then
      docker logs "$cid" --tail 200 >"$diag_dir/${service}.logs.txt" 2>&1 || true
    fi
  done
}

verify_preload_invariants() {
  local status_log="$OUT_DIR/preload-status.log"
  local status_json="$OUT_DIR/preload-status.json"

  for service in "${SERVICES[@]}"; do
    local cid
    local running
    cid="$($COMPOSE_CMD ps -q "$service" || true)"
    if [[ -z "$cid" ]]; then
      echo "Preload invariant FAILED: missing container id for $service" >&2
      write_preload_diagnostics
      return 1
    fi
    running="$(docker inspect -f '{{.State.Running}}' "$cid" 2>/dev/null || true)"
    if [[ "$running" != "true" ]]; then
      echo "Preload invariant FAILED: $service not running (state=$running)" >&2
      write_preload_diagnostics
      return 1
    fi
  done

  local peers=0
  local peer_deadline=$(( $(date +%s) + PRELOAD_PEER_READY_TIMEOUT_SECONDS ))
  local peer_probe_ok=0
  while true; do
    if (
      cd "$RUST_CLIENT_DIR" &&
      timeout 30 cargo run --quiet -- status -H "$DEPLOY_HOST" -p "$DEPLOY_HTTP_PORT"
    ) >"$status_log" 2>&1; then
      if sed -n '/^{/,/^}/p' "$status_log" >"$status_json"; then
        peers="$(awk -F: '/"peers"/ { gsub(/[^0-9]/, "", $2); if ($2 != "") { print $2; exit } }' "$status_json")"
        if [[ -n "$peers" ]] && (( peers >= PRELOAD_REQUIRE_PEERS_MIN )); then
          peer_probe_ok=1
          break
        fi
      fi
    fi

    if (( $(date +%s) >= peer_deadline )); then
      break
    fi
    sleep "$PRELOAD_PEER_READY_SLEEP_SECONDS"
  done

  if (( peer_probe_ok == 0 )); then
    if [[ -z "$peers" ]]; then
      echo "Preload invariant FAILED: peers field unavailable after ${PRELOAD_PEER_READY_TIMEOUT_SECONDS}s. See $status_log" >&2
    else
      echo "Preload invariant FAILED: peers=$peers < required minimum ${PRELOAD_REQUIRE_PEERS_MIN} after ${PRELOAD_PEER_READY_TIMEOUT_SECONDS}s" >&2
    fi
    write_preload_diagnostics
    return 1
  fi

  local ratio_service="validator1"
  local ratio_file="$baseline_metrics_dir/${ratio_service}.metrics.prom"
  if [[ ! -s "$ratio_file" ]]; then
    ratio_service="validator2"
    ratio_file="$baseline_metrics_dir/${ratio_service}.metrics.prom"
  fi
  if [[ ! -s "$ratio_file" ]]; then
    ratio_service="validator3"
    ratio_file="$baseline_metrics_dir/${ratio_service}.metrics.prom"
  fi

  if [[ ! -s "$ratio_file" ]]; then
    echo "Preload invariant FAILED: no baseline metrics file found for retry-ratio check" >&2
    write_preload_diagnostics
    return 1
  fi

  local total retries ratio ratio_over_limit
  total="$(sum_metric_from_file "$ratio_file" "block_requests_total")"
  retries="$(sum_metric_from_file "$ratio_file" "block_requests_retries")"
  ratio="$(awk -v t="$total" -v r="$retries" 'BEGIN { if (t > 0) printf "%.4f", r/t; else print "0.0000" }')"
  ratio_over_limit="$(awk -v t="$total" -v ratio="$ratio" -v min_t="$PRELOAD_RETRY_RATIO_MIN_REQUESTS" -v max_ratio="$PRELOAD_RETRY_RATIO_MAX" 'BEGIN { if (t >= min_t && ratio > max_ratio) print 1; else print 0 }')"

  echo "preload peer check OK: peers=$peers (min=${PRELOAD_REQUIRE_PEERS_MIN})"
  echo "preload retry ratio baseline (${ratio_service}): total=$total retries=$retries ratio=$ratio limit=${PRELOAD_RETRY_RATIO_MAX} (enforced when total>=${PRELOAD_RETRY_RATIO_MIN_REQUESTS})"

  if [[ "$ratio_over_limit" == "1" ]]; then
    echo "Preload invariant FAILED: baseline retry ratio ${ratio_service}=${ratio} exceeds ${PRELOAD_RETRY_RATIO_MAX} with total=${total}" >&2
    write_preload_diagnostics
    return 1
  fi

  return 0
}

if [[ ! -d "$RUST_CLIENT_DIR" ]]; then
  echo "Rust client directory not found: $RUST_CLIENT_DIR" >&2
  exit 2
fi
if [[ ! -f "$DEPLOY_FILE" ]]; then
  echo "Deploy file not found: $DEPLOY_FILE" >&2
  exit 2
fi

detect_compose_cmd

echo "Benchmark start"
echo "  compose_file: $COMPOSE_FILE"
echo "  duration_seconds: $DURATION_SECONDS"
echo "  deploy_file: $DEPLOY_FILE"
echo "  deploy_target: ${DEPLOY_HOST}:${DEPLOY_GRPC_PORT} (grpc), ${DEPLOY_HOST}:${DEPLOY_HTTP_PORT} (http)"
echo "  deploy_interval_seconds: $DEPLOY_INTERVAL_SECONDS"
echo "  output_dir: $OUT_DIR"
echo "  auto_recreate_on_preload_fail: $AUTO_RECREATE_ON_PRELOAD_FAIL (max_attempts=$AUTO_RECREATE_MAX_ATTEMPTS)"
if [[ -n "$POSTLOAD_RETRY_RATIO_MAX" ]]; then
  echo "  postload_retry_ratio_gate: max=$POSTLOAD_RETRY_RATIO_MAX (min_requests=$POSTLOAD_RETRY_RATIO_MIN_REQUESTS)"
fi

preload_attempt=0
while true; do
  preload_attempt=$((preload_attempt + 1))
  attempt_prefix="$OUT_DIR/preload-attempt-${preload_attempt}"
  mkdir -p "$attempt_prefix"

  echo "Step 1/5: ensuring validators are initialized (attempt ${preload_attempt})"
  ./scripts/ci/check-casper-init-sla.sh "$COMPOSE_FILE" "$CASPER_INIT_SLA_SECONDS" >"$attempt_prefix/init-sla.log" 2>&1

  echo "Step 2/5: clearing rust-client tip cache and warming rust-client build cache (attempt ${preload_attempt})"
  ./scripts/ci/clear-rust-client-tip-cache.sh >"$attempt_prefix/rust-client-tip-cache.log" 2>&1
  (
    cd "$RUST_CLIENT_DIR"
    timeout 240 cargo build --quiet
  ) >"$attempt_prefix/rust-client-build.log" 2>&1

  echo "Step 3/5: waiting for deploy endpoint stability (attempt ${preload_attempt})"
  wait_for_endpoint_stability

  baseline_metrics_dir="$attempt_prefix/metrics-baseline"
  collect_metrics_snapshot "$baseline_metrics_dir"
  echo "Step 3.5/5: verifying pre-load correctness invariants (attempt ${preload_attempt})"
  if verify_preload_invariants; then
    cp "$attempt_prefix/init-sla.log" "$OUT_DIR/init-sla.log"
    cp "$attempt_prefix/rust-client-tip-cache.log" "$OUT_DIR/rust-client-tip-cache.log"
    cp "$attempt_prefix/rust-client-build.log" "$OUT_DIR/rust-client-build.log"
    rm -rf "$OUT_DIR/metrics-baseline"
    cp -r "$baseline_metrics_dir" "$OUT_DIR/metrics-baseline"
    baseline_metrics_dir="$OUT_DIR/metrics-baseline"
    break
  fi

  if [[ "$AUTO_RECREATE_ON_PRELOAD_FAIL" != "1" ]] || (( preload_attempt >= AUTO_RECREATE_MAX_ATTEMPTS + 1 )); then
    echo "Pre-load invariants failed and auto-recreate is disabled or exhausted (attempt ${preload_attempt})." >&2
    exit 1
  fi

  echo "Pre-load invariants failed; auto-recreating cluster before retry (attempt ${preload_attempt}/${AUTO_RECREATE_MAX_ATTEMPTS})."
  $COMPOSE_CMD up -d --force-recreate >/dev/null
done

echo "Step 4/5: running fixed-duration deploy load"
profile_log_since_utc="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
load_log="$OUT_DIR/load.log"
success=0
failure=0
propose_ok=0
propose_transient=0
propose_bug_error=0
propose_fail=0
start_epoch="$(date +%s)"
deadline=$((start_epoch + DURATION_SECONDS))
iteration=0

while true; do
  now="$(date +%s)"
  if (( now >= deadline )); then
    break
  fi

  iteration=$((iteration + 1))

  if (
    cd "$RUST_CLIENT_DIR" &&
    timeout 30 cargo run --quiet -- deploy --file "$DEPLOY_FILE" -H "$DEPLOY_HOST" -p "$DEPLOY_GRPC_PORT" -b
  ) >>"$load_log" 2>&1; then
    success=$((success + 1))
  else
    failure=$((failure + 1))
  fi

  if (( PROPOSE_EVERY > 0 )) && (( iteration % PROPOSE_EVERY == 0 )); then
    propose_tmp="$OUT_DIR/propose-${iteration}.log"
    if (
      cd "$RUST_CLIENT_DIR" &&
      timeout 30 cargo run --quiet -- propose -H "$DEPLOY_HOST" -p "$DEPLOY_GRPC_PORT"
    ) >"$propose_tmp" 2>&1; then
      propose_ok=$((propose_ok + 1))
    else
      if rg -q "Propose skipped due to transient proposal race" "$propose_tmp"; then
        propose_transient=$((propose_transient + 1))
      elif rg -q "Proposal failed: BugError" "$propose_tmp"; then
        propose_bug_error=$((propose_bug_error + 1))
        propose_fail=$((propose_fail + 1))
      else
        propose_fail=$((propose_fail + 1))
      fi
    fi
    cat "$propose_tmp" >>"$load_log"
    rm -f "$propose_tmp"
  fi

  sleep "$DEPLOY_INTERVAL_SECONDS"
done

end_epoch="$(date +%s)"
elapsed=$((end_epoch - start_epoch))
attempts=$((success + failure))
deploy_rate="$(awk -v a="$attempts" -v t="$elapsed" 'BEGIN { if (t > 0) printf "%.2f", a/t; else print "0.00" }')"

cat >"$OUT_DIR/load-summary.txt" <<EOF
load_duration_seconds: $elapsed
deploy_attempts: $attempts
deploy_success: $success
deploy_failure: $failure
deploy_attempt_rate_per_s: $deploy_rate
propose_ok: $propose_ok
propose_transient: $propose_transient
propose_bug_error: $propose_bug_error
propose_fail: $propose_fail
EOF

echo "Step 5/5: collecting latency profile"
./scripts/ci/profile-casper-latency.sh "$COMPOSE_FILE" "$OUT_DIR/profile" "$profile_log_since_utc" "$baseline_metrics_dir" >"$OUT_DIR/profile.log" 2>&1

echo "Benchmark completed"
cat "$OUT_DIR/load-summary.txt"
echo
echo "Profile summary:"
cat "$OUT_DIR/profile/summary.txt"
echo

if [[ -n "$POSTLOAD_RETRY_RATIO_MAX" ]]; then
  post_requests="$(awk -F': *' '/^block_requests_total:/ {print $2}' "$OUT_DIR/profile/summary.txt" | head -n1)"
  post_retries="$(awk -F': *' '/^block_requests_retries:/ {print $2}' "$OUT_DIR/profile/summary.txt" | head -n1)"
  post_ratio="$(awk -F': *' '/^block_requests_retry_ratio:/ {print $2}' "$OUT_DIR/profile/summary.txt" | head -n1)"
  post_requests="${post_requests:-0}"
  post_retries="${post_retries:-0}"
  post_ratio="${post_ratio:-0}"
  post_over_limit="$(awk -v t="$post_requests" -v ratio="$post_ratio" -v min_t="$POSTLOAD_RETRY_RATIO_MIN_REQUESTS" -v max_ratio="$POSTLOAD_RETRY_RATIO_MAX" 'BEGIN { if (t >= min_t && ratio > max_ratio) print 1; else print 0 }')"
  echo "Post-load retry ratio: total=$post_requests retries=$post_retries ratio=$post_ratio limit=$POSTLOAD_RETRY_RATIO_MAX (enforced when total>=${POSTLOAD_RETRY_RATIO_MIN_REQUESTS})"
  if [[ "$post_over_limit" == "1" ]]; then
    echo "Post-load quality gate FAILED: retry ratio $post_ratio exceeds $POSTLOAD_RETRY_RATIO_MAX with total=$post_requests" >&2
    exit 1
  fi
fi

echo "Artifacts written to $OUT_DIR"
