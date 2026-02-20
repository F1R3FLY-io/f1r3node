#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${1:-docker/shard-with-autopropose.yml}"
DURATION_SECONDS="${2:-90}"
OUT_DIR="${3:-/tmp/casper-latency-benchmark-$(date +%Y%m%d-%H%M%S)}"

RUST_CLIENT_DIR="${RUST_CLIENT_DIR:-/home/purplezky/work/asi/tests/rust-client}"
DEPLOY_FILE="${DEPLOY_FILE:-/home/purplezky/work/asi/tests/firefly-rholang-tests-finality-suite-v2/corpus/core/hello_world/contract.rho}"
DEPLOY_INTERVAL_SECONDS="${DEPLOY_INTERVAL_SECONDS:-1}"
PROPOSE_EVERY="${PROPOSE_EVERY:-15}"
CASPER_INIT_SLA_SECONDS="${CASPER_INIT_SLA_SECONDS:-240}"

mkdir -p "$OUT_DIR"

if [[ ! -d "$RUST_CLIENT_DIR" ]]; then
  echo "Rust client directory not found: $RUST_CLIENT_DIR" >&2
  exit 2
fi
if [[ ! -f "$DEPLOY_FILE" ]]; then
  echo "Deploy file not found: $DEPLOY_FILE" >&2
  exit 2
fi

echo "Benchmark start"
echo "  compose_file: $COMPOSE_FILE"
echo "  duration_seconds: $DURATION_SECONDS"
echo "  deploy_file: $DEPLOY_FILE"
echo "  deploy_interval_seconds: $DEPLOY_INTERVAL_SECONDS"
echo "  output_dir: $OUT_DIR"

echo "Step 1/4: ensuring validators are initialized"
./scripts/ci/check-casper-init-sla.sh "$COMPOSE_FILE" "$CASPER_INIT_SLA_SECONDS" >"$OUT_DIR/init-sla.log" 2>&1

echo "Step 2/4: clearing rust-client tip cache and warming rust-client build cache"
./scripts/ci/clear-rust-client-tip-cache.sh >"$OUT_DIR/rust-client-tip-cache.log" 2>&1
(
  cd "$RUST_CLIENT_DIR"
  timeout 240 cargo build --quiet
) >"$OUT_DIR/rust-client-build.log" 2>&1

echo "Step 3/4: running fixed-duration deploy load"
load_log="$OUT_DIR/load.log"
success=0
failure=0
propose_ok=0
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
    timeout 30 cargo run --quiet -- deploy --file "$DEPLOY_FILE" -b
  ) >>"$load_log" 2>&1; then
    success=$((success + 1))
  else
    failure=$((failure + 1))
  fi

  if (( PROPOSE_EVERY > 0 )) && (( iteration % PROPOSE_EVERY == 0 )); then
    if (
      cd "$RUST_CLIENT_DIR" &&
      timeout 30 cargo run --quiet -- propose
    ) >>"$load_log" 2>&1; then
      propose_ok=$((propose_ok + 1))
    else
      propose_fail=$((propose_fail + 1))
    fi
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
propose_fail: $propose_fail
EOF

echo "Step 4/4: collecting latency profile"
./scripts/ci/profile-casper-latency.sh "$COMPOSE_FILE" "$OUT_DIR/profile" >"$OUT_DIR/profile.log" 2>&1

echo "Benchmark completed"
cat "$OUT_DIR/load-summary.txt"
echo
echo "Profile summary:"
cat "$OUT_DIR/profile/summary.txt"
echo
echo "Artifacts written to $OUT_DIR"
