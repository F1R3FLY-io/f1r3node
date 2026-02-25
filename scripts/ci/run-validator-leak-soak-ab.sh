#!/usr/bin/env bash
set -euo pipefail

RUN_SCRIPT="${RUN_SCRIPT:-./scripts/ci/run-validator-leak-soak.sh}"

BASELINE_COMPOSE="${1:-docker/shard-with-autopropose.yml}"
NO_HEARTBEAT_COMPOSE="${2:-docker/shard-with-autopropose.no-heartbeat.yml}"
DURATION_SECONDS="${3:-1800}"
SAMPLE_EVERY_SECONDS="${4:-10}"
OUT_DIR="${5:-/tmp/casper-validator-leak-soak-ab-$(date -u +%Y%m%dT%H%M%SZ)}"

SOAK_RESTART_CLEAN="${SOAK_RESTART_CLEAN:-1}"
SOAK_WARMUP_SECONDS="${SOAK_WARMUP_SECONDS:-20}"
SOAK_WAIT_FOR_READY="${SOAK_WAIT_FOR_READY:-1}"
SOAK_READY_TIMEOUT_SECONDS="${SOAK_READY_TIMEOUT_SECONDS:-180}"
SOAK_PROFILE_FINALIZER="${SOAK_PROFILE_FINALIZER:-1}"

RUN_ALLOCATOR_HOTSPOTS="${RUN_ALLOCATOR_HOTSPOTS:-0}"
HOTSPOT_DURATION_SECONDS="${HOTSPOT_DURATION_SECONDS:-$DURATION_SECONDS}"
HOTSPOT_VALIDATOR_SERVICE="${HOTSPOT_VALIDATOR_SERVICE:-validator2}"
HOTSPOT_TOP_STACKS="${HOTSPOT_TOP_STACKS:-15}"
HOTSPOT_TOP_FRAMES="${HOTSPOT_TOP_FRAMES:-12}"
HOTSPOT_CALLSITE_CATEGORIES="${HOTSPOT_CALLSITE_CATEGORIES:-block_proto_decode,interpreter_eval,proposer_compute_state}"
AB_REQUIRE_IMPROVEMENT="${AB_REQUIRE_IMPROVEMENT:-0}"
AB_TARGET_RSS_IMPROVE_PCT="${AB_TARGET_RSS_IMPROVE_PCT:-0}"
AB_TARGET_ANON_IMPROVE_PCT="${AB_TARGET_ANON_IMPROVE_PCT:-0}"

SERVICES=(validator1 validator2 validator3)

if [[ "${BASELINE_COMPOSE}" == "${NO_HEARTBEAT_COMPOSE}" ]]; then
  echo "Baseline and no-heartbeat compose files must differ" >&2
  exit 2
fi

if [[ ! -x "$RUN_SCRIPT" ]]; then
  echo "Run script not executable: $RUN_SCRIPT" >&2
  exit 2
fi

run_case() {
  local label="$1"
  local compose_file="$2"
  local out_dir="$3"
  local start_utc

  mkdir -p "$out_dir"
  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  echo "[${label}] compose=$compose_file"
  echo "[${label}] start=${start_utc}"

  SOAK_RESTART_CLEAN="$SOAK_RESTART_CLEAN" \
  SOAK_WARMUP_SECONDS="$SOAK_WARMUP_SECONDS" \
  SOAK_WAIT_FOR_READY="$SOAK_WAIT_FOR_READY" \
  SOAK_READY_TIMEOUT_SECONDS="$SOAK_READY_TIMEOUT_SECONDS" \
  SOAK_PROFILE_PROC=1 \
  SOAK_PROC_SAMPLE_EVERY_SECONDS="$SAMPLE_EVERY_SECONDS" \
  SOAK_PROFILE_FINALIZER="$SOAK_PROFILE_FINALIZER" \
  "$RUN_SCRIPT" "$compose_file" "$DURATION_SECONDS" "$SAMPLE_EVERY_SECONDS" "$out_dir" >"$out_dir/run.log" 2>&1

  echo "[${label}] complete (log: $out_dir/run.log)"
}

run_allocator_hotspots() {
  local compose_file="$1"
  local out_dir="$2"
  local hotspot_out="$out_dir/hotspots"

  mkdir -p "$hotspot_out"

  echo "[${out_dir}] running allocator hotspots (duration=${HOTSPOT_DURATION_SECONDS}s, validator=${HOTSPOT_VALIDATOR_SERVICE})"

  if ! VALIDATOR_SERVICE="$HOTSPOT_VALIDATOR_SERVICE" \
    TOP_STACKS="$HOTSPOT_TOP_STACKS" \
    TOP_FRAMES="$HOTSPOT_TOP_FRAMES" \
    ./scripts/ci/profile-validator-allocator-hotspots.sh "$compose_file" "$HOTSPOT_DURATION_SECONDS" "$hotspot_out" >"$out_dir/hotspots.log" 2>&1; then
    echo "Allocator hotspot profiling failed (see $out_dir/hotspots.log)" >&2
    return 1
  fi

  echo "[${out_dir}] allocator hotspot complete (summary: $hotspot_out/summary.txt)"
}

extract_comma_field() {
  local file="$1"
  local service="$2"
  local field="$3"
  awk -v service="$service" -v field="$field" -F',' '
    $1 ~ "^" service ":" {
      for (i = 2; i <= NF; i++) {
        s = $i
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", s)
        if (s ~ "^" field "=") {
          gsub("^" field "=", "", s)
          print s
          exit
        }
      }
    }
  ' "$file" | head -n1
}

extract_summary_field() {
  local file="$1"
  local service="$2"
  local field="$3"
  local value
  value="$(extract_comma_field "$file" "$service" "$field")"
  echo "${value:-0}"
}

sum_service_field() {
  local file="$1"
  local field="$2"
  local total=0

  for service in "${SERVICES[@]}"; do
    local value
    value="$(extract_comma_field "$file" "$service" "$field")"
    total="$(awk -v a="$total" -v b="${value:-0}" 'BEGIN { printf "%.10f", a + b }')"
  done

  echo "$total"
}

extract_hotspot_category_bytes() {
  local file="$1"
  local category="$2"
  if [[ ! -f "$file" ]]; then
    echo 0
    return 0
  fi
  awk -F'\t' -v category="$category" '
    {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
      if ($2 == category) {
        print $1
        exit
      }
    }
  ' "$file"
}

run_case_and_report() {
  local label="$1"
  local compose_file="$2"
  local out_dir="$3"

  run_case "$label" "$compose_file" "$out_dir"

  if [[ "$RUN_ALLOCATOR_HOTSPOTS" == "1" ]]; then
    run_allocator_hotspots "$compose_file" "$out_dir" || true
  fi
}

BASELINE_OUT="$OUT_DIR/baseline"
NO_HEARTBEAT_OUT="$OUT_DIR/no-heartbeat"
COMPARISON_CSV="$OUT_DIR/comparison.csv"

mkdir -p "$OUT_DIR"
echo "Output dir: $OUT_DIR"

run_case_and_report "baseline" "$BASELINE_COMPOSE" "$BASELINE_OUT"
run_case_and_report "no-heartbeat" "$NO_HEARTBEAT_COMPOSE" "$NO_HEARTBEAT_OUT"

BASELINE_SUMMARY="$BASELINE_OUT/summary.txt"
BASELINE_PROC="$BASELINE_OUT/proc-summary.txt"
NO_HEARTBEAT_SUMMARY="$NO_HEARTBEAT_OUT/summary.txt"
NO_HEARTBEAT_PROC="$NO_HEARTBEAT_OUT/proc-summary.txt"

for required in "$BASELINE_SUMMARY" "$BASELINE_PROC" "$NO_HEARTBEAT_SUMMARY" "$NO_HEARTBEAT_PROC"; do
  if [[ ! -f "$required" ]]; then
    echo "Expected output file not found: $required" >&2
    exit 1
  fi
done

: >"$COMPARISON_CSV"
{
  echo "metric,service,baseline,no_heartbeat,delta,improvement_pct"
} >"$COMPARISON_CSV"

if [[ "$RUN_ALLOCATOR_HOTSPOTS" == "1" ]]; then
  BASELINE_HOTSPOT_CALLSITES="$BASELINE_OUT/hotspots/stacks/delta-positive.top${HOTSPOT_TOP_STACKS}.callsites.txt"
  NO_HEARTBEAT_HOTSPOT_CALLSITES="$NO_HEARTBEAT_OUT/hotspots/stacks/delta-positive.top${HOTSPOT_TOP_STACKS}.callsites.txt"

  if [[ ! -f "$BASELINE_HOTSPOT_CALLSITES" || ! -f "$NO_HEARTBEAT_HOTSPOT_CALLSITES" ]]; then
    echo "Allocator hotspot output missing for one or both cases; skipping category diff" >&2
    RUN_ALLOCATOR_HOTSPOTS=0
  fi
fi

echo
echo "A/B leak delta summary"
echo "  baseline compose: $BASELINE_COMPOSE"
echo "  no-heartbeat compose: $NO_HEARTBEAT_COMPOSE"
echo

baseline_slope_sum=0
noheartbeat_slope_sum=0

for service in "${SERVICES[@]}"; do
  base_slope="$(extract_summary_field "$BASELINE_SUMMARY" "$service" "rss_slope_mib_per_s")"
  no_slope="$(extract_summary_field "$NO_HEARTBEAT_SUMMARY" "$service" "rss_slope_mib_per_s")"
  base_anon="$(extract_comma_field "$BASELINE_PROC" "$service" "Anonymous_delta_kb")"
  no_anon="$(extract_comma_field "$NO_HEARTBEAT_PROC" "$service" "Anonymous_delta_kb")"

  baseline_slope_sum="$(awk -v a="$baseline_slope_sum" -v b="$base_slope" 'BEGIN { printf "%.10f", a + b }')"
  noheartbeat_slope_sum="$(awk -v a="$noheartbeat_slope_sum" -v b="$no_slope" 'BEGIN { printf "%.10f", a + b }')"

  slope_delta="$(awk -v b="$base_slope" -v n="$no_slope" 'BEGIN { printf "%.10f", n - b }')"
  anon_delta="$(awk -v b="$base_anon" -v n="$no_anon" 'BEGIN { printf "%.10f", n - b }')"
  slope_delta_pct="$(awk -v b="$base_slope" -v n="$no_slope" 'BEGIN { if (b == 0) print "n/a"; else printf "%.3f", ((b - n) / b * 100) }')"
  anon_delta_pct="$(awk -v b="$base_anon" -v n="$no_anon" 'BEGIN { if (b == 0) print "n/a"; else printf "%.3f", ((b - n) / b * 100) }')"

  echo "  $service"
  echo "    rss_slope_mib_per_s: baseline=$base_slope no_heartbeat=$no_slope delta=$slope_delta (${slope_delta_pct}% improvement)"
  echo "    Anonymous_delta_kb: baseline=$base_anon no_heartbeat=$no_anon delta=$anon_delta"
  printf "rss_slope_mib_per_s,%s,%s,%s,%s,%s\n" \
    "$service" "$base_slope" "$no_slope" "$slope_delta" "$slope_delta_pct" >>"$COMPARISON_CSV"
  printf "Anonymous_delta_kb,%s,%s,%s,%s,%s\n" \
    "$service" "$base_anon" "$no_anon" "$anon_delta" "$anon_delta_pct" >>"$COMPARISON_CSV"
done

service_count="${#SERVICES[@]}"
baseline_slope_avg="$(awk -v t="$baseline_slope_sum" -v n="$service_count" 'BEGIN { printf "%.10f", t / n }')"
noheartbeat_slope_avg="$(awk -v t="$noheartbeat_slope_sum" -v n="$service_count" 'BEGIN { printf "%.10f", t / n }')"
baseline_slope_avg_delta="$(awk -v b="$baseline_slope_avg" -v n="$noheartbeat_slope_avg" 'BEGIN { printf "%.10f", n - b }')"
baseline_anon_total="$(sum_service_field "$BASELINE_PROC" "Anonymous_delta_kb")"
noheartbeat_anon_total="$(sum_service_field "$NO_HEARTBEAT_PROC" "Anonymous_delta_kb")"
anon_total_delta="$(awk -v b="$baseline_anon_total" -v n="$noheartbeat_anon_total" 'BEGIN { printf "%.10f", n - b }')"

echo
echo "Aggregate totals"
echo "  baseline_rss_slope_avg_mib_per_s=$baseline_slope_avg"
echo "  no_heartbeat_rss_slope_avg_mib_per_s=$noheartbeat_slope_avg"
echo "  avg_slope_delta=($noheartbeat_slope_avg - $baseline_slope_avg) $baseline_slope_avg_delta"
echo "  baseline_anonymous_delta_kb_total=$baseline_anon_total"
echo "  no_heartbeat_anonymous_delta_kb_total=$noheartbeat_anon_total"
echo "  anonymous_total_delta=($noheartbeat_anon_total - $baseline_anon_total) $anon_total_delta"
echo "  comparison_csv: $COMPARISON_CSV"
baseline_anon_delta_pct="$(awk -v b="$baseline_anon_total" -v n="$noheartbeat_anon_total" 'BEGIN { if (b == 0) print "n/a"; else printf "%.3f", ((b - n) / b * 100) }')"
baseline_slope_avg_pct="$(awk -v b="$baseline_slope_avg" -v n="$noheartbeat_slope_avg" 'BEGIN { if (b == 0) print "n/a"; else printf "%.3f", ((b - n) / b * 100) }')"
printf "rss_slope_mib_per_s,aggregate,$baseline_slope_avg,$noheartbeat_slope_avg,$baseline_slope_avg_delta,$baseline_slope_avg_pct\n" >>"$COMPARISON_CSV"
printf "Anonymous_delta_kb_aggregate,total,$baseline_anon_total,$noheartbeat_anon_total,$anon_total_delta,$baseline_anon_delta_pct\n" >>"$COMPARISON_CSV"

if [[ "$RUN_ALLOCATOR_HOTSPOTS" == "1" ]]; then
  echo
  echo "Allocator hotspot deltas (delta-positive callsite categories)"
  echo "  hotspot categories: ${HOTSPOT_CALLSITE_CATEGORIES}"
  echo "  hotspot_top_stacks: $HOTSPOT_TOP_STACKS"
  echo "  hotspot_top_frames: $HOTSPOT_TOP_FRAMES"
  echo "  baseline_hotspots_summary: $BASELINE_OUT/hotspots/summary.txt"
  echo "  no_heartbeat_hotspots_summary: $NO_HEARTBEAT_OUT/hotspots/summary.txt"
  IFS=',' read -r -a categories <<< "$HOTSPOT_CALLSITE_CATEGORIES"
  for category in "${categories[@]}"; do
    base_category="$(extract_hotspot_category_bytes "$BASELINE_HOTSPOT_CALLSITES" "$category")"
    no_category="$(extract_hotspot_category_bytes "$NO_HEARTBEAT_HOTSPOT_CALLSITES" "$category")"
    base_category="${base_category:-0}"
    no_category="${no_category:-0}"
    category_delta="$(awk -v b="$base_category" -v n="$no_category" 'BEGIN { printf "%.10f", n - b }')"
    category_improvement="$(awk -v b="$base_category" -v n="$no_category" 'BEGIN { if (b == 0) print "n/a"; else printf "%.3f", ((b - n) / b * 100) }')"
    echo "  $category"
    echo "    baseline_bytes=$base_category no_heartbeat_bytes=$no_category delta=$category_delta (${category_improvement}% improvement)"
    printf "allocator_category_%s,aggregate,%s,%s,%s,%s\n" \
      "$category" "$base_category" "$no_category" "$category_delta" "$category_improvement" >>"$COMPARISON_CSV"
  done
  echo "  comparison_csv: $COMPARISON_CSV"
fi

if [[ "$AB_REQUIRE_IMPROVEMENT" == "1" ]]; then
  echo
  echo "A/B verdict"
  if [[ "${baseline_slope_avg_pct}" == "n/a" || "${baseline_anon_delta_pct}" == "n/a" ]]; then
    echo "  result=INCONCLUSIVE"
    echo "  reason=baseline values were zero or unavailable for percent-diff computation"
  elif (( $(awk -v b="$baseline_slope_avg_pct" 'BEGIN { if (b + 0 >= 0) print 1; else print 0 }') == 1 )) && \
       (( $(awk -v b="$baseline_anon_delta_pct" 'BEGIN { if (b + 0 >= 0) print 1; else print 0 }') == 1 )) && \
       (( $(awk -v b="$baseline_slope_avg_pct" -v t="$AB_TARGET_RSS_IMPROVE_PCT" 'BEGIN { if (b >= t) print 1; else print 0 }') == 1 )) && \
       (( $(awk -v b="$baseline_anon_delta_pct" -v t="$AB_TARGET_ANON_IMPROVE_PCT" 'BEGIN { if (b >= t) print 1; else print 0 }') == 1 )); then
    echo "  result=PASS"
    echo "  rss_improvement_pct=$baseline_slope_avg_pct (target>=${AB_TARGET_RSS_IMPROVE_PCT})"
    echo "  anon_improvement_pct=$baseline_anon_delta_pct (target>=${AB_TARGET_ANON_IMPROVE_PCT})"
  else
    echo "  result=FAIL"
    echo "  rss_improvement_pct=$baseline_slope_avg_pct (target>=${AB_TARGET_RSS_IMPROVE_PCT})"
    echo "  anon_improvement_pct=$baseline_anon_delta_pct (target>=${AB_TARGET_ANON_IMPROVE_PCT})"
  fi
fi
