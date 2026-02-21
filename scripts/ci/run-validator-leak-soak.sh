#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${1:-docker/shard-with-autopropose.yml}"
DURATION_SECONDS="${2:-600}"
SAMPLE_EVERY_SECONDS="${3:-10}"
OUT_DIR="${4:-/tmp/casper-validator-leak-soak-$(date -u +%Y%m%dT%H%M%SZ)}"

SERVICES=(validator1 validator2 validator3)

mkdir -p "$OUT_DIR"
SAMPLES_CSV="$OUT_DIR/samples.csv"
SUMMARY_TXT="$OUT_DIR/summary.txt"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 2
fi

if ! docker compose -f "$COMPOSE_FILE" ps >/dev/null 2>&1; then
  echo "Unable to access compose services using $COMPOSE_FILE" >&2
  exit 2
fi

echo "timestamp_utc,elapsed_s,service,rss_mib,block_requests_total,block_requests_retries,block_requests_retry_ratio,dep_tracking_size,broadcast_tracking_size,active_validators_cache_size" >"$SAMPLES_CSV"

to_mib() {
  # Input examples: 89.34MiB / 7.75GiB, 1.2GiB / 7.75GiB, 500KiB / 7.75GiB
  local raw="$1"
  awk '
    function trim(s) { sub(/^[[:space:]]+/, "", s); sub(/[[:space:]]+$/, "", s); return s }
    BEGIN {
      v = trim(ARGV[1])
      sub(/[[:space:]]*\/.*/, "", v)
      if (v ~ /GiB$/) { sub(/GiB$/, "", v); printf "%.6f", (v + 0) * 1024.0; exit }
      if (v ~ /MiB$/) { sub(/MiB$/, "", v); printf "%.6f", (v + 0); exit }
      if (v ~ /KiB$/) { sub(/KiB$/, "", v); printf "%.6f", (v + 0) / 1024.0; exit }
      if (v ~ /B$/)   { sub(/B$/, "", v);   printf "%.6f", (v + 0) / (1024.0 * 1024.0); exit }
      # fallback assume MiB-like numeric
      printf "%.6f", (v + 0)
    }
  ' "$raw"
}

metric_sum() {
  local metrics_text="$1"
  local metric_name="$2"
  awk -v metric_name="$metric_name" '
    index($0, "#") == 1 { next }
    $1 ~ ("^" metric_name "({.*})?$") { sum += $2; found = 1 }
    END { if (found) printf "%.10f", sum; else print "0" }
  ' <<<"$metrics_text"
}

start_epoch="$(date +%s)"
deadline=$((start_epoch + DURATION_SECONDS))

while true; do
  now_epoch="$(date +%s)"
  if (( now_epoch > deadline )); then
    break
  fi
  elapsed=$((now_epoch - start_epoch))
  timestamp_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  for service in "${SERVICES[@]}"; do
    cid="$(docker compose -f "$COMPOSE_FILE" ps -q "$service" || true)"
    if [[ -z "$cid" ]]; then
      continue
    fi

    rss_raw="$(docker stats --no-stream --format '{{.MemUsage}}' "$cid" 2>/dev/null || echo "0B / 0B")"
    rss_mib="$(to_mib "$rss_raw")"

    metrics=""
    host_port="$(docker port "$cid" 40403/tcp 2>/dev/null | awk -F: 'NR==1 {print $NF}')"
    if [[ -n "$host_port" ]]; then
      if command -v curl >/dev/null 2>&1; then
        metrics="$(curl -fsS --max-time 3 "http://127.0.0.1:${host_port}/metrics" 2>/dev/null || true)"
      elif command -v wget >/dev/null 2>&1; then
        metrics="$(wget -q -T 3 -O - "http://127.0.0.1:${host_port}/metrics" 2>/dev/null || true)"
      fi
    fi

    block_requests_total="$(metric_sum "$metrics" "block_requests_total")"
    block_requests_retries="$(metric_sum "$metrics" "block_requests_retries")"
    dep_tracking_size="$(metric_sum "$metrics" "block_retriever_dep_recovery_tracking_size")"
    broadcast_tracking_size="$(metric_sum "$metrics" "block_retriever_broadcast_tracking_size")"
    active_validators_cache_size="$(metric_sum "$metrics" "active_validators_cache_size")"
    retry_ratio="$(awk -v t="$block_requests_total" -v r="$block_requests_retries" 'BEGIN { if (t > 0) printf "%.6f", r/t; else print "0.000000" }')"

    echo "${timestamp_utc},${elapsed},${service},${rss_mib},${block_requests_total},${block_requests_retries},${retry_ratio},${dep_tracking_size},${broadcast_tracking_size},${active_validators_cache_size}" >>"$SAMPLES_CSV"
  done

  sleep "$SAMPLE_EVERY_SECONDS"
done

awk -F, '
  NR == 1 { next }
  {
    svc = $3
    t = $2 + 0
    rss = $4 + 0
    dep = $8 + 0
    brd = $9 + 0
    avc = $10 + 0

    if (!(svc in start_t)) {
      start_t[svc] = t
      start_rss[svc] = rss
      start_dep[svc] = dep
      start_brd[svc] = brd
      start_avc[svc] = avc
    }
    end_t[svc] = t
    end_rss[svc] = rss
    end_dep[svc] = dep
    end_brd[svc] = brd
    end_avc[svc] = avc
    count[svc]++
  }
  END {
    print "Validator leak soak summary"
    for (svc in count) {
      dt = end_t[svc] - start_t[svc]
      if (dt <= 0) dt = 1
      rss_delta = end_rss[svc] - start_rss[svc]
      dep_delta = end_dep[svc] - start_dep[svc]
      brd_delta = end_brd[svc] - start_brd[svc]
      avc_delta = end_avc[svc] - start_avc[svc]
      rss_slope = rss_delta / dt

      printf "%s: samples=%d, elapsed_s=%.0f, rss_start_mib=%.3f, rss_end_mib=%.3f, rss_delta_mib=%.3f, rss_slope_mib_per_s=%.6f, dep_tracking_delta=%.0f, broadcast_tracking_delta=%.0f, active_validators_cache_delta=%.0f\n",
        svc, count[svc], dt, start_rss[svc], end_rss[svc], rss_delta, rss_slope, dep_delta, brd_delta, avc_delta
    }
  }
' "$SAMPLES_CSV" >"$SUMMARY_TXT"

echo "Leak soak complete"
echo "  samples: $SAMPLES_CSV"
echo "  summary: $SUMMARY_TXT"
cat "$SUMMARY_TXT"
