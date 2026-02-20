#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${1:-docker/shard-with-autopropose.yml}"
OUT_DIR="${2:-/tmp/casper-latency-profile-$(date +%Y%m%d-%H%M%S)}"
SERVICES=(validator1 validator2 validator3)

mkdir -p "$OUT_DIR"

if docker compose -f "$COMPOSE_FILE" ps >/dev/null 2>&1; then
  DC="docker compose -f $COMPOSE_FILE"
elif docker-compose -f "$COMPOSE_FILE" ps >/dev/null 2>&1; then
  DC="docker-compose -f $COMPOSE_FILE"
else
  echo "Unable to access compose services using $COMPOSE_FILE" >&2
  exit 2
fi

LOG_FILE="$OUT_DIR/compose.log"
$DC logs --no-color >"$LOG_FILE" || true

extract_ms_series() {
  local target="$1"
  local key="$2"
  local out_file="$3"
  rg "\"target\":\"${target}\"" "$LOG_FILE" | rg -o "${key}=[0-9]+" | cut -d= -f2 >"$out_file" || true
}

summarize_series() {
  local label="$1"
  local file="$2"
  local sorted="$OUT_DIR/.tmp-${label// /_}.sorted"

  if [[ ! -s "$file" ]]; then
    echo "${label}: n=0"
    return 0
  fi

  sort -n "$file" >"$sorted"
  local n sum avg p50 p95 p50_idx p95_idx
  n="$(wc -l <"$sorted" | tr -d ' ')"
  sum="$(awk '{s+=$1} END{printf "%.3f", s+0}' "$sorted")"
  avg="$(awk -v n="$n" -v sum="$sum" 'BEGIN{if(n>0) printf "%.2f", sum/n; else print "0.00"}')"
  p50_idx="$(awk -v n="$n" 'BEGIN{print int((n-1)*0.50)+1}')"
  p95_idx="$(awk -v n="$n" 'BEGIN{print int((n-1)*0.95)+1}')"
  p50="$(sed -n "${p50_idx}p" "$sorted")"
  p95="$(sed -n "${p95_idx}p" "$sorted")"

  echo "${label}: n=${n}, avg_ms=${avg}, p50_ms=${p50}, p95_ms=${p95}"
}

metric_value_sum() {
  local metric_name="$1"
  local total=0
  for service in "${SERVICES[@]}"; do
    local metrics_file="$OUT_DIR/${service}.metrics.prom"
    if [[ ! -f "$metrics_file" ]]; then
      continue
    fi
    local v
    v="$(awk -v metric_name="$metric_name" '
      index($0, "#") == 1 { next }
      $1 ~ ("^" metric_name "({.*})?$") { s += $2; f = 1 }
      END { if (f) printf "%.10f\n", s; else print "0" }
    ' "$metrics_file")"
    total="$(awk -v a="$total" -v b="$v" 'BEGIN{printf "%.10f", a+b}')"
  done
  printf "%s\n" "$total"
}

collect_metrics() {
  local service="$1"
  local cid
  cid="$($DC ps -q "$service" || true)"
  if [[ -z "$cid" ]]; then
    return 0
  fi
  local host_port
  host_port="$(docker port "$cid" 40403/tcp 2>/dev/null | awk -F: 'NR==1 {print $NF}')"
  if [[ -z "$host_port" ]]; then
    return 0
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 3 "http://127.0.0.1:${host_port}/metrics" >"$OUT_DIR/${service}.metrics.prom" 2>/dev/null || true
  elif command -v wget >/dev/null 2>&1; then
    wget -q -T 3 -O "$OUT_DIR/${service}.metrics.prom" "http://127.0.0.1:${host_port}/metrics" 2>/dev/null || true
  fi
}

for service in "${SERVICES[@]}"; do
  collect_metrics "$service"
done

extract_ms_series "f1r3fly.propose.timing" "total_ms" "$OUT_DIR/propose_total_ms.txt"
extract_ms_series "f1r3fly.block_creator.timing" "total_create_block_ms" "$OUT_DIR/block_creator_total_ms.txt"
extract_ms_series "f1r3fly.block_creator.timing" "compute_deploys_checkpoint_ms" "$OUT_DIR/block_creator_checkpoint_ms.txt"
extract_ms_series "f1r3fly.finalizer.timing" "total_ms" "$OUT_DIR/finalizer_total_ms.txt"

block_validation_sum="$(metric_value_sum "block_validation_time_sum")"
block_validation_count="$(metric_value_sum "block_validation_time_count")"
block_replay_sum="$(metric_value_sum "block_processing_stage_replay_time_sum")"
block_replay_count="$(metric_value_sum "block_processing_stage_replay_time_count")"
requests_total="$(metric_value_sum "block_requests_total")"
requests_retries="$(metric_value_sum "block_requests_retries")"

validation_mean_ms="$(awk -v s="$block_validation_sum" -v c="$block_validation_count" 'BEGIN{if(c>0) printf "%.2f", (s/c)*1000; else print "0.00"}')"
replay_mean_ms="$(awk -v s="$block_replay_sum" -v c="$block_replay_count" 'BEGIN{if(c>0) printf "%.2f", (s/c)*1000; else print "0.00"}')"
retry_ratio="$(awk -v r="$requests_retries" -v t="$requests_total" 'BEGIN{if(t>0) printf "%.2f", r/t; else print "0.00"}')"

SUMMARY_FILE="$OUT_DIR/summary.txt"
{
  echo "Casper latency profile"
  echo "generated_at_utc: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "compose_file: $COMPOSE_FILE"
  echo
  summarize_series "propose_total" "$OUT_DIR/propose_total_ms.txt"
  summarize_series "block_creator_total_create_block" "$OUT_DIR/block_creator_total_ms.txt"
  summarize_series "block_creator_compute_deploys_checkpoint" "$OUT_DIR/block_creator_checkpoint_ms.txt"
  summarize_series "finalizer_total" "$OUT_DIR/finalizer_total_ms.txt"
  echo
  echo "block_validation_mean_ms: $validation_mean_ms (sum_s=$block_validation_sum, count=$block_validation_count)"
  echo "block_replay_mean_ms: $replay_mean_ms (sum_s=$block_replay_sum, count=$block_replay_count)"
  echo "block_requests_total: $requests_total"
  echo "block_requests_retries: $requests_retries"
  echo "block_requests_retry_ratio: $retry_ratio"
} >"$SUMMARY_FILE"

cat "$SUMMARY_FILE"
echo
echo "Artifacts written to $OUT_DIR"
