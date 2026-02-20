#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${1:-docker/shard-with-autopropose.yml}"
OUT_DIR="${2:-/tmp/casper-latency-profile-$(date +%Y%m%d-%H%M%S)}"
LOG_SINCE_UTC="${3:-}"
BASELINE_METRICS_DIR="${4:-}"
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
if [[ -n "$LOG_SINCE_UTC" ]]; then
  $DC logs --no-color --since "$LOG_SINCE_UTC" >"$LOG_FILE" || true
else
  $DC logs --no-color >"$LOG_FILE" || true
fi

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

summarize_top_replay_retry_hashes() {
  local out_file="$OUT_DIR/replay-retry-top.txt"
  rg "Replay block [0-9a-f]{64} got error .*retries details: will retry" "$LOG_FILE" \
    | rg -o "Replay block [0-9a-f]{64}" \
    | awk '{print $3}' \
    | sort \
    | uniq -c \
    | sort -nr \
    | head -n 5 \
    | awk '{printf "%s:%s\n", $2, $1}' >"$out_file" || true

  if [[ -s "$out_file" ]]; then
    echo "replay_retry_top_hashes:"
    cat "$out_file"
  else
    echo "replay_retry_top_hashes: none"
  fi
}

metric_value_sum() {
  local metric_name="$1"
  local metrics_dir="$2"
  local total=0
  for service in "${SERVICES[@]}"; do
    local metrics_file="$metrics_dir/${service}.metrics.prom"
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

metric_delta() {
  local current="$1"
  local baseline="$2"
  awk -v c="$current" -v b="$baseline" 'BEGIN { d = c - b; if (d < 0) d = 0; printf "%.10f", d }'
}

metric_value_sum_with_label() {
  local metric_name="$1"
  local label_name="$2"
  local label_value="$3"
  local metrics_dir="$4"
  local total=0
  for service in "${SERVICES[@]}"; do
    local metrics_file="$metrics_dir/${service}.metrics.prom"
    if [[ ! -f "$metrics_file" ]]; then
      continue
    fi
    local v
    v="$(awk -v metric_name="$metric_name" -v label_name="$label_name" -v label_value="$label_value" '
      index($0, "#") == 1 { next }
      $1 ~ ("^" metric_name "({.*})?$") {
        if ($1 ~ (".*" label_name "=\"" label_value "\".*")) {
          s += $2
          f = 1
        }
      }
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

block_validation_sum_current="$(metric_value_sum "block_validation_time_sum" "$OUT_DIR")"
block_validation_count_current="$(metric_value_sum "block_validation_time_count" "$OUT_DIR")"
block_replay_sum_current="$(metric_value_sum "block_processing_stage_replay_time_sum" "$OUT_DIR")"
block_replay_count_current="$(metric_value_sum "block_processing_stage_replay_time_count" "$OUT_DIR")"
requests_total_current="$(metric_value_sum "block_requests_total" "$OUT_DIR")"
requests_retries_current="$(metric_value_sum "block_requests_retries" "$OUT_DIR")"

block_validation_sum_base=0
block_validation_count_base=0
block_replay_sum_base=0
block_replay_count_base=0
requests_total_base=0
requests_retries_base=0
if [[ -n "$BASELINE_METRICS_DIR" ]]; then
  block_validation_sum_base="$(metric_value_sum "block_validation_time_sum" "$BASELINE_METRICS_DIR")"
  block_validation_count_base="$(metric_value_sum "block_validation_time_count" "$BASELINE_METRICS_DIR")"
  block_replay_sum_base="$(metric_value_sum "block_processing_stage_replay_time_sum" "$BASELINE_METRICS_DIR")"
  block_replay_count_base="$(metric_value_sum "block_processing_stage_replay_time_count" "$BASELINE_METRICS_DIR")"
  requests_total_base="$(metric_value_sum "block_requests_total" "$BASELINE_METRICS_DIR")"
  requests_retries_base="$(metric_value_sum "block_requests_retries" "$BASELINE_METRICS_DIR")"
fi

block_validation_sum="$(metric_delta "$block_validation_sum_current" "$block_validation_sum_base")"
block_validation_count="$(metric_delta "$block_validation_count_current" "$block_validation_count_base")"
block_replay_sum="$(metric_delta "$block_replay_sum_current" "$block_replay_sum_base")"
block_replay_count="$(metric_delta "$block_replay_count_current" "$block_replay_count_base")"
requests_total="$(metric_delta "$requests_total_current" "$requests_total_base")"
requests_retries="$(metric_delta "$requests_retries_current" "$requests_retries_base")"

retry_action_peer_current="$(metric_value_sum_with_label "block_requests_retry_action" "action" "peer_request" "$OUT_DIR")"
retry_action_peer_requery_current="$(metric_value_sum_with_label "block_requests_retry_action" "action" "peer_requery" "$OUT_DIR")"
retry_action_broadcast_current="$(metric_value_sum_with_label "block_requests_retry_action" "action" "broadcast_only" "$OUT_DIR")"
retry_action_none_current="$(metric_value_sum_with_label "block_requests_retry_action" "action" "none" "$OUT_DIR")"
retry_action_broadcast_suppressed_current="$(metric_value_sum_with_label "block_requests_retry_action" "action" "broadcast_suppressed" "$OUT_DIR")"
retry_action_peer_base=0
retry_action_peer_requery_base=0
retry_action_broadcast_base=0
retry_action_none_base=0
retry_action_broadcast_suppressed_base=0
if [[ -n "$BASELINE_METRICS_DIR" ]]; then
  retry_action_peer_base="$(metric_value_sum_with_label "block_requests_retry_action" "action" "peer_request" "$BASELINE_METRICS_DIR")"
  retry_action_peer_requery_base="$(metric_value_sum_with_label "block_requests_retry_action" "action" "peer_requery" "$BASELINE_METRICS_DIR")"
  retry_action_broadcast_base="$(metric_value_sum_with_label "block_requests_retry_action" "action" "broadcast_only" "$BASELINE_METRICS_DIR")"
  retry_action_none_base="$(metric_value_sum_with_label "block_requests_retry_action" "action" "none" "$BASELINE_METRICS_DIR")"
  retry_action_broadcast_suppressed_base="$(metric_value_sum_with_label "block_requests_retry_action" "action" "broadcast_suppressed" "$BASELINE_METRICS_DIR")"
fi
retry_action_peer="$(metric_delta "$retry_action_peer_current" "$retry_action_peer_base")"
retry_action_peer_requery="$(metric_delta "$retry_action_peer_requery_current" "$retry_action_peer_requery_base")"
retry_action_broadcast="$(metric_delta "$retry_action_broadcast_current" "$retry_action_broadcast_base")"
retry_action_none="$(metric_delta "$retry_action_none_current" "$retry_action_none_base")"
retry_action_broadcast_suppressed="$(metric_delta "$retry_action_broadcast_suppressed_current" "$retry_action_broadcast_suppressed_base")"

validation_mean_ms="$(awk -v s="$block_validation_sum" -v c="$block_validation_count" 'BEGIN{if(c>0) printf "%.2f", (s/c)*1000; else print "0.00"}')"
replay_mean_ms="$(awk -v s="$block_replay_sum" -v c="$block_replay_count" 'BEGIN{if(c>0) printf "%.2f", (s/c)*1000; else print "0.00"}')"
retry_ratio="$(awk -v r="$requests_retries" -v t="$requests_total" 'BEGIN{if(t>0) printf "%.2f", r/t; else print "0.00"}')"

SUMMARY_FILE="$OUT_DIR/summary.txt"
{
  echo "Casper latency profile"
  echo "generated_at_utc: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "compose_file: $COMPOSE_FILE"
  if [[ -n "$LOG_SINCE_UTC" ]]; then
    echo "log_since_utc: $LOG_SINCE_UTC"
  fi
  if [[ -n "$BASELINE_METRICS_DIR" ]]; then
    echo "metrics_mode: delta_from_baseline"
    echo "baseline_metrics_dir: $BASELINE_METRICS_DIR"
  fi
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
  echo "block_requests_retry_action_peer_request: $retry_action_peer"
  echo "block_requests_retry_action_peer_requery: $retry_action_peer_requery"
  echo "block_requests_retry_action_broadcast_only: $retry_action_broadcast"
  echo "block_requests_retry_action_none: $retry_action_none"
  echo "block_requests_retry_action_broadcast_suppressed: $retry_action_broadcast_suppressed"
  summarize_top_replay_retry_hashes
} >"$SUMMARY_FILE"

cat "$SUMMARY_FILE"
echo
echo "Artifacts written to $OUT_DIR"
