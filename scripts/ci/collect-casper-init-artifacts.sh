#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${1:-docker/shard-with-autopropose.yml}"
OUT_DIR="${2:-/tmp/casper-init-artifacts}"
SERVICES=(boot validator1 validator2 validator3 readonly)
declare -A SERVICE_CIDS=()

mkdir -p "$OUT_DIR"

if docker compose -f "$COMPOSE_FILE" ps >/dev/null 2>&1; then
  DC="docker compose -f $COMPOSE_FILE"
elif docker-compose -f "$COMPOSE_FILE" ps >/dev/null 2>&1; then
  DC="docker-compose -f $COMPOSE_FILE"
else
  echo "Unable to access compose services using $COMPOSE_FILE" >&2
  exit 2
fi

echo "Collecting Casper init artifacts into $OUT_DIR"

$DC ps >"$OUT_DIR/compose-ps.txt" || true
$DC logs --no-color >"$OUT_DIR/compose.log" || true

sum_metric_from_file() {
  local file="$1"
  local metric_name="$2"
  if [[ ! -f "$file" ]]; then
    echo "n/a"
    return 0
  fi
  awk -v metric_name="$metric_name" '
    index($0, "#") == 1 { next }
    $1 ~ ("^" metric_name "({.*})?$") {
      sum += $2
      found = 1
    }
    END {
      if (found) {
        printf "%.0f\n", sum
      } else {
        print "absent"
      }
    }
  ' "$file"
}

fetch_metrics_via_container() {
  local cid="$1"
  docker exec "$cid" sh -lc '
    if command -v curl >/dev/null 2>&1; then
      curl -fsS --max-time 3 http://127.0.0.1:40403/metrics || true
    elif command -v wget >/dev/null 2>&1; then
      wget -q -T 3 -O - http://127.0.0.1:40403/metrics || true
    fi
  ' 2>/dev/null || true
}

fetch_metrics_via_host() {
  local cid="$1"
  local host_port
  host_port="$(docker port "$cid" 40403/tcp 2>/dev/null | awk -F: 'NR==1 {print $NF}')"
  if [[ -z "$host_port" ]]; then
    return 0
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 3 "http://127.0.0.1:${host_port}/metrics" 2>/dev/null || true
  elif command -v wget >/dev/null 2>&1; then
    wget -q -T 3 -O - "http://127.0.0.1:${host_port}/metrics" 2>/dev/null || true
  fi
}

for service in "${SERVICES[@]}"; do
  cid="$($DC ps -q "$service" || true)"
  if [[ -z "$cid" ]]; then
    echo "Service $service not running" >"$OUT_DIR/${service}.status.txt"
    continue
  fi
  SERVICE_CIDS["$service"]="$cid"

  docker inspect "$cid" >"$OUT_DIR/${service}.inspect.json" || true
  docker logs "$cid" >"$OUT_DIR/${service}.log" 2>&1 || true

  metrics="$(fetch_metrics_via_container "$cid")"
  if [[ -z "$metrics" ]]; then
    metrics="$(fetch_metrics_via_host "$cid")"
  fi
  if [[ -n "$metrics" ]]; then
    printf "%s\n" "$metrics" >"$OUT_DIR/${service}.metrics.prom"
  else
    echo "Metrics unavailable for $service" >"$OUT_DIR/${service}.metrics.error.txt"
  fi
done

summary_file="$OUT_DIR/summary.txt"
{
  echo "Casper init artifact summary"
  echo "generated_at_utc: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "compose_file: $COMPOSE_FILE"
  echo
} >"$summary_file"

for service in "${SERVICES[@]}"; do
  status="not_running"
  started_at="n/a"
  running_line="n/a"

  if [[ -f "$OUT_DIR/${service}.inspect.json" ]]; then
    status="running"
    cid="${SERVICE_CIDS[$service]:-}"
    started_at="$(docker inspect -f '{{.State.StartedAt}}' "$cid" 2>/dev/null || true)"
    if [[ -z "$started_at" ]]; then
      started_at="n/a"
    fi
  fi

  if [[ -f "$OUT_DIR/${service}.log" ]]; then
    running_line="$(grep -m1 'Making a transition to Running state' "$OUT_DIR/${service}.log" || true)"
    if [[ -z "$running_line" ]]; then
      running_line="not_found"
    fi
  fi

  attempts="$(sum_metric_from_file "$OUT_DIR/${service}.metrics.prom" "casper_init_attempts")"
  retries="$(sum_metric_from_file "$OUT_DIR/${service}.metrics.prom" "casper_init_retry_no_approved_block")"
  approved="$(sum_metric_from_file "$OUT_DIR/${service}.metrics.prom" "casper_init_approved_block_received")"
  transitions="$(sum_metric_from_file "$OUT_DIR/${service}.metrics.prom" "casper_init_transition_to_running")"
  ttr_count="$(sum_metric_from_file "$OUT_DIR/${service}.metrics.prom" "casper_init_time_to_running_count")"
  ttab_count="$(sum_metric_from_file "$OUT_DIR/${service}.metrics.prom" "casper_init_time_to_approved_block_count")"
  validator_gate="n/a"
  if [[ "$service" == validator1 || "$service" == validator2 || "$service" == validator3 ]]; then
    validator_gate="FAIL"
    # Keep artifact gating consistent with check-casper-init-sla.sh:
    # direct-to-running validators may not expose Initializing-specific metrics.
    if [[ "$transitions" != "absent" ]] && (( transitions >= 1 )); then
      if [[ "$attempts" == "absent" && "$approved" == "absent" && "$ttr_count" == "absent" ]]; then
        validator_gate="PASS_DIRECT_RUNNING"
      elif [[ "$attempts" != "absent" && "$approved" != "absent" && "$ttr_count" != "absent" ]] \
        && (( attempts >= 1 && approved >= 1 && ttr_count >= 1 )); then
        validator_gate="PASS"
      fi
    fi
  fi

  {
    echo "service: $service"
    echo "  container_status: $status"
    echo "  container_started_at: $started_at"
    echo "  running_transition_log: $running_line"
    echo "  metric.casper_init_attempts: $attempts"
    echo "  metric.casper_init_retry_no_approved_block: $retries"
    echo "  metric.casper_init_approved_block_received: $approved"
    echo "  metric.casper_init_transition_to_running: $transitions"
    echo "  metric.casper_init_time_to_running_count: $ttr_count"
    echo "  metric.casper_init_time_to_approved_block_count: $ttab_count"
    echo "  validator_init_gate: $validator_gate"
    echo
  } >>"$summary_file"
done

echo "Artifacts collected in $OUT_DIR"
