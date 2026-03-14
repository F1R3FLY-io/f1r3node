#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${1:-docker/shard-with-autopropose.yml}"
SLA_SECONDS="${2:-180}"
POLL_SECONDS="${POLL_SECONDS:-5}"
METRICS_CHECK_SECONDS="${METRICS_CHECK_SECONDS:-30}"
CASPER_INIT_REQUIRE_GENESIS_FINALIZED="${CASPER_INIT_REQUIRE_GENESIS_FINALIZED:-1}"
SERVICES=(validator1 validator2 validator3)
declare -A SERVICE_METRICS_OK=()

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 2
fi

if ! docker compose -f "$COMPOSE_FILE" ps >/dev/null 2>&1; then
  echo "Unable to access compose services using $COMPOSE_FILE" >&2
  exit 2
fi

start_epoch="$(date +%s)"
deadline=$((start_epoch + SLA_SECONDS))

echo "Checking Casper init SLA ($SLA_SECONDS s) for services: ${SERVICES[*]}"
echo "Compose file: $COMPOSE_FILE"

declare -A SERVICE_CIDS=()
declare -A SERVICE_STARTED_AT=()
for service in "${SERVICES[@]}"; do
  cid="$(docker compose -f "$COMPOSE_FILE" ps -q "$service" || true)"
  if [[ -z "$cid" ]]; then
    echo "Service $service is not running (no container id)" >&2
    exit 1
  fi
  SERVICE_CIDS["$service"]="$cid"
  started_at="$(docker inspect -f '{{.State.StartedAt}}' "$cid")"
  SERVICE_STARTED_AT["$service"]="$started_at"
done

running_map=""
for service in "${SERVICES[@]}"; do
  running_map+="$service=0 "
done

is_done() {
  local service="$1"
  grep -q "\b${service}=1\b" <<<"$running_map"
}

fetch_validator_api_port() {
  local cid="$1"
  local host_port
  host_port="$(docker port "$cid" 40403/tcp 2>/dev/null | awk -F: 'NR==1 {print $NF}')"
  echo "$host_port"
}

fetch_last_finalized_block() {
  local cid="$1"
  local host_port
  host_port="$(fetch_validator_api_port "$cid")"
  if [[ -z "$host_port" ]]; then
    return 1
  fi

  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 3 "http://127.0.0.1:${host_port}/api/last-finalized-block" || return 1
  elif command -v wget >/dev/null 2>&1; then
    wget -q -T 3 -O - "http://127.0.0.1:${host_port}/api/last-finalized-block" || return 1
  else
    return 1
  fi
}

parse_last_finalized_seq() {
  local payload="$1"
  awk 'match($0, /"seqNum"[[:space:]]*:[[:space:]]*([0-9]+)/, m) { print m[1]; exit }
       match($0, /"blockNumber"[[:space:]]*:[[:space:]]*([0-9]+)/, m) { print m[1]; exit }' <<<"$payload"
}

mark_done() {
  local service="$1"
  running_map="$(sed -E "s/\b${service}=0\b/${service}=1/g" <<<"$running_map")"
}

all_done() {
  for service in "${SERVICES[@]}"; do
    if ! is_done "$service"; then
      return 1
    fi
  done
  return 0
}

sum_metric_from_text() {
  local metrics_text="$1"
  local metric_name="$2"
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
        print "0"
      }
    }
  ' <<<"$metrics_text"
}

metric_exists_in_text() {
  local metrics_text="$1"
  local metric_name="$2"
  awk -v metric_name="$metric_name" '
    index($0, "#") == 1 { next }
    $1 ~ ("^" metric_name "({.*})?$") {
      found = 1
    }
    END {
      if (found) print "1"; else print "0"
    }
  ' <<<"$metrics_text"
}

check_service_metrics() {
  local service="$1"
  local cid="$2"

  local host_port
  host_port="$(docker port "$cid" 40403/tcp 2>/dev/null | awk -F: 'NR==1 {print $NF}')"
  if [[ -z "$host_port" ]]; then
    echo "No mapped Prometheus port for $service (expected container 40403/tcp)." >&2
    return 1
  fi

  local metrics_url="http://127.0.0.1:${host_port}/metrics"
  local metrics
  if command -v curl >/dev/null 2>&1; then
    metrics="$(curl -fsS --max-time 3 "$metrics_url" 2>/dev/null || true)"
  elif command -v wget >/dev/null 2>&1; then
    metrics="$(wget -q -T 3 -O - "$metrics_url" 2>/dev/null || true)"
  else
    echo "Neither curl nor wget is available to fetch Prometheus metrics." >&2
    return 1
  fi
  if [[ -z "$metrics" ]]; then
    echo "Unable to fetch metrics for $service from $metrics_url" >&2
    return 1
  fi

  local attempts
  local approved
  local transitions
  local time_to_running_count
  local attempts_present
  local approved_present
  local time_to_running_count_present
  attempts="$(sum_metric_from_text "$metrics" "casper_init_attempts")"
  approved="$(sum_metric_from_text "$metrics" "casper_init_approved_block_received")"
  transitions="$(sum_metric_from_text "$metrics" "casper_init_transition_to_running")"
  time_to_running_count="$(sum_metric_from_text "$metrics" "casper_init_time_to_running_count")"
  attempts_present="$(metric_exists_in_text "$metrics" "casper_init_attempts")"
  approved_present="$(metric_exists_in_text "$metrics" "casper_init_approved_block_received")"
  time_to_running_count_present="$(metric_exists_in_text "$metrics" "casper_init_time_to_running_count")"

  # Validators may legitimately go directly to Running on startup from an approved genesis block
  # without entering Initializing. In that path, init-attempt/approved/time-to-running metrics are
  # absent. If any of those metrics are present, require all of them to be valid.
  if (( attempts_present == 0 && approved_present == 0 && time_to_running_count_present == 0 )); then
    echo "$service metrics OK: transition_to_running=$transitions (direct-to-running path; initializing metrics not exposed)"
    return 0
  fi

  if (( attempts_present == 0 || approved_present == 0 || time_to_running_count_present == 0 )); then
    echo "Metric check failed for $service: partial init metric exposure (attempts_present=$attempts_present, approved_present=$approved_present, time_to_running_count_present=$time_to_running_count_present)." >&2
    return 1
  fi
  if (( attempts < 1 )); then
    echo "Metric check failed for $service: casper_init_attempts=$attempts (expected >=1)." >&2
    return 1
  fi
  if (( approved < 1 )); then
    echo "Metric check failed for $service: casper_init_approved_block_received=$approved (expected >=1)." >&2
    return 1
  fi
  if (( time_to_running_count < 1 )); then
    echo "Metric check failed for $service: casper_init_time_to_running_count=$time_to_running_count (expected >=1)." >&2
    return 1
  fi

  if (( transitions < 1 )); then
    echo "Metric check failed for $service: casper_init_transition_to_running=$transitions (expected >=1)." >&2
    return 1
  fi

  echo "$service metrics OK: attempts=$attempts, approved=$approved, transitions=$transitions, time_to_running_count=$time_to_running_count"
  return 0
}

while true; do
  now="$(date +%s)"
  if (( now > deadline )); then
    echo "SLA FAILED: validators did not all reach Running within ${SLA_SECONDS}s" >&2
    for service in "${SERVICES[@]}"; do
      if ! is_done "$service"; then
        echo "--- ${service} recent init logs ---" >&2
        docker logs "${SERVICE_CIDS[$service]}" --since "${SERVICE_STARTED_AT[$service]}" 2>&1 \
          | tail -n 200 \
          | grep -En "No approved block available|transition to Running|Casper engine present but Casper not initialized yet|Making a transition to Initializing state" || true
      fi
    done
    exit 1
  fi

  for service in "${SERVICES[@]}"; do
    if is_done "$service"; then
      continue
    fi

    logs="$(docker logs "${SERVICE_CIDS[$service]}" --since "${SERVICE_STARTED_AT[$service]}" 2>&1 || true)"
    if grep -q -E "Making a transition to Running state|GenesisValidator: engine transitioned" <<<"$logs"; then
      mark_done "$service"
      elapsed=$((now - start_epoch))
      echo "${service} reached Running in ${elapsed}s"
    fi
  done

  if all_done; then
    total=$((now - start_epoch))
    metrics_deadline=$((now + METRICS_CHECK_SECONDS))
    echo "Running-state SLA met in ${total}s; validating init metrics and genesis finality (timeout ${METRICS_CHECK_SECONDS}s)..."

    while true; do
      metrics_pending=0
      for service in "${SERVICES[@]}"; do
        if [[ "${SERVICE_METRICS_OK[$service]:-0}" == "1" ]]; then
          continue
        fi
        if check_service_metrics "$service" "${SERVICE_CIDS[$service]}"; then
          SERVICE_METRICS_OK["$service"]="1"
        else
          metrics_pending=1
        fi
      done

      if (( metrics_pending == 0 )); then
      if (( CASPER_INIT_REQUIRE_GENESIS_FINALIZED > 0 )); then
          finalized_ok=1
          while true; do
            if (( $(date +%s) > metrics_deadline )); then
              echo "SLA FAILED: validators reached Running and exposed init metrics, but genesis finality threshold (${CASPER_INIT_REQUIRE_GENESIS_FINALIZED}) was not met in time." >&2
              for service in "${SERVICES[@]}"; do
                if ! is_done "$service"; then
                  continue
                fi
                latest_finalized=
                latest_finalized="$(fetch_last_finalized_block "${SERVICE_CIDS[$service]}" 2>/dev/null || echo "")"
                echo "${service} last-finalized endpoint sample: ${latest_finalized:-<unavailable>}"
              done
              exit 1
            fi

            finalized_ok=1
            for service in "${SERVICES[@]}"; do
              last_finalized=
              block_number=
              if ! last_finalized="$(fetch_last_finalized_block "${SERVICE_CIDS[$service]}" 2>/dev/null || true)"; then
                finalized_ok=0
                break
              fi
              block_number="$(parse_last_finalized_seq "$last_finalized")"
              if [[ -z "$block_number" ]] || (( block_number < CASPER_INIT_REQUIRE_GENESIS_FINALIZED )); then
                finalized_ok=0
                break
              fi
            done

            if (( finalized_ok == 1 )); then
              break
            fi
            sleep "$POLL_SECONDS"
          done
        fi

        echo "SLA PASSED: all validators reached Running, exposed init metrics and satisfied genesis-finality threshold (${CASPER_INIT_REQUIRE_GENESIS_FINALIZED})."
        exit 0
      fi

      now="$(date +%s)"
      if (( now > metrics_deadline )); then
        echo "SLA FAILED: validators reached Running, but init metrics were incomplete." >&2
        exit 1
      fi

      sleep "$POLL_SECONDS"
    done
  fi

  sleep "$POLL_SECONDS"
done
