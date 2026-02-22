#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${1:-docker/shard-with-autopropose.yml}"
DURATION_SECONDS="${2:-600}"
SAMPLE_EVERY_SECONDS="${3:-10}"
OUT_DIR="${4:-/tmp/casper-validator-leak-soak-$(date -u +%Y%m%dT%H%M%SZ)}"
SOAK_RESTART_CLEAN="${SOAK_RESTART_CLEAN:-0}"
SOAK_CLEAN_DATA_DIR="${SOAK_CLEAN_DATA_DIR:-}"
SOAK_WARMUP_SECONDS="${SOAK_WARMUP_SECONDS:-20}"
SOAK_WAIT_FOR_READY="${SOAK_WAIT_FOR_READY:-1}"
SOAK_READY_TIMEOUT_SECONDS="${SOAK_READY_TIMEOUT_SECONDS:-180}"
SOAK_PROFILE_PROC="${SOAK_PROFILE_PROC:-0}"
SOAK_PROC_SAMPLE_EVERY_SECONDS="${SOAK_PROC_SAMPLE_EVERY_SECONDS:-10}"
SOAK_PROFILE_FINALIZER="${SOAK_PROFILE_FINALIZER:-1}"

SERVICES=(validator1 validator2 validator3)

mkdir -p "$OUT_DIR"
SAMPLES_CSV="$OUT_DIR/samples.csv"
SUMMARY_TXT="$OUT_DIR/summary.txt"
PROC_SUMMARY_TXT="$OUT_DIR/proc-summary.txt"
FINALIZER_SUMMARY_TXT="$OUT_DIR/finalizer-summary.txt"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 2
fi

if ! docker compose -f "$COMPOSE_FILE" ps >/dev/null 2>&1; then
  echo "Unable to access compose services using $COMPOSE_FILE" >&2
  exit 2
fi

if [[ "$SOAK_RESTART_CLEAN" == "1" ]]; then
  echo "Clean restart requested (SOAK_RESTART_CLEAN=1)"
  docker compose -f "$COMPOSE_FILE" down

  if [[ -n "$SOAK_CLEAN_DATA_DIR" ]]; then
    if [[ ! -d "$SOAK_CLEAN_DATA_DIR" ]]; then
      echo "SOAK_CLEAN_DATA_DIR does not exist: $SOAK_CLEAN_DATA_DIR" >&2
      exit 2
    fi
    echo "Wiping bind-mounted data dir: $SOAK_CLEAN_DATA_DIR"
    docker run --rm -v "$SOAK_CLEAN_DATA_DIR:/data" alpine sh -lc 'rm -rf /data/*'
  fi

  docker compose -f "$COMPOSE_FILE" up -d
  sleep "$SOAK_WARMUP_SECONDS"
fi

echo "timestamp_utc,elapsed_s,service,rss_mib,block_requests_total,block_requests_retries,block_requests_retry_ratio,dep_tracking_size,broadcast_tracking_size,waiting_list_total_size,peers_total_size,stream_cache_entries,stream_cache_bytes,init_block_message_queue_pending,init_tuple_space_queue_pending,proposer_queue_pending,proposer_queue_rejected_total,runtime_soft_checkpoint_total,runtime_checkpoint_total,runtime_revert_soft_checkpoint_total,runtime_take_event_log_total,runtime_take_event_log_events_total,active_validators_cache_size,deploys_in_scope_size,deploys_in_scope_sig_bytes_estimate,block_index_cache_size,parents_post_state_cache_size,hot_store_history_continuations_cache_size,hot_store_history_data_cache_size,hot_store_history_joins_cache_size,hot_store_state_continuations_size,hot_store_state_data_size,hot_store_state_joins_size,hot_store_history_continuations_cache_items,hot_store_history_data_cache_items,hot_store_history_joins_cache_items,hot_store_state_continuations_items,hot_store_state_data_items,hot_store_state_joins_items,dag_blocks_size,dag_children_index_size,dag_heights_size,dag_finalized_blocks_size,casper_buffer_approx_nodes" >"$SAMPLES_CSV"

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

fetch_metrics_for_service() {
  local service="$1"
  local cid host_port
  cid="$(docker compose -f "$COMPOSE_FILE" ps -q "$service" || true)"
  if [[ -z "$cid" ]]; then
    return 1
  fi
  host_port="$(docker port "$cid" 40403/tcp 2>/dev/null | awk -F: 'NR==1 {print $NF}')"
  if [[ -z "$host_port" ]]; then
    return 1
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 3 "http://127.0.0.1:${host_port}/metrics" 2>/dev/null || return 1
  elif command -v wget >/dev/null 2>&1; then
    wget -q -T 3 -O - "http://127.0.0.1:${host_port}/metrics" 2>/dev/null || return 1
  else
    return 1
  fi
}

wait_for_validators_ready() {
  local deadline now metrics dag_blocks
  deadline=$(( $(date +%s) + SOAK_READY_TIMEOUT_SECONDS ))
  while true; do
    now="$(date +%s)"
    if (( now > deadline )); then
      echo "Timed out waiting for validator readiness after ${SOAK_READY_TIMEOUT_SECONDS}s" >&2
      return 1
    fi

    local ready_count=0
    for service in "${SERVICES[@]}"; do
      metrics="$(fetch_metrics_for_service "$service" || true)"
      if [[ -z "$metrics" ]]; then
        continue
      fi
      dag_blocks="$(metric_sum "$metrics" "dag_blocks_size")"
      if awk -v v="$dag_blocks" 'BEGIN{exit !(v > 0)}'; then
        ready_count=$((ready_count + 1))
      fi
    done

    if (( ready_count == ${#SERVICES[@]} )); then
      echo "All validators ready (dag_blocks_size > 0 for all services)"
      return 0
    fi
    sleep 2
  done
}

if [[ "$SOAK_WAIT_FOR_READY" == "1" ]]; then
  wait_for_validators_ready
fi

start_proc_samplers() {
  local duration="$1"
  local period="$2"
  local loops=$(( duration / period + 2 ))

  for service in "${SERVICES[@]}"; do
    (
      local csv="$OUT_DIR/${service}-proc.csv"
      echo "ts,elapsed_s,rss_kb,anonymous_kb,private_dirty_kb,file_approx_kb" >"$csv"
      local start now elapsed rss anon pd file_approx
      start="$(date +%s)"
      for _ in $(seq 1 "$loops"); do
        now="$(date +%s)"
        elapsed=$((now - start))
        rss="$(docker exec rnode.$service cat /proc/1/smaps_rollup 2>/dev/null | awk '/^Rss:/ {print $2; exit}' || echo 0)"
        anon="$(docker exec rnode.$service cat /proc/1/smaps_rollup 2>/dev/null | awk '/^Anonymous:/ {print $2; exit}' || echo 0)"
        pd="$(docker exec rnode.$service cat /proc/1/smaps_rollup 2>/dev/null | awk '/^Private_Dirty:/ {print $2; exit}' || echo 0)"
        rss="${rss:-0}"; anon="${anon:-0}"; pd="${pd:-0}"
        file_approx=$((rss - anon))
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$elapsed,$rss,$anon,$pd,$file_approx" >>"$csv"
        sleep "$period"
      done
    ) &
    PROC_SAMPLER_PIDS+=("$!")
  done
}

write_proc_summary() {
  {
    echo "Validator proc memory-class deltas"
    for service in "${SERVICES[@]}"; do
      local csv="$OUT_DIR/${service}-proc.csv"
      if [[ ! -f "$csv" ]]; then
        continue
      fi
      awk -F, -v v="$service" '
        NR==2 {s_r=$3; s_a=$4; s_pd=$5; s_f=$6; s_t=$2}
        NR>=2 {e_r=$3; e_a=$4; e_pd=$5; e_f=$6; e_t=$2}
        END {
          dt=e_t-s_t; if (dt<=0) dt=1;
          printf "%s: elapsed_s=%d, Rss_delta_kb=%d, Anonymous_delta_kb=%d, PrivateDirty_delta_kb=%d, FileApprox_delta_kb=%d\n",
            v, dt, e_r-s_r, e_a-s_a, e_pd-s_pd, e_f-s_f
        }
      ' "$csv"
    done
  } >"$PROC_SUMMARY_TXT"
}

count_log_pattern() {
  local service="$1"
  local since_utc="$2"
  local pattern="$3"
  local cnt
  if command -v timeout >/dev/null 2>&1; then
    cnt="$(timeout 20s docker logs --since "$since_utc" "rnode.$service" 2>&1 | grep -cE "$pattern" || true)"
  else
    cnt="$(docker logs --since "$since_utc" "rnode.$service" 2>&1 | grep -cE "$pattern" || true)"
  fi
  if [[ -z "$cnt" ]]; then
    cnt=0
  fi
  echo "$cnt"
}

write_finalizer_summary() {
  local since_utc="$1"
  {
    echo "Validator finalizer/log health summary"
    echo "since_utc=$since_utc"
    for service in "${SERVICES[@]}"; do
      local run_started run_finished lfb_true lfb_false filtered_zero skipped timed_out
      run_started="$(count_log_pattern "$service" "$since_utc" "finalizer-run-started")"
      run_finished="$(count_log_pattern "$service" "$since_utc" "finalizer-run-finished")"
      lfb_true="$(count_log_pattern "$service" "$since_utc" "new_lfb_found=true")"
      lfb_false="$(count_log_pattern "$service" "$since_utc" "new_lfb_found=false")"
      filtered_zero="$(count_log_pattern "$service" "$since_utc" "filtered_agreements=0")"
      skipped="$(count_log_pattern "$service" "$since_utc" "Skipping finalizer schedule")"
      timed_out="$(count_log_pattern "$service" "$since_utc" "finalizer-run timed out")"

      echo "$service: finalizer_run_started=$run_started, finalizer_run_finished=$run_finished, new_lfb_found_true=$lfb_true, new_lfb_found_false=$lfb_false, filtered_agreements_0=$filtered_zero, skipped_finalizer_schedule=$skipped, finalizer_run_timed_out=$timed_out"
    done
  } >"$FINALIZER_SUMMARY_TXT"
}

PROC_SAMPLER_PIDS=()
if [[ "$SOAK_PROFILE_PROC" == "1" ]]; then
  start_proc_samplers "$DURATION_SECONDS" "$SOAK_PROC_SAMPLE_EVERY_SECONDS"
fi

LOG_SINCE_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
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
    waiting_list_total_size="$(metric_sum "$metrics" "block_retriever_waiting_list_total_size")"
    peers_total_size="$(metric_sum "$metrics" "block_retriever_peers_total_size")"
    stream_cache_entries="$(metric_sum "$metrics" "stream_cache_entries")"
    stream_cache_bytes="$(metric_sum "$metrics" "stream_cache_bytes")"
    init_block_message_queue_pending="$(metric_sum "$metrics" "init_block_message_queue_pending")"
    init_tuple_space_queue_pending="$(metric_sum "$metrics" "init_tuple_space_queue_pending")"
    proposer_queue_pending="$(metric_sum "$metrics" "proposer_queue_pending")"
    proposer_queue_rejected_total="$(metric_sum "$metrics" "proposer_queue_rejected_total")"
    runtime_soft_checkpoint_total="$(metric_sum "$metrics" "runtime_soft_checkpoint_total")"
    runtime_checkpoint_total="$(metric_sum "$metrics" "runtime_checkpoint_total")"
    runtime_revert_soft_checkpoint_total="$(metric_sum "$metrics" "runtime_revert_soft_checkpoint_total")"
    runtime_take_event_log_total="$(metric_sum "$metrics" "runtime_take_event_log_total")"
    runtime_take_event_log_events_total="$(metric_sum "$metrics" "runtime_take_event_log_events_total")"
    active_validators_cache_size="$(metric_sum "$metrics" "active_validators_cache_size")"
    deploys_in_scope_size="$(metric_sum "$metrics" "deploys_in_scope_size")"
    deploys_in_scope_sig_bytes_estimate="$(metric_sum "$metrics" "deploys_in_scope_sig_bytes_estimate")"
    block_index_cache_size="$(metric_sum "$metrics" "block_index_cache_size")"
    parents_post_state_cache_size="$(metric_sum "$metrics" "parents_post_state_cache_size")"
    hot_store_history_cont_cache_size="$(metric_sum "$metrics" "hot_store_history_continuations_cache_size")"
    hot_store_history_data_cache_size="$(metric_sum "$metrics" "hot_store_history_data_cache_size")"
    hot_store_history_joins_cache_size="$(metric_sum "$metrics" "hot_store_history_joins_cache_size")"
    hot_store_state_cont_size="$(metric_sum "$metrics" "hot_store_state_continuations_size")"
    hot_store_state_data_size="$(metric_sum "$metrics" "hot_store_state_data_size")"
    hot_store_state_joins_size="$(metric_sum "$metrics" "hot_store_state_joins_size")"
    hot_store_history_cont_cache_items="$(metric_sum "$metrics" "hot_store_history_continuations_cache_items")"
    hot_store_history_data_cache_items="$(metric_sum "$metrics" "hot_store_history_data_cache_items")"
    hot_store_history_joins_cache_items="$(metric_sum "$metrics" "hot_store_history_joins_cache_items")"
    hot_store_state_cont_items="$(metric_sum "$metrics" "hot_store_state_continuations_items")"
    hot_store_state_data_items="$(metric_sum "$metrics" "hot_store_state_data_items")"
    hot_store_state_joins_items="$(metric_sum "$metrics" "hot_store_state_joins_items")"
    dag_blocks_size="$(metric_sum "$metrics" "dag_blocks_size")"
    dag_children_index_size="$(metric_sum "$metrics" "dag_children_index_size")"
    dag_heights_size="$(metric_sum "$metrics" "dag_heights_size")"
    dag_finalized_blocks_size="$(metric_sum "$metrics" "dag_finalized_blocks_size")"
    casper_buffer_approx_nodes="$(metric_sum "$metrics" "casper_buffer_approx_nodes")"
    retry_ratio="$(awk -v t="$block_requests_total" -v r="$block_requests_retries" 'BEGIN { if (t > 0) printf "%.6f", r/t; else print "0.000000" }')"

    echo "${timestamp_utc},${elapsed},${service},${rss_mib},${block_requests_total},${block_requests_retries},${retry_ratio},${dep_tracking_size},${broadcast_tracking_size},${waiting_list_total_size},${peers_total_size},${stream_cache_entries},${stream_cache_bytes},${init_block_message_queue_pending},${init_tuple_space_queue_pending},${proposer_queue_pending},${proposer_queue_rejected_total},${runtime_soft_checkpoint_total},${runtime_checkpoint_total},${runtime_revert_soft_checkpoint_total},${runtime_take_event_log_total},${runtime_take_event_log_events_total},${active_validators_cache_size},${deploys_in_scope_size},${deploys_in_scope_sig_bytes_estimate},${block_index_cache_size},${parents_post_state_cache_size},${hot_store_history_cont_cache_size},${hot_store_history_data_cache_size},${hot_store_history_joins_cache_size},${hot_store_state_cont_size},${hot_store_state_data_size},${hot_store_state_joins_size},${hot_store_history_cont_cache_items},${hot_store_history_data_cache_items},${hot_store_history_joins_cache_items},${hot_store_state_cont_items},${hot_store_state_data_items},${hot_store_state_joins_items},${dag_blocks_size},${dag_children_index_size},${dag_heights_size},${dag_finalized_blocks_size},${casper_buffer_approx_nodes}" >>"$SAMPLES_CSV"
  done

  sleep "$SAMPLE_EVERY_SECONDS"
done

if [[ "$SOAK_PROFILE_PROC" == "1" ]]; then
  for pid in "${PROC_SAMPLER_PIDS[@]}"; do
    wait "$pid" || true
  done
  write_proc_summary
fi

if [[ "$SOAK_PROFILE_FINALIZER" == "1" ]]; then
  write_finalizer_summary "$LOG_SINCE_UTC"
fi

awk -F, '
  NR == 1 { next }
  {
    svc = $3
    t = $2 + 0
    rss = $4 + 0
    dep = $8 + 0
    brd = $9 + 0
    wlt = $10 + 0
    prs = $11 + 0
    sce = $12 + 0
    scb = $13 + 0
    ibq = $14 + 0
    itq = $15 + 0
    pqp = $16 + 0
    pqr = $17 + 0
    rsc = $18 + 0
    rcp = $19 + 0
    rrv = $20 + 0
    rtl = $21 + 0
    rte = $22 + 0
    avc = $23 + 0
    dis = $24 + 0
    disb = $25 + 0
    bic = $26 + 0
    ppsc = $27 + 0
    hshc = $28 + 0
    hsdc = $29 + 0
    hsjc = $30 + 0
    hssc = $31 + 0
    hssd = $32 + 0
    hssj = $33 + 0
    hshci = $34 + 0
    hsdci = $35 + 0
    hsjci = $36 + 0
    hssci = $37 + 0
    hssdi = $38 + 0
    hssji = $39 + 0
    dgs = $40 + 0
    dci = $41 + 0
    dhs = $42 + 0
    dfs = $43 + 0
    cban = $44 + 0

    if (!(svc in start_t)) {
      start_t[svc] = t
      start_rss[svc] = rss
      start_dep[svc] = dep
      start_brd[svc] = brd
      start_wlt[svc] = wlt
      start_prs[svc] = prs
      start_sce[svc] = sce
      start_scb[svc] = scb
      start_ibq[svc] = ibq
      start_itq[svc] = itq
      start_pqp[svc] = pqp
      start_pqr[svc] = pqr
      start_rsc[svc] = rsc
      start_rcp[svc] = rcp
      start_rrv[svc] = rrv
      start_rtl[svc] = rtl
      start_rte[svc] = rte
      start_avc[svc] = avc
      start_dis[svc] = dis
      start_disb[svc] = disb
      start_bic[svc] = bic
      start_ppsc[svc] = ppsc
      start_hshc[svc] = hshc
      start_hsdc[svc] = hsdc
      start_hsjc[svc] = hsjc
      start_hssc[svc] = hssc
      start_hssd[svc] = hssd
      start_hssj[svc] = hssj
      start_hshci[svc] = hshci
      start_hsdci[svc] = hsdci
      start_hsjci[svc] = hsjci
      start_hssci[svc] = hssci
      start_hssdi[svc] = hssdi
      start_hssji[svc] = hssji
      start_dgs[svc] = dgs
      start_dci[svc] = dci
      start_dhs[svc] = dhs
      start_dfs[svc] = dfs
      start_cban[svc] = cban
    }
    end_t[svc] = t
    end_rss[svc] = rss
    end_dep[svc] = dep
    end_brd[svc] = brd
    end_wlt[svc] = wlt
    end_prs[svc] = prs
    end_sce[svc] = sce
    end_scb[svc] = scb
    end_ibq[svc] = ibq
    end_itq[svc] = itq
    end_pqp[svc] = pqp
    end_pqr[svc] = pqr
    end_rsc[svc] = rsc
    end_rcp[svc] = rcp
    end_rrv[svc] = rrv
    end_rtl[svc] = rtl
    end_rte[svc] = rte
    end_avc[svc] = avc
    end_dis[svc] = dis
    end_disb[svc] = disb
    end_bic[svc] = bic
    end_ppsc[svc] = ppsc
    end_hshc[svc] = hshc
    end_hsdc[svc] = hsdc
    end_hsjc[svc] = hsjc
    end_hssc[svc] = hssc
    end_hssd[svc] = hssd
    end_hssj[svc] = hssj
    end_hshci[svc] = hshci
    end_hsdci[svc] = hsdci
    end_hsjci[svc] = hsjci
    end_hssci[svc] = hssci
    end_hssdi[svc] = hssdi
    end_hssji[svc] = hssji
    end_dgs[svc] = dgs
    end_dci[svc] = dci
    end_dhs[svc] = dhs
    end_dfs[svc] = dfs
    end_cban[svc] = cban
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
      wlt_delta = end_wlt[svc] - start_wlt[svc]
      prs_delta = end_prs[svc] - start_prs[svc]
      sce_delta = end_sce[svc] - start_sce[svc]
      scb_delta = end_scb[svc] - start_scb[svc]
      ibq_delta = end_ibq[svc] - start_ibq[svc]
      itq_delta = end_itq[svc] - start_itq[svc]
      pqp_delta = end_pqp[svc] - start_pqp[svc]
      pqr_delta = end_pqr[svc] - start_pqr[svc]
      rsc_delta = end_rsc[svc] - start_rsc[svc]
      rcp_delta = end_rcp[svc] - start_rcp[svc]
      rrv_delta = end_rrv[svc] - start_rrv[svc]
      rtl_delta = end_rtl[svc] - start_rtl[svc]
      rte_delta = end_rte[svc] - start_rte[svc]
      avc_delta = end_avc[svc] - start_avc[svc]
      dis_delta = end_dis[svc] - start_dis[svc]
      disb_delta = end_disb[svc] - start_disb[svc]
      bic_delta = end_bic[svc] - start_bic[svc]
      ppsc_delta = end_ppsc[svc] - start_ppsc[svc]
      hshc_delta = end_hshc[svc] - start_hshc[svc]
      hsdc_delta = end_hsdc[svc] - start_hsdc[svc]
      hsjc_delta = end_hsjc[svc] - start_hsjc[svc]
      hssc_delta = end_hssc[svc] - start_hssc[svc]
      hssd_delta = end_hssd[svc] - start_hssd[svc]
      hssj_delta = end_hssj[svc] - start_hssj[svc]
      hshci_delta = end_hshci[svc] - start_hshci[svc]
      hsdci_delta = end_hsdci[svc] - start_hsdci[svc]
      hsjci_delta = end_hsjci[svc] - start_hsjci[svc]
      hssci_delta = end_hssci[svc] - start_hssci[svc]
      hssdi_delta = end_hssdi[svc] - start_hssdi[svc]
      hssji_delta = end_hssji[svc] - start_hssji[svc]
      dgs_delta = end_dgs[svc] - start_dgs[svc]
      dci_delta = end_dci[svc] - start_dci[svc]
      dhs_delta = end_dhs[svc] - start_dhs[svc]
      dfs_delta = end_dfs[svc] - start_dfs[svc]
      cban_delta = end_cban[svc] - start_cban[svc]
      rss_slope = rss_delta / dt

      printf "%s: samples=%d, elapsed_s=%.0f, rss_start_mib=%.3f, rss_end_mib=%.3f, rss_delta_mib=%.3f, rss_slope_mib_per_s=%.6f, dep_tracking_delta=%.0f, broadcast_tracking_delta=%.0f, waiting_list_total_delta=%.0f, peers_total_delta=%.0f, stream_cache_entries_delta=%.0f, stream_cache_bytes_delta=%.0f, init_block_message_queue_pending_delta=%.0f, init_tuple_space_queue_pending_delta=%.0f, proposer_queue_pending_delta=%.0f, proposer_queue_rejected_total_delta=%.0f, runtime_soft_checkpoint_total_delta=%.0f, runtime_checkpoint_total_delta=%.0f, runtime_revert_soft_checkpoint_total_delta=%.0f, runtime_take_event_log_total_delta=%.0f, runtime_take_event_log_events_total_delta=%.0f, active_validators_cache_delta=%.0f, deploys_in_scope_delta=%.0f, deploys_in_scope_sig_bytes_estimate_delta=%.0f, block_index_cache_delta=%.0f, parents_post_state_cache_delta=%.0f, hot_store_history_cont_cache_delta=%.0f, hot_store_history_data_cache_delta=%.0f, hot_store_history_joins_cache_delta=%.0f, hot_store_state_cont_delta=%.0f, hot_store_state_data_delta=%.0f, hot_store_state_joins_delta=%.0f, hot_store_history_cont_items_delta=%.0f, hot_store_history_data_items_delta=%.0f, hot_store_history_joins_items_delta=%.0f, hot_store_state_cont_items_delta=%.0f, hot_store_state_data_items_delta=%.0f, hot_store_state_joins_items_delta=%.0f, dag_blocks_delta=%.0f, dag_children_index_delta=%.0f, dag_heights_delta=%.0f, dag_finalized_blocks_delta=%.0f, casper_buffer_approx_nodes_delta=%.0f\n",
        svc, count[svc], dt, start_rss[svc], end_rss[svc], rss_delta, rss_slope, dep_delta, brd_delta, wlt_delta, prs_delta, sce_delta, scb_delta, ibq_delta, itq_delta, pqp_delta, pqr_delta, rsc_delta, rcp_delta, rrv_delta, rtl_delta, rte_delta, avc_delta, dis_delta, disb_delta, bic_delta, ppsc_delta, hshc_delta, hsdc_delta, hsjc_delta, hssc_delta, hssd_delta, hssj_delta, hshci_delta, hsdci_delta, hsjci_delta, hssci_delta, hssdi_delta, hssji_delta, dgs_delta, dci_delta, dhs_delta, dfs_delta, cban_delta
    }
  }
' "$SAMPLES_CSV" >"$SUMMARY_TXT"

echo "Leak soak complete"
echo "  samples: $SAMPLES_CSV"
echo "  summary: $SUMMARY_TXT"
cat "$SUMMARY_TXT"
if [[ "$SOAK_PROFILE_PROC" == "1" ]]; then
  echo "  proc_summary: $PROC_SUMMARY_TXT"
  cat "$PROC_SUMMARY_TXT"
fi
if [[ "$SOAK_PROFILE_FINALIZER" == "1" ]]; then
  echo "  finalizer_summary: $FINALIZER_SUMMARY_TXT"
  cat "$FINALIZER_SUMMARY_TXT"
fi
