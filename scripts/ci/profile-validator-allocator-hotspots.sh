#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${1:-docker/shard-with-autopropose.yml}"
DURATION_SECONDS="${2:-120}"
OUT_DIR="${3:-/tmp/casper-allocator-hotspots-$(date -u +%Y%m%dT%H%M%SZ)}"

VALIDATOR_SERVICE="${VALIDATOR_SERVICE:-validator2}"
HEAP_DIR="${HEAP_DIR:-docker/data/rnode.${VALIDATOR_SERVICE}/heap}"
HEAP_GLOB="${HEAP_GLOB:-jeprof.*.heap}"
TOP_STACKS="${TOP_STACKS:-15}"
TOP_FRAMES="${TOP_FRAMES:-12}"
NODE_BIN_OUT="${NODE_BIN_OUT:-$OUT_DIR/node-bin}"

mkdir -p "$OUT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 2
fi
if ! command -v x86_64-linux-gnu-addr2line >/dev/null 2>&1; then
  echo "x86_64-linux-gnu-addr2line is required" >&2
  exit 2
fi

if [[ ! -d "$HEAP_DIR" ]]; then
  echo "Heap directory not found: $HEAP_DIR" >&2
  echo "Hint: run validator with jemalloc profiling enabled (MALLOC_CONF=prof:true,...)." >&2
  exit 2
fi

detect_compose_cmd() {
  if docker compose -f "$COMPOSE_FILE" ps >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose -f "$COMPOSE_FILE")
  elif docker-compose -f "$COMPOSE_FILE" ps >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose -f "$COMPOSE_FILE")
  else
    echo "Unable to access compose services using $COMPOSE_FILE" >&2
    exit 2
  fi
}

aggregate_heap_stacks() {
  local heap_file="$1"
  local out_tsv="$2"
  awk '
    /^[[:space:]]*t\*:[[:space:]]/ {
      bytes = $3
      gsub(":", "", bytes)
      curr = bytes + 0
      next
    }
    /^@ / {
      if (curr > 0) {
        stack = substr($0, 3)
        agg[stack] += curr
      }
      curr = 0
    }
    END {
      for (s in agg) {
        printf "%.0f\t%s\n", agg[s], s
      }
    }
  ' "$heap_file" | sort -nr >"$out_tsv"
}

extract_node_map() {
  local heap_file="$1"
  local node_map_file="$2"
  awk '
    $0 ~ /\/opt\/docker\/bin\/node$/ || $0 ~ /\/usr\/local\/bin\/node$/ {
      if ($2 ~ /r-xp/) {
        print $0
        exit
      }
    }
  ' "$heap_file" >"$node_map_file"
}

symbolize_top_stacks() {
  local stacks_tsv="$1"
  local node_map_file="$2"
  local node_bin="$3"
  local top_stacks="$4"
  local top_frames="$5"
  local out_txt="$6"

  if [[ ! -s "$node_map_file" ]]; then
    {
      echo "Node text mapping not found in heap file."
      echo "Raw top stacks:"
      head -n "$top_stacks" "$stacks_tsv"
    } >"$out_txt"
    return 0
  fi

  local range file_off
  range="$(awk 'NR==1 {print $1}' "$node_map_file")"
  file_off="$(awk 'NR==1 {print $3}' "$node_map_file")"
  local start_hex end_hex off_hex
  start_hex="${range%-*}"
  end_hex="${range#*-}"
  off_hex="$file_off"
  local start_dec end_dec off_dec
  start_dec=$((16#${start_hex}))
  end_dec=$((16#${end_hex}))
  off_dec=$((16#${off_hex}))

  {
    echo "node_map: $(cat "$node_map_file")"
    echo
  } >"$out_txt"

  local line_no=0
  while IFS=$'\t' read -r bytes stack_addrs; do
    line_no=$((line_no + 1))
    if (( line_no > top_stacks )); then
      break
    fi
    echo "bytes=$bytes" >>"$out_txt"
    echo "stack=$stack_addrs" >>"$out_txt"
    local frame_idx=0
    for addr in $stack_addrs; do
      frame_idx=$((frame_idx + 1))
      if (( frame_idx > top_frames )); then
        break
      fi
      local addr_clean addr_dec rel_dec rel_hex sym
      addr_clean="${addr#0x}"
      addr_dec=$((16#${addr_clean}))
      if (( addr_dec < start_dec || addr_dec >= end_dec )); then
        continue
      fi
      rel_dec=$((addr_dec - start_dec + off_dec))
      rel_hex="$(printf '0x%x' "$rel_dec")"
      sym="$(x86_64-linux-gnu-addr2line -Cfpe "$node_bin" "$rel_hex" 2>/dev/null || true)"
      if [[ -n "$sym" ]]; then
        echo "  $addr -> $sym" >>"$out_txt"
      fi
    done
    echo >>"$out_txt"
  done <"$stacks_tsv"
}

summarize_symbolized_callsites() {
  local symbolized_file="$1"
  local out_file="$2"
  awk '
    function flush() {
      if (bytes <= 0) {
        return
      }
      if (first_fn != "") {
        first_frame_sum[first_fn] += bytes
      }
      category = "other"
      if (block ~ /tonic_reflection::server::/) {
        category = "tonic_reflection_startup"
      } else if (block ~ /mdb_env_open|heed::env::EnvOpenOptions::open/) {
        category = "lmdb_env_open"
      } else if (block ~ /RadixTreeImpl::load_node|RSpaceHistoryReaderImpl|HistoryReaderBase::get_data|HistoryReaderBase::get_joins/) {
        category = "rspace_history_read"
      } else if (block ~ /bytes_to_block_proto|KeyValueBlockStore::bytes_to_block_proto/) {
        category = "block_proto_decode"
      } else if (block ~ /DebruijnInterpreter|ChargingRSpace::.*::consume|ChargingRSpace::.*::produce/) {
        category = "interpreter_eval"
      } else if (block ~ /RuntimeOps::compute_state|block_creator::create::\{\{closure\}\}|Proposer<.*>::do_propose/) {
        category = "proposer_compute_state"
      } else if (block ~ /InMemHotStore<.*>::put_datum/) {
        category = "hot_store_put_datum"
      }
      category_sum[category] += bytes
      bytes = 0
      first_fn = ""
      block = ""
    }
    /^bytes=/ {
      flush()
      bytes = substr($0, 7) + 0
      next
    }
    /^  0x/ {
      block = block "\n" $0
      if (first_fn == "") {
        fn = $0
        sub(/^  0x[0-9a-f]+ -> /, "", fn)
        sub(/ at .*/, "", fn)
        first_fn = fn
      }
      next
    }
    END {
      flush()
      print "Top first symbolized frame (aggregated bytes):"
      for (k in first_frame_sum) {
        printf "%.0f\t%s\n", first_frame_sum[k], k | "sort -nr"
      }
      close("sort -nr")
      print ""
      print "Top categorized paths (aggregated bytes):"
      for (k in category_sum) {
        printf "%.0f\t%s\n", category_sum[k], k | "sort -nr"
      }
      close("sort -nr")
    }
  ' "$symbolized_file" >"$out_file"
}

detect_compose_cmd

echo "allocator hotspot profiling start"
echo "  compose_file: $COMPOSE_FILE"
echo "  validator_service: $VALIDATOR_SERVICE"
echo "  duration_seconds: $DURATION_SECONDS"
echo "  heap_dir: $HEAP_DIR"
echo "  output_dir: $OUT_DIR"

# Use find instead of shell glob expansion to avoid ARG_MAX failures when
# jemalloc emits a very large number of snapshots.
find "$HEAP_DIR" -maxdepth 1 -type f -name "$HEAP_GLOB" -delete 2>/dev/null || true

SOAK_OUT="$OUT_DIR/soak"
AUTO_RECREATE_ON_PRELOAD_FAIL=1 AUTO_RECREATE_MAX_ATTEMPTS=1 \
  ./scripts/ci/run-latency-benchmark.sh "$COMPOSE_FILE" "$DURATION_SECONDS" "$SOAK_OUT"

mapfile -t heaps < <(find "$HEAP_DIR" -maxdepth 1 -type f -name "$HEAP_GLOB" -print | sort -V)
non_empty_heaps=()
for heap_file in "${heaps[@]}"; do
  if [[ -s "$heap_file" ]]; then
    non_empty_heaps+=("$heap_file")
  fi
done

if (( ${#non_empty_heaps[@]} < 2 )); then
  echo "Expected at least 2 non-empty heap snapshots in $HEAP_DIR after soak, found ${#non_empty_heaps[@]}" >&2
  echo "All heap files found:" >&2
  printf '  %s\n' "${heaps[@]}" >&2
  exit 1
fi

EARLY_HEAP="${non_empty_heaps[0]}"
LATEST_HEAP="${non_empty_heaps[${#non_empty_heaps[@]}-1]}"
EARLY_SELECTION_MODE="first_non_empty"

# Prefer first heap generated at/after soak log window start to avoid startup-heavy i0 baseline.
PROFILE_SUMMARY="$SOAK_OUT/profile/summary.txt"
if [[ -s "$PROFILE_SUMMARY" ]]; then
  log_since_utc="$(awk -F': ' '/^log_since_utc:/ {print $2; exit}' "$PROFILE_SUMMARY" || true)"
  if [[ -n "$log_since_utc" ]]; then
    if log_since_epoch="$(date -u -d "$log_since_utc" +%s 2>/dev/null)"; then
      for heap_file in "${non_empty_heaps[@]}"; do
        heap_epoch="$(stat -c %Y "$heap_file" 2>/dev/null || true)"
        if [[ -n "$heap_epoch" ]] && (( heap_epoch >= log_since_epoch )); then
          EARLY_HEAP="$heap_file"
          EARLY_SELECTION_MODE="first_non_empty_at_or_after_log_since_utc"
          break
        fi
      done
    fi
  fi
fi

mkdir -p "$OUT_DIR/stacks"

EARLY_TSV="$OUT_DIR/stacks/early.stacks.tsv"
LATEST_TSV="$OUT_DIR/stacks/latest.stacks.tsv"
DELTA_TSV="$OUT_DIR/stacks/delta-positive.stacks.tsv"

aggregate_heap_stacks "$EARLY_HEAP" "$EARLY_TSV"
aggregate_heap_stacks "$LATEST_HEAP" "$LATEST_TSV"

awk -F'\t' 'NR==FNR { early[$2]=$1; next } { d=$1-early[$2]; if (d > 0) printf "%.0f\t%s\n", d, $2 }' \
  "$EARLY_TSV" "$LATEST_TSV" | sort -nr >"$DELTA_TSV"

cid="$("${COMPOSE_CMD[@]}" ps -q "$VALIDATOR_SERVICE" || true)"
if [[ -z "$cid" ]]; then
  echo "Unable to resolve container id for $VALIDATOR_SERVICE" >&2
  exit 1
fi

if ! docker cp "$cid:/opt/docker/bin/node" "$NODE_BIN_OUT" 2>/dev/null; then
  if ! docker cp "$cid:/usr/local/bin/node" "$NODE_BIN_OUT" 2>/dev/null; then
    echo "Unable to copy node binary from container $cid for symbolization" >&2
    exit 1
  fi
fi
chmod +x "$NODE_BIN_OUT"

EARLY_NODE_MAP="$OUT_DIR/stacks/early.node-map.txt"
LATEST_NODE_MAP="$OUT_DIR/stacks/latest.node-map.txt"
extract_node_map "$EARLY_HEAP" "$EARLY_NODE_MAP"
extract_node_map "$LATEST_HEAP" "$LATEST_NODE_MAP"

symbolize_top_stacks \
  "$LATEST_TSV" \
  "$LATEST_NODE_MAP" \
  "$NODE_BIN_OUT" \
  "$TOP_STACKS" \
  "$TOP_FRAMES" \
  "$OUT_DIR/stacks/latest.top${TOP_STACKS}.symbolized.txt"

symbolize_top_stacks \
  "$DELTA_TSV" \
  "$LATEST_NODE_MAP" \
  "$NODE_BIN_OUT" \
  "$TOP_STACKS" \
  "$TOP_FRAMES" \
  "$OUT_DIR/stacks/delta-positive.top${TOP_STACKS}.symbolized.txt"

CALLSITE_SUMMARY="$OUT_DIR/stacks/delta-positive.top${TOP_STACKS}.callsites.txt"
summarize_symbolized_callsites \
  "$OUT_DIR/stacks/delta-positive.top${TOP_STACKS}.symbolized.txt" \
  "$CALLSITE_SUMMARY"

{
  echo "Allocator hotspot summary"
  echo "compose_file=$COMPOSE_FILE"
  echo "validator_service=$VALIDATOR_SERVICE"
  echo "soak_out=$SOAK_OUT"
  echo "early_selection_mode=$EARLY_SELECTION_MODE"
  echo "early_heap=$EARLY_HEAP"
  echo "latest_heap=$LATEST_HEAP"
  echo
  echo "Top stacks by bytes in latest heap:"
  head -n "$TOP_STACKS" "$LATEST_TSV"
  echo
  echo "Top positive stack growth (early -> latest):"
  head -n "$TOP_STACKS" "$DELTA_TSV"
  echo
  echo "Symbolized reports:"
  echo "  $OUT_DIR/stacks/latest.top${TOP_STACKS}.symbolized.txt"
  echo "  $OUT_DIR/stacks/delta-positive.top${TOP_STACKS}.symbolized.txt"
  echo "  $CALLSITE_SUMMARY"
} >"$OUT_DIR/summary.txt"

echo "allocator hotspot profiling complete"
echo "  summary: $OUT_DIR/summary.txt"
