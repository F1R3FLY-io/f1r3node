#!/usr/bin/env bash
set -euo pipefail

# rust-client monotonic tip cache can become stale after chain reset/rebuild
# and force deploys into future-only windows. Clear it before correctness/perf runs.
rm -f /tmp/f1r3fly_tip_floor_*.txt

echo "Cleared rust-client tip cache files in /tmp"
