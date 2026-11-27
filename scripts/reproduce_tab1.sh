#!/usr/bin/env bash
# Tab. 1: lookup latency vs. memory-table size. Needs no checkpoint; run on an idle GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/benchmark_lookup_latency.py --out results/tab1_lookup_latency.json
