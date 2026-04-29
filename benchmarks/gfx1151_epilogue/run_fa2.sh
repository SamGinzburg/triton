#!/usr/bin/env bash
# Phase 4: FA2 perf comparing default (gate disabled on gfx1151) vs force_on (upstream behavior).
set -euo pipefail
cd "$(dirname "$0")"
ROOT=/home/samginzburg/triton
source "$ROOT/.venv/bin/activate"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export TRITON_ALWAYS_COMPILE=1

run_one() {
  local label=$1
  local knob=$2
  local OUT="$ROOT/benchmarks/gfx1151_epilogue/fa2_$label"
  mkdir -p "$OUT"
  export TRITON_CACHE_DIR="/tmp/triton-cache-fa2-$label"
  rm -rf "$TRITON_CACHE_DIR"
  if [ "$knob" = "unset" ]; then
    unset TRITON_HIP_USE_OPTIMIZE_EPILOGUE
  else
    export TRITON_HIP_USE_OPTIMIZE_EPILOGUE=$knob
  fi
  echo ">>> [$label knob=$knob] FA2 sweep"
  python bench_fa2.py --out "$OUT/fa2.json" 2>&1 | tee "$OUT/fa2.txt"
}

run_one default unset
run_one force_on 1

echo "DONE."
