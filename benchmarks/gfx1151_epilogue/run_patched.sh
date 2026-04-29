#!/usr/bin/env bash
# Phase 3 (post-patch): run BF16 + FP16 tutorial three ways: default, force-on, force-off.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=/home/samginzburg/triton
source "$ROOT/.venv/bin/activate"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export TRITON_ALWAYS_COMPILE=1

run_one() {
  local label=$1
  local knob=$2
  local OUT="$ROOT/benchmarks/gfx1151_epilogue/patched_$label"
  mkdir -p "$OUT"
  export TRITON_CACHE_DIR="/tmp/triton-cache-$label"
  rm -rf "$TRITON_CACHE_DIR"
  if [ "$knob" = "unset" ]; then
    unset TRITON_HIP_USE_OPTIMIZE_EPILOGUE
  else
    export TRITON_HIP_USE_OPTIMIZE_EPILOGUE=$knob
  fi
  {
    echo "label=$label knob=$knob"
    echo "git_rev=$(git -C $ROOT rev-parse HEAD)"
    python -c "import triton; print('triton',triton.__version__); print('target',triton.runtime.driver.active.get_current_target())"
  } > "$OUT/provenance.txt"
  echo ">>> [$label knob=$knob] BF16 GEMM"
  python bench_bf16_gemm.py --layouts NN,TN,NT --out "$OUT/bf16_gemm.json" 2>&1 | tee "$OUT/bf16_gemm.txt"
  echo ">>> [$label knob=$knob] tutorial 03 FP16"
  python "$ROOT/python/tutorials/03-matrix-multiplication.py" 2>&1 | tee "$OUT/tutorial03.txt"
}

run_one default unset
run_one force_on 1
run_one force_off 0

echo "DONE."
