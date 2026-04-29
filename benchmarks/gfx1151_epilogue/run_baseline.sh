#!/usr/bin/env bash
# Phase 0 baseline: run upstream Triton (no patches yet) on BF16 GEMM matrix
# and FP16 tutorial-03. Stash outputs under benchmarks/gfx1151_epilogue/baseline/
set -euo pipefail
cd "$(dirname "$0")"
ROOT=/home/samginzburg/triton
OUT="$ROOT/benchmarks/gfx1151_epilogue/baseline"
mkdir -p "$OUT"

source "$ROOT/.venv/bin/activate"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export TRITON_ALWAYS_COMPILE=1
export TRITON_CACHE_DIR=/tmp/triton-cache-baseline
rm -rf "$TRITON_CACHE_DIR"

# Provenance
{
  echo "git_rev=$(git -C $ROOT rev-parse HEAD)"
  python -c "import torch,triton; print('torch',torch.__version__); print('triton',triton.__version__); print('device',torch.cuda.get_device_name(0)); print('target',triton.runtime.driver.active.get_current_target())"
  rocm-smi --showproductname --showuniqueid 2>&1 | head -20 || true
} > "$OUT/provenance.txt"

echo ">>> BF16 GEMM matrix (full)"
python bench_bf16_gemm.py --layouts NN,TN,NT --out "$OUT/bf16_gemm.json" 2>&1 | tee "$OUT/bf16_gemm.txt"

echo ">>> Tutorial 03 (FP16)"
python "$ROOT/python/tutorials/03-matrix-multiplication.py" 2>&1 | tee "$OUT/tutorial03.txt"

echo "DONE. Output -> $OUT"
