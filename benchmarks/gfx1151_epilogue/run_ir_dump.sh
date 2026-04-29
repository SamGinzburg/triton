#!/usr/bin/env bash
# Dump TTIR/TTGIR/LLIR/AMDGCN twice: knob=1 (force pass on) and knob=0 (force pass off).
set -euo pipefail
cd "$(dirname "$0")"
ROOT=/home/samginzburg/triton
source "$ROOT/.venv/bin/activate"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export AMDGCN_ENABLE_DUMP=1
export MLIR_ENABLE_DUMP=1

for knob in 1 0; do
  OUT="$ROOT/benchmarks/gfx1151_epilogue/ir_dumps/knob_$knob"
  rm -rf "$OUT"
  mkdir -p "$OUT"
  export TRITON_CACHE_DIR="$OUT/cache"
  export TRITON_DUMP_DIR="$OUT/dump"
  export TRITON_HIP_USE_OPTIMIZE_EPILOGUE=$knob
  echo ">>> dumping knob=$knob -> $OUT"
  python dump_ir.py 2>&1 | tee "$OUT/run.log"
done

echo "DONE. Look under $ROOT/benchmarks/gfx1151_epilogue/ir_dumps/"
