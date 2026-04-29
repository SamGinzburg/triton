# gfx1151 epilogue optimization investigation

Companion benchmark + evidence harness for the patch that gates the AMD
`tritonamdgpu-optimize-epilogue` pass behind a tri-state knob and
default-disables it on `gfx1151`.

## Code change

Two files, +15/-1 lines total:

- `python/triton/knobs.py` — declare `TRITON_HIP_USE_OPTIMIZE_EPILOGUE` next
  to the other HIP feature knobs.
- `third_party/amd/backend/compiler.py` — add `is_optimize_epilogue_enabled`
  helper (default-off on `gfx1151`, default-on elsewhere; honors the env
  knob in either direction) and gate `add_optimize_epilogue(pm)` behind it.

The MLIR pass itself (`OptimizeEpilogue.cpp`) is untouched.

## Why

On gfx1151 (RDNA3.5 / Strix Halo), the existing `BypassEpilogueSMEM`
rewrite produces a direct WMMA-layout store path that maps to many
narrow 16-bit stores. Letting the normal `xmma → blocked` layout
conversion survive instead lets the lowerer move data through LDS and
issue 128-bit coalesced stores. On this hardware, that tradeoff wins for
dense BF16 GEMM and is roughly neutral for fused attention.

## Evidence

### ISA (single shape: 4096 × 4096 × 128 BF16)

`ir_dumps/knob_{0,1}/cache/.../matmul_kernel.amdgcn`:

- `knob=1` (current upstream behavior): 64 × `buffer_store_b16`
- `knob=0` (new gfx1151 default): 8 × `buffer_store_b128`

8× fewer store ops, 8× wider.

### BF16 GEMM perf (TFLOP/s, NN layout)

| shape | baseline | default (gate off) | Δ | rocBLAS |
|---|---:|---:|---:|---:|
| 4096×4096×64 | 5.49 | 9.77 | +78% | 7.40 |
| 4096×4096×128 | 10.15 | 15.99 | +58% | 16.37 |
| 4096×4096×256 | 16.38 | 24.21 | +48% | 26.86 |
| 4096×4096×1024 | 26.83 | 28.83 | +7% | 31.69 |
| 8192×8192×128 | 10.85 | 17.74 | +63% | 17.99 |
| 8192×1024×128 | 9.56 | 17.30 | +81% | 16.84 |
| 1024×8192×128 | 9.44 | 13.99 | +48% | 10.45 |
| 4097×4099×127 (mask) | 11.57 | 14.90 | +29% | 15.62 |

Knob-plumbing sanity: `default ≈ force_off`, `force_on ≈ baseline`.
Full TN/NT layout data lives under `patched_*/bf16_gemm.{txt,json}`.

### FP16 tutorial 03

Within noise across all three knob settings (no regression). See
`*/tutorial03.txt`.

### FA2 / fused attention

Numerically unchanged (identical `max_abs` and `max_rel` against an fp32
reference, both knob settings). Forward pass within ±2% across most
shapes; backward consistently +1% to +9%. One outlier on D=64 /
N_CTX=8192 / causal=True / fwd at -4.6%, right at the edge of the
acceptance threshold. See `fa2_*/fa2.{txt,json}`.

## Reproducer scripts

- `bench_bf16_gemm.py` — BF16 GEMM matrix from the plan, NN/TN/NT.
- `bench_fa2.py` — focused FA2 sweep using the tutorial's `attention()`.
- `dump_ir.py` — dump TTIR / TTGIR / LLIR / AMDGCN for one BF16 shape.
- `run_baseline.sh` — Phase 0 baseline (apply with patches stashed).
- `run_patched.sh` — Phase 3 default vs force_on vs force_off.
- `run_fa2.sh` — Phase 4 FA2 default vs force_on.
- `run_ir_dump.sh` — Phase 2 IR dumps with knob=1 and knob=0.

Each runner uses an isolated `TRITON_CACHE_DIR` because the kernel cache
key does not include the new env knob; toggling the knob in the same
cache dir would alias.

## Caveats / open follow-ups

- The cache key does not include `TRITON_HIP_USE_OPTIMIZE_EPILOGUE`.
  Cleanest fix is to thread the knob value into the kernel cache key or
  into `HIPOptions.hash`. Not necessary for correctness but a footgun
  for users toggling the knob.
- gfx1151-only string match. If broader gfx11 coverage is wanted, the
  gate can fan out trivially; the FA2 sample size here only covers
  gfx1151.
- The remaining FA2 forward regression at one shape is plausibly the
  case where `chooseMfmaLikeStoreLayout` would have succeeded for the
  bypass path. Alternative C from the plan (skip bypass only when the
  wide-store layout cannot be chosen) is the principled long-term fix.
