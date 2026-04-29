"""BF16 GEMM benchmark for gfx1151 epilogue investigation.

Mirrors the structure of python/tutorials/03-matrix-multiplication.py but in BF16
and across the full shape matrix from the GFX1151 epilogue plan.

Output: a CSV-ish text table to stdout. Use TRITON_HIP_USE_OPTIMIZE_EPILOGUE to
control the AMD OptimizeEpilogue pass (after patches land).
"""
import argparse
import json
import os
import sys

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def get_autotune_configs():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=2, num_warps=8),
    ]


@triton.autotune(configs=get_autotune_configs(), key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1),
                        b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1))
    return c


# Full matrix from the plan.
SHAPES = [
    # (M, N, K, label)
    (4096, 4096, 64,   "small_K_square_64"),
    (4096, 4096, 128,  "small_K_square_128"),
    (4096, 4096, 256,  "medium_K_square_256"),
    (4096, 4096, 1024, "large_K_square_1024"),
    (8192, 8192, 128,  "large_square_8k_128"),
    (8192, 1024, 128,  "tall_skinny_128"),
    (1024, 8192, 128,  "wide_128"),
    (4097, 4099, 127,  "tail_mask_127"),
]


def make_inputs(M, N, K, layout):
    """layout: NN, TN, NT.

    NN: a is [M,K] row-major,   b is [K,N] row-major
    TN: a is [M,K] col-major,   b is [K,N] row-major   (a non-K-contig)
    NT: a is [M,K] row-major,   b is [K,N] col-major   (b K-contig)
    """
    if layout == "NN":
        a = torch.randn((M, K), device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.bfloat16)
    elif layout == "TN":
        a = torch.randn((K, M), device=DEVICE, dtype=torch.bfloat16).t()
        b = torch.randn((K, N), device=DEVICE, dtype=torch.bfloat16)
    elif layout == "NT":
        a = torch.randn((M, K), device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn((N, K), device=DEVICE, dtype=torch.bfloat16).t()
    else:
        raise ValueError(f"unknown layout {layout}")
    return a, b


def bench_one(M, N, K, layout, provider, warmup=25, rep=100):
    a, b = make_inputs(M, N, K, layout)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "rocblas":
        fn = lambda: torch.matmul(a, b)
    elif provider == "triton":
        fn = lambda: matmul(a, b)
    else:
        raise ValueError(provider)
    ms, lo, hi = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=quantiles)
    tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return ms, lo, hi, tflops


def correctness(M, N, K, layout):
    a, b = make_inputs(M, N, K, layout)
    out = matmul(a, b)
    ref = torch.matmul(a, b)
    # bf16 matmul accumulates in fp32 already; tolerance per project guidance.
    return torch.allclose(out, ref, atol=1e-2, rtol=1e-2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", default="NN,TN,NT")
    parser.add_argument("--shapes", default="all", help="all or comma-separated labels")
    parser.add_argument("--providers", default="triton,rocblas")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--out", default=None, help="optional JSON output file")
    parser.add_argument("--check", action="store_true", help="run correctness check first")
    args = parser.parse_args()

    layouts = args.layouts.split(",")
    providers = args.providers.split(",")
    if args.shapes == "all":
        shapes = SHAPES
    else:
        wanted = set(args.shapes.split(","))
        shapes = [s for s in SHAPES if s[3] in wanted]

    print(f"# device: {torch.cuda.get_device_name(0)}")
    print(f"# target: {triton.runtime.driver.active.get_current_target()}")
    print(f"# torch: {torch.__version__}")
    print(f"# triton: {triton.__version__}")
    print(f"# TRITON_HIP_USE_OPTIMIZE_EPILOGUE={os.environ.get('TRITON_HIP_USE_OPTIMIZE_EPILOGUE', 'unset')}")
    print(f"# TRITON_CACHE_DIR={os.environ.get('TRITON_CACHE_DIR', 'default')}")
    print()
    print(f"{'shape':<24} {'layout':<6} {'provider':<10} {'ms':>10} {'lo':>10} {'hi':>10} {'TFLOP/s':>10}")
    print("-" * 90)

    rows = []
    for M, N, K, label in shapes:
        for layout in layouts:
            if args.check:
                ok = correctness(M, N, K, layout)
                if not ok:
                    print(f"# WARN correctness failed for {label} {layout}", file=sys.stderr)
            for provider in providers:
                try:
                    ms, lo, hi, tflops = bench_one(M, N, K, layout, provider, args.warmup, args.rep)
                except Exception as e:
                    print(f"{label:<24} {layout:<6} {provider:<10}  ERROR  {e}")
                    continue
                print(f"{label:<24} {layout:<6} {provider:<10} {ms:>10.4f} {lo:>10.4f} {hi:>10.4f} {tflops:>10.2f}")
                rows.append({"label": label, "M": M, "N": N, "K": K,
                             "layout": layout, "provider": provider,
                             "ms": ms, "lo": lo, "hi": hi, "tflops": tflops})

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "device": torch.cuda.get_device_name(0),
                "target": str(triton.runtime.driver.active.get_current_target()),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "knob": os.environ.get("TRITON_HIP_USE_OPTIMIZE_EPILOGUE", "unset"),
                "rows": rows,
            }, f, indent=2)


if __name__ == "__main__":
    main()
