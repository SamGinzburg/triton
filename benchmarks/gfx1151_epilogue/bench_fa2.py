"""FA2 perf + correctness for gfx1151 epilogue investigation.

Reuses python/tutorials/06-fused-attention.py's attention(...) kernel. Sweeps a
focused set of (HEAD_DIM, causal, mode, N_CTX) and reports TFLOP/s, with a
fp32 reference correctness check on a small case.
"""
import argparse
import importlib.util
import json
import os
import sys

import torch
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def load_attention():
    here = os.path.dirname(os.path.abspath(__file__))
    tut = os.path.normpath(os.path.join(here, "..", "..", "python", "tutorials", "06-fused-attention.py"))
    spec = importlib.util.spec_from_file_location("fa2_tut", tut)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.attention


attention = load_attention()


def reference(q, k, v, causal, sm_scale):
    # naive fp32 reference
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * sm_scale
    if causal:
        N = scores.shape[-1]
        mask = torch.triu(torch.ones(N, N, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, vf).to(q.dtype)


def correctness(B=1, H=2, N=256, D=64, causal=True, sm_scale=1.3):
    torch.manual_seed(0)
    q = torch.randn((B, H, N, D), dtype=torch.float16, device=DEVICE, requires_grad=False)
    k = torch.randn((B, H, N, D), dtype=torch.float16, device=DEVICE, requires_grad=False)
    v = torch.randn((B, H, N, D), dtype=torch.float16, device=DEVICE, requires_grad=False)
    out = attention(q, k, v, causal, sm_scale, False)
    ref = reference(q, k, v, causal, sm_scale)
    diff = (out.to(torch.float32) - ref.to(torch.float32)).abs()
    rel = diff / (ref.to(torch.float32).abs() + 1e-3)
    return float(diff.max()), float(rel.max())


def bench(B, H, N, D, causal, mode, sm_scale=1.3, warmup=10, rep=40):
    q = torch.randn((B, H, N, D), dtype=torch.float16, device=DEVICE, requires_grad=True)
    k = torch.randn((B, H, N, D), dtype=torch.float16, device=DEVICE, requires_grad=True)
    v = torch.randn((B, H, N, D), dtype=torch.float16, device=DEVICE, requires_grad=True)
    fn = lambda: attention(q, k, v, causal, sm_scale, False)
    if mode == "bwd":
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops = 2.0 * B * H * N * N * D * 2  # qk + pv
    if causal:
        flops *= 0.5
    if mode == "bwd":
        flops *= 2.5
    return ms, flops * 1e-12 / (ms * 1e-3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    p.add_argument("--quick", action="store_true", help="reduced sweep")
    args = p.parse_args()

    print(f"# device: {torch.cuda.get_device_name(0)}")
    print(f"# target: {triton.runtime.driver.active.get_current_target()}")
    print(f"# triton: {triton.__version__}")
    print(f"# TRITON_HIP_USE_OPTIMIZE_EPILOGUE={os.environ.get('TRITON_HIP_USE_OPTIMIZE_EPILOGUE', 'unset')}")
    print(f"# TRITON_CACHE_DIR={os.environ.get('TRITON_CACHE_DIR', 'default')}")
    print()

    print("# correctness (fp16 vs fp32 ref, small case)")
    for causal in [False, True]:
        ad, rd = correctness(causal=causal)
        ok = ad < 1e-1 and rd < 5e-2  # fp16 tolerance
        print(f"# causal={causal}: max_abs={ad:.4g} max_rel={rd:.4g} ok={ok}")
    print()

    if args.quick:
        head_dims = [64, 128]
        ncxs = [1024, 4096]
        causals = [False, True]
        modes = ["fwd", "bwd"]
    else:
        head_dims = [64, 128]
        ncxs = [1024, 2048, 4096, 8192]
        causals = [False, True]
        modes = ["fwd", "bwd"]

    B, H = 4, 32
    print(f"{'head_dim':>8} {'N_CTX':>6} {'causal':>6} {'mode':>4} {'ms':>10} {'TFLOP/s':>10}")
    print("-" * 60)
    rows = []
    for D in head_dims:
        for N in ncxs:
            for causal in causals:
                for mode in modes:
                    try:
                        ms, tflops = bench(B, H, N, D, causal, mode)
                    except Exception as e:
                        print(f"{D:>8} {N:>6} {str(causal):>6} {mode:>4}  ERROR  {e}")
                        continue
                    print(f"{D:>8} {N:>6} {str(causal):>6} {mode:>4} {ms:>10.4f} {tflops:>10.2f}")
                    rows.append({"head_dim": D, "n_ctx": N, "causal": causal,
                                 "mode": mode, "ms": ms, "tflops": tflops})

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"knob": os.environ.get("TRITON_HIP_USE_OPTIMIZE_EPILOGUE", "unset"),
                       "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
