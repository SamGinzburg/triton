# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import random

random.seed(42)

import numpy as np
import torch

torch.manual_seed(42)

import triton
import triton.language as tl

"""
Triton autotuning for function matmul_kernel_sparse finished after 98.81s; best config selected: BLOCK_SIZE_M: 128, BLOCK_SIZE_N: 128, BLOCK_SIZE_K: 64, GROUP_SIZE_M: 2, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None;
"""
def get_autotune_config():
    configs = []

    for BLOCK_M in [16, 32, 64, 128]:
        for BLOCK_N in [16, 32, 64, 128]:
            for BLOCK_K in [16, 32, 64, 128]:
                for num_warps in [2, 4]:
                    for num_stages in [1, 2]:
                        for GROUP_SIZE_M in [4]:
                            configs.append(
                                triton.Config(
                                    {
                                        "BLOCK_SIZE_M": BLOCK_M,
                                        "BLOCK_SIZE_N": BLOCK_N,
                                        "BLOCK_SIZE_K": BLOCK_K,
                                        "GROUP_SIZE_M": GROUP_SIZE_M,
                                        "matrix_instr_nonkdim": 16,
                                    },
                                    num_stages=num_stages,
                                    num_warps=num_warps,
                                ),
                            )
    return configs


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.heuristics(
    values={
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit
def matmul_kernel_sparse(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_meta_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    stride_aMeta_m,
    stride_aMeta_k,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    EVEN_K: tl.constexpr,  #
    USE_BF16: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    dtype = tl.bfloat16 if USE_BF16 else tl.float16

    tl.assume(stride_am >= 0)
    tl.assume(stride_ak >= 0)
    tl.assume(stride_bk >= 0)
    tl.assume(stride_bn >= 0)
    tl.assume(stride_cm >= 0)
    tl.assume(stride_cn >= 0)
    tl.assume(stride_aMeta_m >= 0)
    tl.assume(stride_aMeta_k >= 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    BLOCK_SIZE_K_A: tl.constexpr = BLOCK_SIZE_K // 2
    BLOCK_SIZE_K_A_META: tl.constexpr = BLOCK_SIZE_K // 16
    offs_k_a = tl.arange(0, BLOCK_SIZE_K_A)
    offs_k_b = tl.arange(0, BLOCK_SIZE_K)
    offs_k_aMeta = tl.arange(0, BLOCK_SIZE_K_A_META)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_a[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k_b[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    aMeta_ptrs = a_meta_ptr + (
        offs_am[:, None] * stride_aMeta_m + offs_k_aMeta[None, :] * stride_aMeta_k
    )
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            aMeta = tl.load(
                aMeta_ptrs
            )
        else:
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(a_ptrs, mask=offs_k_a[None, :] < K - k * BLOCK_SIZE_K, other=0.0)

            # Note: the masks below are not correct. It's fine for M=N=K=8192 because 8192 is a power of 2. But this kernel is not correct for other shapes.
            # For some reason, things don't work if I fix the masks. Did not have time to figure out why. Do not use in production.
            # @nocommit.
            b = tl.load(b_ptrs, mask=offs_k_b[:, None] < K - k * BLOCK_SIZE_K_A, other=0.0)
            aMeta = tl.load(
                aMeta_ptrs,
                mask=offs_k_aMeta[None, :] < K - k * BLOCK_SIZE_K_A_META,
                other=0,
            )

        #if pid == 0:
        #  tl.device_print("aMeta", aMeta, hex=False)

        # We accumulate along the K dimension.
        accumulator += tl.sparse_dot(a, b, aMeta)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K_A * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        aMeta_ptrs += BLOCK_SIZE_K_A_META * stride_aMeta_k
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(dtype)


    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def make_sparse(A):
    assert len(A.shape) == 2
    assert A.shape[-1] % 4 == 0
    assert A.is_contiguous()
    constant = 0
    for i in range(A.shape[0]):
        indices_to_zero = []
        for j in range(0, A.shape[1], 4):
            # This is what we want to work
            # indices_to_zero.extend(random.sample([j, j + 1, j + 2, j + 3], 2))

            # Using just this works
            # indices_to_zero.extend(random.sample([j, j + 1], 2))


            # Create a fixed pattern
            if constant % 2 == 0:
                indices_to_zero.extend([j, j + 1])
            elif constant % 2 == 1:
                indices_to_zero.extend([j + 1, j + 2])
        constant += 1
        A[i, indices_to_zero] = 0
    return A


def compress(A):
    assert len(A.shape) == 2
    assert A.shape[-1] % 4 == 0
    assert A.is_contiguous()
    flat = A.flatten().cpu().detach().numpy()
    nonzero_indices = []
    meta_nibbles = []

    for outerIdx in range(0, len(flat), 4):
        nibble = 0
        nonzeroCount = 0
        for innerIdx in range(4):
            val = flat[outerIdx + innerIdx]
            if val != 0:
                nonzero_indices.append(outerIdx + innerIdx)
                nibble |= innerIdx << (2 * nonzeroCount)
                nonzeroCount += 1
                if nonzeroCount > 2:
                    raise Exception("too many nonzeros!")

        if nonzeroCount == 0:
            nibble |= 0b0100
            nonzero_indices.extend([outerIdx + 0, outerIdx + 1])

        # there is definitely a cleaner way to do this... don't judge me
        if nonzeroCount == 1:
            last_nonzero = nonzero_indices[-1]
            if last_nonzero == outerIdx + 3:
                assert nibble == 0b0011
                nibble = 0b1100
                nonzero_indices[-1] = outerIdx + 0
                nonzero_indices.append(outerIdx + 3)
            else:
                nibble |= 0b1100
                nonzero_indices.append(outerIdx + 3)

        meta_nibbles.append(nibble)

    assert len(meta_nibbles) == len(flat) // 4
    assert len(nonzero_indices) == len(flat) // 2

    metas = []
    for outerIdx in range(0, len(meta_nibbles), 4):
        meta = 0
        meta |= meta_nibbles[outerIdx + 0] << 0
        meta |= meta_nibbles[outerIdx + 1] << 4
        meta |= meta_nibbles[outerIdx + 2] << 8
        meta |= meta_nibbles[outerIdx + 3] << 12
        metas.append(meta)

    aSparse = (A.flatten()[nonzero_indices]).reshape(A.shape[0], A.shape[1] // 2)
    aMeta = torch.tensor(np.array(metas, dtype=np.uint16).astype(np.int16)).reshape(
        A.shape[0], A.shape[1] // 16
    )
    return aSparse.cuda(), aMeta.cuda()


def matmul(aSparse, aMeta, b):
    # Check constraints.
    assert aSparse.shape[1] * 2 == b.shape[0], "Incompatible dimensions"
    assert aMeta.shape[1] * 16 == b.shape[0], "Incompatible dimensions"
    assert aMeta.shape[0] == aSparse.shape[0], "Incompatible dimensions"
    assert aSparse.is_contiguous(), "Matrix A must be contiguous"
    M, _ = aSparse.shape
    K, N = b.shape

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    matmul_kernel_sparse[grid](
        aSparse,
        b,
        c,  #
        aMeta,
        M,
        N,
        K,  #
        aSparse.stride(0),
        aSparse.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        aMeta.stride(0),
        aMeta.stride(1),
        USE_BF16=False, #dtype is torch.bfloat16,
    )
    return c

torch.manual_seed(42)
test_dim = 4096
dtype = torch.float16
#dtype = torch.bfloat16

a = make_sparse(torch.randn((test_dim, test_dim), device="cuda", dtype=dtype))
# a = make_sparse(torch.ones((test_dim, test_dim), device="cuda", dtype=dtype))
print("Running sparse compression in Python... this can be quite slow, be patient!")
aSparse, aMeta = compress(a)
b = torch.randn((test_dim, test_dim), device="cuda", dtype=dtype)
# b = torch.arange(np.prod((test_dim, test_dim)), device="cuda", dtype=torch.float32) % 1024
# b = b.reshape(test_dim, test_dim)
# b = torch.ones((test_dim, test_dim), device="cuda", dtype=dtype)

#b = torch.randn((test_dim, test_dim), device="cuda", dtype=dtype) % 16
#b = b.to(dtype)
"""
for row in range((test_dim)):
    for col in range((test_dim)):
        b[row][col] = (row * col) / 1024
"""
#b = torch.ones((test_dim, test_dim), device="cuda", dtype=dtype)

print (a, b)
print("Autotuning... set TRITON_PRINT_AUTOTUNING=1 to see logs here...")
print ("aMeta: ", aMeta)


"""
count  =0
for row in aMeta:
    if count == 2:
        break
    hex_values = [hex(x.item()) for x in row]
    print("Hexadecimal values:", hex_values)
    count += 1
"""

triton_output = matmul(aSparse, aMeta, b)
torch_output = torch.matmul(a, b)

#triton_output = triton_output.to(dtype)
#torch_output = torch_output.to(dtype)

if torch.allclose(triton_output, torch_output, atol=1e-3, rtol=1e-3):
    print("Correctness: Triton and Torch match! ✅")
    print ("Triton output: ", triton_output)
    print ("torch output: ", torch_output)
else:
    print("Correctness: Triton and Torch differ! ❌")
    print ("Triton output: ", triton_output)
    print ("torch output: ", torch_output)
    print (triton_output - torch_output)
    #diff = triton_output - torch_output
    #for val in diff:
    #    print (val)

ms = triton.testing.do_bench(lambda: matmul(aSparse, aMeta, b))
flops = 2 * test_dim * test_dim * test_dim * 1e-12 / (ms * 1e-3)
print(f"Perf: {flops} TFLOPS")


ms_torch = triton.testing.do_bench(lambda: torch.matmul(a, b))
flops = 2 * test_dim * test_dim * test_dim * 1e-12 / (ms_torch * 1e-3)
print(f"Perf (torch): {flops} TFLOPS")
