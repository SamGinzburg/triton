# WMMAv1 MFMA-like Store Layout Report

## Change Summary

This change adds a WMMAv1 branch to
`chooseMfmaLikeStoreLayout(RankedTensorType valType)` for AMD WMMA
accumulator encodings on wave32 RDNA3. It is intentionally limited to:

- `AMDWmmaEncodingAttr`
- `version == 1`
- `isTranspose == true`
- instruction shape `16x16x16`
- rank-2 F16/BF16 result tensors
- 32-lane wave layouts

Other WMMA versions, RDNA4 layouts, non-transposed layouts, and non-F16/BF16
element types continue to return `std::nullopt`.

## Basis Permutation

The confirmed transposed WMMAv1 output layout places the low N bits across
register and lane dimensions as follows:

| N bit | Original owner | Original basis |
| --- | --- | --- |
| N0 | lane | `[0, 1]` |
| N1 | register | `[0, 2]` |
| N2 | register | `[0, 4]` |
| N3 | register | `[0, 8]` |

The chosen swap layout rotates the low four N basis vectors right:

```text
identity N bases: [1, 2, 4, 8, 16, 32, 64, ...]
swap N bases:     [8, 1, 2, 4, 16, 32, 64, ...]
```

Composing the canonical WMMAv1 layout with this swap moves `N0`, `N1`, and
`N2` into the register dimension, while the lane-16 bit selects `N3`. That
gives each thread eight consecutive elements along the innermost N dimension.

For a `128x128xbf16` tile, the resulting store layout is:

| Input dim | Basis vectors after permutation |
| --- | --- |
| `register` | `[[0, 1], [0, 2], [0, 4], [0, 64], [32, 0], [64, 0]]` |
| `lane` | `[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]]` |
| `warp` | `[[0, 16], [0, 32], [16, 0]]` |
| `block` | `[]` |

The first three register basis vectors cover `N0..N2`, so
`LinearEncodingAttr::getContigPerThread()[1] == 8`.

## IR Dump Histograms

The GPU runtime is not usable from this sandbox (`/dev/kfd` and `/dev/dri` are
not visible; Triton reports no active drivers), so the IR check used offline
compilation with target `GPUTarget("hip", "gfx1151", 32)` and the requested
matmul reproducer shape.

### Bypass On, Before Change

Source: clean parent source archive at `/tmp/triton-before-qbev2e`.

```text
64 ds_load_u16_d16_hi
64 ds_load_u16
64 buffer_store_b16
32 ds_load_b128
 8 ds_store_b128
```

The TTGIR epilogue stored directly from `#ttg.amd_wmma`, so no permlane
conversion was emitted:

```text
amdg.buffer_store %c, ... : tensor<128x128xbf16, #mma>
```

### Bypass On, After Change

```text
64 ds_load_u16_d16_hi
64 ds_load_u16
32 ds_load_b128
16 v_permlanex16_b32
 8 ds_store_b128
 8 buffer_store_b128
```

The TTGIR epilogue now converts from WMMA to the new linear layout and stores
with contiguity 8:

```text
%c_77 = ttg.convert_layout %c : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
amdg.buffer_store %c_77, ... {contiguity = 8 : i32} : tensor<128x128xbf16, #linear>
```

The LLIR contains 16 calls to `llvm.amdgcn.permlanex16.i32`, and the final
AMDGCN contains 8 `buffer_store_b128`. The remaining LDS traffic in the
histogram is from the matmul mainloop staging; there is no epilogue LDS
round-trip between the permlane sequence and the global stores.

### Bypass Off, After Change

```text
64 ds_load_u16_d16_hi
64 ds_load_u16
40 ds_load_b128
16 ds_store_b128
 8 buffer_store_b128
```

This remains the LDS-lowered fallback path and is unchanged in behavior by the
new WMMA branch.

## Performance and Correctness Benchmarks

Real GPU benchmark execution was blocked in this environment. The ROCm runtime
device nodes are not available inside the sandbox:

```text
/dev/kfd: false
/dev/dri: false
python -c "import triton; print(triton.runtime.driver.active.get_current_target())"
RuntimeError: 0 active drivers ([]). There should only be one.
```

| Suite | Requested coverage | Result |
| --- | --- | --- |
| Training BF16 GEMM | square_4k/8k, linear_out_proj, qkv_combined, mlp_up, mlp_down, linear_big_batch | Not run; GPU runtime unavailable |
| Small-K BF16 GEMM | M=4096, K in `{64,128,256}`, NN/TN/NT | Not run; GPU runtime unavailable |
| FA2 forward | HEAD_DIM/N_CTX/causal cube | Not run; GPU runtime unavailable |
| Correctness vs FP32 reference | GEMM and FA2 | Not run; GPU runtime unavailable |

The offline codegen signal is the expected prerequisite for those benchmarks:
the bypass path now emits permlane shuffles plus `buffer_store_b128`, instead
of scalar `buffer_store_b16`.

## Build and Test Results

Commands run from `/home/samginzburg/triton` unless noted:

| Command | Outcome |
| --- | --- |
| `CCACHE_DIR=/tmp/ccache-triton make` | Passed; rebuilt Triton incrementally |
| `build/cmake.linux-x86_64-cpython-3.12/unittest/Dialect/TritonGPU/LinearLayoutConversions --gtest_filter=LinearLayoutConversionsTest.WMMA_v1_ChooseMfmaLikeStoreLayout` | Passed, 1/1 test |
| `build/cmake.linux-x86_64-cpython-3.12/unittest/Dialect/TritonGPU/LinearLayoutConversions` | Passed, 110/110 tests |
| `cmake --build build/cmake.linux-x86_64-cpython-3.12 --target check-triton` | Blocked; this build tree has no `check-triton` target |
| Offline gfx1151 compile with `TRITON_HIP_USE_OPTIMIZE_EPILOGUE=1` | Passed; emitted 16 `v_permlanex16_b32` and 8 `buffer_store_b128` |
| Offline gfx1151 compile with `TRITON_HIP_USE_OPTIMIZE_EPILOGUE=0` | Passed; remained LDS fallback with 8 `buffer_store_b128` |

## Gate Removal Recommendation

The offline IR evidence supports removing the gfx1151 default-disable gate for
`OptimizeEpilogue` in a follow-up commit, but not in this patch. The follow-up
should first run the requested real GPU performance and correctness matrix on
gfx1151, because this session could not execute ROCm kernels.
