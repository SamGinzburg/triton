// RUN: triton-opt %s -split-input-file -tritonamdgpu-optimize-epilogue | FileCheck %s

// CHECK-LABEL: one_op_in_chain
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK: tt.store %{{.*}}, %{{.*}} : tensor<32x32x!tt.ptr<f16>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @one_op_in_chain(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %2 = arith.truncf %1 : tensor<32x32xf32, #blocked> to tensor<32x32xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: store_dword_mfma32_small_n
// CHECK-NOT: tensor<32x8x!tt.ptr<f16>, #linear>
// CHECK: tt.store %{{.*}}, %{{.*}} : tensor<32x8x!tt.ptr<f16>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @store_dword_mfma32_small_n(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x8xf32, #mma>
    %0 = ttg.convert_layout %cst : tensor<32x8xf32, #mma> -> tensor<32x8xf32, #blocked>
    %1 = arith.truncf %0 : tensor<32x8xf32, #blocked> to tensor<32x8xf16, #blocked>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x8x!tt.ptr<f16>, #blocked>
    tt.store %2, %1 : tensor<32x8x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: two_ops_in_chain
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK: tt.store %{{.*}}, %{{.*}} : tensor<32x32x!tt.ptr<f16>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @two_ops_in_chain(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %2 = math.exp2 %1 : tensor<32x32xf32, #blocked>
    %3 = arith.truncf %2 : tensor<32x32xf32, #blocked> to tensor<32x32xf16, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.store %4, %3 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [64, 0]], block = []}>
// CHECK-LABEL: store_dword_128x128
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
// CHECK-DAG: %[[PTR:.+]] = ttg.convert_layout %{{.*}} : tensor<128x128x!tt.ptr<f16>, #mma> -> tensor<128x128x!tt.ptr<f16>, #linear>
// CHECK-DAG: %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #linear>
// CHECK: tt.store %[[PTR]], %[[VAL]] : tensor<128x128x!tt.ptr<f16>, #linear>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @store_dword_128x128(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    %2 = arith.truncf %1 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 128], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [0, 64], [32, 0]], block = []}>
// CHECK-LABEL: store_dword_256x256
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<256x256xf32, #mma> -> tensor<256x256xf32, #blocked>
// CHECK-DAG: %[[PTR:.+]] = ttg.convert_layout %{{.*}} : tensor<256x256x!tt.ptr<f16>, #mma> -> tensor<256x256x!tt.ptr<f16>, #linear>
// CHECK-DAG: %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<256x256xf16, #mma> -> tensor<256x256xf16, #linear>
// CHECK: tt.store %[[PTR]], %[[VAL]] : tensor<256x256x!tt.ptr<f16>, #linear>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [32, 32, 16], isTransposed = true}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @store_dword_256x256(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<256x256xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<256x256xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<256x256xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<256x256xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<256x256xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<256x256xf32, #mma> -> tensor<256x256xf32, #blocked>
    %2 = arith.truncf %1 : tensor<256x256xf32, #blocked> to tensor<256x256xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x256x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<256x256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
// CHECK-LABEL: store_dword_16x16
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
// CHECK-DAG: %[[PTR:.+]] = ttg.convert_layout %{{.*}} : tensor<128x128x!tt.ptr<f16>, #mma> -> tensor<128x128x!tt.ptr<f16>, #linear>
// CHECK-DAG: %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #linear>
// CHECK: tt.store %[[PTR]], %[[VAL]] : tensor<128x128x!tt.ptr<f16>, #linear>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @store_dword_16x16(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    %2 = arith.truncf %1 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----
// On gfx11 WMMA v1 transposed BF16, BypassEpilogueSMEM should pick the new
// linear layout that places 8 consecutive [B]F16 elements per thread along
// the N axis, enabling 128-bit global stores.
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 64], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]], warp = [[0, 16], [0, 32], [16, 0]], block = []}>
// CHECK-LABEL: wmma_v1_bf16_128x128
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
// CHECK-DAG: %[[PTR:.+]] = ttg.convert_layout %{{.*}} : tensor<128x128x!tt.ptr<bf16>, #mma> -> tensor<128x128x!tt.ptr<bf16>, #linear>
// CHECK-DAG: %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
// CHECK: tt.store %[[PTR]], %[[VAL]] : tensor<128x128x!tt.ptr<bf16>, #linear>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [2, 4], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 1, isTranspose = true, ctaLayout = {warp = [[0, 1], [0, 2], [1, 0]]}}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_v1_bf16_128x128(%arg0: !tt.ptr<bf16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<128x128xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    %2 = arith.truncf %1 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    tt.store %3, %2 : tensor<128x128x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----
// WMMA v1 with isTranspose=false is not supported by the wide-store helper:
// the bypass still rewrites the store to the WMMA accumulator layout, but no
// #linear is introduced.
// CHECK-LABEL: wmma_v1_not_transposed
// CHECK-NOT: #linear
// CHECK-NOT: ttg.convert_layout %{{.*}} -> tensor<128x128xf32, #blocked>
// CHECK: tt.store %{{.*}} : tensor<128x128x!tt.ptr<bf16>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [2, 4], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 1, isTranspose = false, ctaLayout = {warp = [[0, 1], [0, 2], [1, 0]]}}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_v1_not_transposed(%arg0: !tt.ptr<bf16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<128x128xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    %2 = arith.truncf %1 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    tt.store %3, %2 : tensor<128x128x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----
// WMMA v2 is not yet handled by the helper (TODO in
// chooseMfmaLikeStoreLayout). The bypass still fires, but no #linear is
// introduced.
// CHECK-LABEL: wmma_v2_should_skip
// CHECK-NOT: #linear
// CHECK-NOT: ttg.convert_layout %{{.*}} -> tensor<128x128xf32, #blocked>
// CHECK: tt.store %{{.*}} : tensor<128x128x!tt.ptr<bf16>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [2, 4], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 2, isTranspose = true, ctaLayout = {warp = [[0, 1], [0, 2], [1, 0]]}}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_v2_should_skip(%arg0: !tt.ptr<bf16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<128x128xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    %2 = arith.truncf %1 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    tt.store %3, %2 : tensor<128x128x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----
// WMMA v1 int4 dots accumulate into i32. The wide-store helper should choose
// the same WMMAv1 #linear layout so the i32 accumulator can be emitted as
// b128 stores instead of falling back to the plain WMMA accumulator layout.
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 64], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]], warp = [[0, 16], [0, 32], [16, 0]], block = []}>
// CHECK-LABEL: wmma_v1_i4_i32_128x128
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<128x128xi32, #mma> -> tensor<128x128xi32, #blocked>
// CHECK-DAG: %[[PTR:.+]] = ttg.convert_layout %{{.*}} : tensor<128x128x!tt.ptr<i32>, #mma> -> tensor<128x128x!tt.ptr<i32>, #linear>
// CHECK-DAG: %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<128x128xi32, #mma> -> tensor<128x128xi32, #linear>
// CHECK: tt.store %[[PTR]], %[[VAL]] : tensor<128x128x!tt.ptr<i32>, #linear>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [2, 4], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 1, isTranspose = true, ctaLayout = {warp = [[0, 1], [0, 2], [1, 0]]}}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_v1_i4_i32_128x128(%a: tensor<128x128xi4, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %b: tensor<128x128xi4, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg0: !tt.ptr<i32>) {
    %cst = arith.constant dense<0> : tensor<128x128xi32, #mma>
    %0 = tt.dot %a, %b, %cst : tensor<128x128xi4, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<128x128xi4, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<128x128xi32, #mma>
    %1 = ttg.convert_layout %0 : tensor<128x128xi32, #mma> -> tensor<128x128xi32, #blocked>
    %2 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x128x!tt.ptr<i32>, #blocked>
    tt.store %2, %1 : tensor<128x128x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----
// Packed int4 uses physical i8 operands but still represents an int4 dot. It
// should get the same WMMAv1 i32 accumulator store layout.
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 64], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]], warp = [[0, 16], [0, 32], [16, 0]], block = []}>
// CHECK-LABEL: wmma_v1_dot_scaled_i4_i32_128x128
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<128x128xi32, #mma> -> tensor<128x128xi32, #blocked>
// CHECK-DAG: %[[PTR:.+]] = ttg.convert_layout %{{.*}} : tensor<128x128x!tt.ptr<i32>, #mma> -> tensor<128x128x!tt.ptr<i32>, #linear>
// CHECK-DAG: %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<128x128xi32, #mma> -> tensor<128x128xi32, #linear>
// CHECK: tt.store %[[PTR]], %[[VAL]] : tensor<128x128x!tt.ptr<i32>, #linear>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [2, 4], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 1, isTranspose = true, ctaLayout = {warp = [[0, 1], [0, 2], [1, 0]]}}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_v1_dot_scaled_i4_i32_128x128(%a: tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>, %b: tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>, %arg0: !tt.ptr<i32>) {
    %cst = arith.constant dense<0> : tensor<128x128xi32, #mma>
    %0 = tt.dot_scaled %a, %b, %cst lhs = int4 rhs = int4 {fastMath = false} : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xi32, #mma>
    %1 = ttg.convert_layout %0 : tensor<128x128xi32, #mma> -> tensor<128x128xi32, #blocked>
    %2 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x128x!tt.ptr<i32>, #blocked>
    tt.store %2, %1 : tensor<128x128x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----
// Truncated int4 accumulators are not a 128-bit-friendly final store shape
// here, so keep the bypass but do not introduce the WMMAv1 #linear layout.
// CHECK-LABEL: wmma_v1_i4_i8_store_should_skip
// CHECK-NOT: #linear
// CHECK-NOT: ttg.convert_layout %{{.*}} -> tensor<128x128xi32, #blocked>
// CHECK: tt.store %{{.*}} : tensor<128x128x!tt.ptr<i8>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [2, 4], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 1, isTranspose = true, ctaLayout = {warp = [[0, 1], [0, 2], [1, 0]]}}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_v1_i4_i8_store_should_skip(%a: tensor<128x128xi4, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %b: tensor<128x128xi4, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg0: !tt.ptr<i8>) {
    %cst = arith.constant dense<0> : tensor<128x128xi32, #mma>
    %0 = tt.dot %a, %b, %cst : tensor<128x128xi4, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<128x128xi4, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<128x128xi32, #mma>
    %1 = ttg.convert_layout %0 : tensor<128x128xi32, #mma> -> tensor<128x128xi32, #blocked>
    %2 = arith.trunci %1 : tensor<128x128xi32, #blocked> to tensor<128x128xi8, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<128x128x!tt.ptr<i8>, #blocked>
    tt.store %3, %2 : tensor<128x128x!tt.ptr<i8>, #blocked>
    tt.return
  }
}

// -----
// To validate if  warpsPerCTA is not expected, no linear layout will be created.
// CHECK-LABEL: store_dword_16x16
// CHECK-NOT: #linear
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @store_dword_16x16(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    %2 = arith.truncf %1 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----
// To validate if N of the input shape is not expected, larger or equal 16X2, no linear layout will be created.
// CHECK-LABEL: store_dword_16x16
// CHECK-NOT: #linear
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @store_dword_16x16(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #blocked>
    %2 = arith.truncf %1 : tensor<16x16xf32, #blocked> to tensor<16x16xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<16x16x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
