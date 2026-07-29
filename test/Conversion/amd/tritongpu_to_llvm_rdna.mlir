// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 --convert-builtin-func-to-llvm | FileCheck %s
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1151 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=CHECK,GFX1151

#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: reduce_dpp_max
  tt.func @reduce_dpp_max(%arg0: tensor<32xf32, #blocked3>) {
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 280, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 276, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 274, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 273, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK: rocdl.permlanex16
    // CHECK: llvm.intr.maxnum
    // CHECK: rocdl.readlane
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<32xf32, #blocked3>) -> f32
    tt.return
  }
}

#linear = #ttg.linear<{register = [[16, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @reduce_linear_layout
tt.func private @reduce_linear_layout(%arg0: tensor<32x2xi32, #linear>) -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #linear}>> {
  // This tensor has 64 elements with the last dimension across the lower and upper 16 lanes.
  // Therefore, we can reduce it with a 16 element butterfly shuffle.

  // CHECK-DAG: [[result0:%.*]] = llvm.mlir.undef
  // CHECK-DAG: [[select_lo:%.*]] = llvm.mlir.constant(1985229328 : i32)
  // CHECK-DAG: [[select_hi:%.*]] = llvm.mlir.constant(-19088744 : i32)
  // CHECK-DAG: [[reg0:%.*]] = llvm.extractvalue %arg0[0]
  // CHECK-DAG: [[reg1:%.*]] = llvm.extractvalue %arg0[1]
  // CHECK: [[permlane0:%.*]] = rocdl.permlanex16 [[reg0]], [[reg0]], [[select_lo]], [[select_hi]], true, false
  // CHECK: [[sum0:%.*]] = llvm.add [[reg0]], [[permlane0]]
  // CHECK: [[permlane1:%.*]] = rocdl.permlanex16 [[reg1]], [[reg1]], [[select_lo]], [[select_hi]], true, false
  // CHECK: [[sum1:%.*]] = llvm.add [[reg1]], [[permlane1]]
  // CHECK: [[result1:%.*]] = llvm.insertvalue [[sum0]], [[result0]][0]
  // CHECK: [[result2:%.*]] = llvm.insertvalue [[sum1]], [[result1]][1]

  %0 = "tt.reduce"(%arg0) ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.reduce.return %1 : i32
  }) {axis = 1 : i32} : (tensor<32x2xi32, #linear>) -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #linear}>>

  // CHECK: llvm.return [[result2]]
  tt.return %0 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #linear}>>
}
}

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @bf16_mulf
tt.func private @bf16_mulf(%arg0: tensor<64xbf16, #blocked>, %arg1: tensor<64xbf16, #blocked>) -> tensor<64xbf16, #blocked> {
  // CHECK-COUNT-2: llvm.call_intrinsic "llvm.amdgcn.fdot2.bf16.bf16"
  %0 = arith.mulf %arg0, %arg1 : tensor<64xbf16, #blocked>
  tt.return %0 : tensor<64xbf16, #blocked>
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @packed_fptoui_u8
tt.func private @packed_fptoui_u8(%arg0: tensor<128xf32, #blocked>) -> tensor<128xi8, #blocked> {
  // CHECK-COUNT-4: llvm.call_intrinsic "llvm.amdgcn.cvt.pk.u8.f32"
  // CHECK: llvm.bitcast {{.*}} : i32 to vector<4xi8>
  %rounded = tt.extern_elementwise %arg0 {libname = "", libpath = "", pure = true, symbol = "__triton_hip_rint"} : (tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
  %0 = arith.fptoui %rounded : tensor<128xf32, #blocked> to tensor<128xi8, #blocked>
  tt.return %0 : tensor<128xi8, #blocked>
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @packed_fptoui_int4_nibbles
tt.func private @packed_fptoui_int4_nibbles(%lo: tensor<128xf32, #blocked>, %hi: tensor<128xf32, #blocked>) -> tensor<128xi8, #blocked> {
  // CHECK-COUNT-8: llvm.call_intrinsic "llvm.amdgcn.cvt.pk.u8.f32"
  %lo_rounded = tt.extern_elementwise %lo {libname = "", libpath = "", pure = true, symbol = "__triton_hip_rint"} : (tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
  %hi_rounded = tt.extern_elementwise %hi {libname = "", libpath = "", pure = true, symbol = "__triton_hip_rint"} : (tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
  %lo_u8 = arith.fptoui %lo_rounded : tensor<128xf32, #blocked> to tensor<128xi8, #blocked>
  %hi_u8 = arith.fptoui %hi_rounded : tensor<128xf32, #blocked> to tensor<128xi8, #blocked>
  %four = arith.constant dense<4> : tensor<128xi8, #blocked>
  %hi_shifted = arith.shli %hi_u8, %four : tensor<128xi8, #blocked>
  %packed = arith.ori %lo_u8, %hi_shifted : tensor<128xi8, #blocked>
  tt.return %packed : tensor<128xi8, #blocked>
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @fptoui_u8_fractional_fallback
tt.func private @fptoui_u8_fractional_fallback(%arg0: tensor<128xf32, #blocked>) -> tensor<128xi8, #blocked> {
  // CHECK-NOT: llvm.amdgcn.cvt.pk.u8.f32
  // CHECK-COUNT-4: llvm.fptoui
  %0 = arith.fptoui %arg0 : tensor<128xf32, #blocked> to tensor<128xi8, #blocked>
  tt.return %0 : tensor<128xi8, #blocked>
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @fptoui_u8_short_group_fallback
tt.func private @fptoui_u8_short_group_fallback(%arg0: tensor<64xf32, #blocked>) -> tensor<64xi8, #blocked> {
  // CHECK-NOT: llvm.amdgcn.cvt.pk.u8.f32
  // CHECK-COUNT-2: llvm.fptoui
  %0 = arith.fptoui %arg0 : tensor<64xf32, #blocked> to tensor<64xi8, #blocked>
  tt.return %0 : tensor<64xi8, #blocked>
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @fptosi_i8_signed_fallback
tt.func private @fptosi_i8_signed_fallback(%arg0: tensor<128xf32, #blocked>) -> tensor<128xi8, #blocked> {
  // CHECK-NOT: llvm.amdgcn.cvt.pk.u8.f32
  // CHECK-COUNT-4: llvm.fptosi
  %0 = arith.fptosi %arg0 : tensor<128xf32, #blocked> to tensor<128xi8, #blocked>
  tt.return %0 : tensor<128xi8, #blocked>
}
}

// -----

#wave32 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// GFX1151-LABEL: @buffer_atomic_wave_reduce_f32
tt.func private @buffer_atomic_wave_reduce_f32(
    %base: !tt.ptr<f32>,
    %offsets: tensor<32xi32, #wave32> {tt.constancy = 32 : i32},
    %values: tensor<32xf32, #wave32>,
    %mask: tensor<32xi1, #wave32>) {
  // GFX1151: rocdl.ballot
  // GFX1151: rocdl.mbcnt.lo
  // GFX1151-NOT: rocdl.mbcnt.hi
  // GFX1151: llvm.call_intrinsic "llvm.amdgcn.ds.permute"
  // GFX1151: llvm.call_intrinsic "llvm.amdgcn.ds.bpermute"
  // GFX1151: llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"
  %unused = amdg.buffer_atomic_rmw fadd, relaxed, gpu, %values, %base[%offsets], %mask : tensor<32xf32, #wave32>
  tt.return
}
}

// -----

#wave32 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// GFX1151-LABEL: @buffer_atomic_wave_reduce_i32
tt.func private @buffer_atomic_wave_reduce_i32(
    %base: !tt.ptr<i32>,
    %offsets: tensor<32xi32, #wave32> {tt.constancy = 32 : i32},
    %values: tensor<32xi32, #wave32>,
    %mask: tensor<32xi1, #wave32>) {
  // GFX1151: rocdl.mbcnt.lo
  // GFX1151-NOT: rocdl.mbcnt.hi
  // GFX1151: llvm.call_intrinsic "llvm.amdgcn.ds.bpermute"
  // GFX1151: llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.add"
  %unused = amdg.buffer_atomic_rmw add, relaxed, cta, %values, %base[%offsets], %mask : tensor<32xi32, #wave32>
  tt.return
}
}

// -----

#wave32 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// GFX1151-LABEL: @buffer_atomic_wave_reduce_used_fallback
tt.func private @buffer_atomic_wave_reduce_used_fallback(
    %base: !tt.ptr<f32>,
    %offsets: tensor<32xi32, #wave32> {tt.constancy = 32 : i32},
    %values: tensor<32xf32, #wave32>) -> tensor<32xf32, #wave32> {
  // GFX1151-NOT: llvm.amdgcn.ds.bpermute
  // GFX1151: llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"
  %old = amdg.buffer_atomic_rmw fadd, relaxed, gpu, %values, %base[%offsets] : tensor<32xf32, #wave32>
  tt.return %old : tensor<32xf32, #wave32>
}
}

// -----

#wave32 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// GFX1151-LABEL: @buffer_atomic_wave_reduce_low_contention_fallback
tt.func private @buffer_atomic_wave_reduce_low_contention_fallback(
    %base: !tt.ptr<f32>,
    %offsets: tensor<32xi32, #wave32> {tt.constancy = 4 : i32},
    %values: tensor<32xf32, #wave32>) {
  // GFX1151-NOT: llvm.amdgcn.ds.bpermute
  // GFX1151: llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"
  %unused = amdg.buffer_atomic_rmw fadd, relaxed, gpu, %values, %base[%offsets] : tensor<32xf32, #wave32>
  tt.return
}
}

// -----

#wave32 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// GFX1151-LABEL: @buffer_atomic_wave_reduce_ordering_fallback
tt.func private @buffer_atomic_wave_reduce_ordering_fallback(
    %base: !tt.ptr<f32>,
    %offsets: tensor<32xi32, #wave32> {tt.constancy = 32 : i32},
    %values: tensor<32xf32, #wave32>) {
  // GFX1151-NOT: llvm.amdgcn.ds.bpermute
  // GFX1151: llvm.fence syncscope("agent") release
  // GFX1151: llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"
  // GFX1151: llvm.fence syncscope("agent") acquire
  %unused = amdg.buffer_atomic_rmw fadd, acq_rel, gpu, %values, %base[%offsets] : tensor<32xf32, #wave32>
  tt.return
}
}
