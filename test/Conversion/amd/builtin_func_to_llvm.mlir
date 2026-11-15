// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942 ftz=True" --convert-builtin-func-to-llvm="ftz=True" | FileCheck %s --check-prefix=LLVM_FTZ
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950 ftz=True" --convert-builtin-func-to-llvm="ftz=True" | FileCheck %s --check-prefix=LLVM_FTZ
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942 ftz=False" --convert-builtin-func-to-llvm="ftz=False" | FileCheck %s --check-prefix=LLVM_NO_FTZ
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950 ftz=False" --convert-builtin-func-to-llvm="ftz=False" | FileCheck %s --check-prefix=LLVM_NO_FTZ

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fast_expf(%arg0: tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked> {
    // CHECK-LABEL: test_fast_expf
    // LLVM_FTZ: rocdl.exp2
    // LLVM_NO_FTZ: llvm.intr.exp2
    %0 = tt.extern_elementwise %arg0 {libname = "libdevice", libpath = "", pure = true, symbol = "__triton_hip_fast_expf"} : (tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked>
    tt.return %0 : tensor<64xf32, #blocked>
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fast_tanhf(%arg0: tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked> {
    // CHECK-LABEL: test_fast_tanhf
    // LLVM_FTZ: rocdl.exp2
    // LLVM_NO_FTZ: llvm.intr.exp2
    %0 = tt.extern_elementwise %arg0 {libname = "libdevice", libpath = "", pure = true, symbol = "__triton_hip_fast_tanhf"} : (tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked>
    tt.return %0 : tensor<64xf32, #blocked>
  }
}

// Identical blocks must not be merged: for scalar atomics in a polling loop
// this creates an irreducible loop with lane-divergent entries.
// RUN: triton-opt %s --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=NO_MERGE
module {
  // NO_MERGE-LABEL: llvm.func @no_block_merging
  llvm.func @no_block_merging(%ptr: !llvm.ptr<1>, %pred: i1, %cmp: i32, %val: i32) {
    llvm.cond_br %pred, ^bb1, ^bb2
  ^bb1:
    // NO_MERGE-COUNT-2: llvm.cmpxchg
    %0 = llvm.cmpxchg %ptr, %cmp, %val syncscope("agent") acquire monotonic : !llvm.ptr<1>, i32
    rocdl.s.barrier
    llvm.br ^bb3
  ^bb2:
    %1 = llvm.cmpxchg %ptr, %cmp, %val syncscope("agent") acquire monotonic : !llvm.ptr<1>, i32
    rocdl.s.barrier
    llvm.br ^bb3
  ^bb3:
    llvm.return
  }
}
