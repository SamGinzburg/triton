// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 | FileCheck %s

#linear_wmma_v1_store = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 64], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]], warp = [[0, 16], [0, 32], [16, 0]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_v1_bf16_linear_store_uses_b128
  tt.func public @wmma_v1_bf16_linear_store_uses_b128(%value: tensor<128x128xbf16, #linear_wmma_v1_store>, %arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %c128_i32 = arith.constant 128 : i32
    %m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear_wmma_v1_store}>>
    %n = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear_wmma_v1_store}>>
    %m_col = tt.expand_dims %m {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear_wmma_v1_store}>> -> tensor<128x1xi32, #linear_wmma_v1_store>
    %n_row = tt.expand_dims %n {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear_wmma_v1_store}>> -> tensor<1x128xi32, #linear_wmma_v1_store>
    %m_bcast = tt.broadcast %m_col : tensor<128x1xi32, #linear_wmma_v1_store> -> tensor<128x128xi32, #linear_wmma_v1_store>
    %n_bcast = tt.broadcast %n_row : tensor<1x128xi32, #linear_wmma_v1_store> -> tensor<128x128xi32, #linear_wmma_v1_store>
    %c128 = tt.splat %c128_i32 : i32 -> tensor<128x128xi32, #linear_wmma_v1_store>
    %m_scaled = arith.muli %m_bcast, %c128 : tensor<128x128xi32, #linear_wmma_v1_store>
    %offs = arith.addi %m_scaled, %n_bcast : tensor<128x128xi32, #linear_wmma_v1_store>
    // CHECK: rocdl.raw.ptr.buffer.store {{.*}} : vector<4xi32>
    amdg.buffer_store %value, %arg0[%offs] : tensor<128x128xbf16, #linear_wmma_v1_store>
    tt.return
  }
}
