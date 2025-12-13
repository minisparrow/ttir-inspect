// RUN: triton-opt %s -split-input-file --triton-nvidia-tma-lowering | FileCheck %s
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: tma_load
// CHECK: ttg.local_alloc : ()
// CHECK: ttg.local_alloc : ()
// CHECK: ttng.init_barrier
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: ttng.wait_barrier
// CHECK: ttng.inval_barrier
// CHECK: ttg.local_load
  tt.func public @tma_load(%arg0: !tt.tensordesc<tensor<128x64xf16, #nvmma_128>>, %arg1: i32) -> tensor<128x64xf16, #blocked> {
    %l = tt.descriptor_load %arg0[%arg1, %arg1] : !tt.tensordesc<tensor<128x64xf16, #nvmma_128>> -> tensor<128x64xf16, #blocked>
    tt.return %l : tensor<128x64xf16, #blocked>
  }
}

