// RUN: triton-opt %s -split-input-file --triton-nvidia-tma-lowering | FileCheck %s
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: tma_store
//       CHECK: ttg.local_alloc {{.*}} -> !ttg.memdesc<128x256xf32, #shared, #smem>
//       CHECK: ttng.fence_async_shared {bCluster = false}
//       CHECK: ttng.async_tma_copy_local_to_global
  tt.func public @tma_store(%arg0: !tt.tensordesc<tensor<128x256xf32, #nvmma_128>>, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: tensor<128x256xf32, #blocked>) {
    tt.descriptor_store %arg0[%arg1, %arg1], %arg2 : !tt.tensordesc<tensor<128x256xf32, #nvmma_128>>, tensor<128x256xf32, #blocked>
    tt.return
  }
}
