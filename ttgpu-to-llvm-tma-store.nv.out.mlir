#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_store(%arg0: !tt.tensordesc<tensor<128x256xf32, #shared>>, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: tensor<128x256xf32, #blocked>) {
    %0 = ttg.local_alloc %arg2 : (tensor<128x256xf32, #blocked>) -> !ttg.memdesc<128x256xf32, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %arg0[%arg1, %arg1] %0 : !tt.tensordesc<tensor<128x256xf32, #shared>>, !ttg.memdesc<128x256xf32, #shared, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}

