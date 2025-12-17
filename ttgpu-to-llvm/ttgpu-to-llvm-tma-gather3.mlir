
// RUN: triton-opt %s --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1], [2], [16], [0]], lane = [[0], [0], [0], [0], [0]], warp = [[4], [8]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-LABEL: @tma_gather_redundant_indices
tt.func @tma_gather_redundant_indices(%arg0: !tt.tensordesc<tensor<1x128xbf16, #shared1>>, %arg1: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %arg2: tensor<32xi32, #linear>, %arg3: i32, %arg4: !ttg.memdesc<32x128xbf16, #shared1, #smem, mutable>, %arg5: i1) {
  // Codegen for this case is actually incorrect due to linear layouts
  // incorrectly handling register broadcasting, but the test outcome is nonetheless
  // the same.

  // CHECK-COUNT-4: cp.async.bulk.tensor
  ttng.async_tma_gather %arg0[%arg2, %arg3] %arg4, %arg1, %arg5 : !tt.tensordesc<tensor<1x128xbf16, #shared1>>, tensor<32xi32, #linear>, i32, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<32x128xbf16, #shared1, #smem, mutable>, i1
  // CHECK-NEXT: ret void
  tt.return
}

}
