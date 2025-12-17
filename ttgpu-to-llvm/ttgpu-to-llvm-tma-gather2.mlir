
// RUN: triton-opt %s --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1], [2], [16], [0]], lane = [[0], [0], [0], [0], [0]], warp = [[4], [8]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-LABEL: @tma_gather_8_consecutive_indices
tt.func @tma_gather_8_consecutive_indices(%arg0: !tt.tensordesc<tensor<1x128xbf16, #shared1>>, %arg1: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %arg2: tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>, %arg3: i32, %arg4: !ttg.memdesc<32x128xbf16, #shared1, #smem, mutable>, %arg5: i1) {
  // Due to the `sizePerThread = [1, 8]`, each warp now handles 8 consecutive
  // rows, where each row is divided into 2 segments for a total of 4 gather4s.
  //
  // t[warpId, 0:128], t[warpId, 128:256], t[warpId+4, 0:128], t[warpId+4, 128:256]
  //
  // So the base addresses are [x, x+2048, x+256, x+2048+256], where `x = warpId*256`.

  // CHECK: [[WARP_ID:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.idx.i32
  // CHECK: [[WARP_STRIDE_TMP:%.*]] = shl i32 [[WARP_ID]], 9
  // CHECK: [[OFFSET0:%.*]] = and i32 [[WARP_STRIDE_TMP]], 1536

  // CHECK: zext nneg i32 [[OFFSET0]] to i64
  // CHECK: [[BASEPTR0:%.*]] = getelementptr bfloat, ptr addrspace(3)
  // CHECK: cp.async.bulk.tensor

  // CHECK: [[OFFSET1:%.*]] = getelementptr i8, ptr addrspace(3) [[BASEPTR0]], i64 4096
  // CHECK: cp.async.bulk.tensor

  // CHECK: [[OFFSET2:%.*]] = getelementptr i8, ptr addrspace(3) [[BASEPTR0]], i64 512
  // CHECK: cp.async.bulk.tensor

  // CHECK: [[OFFSET3:%.*]] = getelementptr i8, ptr addrspace(3) [[BASEPTR0]], i64 4608
  // CHECK: cp.async.bulk.tensor
  ttng.async_tma_gather %arg0[%arg2, %arg3] %arg4, %arg1, %arg5 : !tt.tensordesc<tensor<1x128xbf16, #shared1>>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>, i32, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<32x128xbf16, #shared1, #smem, mutable>, i1

  // CHECK-NEXT: ret void
  tt.return
}
}
