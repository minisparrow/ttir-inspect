
// RUN: triton-opt %s --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1], [2], [16], [0]], lane = [[0], [0], [0], [0], [0]], warp = [[4], [8]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-LABEL: @tma_scatter
tt.func @tma_scatter(%arg0: !tt.tensordesc<tensor<1x128xbf16, #shared1>>, %arg1: tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, %arg2: i32, %arg3: !ttg.memdesc<32x128xbf16, #shared1, #smem, mutable>) {
  // The lowering for `async_tma_scatter` shares practically all of its logic
  // with `async_tma_gather`, so we don't need to re-test the indexing logic.

  // CHECK: [[BASE_PTR:%.*]] = extractvalue {{.*}} %3, 0
  // CHECK: [[ELECT:%.*]] = tail call { i32, i1 } @llvm.nvvm.elect.sync
  // CHECK: [[PRED:%.*]] = extractvalue { i32, i1 } [[ELECT]], 1

  // CHECK: [[PTR:%.*]] = getelementptr {{.*}} [[BASE_PTR]]
  // CHECK-NEXT: "@$0 cp.async.bulk.tensor.2d.tile::scatter4.global.shared::cta.bulk_group [$1, {$2, $3, $4, $5, $6}], [$7];"
  // CHECK-SAME: (i1 [[PRED]], ptr nonnull %0, i32 %2, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, ptr addrspace(3) [[PTR]])
  ttng.async_tma_scatter %arg0[%arg1, %arg2] %arg3 : !tt.tensordesc<tensor<1x128xbf16, #shared1>>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, !ttg.memdesc<32x128xbf16, #shared1, #smem, mutable>

  // CHECK: nvvm.cp.async.bulk.commit.group()

  // CHECK-NEXT: ret void
  tt.return
}
}
