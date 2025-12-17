// RUN: triton-opt %s --allocate-shared-memory-nv --convert-triton-gpu-to-llvm -reconcile-unrealized-casts 2>/dev/null | FileCheck %s --dump-input-context 20
// RUN: triton-opt $s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942
#shared0 = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: basic_alloc_tensor
  tt.func @basic_alloc_tensor() {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK-NEXT: llvm.getelementptr
    // CHECK-NEXT: llvm.mlir.constant
    %0 = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #shared0, #smem, mutable>
    tt.return
  }
}
