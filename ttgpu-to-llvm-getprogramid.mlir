// RUN: triton-opt %s --allocate-shared-memory-nv --convert-triton-gpu-to-llvm -reconcile-unrealized-casts 2>/dev/null | FileCheck %s --dump-input-context 20
// RUN: triton-opt $s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_program_id
  tt.func @basic_program_id() {
    // CHECK: nvvm.read.ptx.sreg.ctaid.x : i32
    %0 = tt.get_program_id x : i32
    tt.return
  }
}

