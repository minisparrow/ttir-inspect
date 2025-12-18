
// RUN: triton-opt %s -allow-unregistered-dialect -split-input-file -tritongpu-schedule-loops -canonicalize | FileCheck %s

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#A = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth=2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 16}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 32]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @one_dep
tt.func @one_dep(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> tensor<128x32xf16, #A> {
  %init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (tensor<128x32xf16, #A>) {
    // CHECK: tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
    %a = tt.load %a_ptr_init {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    // CHECK: arith.addf {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %res = arith.addf %acc, %a : tensor<128x32xf16, #A>
    scf.yield %res : tensor<128x32xf16, #A>
  }
  // CHECK: tt.scheduled_max_stage
  tt.return %loop#0 : tensor<128x32xf16, #A>
}
}