#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @parallel_deps(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %arg4: tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) -> (tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %0:2 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %cst, %arg7 = %cst) -> (tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) {
      %1 = tt.load %arg3 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %2 = tt.load %arg3 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %3 = arith.addf %arg6, %1 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %4 = arith.addf %arg7, %2 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      scf.yield %3, %4 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    } {tt.scheduled_max_stage = 2 : i32}
    tt.return %0#0, %0#1 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
  }
}

