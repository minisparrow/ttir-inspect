#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @one_dep(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> {
    %c2 = arith.constant 2 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %0 = arith.cmpi slt, %arg0, %arg1 : index
    %1 = tt.splat %0 : i1 -> tensor<128x32xi1, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %2 = tt.load %arg3, %1 : tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %3 = arith.addi %arg0, %arg2 : index
    %4 = arith.cmpi slt, %3, %arg1 : index
    %5 = tt.splat %4 : i1 -> tensor<128x32xi1, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %6 = tt.load %arg3, %5 : tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %7:3 = scf.for %arg4 = %arg0 to %arg1 step %arg2 iter_args(%arg5 = %cst, %arg6 = %2, %arg7 = %6) -> (tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) {
      %8 = arith.muli %arg2, %c2 : index
      %9 = arith.subi %arg1, %8 : index
      %10 = arith.cmpi slt, %arg4, %9 : index
      %11 = arith.addf %arg5, %arg6 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %12 = tt.splat %10 : i1 -> tensor<128x32xi1, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %13 = tt.load %arg3, %12 : tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      scf.yield %11, %arg7, %13 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    }
    tt.return %7#0 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
  }
}

