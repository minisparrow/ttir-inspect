#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @parallel_deps(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %arg4: tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) -> (tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) {
    %c2 = arith.constant 2 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %0 = arith.cmpi slt, %arg0, %arg1 : index
    %1 = tt.splat %0 : i1 -> tensor<128x32xi1, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %2 = tt.load %arg3, %1 : tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %3 = tt.splat %0 : i1 -> tensor<128x32xi1, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %4 = tt.load %arg3, %3 : tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %5 = arith.addi %arg0, %arg2 : index
    %6 = arith.cmpi slt, %5, %arg1 : index
    %7 = tt.splat %6 : i1 -> tensor<128x32xi1, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %8 = tt.load %arg3, %7 : tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %9 = tt.splat %6 : i1 -> tensor<128x32xi1, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %10 = tt.load %arg3, %9 : tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %11:6 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %cst, %arg7 = %cst, %arg8 = %2, %arg9 = %8, %arg10 = %4, %arg11 = %10) -> (tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) {
      %12 = arith.muli %arg2, %c2 : index
      %13 = arith.subi %arg1, %12 : index
      %14 = arith.cmpi slt, %arg5, %13 : index
      %15 = arith.addf %arg6, %arg8 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %16 = arith.addf %arg7, %arg10 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %17 = tt.splat %14 : i1 -> tensor<128x32xi1, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %18 = tt.load %arg3, %17 : tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %19 = tt.splat %14 : i1 -> tensor<128x32xi1, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %20 = tt.load %arg3, %19 : tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      scf.yield %15, %16, %arg9, %18, %arg11, %20 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    }
    tt.return %11#0, %11#1 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
  }
}

