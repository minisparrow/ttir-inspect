#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @matmul_loop(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    %0 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<1x32xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %4 = tt.addptr %0, %3 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
    %5 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked1>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %8 = tt.broadcast %7 : tensor<1x128xi32, #blocked1> -> tensor<32x128xi32, #blocked1>
    %9 = tt.addptr %5, %8 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
    %cst = arith.constant dense<true> : tensor<128x32xi1, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #blocked>
    %cst_1 = arith.constant dense<true> : tensor<32x128xi1, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_4 = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %cst_5 = arith.constant dense<4> : tensor<32x128xi32, #blocked1>
    %cst_6 = arith.constant dense<4.000000e+00> : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %10:3 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %4, %arg7 = %9, %arg8 = %cst_3) -> (tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>) {
      %11 = tt.load %arg6 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
      %12 = ttg.convert_layout %11 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %13 = tt.load %arg7, %cst_1, %cst_2 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %14 = ttg.convert_layout %13 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %15 = arith.mulf %14, %cst_6 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %16 = tt.dot %12, %15, %arg8 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %17 = tt.addptr %arg6, %cst_4 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %18 = tt.addptr %arg7, %cst_5 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
      scf.yield %17, %18, %16 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>
    } {tt.scheduled_max_stage = 2 : i32}
    tt.return %10#2 : tensor<128x128xf32, #mma>
  }
}

