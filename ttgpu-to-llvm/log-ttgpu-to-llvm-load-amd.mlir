#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @basic_load(%arg0: !llvm.struct<(ptr<1>, ptr<1>)>, %arg1: !llvm.struct<(i1, i1)>, %arg2: !llvm.struct<(f32, f32)>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>) attributes {nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
    %0 = builtin.unrealized_conversion_cast %arg2 : !llvm.struct<(f32, f32)> to tensor<256xf32, #blocked>
    %1 = builtin.unrealized_conversion_cast %arg1 : !llvm.struct<(i1, i1)> to tensor<256xi1, #blocked>
    %2 = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(ptr<1>, ptr<1>)> to tensor<256x!tt.ptr<f32>, #blocked>
    %3 = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<1>, ptr<1>)> 
    %4 = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<1>, ptr<1>)> 
    %5 = llvm.extractvalue %arg1[0] : !llvm.struct<(i1, i1)> 
    %6 = llvm.extractvalue %arg1[1] : !llvm.struct<(i1, i1)> 
    %7 = llvm.extractvalue %arg2[0] : !llvm.struct<(f32, f32)> 
    %8 = llvm.extractvalue %arg2[1] : !llvm.struct<(f32, f32)> 
    %9 = llvm.mlir.constant(dense<0.000000e+00> : vector<1xf32>) : vector<1xf32>
    %10 = llvm.mlir.undef : vector<1xf32>
    %11 = llvm.mlir.constant(0 : i32) : i32
    %12 = llvm.insertelement %7, %10[%11 : i32] : vector<1xf32>
    llvm.cond_br %5, ^bb1, ^bb2(%12 : vector<1xf32>)
  ^bb1:  // pred: ^bb0
    %13 = llvm.load %3 : !llvm.ptr<1> -> vector<1xf32>
    llvm.br ^bb2(%13 : vector<1xf32>)
  ^bb2(%14: vector<1xf32>):  // 2 preds: ^bb0, ^bb1
    %15 = llvm.mlir.constant(0 : index) : i32
    %16 = llvm.extractelement %14[%15 : i32] : vector<1xf32>
    %17 = llvm.mlir.constant(dense<0.000000e+00> : vector<1xf32>) : vector<1xf32>
    %18 = llvm.mlir.undef : vector<1xf32>
    %19 = llvm.mlir.constant(0 : i32) : i32
    %20 = llvm.insertelement %8, %18[%19 : i32] : vector<1xf32>
    llvm.cond_br %6, ^bb3, ^bb4(%20 : vector<1xf32>)
  ^bb3:  // pred: ^bb2
    %21 = llvm.load %4 : !llvm.ptr<1> -> vector<1xf32>
    llvm.br ^bb4(%21 : vector<1xf32>)
  ^bb4(%22: vector<1xf32>):  // 2 preds: ^bb2, ^bb3
    %23 = llvm.mlir.constant(0 : index) : i32
    %24 = llvm.extractelement %22[%23 : i32] : vector<1xf32>
    %25 = llvm.mlir.undef : !llvm.struct<(f32, f32)>
    %26 = llvm.insertvalue %16, %25[0] : !llvm.struct<(f32, f32)> 
    %27 = llvm.insertvalue %24, %26[1] : !llvm.struct<(f32, f32)> 
    llvm.return
  }
}

