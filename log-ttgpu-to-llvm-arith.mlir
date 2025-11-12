#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @basic_addf(%arg0: !llvm.struct<(f32, f32)>, %arg1: !llvm.struct<(f32, f32)>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
    %0 = builtin.unrealized_conversion_cast %arg1 : !llvm.struct<(f32, f32)> to tensor<256xf32, #blocked>
    %1 = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(f32, f32)> to tensor<256xf32, #blocked>
    %2 = llvm.extractvalue %arg0[0] : !llvm.struct<(f32, f32)> 
    %3 = llvm.extractvalue %arg0[1] : !llvm.struct<(f32, f32)> 
    %4 = llvm.extractvalue %arg1[0] : !llvm.struct<(f32, f32)> 
    %5 = llvm.extractvalue %arg1[1] : !llvm.struct<(f32, f32)> 
    %6 = llvm.fadd %2, %4 : f32
    %7 = llvm.fadd %3, %5 : f32
    %8 = llvm.mlir.undef : !llvm.struct<(f32, f32)>
    %9 = llvm.insertvalue %6, %8[0] : !llvm.struct<(f32, f32)> 
    %10 = llvm.insertvalue %7, %9[1] : !llvm.struct<(f32, f32)> 
    llvm.return
  }
}

