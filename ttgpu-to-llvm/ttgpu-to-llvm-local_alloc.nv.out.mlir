module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 512 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @basic_alloc_tensor(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %2 = llvm.getelementptr %1[%0] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.undef : !llvm.struct<(ptr<3>, i32, i32)>
    %5 = llvm.insertvalue %2, %4[0] : !llvm.struct<(ptr<3>, i32, i32)> 
    %6 = llvm.insertvalue %3, %5[1] : !llvm.struct<(ptr<3>, i32, i32)> 
    %7 = llvm.insertvalue %3, %6[2] : !llvm.struct<(ptr<3>, i32, i32)> 
    llvm.return
  }
}

