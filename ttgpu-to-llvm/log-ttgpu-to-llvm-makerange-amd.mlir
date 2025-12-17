module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @basic_make_range(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 32>} {
    %0 = llvm.mlir.constant(0 : index) : i32
    %1 = rocdl.workitem.id.x : i32
    %2 = llvm.mlir.constant(31 : i32) : i32
    %3 = llvm.and %1, %2 : i32
    %4 = llvm.mlir.constant(32 : i32) : i32
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.constant(0 : i32) : i32
    %11 = llvm.mlir.constant(0 : i32) : i32
    %12 = llvm.shl %3, %11 : i32
    %13 = llvm.or %10, %12 : i32
    %14 = llvm.mlir.constant(31 : i32) : i32
    %15 = llvm.and %13, %14 : i32
    %16 = llvm.mlir.constant(1 : i32) : i32
    %17 = llvm.shl %15, %16 : i32
    %18 = llvm.mlir.constant(0 : i32) : i32
    %19 = llvm.mlir.constant(0 : i32) : i32
    %20 = llvm.or disjoint %17, %19 : i32
    %21 = llvm.xor %9, %20 : i32
    %22 = llvm.mlir.constant(0 : i32) : i32
    %23 = llvm.xor %21, %22 : i32
    %24 = llvm.mlir.constant(1 : i32) : i32
    %25 = llvm.xor %21, %24 : i32
    %26 = llvm.mlir.constant(64 : i32) : i32
    %27 = llvm.xor %21, %26 : i32
    %28 = llvm.mlir.constant(65 : i32) : i32
    %29 = llvm.xor %21, %28 : i32
    %30 = llvm.mlir.constant(128 : i32) : i32
    %31 = llvm.xor %21, %30 : i32
    %32 = llvm.mlir.constant(129 : i32) : i32
    %33 = llvm.xor %21, %32 : i32
    %34 = llvm.mlir.constant(192 : i32) : i32
    %35 = llvm.xor %21, %34 : i32
    %36 = llvm.mlir.constant(193 : i32) : i32
    %37 = llvm.xor %21, %36 : i32
    %38 = llvm.add %23, %0 : i32
    %39 = llvm.add %25, %0 : i32
    %40 = llvm.add %27, %0 : i32
    %41 = llvm.add %29, %0 : i32
    %42 = llvm.add %31, %0 : i32
    %43 = llvm.add %33, %0 : i32
    %44 = llvm.add %35, %0 : i32
    %45 = llvm.add %37, %0 : i32
    %46 = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    %47 = llvm.insertvalue %38, %46[0] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)> 
    %48 = llvm.insertvalue %39, %47[1] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)> 
    %49 = llvm.insertvalue %40, %48[2] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)> 
    %50 = llvm.insertvalue %41, %49[3] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)> 
    %51 = llvm.insertvalue %42, %50[4] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)> 
    %52 = llvm.insertvalue %43, %51[5] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)> 
    %53 = llvm.insertvalue %44, %52[6] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)> 
    %54 = llvm.insertvalue %45, %53[7] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)> 
    llvm.return
  }
}

