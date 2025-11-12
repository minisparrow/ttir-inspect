module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @sliced_layout_make_range(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 32>} {
    %0 = llvm.mlir.constant(0 : index) : i32
    %1 = nvvm.read.ptx.sreg.tid.x : i32
    %2 = llvm.mlir.constant(31 : i32) : i32
    %3 = llvm.and %1, %2 : i32
    %4 = llvm.mlir.constant(32 : i32) : i32
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = nvgpu.cluster_id
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.constant(0 : i32) : i32
    %11 = llvm.mlir.constant(0 : i32) : i32
    %12 = llvm.mlir.constant(0 : i32) : i32
    %13 = llvm.shl %3, %12 : i32
    %14 = llvm.or %11, %13 : i32
    %15 = llvm.mlir.constant(5 : i32) : i32
    %16 = llvm.shl %6, %15 : i32
    %17 = llvm.or %14, %16 : i32
    %18 = llvm.mlir.constant(3 : i32) : i32
    %19 = llvm.and %17, %18 : i32
    %20 = llvm.mlir.constant(2 : i32) : i32
    %21 = llvm.shl %19, %20 : i32
    %22 = llvm.mlir.constant(0 : i32) : i32
    %23 = llvm.mlir.constant(0 : i32) : i32
    %24 = llvm.or disjoint %21, %23 : i32
    %25 = llvm.xor %10, %24 : i32
    %26 = llvm.mlir.constant(0 : i32) : i32
    %27 = llvm.xor %25, %26 : i32
    %28 = llvm.mlir.constant(1 : i32) : i32
    %29 = llvm.xor %25, %28 : i32
    %30 = llvm.mlir.constant(2 : i32) : i32
    %31 = llvm.xor %25, %30 : i32
    %32 = llvm.mlir.constant(3 : i32) : i32
    %33 = llvm.xor %25, %32 : i32
    %34 = llvm.add %27, %0 : i32
    %35 = llvm.add %29, %0 : i32
    %36 = llvm.add %31, %0 : i32
    %37 = llvm.add %33, %0 : i32
    %38 = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32)>
    %39 = llvm.insertvalue %34, %38[0] : !llvm.struct<(i32, i32, i32, i32)> 
    %40 = llvm.insertvalue %35, %39[1] : !llvm.struct<(i32, i32, i32, i32)> 
    %41 = llvm.insertvalue %36, %40[2] : !llvm.struct<(i32, i32, i32, i32)> 
    %42 = llvm.insertvalue %37, %41[3] : !llvm.struct<(i32, i32, i32, i32)> 
    llvm.return
  }
}

