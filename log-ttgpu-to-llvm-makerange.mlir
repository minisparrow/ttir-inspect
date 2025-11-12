module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @basic_make_range(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 32>} {
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
    %18 = llvm.mlir.constant(31 : i32) : i32
    %19 = llvm.and %17, %18 : i32
    %20 = llvm.mlir.constant(1 : i32) : i32
    %21 = llvm.shl %19, %20 : i32
    %22 = llvm.mlir.constant(0 : i32) : i32
    %23 = llvm.mlir.constant(0 : i32) : i32
    %24 = llvm.or disjoint %21, %23 : i32
    %25 = llvm.xor %10, %24 : i32
    %26 = llvm.mlir.constant(0 : i32) : i32
    %27 = llvm.xor %25, %26 : i32
    %28 = llvm.mlir.constant(1 : i32) : i32
    %29 = llvm.xor %25, %28 : i32
    %30 = llvm.mlir.constant(64 : i32) : i32
    %31 = llvm.xor %25, %30 : i32
    %32 = llvm.mlir.constant(65 : i32) : i32
    %33 = llvm.xor %25, %32 : i32
    %34 = llvm.mlir.constant(128 : i32) : i32
    %35 = llvm.xor %25, %34 : i32
    %36 = llvm.mlir.constant(129 : i32) : i32
    %37 = llvm.xor %25, %36 : i32
    %38 = llvm.mlir.constant(192 : i32) : i32
    %39 = llvm.xor %25, %38 : i32
    %40 = llvm.mlir.constant(193 : i32) : i32
    %41 = llvm.xor %25, %40 : i32
    %42 = llvm.add %27, %0 : i32
    %43 = llvm.add %29, %0 : i32
    %44 = llvm.add %31, %0 : i32
    %45 = llvm.add %33, %0 : i32
    %46 = llvm.add %35, %0 : i32
    %47 = llvm.add %37, %0 : i32
    %48 = llvm.add %39, %0 : i32
    %49 = llvm.add %41, %0 : i32
    %50 = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    %51 = llvm.insertvalue %42, %50[0] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    %52 = llvm.insertvalue %43, %51[1] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    %53 = llvm.insertvalue %44, %52[2] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    %54 = llvm.insertvalue %45, %53[3] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    %55 = llvm.insertvalue %46, %54[4] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    %56 = llvm.insertvalue %47, %55[5] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    %57 = llvm.insertvalue %48, %56[6] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    %58 = llvm.insertvalue %49, %57[7] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    llvm.return
  }
}

