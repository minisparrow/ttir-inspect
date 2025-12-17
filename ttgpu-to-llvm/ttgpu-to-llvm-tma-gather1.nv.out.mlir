module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @tma_gather_simple(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: !llvm.struct<(ptr<3>, i32)>, %arg2: !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>, %arg3: i32, %arg4: !llvm.struct<(ptr<3>, i32, i32)>, %arg5: i1, %arg6: !llvm.ptr<1>, %arg7: !llvm.ptr<1>) attributes {nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
    %0 = llvm.mlir.constant(-1 : i32) : i32
    %1 = llvm.mlir.constant(31 : i32) : i32
    %2 = llvm.mlir.constant(32 : i32) : i32
    %3 = llvm.mlir.constant(3072 : i32) : i32
    %4 = llvm.mlir.constant(1024 : i32) : i32
    %5 = llvm.mlir.constant(64 : i32) : i32
    %6 = llvm.mlir.constant(2048 : i32) : i32
    %7 = llvm.mlir.constant(8 : i32) : i32
    %8 = llvm.mlir.constant(3 : i32) : i32
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.extractvalue %arg1[0] : !llvm.struct<(ptr<3>, i32)> 
    %11 = llvm.extractvalue %arg4[0] : !llvm.struct<(ptr<3>, i32, i32)> 
    %12 = nvvm.read.ptx.sreg.tid.x : i32
    %13 = llvm.udiv %12, %2 : i32
    %14 = nvvm.shfl.sync  idx %0, %13, %9, %1 : i32 -> i32
    %15 = llvm.and %14, %9 : i32
    %16 = llvm.icmp "eq" %9, %15 : i32
    %17 = llvm.and %arg5, %16 : i1
    %18 = nvvm.elect.sync -> i1
    %19 = llvm.and %17, %18 : i1
    %20 = llvm.extractvalue %arg2[0] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)> 
    %21 = llvm.extractvalue %arg2[1] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)> 
    %22 = llvm.extractvalue %arg2[2] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)> 
    %23 = llvm.extractvalue %arg2[3] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)> 
    %24 = llvm.extractvalue %arg2[4] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)> 
    %25 = llvm.extractvalue %arg2[5] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)> 
    %26 = llvm.extractvalue %arg2[6] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)> 
    %27 = llvm.extractvalue %arg2[7] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)> 
    %28 = llvm.shl %14, %9 : i32
    %29 = llvm.or %9, %28 : i32
    %30 = llvm.or %29, %9 : i32
    %31 = llvm.and %30, %8 : i32
    %32 = llvm.shl %31, %7 : i32
    %33 = llvm.or disjoint %32, %9 : i32
    %34 = llvm.xor %9, %33 : i32
    %35 = llvm.getelementptr %11[%34] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, bf16
    %36 = llvm.add %arg3, %9 : i32
    %37 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.tile::gather4.shared::cta.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4, $5, $6, $7}], [$8];", "b,r,l,r,r,r,r,r,r" %19, %35, %arg0, %36, %20, %21, %22, %23, %10 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr<3>) -> !llvm.void
    %38 = llvm.shl %14, %9 : i32
    %39 = llvm.or %9, %38 : i32
    %40 = llvm.or %39, %9 : i32
    %41 = llvm.and %40, %8 : i32
    %42 = llvm.shl %41, %7 : i32
    %43 = llvm.or disjoint %42, %9 : i32
    %44 = llvm.xor %6, %43 : i32
    %45 = llvm.getelementptr %11[%44] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, bf16
    %46 = llvm.add %arg3, %5 : i32
    %47 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.tile::gather4.shared::cta.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4, $5, $6, $7}], [$8];", "b,r,l,r,r,r,r,r,r" %19, %45, %arg0, %46, %20, %21, %22, %23, %10 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr<3>) -> !llvm.void
    %48 = llvm.shl %14, %9 : i32
    %49 = llvm.or %9, %48 : i32
    %50 = llvm.or %49, %9 : i32
    %51 = llvm.and %50, %8 : i32
    %52 = llvm.shl %51, %7 : i32
    %53 = llvm.or disjoint %52, %9 : i32
    %54 = llvm.xor %4, %53 : i32
    %55 = llvm.getelementptr %11[%54] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, bf16
    %56 = llvm.add %arg3, %9 : i32
    %57 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.tile::gather4.shared::cta.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4, $5, $6, $7}], [$8];", "b,r,l,r,r,r,r,r,r" %19, %55, %arg0, %56, %24, %25, %26, %27, %10 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr<3>) -> !llvm.void
    %58 = llvm.shl %14, %9 : i32
    %59 = llvm.or %9, %58 : i32
    %60 = llvm.or %59, %9 : i32
    %61 = llvm.and %60, %8 : i32
    %62 = llvm.shl %61, %7 : i32
    %63 = llvm.or disjoint %62, %9 : i32
    %64 = llvm.xor %3, %63 : i32
    %65 = llvm.getelementptr %11[%64] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, bf16
    %66 = llvm.add %arg3, %5 : i32
    %67 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.tile::gather4.shared::cta.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4, $5, $6, $7}], [$8];", "b,r,l,r,r,r,r,r,r" %19, %65, %arg0, %66, %24, %25, %26, %27, %10 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr<3>) -> !llvm.void
    llvm.return
  }
}

