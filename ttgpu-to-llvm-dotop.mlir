
#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase=1, maxPhase=1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma0 = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma0, kWidth=2}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_ldmatrix
  tt.func @convert_dot_ldmatrix(%A: tensor<16x16xf16, #blocked0>, %B: tensor<16x16xf16, #blocked0>) {
    %AA = ttg.local_alloc %A : (tensor<16x16xf16, #blocked0>) -> !ttg.memdesc<16x16xf16, #shared0, #smem>
    %BB = ttg.local_alloc %B : (tensor<16x16xf16, #blocked0>) -> !ttg.memdesc<16x16xf16, #shared0, #smem>
    // CHECK: nvvm.ldmatrix %{{.*}} {eltType = #nvvm.ld_st_matrix_elt_type<b16>, layout = #nvvm.mma_layout<row>, num = 4 : i32, shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    // CHECK: nvvm.ldmatrix %{{.*}} {eltType = #nvvm.ld_st_matrix_elt_type<b16>, layout = #nvvm.mma_layout<col>, num = 4 : i32, shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    // CHECK-NOT: nvvm.ldmatrix
    %AA_DOT = ttg.local_load %AA : !ttg.memdesc<16x16xf16, #shared0, #smem> -> tensor<16x16xf16, #dot_operand_a>
    %BB_DOT = ttg.local_load %BB : !ttg.memdesc<16x16xf16, #shared0, #smem> -> tensor<16x16xf16, #dot_operand_b>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma0>

    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    %D = tt.dot %AA_DOT, %BB_DOT, %cst0 : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>

    tt.return
  }
}

