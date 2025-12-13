#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase=1, maxPhase=1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
#mma0 = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma0, kWidth=2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_ldmatrix
  tt.func @local_alloc_ldmatrix(%A: tensor<16x16xf16, #blocked0>) {
    %AA = ttg.local_alloc %A : (tensor<16x16xf16, #blocked0>) -> !ttg.memdesc<16x16xf16, #shared0, #smem>
    %AA_DOT = ttg.local_load %AA : !ttg.memdesc<16x16xf16, #shared0, #smem> -> tensor<16x16xf16, #dot_operand_a>
    tt.return
  }
}
