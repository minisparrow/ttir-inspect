#nvmma_32 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: make_tensor_descriptor
  // CHECK: %0 = arith.extsi %arg2 : i32 to i64
  // CHECK: %1 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
  // CHECK: ttng.tensormap_create %1, %arg0, [%c32_i32, %c8_i32], [%arg2, %arg1], [%0], [%c1_i32, %c1_i32] {elem_type = 0 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 1 : i32} : (!tt.ptr<i8>, !tt.ptr<i8>, i32, i32, i32, i32, i64, i32, i32) -> ()
  // CHECK: ttng.tensormap_fenceproxy_acquire %1 : !tt.ptr<i8>
  // CHECK: ttng.reinterpret_tensor_descriptor %1 : !tt.ptr<i8> to !tt.tensordesc<tensor<8x32xi8, #shared>>
  tt.func public @make_tensor_descriptor(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32} ) -> !tt.tensordesc<tensor<8x32xi8, #nvmma_32>> {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<32> : tensor<8x1xi32>
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = arith.extsi %arg2 : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : !tt.ptr<i8>, !tt.tensordesc<tensor<8x32xi8, #nvmma_32>>
    tt.return %1 : !tt.tensordesc<tensor<8x32xi8, #nvmma_32>>
  }
}
