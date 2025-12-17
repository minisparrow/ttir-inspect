ttgpu-to-llvm.mlir:10:10: remark: Warning: vectorization fails vec = 1 origin vec = 1 numElems = 2 mask is 1

    %1 = tt.load %a_ptr_init, %cst, %cst_0 : tensor<256x!tt.ptr<f32>, #blocked0>
         ^
ttgpu-to-llvm.mlir:10:10: note: see current operation: %6 = "tt.load"(%4, %2, %0) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<256x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<256xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<256xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> tensor<256xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
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
    %9 = llvm.mlir.undef : vector<1xf32>
    %10 = llvm.mlir.constant(0 : index) : i32
    %11 = llvm.insertelement %7, %9[%10 : i32] : vector<1xf32>
    %12 = llvm.bitcast %11 : vector<1xf32> to i32
    %13 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, $1;\0A\09@$3 ld.global.b32 { $0 }, [ $2 + 0 ];", "=r,r,l,b" %12, %3, %5 : (i32, !llvm.ptr<1>, i1) -> i32
    %14 = llvm.bitcast %13 : i32 to vector<1xf32>
    %15 = llvm.mlir.constant(0 : index) : i32
    %16 = llvm.extractelement %14[%15 : i32] : vector<1xf32>
    %17 = llvm.mlir.undef : vector<1xf32>
    %18 = llvm.mlir.constant(0 : index) : i32
    %19 = llvm.insertelement %8, %17[%18 : i32] : vector<1xf32>
    %20 = llvm.bitcast %19 : vector<1xf32> to i32
    %21 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, $1;\0A\09@$3 ld.global.b32 { $0 }, [ $2 + 0 ];", "=r,r,l,b" %20, %4, %6 : (i32, !llvm.ptr<1>, i1) -> i32
    %22 = llvm.bitcast %21 : i32 to vector<1xf32>
    %23 = llvm.mlir.constant(0 : index) : i32
    %24 = llvm.extractelement %22[%23 : i32] : vector<1xf32>
    %25 = llvm.mlir.undef : !llvm.struct<(f32, f32)>
    %26 = llvm.insertvalue %16, %25[0] : !llvm.struct<(f32, f32)> 
    %27 = llvm.insertvalue %24, %26[1] : !llvm.struct<(f32, f32)> 
    llvm.return
  }
}

