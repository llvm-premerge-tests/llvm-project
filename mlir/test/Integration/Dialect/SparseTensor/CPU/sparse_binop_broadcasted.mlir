// DEFINE: %{option} = "enable-runtime-library=true enable-index-reduction=true"
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = "enable-runtime-library=false enable-index-reduction=true"
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false enable-index-reduction=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{compile} | %{run}

// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=4 enable-index-reduction=true enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli_host_or_aarch64_cmd \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#trait = {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>
  ],
  iterator_types = ["parallel", "parallel"]
}

#COO = #sparse_tensor.encoding<{lvlTypes = ["compressed-nu", "singleton"]}>
#CSR = #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>
#DCSR = #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed" ] }
>
// An example of a 2D convolution with a sparse filter.
module {
  func.func @broadcast_add(%lhs: tensor<8x8xi32>, %rhs: tensor<1x8xi32>) -> tensor<8x8xi32> {
    %out = tensor.empty() : tensor<8x8xi32>
    %4 = linalg.generic #trait ins(%lhs, %rhs : tensor<8x8xi32>, tensor<1x8xi32>) outs(%out : tensor<8x8xi32>) {
    ^bb0(%in: i32, %in_4: i32, %o: i32):
      %9 = arith.addi %in, %in_4 : i32
      linalg.yield %9 : i32
    } -> tensor<8x8xi32>
    return %4 : tensor<8x8xi32>
  }

  func.func @broadcast_add_coo(%lhs: tensor<8x8xi32, #COO>, %rhs: tensor<1x8xi32, #COO>) -> tensor<8x8xi32, #COO> {
    %out = tensor.empty() : tensor<8x8xi32, #COO>
    %4 = linalg.generic #trait ins(%lhs, %rhs : tensor<8x8xi32, #COO>, tensor<1x8xi32, #COO>) outs(%out : tensor<8x8xi32, #COO>) {
    ^bb0(%in: i32, %in_4: i32, %o: i32):
      %9 = arith.addi %in, %in_4 : i32
      linalg.yield %9 : i32
    } -> tensor<8x8xi32, #COO>
    return %4 : tensor<8x8xi32, #COO>
  }

  func.func @broadcast_add_csr(%lhs: tensor<8x8xi32, #CSR>, %rhs: tensor<1x8xi32, #CSR>) -> tensor<8x8xi32, #CSR> {
    %out = tensor.empty() : tensor<8x8xi32, #CSR>
    %4 = linalg.generic #trait ins(%lhs, %rhs : tensor<8x8xi32, #CSR>, tensor<1x8xi32, #CSR>) outs(%out : tensor<8x8xi32, #CSR>) {
    ^bb0(%in: i32, %in_4: i32, %o: i32):
      %9 = arith.addi %in, %in_4 : i32
      linalg.yield %9 : i32
    } -> tensor<8x8xi32, #CSR>
    return %4 : tensor<8x8xi32, #CSR>
  }

  func.func @broadcast_add_dcsr(%lhs: tensor<8x8xi32, #DCSR>, %rhs: tensor<1x8xi32, #DCSR>) -> tensor<8x8xi32, #DCSR> {
    %out = tensor.empty() : tensor<8x8xi32, #DCSR>
    %4 = linalg.generic #trait ins(%lhs, %rhs : tensor<8x8xi32, #DCSR>, tensor<1x8xi32, #DCSR>) outs(%out : tensor<8x8xi32, #DCSR>) {
    ^bb0(%in: i32, %in_4: i32, %o: i32):
      %9 = arith.addi %in, %in_4 : i32
      linalg.yield %9 : i32
    } -> tensor<8x8xi32, #DCSR>
    return %4 : tensor<8x8xi32, #DCSR>
  }

  func.func @entry() {
    %c0 = arith.constant 0 : index
    %i0 = arith.constant 0 : i32

    %lhs = arith.constant dense<[
      [  1,  2,  3,  4,  0,  6,  7,  8 ],
      [  2,  2,  4,  4,  0,  0,  6,  8 ],
      [  2,  0,  4,  4,  0,  0,  6,  8 ],
      [  2,  0,  3,  4,  0,  0,  7,  8 ],
      [  1,  3,  0,  4,  0,  0,  6,  8 ],
      [  3,  2,  0,  4,  0,  0,  7,  8 ],
      [  1,  3,  3,  4,  3,  6,  6,  0 ],
      [  1,  3,  3,  4,  3,  0,  7,  0 ]
    ]> : tensor<8x8xi32>

    %rhs = arith.constant dense<[
      [  1,  2,  0,  4,  0,  6,  0,  8 ]
    ]> : tensor<1x8xi32>

    %rhs_DCSR = sparse_tensor.convert %rhs
      : tensor<1x8xi32> to tensor<1x8xi32, #DCSR>
    %rhs_CSR = sparse_tensor.convert %rhs
      : tensor<1x8xi32> to tensor<1x8xi32, #CSR>
    %rhs_COO = sparse_tensor.convert %rhs
      : tensor<1x8xi32> to tensor<1x8xi32, #COO>

    %lhs_DCSR = sparse_tensor.convert %lhs
      : tensor<8x8xi32> to tensor<8x8xi32, #DCSR>
    %lhs_CSR = sparse_tensor.convert %lhs
      : tensor<8x8xi32> to tensor<8x8xi32, #CSR>
    %lhs_COO = sparse_tensor.convert %lhs
      : tensor<8x8xi32> to tensor<8x8xi32, #COO>


    // Call the kernel.
    %0 = call @broadcast_add(%lhs, %rhs)
       : (tensor<8x8xi32>, tensor<1x8xi32>) -> tensor<8x8xi32>
    %1 = call @broadcast_add_dcsr(%lhs_DCSR, %rhs_DCSR)
       : (tensor<8x8xi32, #DCSR>, tensor<1x8xi32, #DCSR>) -> tensor<8x8xi32, #DCSR>
    %2 = call @broadcast_add_csr(%lhs_CSR, %rhs_CSR)
       : (tensor<8x8xi32, #CSR>, tensor<1x8xi32, #CSR>) -> tensor<8x8xi32, #CSR>
    %3 = call @broadcast_add_coo(%lhs_COO, %rhs_COO)
       : (tensor<8x8xi32, #COO>, tensor<1x8xi32, #COO>) -> tensor<8x8xi32, #COO>


    // Verify the output.
    // CHECK:     ( ( 2, 4, 3, 8, 0, 12, 7, 16 ),
    // CHECK-SAME:  ( 3, 4, 4, 8, 0, 6, 6, 16 ),
    // CHECK-SAME:  ( 3, 2, 4, 8, 0, 6, 6, 16 ),
    // CHECK-SAME:  ( 3, 2, 3, 8, 0, 6, 7, 16 ),
    // CHECK-SAME:  ( 2, 5, 0, 8, 0, 6, 6, 16 ),
    // CHECK-SAME:  ( 4, 4, 0, 8, 0, 6, 7, 16 ),
    // CHECK-SAME:  ( 2, 5, 3, 8, 3, 12, 6, 8 ),
    // CHECK-SAME:  ( 2, 5, 3, 8, 3, 6, 7, 8 ) )
    %v = vector.transfer_read %0[%c0, %c0], %i0 : tensor<8x8xi32>, vector<8x8xi32>
    vector.print %v : vector<8x8xi32>

    // Verify the output.
    // CHECK-NEXT:     ( ( 2, 4, 3, 8, 0, 12, 7, 16 ),
    // CHECK-SAME:  ( 3, 4, 4, 8, 0, 6, 6, 16 ),
    // CHECK-SAME:  ( 3, 2, 4, 8, 0, 6, 6, 16 ),
    // CHECK-SAME:  ( 3, 2, 3, 8, 0, 6, 7, 16 ),
    // CHECK-SAME:  ( 2, 5, 0, 8, 0, 6, 6, 16 ),
    // CHECK-SAME:  ( 4, 4, 0, 8, 0, 6, 7, 16 ),
    // CHECK-SAME:  ( 2, 5, 3, 8, 3, 12, 6, 8 ),
    // CHECK-SAME:  ( 2, 5, 3, 8, 3, 6, 7, 8 ) )
    %dcsr_ret = sparse_tensor.convert %1 : tensor<8x8xi32, #DCSR> to tensor<8x8xi32>
    %v0 = vector.transfer_read %dcsr_ret[%c0, %c0], %i0 : tensor<8x8xi32>, vector<8x8xi32>
    vector.print %v0 : vector<8x8xi32>

    // Verify the output.
    // CHECK-NEXT:     ( ( 2, 4, 3, 8, 0, 12, 7, 16 ),
    // CHECK-SAME:  ( 3, 4, 4, 8, 0, 6, 6, 16 ),
    // CHECK-SAME:  ( 3, 2, 4, 8, 0, 6, 6, 16 ),
    // CHECK-SAME:  ( 3, 2, 3, 8, 0, 6, 7, 16 ),
    // CHECK-SAME:  ( 2, 5, 0, 8, 0, 6, 6, 16 ),
    // CHECK-SAME:  ( 4, 4, 0, 8, 0, 6, 7, 16 ),
    // CHECK-SAME:  ( 2, 5, 3, 8, 3, 12, 6, 8 ),
    // CHECK-SAME:  ( 2, 5, 3, 8, 3, 6, 7, 8 ) )
    %csr_ret = sparse_tensor.convert %2: tensor<8x8xi32, #CSR> to tensor<8x8xi32>
    %v1 = vector.transfer_read %csr_ret[%c0, %c0], %i0 : tensor<8x8xi32>, vector<8x8xi32>
    vector.print %v1 : vector<8x8xi32>

    // Verify the output.
    // CHECK-NEXT:     ( ( 2, 4, 3, 8, 0, 12, 7, 16 ),
    // CHECK-SAME:  ( 3, 4, 4, 8, 0, 6, 6, 16 ),
    // CHECK-SAME:  ( 3, 2, 4, 8, 0, 6, 6, 16 ),
    // CHECK-SAME:  ( 3, 2, 3, 8, 0, 6, 7, 16 ),
    // CHECK-SAME:  ( 2, 5, 0, 8, 0, 6, 6, 16 ),
    // CHECK-SAME:  ( 4, 4, 0, 8, 0, 6, 7, 16 ),
    // CHECK-SAME:  ( 2, 5, 3, 8, 3, 12, 6, 8 ),
    // CHECK-SAME:  ( 2, 5, 3, 8, 3, 6, 7, 8 ) )
    %coo_ret = sparse_tensor.convert %3 : tensor<8x8xi32, #COO> to tensor<8x8xi32>
    %v2= vector.transfer_read %coo_ret[%c0, %c0], %i0 : tensor<8x8xi32>, vector<8x8xi32>
    vector.print %v2 : vector<8x8xi32>

    // TODO: Release the resources.
    return
  }
}
