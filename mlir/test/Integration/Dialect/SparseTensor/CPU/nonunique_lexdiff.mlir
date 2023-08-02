// This is a regression test to ensure proper handling of non-unique levels.
// In particular, without the D156946 changes to `SparseTensorStorage::lexDiff`
// this code causes the size-assertion in `SparseTensorStorage::toCOO` to fail.

// RUN: mlir-opt %s \
// RUN:  --sparse-compiler=enable-runtime-library=true | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils | \
// RUN: FileCheck %s

#COO = #sparse_tensor.encoding<{lvlTypes = ["compressed-nu", "singleton"]}>
module {
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %i0 = arith.constant 0 : i32

    %input = arith.constant dense<[
      [ 2, 4, 3, 8, 0, 12, 7, 16 ],
      [ 3, 4, 4, 8, 0, 6, 6, 16 ],
      [ 3, 2, 4, 8, 0, 6, 6, 16 ],
      [ 3, 2, 3, 8, 0, 6, 7, 16 ],
      [ 2, 5, 0, 8, 0, 6, 6, 16 ],
      [ 4, 4, 0, 8, 0, 6, 7, 16 ],
      [ 2, 5, 3, 8, 3, 12, 6, 8 ],
      [ 2, 5, 3, 8, 3, 6, 7, 8 ]
    ]> : tensor<8x8xi32>
    %coo = sparse_tensor.convert %input : tensor<8x8xi32> to tensor<8x8xi32, #COO>
    %output = sparse_tensor.convert %coo : tensor<8x8xi32, #COO> to tensor<8x8xi32>

    // CHECK:    ( ( 2, 4, 3, 8, 0, 12, 7, 16 ),
    // CHECK-SAME: ( 3, 4, 4, 8, 0, 6, 6, 16 ),
    // CHECK-SAME: ( 3, 2, 4, 8, 0, 6, 6, 16 ),
    // CHECK-SAME: ( 3, 2, 3, 8, 0, 6, 7, 16 ),
    // CHECK-SAME: ( 2, 5, 0, 8, 0, 6, 6, 16 ),
    // CHECK-SAME: ( 4, 4, 0, 8, 0, 6, 7, 16 ),
    // CHECK-SAME: ( 2, 5, 3, 8, 3, 12, 6, 8 ),
    // CHECK-SAME: ( 2, 5, 3, 8, 3, 6, 7, 8 ) )
    %v = vector.transfer_read %output[%c0, %c0], %i0 : tensor<8x8xi32>, vector<8x8xi32>
    vector.print %v : vector<8x8xi32>

    // Release the resources.
    bufferization.dealloc_tensor %coo : tensor<8x8xi32, #COO>

    return
  }
}
