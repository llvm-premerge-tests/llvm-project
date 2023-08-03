// This test ensures proper handling of non-unique levels.
// In particular, without the D156946 changes to `SparseTensorStorage::lexDiff`
// this code causes the size-assertion in `SparseTensorStorage::toCOO` to fail.

// RUN: mlir-opt %s \
// RUN:   --sparse-compiler=enable-runtime-library=true \
// RUN: | mlir-cpu-runner \
// RUN:   -e entry -entry-point-result=void \
// RUN:   -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils \
// RUN: | FileCheck %s

!Dense = tensor<2x2xi32>
!COO = tensor<2x2xi32, #sparse_tensor.encoding<{lvlTypes = ["compressed-nu", "singleton"]}>>
!Vec = vector<2x2xi32>
module {
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %i0 = arith.constant 0 : i32
    %i2 = arith.constant 2 : i32

    %t0 = arith.constant dense<[[1, 0], [3, 4]]> : !Dense
    // This conversion works just fine.
    %t1 = sparse_tensor.convert %t0 : !Dense to !COO
    // This insertion is needed to trigger the crash;
    // without it, the conversion back to dense will work just fine.
    %t2 = sparse_tensor.insert %i2 into %t1[%c0, %c1] : !COO
    // This conversion is the thing that crashes.
    %t3 = sparse_tensor.convert %t2 : !COO to !Dense

    // CHECK: ( ( 1, 2 ), ( 3, 4 ) )
    %v = vector.transfer_read %t3[%c0, %c0], %i0 : !Dense, !Vec
    vector.print %v : !Vec

    // Release the resources.
    bufferization.dealloc_tensor %t1 : !COO
    bufferization.dealloc_tensor %t2 : !COO

    return
  }
}
