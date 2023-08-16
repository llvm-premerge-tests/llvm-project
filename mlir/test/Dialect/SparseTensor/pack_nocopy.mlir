// RUN: mlir-opt %s --sparse-compiler="enable-runtime-library=false" | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  lvlTypes = ["dense","compressed"],
  dimToLvl = affine_map<(i,j) -> (i,j)>,
  crdWidth = 32,
  posWidth = 32
}>

#trait_scale = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = X(i,j) * 2"
}

//
// Pass in the buffers of the sparse tensor, marked non-writable.
//
func.func @foo(%arg0: tensor<3xf64>  {bufferization.writable = false},
               %arg1: tensor<3xi32>  {bufferization.writable = false},
	       %arg2: tensor<11xi32> {bufferization.writable = false}) -> (index) {
    //
    // Pack the buffers into a sparse tensors. Note that
    // using the buffers after this in any aliasing manner
    // is undefined, so we should avoid introducing copies.
    //
    // CHECK-LABEL: llvm.func @foo
    // CHECK-NOT: malloc
    //
    %pack = sparse_tensor.pack %arg0, %arg2, %arg1
      : tensor<3xf64>,
        tensor<11xi32>,
	tensor<3xi32> to tensor<10x10xf64, #CSR>

    //
    // Scale the sparse tensor "in-place" (this has no impact on the final
    // number of entries, but introduces a write into the value buffer).
    //
    %c = arith.constant 2.0 : f64
    %s = linalg.generic #trait_scale
      outs(%pack: tensor<10x10xf64, #CSR>) {
         ^bb(%x: f64):
          %1 = arith.mulf %x, %c : f64
          linalg.yield %1 : f64
    } -> tensor<10x10xf64, #CSR>

    //
    // Return number of entries in the scaled sparse tensor.
    //
    %nse = sparse_tensor.number_of_entries %s : tensor<10x10xf64, #CSR>
    return %nse : index
}
