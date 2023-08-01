// RUN: mlir-opt %s -test-transform-dialect-interpreter -test-transform-dialect-erase-schedule -one-shot-bufferize -func-bufferize -lower-vector-mask --test-lower-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
func.func @main() {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  %da = arith.constant dense<[
          [ 1.1, 2.1 ],
          [ 1.2, 2.2 ],
          [ 1.3, 2.3 ],
          [ 1.4, 2.4 ],
          [ 1.5, 2.5 ],
          [ 1.6, 2.6 ],
          [ 1.7, 2.7 ],
          [ 1.8, 2.8 ]
      ]> : tensor<8x2xf32>
  %db = arith.constant dense<[
          [ 10.1, 11.1, 12.1, 13.1 ],
          [ 10.2, 11.2, 12.2, 13.2 ]
      ]> : tensor<2x4xf32>
  %C = bufferization.alloc_tensor(%c8, %c4) : tensor<?x?xf32>

  %da1 = tensor.cast %da : tensor<8x2xf32> to tensor<?x?xf32>
  %db1 = tensor.cast %db : tensor<2x4xf32> to tensor<?x?xf32>

  %c0_i32 = arith.constant  0 : i32
  %C0 = linalg.fill ins(%c0_i32 : i32) outs(%C : tensor<?x?xf32>) -> tensor<?x?xf32>

  %1 = linalg.matmul ins(%da1, %db1: tensor<?x?xf32>, tensor<?x?xf32>)
            outs(%C0: tensor<?x?xf32>) -> tensor<?x?xf32>
  %xu = tensor.cast %1 : tensor<?x?xf32> to tensor<*xf32>

  // CHECK:      {{\[}}[32.53,   35.73,   38.93,   42.13],
  // CHECK-NEXT: [34.56,   37.96,   41.36,   44.76],
  // CHECK-NEXT: [36.59,   40.19,   43.79,   47.39],
  // CHECK-NEXT: [38.62,   42.42,   46.22,   50.02],
  // CHECK-NEXT: [40.65,   44.65,   48.65,   52.65],
  // CHECK-NEXT: [42.68,   46.88,   51.08,   55.28],
  // CHECK-NEXT: [44.71,   49.11,   53.51,   57.91],
  // CHECK-NEXT: [46.74,   51.34,   55.94,   60.54]]
  call @printMemrefF32(%xu) : (tensor<*xf32>) -> ()

  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %func_op = get_parent_op %0 : (!transform.any_op) -> !transform.op<"func.func">
  transform.structured.masked_vectorize %0 vector_sizes [8, 4, 2] : !transform.any_op
  transform.apply_patterns to %func_op {
    transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerreduction"
  } : !transform.op<"func.func">
}
