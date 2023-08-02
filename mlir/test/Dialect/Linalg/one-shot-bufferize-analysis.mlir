// RUN: mlir-opt %s -one-shot-bufferize="allow-return-allocs bufferize-function-boundaries test-analysis-only" -split-input-file | FileCheck %s

// CHECK-LABEL: @elementwise_no_conflict
func.func @elementwise_no_conflict(%a: tensor<5xf32>,
                                   %b: tensor<5xf32>) -> tensor<5xf32> {
  // CHECK: linalg.elemwise_binary
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "true"], fun = #linalg.binary_fn<add>}
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
      ins(%a, %b : tensor<5xf32>, tensor<5xf32>)
      outs(%a : tensor<5xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
}

// -----

// CHECK-LABEL: @elementwise_no_conflict_2
func.func @elementwise_no_conflict_2(%a: tensor<5xf32>) -> tensor<5xf32> {
  // CHECK: linalg.elemwise_binary
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "true"], fun = #linalg.binary_fn<add>}
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
      ins(%a, %a : tensor<5xf32>, tensor<5xf32>)
      outs(%a : tensor<5xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
}
