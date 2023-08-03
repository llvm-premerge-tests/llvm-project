// RUN: mlir-opt %s --test-transform-dialect-interpreter --verify-diagnostics --split-input-file

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  %sp_kernel = transform.sparse_tensor.match.sparse_inout %matmul
        : !transform.any_op -> !transform.any_op
}

func.func @matmul_simple(%lhs: tensor<10x20xf16>,
                         %rhs: tensor<20x15xf32>) -> tensor<10x15xf64>{
  %cst = arith.constant 0.0 : f64
  %empty = tensor.empty() : tensor<10x15xf64>
  %fill = linalg.fill ins(%cst : f64) outs(%empty : tensor<10x15xf64>) -> tensor<10x15xf64>
  // expected-error @below {{Operation has no sparse input or output}}
  %result = linalg.matmul ins(%lhs, %rhs: tensor<10x20xf16>, tensor<20x15xf32>) outs(%fill: tensor<10x15xf64>) -> tensor<10x15xf64>
  return %result : tensor<10x15xf64>
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  %sp_kernel = transform.sparse_tensor.match.sparse_inout %matmul
        : !transform.any_op -> !transform.any_op
  transform.test_print_remark_at_operand %sp_kernel, "sparse_kernel" : !transform.any_op
}

#CSR = #sparse_tensor.encoding<{lvlTypes = ["dense", "compressed"]}>

func.func @matmul_simple(%lhs: tensor<10x20xf16, #CSR>,
                         %rhs: tensor<20x15xf32>) -> tensor<10x15xf64>{
  %cst = arith.constant 0.0 : f64
  %empty = tensor.empty() : tensor<10x15xf64>
  %fill = linalg.fill ins(%cst : f64) outs(%empty : tensor<10x15xf64>) -> tensor<10x15xf64>
  // expected-remark @below {{sparse_kernel}}
  %result = linalg.matmul ins(%lhs, %rhs: tensor<10x20xf16, #CSR>, tensor<20x15xf32>) outs(%fill: tensor<10x15xf64>) -> tensor<10x15xf64>
  return %result : tensor<10x15xf64>
}
