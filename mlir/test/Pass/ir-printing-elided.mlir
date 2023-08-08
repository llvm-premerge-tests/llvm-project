// RUN: mlir-opt %s -mlir-disable-threading=true -pass-pipeline='builtin.module(func.func(cse))' -mlir-print-ir-after-all -mlir-print-ir-elided=2 -o /dev/null 2>&1 | FileCheck -check-prefix=ELIDED %s

// The testing output should be elided:
// ELIDED: // -----// IR Dump After{{.*}}CSE (cse) //----- //
// ELIDED-NEXT: func @foo() {
func.func @foo() {
// ELIDED-NEXT:   "test.dense_attr"() {foo.dense_attr = dense_resource<__elided__> : tensor<3xi32>} : () -> ()
  "test.dense_attr"() {foo.dense_attr = dense<[1, 2, 3]> : tensor<3xi32>}: () -> ()
  return
}

// RUN: mlir-opt %s -mlir-disable-threading=true -pass-pipeline='builtin.module(func.func(cse))' -mlir-print-ir-after-all -mlir-print-ir-elided=2 2>/dev/null | FileCheck -check-prefix=UNELIDED %s

// The actual output file shouldn't be elided:
// UNELIDED: module {
// UNELIDED-NEXT:   func.func @foo() {
// UNELIDED-NEXT:     "test.dense_attr"() {foo.dense_attr = dense<[1, 2, 3]> : tensor<3xi32>} : () -> ()
