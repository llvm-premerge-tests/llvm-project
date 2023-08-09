// RUN: mlir-opt -convert-func-to-llvm='use-opaque-pointers=1' %s | FileCheck %s

// RUN-32: mlir-opt -convert-func-to-llvm='data-layout=p:32:32:32 use-opaque-pointers=1' %s | FileCheck %s

// RUN-32: mlir-opt -test-transform-dialect-interpreter %s | FileCheck %s

// CHECK: module attributes {llvm.data_layout = ""}
// CHECK-32: module attributes {llvm.data_layout ="p:32:32:32"}
module {}

// TODO: right now this cannot apply as we cannot put the transform.sequence 
// anywhere that would let us apply a transform to the top-level module.
// Encapsulating in another module does not help as the data layout is not 
// current applied to anything else than the top-level module.
transform.sequence failures(propagate) {
^bb1(%toplevel_module: !transform.any_op):
  transform.apply_conversion_patterns to %toplevel_module {
    transform.apply_conversion_patterns.func.func_to_llvm
  } with type_converter {
    transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
      {data_layout = "p:32:32:32", use_opaque_pointers = true}
  } {
    legal_dialects = ["llvm"], 
    legal_ops = ["builtin.module"], 
    partial_conversion
  } : !transform.any_op
}
