set(LLVM_TARGET_DEFINITIONS ArithTransformOps.td)
mlir_tablegen(ArithTransformOps.h.inc -gen-op-decls)
mlir_tablegen(ArithTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRArithTransformOpsIncGen)

add_mlir_doc(ArithTransformOps ArithTransformOps Dialects/ -gen-op-doc)
