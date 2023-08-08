set(LLVM_TARGET_DEFINITIONS ControlFlowTransformOps.td)
mlir_tablegen(ControlFlowTransformOps.h.inc -gen-op-decls)
mlir_tablegen(ControlFlowTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRControlFlowTransformOpsIncGen)

add_mlir_doc(ControlFlowTransformOps ControlFlowTransformOps Dialects/ -gen-op-doc)
