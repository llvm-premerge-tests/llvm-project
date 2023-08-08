add_mlir_dialect_library(MLIRControlFlowTransformOps
  ArithTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ControlFlow/TransformOps

  DEPENDS
  MLIRControlFlowTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRControlFlowDialect
  MLIRControlFlowToLLVM
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRTransformDialect
)
