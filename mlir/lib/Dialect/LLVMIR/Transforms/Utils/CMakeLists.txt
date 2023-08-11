add_mlir_dialect_library(MLIRLLVMIRTransformsUtils
  EnsureDistinctSuccessors.cpp

  DEPENDS
  MLIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  )
