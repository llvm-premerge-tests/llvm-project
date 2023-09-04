add_mlir_dialect_library(MLIRBufferizationDialect
  AllocationOpInterface.cpp
  BufferizableOpInterface.cpp
  BufferizationOps.cpp
  BufferizationDialect.cpp
  InferDestinationOpInterface

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Bufferization

  DEPENDS
  MLIRAllocationOpInterfaceIncGen
  MLIRBufferizationOpsIncGen
  MLIRBufferizationEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRDestinationStyleOpInterface
  MLIRDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSparseTensorDialect
  MLIRTensorDialect
  MLIRMemRefDialect
  )
