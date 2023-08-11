add_subdirectory(Utils)

add_mlir_dialect_library(MLIRLLVMIRTransforms
  DIScopeForLLVMFuncOp.cpp
  LegalizeForExport.cpp
  NVVMAttachTarget.cpp
  OptimizeForNVVM.cpp
  RequestCWrappers.cpp
  ROCDLAttachTarget.cpp
  TypeConsistency.cpp

  DEPENDS
  MLIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
  MLIRNVVMDialect
  MLIRNVVMTarget
  MLIRROCDLTarget
  )
