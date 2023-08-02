add_mlir_library(MLIRTargetLLVM
  ModuleToObject.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVM

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  IPO
  Passes
  Support
  Target
  TargetParser
  LINK_LIBS PUBLIC
  MLIRExecutionEngineUtils
  MLIRTargetLLVMIRExport
)

if (MLIR_ENABLE_CUDA_CONVERSIONS)
  set(NVPTX_LIBS
    NVPTXCodeGen
    NVPTXDesc
    NVPTXInfo
  )
endif()

add_mlir_dialect_library(MLIRNVVMTarget
  NVVM/Target.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  LINK_COMPONENTS
  Core
  MC
  Target
  ${NVPTX_LIBS}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRExecutionEngineUtils
  MLIRSupport
  MLIRGPUDialect
  MLIRTargetLLVM
  MLIRNVVMToLLVMIRTranslation
  )

if(MLIR_ENABLE_CUDA_CONVERSIONS)
  # Find the CUDA toolkit.
  if (NOT DEFINED CUDAToolkit_ROOT)
    find_package(CUDAToolkit)
    get_filename_component(CUDAToolkit_ROOT ${CUDAToolkit_BIN_DIR} DIRECTORY ABSOLUTE)
  endif()
  message(VERBOSE "MLIR Default CUDA toolkit path: ${CUDAToolkit_ROOT}")

  # Define the `CUDAToolkit` path.
  target_compile_definitions(obj.MLIRNVVMTarget
    PRIVATE
    __DEFAULT_CUDATOOLKIT_PATH__="${CUDAToolkit_ROOT}"
  )
endif()
