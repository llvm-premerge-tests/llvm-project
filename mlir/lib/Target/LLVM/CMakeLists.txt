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
  )

if(MLIR_ENABLE_CUDA_RUNNER)
  if(NOT MLIR_ENABLE_CUDA_CONVERSIONS)
    message(SEND_ERROR
      "Building mlir with cuda support requires the NVPTX backend")
  endif()

  # Configure CUDA language support. Using check_language first allows us to
  # give a custom error message.
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(SEND_ERROR
      "Building mlir with cuda support requires a working CUDA install")
  endif()

  # Find the CUDA toolkit.
  if (NOT DEFINED CUDAToolkit_ROOT)
    find_package(CUDAToolkit)
    get_filename_component(CUDAToolkit_ROOT ${CUDAToolkit_BIN_DIR} DIRECTORY ABSOLUTE)
  endif()
  message(VERBOSE "MLIR Default CUDA toolkit path: ${CUDAToolkit_ROOT}")

  # Enable the gpu to cubin target.
  target_compile_definitions(obj.MLIRNVVMTarget
    PRIVATE
    MLIR_GPU_NVPTX_TARGET_ENABLED=1
    __DEFAULT_CUDATOOLKIT_PATH__="${CUDAToolkit_ROOT}"
  )

  # Add CUDA headers includes and the libcuda.so library.
  target_include_directories(obj.MLIRNVVMTarget
    PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
  target_link_libraries(MLIRNVVMTarget
    PRIVATE
    ${CUDA_DRIVER_LIBRARY}
  )
endif()
