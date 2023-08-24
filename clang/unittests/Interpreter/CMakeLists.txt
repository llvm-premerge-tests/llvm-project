set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  OrcJIT
  Support
  TargetParser
  )

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
  get_property(LIBCUDART_PATH TARGET CUDA::cudart PROPERTY LOCATION)
  add_compile_definitions(LIBCUDART_PATH="${LIBCUDART_PATH}")
endif()

add_clang_unittest(ClangReplInterpreterTests
  IncrementalProcessingTest.cpp
  InteractiveCudaTest.cpp
  InterpreterTest.cpp
  )
target_link_libraries(ClangReplInterpreterTests PUBLIC
  clangAST
  clangBasic
  clangInterpreter
  clangFrontend
  clangSema
  )

# Exceptions on Windows are not yet supported.
if(NOT WIN32)
  add_subdirectory(ExceptionTests)
endif()

export_executable_symbols(ClangReplInterpreterTests)
