add_mlir_unittest(MLIRTargetLLVMTests
  Serialize.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

llvm_map_components_to_libnames(llvm_libs nativecodegen)

target_link_libraries(MLIRTargetLLVMTests
  PRIVATE
  MLIRTargetLLVM
  ${dialect_libs}
  ${llvm_libs}
)
