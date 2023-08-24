add_mlir_unittest(MLIRTransformUtilsTests
  CommutativityUtils.cpp
)
target_link_libraries(MLIRTransformUtilsTests
  PRIVATE
  MLIRTransformUtils)
