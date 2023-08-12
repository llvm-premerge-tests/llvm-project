add_mlir_library(MLIRQueryMatcher
  Parser.cpp
  Registry.cpp
  VariantValue.cpp
  Diagnostics.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Query/Matcher
  )
