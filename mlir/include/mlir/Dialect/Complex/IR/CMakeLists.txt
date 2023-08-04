add_mlir_dialect(ComplexOps complex)
add_mlir_doc(ComplexOps ComplexOps Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS ComplexAttributes.td)
mlir_tablegen(ComplexAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=complex)
mlir_tablegen(ComplexAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=complex)
add_public_tablegen_target(MLIRComplexAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS ComplexOpsInterfaces.td)
mlir_tablegen(ComplexOpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ComplexOpsInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRComplexOpsInterfacesIncGen)
