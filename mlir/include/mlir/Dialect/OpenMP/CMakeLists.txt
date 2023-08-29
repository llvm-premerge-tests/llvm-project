set(LLVM_TARGET_DEFINITIONS ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend/OpenMP/OMP.td)
mlir_tablegen(OmpCommon.td --gen-directive-decl --directives-dialect=OpenMP)
add_public_tablegen_target(omp_common_td)

add_mlir_dialect(OpenMPOps omp)
add_mlir_doc(OpenMPOps OpenMPDialect Dialects/ -gen-dialect-doc -dialect=omp)
add_dependencies(OpenMPDialectDocGen omp_common_td)

add_mlir_interface(OpenMPOpsInterfaces)

set(LLVM_TARGET_DEFINITIONS OpenMPTypeInterfaces.td)
mlir_tablegen(OpenMPTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(OpenMPTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIROpenMPTypeInterfacesIncGen)
add_dependencies(mlir-generic-headers MLIROpenMPTypeInterfacesIncGen)
