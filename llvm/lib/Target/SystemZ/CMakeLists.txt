add_llvm_component_group(SystemZ HAS_JIT)

set(LLVM_TARGET_DEFINITIONS SystemZ.td)

tablegen(LLVM SystemZGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM SystemZGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM SystemZGenCallingConv.inc -gen-callingconv)
tablegen(LLVM SystemZGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM SystemZGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM SystemZGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM SystemZGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM SystemZGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM SystemZGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM SystemZ.txt -print-records)

add_public_tablegen_target(SystemZCommonTableGen)

############### Begin Mdl
if(LLVM_ENABLE_MDL)
  add_custom_target(TdScanSystemZ
    COMMAND tdscan -gen_arch_spec --nowarnings SystemZ.txt
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating SystemZ.mdl..."
                    DEPENDS SystemZCommonTableGen
                    VERBATIM)

  set(MdlDatabaseSystemZ
    "${CMAKE_CURRENT_BINARY_DIR}/SystemZGenMdlInfo.inc"
    "${CMAKE_CURRENT_BINARY_DIR}/SystemZGenMdlInfo.h"
    "${CMAKE_CURRENT_BINARY_DIR}/SystemZGenMdlTarget.inc"
  )

  add_custom_command(OUTPUT ${MdlDatabaseSystemZ}
    COMMAND mdl SystemZ.mdl
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating SystemZGenMdlInfo.inc..."
                    DEPENDS TdScanSystemZ
                    VERBATIM)

  # tablegen macro
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${MdlDatabaseSystemZ})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/SystemZGenMdlInfo.inc")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/SystemZGenMdlInfo.h")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/SystemZGenMdlTarget.inc")
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${MdlDatabaseSystemZ} PROPERTIES GENERATED 1)

  # add_public_tablegen_target macro
  add_custom_target(MdlSystemZ DEPENDS ${TABLEGEN_OUTPUT})

  if(LLVM_COMMON_DEPENDS)
    add_dependencies(MdlSystemZ ${LLVM_COMMON_DEPENDS})
  endif()

  set_target_properties(MdlSystemZ PROPERTIES FOLDER "MdlGeneration")
  set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} MdlSystemZ)
endif()
############### End Md

add_llvm_target(SystemZCodeGen
  SystemZAsmPrinter.cpp
  SystemZCallingConv.cpp
  SystemZConstantPoolValue.cpp
  SystemZCopyPhysRegs.cpp
  SystemZElimCompare.cpp
  SystemZFrameLowering.cpp
  SystemZHazardRecognizer.cpp
  SystemZISelDAGToDAG.cpp
  SystemZISelLowering.cpp
  SystemZInstrInfo.cpp
  SystemZLDCleanup.cpp
  SystemZLongBranch.cpp
  SystemZMachineFunctionInfo.cpp
  SystemZMachineScheduler.cpp
  SystemZMCInstLower.cpp
  SystemZPostRewrite.cpp
  SystemZRegisterInfo.cpp
  SystemZSelectionDAGInfo.cpp
  SystemZShortenInst.cpp
  SystemZSubtarget.cpp
  SystemZTargetMachine.cpp
  SystemZTargetTransformInfo.cpp
  SystemZTDC.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  MC
  Scalar
  SelectionDAG
  Support
  SystemZDesc
  SystemZInfo
  Target
  TargetParser

  ADD_TO_COMPONENT
  SystemZ
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
