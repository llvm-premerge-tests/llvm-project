add_llvm_component_group(Lanai)

set(LLVM_TARGET_DEFINITIONS Lanai.td)

tablegen(LLVM LanaiGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM LanaiGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM LanaiGenCallingConv.inc -gen-callingconv)
tablegen(LLVM LanaiGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM LanaiGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM LanaiGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM LanaiGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM LanaiGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM LanaiGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM Lanai.txt -print-records)

add_public_tablegen_target(LanaiCommonTableGen)

############### Begin Mdl
if(LLVM_ENABLE_MDL)
  add_custom_target(TdScanLanai
    COMMAND tdscan -gen_arch_spec --nowarnings Lanai.txt
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating Lanai.mdl..."
                    DEPENDS LanaiCommonTableGen
                    VERBATIM)

  set(MdlDatabaseLanai
    "${CMAKE_CURRENT_BINARY_DIR}/LanaiGenMdlInfo.inc"
    "${CMAKE_CURRENT_BINARY_DIR}/LanaiGenMdlInfo.h"
    "${CMAKE_CURRENT_BINARY_DIR}/LanaiGenMdlTarget.inc"
  )

  add_custom_command(OUTPUT ${MdlDatabaseLanai}
    COMMAND mdl Lanai.mdl
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating LanaiGenMdlInfo.inc..."
                    DEPENDS TdScanLanai
                    VERBATIM)

  # tablegen macro
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${MdlDatabaseLanai})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/LanaiGenMdlInfo.inc")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/LanaiGenMdlInfo.h")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/LanaiGenMdlTarget.inc")
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${MdlDatabaseLanai} PROPERTIES GENERATED 1)

  # add_public_tablegen_target macro
  add_custom_target(MdlLanai DEPENDS ${TABLEGEN_OUTPUT})

  if(LLVM_COMMON_DEPENDS)
    add_dependencies(MdlLanai ${LLVM_COMMON_DEPENDS})
  endif()

  set_target_properties(MdlLanai PROPERTIES FOLDER "MdlGeneration")
  set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} MdlLanai)
endif()
############### End Mdl

add_llvm_target(LanaiCodeGen
  LanaiAsmPrinter.cpp
  LanaiDelaySlotFiller.cpp
  LanaiFrameLowering.cpp
  LanaiInstrInfo.cpp
  LanaiISelDAGToDAG.cpp
  LanaiISelLowering.cpp
  LanaiMachineFunctionInfo.cpp
  LanaiMCInstLower.cpp
  LanaiMemAluCombiner.cpp
  LanaiRegisterInfo.cpp
  LanaiSelectionDAGInfo.cpp
  LanaiSubtarget.cpp
  LanaiTargetMachine.cpp
  LanaiTargetObjectFile.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  LanaiAsmParser
  LanaiDesc
  LanaiInfo
  MC
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  Lanai
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
