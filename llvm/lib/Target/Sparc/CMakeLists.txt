add_llvm_component_group(Sparc)

set(LLVM_TARGET_DEFINITIONS Sparc.td)

tablegen(LLVM SparcGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM SparcGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM SparcGenCallingConv.inc -gen-callingconv)
tablegen(LLVM SparcGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM SparcGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM SparcGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM SparcGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM SparcGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM SparcGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM Sparc.txt -print-records)

add_public_tablegen_target(SparcCommonTableGen)

############### Begin Mdl
if(LLVM_ENABLE_MDL)
  add_custom_target(TdScanSparc
    COMMAND tdscan --family_name=SP -gen_arch_spec --nowarnings Sparc.txt
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating Sparc.mdl..."
                    DEPENDS SparcCommonTableGen
                    VERBATIM)

  set(MdlDatabaseSparc
    "${CMAKE_CURRENT_BINARY_DIR}/SparcGenMdlInfo.inc"
    "${CMAKE_CURRENT_BINARY_DIR}/SparcGenMdlInfo.h"
    "${CMAKE_CURRENT_BINARY_DIR}/SparcGenMdlTarget.inc"
  )

  add_custom_command(OUTPUT ${MdlDatabaseSparc}
    COMMAND mdl Sparc.mdl
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating SparcGenMdlInfo.inc..."
                    DEPENDS TdScanSparc
                    VERBATIM)

  # tablegen macro
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${MdlDatabaseSparc})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/SparcGenMdlInfo.inc")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/SparcGenMdlInfo.h")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/SparcGenMdlTarget.inc")
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${MdlDatabaseSparc} PROPERTIES GENERATED 1)

  # add_public_tablegen_target macro
  add_custom_target(MdlSparc DEPENDS ${TABLEGEN_OUTPUT})

  if(LLVM_COMMON_DEPENDS)
    add_dependencies(MdlSparc ${LLVM_COMMON_DEPENDS})
  endif()

  set_target_properties(MdlSparc PROPERTIES FOLDER "MdlGeneration")
  set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} MdlSparc)
endif()
############### End Mdl

add_llvm_target(SparcCodeGen
  DelaySlotFiller.cpp
  LeonPasses.cpp
  SparcAsmPrinter.cpp
  SparcInstrInfo.cpp
  SparcISelDAGToDAG.cpp
  SparcISelLowering.cpp
  SparcFrameLowering.cpp
  SparcMachineFunctionInfo.cpp
  SparcRegisterInfo.cpp
  SparcSubtarget.cpp
  SparcTargetMachine.cpp
  SparcMCInstLower.cpp
  SparcTargetObjectFile.cpp

  LINK_COMPONENTS
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  MC
  SelectionDAG
  SparcDesc
  SparcInfo
  Support
  Target
  TargetParser

  ADD_TO_COMPONENT
  Sparc
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
