add_llvm_component_group(RISCV)

set(LLVM_TARGET_DEFINITIONS RISCV.td)

tablegen(LLVM RISCVGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM RISCVGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM RISCVGenCompressInstEmitter.inc -gen-compress-inst-emitter)
tablegen(LLVM RISCVGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM RISCVGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM RISCVGenGlobalISel.inc -gen-global-isel)
tablegen(LLVM RISCVGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM RISCVGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM RISCVGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM RISCVGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM RISCVGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM RISCVGenSearchableTables.inc -gen-searchable-tables)
tablegen(LLVM RISCVGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM RISCV.txt -print-records)

add_public_tablegen_target(RISCVCommonTableGen)

############### Begin Mdl
if(LLVM_ENABLE_MDL)
  add_custom_target(TdScanRISCV
    COMMAND tdscan -gen_arch_spec --nowarnings RISCV.txt
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating RISCV.mdl..."
                    DEPENDS RISCVCommonTableGen
                    VERBATIM)

  set(MdlDatabaseRISCV
    "${CMAKE_CURRENT_BINARY_DIR}/RISCVGenMdlInfo.inc"
    "${CMAKE_CURRENT_BINARY_DIR}/RISCVGenMdlInfo.h"
    "${CMAKE_CURRENT_BINARY_DIR}/RISCVGenMdlTarget.inc"
  )

  add_custom_command(OUTPUT ${MdlDatabaseRISCV}
    COMMAND mdl RISCV.mdl
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating RISCVGenMdlInfo.inc..."
                    DEPENDS TdScanRISCV
                    VERBATIM)

  # tablegen macro
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${MdlDatabaseRISCV})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/RISCVGenMdlInfo.inc")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/RISCVGenMdlInfo.h")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/RISCVGenMdlTarget.inc")
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${MdlDatabaseRISCV} PROPERTIES GENERATED 1)

  # add_public_tablegen_target macro
  add_custom_target(MdlRISCV DEPENDS ${TABLEGEN_OUTPUT})

  if(LLVM_COMMON_DEPENDS)
    add_dependencies(MdlRISCV ${LLVM_COMMON_DEPENDS})
  endif()

  set_target_properties(MdlRISCV PROPERTIES FOLDER "MdlGeneration")
  set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} MdlRISCV)
endif()
############### End Mdl

add_llvm_target(RISCVCodeGen
  RISCVAsmPrinter.cpp
  RISCVCodeGenPrepare.cpp
  RISCVMakeCompressible.cpp
  RISCVExpandAtomicPseudoInsts.cpp
  RISCVExpandPseudoInsts.cpp
  RISCVFrameLowering.cpp
  RISCVGatherScatterLowering.cpp
  RISCVInsertVSETVLI.cpp
  RISCVInsertReadWriteCSR.cpp
  RISCVInstrInfo.cpp
  RISCVISelDAGToDAG.cpp
  RISCVISelLowering.cpp
  RISCVMachineFunctionInfo.cpp
  RISCVMacroFusion.cpp
  RISCVMergeBaseOffset.cpp
  RISCVOptWInstrs.cpp
  RISCVRedundantCopyElimination.cpp
  RISCVMoveMerger.cpp
  RISCVPushPopOptimizer.cpp
  RISCVRegisterInfo.cpp
  RISCVRVVInitUndef.cpp
  RISCVSubtarget.cpp
  RISCVTargetMachine.cpp
  RISCVTargetObjectFile.cpp
  RISCVTargetTransformInfo.cpp
  GISel/RISCVCallLowering.cpp
  GISel/RISCVInstructionSelector.cpp
  GISel/RISCVLegalizerInfo.cpp
  GISel/RISCVRegisterBankInfo.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  IPO
  MC
  RISCVDesc
  RISCVInfo
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  RISCV
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(MCA)
add_subdirectory(TargetInfo)
