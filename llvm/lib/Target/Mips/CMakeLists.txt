add_llvm_component_group(Mips HAS_JIT)

set(LLVM_TARGET_DEFINITIONS Mips.td)

tablegen(LLVM MipsGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM MipsGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM MipsGenCallingConv.inc -gen-callingconv)
tablegen(LLVM MipsGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM MipsGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM MipsGenFastISel.inc -gen-fast-isel)
tablegen(LLVM MipsGenGlobalISel.inc -gen-global-isel)
tablegen(LLVM MipsGenPostLegalizeGICombiner.inc -gen-global-isel-combiner-matchtable
              -combiners="MipsPostLegalizerCombiner")
tablegen(LLVM MipsGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM MipsGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM MipsGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM MipsGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM MipsGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM MipsGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM MipsGenExegesis.inc -gen-exegesis)
tablegen(LLVM Mips.txt -print-records)

add_public_tablegen_target(MipsCommonTableGen)

############### Begin Mdl
if(LLVM_ENABLE_MDL)
  add_custom_target(TdScanMips
    COMMAND tdscan -gen_arch_spec --nowarnings Mips.txt
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating Mips.mdl..."
                    DEPENDS MipsCommonTableGen
                    VERBATIM)

  set(MdlDatabaseMips
    "${CMAKE_CURRENT_BINARY_DIR}/MipsGenMdlInfo.inc"
    "${CMAKE_CURRENT_BINARY_DIR}/MipsGenMdlInfo.h"
    "${CMAKE_CURRENT_BINARY_DIR}/MipsGenMdlTarget.inc"
  )

  add_custom_command(OUTPUT ${MdlDatabaseMips}
    COMMAND mdl Mips.mdl
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating MipsGenMdlInfo.inc..."
                    DEPENDS TdScanMips
                    VERBATIM)

  # tablegen macro
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${MdlDatabaseMips})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/MipsGenMdlInfo.inc")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/MipsGenMdlInfo.h")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/MipsGenMdlTarget.inc")
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${MdlDatabaseMips} PROPERTIES GENERATED 1)

  # add_public_tablegen_target macro
  add_custom_target(MdlMips DEPENDS ${TABLEGEN_OUTPUT})

  if(LLVM_COMMON_DEPENDS)
    add_dependencies(MdlMips ${LLVM_COMMON_DEPENDS})
  endif()

  set_target_properties(MdlMips PROPERTIES FOLDER "MdlGeneration")
  set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} MdlMips)
endif()
############### End Mdl

add_llvm_target(MipsCodeGen
  Mips16FrameLowering.cpp
  Mips16HardFloat.cpp
  Mips16HardFloatInfo.cpp
  Mips16InstrInfo.cpp
  Mips16ISelDAGToDAG.cpp
  Mips16ISelLowering.cpp
  Mips16RegisterInfo.cpp
  MipsAnalyzeImmediate.cpp
  MipsAsmPrinter.cpp
  MipsCallLowering.cpp
  MipsCCState.cpp
  MipsConstantIslandPass.cpp
  MipsDelaySlotFiller.cpp
  MipsExpandPseudo.cpp
  MipsFastISel.cpp
  MipsInstrInfo.cpp
  MipsInstructionSelector.cpp
  MipsISelDAGToDAG.cpp
  MipsISelLowering.cpp
  MipsFrameLowering.cpp
  MipsLegalizerInfo.cpp
  MipsBranchExpansion.cpp
  MipsMCInstLower.cpp
  MipsMachineFunction.cpp
  MipsModuleISelDAGToDAG.cpp
  MipsOptimizePICCall.cpp
  MipsOs16.cpp
  MipsPreLegalizerCombiner.cpp
  MipsPostLegalizerCombiner.cpp
  MipsRegisterBankInfo.cpp
  MipsRegisterInfo.cpp
  MipsSEFrameLowering.cpp
  MipsSEInstrInfo.cpp
  MipsSEISelDAGToDAG.cpp
  MipsSEISelLowering.cpp
  MipsSERegisterInfo.cpp
  MipsSubtarget.cpp
  MipsTargetMachine.cpp
  MipsTargetObjectFile.cpp
  MipsTargetTransformInfo.cpp
  MicroMipsSizeReduction.cpp
  MipsMulMulBugPass.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  MC
  MipsDesc
  MipsInfo
  SelectionDAG
  Support
  Target
  TargetParser

  ADD_TO_COMPONENT
  Mips
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
