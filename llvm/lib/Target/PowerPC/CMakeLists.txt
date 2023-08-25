add_llvm_component_group(PowerPC HAS_JIT)

set(LLVM_TARGET_DEFINITIONS PPC.td)

tablegen(LLVM PPCGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM PPCGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM PPCGenCallingConv.inc -gen-callingconv)
tablegen(LLVM PPCGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM PPCGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM PPCGenFastISel.inc -gen-fast-isel)
tablegen(LLVM PPCGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM PPCGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM PPCGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM PPCGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM PPCGenExegesis.inc -gen-exegesis)
tablegen(LLVM PPCGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM PPCGenGlobalISel.inc -gen-global-isel)
tablegen(LLVM PPC.txt -print-records)

add_public_tablegen_target(PowerPCCommonTableGen)

############### Begin Mdl
if(LLVM_ENABLE_MDL)
  add_custom_target(TdScanPPC
    COMMAND tdscan -gen_arch_spec --nowarnings PPC.txt
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating PPC.mdl..."
                    DEPENDS PowerPCCommonTableGen
                    VERBATIM)

  set(MdlDatabasePPC
    "${CMAKE_CURRENT_BINARY_DIR}/PPCGenMdlInfo.inc"
    "${CMAKE_CURRENT_BINARY_DIR}/PPCGenMdlInfo.h"
    "${CMAKE_CURRENT_BINARY_DIR}/PPCGenMdlTarget.inc"
  )

  add_custom_command(OUTPUT ${MdlDatabasePPC}
    COMMAND mdl PPC.mdl
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating PPCGenMdlInfo.inc..."
                    DEPENDS TdScanPPC
                    VERBATIM)

  # tablegen macro
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${MdlDatabasePPC})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/PPCGenMdlInfo.inc")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/PPCGenMdlInfo.h")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/PPCGenMdlTarget.inc")
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${MdlDatabasePPC} PROPERTIES GENERATED 1)

  # add_public_tablegen_target macro
  add_custom_target(MdlPPC DEPENDS ${TABLEGEN_OUTPUT})

  if(LLVM_COMMON_DEPENDS)
    add_dependencies(MdlPPC ${LLVM_COMMON_DEPENDS})
  endif()

  set_target_properties(MdlPPC PROPERTIES FOLDER "MdlGeneration")
  set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} MdlPPC)
endif()
############### End Mdl

add_llvm_target(PowerPCCodeGen
  GISel/PPCInstructionSelector.cpp
  PPCBoolRetToInt.cpp
  PPCAsmPrinter.cpp
  PPCBranchSelector.cpp
  PPCBranchCoalescing.cpp
  PPCCallingConv.cpp
  PPCCCState.cpp
  PPCCTRLoops.cpp
  PPCCTRLoopsVerify.cpp
  PPCExpandAtomicPseudoInsts.cpp
  PPCHazardRecognizers.cpp
  PPCInstrInfo.cpp
  PPCISelDAGToDAG.cpp
  PPCISelLowering.cpp
  PPCEarlyReturn.cpp
  PPCFastISel.cpp
  PPCFrameLowering.cpp
  PPCLoopInstrFormPrep.cpp
  PPCMCInstLower.cpp
  PPCMachineFunctionInfo.cpp
  PPCMachineScheduler.cpp
  PPCMacroFusion.cpp
  PPCMIPeephole.cpp
  PPCRegisterInfo.cpp
  PPCSubtarget.cpp
  PPCTargetMachine.cpp
  PPCTargetObjectFile.cpp
  PPCTargetTransformInfo.cpp
  PPCTOCRegDeps.cpp
  PPCTLSDynamicCall.cpp
  PPCVSXCopy.cpp
  PPCReduceCRLogicals.cpp
  PPCVSXFMAMutate.cpp
  PPCVSXSwapRemoval.cpp
  PPCExpandISEL.cpp
  PPCPreEmitPeephole.cpp
  PPCLowerMASSVEntries.cpp
  PPCGenScalarMASSEntries.cpp
  GISel/PPCCallLowering.cpp
  GISel/PPCRegisterBankInfo.cpp
  GISel/PPCLegalizerInfo.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  BinaryFormat
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  MC
  PowerPCDesc
  PowerPCInfo
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  PowerPC
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
