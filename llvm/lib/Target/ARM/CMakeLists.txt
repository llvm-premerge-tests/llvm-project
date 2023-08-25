add_llvm_component_group(ARM HAS_JIT)

set(LLVM_TARGET_DEFINITIONS ARM.td)

tablegen(LLVM ARMGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM ARMGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM ARMGenCallingConv.inc -gen-callingconv)
tablegen(LLVM ARMGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM ARMGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM ARMGenFastISel.inc -gen-fast-isel)
tablegen(LLVM ARMGenGlobalISel.inc -gen-global-isel)
tablegen(LLVM ARMGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM ARMGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM ARMGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM ARMGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM ARMGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM ARMGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM ARMGenSystemRegister.inc -gen-searchable-tables)
tablegen(LLVM ARM.txt -print-records)

add_public_tablegen_target(ARMCommonTableGen)

############### Begin Mdl
if(LLVM_ENABLE_MDL)
  add_custom_target(TdScanARM
    COMMAND tdscan -gen_arch_spec --nowarnings ARM.txt
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating ARM.mdl..."
                    DEPENDS ARMCommonTableGen
                    VERBATIM)

  set(MdlDatabaseARM
    "${CMAKE_CURRENT_BINARY_DIR}/ARMGenMdlInfo.inc"
    "${CMAKE_CURRENT_BINARY_DIR}/ARMGenMdlInfo.h"
    "${CMAKE_CURRENT_BINARY_DIR}/ARMGenMdlTarget.inc"
  )

  add_custom_command(OUTPUT ${MdlDatabaseARM}
    COMMAND mdl ARM.mdl
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating ARMGenMdlInfo.inc..."
                    DEPENDS TdScanARM
                    VERBATIM)

  # tablegen macro
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${MdlDatabaseARM})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/ARMGenMdlInfo.inc")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/ARMGenMdlInfo.h")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/ARMGenMdlTarget.inc")
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${MdlDatabaseARM} PROPERTIES GENERATED 1)

  # add_public_tablegen_target macro
  add_custom_target(MdlARM DEPENDS ${TABLEGEN_OUTPUT})

  if(LLVM_COMMON_DEPENDS)
    add_dependencies(MdlARM ${LLVM_COMMON_DEPENDS})
  endif()

  set_target_properties(MdlARM PROPERTIES FOLDER "MdlGeneration")
  set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} MdlARM)
endif()
############### End Mdl

add_llvm_target(ARMCodeGen
  A15SDOptimizer.cpp
  ARMAsmPrinter.cpp
  ARMBaseInstrInfo.cpp
  ARMBaseRegisterInfo.cpp
  ARMBasicBlockInfo.cpp
  ARMBranchTargets.cpp
  ARMCallingConv.cpp
  ARMCallLowering.cpp
  ARMConstantIslandPass.cpp
  ARMConstantPoolValue.cpp
  ARMExpandPseudoInsts.cpp
  ARMFastISel.cpp
  ARMFixCortexA57AES1742098Pass.cpp
  ARMFrameLowering.cpp
  ARMHazardRecognizer.cpp
  ARMInstructionSelector.cpp
  ARMISelDAGToDAG.cpp
  ARMISelLowering.cpp
  ARMInstrInfo.cpp
  ARMLegalizerInfo.cpp
  ARMParallelDSP.cpp
  ARMLoadStoreOptimizer.cpp
  ARMLowOverheadLoops.cpp
  ARMBlockPlacement.cpp
  ARMMCInstLower.cpp
  ARMMachineFunctionInfo.cpp
  ARMMacroFusion.cpp
  ARMRegisterInfo.cpp
  ARMOptimizeBarriersPass.cpp
  ARMRegisterBankInfo.cpp
  ARMSelectionDAGInfo.cpp
  ARMSLSHardening.cpp
  ARMSubtarget.cpp
  ARMTargetMachine.cpp
  ARMTargetObjectFile.cpp
  ARMTargetTransformInfo.cpp
  MLxExpansionPass.cpp
  MVEGatherScatterLowering.cpp
  MVELaneInterleavingPass.cpp
  MVETailPredication.cpp
  MVEVPTBlockPass.cpp
  MVETPAndVPTOptimisationsPass.cpp
  Thumb1FrameLowering.cpp
  Thumb1InstrInfo.cpp
  ThumbRegisterInfo.cpp
  Thumb2ITBlockPass.cpp
  Thumb2InstrInfo.cpp
  Thumb2SizeReduction.cpp

  LINK_COMPONENTS
  ARMDesc
  ARMInfo
  ARMUtils
  Analysis
  AsmPrinter
  CFGuard
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  IPO
  MC
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  ARM
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Utils)
