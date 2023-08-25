add_llvm_component_group(X86 HAS_JIT)

set(LLVM_TARGET_DEFINITIONS X86.td)

tablegen(LLVM X86GenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM X86GenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM X86GenAsmWriter1.inc -gen-asm-writer -asmwriternum=1)
tablegen(LLVM X86GenCallingConv.inc -gen-callingconv)
tablegen(LLVM X86GenDAGISel.inc -gen-dag-isel)
tablegen(LLVM X86GenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM X86GenEVEX2VEXTables.inc -gen-x86-EVEX2VEX-tables)
tablegen(LLVM X86GenExegesis.inc -gen-exegesis)
tablegen(LLVM X86GenFastISel.inc -gen-fast-isel)
tablegen(LLVM X86GenGlobalISel.inc -gen-global-isel)
tablegen(LLVM X86GenInstrInfo.inc -gen-instr-info
                                  -instr-info-expand-mi-operand-info=0)
tablegen(LLVM X86GenMnemonicTables.inc -gen-x86-mnemonic-tables -asmwriternum=1)
tablegen(LLVM X86GenRegisterBank.inc -gen-register-bank)
tablegen(LLVM X86GenRegisterInfo.inc -gen-register-info)
tablegen(LLVM X86GenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM X86.txt -print-records)
tablegen(LLVM X86GenFoldTables.inc -gen-x86-fold-tables -asmwriternum=1)

add_public_tablegen_target(X86CommonTableGen)

############### Begin Mdl
if(LLVM_ENABLE_MDL)
  add_custom_target(TdScanX86
    COMMAND tdscan -gen_arch_spec --nowarnings X86.txt
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                    COMMENT "Generating X86.mdl..."
                    DEPENDS X86CommonTableGen
                    VERBATIM)

  set(MdlDatabaseX86
    "${CMAKE_CURRENT_BINARY_DIR}/X86GenMdlInfo.inc"
    "${CMAKE_CURRENT_BINARY_DIR}/X86GenMdlInfo.h"
    "${CMAKE_CURRENT_BINARY_DIR}/X86GenMdlTarget.inc"
  )

  add_custom_command(OUTPUT ${MdlDatabaseX86}
    COMMAND mdl X86.mdl
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating X86GenMdlInfo.inc..."
                    DEPENDS TdScanX86
                    VERBATIM)

  # tablegen macro
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${MdlDatabaseX86})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/X86GenMdlInfo.inc")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/X86GenMdlInfo.h")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/X86GenMdlTarget.inc")
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${MdlDatabaseX86} PROPERTIES GENERATED 1)

  # add_public_tablegen_target macro
  add_custom_target(MdlX86 DEPENDS ${TABLEGEN_OUTPUT})

  if(LLVM_COMMON_DEPENDS)
    add_dependencies(MdlX86 ${LLVM_COMMON_DEPENDS})
  endif()

  set_target_properties(MdlX86 PROPERTIES FOLDER "MdlGeneration")
  set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} MdlX86)
endif()
############### End Mdl

set(sources
  X86ArgumentStackSlotRebase.cpp
  X86AsmPrinter.cpp
  X86AvoidTrailingCall.cpp
  X86CallFrameOptimization.cpp
  X86CallingConv.cpp
  X86CmovConversion.cpp
  X86DomainReassignment.cpp
  X86DiscriminateMemOps.cpp
  X86LowerTileCopy.cpp
  X86LowerAMXType.cpp
  X86PreAMXConfig.cpp
  X86LowerAMXIntrinsics.cpp
  X86TileConfig.cpp
  X86FastPreTileConfig.cpp
  X86FastTileConfig.cpp
  X86PreTileConfig.cpp
  X86ExpandPseudo.cpp
  X86FastISel.cpp
  X86FixupBWInsts.cpp
  X86FixupLEAs.cpp
  X86FixupInstTuning.cpp
  X86FixupVectorConstants.cpp
  X86AvoidStoreForwardingBlocks.cpp
  X86DynAllocaExpander.cpp
  X86FixupSetCC.cpp
  X86FlagsCopyLowering.cpp
  X86FloatingPoint.cpp
  X86FrameLowering.cpp
  X86ISelDAGToDAG.cpp
  X86ISelLowering.cpp
  X86ISelLoweringCall.cpp
  X86IndirectBranchTracking.cpp
  X86IndirectThunks.cpp
  X86InterleavedAccess.cpp
  X86InsertPrefetch.cpp
  X86InstCombineIntrinsic.cpp
  X86InstrFMA3Info.cpp
  X86InstrFoldTables.cpp
  X86InstrInfo.cpp
  X86EvexToVex.cpp
  X86LoadValueInjectionLoadHardening.cpp
  X86LoadValueInjectionRetHardening.cpp
  X86MCInstLower.cpp
  X86MachineFunctionInfo.cpp
  X86MacroFusion.cpp
  X86OptimizeLEAs.cpp
  X86PadShortFunction.cpp
  X86PartialReduction.cpp
  X86RegisterInfo.cpp
  X86ReturnThunks.cpp
  X86SelectionDAGInfo.cpp
  X86ShuffleDecodeConstantPool.cpp
  X86SpeculativeLoadHardening.cpp
  X86SpeculativeExecutionSideEffectSuppression.cpp
  X86Subtarget.cpp
  X86TargetMachine.cpp
  X86TargetObjectFile.cpp
  X86TargetTransformInfo.cpp
  X86VZeroUpper.cpp
  X86WinEHState.cpp
  X86InsertWait.cpp
  GISel/X86CallLowering.cpp
  GISel/X86InstructionSelector.cpp
  GISel/X86LegalizerInfo.cpp
  GISel/X86RegisterBankInfo.cpp
  )

add_llvm_target(X86CodeGen ${sources}
  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CFGuard
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  Instrumentation
  MC
  ProfileData
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  X86Desc
  X86Info

  ADD_TO_COMPONENT
  X86
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCA)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
