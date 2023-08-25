add_llvm_component_group(Hexagon)

set(LLVM_TARGET_DEFINITIONS Hexagon.td)

tablegen(LLVM HexagonGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM HexagonGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM HexagonGenCallingConv.inc -gen-callingconv)
tablegen(LLVM HexagonGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM HexagonGenDFAPacketizer.inc -gen-dfa-packetizer)
tablegen(LLVM HexagonGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM HexagonGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM HexagonGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM HexagonGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM HexagonGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM Hexagon.txt -print-records)

add_public_tablegen_target(HexagonCommonTableGen)

############### Begin Mdl
if(LLVM_ENABLE_MDL)
  add_custom_target(TdScanHexagon
    COMMAND tdscan -gen_arch_spec --nowarnings Hexagon.txt
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating Hexagon.mdl..."
                    DEPENDS HexagonCommonTableGen
                    VERBATIM)

  set(MdlDatabaseHexagon
    "${CMAKE_CURRENT_BINARY_DIR}/HexagonGenMdlInfo.inc"
    "${CMAKE_CURRENT_BINARY_DIR}/HexagonGenMdlInfo.h"
    "${CMAKE_CURRENT_BINARY_DIR}/HexagonGenMdlTarget.inc"
  )

  add_custom_command(OUTPUT ${MdlDatabaseHexagon}
    COMMAND mdl Hexagon.mdl
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating HexagonGenMdlInfo.inc..."
                    DEPENDS TdScanHexagon
                    VERBATIM)

  # tablegen macro
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${MdlDatabaseHexagon})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/HexagonGenMdlInfo.inc")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/HexagonGenMdlInfo.h")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/HexagonGenMdlTarget.inc")
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${MdlDatabaseHexagon} PROPERTIES GENERATED 1)

  # add_public_tablegen_target macro
  add_custom_target(MdlHexagon DEPENDS ${TABLEGEN_OUTPUT})

  if(LLVM_COMMON_DEPENDS)
    add_dependencies(MdlHexagon ${LLVM_COMMON_DEPENDS})
  endif()

  set_target_properties(MdlHexagon PROPERTIES FOLDER "MdlGeneration")
  set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} MdlHexagon)
endif()
############### End Mdl

add_llvm_target(HexagonCodeGen
  BitTracker.cpp
  HexagonAsmPrinter.cpp
  HexagonBitSimplify.cpp
  HexagonBitTracker.cpp
  HexagonBlockRanges.cpp
  HexagonBranchRelaxation.cpp
  HexagonCFGOptimizer.cpp
  HexagonCommonGEP.cpp
  HexagonConstExtenders.cpp
  HexagonConstPropagation.cpp
  HexagonCopyToCombine.cpp
  HexagonEarlyIfConv.cpp
  HexagonExpandCondsets.cpp
  HexagonFixupHwLoops.cpp
  HexagonFrameLowering.cpp
  HexagonGenExtract.cpp
  HexagonGenInsert.cpp
  HexagonGenMux.cpp
  HexagonGenPredicate.cpp
  HexagonHardwareLoops.cpp
  HexagonHazardRecognizer.cpp
  HexagonInstrInfo.cpp
  HexagonISelDAGToDAG.cpp
  HexagonISelDAGToDAGHVX.cpp
  HexagonISelLowering.cpp
  HexagonISelLoweringHVX.cpp
  HexagonLoopIdiomRecognition.cpp
  HexagonMachineFunctionInfo.cpp
  HexagonMachineScheduler.cpp
  HexagonMCInstLower.cpp
  HexagonNewValueJump.cpp
  HexagonOptAddrMode.cpp
  HexagonOptimizeSZextends.cpp
  HexagonPeephole.cpp
  HexagonRDFOpt.cpp
  HexagonRegisterInfo.cpp
  HexagonSelectionDAGInfo.cpp
  HexagonSplitConst32AndConst64.cpp
  HexagonSplitDouble.cpp
  HexagonStoreWidening.cpp
  HexagonSubtarget.cpp
  HexagonTargetMachine.cpp
  HexagonTargetObjectFile.cpp
  HexagonTargetTransformInfo.cpp
  HexagonVectorCombine.cpp
  HexagonVectorLoopCarriedReuse.cpp
  HexagonVectorPrint.cpp
  HexagonVExtract.cpp
  HexagonVLIWPacketizer.cpp
  RDFCopy.cpp
  RDFDeadCode.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  HexagonAsmParser
  HexagonDesc
  HexagonInfo
  IPO
  MC
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  Hexagon
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
