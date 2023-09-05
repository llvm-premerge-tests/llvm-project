// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def TestMutate : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $src, MIFlag<FmNoInfs>, MIFlags<[FmNoNans, FmNoInfs]>)),
  (apply (G_SEXT $dst, $src, MIFlag<FmReassoc>, MIFlags<[FmReassoc, FmAfn]>))>;

def TestNewInst : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $src, MIFlag<FmReassoc>, MIFlags<[FmAfn, FmReassoc]>)),
  (apply (G_FADD $dst, $src, $src, MIFlag<FmNoNans>, MIFlags<[FmNoNans, FmAfn]>))>;

// CHECK:      const int64_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static int64_t MatchTable0[] = {
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 0*/ 49,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_ZEXT,
// CHECK-NEXT:       GIM_Try, /*On fail goto*//*Label 1*/ 20, // Rule ID 0 //
// CHECK-NEXT:         GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule0Enabled,
// CHECK-NEXT:         GIM_CheckHasMIFlag, /*MI*/0, MachineInstr::FmNoInfs | MachineInstr::FmNoNans,
// CHECK-NEXT:         // MIs[0] dst
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] src
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // Combiner Rule #0: TestMutate
// CHECK-NEXT:         GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::G_SEXT,
// CHECK-NEXT:         GIR_SetMIFlags, /*InsnID*/0, MachineInstr::FmReassoc | MachineInstr::FmAfn,
// CHECK-NEXT:         GIR_Done,
// CHECK-NEXT:       // Label 1: @20
// CHECK-NEXT:       GIM_Try, /*On fail goto*//*Label 2*/ 48, // Rule ID 1 //
// CHECK-NEXT:         GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule1Enabled,
// CHECK-NEXT:         GIM_CheckHasMIFlag, /*MI*/0, MachineInstr::FmAfn | MachineInstr::FmReassoc,
// CHECK-NEXT:         // MIs[0] dst
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] src
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // Combiner Rule #1: TestNewInst
// CHECK-NEXT:         GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::G_FADD,
// CHECK-NEXT:         GIR_SetMIFlags, /*InsnID*/0, MachineInstr::FmNoNans | MachineInstr::FmAfn,
// CHECK-NEXT:         GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:         GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
// CHECK-NEXT:         GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
// CHECK-NEXT:         GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:         GIR_Done,
// CHECK-NEXT:       // Label 2: @48
// CHECK-NEXT:       GIM_Reject,
// CHECK-NEXT:     // Label 0: @49
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     };
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }

def MyCombiner: GICombiner<"GenMyCombiner", [
  TestMutate,
  TestNewInst
]>;
