//===-- RISCVInsertWriteVXRM.cpp - Insert Write of RISC-V VXRM CSR --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function pass that inserts WriteVXRMImm
// for rvv pseudos with a static round mode operand.
//
// To work with the intrinsics that have SideEffects, it checks if there are
// any VXRM uses in the given MachineFunction.
// If there are any, it runs emitWriteVXRMLocal instead which always saves
// incoming VCSR value and restores it for InlineAsm and VXRM users.
//
// For functions without dynamic mode, the pass consists of 3 phases:
//
// Phase 1 collects static round mode changes in each basic block
// Phase 2 propogates the round mode state to successor blocks.
// Between phase 2 and 3 we do a partial redundancy elimination to hoist VXRM
// writes out of simple loops.
// Phase 3 emits WriteVXRMImm, and assumes the incoming VXRM value based on the
// information from Phase 2.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <queue>

using namespace llvm;

#define DEBUG_TYPE "riscv-insert-write-vxrm"
#define RISCV_INSERT_WRITE_VXRM_NAME "RISC-V Insert Write VXRM Pass"

namespace {

class VXRMInfo {
  uint8_t VXRMImm = 0;

  enum : uint8_t {
    Uninitialized,
    Static,
    Unknown,
  } State = Uninitialized;

public:
  VXRMInfo() {}

  static VXRMInfo getUnknown() {
    VXRMInfo Info;
    Info.setUnknown();
    return Info;
  }

  bool isValid() const { return State != Uninitialized; }
  void setUnknown() { State = Unknown; }
  bool isUnknown() const { return State == Unknown; }

  bool isStatic() const { return State == Static; }

  void setVXRMImm(unsigned Imm) {
    assert(Imm <= 3 && "Unexpected VXRM value");
    VXRMImm = Imm;
    State = Static;
  }
  unsigned getVXRMImm() const {
    assert(isStatic() && VXRMImm <= 3 && "Unexpected state");
    return VXRMImm;
  }

  bool operator==(const VXRMInfo &Other) const {
    // Uninitialized is only equal to another Uninitialized.
    if (!isValid())
      return !Other.isValid();
    if (!Other.isValid())
      return !isValid();

    // Unknown is only equal to another Unknown.
    if (isUnknown())
      return Other.isUnknown();
    if (Other.isUnknown())
      return isUnknown();

    return VXRMImm == Other.VXRMImm;
  }

  bool operator!=(const VXRMInfo &Other) const { return !(*this == Other); }

  // Calculate the VXRMInfo visible to a block assuming this and Other are
  // both predecessors.
  VXRMInfo intersect(const VXRMInfo &Other) const {
    // If the new value isn't valid, ignore it.
    if (!Other.isValid())
      return *this;

    // If this value isn't valid, this must be the first predecessor, use it.
    if (!isValid())
      return Other;

    // If either is unknown, the result is unknown.
    if (isUnknown() || Other.isUnknown())
      return VXRMInfo::getUnknown();

    // If we have an exact match, return this.
    if (*this == Other)
      return *this;

    // Otherwise the result is unknown.
    return VXRMInfo::getUnknown();
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Support for debugging, callable in GDB: V->dump()
  LLVM_DUMP_METHOD void dump() const {
    print(dbgs());
    dbgs() << "\n";
  }

  void print(raw_ostream &OS) const {
    OS << '{';
    if (!isValid())
      OS << "Uninitialized";
    else if (isUnknown())
      OS << "Unknown";
    else
      OS << getVXRMImm();
    OS << '}';
  }
#endif
};

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_ATTRIBUTE_USED
inline raw_ostream &operator<<(raw_ostream &OS, const VXRMInfo &V) {
  V.print(OS);
  return OS;
}
#endif

struct BlockData {
  // The VXRMInfo that represents the net changes to the VXRM registers
  // made by this block. Calculated in Phase 1.
  VXRMInfo Change;

  // The VXRMInfo that represents the VXRM settings on exit from this
  // block. Calculated in Phase 2.
  VXRMInfo Exit;

  // The VXRMInfo that represents the intersection of the VXRM settings from all
  // predecessor blocks. Calculated in Phase 2, and used by Phase 3.
  VXRMInfo Pred;

  // Keeps track of whether the block is already in the queue.
  bool InQueue;

  BlockData() = default;
};

class RISCVInsertWriteVXRM : public MachineFunctionPass {
  const TargetInstrInfo *TII;

  std::vector<BlockData> BlockInfo;
  std::queue<const MachineBasicBlock *> WorkList;

public:
  static char ID;

  RISCVInsertWriteVXRM() : MachineFunctionPass(ID) {
    initializeRISCVInsertWriteVXRMPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return RISCV_INSERT_WRITE_VXRM_NAME;
  }

private:
  bool computeVXRMChanges(const MachineBasicBlock &MBB, VXRMInfo &CurInfo) const;
  void computeIncomingVXRM(const MachineBasicBlock &MBB);
  void emitWriteVXRM(MachineBasicBlock &MBB);
};

} // end anonymous namespace

char RISCVInsertWriteVXRM::ID = 0;

INITIALIZE_PASS(RISCVInsertWriteVXRM, DEBUG_TYPE, RISCV_INSERT_WRITE_VXRM_NAME,
                false, false)

bool RISCVInsertWriteVXRM::computeVXRMChanges(const MachineBasicBlock &MBB,
                                              VXRMInfo &CurInfo) const {
  bool NeedVXRMChange = false;
  CurInfo = BlockInfo[MBB.getNumber()].Pred;

  for (const MachineInstr &MI : MBB) {
    int VXRMIdx = RISCVII::getVXRMOpNum(MI.getDesc());
    if (VXRMIdx >= 0) {
      unsigned NewVXRMImm = MI.getOperand(VXRMIdx).getImm() & 7;
      NeedVXRMChange = true;
      CurInfo.setVXRMImm(NewVXRMImm);
    }

    if (MI.isCall() || MI.isInlineAsm() || MI.modifiesRegister(RISCV::VXRM))
      CurInfo.setUnknown();
  }

  return NeedVXRMChange;
}

void RISCVInsertWriteVXRM::computeIncomingVXRM(const MachineBasicBlock &MBB) {
  BlockData &BBInfo = BlockInfo[MBB.getNumber()];
  BBInfo.InQueue = false;
  VXRMInfo InInfo;

  if (MBB.pred_empty()) {
    // There are no predecessors, so use the default starting status.
    InInfo.setUnknown();
  } else {
    for (const MachineBasicBlock *P : MBB.predecessors())
      InInfo = InInfo.intersect(BlockInfo[P->getNumber()].Exit);
  }

  // If we don't have any valid predecessor value, wait until we do.
  if (!InInfo.isValid())
    return;

  // If no change, no need to rerun block
  if (InInfo == BBInfo.Pred)
    return;

  BBInfo.Pred = InInfo;
  LLVM_DEBUG(dbgs() << "Entry state of " << printMBBReference(MBB)
                    << " changed to " << BBInfo.Pred << "\n");

  VXRMInfo TmpStatus;
  computeVXRMChanges(MBB, TmpStatus);

  if (BBInfo.Exit == TmpStatus)
    return;

  BBInfo.Exit = TmpStatus;
  LLVM_DEBUG(dbgs() << "Exit state of " << printMBBReference(MBB)
                    << " changed to " << BBInfo.Exit << "\n");

  // Add the successors to the work list so we can propagate the changed exit
  // status.
  for (MachineBasicBlock *S : MBB.successors())
    if (!BlockInfo[S->getNumber()].InQueue) {
      BlockInfo[S->getNumber()].InQueue = true;
      WorkList.push(S);
    }
}

void RISCVInsertWriteVXRM::emitWriteVXRM(MachineBasicBlock &MBB) {
  VXRMInfo CurInfo = BlockInfo[MBB.getNumber()].Pred;

  for (MachineInstr &MI : MBB) {
    int VXRMIdx = RISCVII::getVXRMOpNum(MI.getDesc());
    if (VXRMIdx >= 0) {
      unsigned NewVXRMImm = MI.getOperand(VXRMIdx).getImm();

      if (!CurInfo.isStatic() || CurInfo.getVXRMImm() != NewVXRMImm) {
        BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(RISCV::WriteVXRMImm))
            .addImm(NewVXRMImm);
        MI.addOperand(MachineOperand::CreateReg(RISCV::VXRM, /*IsDef*/ false,
                                                /*IsImp*/ true));
      }

      CurInfo.setVXRMImm(NewVXRMImm);
    }

    if (MI.isCall() || MI.isInlineAsm() || MI.modifiesRegister(RISCV::VXRM))
      CurInfo.setUnknown();
  }

  if (CurInfo.isValid() && CurInfo != BlockInfo[MBB.getNumber()].Exit)
    report_fatal_error("Mismatched VXRM state");
}

bool RISCVInsertWriteVXRM::runOnMachineFunction(MachineFunction &MF) {
  // Skip if the vector extension is not enabled.
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasVInstructions())
    return false;

  TII = ST.getInstrInfo();

  assert(BlockInfo.empty() && "Expect empty block infos");
  BlockInfo.resize(MF.getNumBlockIDs());

  bool NeedVXRMChange = false;
  // Phase 1 - Collect VXRM info
  for (const MachineBasicBlock &MBB : MF) {
    VXRMInfo TmpStatus;
    NeedVXRMChange |= computeVXRMChanges(MBB, TmpStatus);
    // Initial exit state is the last change made in the block
    BlockData &BBInfo = BlockInfo[MBB.getNumber()];
    BBInfo.Exit = TmpStatus;
    LLVM_DEBUG(dbgs() << "Initial exit state of " << printMBBReference(MBB)
                      << " is " << BBInfo.Exit << "\n");
  }

  if (NeedVXRMChange) {
    // Phase 2 - propagate the exit state to successors. We add all blocks to
    // the list here, but will also add any that need to be revisited during
    // Phase 2 processing.
    for (const MachineBasicBlock &MBB : MF) {
      WorkList.push(&MBB);
      BlockInfo[MBB.getNumber()].InQueue = true;
    }
    while (!WorkList.empty()) {
      const MachineBasicBlock &MBB = *WorkList.front();
      WorkList.pop();
      computeIncomingVXRM(MBB);
    }

    // Phase 3 - add any VXRM writes needed.
    for (MachineBasicBlock &MBB : MF)
      emitWriteVXRM(MBB);
  }

  BlockInfo.clear();

  return NeedVXRMChange;
}

FunctionPass *llvm::createRISCVInsertWriteVXRMPass() {
  return new RISCVInsertWriteVXRM();
}
