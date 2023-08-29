//===- SMEPeepholeOpt.cpp - SME peephole optimization pass ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "Utils/AArch64SMEAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-sme-peephole-opt"

namespace {

struct SMEPeepholeOpt : public MachineFunctionPass {
  static char ID;

  SMEPeepholeOpt() : MachineFunctionPass(ID) {
    initializeSMEPeepholeOptPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SME Peephole Optimization pass";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool optimizeStartStopPairs(MachineBasicBlock &MBB) const;
};

char SMEPeepholeOpt::ID = 0;

} // end anonymous namespace

static bool isConditionalStartStop(const MachineInstr *MI) {
  return MI->getOpcode() == AArch64::MSRpstatePseudo;
}

static bool isMatchingStartStopPair(const MachineInstr *MI1,
                                    const MachineInstr *MI2) {
  // We only consider the same type of streaming mode change here, i.e.
  // start/stop SM, or start/stop ZA pairs.
  if (MI1->getOperand(0).getImm() != MI2->getOperand(0).getImm())
    return false;

  bool IsConditional = isConditionalStartStop(MI2);
  if (isConditionalStartStop(MI1) != IsConditional)
    return false;

  if (!IsConditional)
    return true;

  // This optimisation is unlikely to happen in practice for conditional
  // smstart/smstop pairs as the virtual registers for pstate.sm will always
  // be different.
  // TODO: For this optimisation to apply to conditional smstart/smstop,
  // this pass will need to do more work to remove redundant calls to
  // __arm_sme_state.

  // Only consider conditional start/stop pairs which read the same register
  // holding the original value of pstate.sm, as some conditional start/stops
  // require the state on entry to the function.
  Register Reg1 = MI1->getOperand(2).getReg();
  Register Reg2 = MI2->getOperand(2).getReg();
  if (Reg1.isPhysical() || Reg2.isPhysical() || Reg1 != Reg2)
    return false;

  // Ensure reg masks are identical.
  if (MI1->getOperand(4).getRegMask() != MI2->getOperand(4).getRegMask())
    return false;

  // Check to make sure the conditional start/stop pairs are identical.
  return MI1->getOperand(3).getImm() == MI2->getOperand(3).getImm();
}

bool SMEPeepholeOpt::optimizeStartStopPairs(MachineBasicBlock &MBB) const {
  SmallVector<MachineInstr *, 4> ToBeRemoved;
  MachineInstr *Prev = nullptr;
  bool PrevIsStart = false;

  for (MachineInstr &MI : make_early_inc_range(MBB)) {
    // Walk through instructions in the block trying to find pairs of smstart
    // and smstop nodes that cancel each other out. We only permit a limited
    // set of instructions to appear between them, otherwise we reset our
    // tracking.
    switch (MI.getOpcode()) {
    default:
      Prev = nullptr;
      break;
    case AArch64::BL: {
      // Permits calls to __arm_sme_state.
      if (!MI.getOperand(0).isSymbol() ||
          strcmp(MI.getOperand(0).getSymbolName(), "__arm_sme_state"))
        Prev = nullptr;
      break;
    }
    case AArch64::COPY: {
      // Permit copies of 32 and 64-bit registers.
      if (!MI.getOperand(1).isReg()) {
        Prev = nullptr;
        break;
      }
      Register Reg = MI.getOperand(1).getReg();
      if (!AArch64::GPR32RegClass.contains(Reg) &&
          !AArch64::GPR64RegClass.contains(Reg))
        Prev = nullptr;
      break;
    }
    case AArch64::ADJCALLSTACKDOWN:
    case AArch64::ADJCALLSTACKUP:
    case AArch64::ANDXri:
      // We permit these as they don't generate SVE/NEON instructions.
      break;
    case AArch64::MSRpstatesvcrImm1:
    case AArch64::MSRpstatePseudo: {
      // Pairs of smstart/smstop nodes must either both be unconditional or
      // both be conditional.
      if (Prev && !isMatchingStartStopPair(Prev, &MI)) {
        Prev = nullptr;
        break;
      }

      assert((MI.getOperand(1).getImm() < 2) && "Invalid SM state");
      bool CurIsStart = (MI.getOperand(1).getImm() != 0);
      if (Prev && CurIsStart != PrevIsStart) {
        ToBeRemoved.push_back(Prev);
        ToBeRemoved.push_back(&MI);
      }

      if (Prev)
        Prev = nullptr;
      else {
        Prev = &MI;
        PrevIsStart = CurIsStart;
      }
      break;
    }
    }
  }

  for (MachineInstr *MI : ToBeRemoved)
    MI->eraseFromParent();

  return ToBeRemoved.size();
}

INITIALIZE_PASS(SMEPeepholeOpt, "aarch64-sme-peephole-opt",
                "SME Peephole Optimization", false, false)

bool SMEPeepholeOpt::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  if (!MF.getSubtarget<AArch64Subtarget>().hasSME())
    return false;

  assert(MF.getRegInfo().isSSA() && "Expected to be run on SSA form!");

  bool Changed = false;

  // Even if the block lives in a function with no SME attributes attached we
  // still have to analyze all the blocks because we may call a streaming
  // function that requires smstart/smstop pairs.
  for (MachineBasicBlock &MBB : MF) {
    Changed |= optimizeStartStopPairs(MBB);
  }

  return Changed;
}

FunctionPass *llvm::createSMEPeepholeOptPass() { return new SMEPeepholeOpt(); }
